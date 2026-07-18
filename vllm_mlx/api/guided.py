# SPDX-License-Identifier: Apache-2.0
"""
Guided generation for structured JSON output using llguidance.

This module provides constrained decoding for JSON schema enforcement,
ensuring model outputs strictly adhere to specified schemas.

Backend
-------
The constraint engine is `llguidance <https://github.com/guidance-ai/llguidance>`_,
driving an ``mlx_lm`` decode loop. On every step llguidance computes a
token bitmask for the current grammar state and its native ``llguidance.mlx``
Metal kernel writes ``-inf`` into the logits of every disallowed token
before sampling. This replaces the former ``outlines``-backed path; the
public surface (``GuidedGenerator``, ``generate_with_schema``,
``is_guided_available``, ``json_schema_to_pydantic``) is unchanged.

Two constraint modes are supported, matching the two the OpenAI
``response_format`` route exposes today:

* ``generate_json``        — a full JSON Schema (``$defs``/``$ref``/
  ``anyOf``/``enum``/numeric bounds/``additionalProperties:false``/nested
  objects all interpreted natively by llguidance).
* ``generate_json_object`` — any syntactically valid JSON *object* (the
  ``response_format={"type":"json_object"}`` mode).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# MUST install the MLX hardware-compat shim BEFORE the `mlx_lm` import below.
# Even though the import is inside a `try`, the body still runs at module
# load time; on success it triggers `mlx_lm/__init__.py` → `mlx_lm.generate`
# → `mx.new_thread_local_stream(...)` capture, which on M5 single-stream
# GPUs would be unusable (#404). The shim is idempotent and a no-op on
# hardware where the original API works.
from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

# Check for llguidance availability. We need three surfaces:
#   * ``llguidance``            — grammar factories (grammar_from_json_schema)
#   * ``llguidance.hf``         — build an LLTokenizer from a HF fast tokenizer
#   * ``llguidance.mlx``        — LLMatcher + the native Metal mask kernel
# and ``mlx_lm`` / ``mlx.core`` for the decode loop. Any of these missing
# means guided generation is not installed; degrade gracefully.
try:
    import llguidance as _llguidance
    import llguidance.hf as _llguidance_hf
    import mlx.core as mx
    import mlx_lm  # noqa: F401  (imported for availability probe / shim trigger)
    from llguidance.mlx import (
        LLMatcher,
        allocate_token_bitmask,
        apply_token_bitmask,
        fill_next_token_bitmask,
    )

    HAS_LLGUIDANCE = True
except ImportError:
    HAS_LLGUIDANCE = False
    mx = None
    mlx_lm = None
    _llguidance = None
    _llguidance_hf = None
    LLMatcher = None
    allocate_token_bitmask = None
    apply_token_bitmask = None
    fill_next_token_bitmask = None


def is_guided_available() -> bool:
    """Check if guided generation with llguidance is available."""
    return HAS_LLGUIDANCE


# A permissive grammar for the ``json_object`` mode: any single, complete
# JSON *object* (``{...}``). We express it as a one-line JSON Schema of
# ``{"type": "object"}`` and let llguidance compile it — this admits
# arbitrary keys/values (nested objects, arrays, numbers, strings, etc.)
# exactly like OpenAI's ``response_format={"type":"json_object"}`` while
# still guaranteeing the top-level value is an object. This replaces the
# previous outlines regex ``\{[^{}]*\}`` which (a) was silently degraded
# to unconstrained on outlines 1.3.x (BUG-1) and (b) could not represent
# nested objects even when it did run.
_JSON_OBJECT_SCHEMA = '{"type": "object"}'


def json_schema_to_pydantic(schema: dict[str, Any]) -> type | None:
    """
    Convert a JSON schema to a Pydantic model dynamically.

    Kept as a public backward-compat surface. It is NOT used on the
    guided-generation hot path — llguidance interprets the raw JSON
    schema natively, so routing the constraint through this shallow
    converter would silently drop ``$defs``/``$ref``/``anyOf``/``enum``/
    numeric-bounds and re-introduce the waybarrios#546-class bug.

    Args:
        schema: JSON schema dict

    Returns:
        Dynamically created Pydantic model class, or None if conversion fails
    """
    try:
        from pydantic import create_model

        # Extract properties from schema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        # Build field definitions for Pydantic
        field_definitions = {}

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "null": type(None),
        }

        for prop_name, prop_spec in properties.items():
            prop_type = prop_spec.get("type", "string")

            # Handle array type. The "object" and "array" element types
            # are special-cased: without this branch they fell through to
            # ``type_mapping.get(items_type, str)`` and silently became
            # ``list[str]``, so the model emitted strings where the schema
            # required objects — producing JSON that fails validation
            # against the user's own schema (R10 sweep, guided.py bug).
            if prop_type == "array":
                items_type = prop_spec.get("items", {}).get("type", "string")
                if items_type == "object":
                    python_type = list[dict]
                elif items_type == "array":
                    python_type = list[list]
                else:
                    inner_type = type_mapping.get(items_type, str)
                    python_type = list[inner_type]
            # Handle object type (nested)
            elif prop_type == "object":
                # For nested objects, use dict
                python_type = dict
            else:
                python_type = type_mapping.get(prop_type, str)

            # Make optional if not required
            if prop_name not in required:
                python_type = python_type | None
                default = None
            else:
                default = ...

            field_definitions[prop_name] = (python_type, default)

        # Create the model dynamically
        model = create_model("DynamicModel", **field_definitions)
        return model

    except Exception as e:
        logger.warning(f"Failed to convert JSON schema to Pydantic: {e}")
        logger.debug(f"Problematic schema: {schema}")
        return None


class GuidedGenerator:
    """
    Guided generation using llguidance for constrained JSON decoding.

    This class wraps an MLX model to provide structured output generation
    that guarantees valid JSON matching a specified schema (or, in
    ``json_object`` mode, any valid JSON object).

    The llguidance ``LLTokenizer`` is built lazily on first use and cached
    on the instance — it is derived from the INNER transformers fast/Rust
    tokenizer (``tokenizer._tokenizer``), not the ``mlx_lm``
    ``TokenizerWrapper``. If the model ships without a fast tokenizer,
    tokenizer construction fails gracefully (logged, returns ``None`` from
    generation) rather than crashing.
    """

    def __init__(self, model, tokenizer):
        """
        Initialize the guided generator.

        Args:
            model: MLX model instance
            tokenizer: Tokenizer instance (mlx_lm ``TokenizerWrapper``)
        """
        if not HAS_LLGUIDANCE:
            raise ImportError(
                "llguidance is required for guided generation. "
                "Install with: pip install 'rapid-mlx[guided]'"
            )

        self._model = model
        self._tokenizer = tokenizer
        # Lazily-built llguidance tokenizer. ``False`` is the
        # "not-yet-attempted" sentinel; ``None`` means "attempted and
        # unavailable" (so we don't rebuild on every call); a real
        # ``LLTokenizer`` otherwise.
        self._lltokenizer: Any = False

    def _get_lltokenizer(self):
        """Get or build the llguidance ``LLTokenizer``.

        llguidance needs the underlying *fast* (Rust-backed) transformers
        tokenizer, exposed by ``mlx_lm``'s ``TokenizerWrapper`` as
        ``._tokenizer``. Models loaded with a slow (pure-Python)
        tokenizer do not have that attribute in a usable form; in that
        case we log once and return ``None`` so the caller can degrade to
        unconstrained generation instead of raising.

        Returns:
            An ``LLTokenizer`` instance, or ``None`` if one cannot be
            built for this model.
        """
        # Cached (either a real tokenizer or the ``None`` "unavailable"
        # sentinel after a prior failed attempt).
        if self._lltokenizer is not False:
            return self._lltokenizer

        hf_tok = getattr(self._tokenizer, "_tokenizer", None)
        if hf_tok is None:
            logger.warning(
                "Guided generation unavailable: the model's tokenizer has "
                "no underlying fast (Rust) tokenizer (`._tokenizer`), which "
                "llguidance requires. Falling back to unconstrained "
                "generation."
            )
            self._lltokenizer = None
            return None

        # A fast tokenizer is required for `llguidance.hf.from_tokenizer`;
        # a slow tokenizer lacks `is_fast`/the Rust internals it reads.
        if getattr(hf_tok, "is_fast", True) is False:
            logger.warning(
                "Guided generation unavailable: the model's tokenizer is a "
                "slow (non-fast) tokenizer, which llguidance cannot consume. "
                "Falling back to unconstrained generation."
            )
            self._lltokenizer = None
            return None

        try:
            self._lltokenizer = _llguidance_hf.from_tokenizer(hf_tok)
        except Exception:
            logger.exception(
                "Guided generation unavailable: failed to build an "
                "llguidance LLTokenizer from the model's fast tokenizer. "
                "Falling back to unconstrained generation."
            )
            self._lltokenizer = None
            return None

        return self._lltokenizer

    def _decode_constrained(
        self,
        grammar: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str | None:
        """Run an ``mlx_lm`` decode loop constrained by an llguidance grammar.

        On every step:
          1. ``fill_next_token_bitmask`` computes the allow-mask for the
             matcher's current state.
          2. Logits are sliced to ``lltok.vocab_size`` (the model's logit
             width can exceed the tokenizer vocab — the padding tail is
             never a real token) and the native ``apply_token_bitmask``
             Metal kernel writes ``-inf`` into disallowed positions.
          3. The next token is chosen (greedy at ``temperature<=0``, else
             temperature sampling) and fed back into the matcher.
          4. The model is advanced by that one token.

        Returns the decoded text, or ``None`` if the grammar failed to
        compile or the tokenizer was unavailable.
        """
        lltok = self._get_lltokenizer()
        if lltok is None:
            return None

        model = self._model
        tokenizer = self._tokenizer

        # llguidance NEVER raises on grammar errors — it stores them on the
        # matcher. Construct, then check ``get_error()`` explicitly.
        matcher = LLMatcher(lltok, grammar)
        err = matcher.get_error()
        if err:
            logger.error("llguidance grammar/compile error: %s", err)
            return None

        vocab = lltok.vocab_size
        bitmask = allocate_token_bitmask(1, vocab)

        prompt_ids = tokenizer.encode(prompt)
        prompt_arr = mx.array(prompt_ids)

        # Prefill.
        logits = model(prompt_arr[None])[:, -1, :]

        generated: list[int] = []
        for _ in range(max_tokens):
            if matcher.is_stopped():
                break

            # 1. allow-mask for the current matcher state.
            fill_next_token_bitmask(matcher, bitmask, 0)

            # 2. slice logits to the tokenizer vocab width, then apply the
            #    mask via the native Metal kernel (disallowed -> -inf).
            model_vocab = logits.shape[1]
            cur_logits = logits[:, :vocab] if model_vocab > vocab else logits
            masked = apply_token_bitmask(cur_logits, bitmask)

            # 3. pick a token.
            if temperature and temperature > 0:
                tok = int(mx.random.categorical(masked / temperature, axis=1).item())
            else:
                tok = int(mx.argmax(masked, axis=1).item())

            # 4. feed back into the matcher. Because we masked, this should
            #    always be accepted; if not, stop cleanly rather than emit
            #    an out-of-grammar token.
            ok = matcher.consume_token(tok)
            if not ok or matcher.is_error():
                e = matcher.get_error()
                if e:
                    logger.error("llguidance rejected token %d: %s", tok, e)
                break

            generated.append(tok)

            # 5. advance the model by one token.
            logits = model(mx.array([tok])[None])[:, -1, :]

        if not generated:
            return None
        return tokenizer.decode(generated)

    def generate_json(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str | None:
        """Generate JSON output constrained to a schema.

        The raw schema dict is compiled by llguidance via
        ``LLMatcher.grammar_from_json_schema``, which natively understands
        ``$defs``, ``$ref``, ``anyOf``, ``enum``, numeric bounds,
        ``additionalProperties: false``, and nested objects. We compile
        with ``overrides={"whitespace_flexible": False}`` so the grammar
        emits compact JSON (no free structural whitespace), matching the
        server's canonical JSON shape.

        We deliberately do NOT route the schema through
        ``json_schema_to_pydantic`` first — that shallow converter silently
        drops every one of those constructs, which on a real-world schema
        with ``$defs`` + ``$ref`` (waybarrios#546 repro) surfaced as a
        valid JSON *array* where the schema required an object. The dict is
        handed to llguidance directly.
        """
        try:
            import json as _json

            schema_str = _json.dumps(json_schema)
            grammar = LLMatcher.grammar_from_json_schema(
                schema_str,
                overrides={"whitespace_flexible": False},
            )
            return self._decode_constrained(
                grammar=grammar,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception:
            logger.exception("Guided generation failed")
            return None

    def generate_json_object(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str | None:
        """
        Generate any valid JSON object.

        Constrains decoding to a generic ``{"type": "object"}`` JSON Schema
        via llguidance — the ``response_format={"type":"json_object"}``
        mode. This is a real constraint (BUG-1 fix): the previous outlines
        path used ``generate.regex(...)`` which was removed in outlines
        1.3.x, so ``generate_json_object`` silently degraded to
        unconstrained output. It now guarantees the top-level value is a
        complete JSON object with arbitrary (nested) contents.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            JSON string, or None on failure
        """
        try:
            grammar = LLMatcher.grammar_from_json_schema(
                _JSON_OBJECT_SCHEMA,
                overrides={"whitespace_flexible": False},
            )
            return self._decode_constrained(
                grammar=grammar,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception:
            logger.exception("JSON object generation failed")
            return None


def generate_with_schema(
    model,
    tokenizer,
    prompt: str,
    json_schema: dict[str, Any],
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str | None:
    """
    Convenience function for one-shot guided JSON generation.

    Args:
        model: MLX model
        tokenizer: Tokenizer
        prompt: Input prompt
        json_schema: JSON schema
        max_tokens: Maximum tokens
        temperature: Sampling temperature

    Returns:
        JSON string or None if guided generation unavailable/failed
    """
    if not HAS_LLGUIDANCE:
        return None

    try:
        generator = GuidedGenerator(model, tokenizer)
        return generator.generate_json(
            prompt=prompt,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        logger.error(f"generate_with_schema failed: {e}")
        return None
