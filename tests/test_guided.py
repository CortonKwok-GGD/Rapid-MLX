# SPDX-License-Identifier: Apache-2.0
"""Tests for the guided generation module (llguidance backend).

These tests prefer exercising REAL llguidance over mocking it — the
package is tiny and installable, and mocking the grammar engine would
only prove we called some functions, not that the constraint holds.

The decode loop is driven against a *fake* MLX model (a small stub that
returns deterministic logits) so the tests stay fast and need no model
download, while the grammar compilation, tokenizer, bitmask and native
mask-apply all run for real. Tests that need real llguidance/mlx are
guarded with ``pytest.importorskip`` so a no-extras environment skips
them cleanly rather than erroring.
"""

import json

import pytest

# Real llguidance + mlx are needed for the constrained-decode tests. In a
# no-``[guided]`` environment these skip cleanly.
_HAS_GUIDED = False
try:  # pragma: no cover - import guard
    import mlx.core as mx  # noqa: F401

    from vllm_mlx.api import guided as _guided_probe

    _HAS_GUIDED = _guided_probe.is_guided_available()
except Exception:  # pragma: no cover
    _HAS_GUIDED = False

requires_guided = pytest.mark.skipif(
    not _HAS_GUIDED,
    reason="requires the [guided] extra (llguidance + mlx) to be installed",
)


# ---------------------------------------------------------------------------
# Fake MLX model + tokenizer helpers (for the decode-loop tests)
# ---------------------------------------------------------------------------


def _build_byte_level_fast_tokenizer():
    """Build a hermetic byte-level BPE ``PreTrainedTokenizerFast`` in memory.

    No network, no ``from_pretrained``, no model weights. The vocab covers
    the full 256-symbol ByteLevel alphabet plus a few specials, so every
    UTF-8 byte is representable and encode/decode round-trips exactly. This
    is the exact tokenizer shape llguidance's ``hf.from_tokenizer`` expects
    (a Rust-backed fast tokenizer with ``is_fast == True``) and mirrors the
    in-repo offline-tokenizer pattern in ``test_streaming_detokenizer_bpe``.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = pre_tokenizers.ByteLevel.alphabet()  # 256 printable byte proxies
    vocab: dict[str, int] = {}
    for special in ("<pad>", "<s>", "</s>"):
        vocab[special] = len(vocab)
    for symbol in sorted(alphabet):
        vocab[symbol] = len(vocab)

    rust = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    rust.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    rust.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=rust,
        eos_token="</s>",
        bos_token="<s>",
        pad_token="<pad>",
    )


def _make_fake_model(lltok, plan, prompt_len):
    """Build a stub model whose logits deterministically pick ``plan`` tokens.

    ``plan`` is a list of token ids the model "wants" to emit in order. On
    each forward call the model returns logits that put +100 on the next
    planned token and 0 elsewhere — but because the guided decode loop
    masks disallowed tokens to -inf, the plan only lands when the grammar
    permits it. This lets a test drive a fully real llguidance mask over a
    predictable model without any weights.

    CONTEXT DEPENDENCE (guards the missing-KV-cache regression). The plan
    position is derived SOLELY from the per-request KV cache's ``offset``
    — i.e. how many tokens have actually flowed through the cache — NOT an
    internal step counter. A correct decode loop (mirroring
    ``mlx_lm.generate.generate_step``) prefills the whole prompt through
    the cache (offset -> ``prompt_len``) and then feeds one token per step
    (offset -> ``prompt_len + 1``, ``+ 2`` …), so ``offset - prompt_len``
    walks the plan 0, 1, 2, …. A BUGGY loop that drops the cache (passes no
    cache, or a fresh cache each step) leaves ``offset`` stuck at the
    single-token width it just saw, so ``offset - prompt_len`` goes
    negative/constant and the model emits the WRONG planned token —
    producing garbage that fails the caller's exact-value assertion. This
    makes the fake model a genuine trap for a #1-class regression instead
    of a step-counter that ignores its inputs.

    The model mimics ``mlx-lm``'s cache contract: it exposes ``make_cache``
    (so ``make_prompt_cache(model)`` defers to it) and updates the KV cache
    on every ``model(ids, cache=cache)`` call, exactly like a real model.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    vocab = lltok.vocab_size

    class _FakeModel:
        class args:  # noqa: N801 - mimic mlx-lm model.args.vocab_size
            vocab_size = vocab

        def make_cache(self):
            # A single real KVCache — its ``offset`` is our position source
            # of truth, and it enforces that callers actually thread it.
            return [KVCache()]

        def __call__(self, ids, cache=None):
            # ids has shape (1, seq_len). We only inspect the last position.
            seq_len = ids.shape[1]

            # A call's last-position logits predict the NEXT token. The plan
            # index is derived from the cache offset AFTER this call appends
            # its tokens, relative to the prompt length: the prefill call
            # (offset 0 -> prompt_len) predicts ``plan[0]``, the step that
            # emits the k-th generated token (offset prompt_len+k-1 ->
            # prompt_len+k) predicts ``plan[k]``. This is read SOLELY from
            # the KV cache — a decode loop that drops the cache leaves
            # ``offset`` stuck at the single-token width it just saw, so the
            # plan index collapses and the model emits the WRONG token,
            # producing garbage that fails the caller's exact-value
            # assertion. That makes this a real trap for a #1-class
            # regression rather than a step counter that ignores its inputs.
            if cache is None:
                raise AssertionError(
                    "fake model called without a KV cache — the constrained "
                    "decode loop must thread make_prompt_cache(model) through "
                    "every model(...) call (mirrors mlx_lm.generate_step)"
                )
            offset_after = cache[0].offset + seq_len
            pos = offset_after - prompt_len

            logits = mx.zeros((1, seq_len, vocab))
            want = None
            if plan:
                if 0 <= pos < len(plan):
                    want = plan[pos]
                elif pos >= len(plan):
                    want = plan[-1]
            if want is not None and want < vocab:
                # Bias the last-position logits toward the planned token.
                row = [0.0] * vocab
                row[want] = 100.0
                logits[0, -1] = mx.array(row)

            # Append this call's tokens into the cache so ``offset`` advances
            # exactly like a real forward pass (prefill: +seq_len; step: +1).
            keys = mx.zeros((1, 1, seq_len, 1))
            cache[0].update_and_fetch(keys, keys)

            return logits

    return _FakeModel()


# ---------------------------------------------------------------------------
# Tests for json_schema_to_pydantic (public backward-compat surface)
# ---------------------------------------------------------------------------


class TestJsonSchemaToPydantic:
    """``json_schema_to_pydantic`` is retained as a public helper. It is NOT
    on the guided hot path anymore (llguidance interprets schemas natively),
    but the conversion contract must keep working for external callers."""

    def test_basic_string_property(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        model = json_schema_to_pydantic(schema)
        assert model is not None
        assert hasattr(model, "model_validate")

        instance = model.model_validate({"name": "test"})
        assert instance.name == "test"

    def test_multiple_property_types(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "score": {"type": "number"},
                "active": {"type": "boolean"},
            },
        }

        model = json_schema_to_pydantic(schema)
        assert model is not None

        instance = model.model_validate(
            {"name": "John", "age": 30, "score": 85.5, "active": True}
        )
        assert instance.name == "John"
        assert instance.age == 30
        assert instance.score == 85.5
        assert instance.active is True

    def test_required_fields(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
            "required": ["name"],
        }

        model = json_schema_to_pydantic(schema)
        assert model is not None

        with pytest.raises(Exception):
            model.model_validate({})

        instance = model.model_validate({"name": "test"})
        assert instance.name == "test"
        assert instance.email is None

    def test_optional_fields(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {"optional_field": {"type": "string"}},
        }

        model = json_schema_to_pydantic(schema)
        assert model is not None

        instance = model.model_validate({})
        assert instance.optional_field is None

    def test_array_type(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "scores": {"type": "array", "items": {"type": "number"}},
            },
        }

        model = json_schema_to_pydantic(schema)
        assert model is not None

        instance = model.model_validate(
            {"tags": ["a", "b", "c"], "scores": [1.0, 2.5, 3.5]}
        )
        assert instance.tags == ["a", "b", "c"]
        assert instance.scores == [1.0, 2.5, 3.5]

    def test_nested_object(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                }
            },
        }

        model = json_schema_to_pydantic(schema)
        assert model is not None

        instance = model.model_validate({"user": {"name": "John", "age": 30}})
        assert instance.user == {"name": "John", "age": 30}

    def test_empty_schema(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {"type": "object", "properties": {}}

        model = json_schema_to_pydantic(schema)
        assert model is not None

        instance = model.model_validate({})
        assert hasattr(instance, "model_validate")

    def test_missing_properties(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {"type": "object"}

        model = json_schema_to_pydantic(schema)
        assert model is not None

    def test_complex_schema(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "count": {"type": "integer"},
                "price": {"type": "number"},
                "is_active": {"type": "boolean"},
                "items": {"type": "array", "items": {"type": "string"}},
                "metadata": {
                    "type": "object",
                    "properties": {"created": {"type": "string"}},
                },
            },
            "required": ["id", "count"],
        }

        model = json_schema_to_pydantic(schema)
        assert model is not None

        instance = model.model_validate(
            {
                "id": "abc123",
                "count": 5,
                "price": 10.99,
                "is_active": True,
                "items": ["a", "b"],
                "metadata": {"created": "2024-01-01"},
            }
        )
        assert instance.id == "abc123"
        assert instance.count == 5


# ---------------------------------------------------------------------------
# is_guided_available
# ---------------------------------------------------------------------------


class TestIsGuidedAvailable:
    """``is_guided_available`` reflects the llguidance availability flag."""

    def test_returns_true_when_llguidance_available(self):
        import vllm_mlx.api.guided as guided

        # Temporarily force the flag True and confirm the helper agrees.
        original = guided.HAS_LLGUIDANCE
        try:
            guided.HAS_LLGUIDANCE = True
            assert guided.is_guided_available() is True
        finally:
            guided.HAS_LLGUIDANCE = original

    def test_returns_false_when_llguidance_not_available(self):
        import vllm_mlx.api.guided as guided

        original = guided.HAS_LLGUIDANCE
        try:
            guided.HAS_LLGUIDANCE = False
            assert guided.is_guided_available() is False
        finally:
            guided.HAS_LLGUIDANCE = original


# ---------------------------------------------------------------------------
# GuidedGenerator construction
# ---------------------------------------------------------------------------


class TestGuidedGenerator:
    """Construction contract + graceful degradation."""

    def test_init_raises_import_error_without_llguidance(self):
        import vllm_mlx.api.guided as guided

        original = guided.HAS_LLGUIDANCE
        try:
            guided.HAS_LLGUIDANCE = False
            with pytest.raises(ImportError, match="llguidance is required"):
                guided.GuidedGenerator(None, None)
        finally:
            guided.HAS_LLGUIDANCE = original

    def test_init_succeeds_with_llguidance(self):
        import vllm_mlx.api.guided as guided

        original = guided.HAS_LLGUIDANCE
        try:
            guided.HAS_LLGUIDANCE = True

            class _Model:
                pass

            class _Tok:
                pass

            model = _Model()
            tok = _Tok()
            generator = guided.GuidedGenerator(model, tok)
            assert generator._model is model
            assert generator._tokenizer is tok
        finally:
            guided.HAS_LLGUIDANCE = original

    def test_degrades_gracefully_without_fast_tokenizer(self):
        """A tokenizer with no underlying fast (``._tokenizer``) tokenizer
        must NOT crash — ``_get_lltokenizer`` logs and returns None, and
        ``generate_json`` returns None (caller falls back to
        unconstrained)."""
        import vllm_mlx.api.guided as guided

        original = guided.HAS_LLGUIDANCE
        try:
            guided.HAS_LLGUIDANCE = True

            class _SlowTokWrapper:
                # No ``_tokenizer`` attribute at all.
                def encode(self, s):
                    return [1, 2, 3]

            generator = guided.GuidedGenerator(object(), _SlowTokWrapper())
            assert generator._get_lltokenizer() is None
            out = generator.generate_json(
                "hi", {"type": "object"}, max_tokens=8, temperature=0.0
            )
            assert out is None
        finally:
            guided.HAS_LLGUIDANCE = original


# ---------------------------------------------------------------------------
# Real-llguidance constrained decode over a fake model
# ---------------------------------------------------------------------------


@requires_guided
class TestConstrainedDecodeWithRealLLGuidance:
    """The heart of the migration: prove the llguidance grammar actually
    constrains an mlx decode loop. Uses a real LLTokenizer (built from a
    real fast tokenizer) + a fake model.
    """

    @pytest.fixture(scope="class")
    def hf_fast_tokenizer(self):
        """A real *fast* (Rust-backed) tokenizer built entirely in memory —
        no network, no cache lookup, no model download. This keeps the
        constrained-decode tests hermetic in a clean/offline CI (the prior
        ``AutoTokenizer.from_pretrained("gpt2")`` did an uncaught
        network/cache fetch and failed offline).

        It is a complete byte-level BPE tokenizer: the vocab spans the full
        256-symbol ByteLevel alphabet, so it can encode ANY UTF-8 string
        (every JSON structural char, ``Sure``, ``[``, arbitrary content)
        and round-trips exactly — which is all llguidance needs to build an
        ``LLTokenizer`` and drive the grammar mask.
        """
        return _build_byte_level_fast_tokenizer()

    @pytest.fixture(scope="class")
    def wrapped_tokenizer(self, hf_fast_tokenizer):
        """Wrap the fast tokenizer to mimic mlx-lm's TokenizerWrapper shape:
        the guided code reads ``._tokenizer`` for the inner fast tokenizer
        and calls ``.encode``/``.decode`` on the wrapper."""

        class _Wrapper:
            def __init__(self, inner):
                self._tokenizer = inner

            def encode(self, s):
                return self._tokenizer.encode(s)

            def decode(self, ids):
                return self._tokenizer.decode(ids)

        return _Wrapper(hf_fast_tokenizer)

    def _make_generator(self, wrapped, plan, prompt="prompt"):
        import vllm_mlx.api.guided as guided

        gen = guided.GuidedGenerator.__new__(guided.GuidedGenerator)
        gen._tokenizer = wrapped
        gen._lltokenizer = False
        lltok = gen._get_lltokenizer()
        assert lltok is not None
        # The fake model reads its plan position from the KV cache offset,
        # so it needs to know how many prompt tokens the prefill consumes.
        prompt_len = len(wrapped.encode(prompt))
        gen._model = _make_fake_model(lltok, plan, prompt_len)
        return gen, lltok

    def test_json_object_grammar_produces_valid_object(self, wrapped_tokenizer):
        """A model that wants to emit ``{"a":1}`` under the json_object
        grammar produces EXACTLY that (all planned tokens are grammar-legal,
        so the mask never blocks them).

        This is also the primary guard against the missing-KV-cache
        regression (#1): the fake model reads its plan position from the KV
        cache offset, so the ``{"a":1}`` sequence only lands if the decode
        loop threads ``make_prompt_cache(model)`` through the prefill and
        every step. A loop that drops the cache de-syncs the plan and the
        exact-value assertion below fails. Asserting only ``isinstance(...,
        dict)`` (the old check) could not catch that — a garbled ``{}`` is
        still a dict."""
        # Plan out the tokens for {"a":1}. Encode piecewise so we know the
        # ids; the fake model then "wants" each in turn — driven by the KV
        # cache offset, not a blind step counter.
        inner = wrapped_tokenizer._tokenizer
        target = '{"a":1}'
        plan = inner.encode(target)
        gen, _ = self._make_generator(wrapped_tokenizer, plan)
        out = gen.generate_json_object("prompt", max_tokens=32, temperature=0.0)
        assert out is not None, (
            "constrained decode returned None — either the grammar never "
            "reached an accepting state or the KV cache was not threaded"
        )
        parsed = json.loads(out)
        assert parsed == {"a": 1}, (
            f"expected exactly {{'a': 1}} but got {parsed!r}; a mismatch here "
            "means the decode loop lost context (missing KV cache) so the "
            "context-dependent fake model emitted the wrong planned tokens"
        )

    def test_negative_control_first_token_mask(self, wrapped_tokenizer):
        """NEGATIVE CONTROL: under a JSON-object grammar the first-step
        allow-mask forbids a plain-text token (e.g. the word ``Sure``) and
        permits ``{`` — proving disallowed logits are driven to -inf."""
        import numpy as np
        from llguidance.mlx import (
            LLMatcher,
            allocate_token_bitmask,
            fill_next_token_bitmask,
        )

        import vllm_mlx.api.guided as guided

        gen = guided.GuidedGenerator.__new__(guided.GuidedGenerator)
        gen._tokenizer = wrapped_tokenizer
        gen._lltokenizer = False
        lltok = gen._get_lltokenizer()

        grammar = LLMatcher.grammar_from_json_schema(
            json.dumps({"type": "object"}),
            overrides={"whitespace_flexible": True},
        )
        matcher = LLMatcher(lltok, grammar)
        assert not matcher.get_error()

        bitmask = allocate_token_bitmask(1, lltok.vocab_size)
        fill_next_token_bitmask(matcher, bitmask, 0)
        arr = np.asarray(bitmask).reshape(-1)

        def _allowed(tok_id: int) -> bool:
            return bool((int(arr[tok_id // 32]) >> (tok_id % 32)) & 1)

        inner = wrapped_tokenizer._tokenizer
        brace = inner.encode("{")[0]
        prose = inner.encode("Sure")[0]
        assert _allowed(brace), "'{' must be allowed at a JSON-object start"
        assert not _allowed(prose), (
            "a plain-text token ('Sure') must be masked out at the start "
            "of a JSON-object grammar — the constraint is not enforcing "
            "the object opener"
        )

    def test_negative_control_masked_logits_are_neg_inf(self, wrapped_tokenizer):
        """Prove the native mask kernel writes -inf into a concretely
        disallowed position: build the step-0 mask, apply it to an
        all-ones logit row, and assert the 'Sure' token slot is -inf while
        the '{' slot is finite."""
        import math

        import mlx.core as mx
        from llguidance.mlx import (
            LLMatcher,
            allocate_token_bitmask,
            apply_token_bitmask,
            fill_next_token_bitmask,
        )

        import vllm_mlx.api.guided as guided

        gen = guided.GuidedGenerator.__new__(guided.GuidedGenerator)
        gen._tokenizer = wrapped_tokenizer
        gen._lltokenizer = False
        lltok = gen._get_lltokenizer()

        grammar = LLMatcher.grammar_from_json_schema(
            json.dumps({"type": "object"}),
            overrides={"whitespace_flexible": True},
        )
        matcher = LLMatcher(lltok, grammar)
        bitmask = allocate_token_bitmask(1, lltok.vocab_size)
        fill_next_token_bitmask(matcher, bitmask, 0)

        logits = mx.ones((1, lltok.vocab_size))
        masked = apply_token_bitmask(logits, bitmask)
        masked_np = masked  # mx array; index below

        inner = wrapped_tokenizer._tokenizer
        brace = inner.encode("{")[0]
        prose = inner.encode("Sure")[0]
        brace_val = float(masked_np[0, brace].item())
        prose_val = float(masked_np[0, prose].item())
        assert math.isfinite(brace_val), "'{' logit must stay finite (allowed)"
        assert prose_val == float("-inf"), (
            "disallowed 'Sure' token logit must be driven to -inf by the "
            f"native mask kernel; got {prose_val}"
        )

    def test_defs_ref_schema_constrains_to_object(self, wrapped_tokenizer):
        """waybarrios#546 regression: a schema using ``$defs`` + ``$ref`` +
        ``enum`` + ``additionalProperties:false`` must compile, and the
        constraint must hold not only at the top-level opener but INSIDE the
        referenced nested object.

        Pre-migration, routing such a schema through the shallow pydantic
        converter dropped the ``$ref`` and the model could emit a JSON
        array; a step-0-only test would miss a converter that drops the
        NESTED constraints while keeping the object opener. This test
        therefore drives the matcher token-by-token through a complete,
        schema-valid document (into the ``$ref`` Inner object, the enum, and
        the anyOf field), validates the finished document against the FULL
        schema, and — at two nested positions — asserts an invalid token is
        driven to ``-inf`` while the valid one stays finite:

          * inside ``$defs.Inner`` (``additionalProperties:false``): the
            only legal key is ``x`` — a ``y`` key char is masked.
          * at the ``kind`` enum value: only ``a``/``b`` are legal — a ``c``
            is masked.
        """
        import mlx.core as mx
        import numpy as np
        from llguidance.mlx import (
            LLMatcher,
            allocate_token_bitmask,
            apply_token_bitmask,
            fill_next_token_bitmask,
        )

        import vllm_mlx.api.guided as guided

        gen = guided.GuidedGenerator.__new__(guided.GuidedGenerator)
        gen._tokenizer = wrapped_tokenizer
        gen._lltokenizer = False
        lltok = gen._get_lltokenizer()

        schema = {
            "type": "object",
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                    "additionalProperties": False,
                }
            },
            "properties": {
                "inner": {"$ref": "#/$defs/Inner"},
                "kind": {"type": "string", "enum": ["a", "b"]},
                "tag": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["inner", "kind", "tag"],
            "additionalProperties": False,
        }
        grammar = LLMatcher.grammar_from_json_schema(
            json.dumps(schema), overrides={"whitespace_flexible": True}
        )
        matcher = LLMatcher(lltok, grammar)
        assert not matcher.get_error(), (
            f"$defs/$ref schema failed to compile: {matcher.get_error()}"
        )

        vocab = lltok.vocab_size
        bitmask = allocate_token_bitmask(1, vocab)
        inner = wrapped_tokenizer._tokenizer

        def _refresh_mask() -> np.ndarray:
            fill_next_token_bitmask(matcher, bitmask, 0)
            return np.asarray(bitmask).reshape(-1)

        def _allowed(arr: np.ndarray, tok_id: int) -> bool:
            return bool((int(arr[tok_id // 32]) >> (tok_id % 32)) & 1)

        def _mask_is_neg_inf(tok_id: int) -> bool:
            # Apply the CURRENT mask to an all-ones row via the native
            # kernel and read back whether this slot is -inf.
            fill_next_token_bitmask(matcher, bitmask, 0)
            row = apply_token_bitmask(mx.ones((1, vocab)), bitmask)
            return float(row[0, tok_id].item()) == float("-inf")

        # ---- Step 0: object opener required, array opener forbidden. ----
        arr0 = _refresh_mask()
        allowed_ids = [t for t in range(vocab) if _allowed(arr0, t)]
        assert allowed_ids, "grammar admitted no tokens at object start"
        bracket = inner.encode("[")[0]
        assert not _allowed(arr0, bracket), (
            "array opener '[' must be forbidden — the schema requires an "
            "object; allowing '[' is the waybarrios#546 regression"
        )
        for tid in allowed_ids:
            piece = inner.decode([tid])
            assert piece.lstrip().startswith("{"), (
                f"allowed step-0 token {tid} decodes to {piece!r}, which does "
                f"not open a JSON object — the $defs/$ref schema must still "
                f"constrain to an object opener"
            )

        # ---- Drive INTO the nested $ref Inner object. ----
        # Consume the prefix up to (but not into) the inner key name, so the
        # matcher is positioned at the key-open of Inner where
        # additionalProperties:false is in force.
        for t in inner.encode('{"inner":{"'):
            assert _allowed(_refresh_mask(), t), (
                f"prefix token {t} ({inner.decode([t])!r}) unexpectedly masked "
                "while descending into the $ref Inner object"
            )
            assert matcher.consume_token(t)

        # NESTED CONSTRAINT #1 — Inner has additionalProperties:false with
        # the single property 'x'. The key char 'x' must be allowed; 'y'
        # must be masked to -inf.
        x_key = inner.encode("x")[0]
        y_key = inner.encode("y")[0]
        assert _allowed(_refresh_mask(), x_key), (
            "the only legal Inner key char 'x' must be allowed — the nested "
            "$ref object's properties were dropped"
        )
        assert _mask_is_neg_inf(y_key), (
            "an out-of-schema key char 'y' must be driven to -inf inside the "
            "$ref Inner object (additionalProperties:false); it was allowed, "
            "so the nested constraint was not enforced"
        )

        # Finish the Inner object and advance to the 'kind' enum value.
        for t in inner.encode('x":1},"kind":"'):
            assert matcher.consume_token(t)

        # NESTED CONSTRAINT #2 — 'kind' is an enum of {'a','b'}. 'a' must be
        # allowed; 'c' must be masked to -inf.
        a_val = inner.encode("a")[0]
        c_val = inner.encode("c")[0]
        assert _allowed(_refresh_mask(), a_val), (
            "enum value 'a' must be allowed at the 'kind' position"
        )
        assert _mask_is_neg_inf(c_val), (
            "out-of-enum value 'c' must be driven to -inf at the 'kind' enum "
            "position; it was allowed, so the enum constraint was not enforced"
        )

        # ---- Finish a full valid document and prove it reaches accepting
        #      and validates against the COMPLETE schema. ----
        for t in inner.encode('a","tag":null}'):
            assert matcher.consume_token(t), (
                f"token {t} ({inner.decode([t])!r}) rejected while completing "
                "the valid document"
            )
        assert matcher.is_accepting(), (
            "the completed document did not drive the matcher to an accepting "
            "state — the nested grammar is over- or under-constrained"
        )
        import jsonschema

        produced = json.loads('{"inner":{"x":1},"kind":"a","tag":null}')
        assert produced == {"inner": {"x": 1}, "kind": "a", "tag": None}
        jsonschema.validate(produced, schema)  # raises if the doc is invalid

    def test_json_object_mode_actually_constrains_bug1(self, wrapped_tokenizer):
        """BUG-1 regression: ``generate_json_object`` must produce a REAL
        constraint (not silently unconstrained).

        Two independent proofs:

        1. Step-0 mask: under the json_object grammar a plain-prose token
           (``Sure``) is masked to ``-inf`` at the opener while ``{`` stays
           finite — the model literally *cannot* emit prose there.
        2. Full decode: a model whose plan is a valid object ``{"ok":1}``
           decodes to exactly that and reaches the grammar's accepting
           state — with the KV cache threaded, the context-dependent fake
           model can only reproduce the object if it kept full context.
        """
        import math

        import mlx.core as mx
        from llguidance.mlx import (
            LLMatcher,
            allocate_token_bitmask,
            apply_token_bitmask,
            fill_next_token_bitmask,
        )

        inner = wrapped_tokenizer._tokenizer

        # ---- Proof 1: prose is masked at the opener. ----
        import vllm_mlx.api.guided as guided

        gen0 = guided.GuidedGenerator.__new__(guided.GuidedGenerator)
        gen0._tokenizer = wrapped_tokenizer
        gen0._lltokenizer = False
        lltok = gen0._get_lltokenizer()
        grammar = LLMatcher.grammar_from_json_schema(
            json.dumps({"type": "object"}),
            overrides={"whitespace_flexible": True},
        )
        matcher = LLMatcher(lltok, grammar)
        bitmask = allocate_token_bitmask(1, lltok.vocab_size)
        fill_next_token_bitmask(matcher, bitmask, 0)
        masked = apply_token_bitmask(mx.ones((1, lltok.vocab_size)), bitmask)
        brace = inner.encode("{")[0]
        prose = inner.encode("Sure")[0]
        assert math.isfinite(float(masked[0, brace].item())), (
            "'{' must stay finite at a json_object opener"
        )
        assert float(masked[0, prose].item()) == float("-inf"), (
            "json_object mode did not mask a prose token at the opener — "
            "BUG-1 (silently unconstrained) is back"
        )

        # ---- Proof 2: a valid-object plan decodes to exactly that object
        #      and completes the grammar. ----
        plan = inner.encode('{"ok":1}')
        gen = self._make_generator(wrapped_tokenizer, plan)[0]
        out = gen.generate_json_object("prompt", max_tokens=16, temperature=0.0)
        assert out is not None, (
            "constrained decode returned None — grammar never completed or "
            "the KV cache was not threaded"
        )
        assert out.lstrip().startswith("{"), (
            f"json_object mode did not constrain to an object opener; "
            f"got {out!r} — BUG-1 (silently unconstrained) is back"
        )
        assert json.loads(out) == {"ok": 1}, (
            f"expected exactly {{'ok': 1}} but got {out!r}; a mismatch means "
            "the decode loop lost context (missing KV cache)"
        )


# ---------------------------------------------------------------------------
# generate_with_schema convenience wrapper
# ---------------------------------------------------------------------------


class TestGenerateWithSchema:
    def test_returns_none_when_llguidance_not_available(self):
        import vllm_mlx.api.guided as guided

        original = guided.HAS_LLGUIDANCE
        try:
            guided.HAS_LLGUIDANCE = False
            result = guided.generate_with_schema(
                model=object(),
                tokenizer=object(),
                prompt="Generate",
                json_schema={"type": "object", "properties": {}},
            )
            assert result is None
        finally:
            guided.HAS_LLGUIDANCE = original

    @requires_guided
    def test_generate_with_schema_delegates_to_generate_json(self):
        """``generate_with_schema`` must construct a GuidedGenerator and
        return whatever ``generate_json`` produced — verified by patching
        ``generate_json`` to a sentinel."""
        import vllm_mlx.api.guided as guided

        called = {}

        def _fake_generate_json(self, *, prompt, json_schema, max_tokens, temperature):
            called["args"] = (prompt, json_schema, max_tokens, temperature)
            return '{"ok": true}'

        original = guided.GuidedGenerator.generate_json
        # Patch __init__ so we don't need a real model/tokenizer.
        original_init = guided.GuidedGenerator.__init__

        def _fake_init(self, model, tokenizer):
            self._model = model
            self._tokenizer = tokenizer
            self._lltokenizer = False

        try:
            guided.GuidedGenerator.generate_json = _fake_generate_json
            guided.GuidedGenerator.__init__ = _fake_init
            result = guided.generate_with_schema(
                model=object(),
                tokenizer=object(),
                prompt="Generate",
                json_schema={"type": "object", "properties": {"a": {"type": "string"}}},
                max_tokens=50,
                temperature=0.3,
            )
            assert result == '{"ok": true}'
            assert called["args"][2] == 50
            assert called["args"][3] == 0.3
        finally:
            guided.GuidedGenerator.generate_json = original
            guided.GuidedGenerator.__init__ = original_init


# ---------------------------------------------------------------------------
# Edge cases for json_schema_to_pydantic (public helper)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_schema_with_required(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {"type": "object", "properties": {}, "required": []}
        model = json_schema_to_pydantic(schema)
        assert model is not None
        instance = model.model_validate({})
        assert instance is not None

    def test_schema_with_all_optional_fields(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {"field1": {"type": "string"}, "field2": {"type": "integer"}},
        }
        model = json_schema_to_pydantic(schema)
        assert model is not None
        instance = model.model_validate({})
        assert instance.field1 is None
        assert instance.field2 is None

    def test_array_with_integer_items(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
        }
        model = json_schema_to_pydantic(schema)
        assert model is not None
        instance = model.model_validate({"ids": [1, 2, 3]})
        assert instance.ids == [1, 2, 3]

    def test_array_with_boolean_items(self):
        from vllm_mlx.api.guided import json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {"flags": {"type": "array", "items": {"type": "boolean"}}},
        }
        model = json_schema_to_pydantic(schema)
        assert model is not None
        instance = model.model_validate({"flags": [True, False, True]})
        assert instance.flags == [True, False, True]


# ---------------------------------------------------------------------------
# Schema-passthrough contract: llguidance sees the RAW schema, not a
# pydantic-reduced superset.
# ---------------------------------------------------------------------------


@requires_guided
class TestGuidedJsonSchemaPassthrough:
    """Contract: ``GuidedGenerator.generate_json`` must hand the *full* JSON
    schema dict to llguidance's ``grammar_from_json_schema`` so the
    constraint engine can interpret ``$defs``/``$ref``/``anyOf``/``enum``/
    numeric bounds/``additionalProperties: false`` itself.

    Pre-migration, the schema was first projected through
    ``json_schema_to_pydantic`` — a shallow converter that silently dropped
    every one of those constructs (waybarrios#546). This test pins that the
    converter is NOT on the hot path and that the raw schema string reaches
    ``grammar_from_json_schema``.
    """

    def test_passes_raw_schema_to_grammar_from_json_schema(self, monkeypatch):
        from llguidance.mlx import LLMatcher

        import vllm_mlx.api.guided as guided

        captured = {}
        real = LLMatcher.grammar_from_json_schema

        def _spy(schema_str, *args, **kwargs):
            captured["schema_str"] = schema_str
            captured["overrides"] = kwargs.get("overrides")
            return real(schema_str, *args, **kwargs)

        monkeypatch.setattr(LLMatcher, "grammar_from_json_schema", staticmethod(_spy))

        schema = {
            "type": "object",
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                }
            },
            "properties": {
                "inner": {"$ref": "#/$defs/Inner"},
                "kind": {"type": "string", "enum": ["a", "b"]},
            },
            "required": ["inner", "kind"],
            "additionalProperties": False,
        }

        # Build a generator whose tokenizer resolves to None so decoding
        # short-circuits after the grammar compile (we only assert the
        # compile args here).
        gen = guided.GuidedGenerator.__new__(guided.GuidedGenerator)

        class _NoFastTok:
            def encode(self, s):
                return [0]

        gen._tokenizer = _NoFastTok()
        gen._model = object()
        gen._lltokenizer = False

        gen.generate_json(prompt="hi", json_schema=schema, max_tokens=8)

        assert "schema_str" in captured, (
            "generate_json did not call grammar_from_json_schema at all"
        )
        # The captured schema string must round-trip to the EXACT raw
        # schema — no pydantic reduction, all $defs/$ref/enum preserved.
        assert json.loads(captured["schema_str"]) == schema
        # And the whitespace override must be present. It is
        # ``whitespace_flexible: True`` (the default): forbidding structural
        # whitespace masks a chat model's natural space-prefixed string
        # opener and derails greedy decoding on unbounded string fields into
        # fluent-but-wrong content (proved on a real model), so the grammar
        # must permit optional whitespace.
        assert captured["overrides"] == {"whitespace_flexible": True}

    def test_converter_not_called_from_generate_json(self, monkeypatch):
        """Negative control: ``json_schema_to_pydantic`` must NOT be invoked
        from ``generate_json`` — using it re-introduces the bug."""
        import vllm_mlx.api.guided as guided

        calls = []

        def _trap(_schema):
            calls.append(_schema)
            raise AssertionError(
                "json_schema_to_pydantic must not be called from "
                "generate_json — it silently drops $defs/$ref/anyOf/enum."
            )

        monkeypatch.setattr(guided, "json_schema_to_pydantic", _trap)

        gen = guided.GuidedGenerator.__new__(guided.GuidedGenerator)

        class _NoFastTok:
            def encode(self, s):
                return [0]

        gen._tokenizer = _NoFastTok()
        gen._model = object()
        gen._lltokenizer = False

        gen.generate_json(
            prompt="hi",
            json_schema={"type": "object", "properties": {}},
            max_tokens=8,
        )
        assert calls == [], (
            "generate_json invoked the shallow pydantic converter instead "
            "of grammar_from_json_schema"
        )
