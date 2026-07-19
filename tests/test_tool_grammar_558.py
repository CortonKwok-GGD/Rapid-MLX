# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the grammar-constrained tool-calling builder (#558 PR-1).

These validate the grammar BUILDER and the ``ToolParser.structure_info()``
ABC contract WITHOUT a model or a decode loop. They exercise:

  * the non-breaking ABC default (``structure_info() -> None``);
  * ``build_tool_lark`` structural output (``<tool_call>`` trigger + a
    ``%json`` schema-constraint region);
  * grammar ENFORCEMENT via llguidance ``LLMatcher.validate_tokens`` — a
    well-formed hermes tool call is accepted in full while a hallucinated
    tool name, an off-schema argument, and a bad enum value are rejected
    mid-stream. This is the load-bearing #558 proof that the constraint is
    grammar-enforced, not merely post-parsed.

No routing, no scheduler, no ``GrammarLogitsProcessor`` (those are PR-3). The
per-family concrete overrides are PR-2; here we drive the builder with a
test-local hermes-style parser stub so PR-1 ships zero behavior change while
still proving the builder against a realistic hermes wire format.

The negative-control tests need a fast (Rust) tokenizer whose
``<tool_call>``/``</tool_call>`` are single special tokens — the pilot
verified this on ``mlx-community/Qwen3.5-4B-MLX-4bit``. If that tokenizer or
llguidance is unavailable the enforcement tests skip; the pure-Python ABC and
Lark-structure tests always run.
"""

import pytest

pytest.importorskip("llguidance")

_TOKENIZER_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"

TOOLS = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_time",
        "parameters": {
            "type": "object",
            "properties": {"tz": {"type": "string"}},
            "required": ["tz"],
            "additionalProperties": False,
        },
    },
]


def _hermes_structure_info():
    """A hermes ``<tool_call>`` JSON-body wire triple, as PR-2 will ship it.

    Declared here (not on the concrete parser) so PR-1 leaves every shipped
    parser's behavior untouched while still exercising the builder against a
    realistic family. ``<tool_call>``/``</tool_call>`` are single special
    tokens in Qwen3/Hermes tokenizers, hence the ``sentinels`` entries.
    """
    from vllm_mlx.api.tool_grammar import StructureInfo

    def _info(name: str):
        return StructureInfo(
            begin=f'<tool_call>\n{{"name": "{name}", "arguments": ',
            end="}\n</tool_call>",
            trigger="<tool_call>",
            sentinels=("<tool_call>", "</tool_call>"),
        )

    return _info


class _HermesStubParser:
    """Minimal parser exposing only ``structure_info`` for builder tests."""

    def structure_info(self):
        return _hermes_structure_info()


# --------------------------------------------------------------------------
# ABC contract (pure Python, always runs).
# --------------------------------------------------------------------------
def test_abc_structure_info_defaults_to_none():
    # PR-1 non-breaking contract: a parser that does not override
    # ``structure_info`` returns None, so callers fall back to today's
    # free-form-then-parse behavior.
    from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

    class _Dummy(ToolParser):
        def extract_tool_calls(self, model_output, request=None):  # noqa: D401
            raise NotImplementedError

    assert _Dummy(tokenizer=None).structure_info() is None


def test_build_tool_grammar_none_when_parser_opts_out():
    # A parser whose structure_info() returns None -> builder returns None
    # (free-form fallback), NOT a grammar.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    class _OptOut:
        def structure_info(self):
            return None

    assert build_tool_grammar(TOOLS, "required", _OptOut()) is None


def test_build_tool_grammar_none_on_empty_tools():
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    assert build_tool_grammar([], "required", _HermesStubParser()) is None


# --------------------------------------------------------------------------
# Lark structural output (pure Python, always runs).
# --------------------------------------------------------------------------
def test_lark_contains_trigger_and_schema_region():
    from vllm_mlx.api.tool_grammar import build_tool_lark

    infos = [_hermes_structure_info()(t["name"]) for t in TOOLS]
    lark = build_tool_lark(TOOLS, "required", infos)

    # trigger sentinel is emitted as a Lark special-token ref (bare <tool_call>)
    assert "<tool_call>" in lark
    assert "</tool_call>" in lark
    # a %json schema-constraint region is present for the arguments object
    assert "%json" in lark
    # the concrete tool names are substituted into the begin bodies
    assert "get_weather" in lark
    assert "get_time" in lark


def test_lark_quantifier_tracks_tool_choice():
    from vllm_mlx.api.tool_grammar import build_tool_lark

    infos = [_hermes_structure_info()(t["name"]) for t in TOOLS]
    # auto -> may emit zero calls -> (...)*
    assert "start: (tag_0 | tag_1)* tag_end" in build_tool_lark(TOOLS, "auto", infos)
    # required/named -> at least one call -> (...)+
    assert "start: (tag_0 | tag_1)+ tag_end" in build_tool_lark(
        TOOLS, "required", infos
    )


# --------------------------------------------------------------------------
# Grammar ENFORCEMENT via offline validate_tokens (the #558 proof).
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(_TOKENIZER_MODEL)
    except Exception:  # pragma: no cover - network / cache miss
        pytest.skip("hermes-wire tokenizer not available")


@pytest.fixture(scope="module")
def lltok(tok):
    """Build an llguidance LLTokenizer from the fast (Rust) tokenizer.

    Mirrors ``guided.py``'s tokenizer resolution: try the wrapper's inner
    fast tokenizer, then the object itself (transformers 5.x exposes a
    ``TokenizersBackend`` that IS the fast tokenizer llguidance wants).
    """
    import llguidance.hf as llg_hf

    candidates = []
    inner = getattr(tok, "_tokenizer", None)
    if inner is not None:
        candidates.append(inner)
    candidates.append(tok)
    for cand in candidates:
        if getattr(cand, "is_fast", True) is False:
            continue
        try:
            return llg_hf.from_tokenizer(cand)
        except Exception:
            continue
    pytest.skip("could not build an LLTokenizer for this tokenizer")


def _accepts_full(grammar, lltok, tok, text):
    """Offline check: does the grammar accept EVERY token of ``text``?

    Uses ``LLMatcher.validate_tokens`` — the count of the longest accepted
    token prefix. ``== len(ids)`` means fully accepted; anything less means
    the grammar mask forbids some token (enforcement).
    """
    from llguidance.mlx import LLMatcher

    ids = tok.encode(text, add_special_tokens=False)
    matcher = LLMatcher(lltok, grammar)
    assert not matcher.get_error(), matcher.get_error()
    accepted = matcher.validate_tokens(ids)
    return accepted == len(ids), accepted, len(ids)


def test_valid_hermes_call_is_accepted(tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _HermesStubParser())
    assert grammar is not None
    full, accepted, total = _accepts_full(
        grammar,
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
    )
    assert full, f"valid hermes call unexpectedly rejected ({accepted}/{total})"


def test_hallucinated_tool_name_is_rejected(tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _HermesStubParser())
    full, accepted, total = _accepts_full(
        grammar, lltok, tok, '<tool_call>\n{"name": "get_stockquote'
    )
    assert not full, "hallucinated tool name was NOT rejected by the grammar"


def test_off_schema_argument_is_rejected(tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _HermesStubParser())
    # `city` must be a string; an integer must be forbidden.
    full, accepted, total = _accepts_full(
        grammar,
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": 4',
    )
    assert not full, "off-schema integer argument was NOT rejected by the grammar"


def test_bad_enum_value_is_rejected(tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _HermesStubParser())
    # `unit` enum is {c, f}; "kelvin" must be forbidden.
    full, accepted, total = _accepts_full(
        grammar,
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "P", "unit": "kelvin',
    )
    assert not full, "invalid enum value was NOT rejected by the grammar"
