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
verified this on ``mlx-community/Qwen3.5-4B-MLX-4bit``. Those tests skip ONLY
on genuine unavailability (llguidance extra absent, or the tokenizer neither
cached nor reachable); any other failure is surfaced, not swallowed. The
pure-Python ABC and Lark-structure tests never skip — they carry no optional
dependency and always run.
"""

import importlib.util

import pytest

# NOTE: llguidance is only needed by the grammar-BUILD and enforcement tests
# below (they compile a Lark grammar / build an LLTokenizer). The ABC-contract
# and pure-Lark-string tests need NOTHING optional, so we do NOT skip at module
# level — a repo without the [guided] extra still exercises the ABC change and
# the builder's string output. Tests that need llguidance guard themselves via
# ``_requires_llguidance``.
_HAS_LLGUIDANCE = importlib.util.find_spec("llguidance") is not None
_requires_llguidance = pytest.mark.skipif(
    not _HAS_LLGUIDANCE, reason="llguidance ([guided] extra) not installed"
)

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


@_requires_llguidance
def test_build_tool_grammar_none_when_parser_opts_out():
    # A parser whose structure_info() returns None -> builder returns None
    # (free-form fallback), NOT a grammar. Requires llguidance so the opt-out
    # branch is reached rather than the ``HAS_LLGUIDANCE`` short-circuit
    # (which would make this pass for the wrong reason).
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    class _OptOut:
        def structure_info(self):
            return None

    assert build_tool_grammar(TOOLS, "required", _OptOut()) is None


@_requires_llguidance
def test_build_tool_grammar_none_on_empty_tools():
    # Empty tools -> None. Requires llguidance so the empty-tools guard is the
    # reason for None, not the availability short-circuit.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    assert build_tool_grammar([], "required", _HermesStubParser()) is None


@_requires_llguidance
def test_build_tool_grammar_none_for_tool_choice_none():
    # tool_choice="none" -> no grammar at all (design §4), never a forced call.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    assert build_tool_grammar(TOOLS, "none", _HermesStubParser()) is None


@_requires_llguidance
def test_build_tool_grammar_degrades_when_factory_raises():
    # A per-family structure_info() factory that raises on a tool name must
    # degrade to free-form (None), not crash the request.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    class _Raises:
        def structure_info(self):
            def _boom(name):
                raise RuntimeError("boom")

            return _boom

    assert build_tool_grammar(TOOLS, "required", _Raises()) is None


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
    # required (and any non-auto choice) -> at least one call -> (...)+
    assert "start: (tag_0 | tag_1)+ tag_end" in build_tool_lark(
        TOOLS, "required", infos
    )


def test_named_choice_narrows_to_single_forced_tag():
    # A NAMED tool_choice is expressed by the caller narrowing ``tools`` to the
    # single requested function before calling the builder (design §4). The
    # builder then emits exactly one forced tag — it never leaks the other
    # tools' alternatives into a named request.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    only = [TOOLS[0]]  # caller pre-filtered to the requested function
    info = [_hermes_structure_info()(only[0]["name"])]
    lark = build_tool_lark(only, "get_weather", info)
    assert "start: (tag_0)+ tag_end" in lark
    assert "tag_1" not in lark  # no other tool's alternative present
    assert "get_time" not in lark


def test_build_tool_lark_rejects_bad_inputs():
    # Public-ish input validation raises ValueError (survives ``python -O``),
    # rather than asserting.
    from vllm_mlx.api.tool_grammar import StructureInfo, build_tool_lark

    good = _hermes_structure_info()("get_weather")
    with pytest.raises(ValueError):
        build_tool_lark([], "required", [])
    with pytest.raises(ValueError):
        # length mismatch
        build_tool_lark(TOOLS, "required", [good])
    with pytest.raises(ValueError):
        # tool_choice="none" must never build a grammar
        build_tool_lark([TOOLS[0]], "none", [good])
    with pytest.raises(ValueError):
        # begin does not start with trigger -> invariant violation
        bad = StructureInfo(
            begin="oops", end="", trigger="<tool_call>", sentinels=("<tool_call>",)
        )
        build_tool_lark([TOOLS[0]], "required", [bad])
    with pytest.raises(ValueError):
        # trigger not declared as a special-token sentinel -> rejected
        bad = StructureInfo(begin="<x>go", end="", trigger="<x>", sentinels=())
        build_tool_lark([TOOLS[0]], "required", [bad])


def test_build_tool_lark_preserves_falsy_schemas():
    # A present-but-falsy JSON Schema ({} = allow-any, false = allow-none) is
    # meaningful and must be embedded verbatim, NOT replaced by the permissive
    # default (the ``... or default`` bug codex flagged).
    from vllm_mlx.api.tool_grammar import build_tool_lark

    info = _hermes_structure_info()("get_weather")
    tool_empty = {"name": "get_weather", "parameters": {}}
    lark = build_tool_lark([tool_empty], "required", [info])
    assert "%json {}" in lark  # empty schema preserved, not defaulted

    tool_false = {"name": "get_weather", "parameters": False}
    lark = build_tool_lark([tool_false], "required", [info])
    assert "%json false" in lark  # false schema preserved verbatim


def test_build_tool_lark_defaults_only_when_parameters_absent():
    from vllm_mlx.api.tool_grammar import build_tool_lark

    info = _hermes_structure_info()("get_weather")
    tool_missing = {"name": "get_weather"}  # no "parameters" key
    lark = build_tool_lark([tool_missing], "required", [info])
    assert '%json {"type": "object", "properties": {}}' in lark


# --------------------------------------------------------------------------
# Grammar ENFORCEMENT via offline validate_tokens (the #558 proof).
#
# These need a fast (Rust) tokenizer whose <tool_call>/</tool_call> are single
# special tokens. The ONLY sanctioned skip is genuine tokenizer/llguidance
# UNAVAILABILITY (no network + not cached, or the optional extra absent) — any
# OTHER failure (a real grammar regression, a matcher error, an unexpected
# tokenizer exception) propagates and FAILS the test, so a green run of these
# tests always means enforcement was actually exercised.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(_TOKENIZER_MODEL)
    except OSError:  # pragma: no cover - offline & uncached
        pytest.skip(
            f"tokenizer {_TOKENIZER_MODEL} not cached and no network — "
            "enforcement tests require it"
        )


@pytest.fixture(scope="module")
def lltok(tok):
    """Build an llguidance LLTokenizer from the fast (Rust) tokenizer.

    Mirrors ``guided.py``'s tokenizer resolution: try the wrapper's inner
    fast tokenizer, then the object itself (transformers 5.x exposes a
    ``TokenizersBackend`` that IS the fast tokenizer llguidance wants). A
    slow tokenizer is the one sanctioned skip; a genuine ``from_tokenizer``
    regression is NOT swallowed — it fails the test.
    """
    import llguidance.hf as llg_hf

    candidates = []
    inner = getattr(tok, "_tokenizer", None)
    if inner is not None:
        candidates.append(inner)
    candidates.append(tok)
    fast_candidates = [
        c for c in candidates if getattr(c, "is_fast", True) is not False
    ]
    if not fast_candidates:
        pytest.skip("tokenizer is not a fast tokenizer — llguidance needs one")
    last_exc = None
    for cand in fast_candidates:
        try:
            return llg_hf.from_tokenizer(cand)
        except Exception as exc:  # noqa: BLE001 - re-raised below if all fail
            last_exc = exc
    # Every fast candidate raised — that is a real regression, not an
    # environment gap. Surface it rather than skipping.
    raise AssertionError(
        f"llguidance could not build an LLTokenizer from any fast candidate: "
        f"{last_exc!r}"
    )


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


@_requires_llguidance
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


@_requires_llguidance
def test_hallucinated_tool_name_is_rejected(tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _HermesStubParser())
    full, accepted, total = _accepts_full(
        grammar, lltok, tok, '<tool_call>\n{"name": "get_stockquote'
    )
    assert not full, "hallucinated tool name was NOT rejected by the grammar"


@_requires_llguidance
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


@_requires_llguidance
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
