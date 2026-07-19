# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the hermes/qwen ``structure_info()`` overrides (#558 PR-2).

PR-1 shipped the grammar builder plus a NON-BREAKING ``structure_info() ->
None`` ABC default, and proved the builder against a *test-local* hermes-style
stub. PR-2 lands the concrete per-family overrides on the REAL
``HermesToolParser`` and ``QwenToolParser`` (which share the
``<tool_call>…</tool_call>`` JSON-body wire). These tests therefore drive the
grammar path through the ACTUAL shipped parsers — not a stub — to prove:

  * each real parser's ``structure_info()`` returns a ``name -> StructureInfo``
    factory whose wire triple is the hermes ``<tool_call>`` JSON body, with the
    ``<tool_call>``/``</tool_call>`` single special tokens declared as
    ``sentinels`` (ground-truth correction #1);
  * feeding a real parser through ``build_tool_grammar`` yields a Lark with the
    ``<tool_call>`` bare special-token trigger + a ``%json`` schema-constraint
    region for the ``arguments`` object;
  * grammar ENFORCEMENT via llguidance ``LLMatcher``: a well-formed hermes/qwen
    tool call is ACCEPTED in full (and is a terminal/accepting state) while a
    hallucinated tool name, an off-schema argument, and a bad enum value are
    REJECTED mid-stream.

Scope note: this PR teaches hermes/qwen to DESCRIBE their grammar; NOTHING in
the request path calls ``structure_info()`` yet (chat.py/scheduler.py routing +
the runtime ``GrammarLogitsProcessor`` are PR-3). So these tests are hermetic —
no server, no decode loop, no live network beyond the pinned tokenizer fetch —
and the overrides are pure no-behavior-change scaffolding until PR-3 wires them.

The enforcement tests need a fast (Rust) tokenizer whose
``<tool_call>``/``</tool_call>`` are single special tokens — the pilot verified
this on ``mlx-community/Qwen3.5-4B-MLX-4bit`` (pinned by revision below for an
immutable artifact). Those tests skip ONLY on genuine unavailability
(llguidance extra absent, or the tokenizer neither cached nor reachable); any
OTHER failure is surfaced, not swallowed. The pure-Python structure-triple and
Lark-structure tests never skip — they carry no optional dependency.
"""

import importlib.util

import pytest

# llguidance is only needed by the grammar-BUILD / enforcement tests (they
# compile a Lark grammar / build an LLTokenizer). The structure-triple and
# pure-Lark-string tests need NOTHING optional, so we do NOT skip at module
# level — a repo without the [guided] extra still exercises the real parsers'
# structure_info triples and the builder's string output.
_HAS_LLGUIDANCE = importlib.util.find_spec("llguidance") is not None
_requires_llguidance = pytest.mark.skipif(
    not _HAS_LLGUIDANCE, reason="llguidance ([guided] extra) not installed"
)

_TOKENIZER_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
# Pin the revision so the enforcement proof runs against an IMMUTABLE artifact
# (an unpinned Hub revision is a mutable third-party dependency). The
# tokenizer's <tool_call>/</tool_call> single-special-token layout is fixed at
# this commit; a different upstream revision must not silently change what the
# enforcement tests exercise. Same pin as tests/test_tool_grammar_558.py.
_TOKENIZER_REVISION = "32f3e8ecf65426fc3306969496342d504bfa13f3"

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

# The two families that share the <tool_call>…</tool_call> JSON-body wire and
# opt into grammar constraint in this PR. Parametrizing over the REAL parser
# classes proves BOTH overrides (not a shared stub) with one test body.
_PARSER_IMPORTS = {
    "hermes": ("vllm_mlx.tool_parsers.hermes_tool_parser", "HermesToolParser"),
    "qwen": ("vllm_mlx.tool_parsers.qwen_tool_parser", "QwenToolParser"),
}


def _make_parser(family: str, tokenizer=None):
    import importlib

    module_name, cls_name = _PARSER_IMPORTS[family]
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(tokenizer=tokenizer)


# --------------------------------------------------------------------------
# structure_info() wire triple (pure Python, always runs — no tokenizer,
# no llguidance: structure_info() is stateless w.r.t. the tokenizer).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_structure_info_returns_hermes_wire_triple(family):
    from vllm_mlx.api.tool_grammar import StructureInfo

    parser = _make_parser(family)
    get_info = parser.structure_info()
    # PR-2 opt-in: the override returns a name->StructureInfo factory, not None.
    assert callable(get_info), f"{family}.structure_info() must return a callable"

    si = get_info("get_weather")
    assert isinstance(si, StructureInfo)
    # The hermes <tool_call> JSON-body wire, with the concrete tool name
    # substituted into ``begin``.
    assert si.trigger == "<tool_call>"
    assert si.begin == '<tool_call>\n{"name": "get_weather", "arguments": '
    assert si.end == "}\n</tool_call>"
    # begin MUST start with trigger (StructTag invariant the builder enforces).
    assert si.begin.startswith(si.trigger)
    # Ground-truth correction #1: <tool_call>/</tool_call> are SINGLE special
    # tokens, so both must be declared as sentinels (the trigger among them) so
    # the builder renders them as special-token refs, not byte strings.
    assert si.sentinels == ("<tool_call>", "</tool_call>")
    assert si.trigger in si.sentinels


@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_structure_info_substitutes_each_tool_name(family):
    # The factory substitutes whatever concrete name it is given — one triple
    # per tool, so a multi-tool request constrains each tool to ITS own schema.
    parser = _make_parser(family)
    get_info = parser.structure_info()
    for name in ("get_weather", "get_time", "any_other_name"):
        si = get_info(name)
        assert si.begin == f'<tool_call>\n{{"name": "{name}", "arguments": '


@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_hermes_and_qwen_share_identical_wire(family):
    # hermes and qwen intentionally emit the SAME wire triple (both are the
    # <tool_call> JSON body). Assert byte-identical triples so the two overrides
    # can never silently diverge.
    hermes_si = _make_parser("hermes").structure_info()("get_weather")
    other_si = _make_parser(family).structure_info()("get_weather")
    assert other_si == hermes_si


# --------------------------------------------------------------------------
# build_tool_grammar / Lark structure via the REAL parsers (needs llguidance).
# --------------------------------------------------------------------------
@_requires_llguidance
@pytest.mark.parametrize("family", ["hermes", "qwen"])
@pytest.mark.parametrize("tool_choice", ["required", "auto"])
def test_real_parser_builds_grammar(family, tool_choice):
    # Driving the REAL parser (not a stub) through the public builder yields a
    # compiled grammar (non-None) for both required and auto.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    parser = _make_parser(family)
    assert build_tool_grammar(TOOLS, tool_choice, parser) is not None


@_requires_llguidance
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_real_parser_lark_has_trigger_and_schema_region(family):
    # Assemble the Lark from the REAL parser's structure_info and assert the
    # load-bearing structure: <tool_call> as a BARE special-token ref (not a
    # quoted byte literal the single <tool_call> token could never satisfy),
    # </tool_call> bare closing ref, and a %json schema-constraint region.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    get_info = _make_parser(family).structure_info()
    infos = [get_info(t["name"]) for t in TOOLS]
    lark = build_tool_lark(TOOLS, "required", infos)

    assert " <tool_call> " in lark  # space-delimited bare token ref (trigger)
    assert lark.rstrip().endswith("</tool_call>")  # bare closing token ref
    assert '"<tool_call>"' not in lark  # NOT a quoted (multi-byte) literal
    assert '"</tool_call>"' not in lark
    assert "%json" in lark  # arguments constrained by JSON Schema
    assert "get_weather" in lark
    assert "get_time" in lark


# --------------------------------------------------------------------------
# Grammar ENFORCEMENT via offline LLMatcher (the #558 proof, real parsers).
#
# Needs a fast (Rust) tokenizer whose <tool_call>/</tool_call> are single
# special tokens. The ONLY sanctioned skip is genuine tokenizer/llguidance
# UNAVAILABILITY (no network + not cached, or the optional extra absent) — any
# OTHER failure (a real grammar regression, a matcher error, an unexpected
# tokenizer exception) propagates and FAILS the test.
# --------------------------------------------------------------------------
def _offline_skip_exc_types():
    """Only genuine network/cache-miss errors are a sanctioned skip.

    A corrupt tokenizer artifact, an invalid revision, or a tokenizer/config
    incompatibility must FAIL the enforcement test, not silently skip it. So we
    skip ONLY on the specific huggingface_hub offline/cache-miss signals (and a
    raw connection error), letting every other exception propagate.
    """
    types: list[type[BaseException]] = []
    try:
        from huggingface_hub.errors import (
            LocalEntryNotFoundError,
            OfflineModeIsEnabled,
        )

        types += [LocalEntryNotFoundError, OfflineModeIsEnabled]
    except Exception:  # pragma: no cover - old hub without these names
        pass
    try:
        from requests.exceptions import ConnectionError as _ReqConnErr

        types.append(_ReqConnErr)
    except Exception:  # pragma: no cover - requests not present
        pass
    return tuple(types) or (OSError,)


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            _TOKENIZER_MODEL, revision=_TOKENIZER_REVISION
        )
    except _offline_skip_exc_types():  # pragma: no cover - offline & uncached
        pytest.skip(
            f"tokenizer {_TOKENIZER_MODEL}@{_TOKENIZER_REVISION[:8]} not "
            "cached and no network — enforcement tests require it"
        )


@pytest.fixture(scope="module")
def lltok(tok):
    """Build an llguidance LLTokenizer from the fast (Rust) tokenizer.

    Mirrors ``guided.py``'s tokenizer resolution: try the wrapper's inner fast
    tokenizer, then the object itself. A slow tokenizer is the one sanctioned
    skip; a genuine ``from_tokenizer`` regression is NOT swallowed.
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
    raise AssertionError(
        f"llguidance could not build an LLTokenizer from any fast candidate: "
        f"{last_exc!r}"
    )


def _consume(grammar, lltok, tok, text):
    """Offline enforcement probe. Returns ``(accepted, total, is_accepting)``.

    Advances real grammar state one token at a time via
    ``LLMatcher.consume_tokens`` (which returns a bool per batch), counting how
    many tokens the grammar accepts before it rejects one. Because this ADVANCES
    matcher state, afterwards ``is_accepting()`` reports whether the grammar can
    TERMINATE there — so a "fully accepted" positive test proves the string is a
    COMPLETE valid derivation, not merely an accepted prefix.
    """
    from llguidance.mlx import LLMatcher

    ids = tok.encode(text, add_special_tokens=False)
    matcher = LLMatcher(lltok, grammar)
    assert not matcher.get_error(), matcher.get_error()
    accepted = 0
    for tid in ids:
        if not matcher.consume_tokens([tid]):
            break
        accepted += 1
    return accepted, len(ids), matcher.is_accepting()


@_requires_llguidance
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_valid_call_is_accepted_and_terminates(family, tok, lltok):
    # A well-formed call through the REAL parser's grammar is accepted in full
    # AND is a terminal/accepting state (a complete valid derivation).
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _make_parser(family))
    assert grammar is not None
    accepted, total, accepting = _consume(
        grammar,
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
    )
    assert accepted == total, f"valid {family} call rejected ({accepted}/{total})"
    assert accepting, f"valid complete {family} call is not an accepting state"


@_requires_llguidance
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_valid_enum_value_is_accepted(family, tok, lltok):
    # Positive enum control (paired with the rejection test below): a VALID enum
    # value is accepted and terminates — so the rejection test cannot pass merely
    # because the grammar forbids the optional `unit` property entirely.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _make_parser(family))
    accepted, total, accepting = _consume(
        grammar,
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "P", "unit": "c"}}\n</tool_call>',
    )
    assert accepted == total, (
        f"valid enum value rejected for {family} ({accepted}/{total})"
    )
    assert accepting, f"valid enum call is not an accepting state for {family}"


@_requires_llguidance
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_hallucinated_tool_name_is_rejected(family, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _make_parser(family))
    accepted, total, _ = _consume(
        grammar, lltok, tok, '<tool_call>\n{"name": "get_stockquote'
    )
    assert accepted < total, f"hallucinated tool name NOT rejected for {family}"


@_requires_llguidance
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_off_schema_argument_is_rejected(family, tok, lltok):
    # `city` must be a string; an integer must be forbidden.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _make_parser(family))
    accepted, total, _ = _consume(
        grammar,
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": 4',
    )
    assert accepted < total, f"off-schema integer argument NOT rejected for {family}"


@_requires_llguidance
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_bad_enum_value_is_rejected(family, tok, lltok):
    # `unit` enum is {c, f}; "kelvin" must be forbidden.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _make_parser(family))
    accepted, total, _ = _consume(
        grammar,
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "P", "unit": "kelvin',
    )
    assert accepted < total, f"invalid enum value NOT rejected for {family}"
