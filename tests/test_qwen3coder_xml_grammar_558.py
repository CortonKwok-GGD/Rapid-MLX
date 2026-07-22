# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the Qwen3-Coder XML grammar constraint (#558 E3).

The JSON-body families (hermes / qwen / harmony) constrain their ``arguments``
as a single ``%json`` object. Qwen3-Coder instead emits an XML arg body —
``<function=NAME>\\n<parameter=KEY>\\nVALUE\\n</parameter>\\n...</function>`` — so
its ``structure_info()`` declares ``arg_style="xml"`` and ``build_tool_lark``
emits a per-parameter XML body (``_emit_xml_arg_body`` / ``_emit_xml_param_value``)
instead of a whole-object ``%json``. These tests prove that path WITHOUT a model
or a decode loop:

  * ``build_tool_lark`` golden output for ``arg_style="xml"`` (function/parameter
    frame, string values as the LAZY ``xml_param_value`` rule, enum alternation,
    ``%json`` scalars with ``$defs``/``$ref`` propagation, required vs optional
    ``( )?``);
  * GAP #1 (the E3 fix): a string parameter value containing a literal ``<``
    (a ``code`` arg such as ``a < b``, ``<html>...``, C++ ``vector<int>``) is
    ACCEPTED and terminates at the real ``</parameter>`` — the pilot's
    ``XMLSTR: /[^<]*/`` terminal SILENTLY TRUNCATED such a value at the first
    ``<``. The fix ports llguidance's own ``[lazy]`` lexeme idiom (the one
    ``StructTag.to_grammar`` uses for "text until a tag"), matching XGrammar's
    ``qwen_xml`` semantics: value = any text up to the FIRST ``</parameter>``;
  * grammar ENFORCEMENT via llguidance ``LLMatcher`` on the REAL Qwen3-Coder
    tokenizer: a valid XML call is accepted + terminal, prose before a forced
    call is masked at token 0, and a bad enum / off-schema scalar is rejected;
  * ROUND-TRIP: the ``qwen3_coder_xml`` parser parses the constrained wire back
    to ``{name, arguments}`` with correct types (str-with-``<`` / int / bool /
    nested object);
  * REGRESSION GUARD: an ``arg_style="json"`` (hermes/qwen/harmony) build is
    byte-identical to before — it never emits the XML string constructs
    (``XML_PARAM_TEXT`` / ``xml_param_value``) and still uses ``%json``.

The enforcement / round-trip tests use the ACTUAL target model tokenizer
``mlx-community/Qwen3-Coder-Next-4bit`` (pinned by revision below for an
immutable artifact — its ``<tool_call>``/``</tool_call>`` single-special-token
layout and the inner XML byte markers are fixed at this commit). They skip ONLY
on genuine unavailability (llguidance extra absent, or the tokenizer neither
cached nor reachable); any OTHER failure is surfaced, not swallowed. The
pure-Python golden / structure-triple / regression tests never skip.
"""

import importlib.util
import json

import pytest

_HAS_LLGUIDANCE = importlib.util.find_spec("llguidance") is not None
_requires_llguidance = pytest.mark.skipif(
    not _HAS_LLGUIDANCE, reason="llguidance ([guided] extra) not installed"
)

# The REAL Qwen3-Coder target model (the pilot verified the XML wire on it).
# Pin the revision so enforcement runs against an IMMUTABLE artifact.
_TOKENIZER_MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
_TOKENIZER_REVISION = "7b9321eabb85ce79625cac3f61ea691e4ea984b5"

# A representative XML tool: required string + required enum + optional int +
# optional bool — exercises every ``_emit_xml_param_value`` branch (lazy string
# rule, enum alternation, ``%json`` scalar) AND required-vs-optional framing.
XML_TOOLS = [
    {
        "name": "run_code",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "language": {"type": "string", "enum": ["python", "cpp"]},
                "timeout": {"type": "integer"},
                "verbose": {"type": "boolean"},
            },
            "required": ["code", "language"],
            "additionalProperties": False,
        },
    },
]

# A nested-object tool with a ``$ref`` into ``$defs`` — proves ``$defs`` is
# propagated into the per-value ``%json`` sub-schema so the ``$ref`` resolves.
XML_REF_TOOL = [
    {
        "name": "place",
        "parameters": {
            "type": "object",
            "properties": {"origin": {"$ref": "#/$defs/point"}},
            "required": ["origin"],
            "$defs": {
                "point": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        },
    },
]


def _xml_structure_info(name: str):
    """The Qwen3-Coder XML wire triple, exactly as ``Qwen3CoderToolParser``
    ships it (``arg_style="xml"``). Declared test-locally so the pure-Python
    golden tests need no tokenizer; the enforcement tests below drive the REAL
    parser instead."""
    from vllm_mlx.api.tool_grammar import StructureInfo

    return StructureInfo(
        begin=f"<tool_call>\n<function={name}>\n",
        end="</function>\n</tool_call>",
        trigger="<tool_call>",
        sentinels=("<tool_call>", "</tool_call>"),
        arg_style="xml",
    )


def _hermes_json_structure_info(name: str):
    """A hermes ``<tool_call>`` JSON-body wire triple (``arg_style="json"``,
    the default) — the regression baseline the XML change must not perturb."""
    from vllm_mlx.api.tool_grammar import StructureInfo

    return StructureInfo(
        begin=f'<tool_call>\n{{"name": "{name}", "arguments": ',
        end="}\n</tool_call>",
        trigger="<tool_call>",
        sentinels=("<tool_call>", "</tool_call>"),
    )


# --------------------------------------------------------------------------
# Golden Lark for the XML arg body (pure Python, always runs). A CHECKED-IN
# golden — comparing against it (not another call of the same implementation)
# pins the EXACT emitted grammar even if the whole builder drifted.
# --------------------------------------------------------------------------
_XML_GOLDEN_LARK = (
    "%llguidance {}\n"
    "start: (tag_0) (SEP (tag_0))* tag_end\n"
    "tag_end: TAG_TEXT\n"
    "SEP: /[ \\t\\r\\n]*/\n"
    "TAG_TEXT: /(.|\\n)*/\n"
    # The lazy string-value construct: XML_PARAM_TEXT admits ANY byte (``<``
    # included); the lazy rule binds it to the FIRST ``</parameter>``.
    "XML_PARAM_TEXT: /(.|\\n)*/\n"
    'xml_param_value[lazy]: XML_PARAM_TEXT "</parameter>"\n'
    "\n"
    # A string value is the lazy rule + only the trailing ``\\n`` separator
    # (the rule already consumed ``\\n</parameter>``). An enum is a literal
    # alternation, a scalar is ``%json``; both keep the ``\\n</parameter>\\n``
    # close. Optional params are wrapped in ``( ... )?``.
    'tag_0: <tool_call> "\\n<function=run_code>\\n" '
    '"<parameter=code>\\n" xml_param_value "\\n" '
    '"<parameter=language>\\n" ("python" | "cpp") "\\n</parameter>\\n" '
    '( "<parameter=timeout>\\n" %json {"type": "integer"} "\\n</parameter>\\n" )? '
    '( "<parameter=verbose>\\n" %json {"type": "boolean"} "\\n</parameter>\\n" )? '
    '"</function>\\n" </tool_call>\n'
)


def test_xml_lark_matches_golden():
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(XML_TOOLS, "required", [_xml_structure_info("run_code")])
    assert lark == _XML_GOLDEN_LARK


def test_xml_lark_frame_and_sentinels():
    # The function/parameter frame: <tool_call> trigger + </tool_call> close as
    # BARE special-token refs (never quoted byte literals the single token could
    # not satisfy), the <function=NAME> header, and per-parameter blocks.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(XML_TOOLS, "required", [_xml_structure_info("run_code")])
    assert " <tool_call> " in lark  # bare trigger ref
    assert lark.rstrip().endswith("</tool_call>")  # bare closing ref
    assert '"<tool_call>"' not in lark  # NOT a quoted literal
    assert '"</tool_call>"' not in lark
    # XML frame markers are ordinary byte-string literals (multi-token text).
    assert '"\\n<function=run_code>\\n"' in lark
    assert '"<parameter=code>\\n"' in lark
    assert '"</function>\\n"' in lark


def test_xml_string_value_uses_lazy_rule_not_raw_charclass():
    # GAP #1 fix, at the grammar-string level: a string param value must be the
    # LAZY ``xml_param_value`` rule (any text up to the first ``</parameter>``),
    # NOT the pilot's ``XMLSTR: /[^<]*/`` terminal that stopped at any ``<``.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(XML_TOOLS, "required", [_xml_structure_info("run_code")])
    # The lazy value construct is declared once and referenced for the string.
    assert 'xml_param_value[lazy]: XML_PARAM_TEXT "</parameter>"' in lark
    assert "XML_PARAM_TEXT: /(.|\\n)*/" in lark
    assert '"<parameter=code>\\n" xml_param_value "\\n"' in lark
    # The truncating raw terminal is GONE.
    assert "XMLSTR" not in lark
    assert "/[^<]*/" not in lark


def test_xml_required_vs_optional_framing():
    # Required params are mandatory (no quantifier); optional params are wrapped
    # in ``( ... )?``. ``code``/``language`` required; ``timeout``/``verbose`` not.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(XML_TOOLS, "required", [_xml_structure_info("run_code")])
    # Required string: bare (not inside a ``( ... )?`` group).
    assert '"<parameter=code>\\n" xml_param_value "\\n" "<parameter=language>' in lark
    # Optional scalars: each wrapped in an optional group.
    assert (
        '( "<parameter=timeout>\\n" %json {"type": "integer"} "\\n</parameter>\\n" )?'
        in lark
    )
    assert (
        '( "<parameter=verbose>\\n" %json {"type": "boolean"} "\\n</parameter>\\n" )?'
        in lark
    )


def test_xml_enum_is_literal_alternation():
    # An enum value renders as an alternation of the literal enum values (raw
    # string form for string enums), NOT ``%json`` and NOT the lazy string rule.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(XML_TOOLS, "required", [_xml_structure_info("run_code")])
    assert '("python" | "cpp") "\\n</parameter>\\n"' in lark


def test_xml_ref_defs_propagated_into_value_schema():
    # A ``$ref`` value must carry the parent's ``$defs`` into its per-value
    # ``%json`` sub-schema so the ``$ref`` resolves. Golden-compare the emitted
    # ``%json`` payload.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(XML_REF_TOOL, "required", [_xml_structure_info("place")])
    expected_schema = {
        "$ref": "#/$defs/point",
        "$defs": {
            "point": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            }
        },
    }
    assert f"%json {json.dumps(expected_schema)}" in lark


def test_xml_no_string_param_tool_still_declares_lazy_rule():
    # An XML tool with NO string params still declares the lazy rule (arg_style
    # is xml). The unused rule must not break the grammar (a reference-free rule
    # is tolerated) — the enforcement test below compiles exactly such a case.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    int_only = [
        {
            "name": "cfg",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
                "additionalProperties": False,
            },
        }
    ]
    lark = build_tool_lark(int_only, "required", [_xml_structure_info("cfg")])
    assert "xml_param_value[lazy]:" in lark
    assert "xml_param_value" not in lark.split("\ntag_0:", 1)[1]  # unused in the tag


# --------------------------------------------------------------------------
# REGRESSION GUARD: arg_style="json" (hermes/qwen/harmony) is byte-identical —
# the XML change must not leak the XML string constructs into JSON families.
# --------------------------------------------------------------------------
def test_json_family_grammar_has_no_xml_constructs():
    from vllm_mlx.api.tool_grammar import build_tool_lark

    infos = [_hermes_json_structure_info(t["name"]) for t in XML_TOOLS]
    lark = build_tool_lark(XML_TOOLS, "required", infos)
    # JSON body: a single ``%json`` object, none of the XML string constructs.
    assert "%json" in lark
    assert "XML_PARAM_TEXT" not in lark
    assert "xml_param_value" not in lark
    assert "XMLSTR" not in lark
    assert "<parameter=" not in lark


def test_json_family_grammar_byte_identical_to_pre_xml_baseline():
    # The exact hermes forced golden (identical to the checked-in golden in
    # test_tool_grammar_558.py). Pinning it here proves the XML feature left the
    # JSON-family grammar byte-for-byte unchanged.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    tools = [
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
        }
    ]
    infos = [_hermes_json_structure_info("get_weather")]
    expected = (
        "%llguidance {}\n"
        "start: (tag_0) (SEP (tag_0))* tag_end\n"
        "tag_end: TAG_TEXT\n"
        "SEP: /[ \\t\\r\\n]*/\n"
        "TAG_TEXT: /(.|\\n)*/\n"
        "\n"
        'tag_0: <tool_call> "\\n{\\"name\\": \\"get_weather\\", '
        '\\"arguments\\": " %json {"type": "object", "properties": {"city": '
        '{"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, '
        '"required": ["city"], "additionalProperties": false} "}\\n" </tool_call>\n'
    )
    assert build_tool_lark(tools, "required", infos) == expected


# --------------------------------------------------------------------------
# Real Qwen3CoderToolParser.structure_info() (pure Python, hermetic tokenizer
# stub — no network). Proves the parser opts in ONLY when the tokenizer proves
# both sentinels are single special tokens, and returns arg_style="xml".
# --------------------------------------------------------------------------
class _FakeAddedToken:
    def __init__(self, content, special=False):
        self.content = content
        self.special = special


class _FakeTokenizer:
    """Models the surfaces the ``are_single_special_tokens`` guard probes:
    ``<tool_call>``/``</tool_call>`` as distinct single ADDED (special=False)
    tokens that round-trip (the real Qwen3-Coder layout: ids 151657/151658)."""

    def __init__(self, added=None):
        self._added = dict(added or {})
        self._id_to_str = {i: s for s, i in self._added.items()}
        self.added_tokens_decoder = {
            i: _FakeAddedToken(s, special=False) for s, i in self._added.items()
        }

    def encode(self, text, add_special_tokens=False):
        if text in self._added:
            return [self._added[text]]
        return [0, 1]  # ordinary multi-token text

    def decode(self, ids):
        return "".join(self._id_to_str.get(i, "<unk>") for i in ids)

    def get_vocab(self):
        return dict(self._added)


def _single_token_tokenizer():
    return _FakeTokenizer(added={"<tool_call>": 151657, "</tool_call>": 151658})


def _make_qwen3coder(tokenizer=None):
    from vllm_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    return Qwen3CoderToolParser(tokenizer=tokenizer)


def test_qwen3coder_structure_info_opts_out_without_tokenizer():
    # No tokenizer -> cannot prove single-token sentinels -> opt out (None).
    assert _make_qwen3coder(tokenizer=None).structure_info() is None


def test_qwen3coder_structure_info_opts_out_on_multitoken_tokenizer():
    # A tokenizer that encodes <tool_call> as ordinary multi-token text -> opt
    # out rather than build an unenforceable special-token grammar.
    assert _make_qwen3coder(tokenizer=_FakeTokenizer(added={})).structure_info() is None


def test_qwen3coder_structure_info_returns_xml_wire_triple():
    from vllm_mlx.api.tool_grammar import StructureInfo

    get_info = _make_qwen3coder(tokenizer=_single_token_tokenizer()).structure_info()
    assert callable(get_info), "opt-in must return a name->StructureInfo factory"
    si = get_info("run_code")
    assert isinstance(si, StructureInfo)
    assert si.arg_style == "xml"  # the load-bearing distinction from hermes/qwen
    assert si.trigger == "<tool_call>"
    assert si.begin == "<tool_call>\n<function=run_code>\n"
    assert si.end == "</function>\n</tool_call>"
    assert si.begin.startswith(si.trigger)  # builder invariant
    assert si.sentinels == ("<tool_call>", "</tool_call>")
    assert si.trigger in si.sentinels


# --------------------------------------------------------------------------
# Grammar ENFORCEMENT + ROUND-TRIP on the REAL Qwen3-Coder tokenizer.
# --------------------------------------------------------------------------
def _offline_skip_exc_types():
    """Genuine network/cache-miss exceptions that are a sanctioned skip.

    A corrupt tokenizer artifact / invalid revision must FAIL the test, not
    skip it, so we skip ONLY on the specific offline/cache-miss signals.
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
            f"tokenizer {_TOKENIZER_MODEL}@{_TOKENIZER_REVISION[:8]} not cached "
            "and no network — XML enforcement tests require it"
        )


@pytest.fixture(scope="module")
def lltok(tok):
    """Build an llguidance LLTokenizer from the fast (Rust) tokenizer via the
    module's own resolver (the spike-proven candidate-3 direct build handles the
    transformers ``from_tokenizer`` isinstance gotcha ``guided.py`` trips on)."""
    from vllm_mlx.api.tool_grammar import build_lltokenizer

    built = build_lltokenizer(tok)
    if built is None:
        pytest.skip("could not build an LLTokenizer for the Qwen3-Coder tokenizer")
    return built


def _xml_grammar(tools, tool_choice, tok):
    """Compile the XML grammar through the REAL parser (opted-in on ``tok``)."""
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    return build_tool_grammar(tools, tool_choice, _make_qwen3coder(tok))


def _consume(grammar, lltok, tok, text):
    """Offline enforcement probe. Returns ``(accepted, total, is_accepting)`` —
    advances real matcher state so ``is_accepting()`` proves a COMPLETE valid
    derivation, not merely an accepted prefix."""
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


def _wire(code, *, language="python", timeout=None, verbose=None):
    """Build the Qwen3-Coder XML wire for the ``run_code`` tool."""
    s = (
        "<tool_call>\n<function=run_code>\n"
        f"<parameter=code>\n{code}\n</parameter>\n"
        f"<parameter=language>\n{language}\n</parameter>\n"
    )
    if timeout is not None:
        s += f"<parameter=timeout>\n{timeout}\n</parameter>\n"
    if verbose is not None:
        s += f"<parameter=verbose>\n{verbose}\n</parameter>\n"
    s += "</function>\n</tool_call>"
    return s


@_requires_llguidance
def test_xml_valid_call_accepted_and_terminates(tok, lltok):
    grammar = _xml_grammar(XML_TOOLS, "required", tok)
    assert grammar is not None
    accepted, total, accepting = _consume(grammar, lltok, tok, _wire("print(1)"))
    assert accepted == total, f"valid XML call rejected ({accepted}/{total})"
    assert accepting, "valid complete XML call is not an accepting (terminal) state"


@_requires_llguidance
@pytest.mark.parametrize(
    "code",
    [
        "a < b && c > d",
        "<html><body>x</body></html>",
        "vector<int> v",
        "if (a < b) { return a; }",
        "for (int i = 0; i < n; i++) x[i] <<= 1;",
    ],
)
def test_xml_string_value_with_angle_bracket_is_accepted(code, tok, lltok):
    # GAP #1 — the E3 fix. A ``code`` arg containing a literal ``<`` must be
    # ACCEPTED in full and terminate at the real ``</parameter>``. The pilot's
    # ``/[^<]*/`` terminal masked the first ``<`` and SILENTLY TRUNCATED here.
    grammar = _xml_grammar(XML_TOOLS, "required", tok)
    assert grammar is not None
    accepted, total, accepting = _consume(grammar, lltok, tok, _wire(code))
    assert accepted == total, (
        f"`<`-containing code value rejected ({accepted}/{total}) for {code!r} — "
        "gap #1 truncation is back (string value stops at the first `<`)"
    )
    assert accepting, f"`<`-containing value {code!r} is not a terminal state"


@_requires_llguidance
def test_xml_normal_value_still_round_trips_no_regression(tok, lltok):
    # A plain value (no ``<``) must still be accepted + terminal — the lazy rule
    # is a strict superset of the old raw terminal for non-``<`` values.
    grammar = _xml_grammar(XML_TOOLS, "required", tok)
    accepted, total, accepting = _consume(grammar, lltok, tok, _wire("Paris"))
    assert accepted == total and accepting, (
        f"plain value regressed ({accepted}/{total}, accepting={accepting})"
    )


@_requires_llguidance
def test_xml_value_containing_close_tag_closes_at_first(tok, lltok):
    # FIRST-``</parameter>`` semantics (same as XGrammar, acceptable): a value
    # that literally contains ``</parameter>`` closes THERE, so the trailing
    # remainder is not part of a single value and the full wire is not a
    # complete derivation (accepted < total). This pins the documented behavior.
    grammar = _xml_grammar(XML_TOOLS, "required", tok)
    accepted, total, _ = _consume(grammar, lltok, tok, _wire("a</parameter>b"))
    assert accepted < total, (
        "a value literally containing </parameter> must close at the FIRST one "
        "(XGrammar semantics), not swallow the rest of the wire"
    )


@_requires_llguidance
def test_xml_optional_and_scalar_params_enforced(tok, lltok):
    # Optional int/bool params present with valid scalar surface forms are
    # accepted + terminal; the enum on the required ``language`` is honored.
    grammar = _xml_grammar(XML_TOOLS, "required", tok)
    accepted, total, accepting = _consume(
        grammar, lltok, tok, _wire("x=1", language="cpp", timeout=30, verbose="true")
    )
    assert accepted == total and accepting, (
        f"valid call with optional scalars rejected ({accepted}/{total})"
    )


@_requires_llguidance
def test_xml_bad_enum_value_is_rejected(tok, lltok):
    # ``language`` enum is {python, cpp}; "rust" must be masked.
    grammar = _xml_grammar(XML_TOOLS, "required", tok)
    accepted, total, _ = _consume(grammar, lltok, tok, _wire("x", language="rust"))
    assert accepted < total, "invalid enum value was NOT rejected by the grammar"


@_requires_llguidance
def test_xml_off_schema_scalar_is_rejected(tok, lltok):
    # ``timeout`` is an integer; a non-numeric value must be masked by ``%json``.
    grammar = _xml_grammar(XML_TOOLS, "required", tok)
    bad = (
        "<tool_call>\n<function=run_code>\n"
        "<parameter=code>\nx\n</parameter>\n"
        "<parameter=language>\npython\n</parameter>\n"
        "<parameter=timeout>\nnot_an_int"
    )
    accepted, total, _ = _consume(grammar, lltok, tok, bad)
    assert accepted < total, "off-schema (non-integer) timeout was NOT rejected"


@_requires_llguidance
def test_xml_forced_rejects_prose_before_the_call(tok, lltok):
    # Forced (required) non-reasoning: the first call sits AT the trigger with no
    # free prefix, so bare prose before it is masked at token 0.
    grammar = _xml_grammar(XML_TOOLS, "required", tok)
    assert grammar is not None
    prose_then_call = "Sure, let me run that. " + _wire("print(1)")
    accepted, _total, _ = _consume(grammar, lltok, tok, prose_then_call)
    assert accepted == 0, (
        f"forced XML grammar accepted {accepted} prose token(s) before the "
        "trigger — the unbounded leading prefix is back (#558 forced-leak)"
    )


# --------------------------------------------------------------------------
# ROUND-TRIP: the qwen3_coder_xml parser parses the constrained wire back to
# {name, arguments} with correct types — the surface forms the grammar emits
# are exactly what the parser type-converts.
# --------------------------------------------------------------------------
def _parse(wire, tools):
    parser = _make_qwen3coder(tokenizer=None)
    req = {
        "tools": [
            {
                "type": "function",
                "function": {"name": t["name"], "parameters": t["parameters"]},
            }
            for t in tools
        ]
    }
    res = parser.extract_tool_calls(wire, request=req)
    assert res.tools_called, "parser did not detect the tool call"
    tc = res.tool_calls[0]
    return tc["name"], json.loads(tc["arguments"])


@pytest.mark.parametrize("code", ["a < b && c > d", "vector<int> v", "print('ok')"])
def test_roundtrip_string_value_with_angle_bracket(code):
    # The constrained wire round-trips back to the EXACT string value (including
    # ``<``) — the grammar and parser agree on the surface form.
    name, args = _parse(_wire(code), XML_TOOLS)
    assert name == "run_code"
    assert args["code"] == code
    assert args["language"] == "python"


def test_roundtrip_scalar_types_int_bool():
    # int / bool params type-convert correctly (not left as strings).
    name, args = _parse(
        _wire("x", language="cpp", timeout=30, verbose="true"), XML_TOOLS
    )
    assert args["timeout"] == 30 and isinstance(args["timeout"], int)
    assert args["verbose"] is True
    assert args["language"] == "cpp"


def test_roundtrip_nested_object_value():
    # A nested-object ($ref) value round-trips into a JSON object with typed
    # fields — the ``%json`` surface form the grammar emits is what the parser
    # json.loads-decodes.
    wire = (
        "<tool_call>\n<function=place>\n"
        '<parameter=origin>\n{"x": 3, "y": 4}\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    name, args = _parse(wire, XML_REF_TOOL)
    assert name == "place"
    assert args["origin"] == {"x": 3, "y": 4}
