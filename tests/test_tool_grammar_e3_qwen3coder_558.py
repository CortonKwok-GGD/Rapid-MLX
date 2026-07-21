# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the Qwen3-Coder XML-args grammar body (#558 E3).

E3 extends the #558 grammar builder beyond the JSON-object body
(``%json <schema>``, used by hermes/qwen/harmony/deepseek-JSON) to the
XML-argument wire the Qwen3-Coder family emits::

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

The wire and the per-property XML value rendering are COPIED from the XGrammar
``qwen_3_coder`` built-in (``style="qwen_xml"``) and verified byte-for-byte
against the Qwen3-Coder chat template's ``message.tool_calls`` rendering and our
own ``qwen3coder_tool_parser`` (generation grammar and parser agree).

Two layers:

  * pure-Python BUILDER tests (never skip): ``build_tool_lark`` emits the
    ``<parameter=...>`` blocks, the shared ``STRVAL``/``INTVAL``/``NUMVAL``
    terminals, enum alternation, ``%json`` for object/array values, and wraps
    optional properties ``( ... )?`` — while the JSON path stays byte-identical;
  * grammar ENFORCEMENT (skip only on genuine tokenizer/llguidance
    unavailability): a well-formed Qwen3-Coder XML call is ACCEPTED in full and
    reaches an accepting/terminal state, while a hallucinated function name, an
    off-schema/undeclared parameter, a bad enum value, a non-integer for an int
    param, a JSON body instead of XML, and a missing required parameter are all
    REJECTED mid-stream.

The enforcement tests use the same pinned Qwen tokenizer as
``tests/test_tool_grammar_558.py`` — its ``<tool_call>``/``</tool_call>`` are
single special tokens (``<function=``/``<parameter=`` are ordinary text), which
is exactly the layout the Qwen3-Coder wire needs.
"""

import importlib.util

import pytest

_HAS_LLGUIDANCE = importlib.util.find_spec("llguidance") is not None
_requires_llguidance = pytest.mark.skipif(
    not _HAS_LLGUIDANCE, reason="llguidance ([guided] extra) not installed"
)

_TOKENIZER_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
# Pinned (immutable artifact) — same pin as tests/test_tool_grammar_558.py.
_TOKENIZER_REVISION = "32f3e8ecf65426fc3306969496342d504bfa13f3"

# A representative Qwen3-Coder tool: a required string, an optional string-enum,
# a required integer, an optional number, a boolean, and an object-typed arg
# (exercises every value-rule branch).
WEATHER_TOOL = {
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
TYPED_TOOL = {
    "name": "configure",
    "parameters": {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
            "meta": {"type": "object", "properties": {"k": {"type": "string"}}},
        },
        "required": ["count"],
        "additionalProperties": False,
    },
}
NOARG_TOOL = {"name": "ping", "parameters": {"type": "object", "properties": {}}}


def _qwen_xml_structure_info(name: str):
    """The Qwen3-Coder XML wire triple, exactly as the real parser ships it."""
    from vllm_mlx.api.tool_grammar import StructureInfo

    return StructureInfo(
        begin=f"<tool_call>\n<function={name}>\n",
        end="</function>\n</tool_call>",
        trigger="<tool_call>",
        sentinels=("<tool_call>", "</tool_call>"),
        arg_format="qwen_xml",
    )


# --------------------------------------------------------------------------
# BUILDER (pure Python, always runs).
# --------------------------------------------------------------------------
def _lark_for(tools, tool_choice="required"):
    from vllm_mlx.api.tool_grammar import build_tool_lark

    infos = [_qwen_xml_structure_info(t["name"]) for t in tools]
    return build_tool_lark(tools, tool_choice, infos)


def test_qwen_xml_body_emits_parameter_blocks_and_shared_terminals():
    lark = _lark_for([WEATHER_TOOL])
    # XML markup, not a JSON object body.
    assert '"<parameter=city>\\n"' in lark
    assert '"<parameter=unit>\\n"' in lark
    # The shared value terminals are declared exactly once.
    assert lark.count("STRVAL: /[^<]*/") == 1
    assert "INTVAL:" in lark and "NUMVAL:" in lark
    # No JSON-object body for an XML tool.
    assert "%json" not in lark


def test_qwen_xml_enum_renders_as_alternation():
    lark = _lark_for([WEATHER_TOOL])
    # enum ["c", "f"] -> ("c" | "f") raw-string alternation.
    assert '"c" | "f"' in lark


def test_qwen_xml_optional_property_is_wrapped_optional():
    lark = _lark_for([WEATHER_TOOL])
    # ``city`` required (bare), ``unit`` optional -> its block is ``( ... )?``.
    # The optional wrapper immediately precedes the ``unit`` parameter open.
    assert '("<parameter=unit>\\n"' in lark
    assert ")?" in lark


def test_qwen_xml_typed_values_use_typed_rules():
    lark = _lark_for([TYPED_TOOL])
    assert "INTVAL" in lark  # count: integer
    assert "NUMVAL" in lark  # ratio: number
    assert '"true" | "false"' in lark  # flag: boolean
    # object-typed arg constrains its value with %json of the subschema.
    assert '%json {"type": "object"' in lark


def test_qwen_xml_no_params_emits_empty_body():
    lark = _lark_for([NOARG_TOOL])
    # No <parameter=...> block for a no-arg tool; begin/end still present.
    assert "<parameter=" not in lark
    assert '"\\n<function=ping>\\n"' in lark
    assert "</function>" in lark


def test_json_arg_format_body_is_unchanged():
    # Regression guard: a default (json) StructureInfo still emits ``%json`` and
    # does NOT pull in the XML terminals — the pure-JSON grammar is untouched.
    from vllm_mlx.api.tool_grammar import StructureInfo, build_tool_lark

    tools = [WEATHER_TOOL]
    infos = [
        StructureInfo(
            begin=f'<tool_call>\n{{"name": "{WEATHER_TOOL["name"]}", "arguments": ',
            end="}\n</tool_call>",
            trigger="<tool_call>",
            sentinels=("<tool_call>", "</tool_call>"),
        )
    ]
    lark = build_tool_lark(tools, "required", infos)
    assert "%json" in lark
    assert "STRVAL" not in lark
    assert "<parameter=" not in lark


# --------------------------------------------------------------------------
# structure_info() on the REAL Qwen3CoderToolParser (pure Python).
# --------------------------------------------------------------------------
class _FakeAddedToken:
    def __init__(self, content, special=False):
        self.content = content
        self.special = special


class _SingleTokenTokenizer:
    """Qwen-like: <tool_call>/</tool_call> are distinct single ADDED tokens."""

    def __init__(self):
        self._added = {"<tool_call>": 100, "</tool_call>": 101}
        self._id_to_str = {i: s for s, i in self._added.items()}
        self.added_tokens_decoder = {
            i: _FakeAddedToken(s) for s, i in self._added.items()
        }

    def encode(self, text, add_special_tokens=False):
        return [self._added[text]] if text in self._added else [0, 1]

    def decode(self, ids):
        return "".join(self._id_to_str.get(i, "<unk>") for i in ids)

    def get_vocab(self):
        return dict(self._added)


def _parser(tokenizer=None):
    from vllm_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    return Qwen3CoderToolParser(tokenizer=tokenizer)


def test_supports_grammar_is_true():
    from vllm_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    assert Qwen3CoderToolParser.supports_grammar() is True


def test_structure_info_opts_out_without_tokenizer():
    assert _parser(tokenizer=None).structure_info() is None


def test_structure_info_opts_out_on_multitoken_tokenizer():
    class _MT:
        added_tokens_decoder = {}

        def encode(self, t, add_special_tokens=False):
            return [0, 1]

        def decode(self, ids):
            return "<unk>"

        def get_vocab(self):
            return {}

    assert _parser(tokenizer=_MT()).structure_info() is None


def test_structure_info_wire_triple_opts_in():
    get_info = _parser(tokenizer=_SingleTokenTokenizer()).structure_info()
    assert callable(get_info)
    si = get_info("get_weather")
    assert si.begin == "<tool_call>\n<function=get_weather>\n"
    assert si.end == "</function>\n</tool_call>"
    assert si.trigger == "<tool_call>"
    assert si.sentinels == ("<tool_call>", "</tool_call>")
    assert si.arg_format == "qwen_xml"
    # Builder invariants: begin starts with trigger; trigger is a sentinel.
    assert si.begin.startswith(si.trigger)
    assert si.trigger in si.sentinels


# --------------------------------------------------------------------------
# Grammar ENFORCEMENT via offline consume (skip only on genuine unavailability).
# --------------------------------------------------------------------------
def _offline_skip_exc_types():
    types: list[type[BaseException]] = []
    try:
        from huggingface_hub.errors import (
            LocalEntryNotFoundError,
            OfflineModeIsEnabled,
        )

        types += [LocalEntryNotFoundError, OfflineModeIsEnabled]
    except Exception:  # pragma: no cover - old hub
        pass
    try:
        from requests.exceptions import ConnectionError as _ReqConnErr

        types.append(_ReqConnErr)
    except Exception:  # pragma: no cover - requests absent
        pass
    return tuple(types) or (OSError,)


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            _TOKENIZER_MODEL, revision=_TOKENIZER_REVISION, use_fast=True
        )
    except _offline_skip_exc_types():  # pragma: no cover - offline & uncached
        pytest.skip(
            f"tokenizer {_TOKENIZER_MODEL}@{_TOKENIZER_REVISION[:8]} not cached "
            "and no network — enforcement tests require it"
        )


@pytest.fixture(scope="module")
def lltok(tok):
    from vllm_mlx.api.tool_grammar import build_lltokenizer

    built = build_lltokenizer(tok)
    if built is None:  # pragma: no cover - only if the fast tokenizer is absent
        pytest.skip("could not build an LLTokenizer for the pinned tokenizer")
    return built


def _consume(grammar, lltok, tok, text):
    """Advance grammar state token-by-token. Returns (accepted, total, terminal)."""
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


def _grammar():
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    return build_tool_grammar([WEATHER_TOOL], "required", _optin_parser())


def _optin_parser():
    return _parser(tokenizer=_SingleTokenTokenizer())


# A valid, fully-formed Qwen3-Coder XML tool call for get_weather.
_VALID = (
    "<tool_call>\n<function=get_weather>\n"
    "<parameter=city>\nParis\n</parameter>\n"
    "<parameter=unit>\nc\n</parameter>\n"
    "</function>\n</tool_call>"
)


@_requires_llguidance
def test_valid_qwen3coder_xml_accepted_and_terminates(tok, lltok):
    grammar = _grammar()
    assert grammar is not None
    accepted, total, terminal = _consume(grammar, lltok, tok, _VALID)
    assert accepted == total, f"rejected at {accepted}/{total}"
    assert terminal, "valid call did not reach an accepting/terminal state"


@_requires_llguidance
def test_valid_call_with_optional_omitted_accepted(tok, lltok):
    grammar = _grammar()
    text = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nTokyo\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    accepted, total, terminal = _consume(grammar, lltok, tok, text)
    assert accepted == total and terminal


@_requires_llguidance
def test_reject_missing_close_tool_call(tok, lltok):
    grammar = _grammar()
    text = _VALID[: -len("</tool_call>")]  # drop the closing wrapper
    accepted, total, terminal = _consume(grammar, lltok, tok, text)
    assert not terminal  # a prefix, not a complete derivation


@_requires_llguidance
def test_reject_json_body_instead_of_xml(tok, lltok):
    grammar = _grammar()
    text = (
        '<tool_call>\n<function=get_weather>\n{"city": "Paris"}\n'
        "</function>\n</tool_call>"
    )
    accepted, total, _ = _consume(grammar, lltok, tok, text)
    assert accepted < total, "a JSON-object body must be rejected under qwen_xml"


@_requires_llguidance
def test_reject_undeclared_parameter_key(tok, lltok):
    grammar = _grammar()
    text = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=country>\nFR\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    accepted, total, _ = _consume(grammar, lltok, tok, text)
    assert accepted < total, "an undeclared parameter key must be rejected"


@_requires_llguidance
def test_reject_bad_enum_value(tok, lltok):
    grammar = _grammar()
    text = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "<parameter=unit>\nkelvin\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    accepted, total, _ = _consume(grammar, lltok, tok, text)
    assert accepted < total, "an off-enum value must be rejected"


@_requires_llguidance
def test_reject_missing_required_parameter(tok, lltok):
    grammar = _grammar()
    text = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=unit>\nc\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    accepted, total, _ = _consume(grammar, lltok, tok, text)
    assert accepted < total, "omitting a required parameter must be rejected"


@_requires_llguidance
def test_reject_hallucinated_function_name(tok, lltok):
    grammar = _grammar()
    text = (
        "<tool_call>\n<function=not_a_tool>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    accepted, total, _ = _consume(grammar, lltok, tok, text)
    assert accepted < total, "a hallucinated tool name must be rejected"


@_requires_llguidance
def test_reject_non_integer_for_int_param(tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar([TYPED_TOOL], "required", _optin_parser())
    assert grammar is not None
    text = (
        "<tool_call>\n<function=configure>\n"
        "<parameter=count>\nnot_a_number\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    accepted, total, _ = _consume(grammar, lltok, tok, text)
    assert accepted < total, "a non-integer for an integer param must be rejected"
