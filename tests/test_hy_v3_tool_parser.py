# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for the Hy3 tool call parser (vLLM HYV3ToolParser port).

Architecture under test (see ``vllm_mlx/tool_parsers/hy_v3_tool_parser.py``):
  * suffix resolved ONCE at ``__init__`` from vocab, pinned as fixed strings
  * token-ID / fixed-string gate on streaming entry (no full-text re-parse)
  * two-phase FSM: SEEKING_NAME → STREAMING_ARGS (withhold trailing ``}``)
  * ``<think>`` lives in the SEPARATE reasoning parser (disjoint stream) — the
    tool parser has zero ``<think>`` code
  * malformed-close salvage on the NON-STREAMING path only

Scenarios that encode REAL model behavior (preserved from the pre-pivot
suite because they were authored from ``pipenetwork/Hy3-REAP50/75-MLX-4bit``
output, 2026-07-09 spike):
  * canonical JSON body with/without the ``:opensource`` suffix
  * malformed close ``<tool_call>NAME</arg_value>`` (4-bit numerical noise)
  * XML-pair argument variant + type coercion
  * multiple tool calls
  * JSON string value containing the literal ``</arg_value>`` substring
  * request ``tools`` allowlist filtering
"""

from __future__ import annotations

import json

from vllm_mlx.tool_parsers import HyV3ToolParser, ToolParserManager


# ---------------------------------------------------------------------------
# Registration / declarative surface
# ---------------------------------------------------------------------------
def test_parser_is_registered():
    """The parser must appear in the registry under both aliases so
    downstream ``tool_call_parser="hy_v3"`` (and the CLI-friendly ``hy3``)
    resolve without a ``KeyError``."""
    assert ToolParserManager.get_tool_parser("hy_v3") is HyV3ToolParser
    assert ToolParserManager.get_tool_parser("hy3") is HyV3ToolParser


def test_expected_wire_formats_declared():
    """Structural test — every parser MUST declare a non-empty
    ``EXPECTED_WIRE_FORMATS`` tuple so the audit matrix stays honest."""
    assert HyV3ToolParser.EXPECTED_WIRE_FORMATS == ("hy3_native",)


def test_supports_native_tool_format_flag():
    """The Hy3 chat template renders assistant ``tool_calls`` back as
    ``<tool_call:opensource>…<end_of_tool_call:opensource>``, so the
    native-format flag MUST be True to prevent the tool-history round-trip
    from being converted to synthetic text."""
    assert HyV3ToolParser.SUPPORTS_NATIVE_TOOL_FORMAT is True


def test_suffix_defaults_to_opensource_without_tokenizer():
    """With no tokenizer the parser MUST fall back to the ``:opensource``
    label every current ``pipenetwork/Hy3-*-MLX-4bit`` checkpoint emits, and
    pin the fixed tag strings accordingly."""
    p = HyV3ToolParser()
    assert p.suffix == ":opensource"
    assert p.tool_call_start_token == "<tool_call:opensource>"
    assert p.tool_sep_token == "<tool_sep:opensource>"
    assert p.tool_call_end_token == "<end_of_tool_call:opensource>"
    assert p.arg_value_end_token == "</arg_value:opensource>"


class _FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


def test_suffix_resolved_from_vocab_suffixless():
    """When the tokenizer vocab exposes the bare ``<tool_call>`` token (a
    future revision that drops the label), the resolver MUST pin the
    suffix-less strings."""
    tok = _FakeTokenizer(
        {
            "<tool_call>": 1000,
            "<tool_sep>": 1001,
            "<end_of_tool_call>": 1002,
        }
    )
    p = HyV3ToolParser(tokenizer=tok)
    assert p.suffix == ""
    assert p.tool_call_start_token == "<tool_call>"
    assert p.tool_call_start_token_id == 1000


def test_suffix_resolved_from_vocab_labelled():
    """When the vocab carries the labelled ``<tool_call:opensource>`` token,
    the resolver pins the labelled strings AND the token id."""
    tok = _FakeTokenizer(
        {
            "<tool_call:opensource>": 2000,
            "<tool_sep:opensource>": 2001,
        }
    )
    p = HyV3ToolParser(tokenizer=tok)
    assert p.suffix == ":opensource"
    assert p.tool_call_start_token_id == 2000


# ---------------------------------------------------------------------------
# Non-streaming extraction — real wire shapes
# ---------------------------------------------------------------------------
def test_canonical_json_body_with_opensource_suffix():
    """The chat-template default emission — every tag carries the
    ``:opensource`` suffix and the body is a JSON object."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>get_weather"
        '<tool_sep:opensource>{"city": "Paris"}'
        "<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["name"] == "get_weather"
    assert json.loads(tc["arguments"]) == {"city": "Paris"}
    assert res.content is None


def test_canonical_json_body_without_suffix():
    """Future-proof: a parser whose vocab pinned the suffix-less strings MUST
    accept the plain variant so upstream can drop the ``:opensource`` label
    in a later revision."""
    tok = _FakeTokenizer({"<tool_call>": 1})
    parser = HyV3ToolParser(tokenizer=tok)
    out = '<tool_call>get_weather<tool_sep>{"city": "Paris"}<end_of_tool_call>'
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "get_weather"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"city": "Paris"}


def test_malformed_close_defensive_strip():
    """4-bit numerical noise workaround: the model skips
    ``<tool_sep>{args}<end_of_tool_call>`` and jumps straight to
    ``</arg_value>``. The NON-STREAMING path MUST still surface the tool name
    (empty arguments) rather than dropping the call silently — otherwise the
    user sees an empty ``tool_calls`` array and thinks the model refused.
    Empirically observed on ``pipenetwork/Hy3-REAP50-MLX-4bit`` +
    ``Hy3-REAP75-MLX-4bit`` (10/10 BFCL simple_python prompts, 2026-07-09
    spike; see bug1_comment.md filed against mlx-lm PR #1211)."""
    parser = HyV3ToolParser()
    out = "<tool_call:opensource>get_weather</arg_value:opensource>"
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["name"] == "get_weather"
    assert tc["arguments"] == "{}"


def test_malformed_close_suffix_less():
    """Same defensive strip works when the suffix is absent (vocab pinned the
    suffix-less strings)."""
    tok = _FakeTokenizer({"<tool_call>": 1})
    parser = HyV3ToolParser(tokenizer=tok)
    res = parser.extract_tool_calls("<tool_call>get_weather</arg_value>")
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "get_weather"
    assert res.tool_calls[0]["arguments"] == "{}"


def test_json_body_then_stray_arg_value_opener_salvages_args():
    """Second real malformed shape from the spike: a well-formed JSON body
    followed by a STRAY ``<arg_value>`` opener before the canonical close
    (``NAME<tool_sep>{"radius":5}<arg_value:opensource><end_of_tool_call>``).
    The JSON prefix ``raw_decode`` MUST recover the real args and ignore the
    trailing stray opener."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>calculate_circle_area"
        '<tool_sep:opensource>{"radius": 5}<arg_value:opensource>'
        "<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "calculate_circle_area"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"radius": 5}


def test_xml_pair_argument_variant():
    """The chat template's second-choice emission — each argument as a
    separate ``<arg_key>K</arg_key><arg_value>V</arg_value>`` pair."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>get_weather"
        "<tool_sep:opensource>"
        "<arg_key:opensource>city</arg_key:opensource>"
        "<arg_value:opensource>Paris</arg_value:opensource>"
        "<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "get_weather"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"city": "Paris"}


def test_xml_pair_multi_key_with_type_coercion():
    """Multi-key XML variant — ``<arg_value>`` payload MUST be JSON-decoded so
    ``1`` → int, ``"two"`` → str, ``true`` → bool."""
    tok = _FakeTokenizer({"<tool_call>": 1})
    parser = HyV3ToolParser(tokenizer=tok)
    out = (
        "<tool_call>lookup<tool_sep>"
        "<arg_key>a</arg_key><arg_value>1</arg_value>"
        '<arg_key>b</arg_key><arg_value>"two"</arg_value>'
        "<arg_key>flag</arg_key><arg_value>true</arg_value>"
        "<end_of_tool_call>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert json.loads(res.tool_calls[0]["arguments"]) == {
        "a": 1,
        "b": "two",
        "flag": True,
    }


def test_sep_less_xml_pair_body():
    """Some 4-bit checkpoints skip ``<tool_sep>`` but still emit full XML
    pairs. The name is the residue before the first ``<arg_key>`` opener and
    the args are recovered from the pairs."""
    tok = _FakeTokenizer({"<tool_call>": 1})
    parser = HyV3ToolParser(tokenizer=tok)
    out = (
        "<tool_call>do_it"
        "<arg_key>x</arg_key><arg_value>1</arg_value>"
        "<arg_key>y</arg_key><arg_value>2</arg_value>"
        "<end_of_tool_call>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "do_it"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"x": 1, "y": 2}


def test_multiple_tool_calls():
    """Two tool_calls in one assistant turn — both must be extracted in wire
    order."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>a<tool_sep:opensource>{}<end_of_tool_call:opensource>"
        '<tool_call:opensource>b<tool_sep:opensource>{"x": 1}'
        "<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 2
    assert res.tool_calls[0]["name"] == "a"
    assert res.tool_calls[1]["name"] == "b"
    assert json.loads(res.tool_calls[1]["arguments"]) == {"x": 1}


def test_json_body_containing_literal_arg_value_close_parses_correctly():
    """A JSON body whose STRING VALUE legitimately contains the literal
    substring ``</arg_value>`` MUST round-trip unchanged. ``raw_decode``
    consumes only a well-formed JSON prefix so the literal inside a string is
    preserved rather than mistaken for a close boundary."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>log_message<tool_sep:opensource>"
        '{"snippet": "The tag </arg_value:opensource> is not a close here.",'
        ' "level": "info"}'
        "<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    args = json.loads(res.tool_calls[0]["arguments"])
    assert args == {
        "snippet": "The tag </arg_value:opensource> is not a close here.",
        "level": "info",
    }


def test_no_tool_call_returns_content_unchanged():
    """A pure text response must pass through as content with
    ``tools_called=False``."""
    parser = HyV3ToolParser()
    res = parser.extract_tool_calls("The answer is Paris.")
    assert res.tools_called is False
    assert res.tool_calls == []
    assert res.content == "The answer is Paris."


def test_tool_name_filter_via_request_tools():
    """When the request supplies a ``tools`` list, unknown tool names MUST be
    filtered out (defence against name hallucination)."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>bogus_tool<tool_sep:opensource>"
        '{"x": 1}<end_of_tool_call:opensource>'
    )
    request = {"tools": [{"function": {"name": "allowed_tool"}}]}
    res = parser.extract_tool_calls(out, request=request)
    assert res.tools_called is False
    assert res.tool_calls == []


def test_valid_names_filter_preserves_rejected_span_in_content():
    """When ``valid_names`` is set and every parsed call is filtered out, the
    raw span of the rejected call MUST be preserved in ``content`` — silently
    dropping it makes the output look like a refusal when the model actually
    tried to invoke an off-list tool."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>bogus_tool<tool_sep:opensource>"
        '{"x": 1}<end_of_tool_call:opensource>'
    )
    request = {"tools": [{"function": {"name": "allowed_tool"}}]}
    res = parser.extract_tool_calls(out, request=request)
    assert res.tools_called is False
    assert res.tool_calls == []
    assert res.content is not None
    assert "bogus_tool" in res.content


def test_valid_names_filter_preserves_mixed_valid_and_rejected():
    """When SOME calls are valid, ``tools_called=True`` and ``content=None``
    (exclusive-turn policy). The rejected span is lost in that case by design
    — the OpenAI-compatible contract forbids mixed ``tool_calls`` +
    ``content`` in a single assistant turn."""
    parser = HyV3ToolParser()
    out = (
        '<tool_call:opensource>allowed_tool<tool_sep:opensource>{"y": 2}'
        "<end_of_tool_call:opensource>"
        "\n"
        '<tool_call:opensource>bogus_tool<tool_sep:opensource>{"x": 1}'
        "<end_of_tool_call:opensource>"
    )
    request = {"tools": [{"function": {"name": "allowed_tool"}}]}
    res = parser.extract_tool_calls(out, request=request)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "allowed_tool"
    assert res.content is None


# ---------------------------------------------------------------------------
# Streaming — token-gate + 2-phase FSM. Boundary cases are now trivially
# green because the opener gate + fixed-string finds replace the bespoke
# straddle machinery.
# ---------------------------------------------------------------------------
def _stepper(parser, request=None):
    """Return a ``step(delta)`` closure that feeds deltas and returns the
    parser's per-delta result."""
    state = {"prev": ""}

    def step(delta: str):
        cur = state["prev"] + delta
        msg = parser.extract_tool_calls_streaming(
            state["prev"], cur, delta, request=request
        )
        state["prev"] = cur
        return msg

    return step


def _collect_stream(parser, chunks, request=None):
    """Feed ``chunks`` and return ``(tool_acc, content)`` where ``tool_acc``
    maps index → {name, args}."""
    parser.reset()
    step = _stepper(parser, request=request)
    tool_acc: dict[int, dict] = {}
    content = ""
    for d in chunks:
        msg = step(d)
        if not msg:
            continue
        if msg.get("content"):
            content += msg["content"]
        for tc in msg.get("tool_calls", []) or []:
            entry = tool_acc.setdefault(tc["index"], {"name": "", "args": ""})
            fn = tc.get("function", {})
            if fn.get("name"):
                entry["name"] = fn["name"]
            if fn.get("arguments"):
                entry["args"] += fn["arguments"]
    return tool_acc, content


def test_streaming_holds_until_close_then_emits_json_body():
    """The name emits when ``<tool_sep>`` lands; the args stream as a JSON
    diff; the delta carrying ``<end_of_tool_call>`` completes the args with
    the trailing ``}``. Reassembled args MUST equal the wire JSON."""
    parser = HyV3ToolParser()
    tool_acc, content = _collect_stream(
        parser,
        [
            "<tool_call:opensource>",
            "get_weather",
            "<tool_sep:opensource>",
            '{"city": "NYC"}',
            "<end_of_tool_call:opensource>",
        ],
    )
    assert 0 in tool_acc
    assert tool_acc[0]["name"] == "get_weather"
    assert json.loads(tool_acc[0]["args"]) == {"city": "NYC"}
    assert content == ""


def test_streaming_char_by_char_json_body_no_leak():
    """Char-by-char delivery (the harshest boundary case) MUST NOT leak any
    raw markup as content and MUST reassemble the args to valid JSON. The
    partial-opener prefix hold catches the char-split opener; the fixed-string
    finds handle every interior boundary."""
    parser = HyV3ToolParser()
    wire = (
        "<tool_call:opensource>get_weather"
        '<tool_sep:opensource>{"city": "NYC"}'
        "<end_of_tool_call:opensource>"
    )
    tool_acc, content = _collect_stream(parser, list(wire))
    assert tool_acc[0]["name"] == "get_weather"
    assert json.loads(tool_acc[0]["args"]) == {"city": "NYC"}
    assert content == "", f"raw markup leaked as content: {content!r}"


def test_streaming_json_body_with_escapes_and_unicode_reassembles():
    """Char-by-char streaming of a JSON body whose string values contain
    escape sequences (``\\n``, ``\\"``, ``\\\\``) AND a ``\\uXXXX`` unicode
    escape MUST reassemble to the exact wire JSON. The parser streams the RAW
    wire text verbatim (never re-serializing via ``json.dumps``), so the
    open prefixes stay byte-aligned with the closed document even when the
    value contains ``\\uXXXX`` — a re-serialize would decode it to the char
    and make the diff non-monotonic (regression guard for the escape-stream
    snapshot bug)."""
    parser = HyV3ToolParser()
    # ``json.dumps`` default (ensure_ascii=True) puts a ``\\uXXXX`` escape on
    # the wire for the é; the other escapes exercise the dangling-backslash
    # and quote-in-value hold logic.
    args_obj = {"path": "/a/b.txt", "content": 'line1\nline2 "q" \\ café'}
    body = json.dumps(args_obj)
    assert "\\u" in body  # confirm the wire really carries a unicode escape
    wire = (
        "<tool_call:opensource>write_file<tool_sep:opensource>"
        + body
        + "<end_of_tool_call:opensource>"
    )
    tool_acc, content = _collect_stream(parser, list(wire))
    assert tool_acc[0]["name"] == "write_file"
    assert json.loads(tool_acc[0]["args"]) == args_obj
    assert content == "", f"raw markup leaked as content: {content!r}"


def test_streaming_json_body_nested_and_mixed_types_reassembles():
    """A JSON body with nested objects/arrays and mixed scalar types streams
    char-by-char and reassembles exactly."""
    parser = HyV3ToolParser()
    args_obj = {"n": 3, "flag": True, "z": None, "list": [1, 2], "obj": {"k": "v"}}
    wire = (
        "<tool_call:opensource>f<tool_sep:opensource>"
        + json.dumps(args_obj)
        + "<end_of_tool_call:opensource>"
    )
    tool_acc, content = _collect_stream(parser, list(wire))
    assert json.loads(tool_acc[0]["args"]) == args_obj
    assert content == ""


def test_streaming_xml_pair_reassembles_args():
    """The XML-pair variant streams pairs into a growing JSON object; the
    reassembled args MUST equal the wire pairs with type coercion."""
    parser = HyV3ToolParser()
    tool_acc, content = _collect_stream(
        parser,
        [
            "<tool_call:opensource>",
            "multi_arg_fn",
            "<tool_sep:opensource>",
            "<arg_key:opensource>city</arg_key:opensource>",
            "<arg_value:opensource>Paris</arg_value:opensource>",
            "<arg_key:opensource>n</arg_key:opensource>",
            "<arg_value:opensource>3</arg_value:opensource>",
            "<end_of_tool_call:opensource>",
        ],
    )
    assert tool_acc[0]["name"] == "multi_arg_fn"
    assert json.loads(tool_acc[0]["args"]) == {"city": "Paris", "n": 3}
    assert content == ""


def test_streaming_first_arg_value_close_does_not_finish_call():
    """A mid-body ``</arg_value>`` (closing the FIRST argument value) MUST NOT
    finish the call — only ``<end_of_tool_call>`` closes it. Otherwise the
    parser would flush truncated args after the first argument."""
    parser = HyV3ToolParser()
    tool_acc, content = _collect_stream(
        parser,
        [
            "<tool_call:opensource>",
            "multi_arg_fn",
            "<tool_sep:opensource>",
            "<arg_key:opensource>city</arg_key:opensource>",
            "<arg_value:opensource>Paris</arg_value:opensource>",
            "<arg_key:opensource>units</arg_key:opensource>",
            "<arg_value:opensource>metric</arg_value:opensource>",
            "<end_of_tool_call:opensource>",
        ],
    )
    assert json.loads(tool_acc[0]["args"]) == {"city": "Paris", "units": "metric"}
    assert content == ""


def test_streaming_close_split_across_sse_boundary_still_emits():
    """The close tag arrives split across two SSE chunks (``<end_of_tool_c``
    then ``all:opensource>``). Because the parser searches for the fixed close
    string in the accumulated buffer, the split resolves on the chunk that
    completes it — no bespoke transition tracking needed."""
    parser = HyV3ToolParser()
    tool_acc, content = _collect_stream(
        parser,
        [
            "<tool_call:opensource>",
            "my_fn",
            "<tool_sep:opensource>",
            "{}",
            "<end_of_tool_c",
            "all:opensource>",
        ],
    )
    assert tool_acc[0]["name"] == "my_fn"
    assert json.loads(tool_acc[0]["args"]) == {}
    assert content == ""


def test_streaming_passthrough_content_when_no_tool_call():
    """Plain content deltas — no ``<tool_call>`` opener seen — pass through as
    content."""
    parser = HyV3ToolParser()
    parser.reset()
    msg = parser.extract_tool_calls_streaming("", "Hello ", "Hello ")
    assert msg is not None
    assert msg["content"] == "Hello "


def test_streaming_content_after_completed_tool_call_is_suppressed():
    """Once an assistant turn is a TOOL-CALL turn (any ``<tool_call>`` opener
    has appeared), post-close plain-content deltas MUST be suppressed —
    OpenAI-compatible clients treat ``tool_calls`` and ``content`` as mutually
    exclusive for a single assistant turn."""
    parser = HyV3ToolParser()
    parser.reset()
    step = _stepper(parser)
    step("<tool_call:opensource>")
    step("do_it")
    step("<tool_sep:opensource>")
    step("{}")
    final = step("<end_of_tool_call:opensource>")
    assert final is not None and "tool_calls" in final
    assert step(" now ") is None
    assert step("what?") is None


def test_streaming_partial_opener_prefix_held_then_released_on_falsify():
    """When a delta ends in a partial-opener prefix that later FALSIFIES into
    ordinary text (``<tool_ca`` then ``rrot recipe`` → ``<tool_carrot``), the
    held bytes MUST surface as content once the tail resolves. No prose is
    lost, and no partial markup leaks before resolution."""
    parser = HyV3ToolParser()
    parser.reset()
    step = _stepper(parser)
    m1 = step("Look at this: <tool_ca")
    # The trailing ``<tool_ca`` is a partial-opener prefix — held back.
    assert (m1 or {}).get("content", "") == "Look at this: "
    m2 = step("rrot recipe")
    assert m2 is not None
    assert m2["content"] == "<tool_carrot recipe"


def test_streaming_partial_opener_prefix_resolves_to_tool_call():
    """When the partial-opener prefix COMPLETES into a real opener, the turn
    becomes a tool-call turn; the held bytes are markup (not content) and the
    call streams normally."""
    parser = HyV3ToolParser()
    tool_acc, content = _collect_stream(
        parser,
        [
            "<tool_ca",
            "ll:opensource>get_weather<tool_sep:opensource>{}",
            "<end_of_tool_call:opensource>",
        ],
    )
    assert tool_acc[0]["name"] == "get_weather"
    # The ``<tool_ca`` prefix was held, then absorbed into the tool-call turn.
    assert content == ""


def test_streaming_respects_request_tool_allowlist():
    """The streaming path MUST honour the request ``tools`` allowlist so a
    hallucinated off-list name is not surfaced as a ``tool_calls`` header."""
    parser = HyV3ToolParser()
    request = {"tools": [{"function": {"name": "allowed_tool"}}]}
    tool_acc, _content = _collect_stream(
        parser,
        [
            "<tool_call:opensource>",
            "hallucinated_tool",
            "<tool_sep:opensource>",
            "{}",
            "<end_of_tool_call:opensource>",
        ],
        request=request,
    )
    # The off-list call must not surface as a tool_call header.
    for entry in tool_acc.values():
        assert entry["name"] != "hallucinated_tool"


def test_streaming_flush_held_content_releases_dangling_prefix():
    """A stream ending in a partial-opener prefix that never completed leaves
    those bytes held; ``flush_held_content`` MUST release them so the final
    chars are not dropped."""
    parser = HyV3ToolParser()
    parser.reset()
    step = _stepper(parser)
    step("done <tool_ca")
    flushed = parser.flush_held_content("done <tool_ca")
    assert flushed == "<tool_ca"


def test_flush_held_content_empty_when_tool_call_opened():
    """When a real opener is present, the held tail is markup, not content —
    ``flush_held_content`` returns empty so nothing leaks."""
    parser = HyV3ToolParser()
    full = (
        "<tool_call:opensource>fn<tool_sep:opensource>{}<end_of_tool_call:opensource>"
    )
    assert parser.flush_held_content(full) == ""


# ---------------------------------------------------------------------------
# Pending predicate
# ---------------------------------------------------------------------------
def test_has_pending_tool_call_recognises_suffix_variant():
    """The pending-call predicate MUST recognise the pinned ``:opensource``
    opener so streaming shutdown handlers can flush the buffer, and MUST NOT
    report pending once the call has closed."""
    parser = HyV3ToolParser()
    assert parser.has_pending_tool_call("<tool_call:opensource>fn") is True
    assert (
        parser.has_pending_tool_call(
            "<tool_call:opensource>fn<end_of_tool_call:opensource>"
        )
        is False
    )
    assert parser.has_pending_tool_call("just a plain message") is False
