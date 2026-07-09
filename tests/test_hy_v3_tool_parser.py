# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for the Hy3 tool call parser.

Covers:
- Canonical wire shape with ``:opensource`` suffix on every tag
- Suffix-less shape (future-proof against upstream dropping the label)
- Malformed close (``<tool_call>NAME</arg_value>`` — 4-bit numerical noise
  workaround, empirically observed on ``mlx-community/Hy3-preview-4bit``)
- XML-pair argument variant (``<arg_key>K</arg_key><arg_value>V</arg_value>``)
- Multiple tool_calls in a single response
- ``<think:opensource>`` reasoning span preceding the tool call
- No tool call at all (pure content passthrough)
- Streaming buffer-until-close semantics
"""

from __future__ import annotations

import json

from vllm_mlx.tool_parsers import HyV3ToolParser, ToolParserManager


def test_parser_is_registered():
    """The parser must appear in the registry under both aliases so
    downstream ``tool_call_parser="hy_v3"`` (and the CLI-friendly
    ``hy3``) resolve without a ``KeyError``."""
    assert ToolParserManager.get_tool_parser("hy_v3") is HyV3ToolParser
    assert ToolParserManager.get_tool_parser("hy3") is HyV3ToolParser


def test_expected_wire_formats_declared():
    """Structural test — every parser MUST declare a non-empty
    ``EXPECTED_WIRE_FORMATS`` tuple so the audit matrix stays honest."""
    assert HyV3ToolParser.EXPECTED_WIRE_FORMATS == ("hy3_native",)


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


def test_canonical_json_body_without_suffix():
    """Future-proof: the same parser MUST accept the plain (suffix-less)
    variant so upstream can drop the ``:opensource`` label in a
    later revision without breaking rapid-mlx."""
    parser = HyV3ToolParser()
    out = '<tool_call>get_weather<tool_sep>{"city": "Paris"}<end_of_tool_call>'
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "get_weather"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"city": "Paris"}


def test_malformed_close_defensive_strip():
    """4-bit numerical noise workaround: model skips
    ``<tool_sep>{args}<end_of_tool_call>`` and jumps straight to
    ``</arg_value>``. Parser MUST still surface the tool name (with
    empty arguments) rather than dropping the call silently — otherwise
    the user sees an empty ``tool_calls`` array and thinks the model
    refused the request. Empirically observed on ``mlx-community/
    Hy3-preview-4bit`` at REAP50 and REAP75 quant levels."""
    parser = HyV3ToolParser()
    out = "<tool_call:opensource>get_weather</arg_value:opensource>"
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["name"] == "get_weather"
    assert tc["arguments"] == "{}"


def test_malformed_close_suffix_less():
    """Same defensive strip works when the suffix is absent."""
    parser = HyV3ToolParser()
    res = parser.extract_tool_calls("<tool_call>get_weather</arg_value>")
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "get_weather"
    assert res.tool_calls[0]["arguments"] == "{}"


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
    """Multi-key XML variant — ``<arg_value>`` payload MUST be
    JSON-decoded so ``1`` → int, ``"two"`` → str, ``true`` → bool."""
    parser = HyV3ToolParser()
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


def test_multiple_tool_calls():
    """Two tool_calls in one assistant turn — both must be extracted
    in wire order."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call>a<tool_sep>{}<end_of_tool_call>"
        '<tool_call:opensource>b<tool_sep:opensource>{"x": 1}'
        "<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 2
    assert res.tool_calls[0]["name"] == "a"
    assert res.tool_calls[1]["name"] == "b"
    assert json.loads(res.tool_calls[1]["arguments"]) == {"x": 1}


def test_think_prefix_stripped_before_tool_extraction():
    """When no reasoning parser is configured, a ``<think:opensource>``
    span preceding the tool call MUST NOT block extraction (defensive
    parity with the hermes / glm47 parsers)."""
    parser = HyV3ToolParser()
    out = (
        "<think:opensource>Let me think...</think:opensource>"
        "<tool_call:opensource>fn<tool_sep:opensource>{}<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "fn"


def test_no_tool_call_returns_content_unchanged():
    """A pure text response must pass through as content with
    ``tools_called=False``."""
    parser = HyV3ToolParser()
    res = parser.extract_tool_calls("The answer is Paris.")
    assert res.tools_called is False
    assert res.tool_calls == []
    assert res.content == "The answer is Paris."


def test_tool_name_filter_via_request_tools():
    """When the request supplies a ``tools`` list, unknown tool names
    MUST be filtered out (defence against name hallucination)."""
    parser = HyV3ToolParser()
    out = '<tool_call>bogus_tool<tool_sep>{"x": 1}<end_of_tool_call>'
    request = {"tools": [{"function": {"name": "allowed_tool"}}]}
    res = parser.extract_tool_calls(out, request=request)
    assert res.tools_called is False
    assert res.tool_calls == []


def test_streaming_holds_until_close_then_emits():
    """The buffer-until-close streaming policy — content deltas that
    arrive INSIDE an open ``<tool_call>`` block are suppressed; the
    delta that carries the close tag emits the parsed tool_calls
    array."""
    parser = HyV3ToolParser()
    parser.reset()
    prev = ""

    def step(delta: str):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_tool_calls_streaming(prev, cur, delta)
        prev = cur
        return msg

    assert step("<tool_call:opensource>") is None
    assert step("get_weather") is None
    assert step("<tool_sep:opensource>") is None
    assert step('{"city": "NYC"}') is None
    final = step("<end_of_tool_call:opensource>")
    assert final is not None
    assert "tool_calls" in final
    calls = final["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "NYC"}


def test_streaming_malformed_close_still_emits():
    """Malformed close on the streaming path — the parser MUST still
    emit the tool_call on the ``</arg_value>`` delta rather than hang
    indefinitely (which would leave the client without a tool_calls
    response and force a timeout)."""
    parser = HyV3ToolParser()
    parser.reset()
    prev = ""

    def step(delta: str):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_tool_calls_streaming(prev, cur, delta)
        prev = cur
        return msg

    assert step("<tool_call:opensource>") is None
    assert step("do_something") is None
    final = step("</arg_value:opensource>")
    assert final is not None
    assert final["tool_calls"][0]["function"]["name"] == "do_something"


def test_streaming_passthrough_content_when_no_tool_call():
    """Plain content deltas — no ``<tool_call>`` opener seen — must pass
    through as content on every delta."""
    parser = HyV3ToolParser()
    parser.reset()
    msg = parser.extract_tool_calls_streaming("", "Hello ", "Hello ")
    assert msg is not None
    assert msg["content"] == "Hello "


def test_streaming_content_after_completed_tool_call_is_suppressed():
    """Codex round-2 BLOCKING #2 regression test. Once an assistant turn
    is a TOOL-CALL turn (any ``<tool_call>`` opener has appeared in the
    accumulated text), post-close plain-content deltas MUST be
    suppressed. OpenAI-compatible clients treat ``tool_calls`` and
    ``content`` as mutually exclusive for a single assistant turn — a
    later ``delta.content`` after ``delta.tool_calls`` breaks parsers
    that dispatch the response as a tool call. This mirrors the
    ``Glm47ToolParser`` policy.

    (Round-1 codex asked us to KEEP post-call content; round-2 codex
    overturned that as wrong per the OpenAI spec — the round-2 fix wins.
    See PR #1070 for the full arc.)"""
    parser = HyV3ToolParser()
    parser.reset()
    prev = ""

    def step(delta: str):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_tool_calls_streaming(prev, cur, delta)
        prev = cur
        return msg

    step("<tool_call:opensource>")
    step("do_it")
    step("<tool_sep:opensource>")
    step("{}")
    final = step("<end_of_tool_call:opensource>")
    assert final is not None and "tool_calls" in final

    # Post-call plain content deltas MUST be suppressed — the assistant
    # turn is already committed to being a tool-call turn.
    m1 = step(" now ")
    assert m1 is None
    m2 = step("what?")
    assert m2 is None


def test_streaming_xml_pair_first_arg_close_does_not_emit_early():
    """Codex round-2 BLOCKING #1 regression test. When the body uses the
    XML-pair variant (``<arg_key>K</arg_key><arg_value>V</arg_value>``),
    the FIRST argument's ``</arg_value>`` closer MUST NOT be treated as
    a tool-call close — otherwise the parser flushes the call after the
    first argument with ``arguments={}`` or a truncated dict. Only the
    canonical ``<end_of_tool_call>`` closes an XML-pair body, and this
    test locks that contract in."""
    parser = HyV3ToolParser()
    parser.reset()
    prev = ""

    def step(delta: str):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_tool_calls_streaming(prev, cur, delta)
        prev = cur
        return msg

    assert step("<tool_call:opensource>") is None
    assert step("multi_arg_fn") is None
    assert step("<tool_sep:opensource>") is None
    assert step("<arg_key:opensource>city</arg_key:opensource>") is None
    assert step("<arg_value:opensource>Paris") is None
    # The FIRST </arg_value> — mid-body, MUST NOT flush.
    early = step("</arg_value:opensource>")
    assert early is None, (
        "First-arg </arg_value> was treated as call-close — regressing "
        "codex round-2 BLOCKING #1."
    )
    assert step("<arg_key:opensource>units</arg_key:opensource>") is None
    assert step("<arg_value:opensource>metric</arg_value:opensource>") is None
    # Canonical close is the ONLY trigger.
    final = step("<end_of_tool_call:opensource>")
    assert final is not None
    tc = final["tool_calls"][0]
    assert tc["function"]["name"] == "multi_arg_fn"
    args = json.loads(tc["function"]["arguments"])
    assert args == {"city": "Paris", "units": "metric"}


def test_streaming_close_split_across_sse_boundary_still_emits():
    """Codex round-1 BLOCKING #3 regression test. The close tag arrives
    split across two SSE chunks (e.g. ``<end_of_tool_c`` then
    ``all:opensource>``). The parser MUST detect the transition to
    "no pending unclosed call" and emit the tool_calls array — the
    earlier gate (which only searched ``delta_text``) would never fire
    on the second chunk because the close token only fully appears in
    ``current_text``, not the second delta alone."""
    parser = HyV3ToolParser()
    parser.reset()
    prev = ""

    def step(delta: str):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_tool_calls_streaming(prev, cur, delta)
        prev = cur
        return msg

    step("<tool_call>")
    step("my_fn")
    step("<tool_sep>")
    step("{}")
    # Split the closer across two chunks.
    m_partial = step("<end_of_tool_c")
    assert m_partial is None
    m_close = step("all>")
    assert m_close is not None
    assert m_close["tool_calls"][0]["function"]["name"] == "my_fn"


def test_has_pending_tool_call_recognises_suffix_variant():
    """The pending-call predicate MUST recognise the ``:opensource``
    variant so streaming shutdown handlers can flush the buffer."""
    parser = HyV3ToolParser()
    assert parser.has_pending_tool_call("<tool_call:opensource>fn") is True
    assert parser.has_pending_tool_call("<tool_call>fn") is True
    assert (
        parser.has_pending_tool_call(
            "<tool_call:opensource>fn<end_of_tool_call:opensource>"
        )
        is False
    )
    assert parser.has_pending_tool_call("just a plain message") is False


def test_supports_native_tool_format_flag():
    """The Hy3 chat template renders assistant ``tool_calls`` back as
    ``<tool_call:opensource>…<end_of_tool_call:opensource>``, so the
    native-format flag MUST be True to prevent the tool-history
    round-trip from being converted to synthetic text."""
    assert HyV3ToolParser.SUPPORTS_NATIVE_TOOL_FORMAT is True
