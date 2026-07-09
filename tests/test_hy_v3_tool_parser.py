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


def test_streaming_xml_pair_without_sep_does_not_flush_early():
    """Codex round-3 BLOCKING #1 regression test. Some 4-bit checkpoints
    emit an XML-pair body WITHOUT the preceding ``<tool_sep>`` — the
    stream goes ``<tool_call>NAME<arg_key>K</arg_key><arg_value>V
    </arg_value>...<end_of_tool_call>``. The round-2 fix gated XML-pair
    detection on ``<tool_sep>`` only, so this shape falls back to the
    salvage-mode ``</arg_value>`` close and flushes with truncated
    ``arguments={}`` on the first ``</arg_value>``. The round-3 fix
    extends the XML-pair signal to ``<arg_key>`` / ``<arg_value>``
    openers so the sep-less stream still waits for
    ``<end_of_tool_call>``.

    This test also covers the split-SSE variant of the same failure
    mode — sep + first arg pair arriving in a later delta than the
    opener — which is the actual scenario codex called out."""
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
    assert step("do_it") is None
    # NO <tool_sep> emitted — the checkpoint jumps straight into args.
    assert step("<arg_key:opensource>x</arg_key:opensource>") is None
    assert step("<arg_value:opensource>1") is None
    # First arg-value close arriving alone MUST NOT flush the call —
    # <arg_key>/<arg_value> openers already flagged XML-pair mode.
    early = step("</arg_value:opensource>")
    assert early is None, (
        "Sep-less XML-pair body flushed on first </arg_value> — "
        "regressing codex round-3 BLOCKING #1."
    )
    # Only the canonical close should trigger the emit.
    assert step("<arg_key:opensource>y</arg_key:opensource>") is None
    assert step("<arg_value:opensource>2</arg_value:opensource>") is None
    final = step("<end_of_tool_call:opensource>")
    assert final is not None
    tc = final["tool_calls"][0]
    assert tc["function"]["name"] == "do_it"
    args = json.loads(tc["function"]["arguments"])
    assert args == {"x": 1, "y": 2}


def test_streaming_partial_opener_withholds_entire_delta():
    """Codex round-5 BLOCKING #1 + round-6 BLOCKING #1/#2 regression
    test. When a delta ends in a strict prefix of ``<tool_call>`` /
    ``<tool_call:LABEL>`` (e.g. delta 1 = ``"Sure, <tool_ca"``, delta
    2 = ``"ll:opensource>..."``), the parser MUST return ``None`` for
    the entire delta — including any pre-straddle prose bytes.
    Emitting ``"Sure, "`` as content before the opener resolves would
    violate the exclusive tool-call turn contract: if the opener
    completes, the assistant turn is a tool-call turn and prose that
    already reached the client can't be un-emitted.

    Round-5's original fix emitted the pre-straddle prefix; codex
    round-6 flagged that as a protocol violation. The stricter policy
    (buffer-until-resolved) is what this test locks in."""
    parser = HyV3ToolParser()
    parser.reset()
    prev = ""

    def step(delta: str):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_tool_calls_streaming(prev, cur, delta)
        prev = cur
        return msg

    # Delta 1: prose + partial opener. Straddle is at end. Contract:
    # WITHHOLD the whole delta — no content emitted, prose stays
    # buffered pending opener resolution.
    m1 = step("Sure, <tool_ca")
    assert m1 is None, f"Partial-opener delta must be fully withheld; got: {m1!r}"
    # Delta 2: opener completes. The turn is now a tool-call turn.
    # All pre-opener prose gets absorbed into the tool-call turn per
    # exclusive-turn policy — no content delta.
    m2 = step("ll:opensource>get_weather<tool_sep:opensource>{}")
    assert m2 is None, f"Post-opener content leaked: {m2!r}"
    # Delta 3: canonical close — emit the tool_calls array.
    m3 = step("<end_of_tool_call:opensource>")
    assert m3 is not None
    assert m3["tool_calls"][0]["function"]["name"] == "get_weather"


def test_streaming_partial_opener_falsified_releases_buffered_prose():
    """Round-6 falsification path — when the straddle turns out NOT
    to be a real opener (e.g. ``"<tool_ca"`` followed by ``"rrot"``
    forming ``"<tool_carrot"``), the previously-withheld pre-straddle
    prose plus the falsifying bytes MUST be emitted as content on the
    tick when the straddle resolves. Otherwise the client would
    permanently lose the ``"Sure, "`` prose that was withheld pending
    opener resolution."""
    parser = HyV3ToolParser()
    parser.reset()
    prev = ""

    def step(delta: str):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_tool_calls_streaming(prev, cur, delta)
        prev = cur
        return msg

    m1 = step("Look at this: <tool_ca")
    assert m1 is None, "straddle must withhold"
    # Falsifying delta — ``rrot`` makes ``<tool_carrot`` which is NOT
    # a valid opener anymore. Watermark releases the withheld bytes.
    m2 = step("rrot recipe")
    assert m2 is not None, "falsification must release buffered prose"
    content = m2.get("content") or ""
    assert "Look at this:" in content, (
        f"buffered prose lost on falsification: {content!r}"
    )
    assert "<tool_carrot recipe" in content, (
        f"falsifying bytes not included on release: {content!r}"
    )


def test_json_body_containing_literal_arg_value_close_parses_correctly():
    """Codex round-5 BLOCKING #3 regression test. A JSON body whose
    STRING VALUE legitimately contains the literal substring
    ``</arg_value>`` MUST round-trip unchanged through the parser.
    The round-1..4 code truncated the tail at the first ``</arg_value>``
    before ``json.loads``, corrupting ``{"snippet": "see </arg_value>
    here"}`` into ``{"snippet": "see``. Using ``JSONDecoder.raw_decode``
    to consume only a well-formed JSON prefix of the tail preserves
    the string content exactly."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>log_message<tool_sep:opensource>"
        '{"snippet": "The tag </arg_value> is not a close here.", "level": "info"}'
        "<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    args = json.loads(res.tool_calls[0]["arguments"])
    assert args == {
        "snippet": "The tag </arg_value> is not a close here.",
        "level": "info",
    }


def test_streaming_respects_request_tool_allowlist():
    """Codex round-6 BLOCKING #1 regression test. The streaming path
    MUST pass ``request`` through to ``extract_tool_calls`` so a
    hallucinated tool name filtered by the request's ``tools``
    allowlist is suppressed in the streaming emit just as it is in the
    non-streaming path. Otherwise streaming leaks off-list tool calls
    that non-streaming would filter."""
    parser = HyV3ToolParser()
    parser.reset()
    prev = ""

    def step(delta: str, request=None):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_tool_calls_streaming(prev, cur, delta, request=request)
        prev = cur
        return msg

    request = {"tools": [{"function": {"name": "allowed_tool"}}]}
    step("<tool_call:opensource>", request=request)
    step("hallucinated_tool", request=request)
    step("<tool_sep:opensource>", request=request)
    step("{}", request=request)
    final = step("<end_of_tool_call:opensource>", request=request)
    # The filtered call MUST NOT surface as tool_calls. Either the
    # streaming emit returns None (call silently dropped, matching
    # exclusive-turn) OR returns content with the raw span preserved
    # — but it MUST NOT emit tool_calls with the off-list name.
    if final is not None:
        tool_calls = final.get("tool_calls") or []
        for tc in tool_calls:
            assert tc["function"]["name"] != "hallucinated_tool", (
                f"Streaming emitted off-list tool_call: {tc!r}"
            )


def test_streaming_json_body_with_literal_arg_value_close_does_not_flush_early():
    """Codex round-6 BLOCKING #2 regression test. A streaming JSON body
    whose value string contains the literal ``</arg_value>`` MUST NOT
    trigger the salvage close mid-body. The round-4 fix used a
    per-``</arg_value>`` body-prefix check (no arg_key/arg_value opener
    in prefix → treat as salvage), but that misfires on a JSON body
    where the ``</arg_value>`` sits inside a string value — the JSON
    hasn't finished yet, and the salvage close would emit truncated
    args. Round-6 guards this by attempting a ``json.raw_decode`` of
    the tail-after-sep; if the ``</arg_value>`` lands INSIDE the JSON
    prefix, it's a string literal, not salvage."""
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
    step("log_message")
    step("<tool_sep:opensource>")
    # The JSON body contains a literal ``</arg_value>`` inside a
    # string value. Emit the whole body at once — no interior chunk
    # boundary — so the salvage check runs on the complete prefix.
    step('{"snippet": "see </arg_value> below", "level": "info"}')
    # The literal ``</arg_value>`` inside the JSON string MUST NOT
    # trigger the salvage close.
    # ``_closed_after_opener`` should recognise the JSON body and
    # skip that ``</arg_value>``.
    # Only the canonical close should fire the emit.
    final = step("<end_of_tool_call:opensource>")
    assert final is not None
    tc = final["tool_calls"][0]
    args = json.loads(tc["function"]["arguments"])
    assert args == {"snippet": "see </arg_value> below", "level": "info"}, (
        f"JSON literal </arg_value> was mishandled: {args!r}"
    )


def test_prefix_check_rejects_unicode_label_matching_isalnum_only():
    r"""Codex round-6 NIT regression test. ``_is_strict_prefix_of_tool_call_opener``
    MUST use the same alphabet as the compiled opener regex
    (``[\w-]+``). A Unicode letter that ``str.isalnum()`` accepts but
    isn't in ``\w`` is a subtle drift point; using the same regex
    keeps them locked in step."""
    from vllm_mlx.tool_parsers.hy_v3_tool_parser import (
        _is_strict_prefix_of_tool_call_opener,
    )

    # Plain ASCII label — accepted.
    assert _is_strict_prefix_of_tool_call_opener("<tool_call:opensource")
    assert _is_strict_prefix_of_tool_call_opener("<tool_call:foo-bar_v2")
    # Space is not in ``[\w-]`` — rejected.
    assert not _is_strict_prefix_of_tool_call_opener("<tool_call:foo bar")
    # Punctuation is not in ``[\w-]`` — rejected.
    assert not _is_strict_prefix_of_tool_call_opener("<tool_call:foo.bar")
    # Complete opener — NOT a strict prefix.
    assert not _is_strict_prefix_of_tool_call_opener("<tool_call>")
    assert not _is_strict_prefix_of_tool_call_opener("<tool_call:opensource>")


def test_streaming_json_body_with_corrupted_arg_value_tail_salvages():
    """Codex round-4 BLOCKING #2 regression test. A JSON-body stream can
    end with a stray ``</arg_value>`` (4-bit noise corrupting the JSON
    tail) — no ``<arg_key>`` opener, no ``<arg_value>`` opener, just
    ``NAME<tool_sep>{"k":"v"}</arg_value>``. The round-3 fix put this
    into XML-pair mode (canonical-only close) because ``<tool_sep>`` was
    present, so the salvage close never fired and the streaming path
    hung waiting for ``<end_of_tool_call>`` that never arrives. The
    round-4 fix inspects the body-prefix of EACH ``</arg_value>``
    individually — a prefix with no arg-key/value opener still fires the
    salvage close, letting the parser emit rather than hang."""
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
    # JSON body — well-formed argument dict, then a stray malformed
    # close in the tail (no <arg_key>/<arg_value> ever emitted).
    assert step('{"city": "NYC"}') is None
    final = step("</arg_value:opensource>")
    assert final is not None, (
        "Corrupted JSON tail with </arg_value> failed to trigger "
        "salvage close — regressing codex round-4 BLOCKING #2."
    )
    tc = final["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"


def test_auto_config_regex_boundary_rejects_incidental_substring():
    """Codex round-4 BLOCKING #1 regression test. The Hy3 auto-config
    regex in ``model_auto_config.py`` MUST reject incidental substring
    matches — an unrelated HF path like ``mymodelhy3embedded`` must NOT
    auto-wire to the Hy3 tool/reasoning parsers just because it happens
    to contain ``hy3`` as a substring.

    Duplicated across ``model_auto_config`` and ``chat_template`` so a
    future refactor to a shared helper can drop one without losing
    coverage — the two entry points share the same boundary policy."""
    import re as _re

    from vllm_mlx.model_auto_config import _MODEL_PATTERNS

    # Locate the Hy3 pattern in the auto-config table.
    hy3_pattern = None
    for pattern, config in _MODEL_PATTERNS:
        if getattr(config, "tool_call_parser", None) == "hy_v3":
            hy3_pattern = pattern
            break
    assert hy3_pattern is not None, "Hy3 auto-config pattern not found"

    # Positives — must match.
    for name in [
        "hy3-preview-4bit",
        "mlx-community/Hy3-preview-4bit",
        "Hunyuan-3-Preview",
        "hunyuan3",
        "hy-v3-experimental",
    ]:
        assert hy3_pattern.search(name), f"expected match on {name!r}"

    # Negatives — must NOT match. These are the exact strings codex
    # cited as auto-wiring wrongly under the unanchored regex.
    for name in [
        "mymodelhy3embedded",
        "not-hunyuanx3-test",
        "qwen3.5-4b-4bit",
        "gemma4-27b-8bit",
    ]:
        assert not hy3_pattern.search(name), (
            f"unexpected match on {name!r} — regressing round-4 BLOCKING #1"
        )
    # Silence unused-import lint on ``re`` — imported for reader clarity.
    _ = _re


def test_valid_names_filter_preserves_rejected_span_in_content():
    """Codex round-3 BLOCKING #2 regression test. When ``valid_names``
    is set and every parsed call is filtered out, the raw XML span of
    the rejected call MUST be preserved in ``content`` — silently
    dropping it makes the model output look like a refusal to the
    caller, when in fact the model tried to invoke an off-list tool.
    Preserving the span lets the caller diagnose the hallucination."""
    parser = HyV3ToolParser()
    out = '<tool_call>bogus_tool<tool_sep>{"x": 1}<end_of_tool_call>'
    request = {"tools": [{"function": {"name": "allowed_tool"}}]}
    res = parser.extract_tool_calls(out, request=request)
    assert res.tools_called is False
    assert res.tool_calls == []
    assert res.content is not None
    # The rejected span MUST survive in the content — the exact bytes
    # of the tool_call block matter for the caller's diagnostic.
    assert "<tool_call>bogus_tool<tool_sep>" in res.content
    assert "bogus_tool" in res.content


def test_valid_names_filter_preserves_mixed_valid_and_rejected():
    """Extension of the round-3 BLOCKING #2 fix: when SOME calls are
    valid, ``tools_called=True`` and ``content=None`` (per the
    "tool-call turn is exclusive" policy from round-2). The rejected
    span is lost in that case by design — the OpenAI-compatible
    contract does not permit mixed ``tool_calls`` + ``content`` in a
    single assistant turn. This test locks the mixed-case behaviour so
    a future edit doesn't accidentally start leaking the rejected span
    back into ``content`` and break the exclusive-turn contract."""
    parser = HyV3ToolParser()
    out = (
        '<tool_call>allowed_tool<tool_sep>{"y": 2}<end_of_tool_call>'
        "\n"
        '<tool_call>bogus_tool<tool_sep>{"x": 1}<end_of_tool_call>'
    )
    request = {"tools": [{"function": {"name": "allowed_tool"}}]}
    res = parser.extract_tool_calls(out, request=request)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "allowed_tool"
    # Exclusive-turn policy — no content when tools_called is True.
    assert res.content is None


def test_streaming_think_close_split_across_deltas_does_not_leak_close_tag():
    """Codex round-5 BLOCKING #1 regression test. A ``<think:opensource>``
    OPENER lands in ``previous_text`` and only the ``</think:opensource>``
    CLOSER arrives in ``delta_text`` alongside real content. The prior
    ``re.sub(<think>.*?</think>, "", delta_text)`` matched full pairs
    INSIDE delta_text only — so the closer literal AND the reasoning
    tail that followed leaked into the emitted content, delivering the
    end-user a broken payload like
    ``about this</think:opensource>reasoning done``.

    Fix: compute clean-baseline / clean-current with think spans
    stripped end-to-end (treating any unclosed opener in prev as a
    boundary that JUST closed), then emit only the diff. This test
    locks the split-delta close case with real content on both
    sides."""
    parser = HyV3ToolParser()
    parser.reset()
    prev = ""

    def step(delta: str):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_tool_calls_streaming(prev, cur, delta)
        prev = cur
        return msg

    # Delta 1 opens the think span with the ``:opensource`` suffix.
    assert step("<think:opensource>let me think") is None
    # Delta 2 closes the span AND carries reasoning-done tail content.
    # The emitted content MUST NOT include the ``</think:opensource>``
    # literal, and MUST NOT include the pre-close think text.
    r = step(" about this</think:opensource>reasoning done")
    assert r is not None
    content = r["content"]
    assert "</think" not in content, (
        f"think closer literal leaked into content: {content!r}"
    )
    assert "let me think" not in content, f"pre-close think content leaked: {content!r}"
    assert "about this" not in content, f"tail-of-think reasoning leaked: {content!r}"
    # The real post-close content MUST survive.
    assert "reasoning done" in content


def test_streaming_sep_less_xml_pair_across_three_deltas_emits_on_end_of_tool_call():
    """Codex round-5 BLOCKING #2 was flagged but IS a false positive —
    the round-4 ``_closed_after_opener`` correctly considers the
    ``<arg_value>`` opener when scanning the body-prefix. This test
    locks the working behaviour so a future refactor doesn't
    accidentally regress on codex's scenario:

    * Delta 1: ``<tool_call>fn<arg_key>x</arg_key>`` — arg_key done,
      no arg_value opener yet.
    * Delta 2: ``<arg_value>1</arg_value>`` — opener AND closer both
      inside delta.
    * Delta 3: ``<end_of_tool_call>`` — canonical close.

    Codex asserted the parser would think delta 2 was already closed
    and never emit on delta 3. In reality ``_closed_after_opener``
    sees the ``<arg_value>`` opener in the body-prefix and correctly
    skips the ``</arg_value>`` as an argument-close (not tool-close),
    so delta 3's canonical close cleanly triggers the emit."""
    parser = HyV3ToolParser()
    parser.reset()
    prev = ""

    def step(delta: str):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_tool_calls_streaming(prev, cur, delta)
        prev = cur
        return msg

    assert step("<tool_call>fn<arg_key>x</arg_key>") is None
    assert step("<arg_value>1</arg_value>") is None
    final = step("<end_of_tool_call>")
    assert final is not None, (
        "Sep-less XML-pair body did not emit on canonical close — "
        "regression against codex round-5 BLOCKING #2 (false positive)."
    )
    assert final["tool_calls"][0]["function"]["name"] == "fn"
    assert json.loads(final["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
