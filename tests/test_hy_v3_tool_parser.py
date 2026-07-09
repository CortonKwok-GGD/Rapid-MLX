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


def test_streaming_partial_opener_does_not_leak_as_content():
    """Codex round-5 BLOCKING #1 regression test. A ``<tool_call:opensource>``
    opener that arrives split across SSE deltas (e.g. delta 1
    ``"Sure, <tool_ca"``, delta 2 ``"ll:opensource>..."``) MUST NOT
    leak the ``<tool_ca`` bytes as plain content — an OpenAI-compatible
    client would render tool markup before the tool-call turn engages,
    then have to interpret it as prose.

    The fix withholds the trailing partial-opener bytes on the delta
    that contains them, and releases the withheld bytes as content
    ONLY if the next chunk falsifies the opener guess (rare but
    handled: the bytes are re-derivable from ``current_text`` on the
    next tick)."""
    parser = HyV3ToolParser()
    parser.reset()
    prev = ""

    def step(delta: str):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_tool_calls_streaming(prev, cur, delta)
        prev = cur
        return msg

    # Delta 1: prose + partial opener. The prose bytes pass through as
    # content; the partial-opener bytes MUST be withheld.
    m1 = step("Sure, <tool_ca")
    if m1 is not None:
        # If anything emitted, it MUST be prose only — no ``<tool_ca``
        # bytes may reach the content channel.
        assert "<tool" not in (m1.get("content") or ""), (
            f"Partial opener leaked as content: {m1!r}"
        )
    # Delta 2: opener completes. The turn is now a tool-call turn.
    m2 = step("ll:opensource>get_weather<tool_sep:opensource>{}")
    # Once the opener has appeared, all further deltas are suppressed
    # until the close arrives. This delta MUST NOT emit content.
    assert m2 is None
    # Delta 3: canonical close — emit the tool_calls array.
    m3 = step("<end_of_tool_call:opensource>")
    assert m3 is not None
    assert m3["tool_calls"][0]["function"]["name"] == "get_weather"


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
