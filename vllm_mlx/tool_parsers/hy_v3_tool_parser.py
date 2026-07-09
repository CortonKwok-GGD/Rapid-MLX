# SPDX-License-Identifier: Apache-2.0
"""
Hy3 (Tencent Hunyuan 3) tool call parser for rapid-mlx.

Handles the Hy3 tool call wire format that emerged from mlx-lm PR #1211 and
Tencent's ``mlx-community/Hy3-preview-4bit`` chat template. The canonical
form is:

    <tool_call:opensource>NAME<tool_sep:opensource>{"k1": "v1", ...}<end_of_tool_call:opensource>

with variant emitting explicit ``<arg_key:opensource>K</arg_key:opensource>
<arg_value:opensource>V</arg_value:opensource>`` pairs instead of a JSON body.

**Suffix tolerance.** The tokenizer bakes ``:opensource`` into every wire
token (marking the "opensource" reasoning-mode variant), but future model
revisions may drop the suffix or swap it for another label (``:v1``,
``:internal``, …). All open/close tags in this parser are matched with
``(?::[\\w-]+)?`` so both the current stream (``<tool_call:opensource>``)
and the future stream (``<tool_call>``) hit the same code path — no branch
for wire-shape drift.

**Defensive close-tag strip (4-bit numerical noise mitigation).** The 4-bit
quantized HY3 checkpoint (both REAP50 and REAP75 variants) occasionally
emits a malformed close boundary — the model skips the ``<tool_sep>`` and
the arguments block entirely and jumps straight to ``</arg_value>``:

    <tool_call:opensource>get_weather</arg_value:opensource>

Empirically observed on 4/32 real prompts against ``Hy3-preview-4bit`` in
the 2026-07-09 spike (agent ``ac2851864dbd17b07``). The upstream fix will
land in a full-precision retraining pass, but until then the parser must
gracefully surface the tool name even when the arguments block is missing —
otherwise the user sees an empty ``tool_calls`` array and thinks the model
refused. The recovery: extract the raw name between ``<tool_call>`` and the
first close-tag artifact, return ``arguments="{}"``, mark tools_called=True.

**Wire format label.** ``hy3_native`` — declared in
``EXPECTED_WIRE_FORMATS`` so ``test_every_parser_declares_formats``
recognizes the new format without requiring a symbol-table update
(``WIRE_FORMAT_LABELS`` in ``abstract_tool_parser.py`` is a documentation
manifest, not an enforced allowlist against subclass declarations).
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def generate_tool_id() -> str:
    """Generate a unique tool call ID (OpenAI-compatible short form)."""
    return f"call_{uuid.uuid4().hex[:8]}"


# Suffix-tolerant tag alternation (``:opensource``, ``:v1``, …). ``\w-``
# covers the label characters upstream is likely to use; a broader
# character class would risk swallowing genuine punctuation in the model
# output. Applied to every open/close tag below.
_SUFFIX = r"(?::[\w-]+)?"

# Canonical tool_call block — non-greedy body up to ``<end_of_tool_call>``.
# This is the shape emitted by a well-formed Hy3 checkpoint. The XML-pair
# variant lives inside this block too because ``</arg_value>`` appears
# BEFORE ``<end_of_tool_call>``.
_TOOL_CALL_BLOCK = re.compile(
    rf"<tool_call{_SUFFIX}>"
    rf"(?P<body>.*?)"
    rf"<end_of_tool_call{_SUFFIX}>",
    re.DOTALL,
)

# Malformed close fallback — ``<tool_call>NAME</arg_value>`` with no
# ``<end_of_tool_call>`` in sight. Fires only when the canonical block
# regex misses. The body is captured up to the FIRST ``</arg_value>``
# because on 4-bit checkpoints the model jumps straight there without
# emitting ``<tool_sep>`` / args (see module docstring). Explicitly
# forbids ``<end_of_tool_call>`` inside the body so we don't shadow the
# canonical parse.
_TOOL_CALL_MALFORMED = re.compile(
    rf"<tool_call{_SUFFIX}>"
    rf"(?P<body>(?:(?!<end_of_tool_call{_SUFFIX}>).)*?)"
    rf"</arg_value{_SUFFIX}>",
    re.DOTALL,
)

# Standalone open-tag detector for suffix-tolerant streaming trigger.
_TOOL_CALL_OPEN = re.compile(rf"<tool_call{_SUFFIX}>")
_TOOL_CALL_CLOSE_ANY = re.compile(rf"<end_of_tool_call{_SUFFIX}>|</arg_value{_SUFFIX}>")

# Body-level tags.
_TOOL_SEP = re.compile(rf"<tool_sep{_SUFFIX}>")
_ARG_KEY_OPEN = re.compile(rf"<arg_key{_SUFFIX}>")
_ARG_KEY_CLOSE = re.compile(rf"</arg_key{_SUFFIX}>")
_ARG_VALUE_OPEN = re.compile(rf"<arg_value{_SUFFIX}>")
_ARG_VALUE_CLOSE = re.compile(rf"</arg_value{_SUFFIX}>")

# Extract an <arg_key>K</arg_key><arg_value>V</arg_value> pair. Both open
# tags are suffix-tolerant. Reusable at parse-time to walk the argument
# block sequentially.
_ARG_PAIR = re.compile(
    rf"<arg_key{_SUFFIX}>\s*(?P<key>.*?)\s*</arg_key{_SUFFIX}>\s*"
    rf"<arg_value{_SUFFIX}>(?P<val>.*?)</arg_value{_SUFFIX}>",
    re.DOTALL,
)


def _deserialize_arg_value(value: str) -> Any:
    """Coerce a raw ``<arg_value>`` payload to a JSON-native Python type.

    Tries ``json.loads`` first so ``true`` / ``42`` / ``[1,2]`` / ``null``
    round-trip cleanly; falls back to the trimmed string for free-form
    text so we don't silently drop the argument.
    """
    stripped = value.strip()
    if not stripped:
        return stripped
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return stripped


def _parse_hy3_body(body: str) -> tuple[str, dict[str, Any]]:
    """Parse the body of a ``<tool_call>…</end_of_tool_call>`` (or the
    malformed ``</arg_value>``) block into ``(name, arguments)``.

    Two on-wire variants coexist:

    1. **JSON body** — ``NAME<tool_sep>{"k": "v", …}`` — the chat-template's
       default emission when the model spells out a JSON object.
    2. **XML pair body** — ``NAME<tool_sep><arg_key>k</arg_key>
       <arg_value>v</arg_value>…`` — the malformed-with-real-args case
       when the model transcribes each pair separately.
    3. **Malformed close** — ``NAME`` alone (the ``</arg_value>`` early
       close ate the sep + args). Returns ``(NAME, {})`` so the outer
       extractor still surfaces the call rather than dropping it.

    We probe (1) first because the JSON body is what the chat template
    documents; (2) is the fallback; (3) is the trivial residue.
    """
    # Split at the first <tool_sep> if present.
    sep_match = _TOOL_SEP.search(body)
    if sep_match is None:
        # No separator — the tool_sep + args block never arrived.
        # Return the whole trimmed body as the name (defensive strip
        # case). Extra ``</arg_value>`` / ``</arg_key>`` residue can
        # leak in when the model emits a partial close pair; strip
        # those literally so ``get_weather`` doesn't become
        # ``get_weather</arg_value>``.
        raw_name = body.strip()
        raw_name = _ARG_VALUE_CLOSE.sub("", raw_name)
        raw_name = _ARG_KEY_CLOSE.sub("", raw_name)
        return raw_name.strip(), {}
    name = body[: sep_match.start()].strip()
    tail = body[sep_match.end() :]

    # Variant 1: JSON object payload.
    tail_stripped = tail.strip()
    # Trim any trailing structural residue (``</arg_value>`` etc.) so
    # ``{"k":"v"}</arg_value>`` still json.loads cleanly.
    trailing_start = _ARG_VALUE_CLOSE.search(tail_stripped)
    if trailing_start is not None:
        tail_stripped = tail_stripped[: trailing_start.start()].strip()
    if tail_stripped.startswith("{"):
        try:
            parsed = json.loads(tail_stripped)
            if isinstance(parsed, dict):
                return name, parsed
        except (json.JSONDecodeError, ValueError):
            pass  # Fall through to variant 2.

    # Variant 2: <arg_key>K</arg_key><arg_value>V</arg_value> pairs.
    args: dict[str, Any] = {}
    for m in _ARG_PAIR.finditer(tail):
        key = m.group("key").strip()
        if not key:
            continue
        args[key] = _deserialize_arg_value(m.group("val"))
    return name, args


@ToolParserManager.register_module(["hy_v3", "hy3"])
class HyV3ToolParser(ToolParser):
    """
    Tool call parser for Tencent Hunyuan 3 (``mlx-community/Hy3-preview-4bit``).

    Format:
        <tool_call:opensource>NAME<tool_sep:opensource>{"k":"v",...}
        <end_of_tool_call:opensource>

    or the XML-pair variant:
        <tool_call:opensource>NAME<tool_sep:opensource>
        <arg_key:opensource>K</arg_key:opensource>
        <arg_value:opensource>V</arg_value:opensource>
        <end_of_tool_call:opensource>

    Suffix-tolerant (``:opensource`` optional) and defensive against the
    4-bit malformed close where the model skips ``<tool_sep>`` and jumps
    straight to ``</arg_value>``.

    Used when ``--enable-auto-tool-choice --tool-call-parser hy_v3`` are
    set, or auto-wired for the ``hy3-preview-4bit`` alias via
    ``aliases.json``.
    """

    # The Hy3 chat template renders assistant ``tool_calls`` back into the
    # SAME ``<tool_call:opensource>…<end_of_tool_call:opensource>`` markup
    # the parser reads. Feed previous-turn tool calls in native format
    # rather than converting them to synthetic text.
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("hy3_native",)

    def _get_tool_names(self, request: dict[str, Any] | None) -> set[str]:
        """Extract valid tool names from the request payload."""
        if not request or "tools" not in request:
            return set()
        return {
            t.get("function", {}).get("name", "")
            for t in request.get("tools", [])
            if isinstance(t, dict)
        }

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract Hy3 tool calls from a complete model response."""
        tool_calls: list[dict[str, Any]] = []

        # Strip <think> and <think:opensource> tags so a call embedded
        # in the reasoning span still surfaces when no reasoning parser
        # is configured (defensive parity with hermes / glm47). Handle
        # the ``:opensource`` variant separately since the base
        # ``strip_think_tags`` only matches the plain form.
        cleaned = re.sub(
            rf"<think{_SUFFIX}>.*?</think{_SUFFIX}>",
            "",
            model_output,
            flags=re.DOTALL,
        )
        cleaned = self.strip_think_tags(cleaned)

        valid_names = self._get_tool_names(request)

        # Walk left-to-right, preferring the canonical
        # ``<end_of_tool_call>`` close. When no canonical block matches at
        # the current cursor position, retry with the malformed-close
        # regex so the 4-bit ``<tool_call>NAME</arg_value>`` shape still
        # surfaces the tool name.
        residual_parts: list[str] = []
        cursor = 0
        length = len(cleaned)
        while cursor < length:
            canonical = _TOOL_CALL_BLOCK.search(cleaned, cursor)
            malformed = _TOOL_CALL_MALFORMED.search(cleaned, cursor)
            # Whichever opener starts earliest wins; canonical wins on ties
            # so a well-formed block never falls to the malformed regex.
            match = None
            if canonical is not None and malformed is not None:
                match = canonical if canonical.start() <= malformed.start() else malformed
            elif canonical is not None:
                match = canonical
            elif malformed is not None:
                match = malformed
            if match is None:
                break
            residual_parts.append(cleaned[cursor : match.start()])
            body = match.group("body")
            name, args = _parse_hy3_body(body)
            if not name:
                cursor = match.end()
                continue
            if valid_names and name not in valid_names:
                cursor = match.end()
                continue
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                }
            )
            cursor = match.end()
        residual_parts.append(cleaned[cursor:])
        residual_text = "".join(residual_parts).strip()

        if tool_calls:
            # Suppress reasoning prose that precedes tool calls (same
            # policy as glm47: content=None when tools_called is True).
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=None,
            )

        # Text-format degradation fallback (``[Calling tool="X" k="v"]``)
        # is shared across parsers; consult it before giving up.
        if self.has_text_format_tool_call(residual_text):
            text_calls = self.extract_text_format_tool_calls(residual_text)
            if text_calls:
                # Normalise the shared helper's ``{name, arguments}`` shape.
                normalised = [
                    {
                        "id": tc.get("id", generate_tool_id()),
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    }
                    for tc in text_calls
                ]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=normalised,
                    content=None,
                )

        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=residual_text or cleaned,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Extract Hy3 tool calls from streaming model output.

        Simple buffer-until-close strategy: while an unclosed
        ``<tool_call>`` opener is in flight, suppress content deltas
        (matches glm47's policy). When a close tag arrives in this
        delta, parse the full accumulated text with
        ``extract_tool_calls`` and emit the tool_calls array.

        Suppresses ``<think:opensource>`` reasoning content (mirrors the
        glm47 streaming policy of holding reasoning bytes when a
        downstream reasoning parser handles them).
        """
        # Skip while inside a thinking span.
        if (
            re.search(rf"<think{_SUFFIX}>", current_text)
            and not re.search(rf"</think{_SUFFIX}>", current_text)
        ):
            return None

        if _TOOL_CALL_OPEN.search(current_text):
            if _TOOL_CALL_CLOSE_ANY.search(delta_text):
                result = self.extract_tool_calls(current_text, request)
                if result.tools_called:
                    return {
                        "tool_calls": [
                            {
                                "index": i,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                            for i, tc in enumerate(result.tool_calls)
                        ]
                    }
            # In-flight tool_call — suppress content until the closer arrives.
            return None

        # Passthrough content — trim think tags if they slipped through
        # (only when the closer is actually in the delta, so mid-stream
        # deltas don't lose inter-word spacing).
        if re.search(rf"</think{_SUFFIX}>", delta_text):
            clean = re.sub(
                rf"<think{_SUFFIX}>.*?</think{_SUFFIX}>",
                "",
                delta_text,
                flags=re.DOTALL,
            )
            clean = self.strip_think_tags(clean)
            if clean:
                return {"content": clean}
            return None
        if delta_text:
            return {"content": delta_text}
        return None

    def has_pending_tool_call(self, text: str) -> bool:
        """Override — Hy3 uses ``<tool_call:opensource>`` (suffix-tolerant)."""
        if _TOOL_CALL_OPEN.search(text) and not _TOOL_CALL_CLOSE_ANY.search(text):
            return True
        return self.has_text_format_tool_call(text)
