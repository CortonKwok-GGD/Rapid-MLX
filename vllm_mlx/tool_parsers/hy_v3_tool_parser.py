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
# Canonical close only — the streaming pending-check MUST use this
# stricter form, not the malformed-tolerant ``_TOOL_CALL_CLOSE_ANY``,
# because the XML-pair body legitimately contains ``</arg_value>`` for
# every argument value. Treating those as closes would emit the call
# prematurely on the first ``</arg_value>`` (codex round-2 BLOCKING #1).
# The malformed-close fallback (``</arg_value>`` without a matching
# ``<end_of_tool_call>``) is handled by the non-streaming extractor
# which sees the whole body at once and can distinguish the "no
# ``<tool_sep>`` in the body" malformed case from the XML-pair case.
_TOOL_CALL_CANONICAL_CLOSE = re.compile(rf"<end_of_tool_call{_SUFFIX}>")
# Malformed OR canonical — used only by ``extract_tool_calls`` (non-
# streaming) where we can safely distinguish which shape applies.
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

# Streaming JSON-decoder used by ``_parse_hy3_body`` to consume only a
# well-formed JSON prefix of the argument tail. Round-5 BLOCKING #3
# fix — avoids corrupting a valid JSON body whose string values
# legitimately contain the literal ``</arg_value>`` substring.
_JSON_DECODER = json.JSONDecoder()

# Same character class as the ``:LABEL`` suffix in the tool_call opener
# regex — kept in lockstep with ``_SUFFIX`` (``(?::[\w-]+)?``) so
# ``_is_strict_prefix_of_tool_call_opener`` accepts exactly the alphabet
# that a real opener would (codex round-6 NIT, PR #1070).
_LABEL_CHAR_RE = re.compile(r"[\w-]+")


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
        # No separator — two shapes to disambiguate (codex round-3
        # BLOCKING #1 extension):
        #   (a) sep-less XML-pair body:
        #       ``NAME<arg_key>K</arg_key><arg_value>V</arg_value>...``
        #       (some 4-bit checkpoints skip <tool_sep> but still emit
        #       full XML-pair args). Detect via the first arg-key /
        #       arg-value opener and treat it as an implicit sep.
        #   (b) short-form salvage: no sep, no arg tags — the whole
        #       body is the tool name (with ``</arg_value>`` residue
        #       stripped defensively).
        arg_open = _ARG_KEY_OPEN.search(body) or _ARG_VALUE_OPEN.search(body)
        if arg_open is not None:
            name = body[: arg_open.start()].strip()
            tail = body[arg_open.start() :]
        else:
            raw_name = body.strip()
            raw_name = _ARG_VALUE_CLOSE.sub("", raw_name)
            raw_name = _ARG_KEY_CLOSE.sub("", raw_name)
            return raw_name.strip(), {}
    else:
        name = body[: sep_match.start()].strip()
        tail = body[sep_match.end() :]

    # Variant 1: JSON object payload.
    tail_stripped = tail.strip()
    if tail_stripped.startswith("{"):
        # Codex round-5 BLOCKING #3 (PR #1070): DO NOT truncate at the
        # first ``</arg_value>`` before json.loads — a valid JSON
        # payload can legitimately contain that literal inside a
        # string value (``{"snippet": "see </arg_value> below"}``) and
        # slicing there would corrupt the args into empty/truncated.
        # Use ``raw_decode`` to consume only a well-formed JSON prefix
        # of the tail; any structural residue after that (the
        # ``</arg_value>`` malformed close, whitespace, extra tags) is
        # discarded harmlessly.
        try:
            parsed, _end = _JSON_DECODER.raw_decode(tail_stripped)
            if isinstance(parsed, dict):
                return name, parsed
        except (json.JSONDecodeError, ValueError):
            # Fall through to variant 2 (XML-pair) — same as before.
            # If raw_decode failed on the untrimmed tail, structural
            # residue is unlikely to be the cause (JSON is delimited);
            # a genuinely malformed JSON body degrades cleanly to the
            # XML-pair walker.
            pass

    # Variant 2: <arg_key>K</arg_key><arg_value>V</arg_value> pairs.
    args: dict[str, Any] = {}
    for m in _ARG_PAIR.finditer(tail):
        key = m.group("key").strip()
        if not key:
            continue
        args[key] = _deserialize_arg_value(m.group("val"))
    return name, args


def _tool_call_open_straddle_suffix_len(text: str) -> int:
    r"""Return the byte length of a trailing partial ``<tool_call>`` /
    ``<tool_call:label>`` opener that hasn't fully arrived yet.

    The Hy3 opener regex is ``<tool_call(?::[\w-]+)?>``; on an SSE
    boundary a delta can end with any strict prefix of that string,
    e.g. ``<``, ``<t``, ``<tool_c``, ``<tool_call``, ``<tool_call:``,
    ``<tool_call:opens``. Codex round-5 BLOCKING #1: emitting those
    bytes as ``content`` leaks tool markup to OpenAI-compatible
    clients seconds before the opener completes and the parser
    switches to tool-call turn suppression.

    Returns 0 when no straddle exists (the trailing bytes cannot
    become a valid opener no matter what arrives next).
    """
    if not text:
        return 0
    # Longest possible partial-opener length is
    #   len("<tool_call:") + reasonable suffix len bound.
    # Any label longer than 32 chars is unrealistic — cap the scan.
    max_len = min(len(text), 48)
    # Walk from longest candidate suffix down to length 1; return the
    # first (longest) one that is a strict prefix of a valid opener.
    for cand_len in range(max_len, 0, -1):
        cand = text[-cand_len:]
        if _is_strict_prefix_of_tool_call_opener(cand):
            return cand_len
    return 0


def _is_strict_prefix_of_tool_call_opener(cand: str) -> bool:
    r"""Return True iff ``cand`` is a strict prefix of some valid Hy3
    tool_call opener (``<tool_call>`` or ``<tool_call:LABEL>`` with
    a non-empty ``[\w-]+`` label). A prefix is "strict" when the
    opener isn't complete yet — the parser hasn't emitted the ``>``.
    """
    # Must start at the ``<`` so it's a real straddle, not incidental
    # text.
    base = "<tool_call"
    if not cand:
        return False
    if not base.startswith(cand) and not cand.startswith(base):
        return False
    if base.startswith(cand):
        # Prefix of the bare opener itself (``<``, ``<t``, ... up to
        # ``<tool_call``). Strict because ``>`` hasn't arrived.
        return cand != base + ">"
    # cand starts with ``<tool_call`` — is the tail either the ``:``
    # prefix, a partial label, or exactly the terminator ``>``?
    tail = cand[len(base) :]
    if tail == "":
        return True
    if tail == ">":
        # Complete bare opener — NOT a straddle.
        return False
    if not tail.startswith(":"):
        # ``<tool_call?`` — cannot become a valid opener.
        return False
    label = tail[1:]
    if label == "":
        # ``<tool_call:`` — awaiting label.
        return True
    # ``<tool_call:LABEL`` — accept if the LABEL matches the same
    # character set as the compiled opener regex (``[\w-]+``). Codex
    # round-6 NIT (PR #1070) flagged an earlier ``str.isalnum()``
    # variant as accepting a slightly different alphabet than the
    # actual opener regex (Python ``\w`` covers underscore too,
    # ``isalnum`` doesn't; both accept Unicode letters). Delegate to
    # the same regex character class to keep the two definitions
    # in lockstep.
    if label.endswith(">"):
        # Complete labelled opener — NOT a straddle.
        return False
    return _LABEL_CHAR_RE.fullmatch(label) is not None


def _closed_after_opener(after_opener: str) -> bool:
    """Return True iff ``after_opener`` (the substring starting AT the
    ``<tool_call>`` opener) already contains an effective close.

    An "effective close" is either:

    * the canonical ``<end_of_tool_call>`` tag anywhere in the substring,
      OR
    * a ``</arg_value>`` tag whose body-prefix (from the opener up to
      that ``</arg_value>``) contains NO ``<arg_value>`` opener AND NO
      ``<arg_key>`` opener — that is, the true malformed-salvage shape
      ``<tool_call>NAME</arg_value>`` (empirically observed on 4-bit
      Hy3 quants). Any earlier ``</arg_value>`` in a body that has
      already emitted an ``<arg_value>`` / ``<arg_key>`` opener is
      closing a real argument value, not the tool_call itself.

    Codex round-4 BLOCKING #2 (PR #1070): the round-3 implementation
    used any-marker XML-pair mode detection (tool_sep OR arg_key OR
    arg_value openers all counted). But ``<tool_sep>`` appears in BOTH
    JSON-body streams (``NAME<tool_sep>{...}``) and XML-pair streams,
    so a JSON stream with corrupted-tail ``</arg_value>`` would fall
    to canonical-only mode and hang until timeout waiting for
    ``<end_of_tool_call>``. Switching the discriminator to
    per-``</arg_value>`` prefix inspection recovers the salvage path
    on corrupted JSON while still preventing early flushes on real
    XML-pair bodies.
    """
    if _TOOL_CALL_CANONICAL_CLOSE.search(after_opener):
        return True
    for m in _ARG_VALUE_CLOSE.finditer(after_opener):
        prefix = after_opener[: m.start()]
        if _ARG_VALUE_OPEN.search(prefix):
            continue
        if _ARG_KEY_OPEN.search(prefix):
            continue
        # Codex round-6 BLOCKING #2 (PR #1070): if a ``<tool_sep>`` is
        # in the body prefix AND the material after it starts with an
        # opening JSON brace, we're in JSON-body mode. A ``</arg_value>``
        # inside a JSON string value is NOT a call close — the
        # non-streaming ``_parse_hy3_body`` uses ``raw_decode`` to
        # tolerate the literal, but the streaming close-check would
        # otherwise flush the call before the JSON body is complete.
        # Only treat the ``</arg_value>`` as salvage when it appears
        # AFTER a well-formed JSON prefix has been decoded.
        sep_m = _TOOL_SEP.search(prefix)
        if sep_m is not None:
            tail_after_sep = prefix[sep_m.end() :].lstrip()
            if tail_after_sep.startswith("{"):
                try:
                    _, consumed = _JSON_DECODER.raw_decode(tail_after_sep)
                except (json.JSONDecodeError, ValueError):
                    # JSON is still incomplete — don't fire salvage yet.
                    continue
                # JSON parsed OK. The ``</arg_value>`` is legitimate
                # salvage only if it lands AFTER the decoded prefix
                # (i.e., the ``</arg_value>`` is trailing structural
                # residue, not inside the JSON body). Otherwise the
                # literal was consumed inside a string value.
                trailing_offset = sep_m.end() + (
                    len(prefix[sep_m.end() :]) - len(tail_after_sep) + consumed
                )
                if m.start() < trailing_offset:
                    continue
        return True
    return False


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

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Watermark tracking how many bytes of ``current_text`` have
        # been emitted as ``content`` on the streaming path. Used to
        # release bytes withheld by the partial-opener straddle guard
        # once the straddle either resolves into an opener (bytes are
        # dropped per exclusive-turn) or falsifies (bytes are emitted
        # on the next tick). Codex round-6 BLOCKING #1 (PR #1070).
        self._streamed_bytes: int = 0

    def reset(self) -> None:
        super().reset()
        self._streamed_bytes = 0

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
                match = (
                    canonical if canonical.start() <= malformed.start() else malformed
                )
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
                # Codex round-3 BLOCKING #2: preserve the raw span of the
                # rejected call in the residual text rather than silently
                # erasing it. Without this, a request that supplies a
                # ``tools`` allowlist and a model that hallucinates an
                # off-list tool name would see ``tools_called=False`` +
                # empty content — the user never learns the model tried
                # to call something. Surfacing the raw XML lets the
                # caller diagnose the hallucination.
                residual_parts.append(cleaned[match.start() : match.end()])
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

    def _last_unclosed_tool_call_position(self, text: str) -> int:
        """Return the offset of the LAST ``<tool_call>`` opener that has
        no matching close tag after it, or ``-1`` when every opener
        already closed.

        Close-tag semantics (codex round-2 / round-3 / round-4 BLOCKING
        #1 on PR #1070):

        * The canonical ``<end_of_tool_call>`` tag ALWAYS closes the
          call — anywhere after the opener.
        * A ``</arg_value>`` closes the call ONLY when it is the
          "malformed-salvage" shape — i.e., the body-prefix between the
          opener and that specific ``</arg_value>`` contains no
          ``<arg_value>`` opener AND no ``<arg_key>`` opener. In both
          the JSON-body wire (``NAME<tool_sep>{...}``) and the
          XML-pair-in-flight case, an ``</arg_value>`` after real body
          content is closing an argument value, not the call.

        The round-4 refinement over round-3: gating on ``<tool_sep>`` +
        arg-pair markers TREATED ``<tool_sep>`` alone as XML-pair mode,
        which mis-classified JSON-body streams — a JSON body with a
        stray ``</arg_value>`` (4-bit noise corrupting the JSON tail)
        would then wait forever for a canonical close that never
        arrives. Switching the discriminator to arg-key / arg-value
        openers ONLY (which are exclusive to the XML-pair shape) lets
        the salvage path recover a corrupted JSON body while still
        preventing early flushes on well-formed XML-pair bodies.
        """
        openers = [m.start() for m in _TOOL_CALL_OPEN.finditer(text)]
        for opener_pos in reversed(openers):
            after_opener = text[opener_pos:]
            if _closed_after_opener(after_opener):
                continue
            return opener_pos
        return -1

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

        Streaming rules (mirrors ``Glm47ToolParser`` policy):

        * Once ANY ``<tool_call>`` opener has entered ``current_text``,
          the assistant turn is a TOOL-CALL turn. Suppress every content
          delta after that — OpenAI-compatible clients treat
          ``tool_calls`` and ``content`` as mutually exclusive for a
          single assistant turn (codex round-2 BLOCKING #2).
        * The FIRST delta on which ``current_text`` transitions from
          "has an unclosed canonical opener" to "canonical close seen"
          is the delta on which we emit the tool_calls array. Compare
          ``previous_text`` state to ``current_text`` state (not "did
          the close token appear entirely in this delta") so a close
          split across SSE chunks still fires (codex round-1 #3).
        * Before any opener appears, pass content deltas through.
        """
        # Skip while inside a suffix-tolerant thinking span. Recompute
        # think-state on ``current_text`` (source of truth).
        if re.search(rf"<think{_SUFFIX}>", current_text) and not re.search(
            rf"</think{_SUFFIX}>", current_text
        ):
            return None

        prev_pending = self._last_unclosed_tool_call_position(previous_text)
        curr_pending = self._last_unclosed_tool_call_position(current_text)
        seen_opener = _TOOL_CALL_OPEN.search(current_text) is not None

        if prev_pending >= 0 and curr_pending < 0:
            # Transition: canonical close arrived (possibly split across
            # chunks). Parse and emit the tool_calls array.
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
            # Structural close but body didn't parse — suppress the
            # residue rather than leak <tool_call> literals to content.
            return None

        if seen_opener:
            # Any opener anywhere in the accumulated text — this turn is
            # a tool-call turn (codex round-2 BLOCKING #2). Suppress
            # further content deltas until the closer arrives (the
            # transition branch above handles that).
            return None

        # Codex round-6 BLOCKING #1/#2 (PR #1070): if ``current_text``
        # ends in a strict prefix of a ``<tool_call>`` opener, WITHHOLD
        # the whole delta — including any pre-straddle prose. Emitting
        # ``"Sure, "`` before the opener resolves would violate the
        # exclusive tool-call turn contract if the opener ends up
        # completing (an OpenAI-compatible client that already routed
        # the turn to a plain-content callback then has to switch to
        # tool_calls). Round-5's "pass through pre-straddle prefix"
        # behaviour was flagged as regressing this contract; the
        # correct policy is buffer-until-resolved, and this branch
        # runs BEFORE the think-close emit path so a straddle in the
        # same delta as a ``</think>`` still short-circuits.
        if _tool_call_open_straddle_suffix_len(current_text) > 0:
            return None

        # No opener seen yet — pass plain content through. Trim any
        # inline think tags that slipped through (only when the closer
        # is actually in this delta, so mid-stream deltas don't lose
        # inter-word spacing).
        #
        # Codex round-5 BLOCKING #1 (PR #1070): when a ``<think>`` OPENER
        # is in ``previous_text`` and only its ``</think>`` CLOSER arrives
        # in ``delta_text``, ``re.sub`` sees just the closer + tail in
        # the delta and does NOT strip anything (the pattern requires
        # opener-in-same-string), leaving the closer literal and any
        # post-closer content to leak to the client. Fix: compute
        # ``clean_current`` and ``clean_previous`` with think spans
        # stripped end-to-end, treating any unclosed opener in
        # ``previous_text`` as a boundary (its span JUST closed in
        # ``delta_text``). Emit only the diff — this handles every
        # straddle pattern (opener-in-prev, opener-and-closer-in-delta,
        # multiple spans, etc.) with one code path.
        if re.search(rf"</think{_SUFFIX}>", delta_text):

            def _strip_all_think(text: str) -> str:
                """Strip all matched ``<think>...</think>`` pairs from
                ``text`` (suffix-tolerant), plus the base parser's
                inline-tag stripper for any unpaired-close residue."""
                stripped = re.sub(
                    rf"<think{_SUFFIX}>.*?</think{_SUFFIX}>",
                    "",
                    text,
                    flags=re.DOTALL,
                )
                return self.strip_think_tags(stripped)

            # If ``previous_text`` ends with an unclosed think opener,
            # treat everything from that opener onwards as span content
            # that JUST closed in this delta — i.e., truncate prev to
            # the opener boundary before computing the clean baseline.
            unclosed_open_in_prev: re.Match | None = None
            for m in re.finditer(rf"<think{_SUFFIX}>", previous_text):
                tail = previous_text[m.end() :]
                if not re.search(rf"</think{_SUFFIX}>", tail):
                    unclosed_open_in_prev = m
                    break
            if unclosed_open_in_prev is not None:
                clean_prev = _strip_all_think(
                    previous_text[: unclosed_open_in_prev.start()]
                )
            else:
                clean_prev = _strip_all_think(previous_text)
            clean_curr = _strip_all_think(current_text)

            if clean_curr.startswith(clean_prev):
                clean_delta = clean_curr[len(clean_prev) :]
            else:
                # Defensive fallback — should not happen with well-formed
                # streams. Emit the full clean current to avoid dropping
                # content on a defensive-inconsistency edge.
                clean_delta = clean_curr

            if clean_delta:
                return {"content": clean_delta}
            return None
        # Straddle-falsified / no-straddle path: emit any bytes past
        # the watermark ``self._streamed_bytes`` as content. This
        # releases the bytes that a prior tick's straddle check
        # withheld, if the straddle turned out not to be an opener.
        # In the common "no prior straddle" case, ``_streamed_bytes``
        # equals ``len(previous_text)`` and the returned content is
        # exactly ``delta_text``.
        emit_start = self._streamed_bytes
        emit_end = len(current_text)
        if emit_end <= emit_start:
            return None
        content = current_text[emit_start:emit_end]
        self._streamed_bytes = emit_end
        if content:
            return {"content": content}
        return None

    def has_pending_tool_call(self, text: str) -> bool:
        """Override — Hy3 uses ``<tool_call:opensource>`` (suffix-tolerant).

        Scoped to UNCLOSED openers so a completed call earlier in
        ``text`` doesn't falsely leave the parser "pending" forever
        (aligns with the streaming-gate fix in codex round-1).
        """
        if self._last_unclosed_tool_call_position(text) >= 0:
            return True
        return self.has_text_format_tool_call(text)
