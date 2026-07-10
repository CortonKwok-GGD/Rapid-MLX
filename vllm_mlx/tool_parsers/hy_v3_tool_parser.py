# SPDX-License-Identifier: Apache-2.0
"""
Hy3 (Tencent Hunyuan 3) tool call parser for rapid-mlx.

Ported from vLLM's ``HYV3ToolParser`` (vllm/tool_parsers/hy_v3_tool_parser.py)
and SGLang's ``hunyuan_detector`` (python/sglang/srt/function_call/
hunyuan_detector.py::resolve_hunyuan_tokens). The wire format is:

    <tool_call:opensource>NAME<tool_sep:opensource>{"k1": "v1", ...}<end_of_tool_call:opensource>

with an ``<arg_key:opensource>K</arg_key:opensource><arg_value:opensource>V
</arg_value:opensource>`` XML-pair variant instead of / in addition to the
JSON body. The label suffix (``:opensource``) marks the checkpoint's
reasoning-mode variant; a future revision may drop it or swap it.

Design (why this file was rewritten from the bespoke full-text-regex design)
===========================================================================

The earlier rapid-mlx implementation re-parsed the entire accumulated text
on every streaming delta, used a suffix-alternation regex ``(?::[\\w-]+)?``
everywhere, and carried ~85 LOC of partial-opener straddle-guard code
(``_tool_call_open_straddle_suffix_len``, ``_is_strict_prefix_of_tool_call_
opener``, a ``_streamed_bytes`` content watermark). Seven codex rounds all
chased symptoms of that architecture. This rewrite ports vLLM/SGLang's
proven approach instead:

1. **Resolve the suffix ONCE at ``__init__``** by scanning
   ``tokenizer.get_vocab()`` for ``<tool_call(:LABEL)?>`` and pinning the
   real tag strings as FIXED strings (``self.tool_call_start_token``,
   ``self.tool_sep_token``, …). No regex alternation on the hot path.
   (SGLang ``resolve_hunyuan_tokens`` pattern.)

2. **Token-ID gate the streaming entry.** ``self.tool_call_start_token_id``
   is resolved from vocab; special tokens are ATOMIC on the tokenizer
   boundary so they cannot straddle SSE chunks. Once the start token id is
   absent from ``current_token_ids`` (and the fixed start string is absent
   from the accumulated text), the delta is pure content — pass it through.
   This single gate deletes the entire straddle-guard family.

3. **Buffer accumulates only INSIDE the tool-call span**, keyed on
   ``str.find`` of the pinned fixed strings — never a full-history regex
   re-scan.

4. **Two-phase state machine**: ``SEEKING_NAME`` (find ``<tool_sep>`` in the
   buffer → emit the function name) → ``STREAMING_ARGS`` (stream the args
   body incrementally, emitting a JSON diff and withholding the trailing
   ``}`` until ``<end_of_tool_call>``).

5. **``<think>`` handling lives entirely in the separate reasoning parser**
   (``vllm_mlx/reasoning/hy3_parser.py::Hy3ReasoningParser``, registered as
   ``--reasoning-parser hy_v3``). This tool parser has ZERO ``<think>`` code
   — the two parsers see disjoint token streams, exactly as vLLM's do.

6. **Watermark on args, not content**: ``self.streamed_args_for_tool`` holds
   the JSON already emitted per open tool call (vLLM base pattern).

7. **Malformed-close salvage** (``<tool_call>NAME</arg_value>`` — 4-bit
   numerical noise empirically observed on ``pipenetwork/Hy3-REAP50-MLX-4bit``
   and ``Hy3-REAP75-MLX-4bit``, 10/10 BFCL simple_python prompts) is
   rapid-mlx's unique value-add. It runs ONLY on the non-streaming
   ``extract_tool_calls`` path — 4-bit noise is rare and streaming clients
   re-parse on completion, so streaming never runs salvage.

**Wire format label.** ``hy3_native`` — declared in ``EXPECTED_WIRE_FORMATS``.
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from transformers import PreTrainedTokenizerBase

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

# Default label baked into ``pipenetwork/Hy3-*-MLX-4bit`` and the
# ``chat_template.jinja`` sentinel scheme. Used only when no tokenizer is
# available to resolve the real suffix from vocab.
_DEFAULT_LABEL = "opensource"

# Matches a Hy3 wire token in the vocab, capturing the bare name. ``resolve``
# uses this to pin the real (possibly suffixed) strings.
_VOCAB_TOKEN_RE = re.compile(r"^<(?P<name>[a-z_]+)(?::[\w-]+)?>$")


def generate_tool_id() -> str:
    """Generate a unique tool call ID (OpenAI-compatible short form)."""
    return f"call_{uuid.uuid4().hex[:8]}"


def _resolve_suffix(tokenizer: PreTrainedTokenizerBase | None) -> str:
    """Resolve the wire label suffix (e.g. ``:opensource``) ONCE from vocab.

    Scans ``tokenizer.get_vocab()`` for ``<tool_call(:LABEL)?>`` and returns
    the ``:LABEL`` string (including the leading colon) if present, or ``""``
    for the bare form. Falls back to ``:opensource`` when no tokenizer is
    available (the shape every current ``pipenetwork/Hy3-*-MLX-4bit``
    checkpoint emits). SGLang ``resolve_hunyuan_tokens`` pattern.
    """
    if tokenizer is not None:
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            vocab = None
        if isinstance(vocab, dict):
            # Anchor on the real ``<tool_call...>`` start token so the resolved
            # suffix is not an incidental sibling.
            for tok in vocab:
                if not isinstance(tok, str):
                    continue
                m = _VOCAB_TOKEN_RE.match(tok)
                if m and m.group("name") == "tool_call":
                    # ``tok`` is ``<tool_call>`` or ``<tool_call:LABEL>``.
                    return tok[len("<tool_call") : -1]  # ``""`` or ``:LABEL``
    return f":{_DEFAULT_LABEL}"


def _deserialize_arg_value(value: str) -> Any:
    """Coerce a raw ``<arg_value>`` payload to a JSON-native Python type.

    Tries ``json.loads`` first so ``true`` / ``42`` / ``[1,2]`` / ``null``
    round-trip cleanly; falls back to the trimmed string for free-form text
    so we do not silently drop the argument.
    """
    stripped = value.strip()
    if not stripped:
        return stripped
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return stripped


@ToolParserManager.register_module(["hy_v3", "hy3"])
class HyV3ToolParser(ToolParser):
    """
    Tool call parser for Tencent Hunyuan 3 (``pipenetwork/Hy3-*-MLX-4bit``).

    Format:
        <tool_call:opensource>NAME<tool_sep:opensource>{"k":"v",...}
        <end_of_tool_call:opensource>

    or the XML-pair argument variant:
        <tool_call:opensource>NAME<tool_sep:opensource>
        <arg_key:opensource>K</arg_key:opensource>
        <arg_value:opensource>V</arg_value:opensource>
        <end_of_tool_call:opensource>

    The label suffix (``:opensource``) is resolved once from the tokenizer
    vocab at ``__init__``; all matching downstream uses the pinned fixed
    strings. The 4-bit malformed close (``<tool_call>NAME</arg_value>``) is
    salvaged on the non-streaming path.

    Used when ``--enable-auto-tool-choice --tool-call-parser hy_v3`` are set,
    or auto-wired for the ``hy3-*`` aliases via ``aliases.json`` /
    ``model_auto_config``.
    """

    # The Hy3 chat template renders assistant ``tool_calls`` back into the
    # same ``<tool_call:opensource>…<end_of_tool_call:opensource>`` markup the
    # parser reads. Feed previous-turn tool calls in native format rather
    # than converting them to synthetic text.
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("hy3_native",)

    def __init__(self, tokenizer: PreTrainedTokenizerBase | None = None):
        super().__init__(tokenizer)

        # --- Resolve the suffix ONCE and pin FIXED tag strings (step 1) ---
        suffix = _resolve_suffix(tokenizer)  # ``":opensource"`` or ``""``
        self.suffix = suffix
        self.tool_call_start_token = f"<tool_call{suffix}>"
        self.tool_sep_token = f"<tool_sep{suffix}>"
        self.tool_call_end_token = f"<end_of_tool_call{suffix}>"
        self.arg_key_start_token = f"<arg_key{suffix}>"
        self.arg_key_end_token = f"</arg_key{suffix}>"
        self.arg_value_start_token = f"<arg_value{suffix}>"
        self.arg_value_end_token = f"</arg_value{suffix}>"

        # Non-streaming regex built from the FIXED strings (no alternation).
        esc = re.escape
        self._tool_call_block_re = re.compile(
            esc(self.tool_call_start_token)
            + r"(?P<body>.*?)"
            + esc(self.tool_call_end_token),
            re.DOTALL,
        )
        # Malformed close: ``<tool_call>NAME</arg_value>`` with no
        # ``<end_of_tool_call>`` — body captured up to the FIRST
        # ``</arg_value>``, explicitly forbidding an interior
        # ``<end_of_tool_call>`` so a well-formed block never falls here.
        self._tool_call_malformed_re = re.compile(
            esc(self.tool_call_start_token)
            + r"(?P<body>(?:(?!"
            + esc(self.tool_call_end_token)
            + r").)*?)"
            + esc(self.arg_value_end_token),
            re.DOTALL,
        )
        self._arg_pair_re = re.compile(
            esc(self.arg_key_start_token)
            + r"\s*(?P<key>.*?)\s*"
            + esc(self.arg_key_end_token)
            + r"\s*"
            + esc(self.arg_value_start_token)
            + r"(?P<val>.*?)"
            + esc(self.arg_value_end_token),
            re.DOTALL,
        )
        self._json_decoder = json.JSONDecoder()

        # --- Token-ID gate (step 2). Special tokens are atomic on the ---
        # tokenizer boundary, so the start id (when present) cannot straddle
        # an SSE chunk. ``None`` when the tokenizer does not expose the token
        # as a single id; the streaming path then falls back to the fixed
        # string containment check, which is still atomic per-delta because
        # the whole opener arrives in one delta once the model emits it.
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        self._reset_streaming_state()

    # ------------------------------------------------------------------
    # Streaming state
    # ------------------------------------------------------------------
    def _reset_streaming_state(self) -> None:
        # Phase: name not sent yet = SEEKING_NAME, else STREAMING_ARGS.
        self.current_tool_id: int = -1
        self._name_sent: bool = False
        self._current_tool_ref: str | None = None
        # JSON already emitted for the current tool's arguments (watermark on
        # args, NOT content). vLLM base ``streamed_args_for_tool`` pattern.
        self.streamed_args_for_tool: list[str] = []
        self.prev_tool_call_arr: list[dict] = []

    def reset(self) -> None:
        super().reset()
        self._reset_streaming_state()

    def _get_tool_names(self, request: dict[str, Any] | None) -> set[str]:
        """Extract valid tool names from the request payload."""
        if not request or "tools" not in request:
            return set()
        return {
            t.get("function", {}).get("name", "")
            for t in request.get("tools", [])
            if isinstance(t, dict)
        }

    def _opener_positions(self, text: str) -> list[int]:
        """Byte offsets of every ``<tool_call>`` opener in ``text``.

        A plain ``str.find`` scan on the pinned fixed opener string — the
        wire never nests tool_call openers, so each occurrence starts a new
        indexed call (the streaming FSM drives one index per opener)."""
        positions: list[int] = []
        start = 0
        tok = self.tool_call_start_token
        step = len(tok)
        while True:
            pos = text.find(tok, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + step
        return positions

    # ------------------------------------------------------------------
    # Body parsing (shared by non-streaming extraction)
    # ------------------------------------------------------------------
    def _parse_body(self, body: str) -> tuple[str, dict[str, Any]]:
        """Parse a ``NAME<tool_sep>ARGS`` body into ``(name, arguments)``.

        Two on-wire arg shapes coexist:

        1. **JSON body** — ``NAME<tool_sep>{"k": "v", …}`` — the chat
           template's default emission.
        2. **XML-pair body** — ``NAME<tool_sep><arg_key>k</arg_key>
           <arg_value>v</arg_value>…`` — each pair transcribed separately.

        A sep-less body (``NAME`` alone, or ``NAME`` followed straight by an
        ``<arg_key>``/``<arg_value>`` opener) is handled too: the residue is
        the name and the args are recovered from the pairs if present.
        """
        sep_idx = body.find(self.tool_sep_token)
        if sep_idx == -1:
            # No separator. Either sep-less XML pairs, or just a bare name.
            ak = body.find(self.arg_key_start_token)
            av = body.find(self.arg_value_start_token)
            candidates = [c for c in (ak, av) if c != -1]
            if candidates:
                arg_open = min(candidates)
                name = body[:arg_open].strip()
                tail = body[arg_open:]
            else:
                raw = body.strip()
                # Strip any close-tag residue defensively (malformed close).
                raw = raw.replace(self.arg_value_end_token, "")
                raw = raw.replace(self.arg_key_end_token, "")
                return raw.strip(), {}
        else:
            name = body[:sep_idx].strip()
            tail = body[sep_idx + len(self.tool_sep_token) :]

        return name, self._parse_args_tail(tail)

    def _parse_args_tail(self, tail: str) -> dict[str, Any]:
        """Parse the post-``<tool_sep>`` tail into an arguments dict.

        Probes the JSON-body shape first (``raw_decode`` consumes only a
        well-formed JSON prefix so a value string containing the literal
        ``</arg_value>`` round-trips unchanged), then falls back to walking
        ``<arg_key>K</arg_key><arg_value>V</arg_value>`` pairs.
        """
        stripped = tail.strip()
        if stripped.startswith("{"):
            try:
                parsed, _end = self._json_decoder.raw_decode(stripped)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        args: dict[str, Any] = {}
        for m in self._arg_pair_re.finditer(tail):
            key = m.group("key").strip()
            if not key:
                continue
            args[key] = _deserialize_arg_value(m.group("val"))
        return args

    # ------------------------------------------------------------------
    # Non-streaming extraction (with malformed-close salvage)
    # ------------------------------------------------------------------
    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract Hy3 tool calls from a complete model response.

        Malformed-close salvage (``<tool_call>NAME</arg_value>``) runs here —
        4-bit noise is rare and this path sees the whole body at once, so it
        can distinguish the malformed close from a legitimate interior
        ``</arg_value>`` in an XML-pair body. Streaming never runs salvage.
        """
        if self.tool_call_start_token not in model_output:
            # No native opener. A low-quant checkpoint may still degrade into
            # the shared ``[Calling tool="X" k="v"]`` text form, so consult
            # that fallback before returning plain content (codex BLOCKING:
            # the early return otherwise made this branch unreachable).
            return self._text_format_or_content(model_output)

        valid_names = self._get_tool_names(request)
        tool_calls: list[dict[str, Any]] = []
        residual_parts: list[str] = []
        cursor = 0
        length = len(model_output)
        while cursor < length:
            canonical = self._tool_call_block_re.search(model_output, cursor)
            malformed = self._tool_call_malformed_re.search(model_output, cursor)
            # Whichever opener starts earliest wins; canonical wins on ties so
            # a well-formed block never falls to the malformed-salvage regex.
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
            residual_parts.append(model_output[cursor : match.start()])
            name, args = self._parse_body(match.group("body"))
            if not name:
                cursor = match.end()
                continue
            if valid_names and name not in valid_names:
                # Preserve the rejected span in residual text so a request
                # with a ``tools`` allowlist + a hallucinated off-list name
                # surfaces the attempted call rather than a silent empty
                # ``tool_calls`` array that looks like a refusal.
                residual_parts.append(model_output[match.start() : match.end()])
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
        residual_parts.append(model_output[cursor:])
        residual_text = "".join(residual_parts).strip()

        if tool_calls:
            # Suppress content that precedes tool calls (exclusive-turn
            # policy — content=None when tools_called is True).
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=None
            )

        # No native call parsed — try the shared text-format degradation
        # fallback on the residual before giving up.
        return self._text_format_or_content(residual_text or model_output)

    def _text_format_or_content(self, text: str) -> ExtractedToolCallInformation:
        """Consult the shared ``[Calling tool="X" k="v"]`` text-format
        fallback, else return ``text`` as plain content.

        Shared by the no-native-opener early return AND the end of
        ``extract_tool_calls`` so the low-quant text-degradation path is
        reachable in both (codex BLOCKING #2)."""
        if self.has_text_format_tool_call(text):
            text_calls = self.extract_text_format_tool_calls(text)
            if text_calls:
                normalised = [
                    {
                        "id": tc.get("id", generate_tool_id()),
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    }
                    for tc in text_calls
                ]
                return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=normalised, content=None
                )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=text
        )

    # ------------------------------------------------------------------
    # Streaming extraction — token-ID gate + 2-phase FSM
    # ------------------------------------------------------------------
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

        Token-ID gate (step 2): before any ``<tool_call>`` opener has entered
        the stream, the delta is pure content — pass it through. A special
        token is atomic on the tokenizer boundary so the opener cannot
        straddle an SSE chunk; no partial-opener buffering is needed.

        2-phase FSM (step 4):
          * SEEKING_NAME — buffer from the opener until ``<tool_sep>`` lands,
            then emit the function name (arguments empty).
          * STREAMING_ARGS — stream the args body incrementally, emitting a
            JSON diff and withholding the trailing ``}`` until
            ``<end_of_tool_call>`` arrives.
        """
        if not previous_text:
            self._reset_streaming_state()

        # ---- Token-ID gate: no tool call has begun → pure content. ----
        if not self._opener_seen(current_text, current_token_ids):
            # Emit content, but hold back a trailing partial-opener prefix so
            # a char-split ``<tool_call:opensou`` (the tokenizer/driver did
            # not deliver the opener as one atomic special token) does not
            # leak raw markup to the client. This is a single-string prefix
            # hold (the established ``_safe_content_prefix`` idiom), NOT the
            # deleted 85-LOC suffix-alternation straddle machinery — the held
            # bytes are released the moment they either complete the opener
            # (handled below) or falsify into ordinary content.
            return self._emit_safe_content(previous_text, current_text)

        # A tool call has opened somewhere in ``current_text``. This turn is a
        # TOOL-CALL turn — content and tool_calls are mutually exclusive for a
        # single assistant turn, so plain content deltas are suppressed and
        # only tool-call deltas flow from here.
        return self._stream_tool_call(current_text, request)

    def _safe_content_prefix(self, text: str) -> str:
        """Return the portion of ``text`` safe to emit as content now.

        Holds back the longest suffix of ``text`` that is a non-empty proper
        prefix of the tool-call opener (``<tool_call:opensource>``), so a
        char-split opener never leaks. Returns ``text`` unchanged when its
        tail cannot begin the opener.
        """
        opener = self.tool_call_start_token
        max_hold = 0
        for length in range(min(len(text), len(opener) - 1), 0, -1):
            if text.endswith(opener[:length]):
                max_hold = length
                break
        return text if max_hold == 0 else text[: len(text) - max_hold]

    def _emit_safe_content(
        self, previous_text: str, current_text: str
    ) -> dict[str, Any] | None:
        """Emit the new content diff with a partial-opener tail held back.

        When everything new is a held opener prefix, returns ``None`` so no
        content event fires this round; the bytes surface once the tail
        resolves (opener completes → tool-call turn; or falsifies → content).
        """
        safe_current = self._safe_content_prefix(current_text)
        safe_previous = self._safe_content_prefix(previous_text)
        if len(safe_current) <= len(safe_previous):
            return None
        return {"content": safe_current[len(safe_previous) :]}

    def flush_held_content(self, full_text: str) -> str:
        """Release any prefix-held opener tail at stream end.

        A stream ending in ``abc<tool_ca`` (a partial opener that never
        completed) has held those bytes back; they are ordinary content and
        must be released so the last chars are not dropped.
        """
        # Only meaningful when no tool call actually opened — a real opener
        # commits the turn to tool_calls and the held tail is markup, not
        # content.
        if self.tool_call_start_token in full_text:
            return ""
        return full_text[len(self._safe_content_prefix(full_text)) :]

    def _opener_seen(
        self, current_text: str, current_token_ids: Sequence[int] | None
    ) -> bool:
        """True once the tool-call opener has entered the stream.

        Prefers the atomic token-ID signal (the opener is a single special
        token that cannot split across SSE chunks); falls back to the pinned
        fixed-string containment check when the tokenizer does not expose the
        opener as a single id.
        """
        if (
            self.tool_call_start_token_id is not None
            and current_token_ids is not None
            and self.tool_call_start_token_id in current_token_ids
        ):
            return True
        return self.tool_call_start_token in current_text

    def _stream_tool_call(
        self, current_text: str, request: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Drive the 2-phase FSM against the accumulated ``current_text``.

        Supports MULTIPLE tool calls in one turn: the parser advances index
        by index. The block for the call currently being streamed is the
        span from ITS opener up to the next opener (or end of text) — found
        by ``str.find`` on the pinned fixed strings, never a full re-parse.
        When the current call's ``<end_of_tool_call>`` lands, the args are
        finalized and the FSM transitions back to SEEKING_NAME so the next
        opener starts a fresh indexed call (codex BLOCKING: ``_name_sent``
        must reset per call).
        """
        opener_positions = self._opener_positions(current_text)
        if not opener_positions:
            return None

        # The call currently being processed. ``current_tool_id`` starts at
        # -1; the first SEEKING_NAME emit bumps it to 0. While streaming a
        # call it stays put; on close we advance to the next opener.
        idx = self.current_tool_id if self.current_tool_id >= 0 else 0
        if idx >= len(opener_positions):
            # We finished the last opener we know about and no new opener has
            # arrived yet — nothing to do this tick.
            return None

        opener_pos = opener_positions[idx]
        block_end = (
            opener_positions[idx + 1]
            if idx + 1 < len(opener_positions)
            else len(current_text)
        )
        buffer = current_text[opener_pos + len(self.tool_call_start_token) : block_end]
        sep_idx = buffer.find(self.tool_sep_token)

        # ---------- Phase 1: SEEKING_NAME ----------
        if not self._name_sent:
            if sep_idx == -1:
                # Name not yet delimited — keep buffering, emit nothing.
                return None
            name = buffer[:sep_idx].strip()
            if not name:
                return None
            valid_names = self._get_tool_names(request)
            if valid_names and name not in valid_names:
                # Hallucinated off-list name — do not emit a header. Suppress
                # (exclusive-turn) but still claim the index so the FSM can
                # advance to the next opener when this call closes; the
                # non-streaming path preserves the raw span for diagnostics.
                self.current_tool_id = idx
                self._name_sent = True
                self._current_tool_ref = None
                if idx >= len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool.append("")
                    self.prev_tool_call_arr.append({"name": name, "arguments": "{}"})
                # Fall through so a same-tick close still advances the FSM.
                return self._maybe_advance_on_close(buffer, sep_idx, emit=False)
            self.current_tool_id = idx
            self._name_sent = True
            self._current_tool_ref = generate_tool_id()
            if idx >= len(self.streamed_args_for_tool):
                self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr.append({"name": name, "arguments": "{}"})
            return {
                "tool_calls": [
                    {
                        "index": idx,
                        "id": self._current_tool_ref,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }
                ]
            }

        # ---------- Phase 2: STREAMING_ARGS ----------
        return self._maybe_advance_on_close(buffer, sep_idx, emit=True)

    def _maybe_advance_on_close(
        self, buffer: str, sep_idx: int, emit: bool
    ) -> dict[str, Any] | None:
        """Stream the current call's args and, on close, advance the FSM.

        ``emit`` is False for a suppressed (off-list) call — args are not
        emitted but the close still transitions the FSM back to SEEKING_NAME
        so the NEXT opener is processed as a fresh indexed call.
        """
        idx = self.current_tool_id
        if sep_idx == -1:
            return None
        args_tail = buffer[sep_idx + len(self.tool_sep_token) :]
        end_idx = args_tail.find(self.tool_call_end_token)
        closed = end_idx != -1
        if closed:
            args_tail = args_tail[:end_idx]

        result: dict[str, Any] | None = None
        if emit:
            snapshot = self._args_snapshot(args_tail, closed)
            if snapshot is not None:
                already = (
                    self.streamed_args_for_tool[idx]
                    if idx < len(self.streamed_args_for_tool)
                    else ""
                )
                if snapshot.startswith(already):
                    diff = snapshot[len(already) :]
                elif closed:
                    # Snapshot no longer extends what we streamed (e.g. a
                    # re-parse flipped JSON→pair mode). On close, reconcile
                    # by emitting the full final document as the diff.
                    diff = snapshot
                else:
                    diff = ""
                if diff:
                    if idx < len(self.streamed_args_for_tool):
                        self.streamed_args_for_tool[idx] = snapshot
                    if idx < len(self.prev_tool_call_arr):
                        self.prev_tool_call_arr[idx]["arguments"] = snapshot
                    result = {
                        "tool_calls": [{"index": idx, "function": {"arguments": diff}}]
                    }

        if closed:
            # Transition back to SEEKING_NAME so the next opener (if any)
            # starts a fresh indexed call on the next tick.
            self.current_tool_id = idx + 1
            self._name_sent = False
            self._current_tool_ref = None
        return result

    def _args_snapshot(self, args_tail: str, closed: bool) -> str | None:
        """Return the JSON args snapshot to have streamed SO FAR.

        While the call is open, the snapshot is a growing valid-JSON PREFIX
        with the trailing ``}`` withheld (vLLM's withhold-close principle) so
        an OpenAI-compatible client can concatenate deltas into ever-valid
        JSON. When ``closed`` is True the full args document (including the
        closing ``}``) is returned.

        Two arg shapes:
          * JSON body — stream the well-formed JSON prefix directly.
          * XML pairs — rebuild the args dict from CLOSED pairs and serialize;
            withhold the trailing ``}`` while open.
        """
        stripped = args_tail.strip()

        # --- JSON body shape ---
        if stripped.startswith("{"):
            if closed:
                # Emit the RAW wire JSON verbatim (validated by raw_decode)
                # rather than a re-serialized ``json.dumps(parsed)`` — the
                # two differ for unicode escapes (``\uXXXX`` vs the decoded
                # char) and whitespace, which would make the closed snapshot
                # non-monotonic vs the raw prefixes streamed while open and
                # corrupt the diff. Consume only the well-formed JSON prefix
                # so trailing structural residue (a stray ``<arg_value>``
                # opener, whitespace) is dropped.
                try:
                    _parsed, end = self._json_decoder.raw_decode(stripped)
                    if isinstance(_parsed, dict):
                        return stripped[:end]
                except (json.JSONDecodeError, ValueError):
                    return None
                return None
            # Open: emit the longest valid raw-JSON prefix with ``}`` withheld.
            return self._partial_json_prefix(stripped)

        # --- XML-pair shape (or empty tail) ---
        args: dict[str, Any] = {}
        for m in self._arg_pair_re.finditer(args_tail):
            key = m.group("key").strip()
            if not key:
                continue
            args[key] = _deserialize_arg_value(m.group("val"))
        if not args and not closed:
            # Nothing complete to emit yet.
            return None
        full = json.dumps(args, ensure_ascii=False)
        if closed:
            return full
        # Withhold the trailing ``}`` so the next pair (or the close) can
        # append cleanly. ``full`` is at least ``"{}"``; drop the last char.
        return full[:-1]

    def _partial_json_prefix(self, text: str) -> str | None:
        """Longest valid-JSON prefix of an in-flight object with ``}`` held.

        Emits the RAW wire text verbatim (the model emits valid JSON) so the
        prefixes streamed while open match the raw document emitted on close
        — re-serializing via ``json.dumps`` would diverge on unicode escapes
        and whitespace and corrupt the monotonic diff.

        If the object already decodes whole (the common single-delta case),
        emit the raw decoded span minus the trailing ``}``. Otherwise
        reconstruct a raw-faithful prefix from the complete members plus a
        partial string value in flight.

        Returns ``None`` when nothing new is safely emittable yet.
        """
        try:
            parsed, end = self._json_decoder.raw_decode(text)
            if isinstance(parsed, dict):
                raw = text[:end]
                # Emit raw up to (not including) the final ``}`` so the closed
                # snapshot — which is the same ``raw`` including ``}`` — is a
                # clean superstring. No rstrip: keep byte-for-byte alignment.
                close = raw.rfind("}")
                if close != -1:
                    return raw[:close]
        except (json.JSONDecodeError, ValueError):
            pass
        return self._decode_json_partial(text)

    def _decode_json_partial(self, text: str) -> str | None:
        """Return a RAW-faithful valid-JSON prefix of an incomplete object.

        Walks the raw wire text ``{ "k": v, "k2": v2, "k3": "partial…`` and
        returns ``text[:cut]`` where ``cut`` is the safe boundary — the end
        of the last COMPLETE ``"key": value`` member, or the safe end of a
        partial STRING value in flight. Emitting the raw span (not a
        ``json.dumps`` reconstruction) keeps every streamed prefix
        byte-aligned with the raw closed snapshot, so the diff stays
        monotonic across the partial→complete transition even when values
        contain unicode escapes.

        Returns ``None`` when no complete member (or partial string) has
        arrived yet.
        """
        s = text
        n = len(s)
        i = 0
        if i >= n or s[i] != "{":
            return None
        i += 1  # past ``{``
        cut = i  # after ``{`` (yields ``{`` alone if no member completes)
        first = True
        while True:
            while i < n and s[i] in " \t\r\n,":
                i += 1
            if i >= n or s[i] == "}":
                break
            # Decode the key string.
            try:
                key, key_end = self._json_decoder.raw_decode(s, i)
            except (json.JSONDecodeError, ValueError):
                break
            if not isinstance(key, str):
                break
            j = key_end
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j >= n or s[j] != ":":
                break
            j += 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j >= n:
                break
            # Try to decode a complete value → advance the safe cut.
            try:
                _value, value_end = self._json_decoder.raw_decode(s, j)
                cut = value_end
                i = value_end
                first = False
                continue
            except (json.JSONDecodeError, ValueError):
                pass
            # Value incomplete. Only a partial STRING can be emitted safely.
            if s[j] == '"':
                partial = self._partial_json_string(s[j:])
                if partial is not None:
                    # Splice the raw member prefix: everything up to the value
                    # start plus the safe partial-string body. ``partial``
                    # already carries the opening quote.
                    return s[:j] + partial
            break
        if cut <= 1:
            # Only ``{`` seen (or nothing complete) — hold until a member or
            # partial string arrives so we never emit a bare ``{``.
            return None
        _ = first
        return s[:cut]

    @staticmethod
    def _partial_json_string(text: str) -> str | None:
        """Return a valid-JSON-string PREFIX for an in-flight string value.

        ``text`` starts at the opening ``"`` of a JSON string value taken
        VERBATIM from the wire — its body is ALREADY JSON-escaped source
        (``\\n``, ``\\"``, ``\\uXXXX`` are literal two/six-char sequences on
        the wire, not decoded). We must NOT re-escape it; we emit the raw
        body unchanged (opening quote + safe body, NO closing quote) so the
        consumer's next chunk extends the same string.

        The safe body is the longest prefix that does not end mid-escape.
        We scan escape sequences so a trailing ``\\`` (dangling escape) or a
        partial ``\\uXX`` (fewer than 4 hex digits) is held back until the
        rest of the escape arrives.
        """
        if not text or text[0] != '"':
            return None
        body = text[1:]
        safe_len = 0
        i = 0
        n = len(body)
        while i < n:
            ch = body[i]
            if ch == "\\":
                # An escape sequence needs its payload to be complete before
                # the position past it is a safe cut point.
                if i + 1 >= n:
                    break  # dangling backslash — hold from here
                esc = body[i + 1]
                if esc == "u":
                    if i + 6 > n:
                        break  # partial \uXXXX — hold from here
                    i += 6
                else:
                    i += 2
                safe_len = i
            else:
                i += 1
                safe_len = i
        if safe_len == 0:
            return '"'
        return '"' + body[:safe_len]

    def has_pending_tool_call(self, text: str) -> bool:
        """Override — Hy3 opener/closer are the pinned fixed strings.

        Pending iff the LAST opener has no ``<end_of_tool_call>`` after it. A
        completed call earlier in ``text`` does not leave the parser pending
        forever.
        """
        opener = text.rfind(self.tool_call_start_token)
        if opener != -1 and self.tool_call_end_token not in text[opener:]:
            return True
        return self.has_text_format_tool_call(text)
