# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Tencent Hunyuan 3 (Hy3) models.

Hy3 emits reasoning content wrapped in ``<think:opensource>…</think:opensource>``
tags instead of the plain ``<think>…</think>`` shape every other thinking
family uses. The ``:opensource`` suffix marks the chat template's
"opensource" reasoning-mode variant; future model revisions may drop the
suffix or swap it for another label (``:v1``, ``:internal``, …), so we
match with ``(?::[\\w-]+)?`` to future-proof.

Implementation strategy
=======================

The base ``BaseThinkingReasoningParser`` state machine depends on exact
string containment (``self.start_token in text``, ``text.find(self.end_token)``).
Rather than duplicate the ~1500-line multi-block streaming state machine
with a regex-based variant, we subclass ``Qwen3ReasoningParser`` and
normalize the input at every public entry point:

  * ``<think:opensource>`` → ``<think>``
  * ``</think:opensource>`` → ``</think>``

The normalized text has identical structure to what a plain-tag stream
would produce, so all of Qwen3's Case 1/2/3/4 + streaming multi-block +
SSE-boundary withhold logic applies verbatim.

SSE-boundary partial-tag withhold. The ``:opensource`` suffix creates
partial-tag prefixes qwen3's own ``_held_partial_tag_len`` doesn't know
about (its withhold logic recognises ``<`` through ``<think>``, not
``<think:`` through ``<think:opensource>``). ``_hy3_straddle_suffix_len``
adds withhold coverage for the ``:[label]`` region so
``previous_norm + delta_norm == current_norm`` holds by construction on
every tick — including the one where the suffixed tag spans the SSE
chunk boundary. See PR #1070 codex round-1 finding #2.
"""

from __future__ import annotations

import re

from .base import DeltaMessage
from .qwen3_parser import Qwen3ReasoningParser

# Suffix-tolerant matcher — captures ``:opensource``, ``:v1``, etc. so
# future model revisions keep parsing without a code change.
_HY3_OPEN_TAG_RE = re.compile(r"<think(?::[\w-]+)?>")
_HY3_CLOSE_TAG_RE = re.compile(r"</think(?::[\w-]+)?>")

# Straddle-boundary detector: a trailing text run that is a non-empty PREFIX
# of a (possibly suffixed) Hy3 think tag and MAY complete on the next delta.
#
# It must match EVERY prefix as the tag builds up char-by-char — ``<``, ``<t``,
# … ``<think``, ``<think:``, ``<think:opensource`` — not only the full
# ``<think`` root (codex R7 BLOCKING #3). If only the full root were held, the
# withheld span would grow NON-monotonically (``see <thin`` holds nothing, then
# ``see <think`` suddenly holds 6 bytes), so the visible span retreats and the
# base machine — which already emitted the earlier bytes — gets an inconsistent
# boundary and duplicates/corrupts output (``see `` re-emitted as ``k>see``).
# Matching all prefixes keeps the hold monotonic: once a ``<`` that could start
# a think tag appears it stays held until the tag COMPLETES (real tag) or
# FALSIFIES into ordinary content (``<thinking``), and either way the held
# bytes are delivered on the tick they resolve — nothing is dropped.
#
# ``<`` / ``</`` alone are ambiguous roots shared by both open and close; the
# open matcher covers ``<`` and ``<t…<think[:label]``, the close matcher covers
# ``</`` and ``</t…</think[:label]``. A run like ``</`` matches the close
# matcher (longer, tried first) so both are reserved.
_HY3_OPEN_STRADDLE_RE = re.compile(r"<(?:t(?:h(?:i(?:n(?:k(?::[\w-]*)?)?)?)?)?)?$")
_HY3_CLOSE_STRADDLE_RE = re.compile(r"</(?:t(?:h(?:i(?:n(?:k(?::[\w-]*)?)?)?)?)?)?$")


def _normalize_hy3_tags(text: str) -> str:
    """Rewrite Hy3's suffixed think tags to the plain Qwen3 shape.

    ``<think:opensource>`` → ``<think>``
    ``</think:opensource>`` → ``</think>``

    Non-matching input is returned unchanged (empty-string safe).

    Parser policy (codex R4 NIT #5). This is an unconditional substitution over
    the whole reasoning stream, so a LITERAL ``<think:opensource>`` string that
    a model happened to emit as visible content would also be rewritten. That
    is the accepted, intentional contract: for an Hy3 model the think-tag
    namespace (``<think(:LABEL)?>``) is reserved for reasoning delimiters — it
    is exactly how every other thinking family's reasoning parser treats
    ``<think>`` (the plain-tag parsers rewrite/strip any literal ``<think>`` in
    content too). The alternative — a full regex-driven state machine that only
    transforms tags at genuine state transitions — would duplicate the
    ~1500-line qwen3 streaming machine for a payload no real Hy3 checkpoint
    produces, so we keep the simple normalization and document the tradeoff.
    """
    if not text:
        return text
    text = _HY3_OPEN_TAG_RE.sub("<think>", text)
    text = _HY3_CLOSE_TAG_RE.sub("</think>", text)
    return text


def _hy3_straddle_suffix_len(text: str) -> int:
    """Length of the trailing suffix that is an in-progress Hy3 tag.

    Returns 0 when ``text`` doesn't end mid-tag. The base qwen3 state
    machine handles the plain ``<think>`` / ``</think>`` partial-tag
    withhold itself (via ``_held_partial_tag_len``); this helper covers
    ONLY the additional ``:[label]`` region that qwen3 has no knowledge
    of. Withholding those bytes on the current tick preserves the
    invariant ``previous_norm + delta_norm == current_norm`` for the
    NEXT tick when the tag completes.

    Codex round-1 BLOCKING fix (PR #1070 finding #2).
    """
    if not text:
        return 0
    m = _HY3_CLOSE_STRADDLE_RE.search(text)
    if m is not None:
        return len(text) - m.start()
    m = _HY3_OPEN_STRADDLE_RE.search(text)
    if m is not None:
        return len(text) - m.start()
    return 0


class Hy3ReasoningParser(Qwen3ReasoningParser):
    """Reasoning parser for Hy3 / Hunyuan 3.

    Suffix-tolerant wrapper over ``Qwen3ReasoningParser``. Normalizes
    ``<think:xxx>`` / ``</think:xxx>`` to the plain form before
    delegating so the entire Qwen3 state machine (Case 1/2/3/4,
    multi-block streaming, SSE-boundary withhold, tool-call promotion,
    D-STOP-THINK finalize suppression) works verbatim.

    Streaming ``previous_text`` / ``current_text`` / ``delta_text``
    are normalized consistently — after normalization the invariant
    ``current_norm = previous_norm + delta_norm`` holds because the
    substitution is a simple length-changing but position-preserving
    rewrite of full tags (and both boundary texts see the SAME rewrite
    when the tag was already complete in ``previous_text``).
    """

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        return super().extract_reasoning(
            _normalize_hy3_tags(model_output),
            enable_thinking=enable_thinking,
        )

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        # Codex round-1 BLOCKING fix (PR #1070 finding #2): withhold
        # trailing bytes that could be an in-progress Hy3 suffixed tag
        # (``<think:opensou`` waiting on ``rce>``) BEFORE normalisation.
        # Without this, ``current_norm`` collapses ``<think:opensource>``
        # to ``<think>`` on the tick the closer arrives while
        # ``previous_norm`` still ends with ``<think:opensou`` — the
        # invariant ``previous_norm + delta_norm == current_norm`` breaks
        # and the base multi-block router routes bytes to the wrong phase.
        # Withhold the trailing bytes of BOTH boundary texts that could still
        # grow into a Hy3 suffixed tag, working on the RAW text so a byte held
        # on tick N is delivered on tick N+1 whether the tag COMPLETES (a real
        # ``<think:opensource>``) or FALSIFIES into ordinary content (``<think``
        # → ``<thinking``). Both cases are covered because the visible span is
        # ``text[: len - hold]`` and any held tail that no longer matches
        # ``_hy3_straddle_suffix_len`` next tick simply becomes part of that
        # tick's visible span — nothing is dropped (codex R7 BLOCKING #3: the
        # old ``startswith``-based delta recompute corrupted output when a held
        # ``<think`` prefix falsified after a completed think block).
        prev_visible = previous_text[
            : len(previous_text) - _hy3_straddle_suffix_len(previous_text)
        ]
        curr_visible = current_text[
            : len(current_text) - _hy3_straddle_suffix_len(current_text)
        ]
        # Derive the RAW newly-visible span, then normalise ONLY that span. The
        # raw delta is unambiguous (curr_visible always extends prev_visible —
        # both are prefixes of the same growing accumulated text truncated at a
        # non-tag-interior point), so we never double-emit or drop.
        if curr_visible.startswith(prev_visible):
            raw_delta = curr_visible[len(prev_visible) :]
        else:
            # Should not happen (both are prefixes of the same text), but guard
            # against a pathological shrink by falling back to the whole raw
            # delta so no bytes are lost.
            raw_delta = delta_text
        if not raw_delta:
            return None
        previous_norm = _normalize_hy3_tags(prev_visible)
        current_norm = _normalize_hy3_tags(curr_visible)
        delta_norm = _normalize_hy3_tags(raw_delta)
        # Guard the base machine's own invariant: only feed it when the
        # normalised boundaries still concatenate (they do whenever the hold
        # points fall outside a tag the normaliser rewrites). Otherwise defer —
        # the withheld bytes surface next tick.
        if previous_norm + delta_norm != current_norm:
            delta_norm = (
                current_norm[len(previous_norm) :]
                if (current_norm.startswith(previous_norm))
                else delta_norm
            )
        if not delta_norm:
            return None
        return super().extract_reasoning_streaming(
            previous_norm, current_norm, delta_norm
        )

    def finalize_streaming(
        self,
        accumulated_text: str,
        *,
        matched_stop: str | None = None,
        prompt_thinking_active: bool = False,
        finish_reason: str | None = None,
    ) -> DeltaMessage | None:
        return super().finalize_streaming(
            _normalize_hy3_tags(accumulated_text),
            matched_stop=matched_stop,
            prompt_thinking_active=prompt_thinking_active,
            finish_reason=finish_reason,
        )

    def is_open_in_think(self, accumulated_text: str) -> bool:
        return super().is_open_in_think(_normalize_hy3_tags(accumulated_text))
