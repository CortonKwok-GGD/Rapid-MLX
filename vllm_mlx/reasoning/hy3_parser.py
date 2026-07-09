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

# Straddle-boundary detector: strict prefix of a suffixed close/open tag
# that MAY complete on the next delta. Codex round-5 BLOCKING #2
# (PR #1070) widened the pattern to include the ``:``-less prefix
# (``<think`` / ``</think``) — the qwen3 base withhold only reserves
# the plain form up to (but not including) the ``>``; without extra
# coverage a boundary split like ``"<think"`` then ``":opensource>"``
# would release ``<think`` from qwen3's hold on tick N and then leak
# ``:opensource>`` as plain content on tick N+1 (because the ``:``
# character isn't the ``>`` qwen3 was waiting for, so qwen3 falls
# through). Anchoring on the exact ``<think`` root plus an optional
# ``:LABEL`` suffix covers both the ``:``-less prefix AND the labelled
# variants uniformly.
_HY3_OPEN_STRADDLE_RE = re.compile(r"<think(?::[\w-]*)?$")
_HY3_CLOSE_STRADDLE_RE = re.compile(r"</think(?::[\w-]*)?$")


def _normalize_hy3_tags(text: str) -> str:
    """Rewrite Hy3's suffixed think tags to the plain Qwen3 shape.

    ``<think:opensource>`` → ``<think>``
    ``</think:opensource>`` → ``</think>``

    Non-matching input is returned unchanged (empty-string safe).
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
        prev_hold = _hy3_straddle_suffix_len(previous_text)
        curr_hold = _hy3_straddle_suffix_len(current_text)
        previous_norm = _normalize_hy3_tags(
            previous_text[: len(previous_text) - prev_hold]
        )
        current_norm = _normalize_hy3_tags(
            current_text[: len(current_text) - curr_hold]
        )
        # Recompute delta from the normalised strings so the invariant
        # ``previous_norm + delta_norm == current_norm`` holds by
        # construction. If the withhold ate the whole delta (all bytes
        # were partial-tag), emit nothing — the next tick will surface
        # them when the tag completes.
        if current_norm.startswith(previous_norm):
            delta_norm = current_norm[len(previous_norm) :]
        else:
            # Defensive fallback for a still-inconsistent boundary — e.g.
            # a normalisation that shrank the previous span more than the
            # withhold reserved. Fall through to plain-delta normalisation
            # so we don't crash; qwen3's own partial-tag withhold and the
            # multi-block router recover on the next tick.
            delta_norm = _normalize_hy3_tags(delta_text)
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
