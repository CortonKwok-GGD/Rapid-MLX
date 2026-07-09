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

Streaming caveat: a delta boundary that lands INSIDE a ``<think:xxx>``
suffix (e.g. delta ends with ``<think:`` and the next delta opens with
``opensource>``) will let ``<think:`` slip into ``reasoning_content``
briefly — the base parser recognises ``<think>`` as complete but treats
``<think:`` as an in-progress prefix of the plain tag it doesn't know
about. The bytes normalise correctly on the FOLLOWING delta because
``current_text`` sees the whole suffix. For the preview HY3 checkpoint
(released 2026-07-06) mlx-lm's SSE chunker emits tag-atomic deltas in
practice, so this is a theoretical concern only — pinned in tests but
not gate-blocking for the launch.
"""

from __future__ import annotations

import re

from .base import DeltaMessage
from .qwen3_parser import Qwen3ReasoningParser

# Suffix-tolerant matcher — captures ``:opensource``, ``:v1``, etc. so
# future model revisions keep parsing without a code change.
_HY3_OPEN_TAG_RE = re.compile(r"<think(?::[\w-]+)?>")
_HY3_CLOSE_TAG_RE = re.compile(r"</think(?::[\w-]+)?>")


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
        # Normalize current + previous. Recompute delta from the
        # normalized strings so ``previous_norm + delta_norm ==
        # current_norm`` (the invariant the base multi-block router
        # relies on). When ``current_text`` doesn't cleanly extend
        # ``previous_text`` under normalization — the tag boundary
        # straddled the SSE chunk boundary — fall back to normalizing
        # the delta directly. The base's partial-tag withhold handles
        # the residue on the next tick.
        current_norm = _normalize_hy3_tags(current_text)
        previous_norm = _normalize_hy3_tags(previous_text)
        if current_norm.startswith(previous_norm):
            delta_norm = current_norm[len(previous_norm) :]
        else:
            delta_norm = _normalize_hy3_tags(delta_text)
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
