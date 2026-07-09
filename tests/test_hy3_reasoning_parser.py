# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for the Hy3 (Tencent Hunyuan 3) reasoning parser.

Hy3 emits ``<think:opensource>…</think:opensource>`` reasoning spans. The
parser normalizes the ``:opensource`` suffix to the plain ``<think>`` /
``</think>`` shape and delegates to the qwen3 parser, so every qwen3
semantic (Case 1/2/3/4, streaming multi-block, SSE-boundary withhold,
tool-call promotion, D-STOP-THINK finalize suppression) applies verbatim.
"""

from __future__ import annotations

import pytest

from vllm_mlx.reasoning import get_parser
from vllm_mlx.reasoning.hy3_parser import Hy3ReasoningParser, _normalize_hy3_tags


def test_parser_is_registered():
    """The parser must appear in the reasoning registry under both
    aliases so ``reasoning_parser="hy_v3"`` and ``reasoning_parser="hy3"``
    (CLI convenience) both resolve."""
    assert get_parser("hy_v3") is Hy3ReasoningParser
    assert get_parser("hy3") is Hy3ReasoningParser


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<think:opensource>a</think:opensource>", "<think>a</think>"),
        ("<think>a</think>", "<think>a</think>"),  # plain — unchanged
        # Mixed open+close suffixes still normalize.
        ("<think:v1>a</think:opensource>", "<think>a</think>"),
        # Non-tag text is unchanged.
        ("normal text with no tags", "normal text with no tags"),
        # Empty-string safe.
        ("", ""),
    ],
)
def test_normalize_hy3_tags(raw, expected):
    assert _normalize_hy3_tags(raw) == expected


def test_extract_reasoning_suffixed_tags():
    """The canonical Hy3 emission — ``<think:opensource>…</think:opensource>``
    — must split cleanly into reasoning and content."""
    parser = Hy3ReasoningParser()
    r, c = parser.extract_reasoning(
        "<think:opensource>Let me think about it.</think:opensource>The answer is 42."
    )
    assert r == "Let me think about it."
    assert c == "The answer is 42."


def test_extract_reasoning_plain_tags_still_work():
    """The parser MUST accept the plain ``<think>`` shape too so a future
    Hy3 revision that drops the suffix (or a mixed-checkpoint dogfood
    session) doesn't regress."""
    parser = Hy3ReasoningParser()
    r, c = parser.extract_reasoning("<think>reasoning</think>content")
    assert r == "reasoning"
    assert c == "content"


def test_extract_reasoning_implicit_close_only():
    """Case-2 (chat template injects ``<think>`` into the prompt) — only
    the close tag appears in the output. Suffixed variant must route the
    same way."""
    parser = Hy3ReasoningParser()
    r, c = parser.extract_reasoning(
        "reasoning here</think:opensource>final answer"
    )
    assert r == "reasoning here"
    assert c == "final answer"


def test_extract_reasoning_no_tags():
    """Case-4 with ``enable_thinking`` unset — no tags → pure content."""
    parser = Hy3ReasoningParser()
    r, c = parser.extract_reasoning("just a response")
    assert r is None
    assert c == "just a response"


def test_streaming_tag_atomic_deltas():
    """The common streaming case — every tag arrives whole in a single
    SSE delta. Reasoning bytes route to ``reasoning``, post-close bytes
    to ``content``."""
    parser = Hy3ReasoningParser()
    parser.reset_state()

    def step(prev: str, delta: str):
        cur = prev + delta
        return parser.extract_reasoning_streaming(prev, cur, delta)

    prev = ""
    # Opener token — nothing emitted (structural).
    m1 = step(prev, "<think:opensource>")
    prev = "<think:opensource>"
    # After normalization the base parser sees ``<think>`` as opener.
    assert m1 is None
    # Reasoning bytes flow to ``reasoning``.
    m2 = step(prev, "Let me ")
    prev += "Let me "
    assert m2 is not None
    assert m2.reasoning == "Let me "
    assert m2.content is None
    m3 = step(prev, "think.")
    prev += "think."
    assert m3.reasoning == "think."
    # Close token — structural.
    m4 = step(prev, "</think:opensource>")
    prev += "</think:opensource>"
    assert m4 is None or m4.content in (None, "")
    # Post-close content bytes.
    m5 = step(prev, "Paris")
    assert m5 is not None
    assert m5.content == "Paris"


def test_finalize_streaming_delegates_to_qwen3():
    """``finalize_streaming`` MUST inherit qwen3's D-STOP-THINK
    suppression semantics on truncation. Smoke-test that the delegation
    path doesn't crash on a suffixed accumulated buffer."""
    parser = Hy3ReasoningParser()
    parser.reset_state()
    # A ``<think:opensource>`` opener with no close — the base returns
    # None (no correction) by default; qwen3's override returns
    # reasoning on ``finish_reason="length"``.
    msg = parser.finalize_streaming(
        "<think:opensource>partial thought",
        finish_reason="length",
    )
    assert msg is not None
    assert msg.reasoning == "partial thought"


def test_is_open_in_think_recognises_suffixed_opener():
    """The finalize-on-truncation router calls ``is_open_in_think`` to
    decide whether to route the buffer as reasoning. The Hy3 parser
    MUST recognise a suffixed opener as such."""
    parser = Hy3ReasoningParser()
    assert parser.is_open_in_think("<think:opensource>partial") is True
    assert parser.is_open_in_think("no think here") is False
    assert (
        parser.is_open_in_think("<think:opensource>closed</think:opensource>tail")
        is False
    )
