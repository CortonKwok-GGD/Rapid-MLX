# SPDX-License-Identifier: Apache-2.0
"""Always-on unit tests for the output-coherence primitives (#1247).

These are pure — no server, no MLX — so they run on every PR on a GitHub-hosted
runner and lock in the *gate logic itself*: that known garbage is flagged, that
real prose is not, and that a coherent-but-wrong answer fails its golden case.
The serve-path half (``evals/coherence_gate.py``) runs on the Apple-Silicon
release gauntlet where a real model is served.

Acceptance for #1247 is "test the gate": the exact class that shipped in #1234
(garbage output) and a coherent-but-wrong regression must both be rejected here.
"""

from __future__ import annotations

import pytest

from vllm_mlx.coherence import GOLDEN, GoldenCase, evaluate_case, looks_like_garbage

# ── real, coherent outputs that must NOT be flagged as garbage ──────────────
COHERENT = [
    "The capital of Japan is Tokyo.",
    "391",
    "Tokyo",
    "banana",
    # Short legitimate answers — regression guard for the dogfood false
    # positive where "7" tripped char-dominance (1/1 > 0.5). A valid single-
    # token answer must never be judged garbage.
    "7",
    "42",
    "9",
    "Yes",
    "No",
    "OK",
    "The ocean is a vast body of saltwater that covers most of Earth's "
    "surface. It teems with life, from microscopic plankton to enormous whales.",
    "A CPU executes instructions: it fetches, decodes, and runs the operations "
    "that make up a program, coordinating the rest of the computer.",
    "Yes — pedestrians are visible walking along the sidewalk on the left.",
]

# ── shipped / plausible garbage that MUST be flagged ────────────────────────
GARBAGE = [
    "",
    "   ",
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",  # prefix-cache poison (#gotchas)
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # doubled-norm degenerate single token
    "。。。。。。。。。。。。。。。。。。。。。。",  # CJK fill
    "the the the the the the the the the the the the the",  # bigram loop
    "cat cat cat cat cat cat cat cat cat cat cat cat",  # word loop
    "Sure! " + "!" * 40,  # long single-char run embedded in otherwise-ok text
]


@pytest.mark.parametrize("text", COHERENT)
def test_coherent_text_not_flagged(text: str) -> None:
    is_garbage, reason = looks_like_garbage(text)
    assert not is_garbage, f"false positive on {text[:50]!r}: {reason}"


@pytest.mark.parametrize("text", GARBAGE)
def test_garbage_text_flagged(text: str) -> None:
    is_garbage, _ = looks_like_garbage(text)
    assert is_garbage, f"missed garbage: {text[:50]!r}"


def test_evaluate_case_accepts_correct_answer() -> None:
    case = next(c for c in GOLDEN if c.id == "capital-japan")
    passed, _ = evaluate_case(case, "**Tokyo.**")
    assert passed


def test_evaluate_case_rejects_coherent_but_wrong() -> None:
    """A fluent, non-garbage, but WRONG answer must fail — this is the class no
    perf/import gate catches."""
    case = next(c for c in GOLDEN if c.id == "capital-japan")
    passed, reason = evaluate_case(case, "Tokyo is incorrect; Osaka is the capital.")
    assert not passed
    assert "exact match" in reason


def test_evaluate_case_rejects_garbage_for_golden_prompt() -> None:
    """The exact #1234 failure: the model emits garbage for a golden prompt."""
    case = next(c for c in GOLDEN if c.id == "arithmetic")
    passed, reason = evaluate_case(case, "!!!!!!!!!!!!!!!!!!!!")
    assert not passed
    assert "garbage" in reason


def test_evaluate_case_arithmetic_wrong_number_fails() -> None:
    case = next(c for c in GOLDEN if c.id == "arithmetic")
    assert evaluate_case(case, "391.")[0]
    assert not evaluate_case(case, "1391")[0]


def test_evaluate_case_days_rejects_number_containing_answer() -> None:
    case = next(c for c in GOLDEN if c.id == "days-in-week")
    assert not evaluate_case(case, "17")[0]


def test_no_think_leak_case_rejects_raw_reasoning_tag() -> None:
    case = next(c for c in GOLDEN if c.id == "no-think-leak")
    # Correct answer but raw reasoning markers leaked into the visible output.
    leaked = "<think>France's capital is Paris</think> Paris"
    passed, reason = evaluate_case(case, leaked)
    assert not passed
    assert "think" in reason.lower() or "reasoning" in reason.lower()
    # Clean answer passes.
    assert evaluate_case(case, "Paris")[0]


def test_not_garbage_case_is_open_ended() -> None:
    case = next(c for c in GOLDEN if c.id == "open-ocean")
    assert case.kind == "not_garbage"
    # Any coherent prose passes; degenerate output fails.
    assert evaluate_case(case, "The ocean is deep and full of life. It is blue.")[0]
    assert not evaluate_case(case, "ocean ocean ocean ocean ocean ocean ocean ocean")[0]


def test_golden_set_is_wellformed() -> None:
    assert len(GOLDEN) >= 5
    valid_kinds = {"exact", "not_garbage", "no_think_leak"}
    ids = set()
    for c in GOLDEN:
        assert isinstance(c, GoldenCase)
        assert c.kind in valid_kinds, f"{c.id}: bad kind {c.kind!r}"
        assert c.id not in ids, f"duplicate case id {c.id!r}"
        ids.add(c.id)
        if c.kind in {"exact", "no_think_leak"}:
            assert c.expect, f"{c.id}: {c.kind} needs a non-empty expect"
        assert c.max_tokens > 0
