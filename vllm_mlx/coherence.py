# SPDX-License-Identifier: Apache-2.0
"""Output-coherence primitives — the reusable core of the release coherence gate.

Motivation (#1247). Qwen3.6-35B-A3B and Qwen3.5-35B-A3B-8bit shipped producing
**pure garbage from the first token** (a doubled RMSNorm scale from an mlx-lm
``+1.0`` norm-shift misfire; fixed in #1234). The garbage passed *every*
automated gate — 278 delta tests, lint, install/import smoke, perf thresholds —
because **no gate ever generates a token and checks whether the output is
coherent**. This module supplies the two pieces that close that hole:

  * :func:`looks_like_garbage` — a cheap, deterministic detector for the two
    collapse classes we have actually shipped (``!!!!!!`` prefix-cache poison
    and doubled-norm token soup). Conservative by design: it only fires on
    unambiguous degeneracy, so coherent prose never trips it and the precise
    correctness work is left to the golden predicates below.
  * :data:`GOLDEN` + :func:`evaluate_case` — fixed prompts with *checkable*
    properties (``capital of Japan`` → ``Tokyo``, ``17 × 23`` → ``391``), so a
    coherent-but-wrong regression fails deterministically at ``temperature=0``.

Everything here is pure (no network, no MLX, no server) so it can be unit-tested
in ordinary CI on a GitHub-hosted runner. The serve-path runner that feeds real
generations through these predicates lives in ``evals/coherence_gate.py`` and
runs on the Apple-Silicon release gauntlet (``scripts/release_check_m3.sh``).
The detector is also intended for reuse by the telemetry garbage-rate alert
(#1250).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

__all__ = ["GoldenCase", "GOLDEN", "looks_like_garbage", "evaluate_case"]

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Literal reasoning-channel markers that must never survive into the visible
# assistant message (the OutputRouter strips them; a routing regression leaks
# them). Kept lowercase — callers compare against a lowercased copy.
_THINK_MARKERS = ("<think>", "</think>", "<reasoning>", "</reasoning>")


def _max_char_run(s: str) -> int:
    """Length of the longest run of a single non-space character."""
    best = run = 0
    prev = ""
    for ch in s:
        if ch == prev and not ch.isspace():
            run += 1
        else:
            run = 1
            prev = ch
        if run > best:
            best = run
    return best


def looks_like_garbage(text: str, *, min_words: int = 10) -> tuple[bool, str]:
    """Return ``(is_garbage, reason)`` for a completion string.

    Conservative: fires only on unambiguous degeneracy so real prose passes.
    Detects the classes we have actually shipped or seen:

    * empty / whitespace-only output;
    * punctuation/symbol-only output (``!!!!!``, CJK fill) at any length;
    * a single character dominating a non-trivial output (``aaaaa``);
    * a very long single-character run embedded in otherwise-mixed text;
    * a single word repeated (``ocean ocean ocean ocean``);
    * a tiny vocabulary over a long output (looping token soup);
    * one bigram dominating a long output (``the the the …``).

    Short legitimate answers (``"7"``, ``"42"``, ``"391"``, ``"Tokyo"``) are
    never flagged: the character-repetition heuristics only apply above a small
    length floor, and a valid answer always carries at least one word
    character. ``min_words`` guards the vocabulary/bigram checks so a short
    answer is never judged a loop.
    """
    s = (text or "").strip()
    if not s:
        return True, "empty"

    non_space = [c for c in s if not c.isspace()]
    if not non_space:
        return True, "whitespace-only"

    words = _WORD_RE.findall(s.lower())

    # (a) no word characters at all -> pure punctuation/symbol collapse
    # ("!!!!!", "。。。。", "?????"). Fires at any length; a legitimate answer to
    # any prompt carries at least one alphanumeric word character.
    if not words:
        return True, "no word characters (punctuation/symbol-only)"

    # (b) a single word repeated -> "ocean ocean ocean ocean". Unambiguous at
    # any length >= 4 (no legitimate answer is one word repeated 4+ times).
    if len(words) >= 4 and len(set(words)) == 1:
        return True, f"single word {words[0]!r} repeated {len(words)}x"

    # (c) character-repetition heuristics. Guarded by a length floor so short
    # legitimate answers ("7", "42", "OK") are never flagged: a doubled-norm /
    # cache-poison collapse that carries word characters ("aaaaa…") is long.
    if len(non_space) >= 5:
        top_char, top_n = Counter(non_space).most_common(1)[0]
        if top_n / len(non_space) > 0.5:
            return True, (
                f"char {top_char!r} is {top_n}/{len(non_space)} of non-space output"
            )
        if _max_char_run(s) >= 20:
            return True, "single-character run >= 20"

    if len(words) >= min_words:
        # (c) tiny vocabulary over a long output -> looping token soup
        uniq_ratio = len(set(words)) / len(words)
        if uniq_ratio < 0.20:
            return True, (
                f"distinct-word ratio {uniq_ratio:.2f} < 0.20 over {len(words)} words"
            )

        # (d) one bigram dominates -> "the the the the …"
        bigrams = list(zip(words, words[1:]))
        if bigrams:
            _, bn = Counter(bigrams).most_common(1)[0]
            if bn / len(bigrams) > 0.30:
                return True, f"top bigram is {bn}/{len(bigrams)} of the output"

    return False, "ok"


@dataclass(frozen=True)
class GoldenCase:
    """A fixed prompt plus a checkable predicate over its completion.

    ``kind`` selects the predicate applied by :func:`evaluate_case`:

    * ``exact``         — normalized completion exactly matches one of ``expect``
    * ``two_sentence``  — non-degenerate response with two sentences
    * ``no_think_leak`` — exact match AND no raw reasoning tag
    """

    id: str
    prompt: str
    kind: str
    expect: tuple[str, ...] = field(default_factory=tuple)
    max_tokens: int = 64


# Deterministic anchors a garbage / doubled-norm model cannot produce, and that
# a coherent-but-wrong regression also fails. Kept small + fast: on the starter
# alias (qwen3.5-4b-4bit) the whole set generates in well under a minute.
GOLDEN: tuple[GoldenCase, ...] = (
    GoldenCase(
        "capital-japan",
        "What is the capital of Japan? Answer in one word.",
        "exact",
        ("Tokyo",),
        max_tokens=32,
    ),
    GoldenCase(
        "arithmetic",
        "What is 17 multiplied by 23? Reply with just the number.",
        "exact",
        ("391",),
        max_tokens=32,
    ),
    GoldenCase(
        "sky-color",
        "What color is a clear daytime sky? Answer in one word.",
        "exact",
        ("blue",),
        max_tokens=32,
    ),
    GoldenCase(
        "days-in-week",
        "How many days are in a week? Reply with just the number.",
        "exact",
        ("7", "seven"),
        max_tokens=32,
    ),
    GoldenCase(
        # Instruction-following + coherence: a garbage / doubled-norm model
        # cannot echo a distinctive token. "banana" is 6 chars and vanishingly
        # unlikely to appear by chance, so this is far more discriminating than
        # a case-insensitive single-letter check.
        "echo-word",
        "Repeat exactly this word back to me, nothing else: banana",
        "exact",
        ("banana",),
        max_tokens=32,
    ),
    GoldenCase(
        "open-ocean",
        "Write a short two-sentence description of the ocean.",
        "two_sentence",
        max_tokens=200,
    ),
    GoldenCase(
        "open-cpu",
        "In two sentences, explain what a computer CPU does.",
        "two_sentence",
        max_tokens=200,
    ),
    GoldenCase(
        "no-think-leak",
        "What is the capital of France? Answer in one word.",
        "no_think_leak",
        ("Paris",),
        max_tokens=64,
    ),
)


_ANSWER_WRAPPER = " \t\r\n`*_~\"'“”‘’.,!?;:。！？；："


def _normalize_exact_answer(text: str) -> str:
    """Normalize harmless presentation around a requested one-token answer."""
    return " ".join(text.casefold().strip(_ANSWER_WRAPPER).split())


def _matches_exact(text: str, expected: tuple[str, ...]) -> bool:
    answer = _normalize_exact_answer(text)
    return any(answer == _normalize_exact_answer(item) for item in expected)


def evaluate_case(case: GoldenCase, text: str) -> tuple[bool, str]:
    """Apply ``case``'s predicate to a completion ``text``.

    Returns ``(passed, reason)``. Every case is first screened by
    :func:`looks_like_garbage` — a golden answer buried in ``!!!!`` still fails
    — and then the kind-specific predicate is applied.
    """
    garbage, why = looks_like_garbage(text)
    if garbage:
        return False, f"garbage output ({why})"

    if case.kind == "two_sentence":
        words = _WORD_RE.findall(text)
        sentence_ends = re.findall(r"[.!?。！？](?:\s|$)", text)
        if len(words) < 8:
            return False, f"too short ({len(words)} words; expected at least 8)"
        if len(sentence_ends) < 2:
            return False, "expected at least two complete sentences"
        return True, "coherent two-sentence response"

    if case.kind == "exact":
        if _matches_exact(text, case.expect):
            return True, f"exactly matches {case.expect!r}"
        return False, f"not an exact match for {case.expect!r}"

    if case.kind == "no_think_leak":
        low = text.lower()
        leaked = [m for m in _THINK_MARKERS if m in low]
        if leaked:
            return False, f"leaked reasoning marker(s) {leaked!r} into visible output"
        if _matches_exact(text, case.expect):
            return True, f"exactly matches {case.expect!r}, no think-leak"
        return False, f"not an exact match for {case.expect!r}"

    return False, f"unknown case kind {case.kind!r}"
