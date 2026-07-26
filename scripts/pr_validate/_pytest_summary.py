# SPDX-License-Identifier: Apache-2.0
"""Shared parser for pytest's short-summary section.

Two steps (``full_unit`` — quarantine partition — and ``flake_tracking``
— rerun classification) need to pull node ids out of pytest's short test
summary. Keeping one parser here means the FAILED/ERROR extraction can't
drift between them (a divergence would silently mis-gate).
"""

from __future__ import annotations

import re

# pytest writes the section header as a full-width separator banner:
# ``==================== short test summary info ====================``.
# Anchoring on that *padded* banner — not a bare ``"short test summary"``
# substring — stops a test from injecting a forged summary block: a test
# that prints ``short test summary`` (or even a fully padded fake banner)
# in its own captured stdout can't be mistaken for the real section,
# because we take the LAST banner and captured output always renders in
# the FAILURES section *above* the genuine summary (codex #1222 r2).
#
# Scope of the guarantee (codex #1222 r3): this defeats the REALISTIC
# vectors — an accidental "short test summary" string, and captured
# stdout above the summary. It does NOT defend against an author who
# deliberately weaponizes their own test process (e.g. an ``atexit``
# handler that prints a forged banner *after* pytest's genuine summary).
# That is out of scope by design: the gate runs the candidate's code
# in-process, so a hostile author owning the interpreter has strictly
# easier sabotage routes (``@pytest.mark.skip``, deleting the test,
# xfail) and no in-process report source — junit-xml included, since it
# too can be rewritten from an ``atexit`` handler — is tamper-proof
# against them. We defend against mistakes and flakes, not self-sabotage.
_SUMMARY_BANNER = re.compile(r"^=+\s*short test summary info\s*=+$")


def summary_node_ids(text: str, *labels: str) -> list[str]:
    """Node ids pytest lists under the given short-summary labels.

    A summary line is ``<LABEL> <nodeid>[ - <message>]``; the node id is
    everything between the label and the ``" - "`` message separator. A
    parametrized id can itself contain ``" - "`` inside its ``[...]``
    (e.g. ``test_x[a - b]``), so we only treat a ``" - "`` that occurs
    *after* the final ``]`` as the separator — inside the bracket it's
    part of the id (codex #1222 r4).

    Only lines inside the genuine ``short test summary info`` banner block
    count — a ``FAILED`` token in a traceback (or a forged banner) above
    it is ignored. Defaults to the ``FAILED`` label when none is given.
    """
    wanted = labels or ("FAILED",)
    lines = (text or "").splitlines()

    # Find the LAST real banner. Anything before it — captured stdout,
    # tracebacks, a test-forged banner — is not the authoritative summary.
    start = None
    for i, line in enumerate(lines):
        if _SUMMARY_BANNER.match(line.strip()):
            start = i
    if start is None:
        return []

    out: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("="):
            break
        for label in wanted:
            if line.startswith(label + " "):
                node_id = _strip_message(line[len(label) + 1 :])
                if node_id:
                    out.append(node_id)
                break
    return out


def _strip_message(rest: str) -> str:
    """Return the node id from a ``<nodeid>[ - <message>]`` tail.

    The message separator is the FIRST ``" - "`` that falls outside the
    node id's parametrization brackets. Scanning left-to-right with a
    bracket-depth counter handles both directions of ambiguity: a
    ``" - "`` inside ``[...]`` is part of the id (``test_x[a - b]``), and
    a ``]`` inside the *message* can't be mistaken for the id's bracket
    (``test_x - AssertionError: [1]`` → ``test_x``) — the earlier
    ``rfind("]")`` approach got the latter wrong (codex #1222 r5).

    Known limitation (codex #1222 r6): a node id whose EXPLICIT param id
    contains an *unmatched* bracket (e.g. ``ids=["["]`` → ``test_x[[]``)
    leaves the depth counter unbalanced, so the message is absorbed into
    the returned id. This is irreducible in pytest's terminal output —
    ``nodeid - message`` is genuinely ambiguous when the id may hold
    ``" - "`` and unbalanced ``[]`` while the message may hold anything.
    It fails SAFE: a mangled id won't match a (normal) quarantine entry,
    so the failure BLOCKS rather than being wrongly downgraded, and a
    mistargeted advisory re-run just yields an inconclusive result. A
    structured report (junit-xml) was considered and rejected: its
    ``classname``/``name`` split can't be mapped back to a pytest node id
    unambiguously for class-based / deep-package tests, trading this rare
    safe failure for a common-path matching risk."""
    depth = 0
    i = 0
    n = len(rest)
    while i < n:
        c = rest[i]
        if c == "[":
            depth += 1
        elif c == "]":
            if depth > 0:
                depth -= 1
        elif depth == 0 and rest.startswith(" - ", i):
            return rest[:i].strip()
        i += 1
    return rest.strip()
