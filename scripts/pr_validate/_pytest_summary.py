# SPDX-License-Identifier: Apache-2.0
"""Extract failing/passing node ids from a pytest run.

Two steps (``full_unit`` — quarantine partition — and ``flake_tracking``
— rerun classification) need node ids out of a pytest run. There are two
sources, in order of preference:

  * ``report_log_node_ids`` — reads the STRUCTURED log written by the
    ``_nodeid_reporter`` plugin (``report.nodeid``, exact + unambiguous).
    This is the source of truth; use it whenever the plugin ran.
  * ``summary_node_ids`` — a FALLBACK that scrapes pytest's terminal
    short-summary when the plugin didn't run. It is irreducibly ambiguous
    for exotic parametrized ids (see ``_strip_message``); it fails SAFE
    (an ambiguous id won't match a quarantine entry → the failure blocks).

Keeping both here means the two steps can't drift on how a failure set is
derived (a divergence would silently mis-gate).
"""

from __future__ import annotations

import re
from pathlib import Path

from ._nodeid_reporter import SESSION_LABEL, unescape_field

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


def report_log_node_ids(path: Path, *labels: str) -> list[str]:
    """Node ids the ``_nodeid_reporter`` plugin recorded under ``labels``.

    The log is append-only ``<LABEL>\\t<nodeid>`` lines (see
    ``_nodeid_reporter``). Returns the ids for the requested labels
    (default ``FAILED``), first-seen order, de-duplicated. Unlike
    ``summary_node_ids`` there is NO parsing ambiguity — ``report.nodeid``
    is pytest's own canonical id. The caller is responsible for checking
    the file exists (a missing log means the plugin didn't run → fall back
    to ``summary_node_ids``)."""
    wanted = set(labels or ("FAILED",))
    out: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        label, tab, raw = line.partition("\t")
        if not (tab and label in wanted and raw):
            continue
        # Decode the escaped value back to the exact node id. Dedup on the
        # DECODED id so an escaped and a raw spelling of the same id can't
        # both slip through (codex #1222 r23).
        nodeid = unescape_field(raw)
        if nodeid and nodeid not in seen:
            seen.add(nodeid)
            out.append(nodeid)
    return out


def raw_session_records(path: Path) -> list[str]:
    """Every ``SESSIONFINISH`` value line in the log, decoded but WITHOUT the
    node-id dedup ``report_log_node_ids`` applies.

    Two identical completion records must count as two so ``session_completed``
    can reject the duplicate (an unexpected xdist/tamper shape) — deduping
    would collapse them to one and slip a second record past the "exactly
    one" rule (codex #1222 r16)."""
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec_label, tab, raw = line.partition("\t")
        if tab and rec_label == SESSION_LABEL and raw:
            out.append(unescape_field(raw))
    return out


def session_completed(path: Path) -> bool:
    """True iff the reporter's session-finish record proves pytest ran the
    WHOLE collected suite — the precondition for a sound quarantine
    downgrade (``full_unit``) and for trusting a rerun classification
    (``flake_tracking``) (codex #1222 r15/r23).

    Both truncation modes are caught STRUCTURALLY, replacing an old
    stdout-banner heuristic that false-positived when a test's own traceback
    contained ``"stopping after"`` / ``"Interrupted:"`` (r15 B2):

      * ABSENT record → ``pytest_sessionfinish`` never fired, i.e. the
        process died mid-suite (``os._exit`` / crash / SIGKILL) (r15 B1).
      * ``ran < collected`` → the session ended early (``-x`` / a hit
        ``--maxfail`` / ``--stepwise`` / a mid-call ``pytest.exit``), so
        tests after the stop never ran.

    Requires EXACTLY one well-formed record (``<ran> <collected>``) with a
    positive collected count and ``ran >= collected``. Anything else —
    missing, malformed, duplicated — withholds trust, the safe direction.

    Kept here (not in a step) so ``full_unit`` and ``flake_tracking`` share
    ONE completeness definition and can't drift on it."""
    records = raw_session_records(path)
    if len(records) != 1:
        return False
    try:
        ran_s, collected_s = records[0].split()
        ran, collected = int(ran_s), int(collected_s)
    except (ValueError, TypeError):
        return False
    return collected > 0 and ran >= collected


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

    Ambiguity is resolved by failing SAFE, never by guessing. Two cases:

      * ``r6``: a node id whose EXPLICIT param id holds an *unmatched*
        bracket (``ids=["["]`` → ``test_x[[]``) leaves the depth counter
        unbalanced, so the message is absorbed into the returned id.
      * ``r9``: a param id that itself contains ``"] - ["`` (e.g.
        ``test_x[a] - [b]``) makes the inner ``]`` close the depth
        prematurely, so the FIRST depth-0 ``" - "`` lands *inside* the
        param sequence. Splitting there would truncate to ``test_x[a]`` —
        which could wrongly match a shorter quarantine entry and DOWNGRADE
        a real failure. To prevent that, a candidate split is rejected when
        the message it would produce begins with ``"["`` (the tell-tale of
        a mid-param split, or a genuine bracket-leading message we equally
        can't disambiguate). We then return the whole line as the id.

    Both cases return a mangled id that won't match a normal quarantine
    entry, so the failure BLOCKS rather than being wrongly waived, and a
    mistargeted advisory re-run just yields an inconclusive result. The
    r4/r5 common cases are unaffected — their message begins with the
    exception text (``AssertionError: …``) or a bare word, not ``"["``.

    A structured report (junit-xml) was considered and rejected: its
    ``classname``/``name`` split can't be mapped back to a pytest node id
    unambiguously for class-based / deep-package tests, trading these rare
    safe failures for a common-path matching risk."""
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
            # A message beginning with "[" means we can't tell a genuine
            # bracket-leading message from a split taken in the MIDDLE of a
            # bracketed param sequence ("test_x[a] - [b]"). Fail safe:
            # return the whole line so an ambiguous id can't be wrongly
            # matched against a shorter quarantine entry (codex #1222 r9).
            if rest[i + 3 :].startswith("["):
                return rest.strip()
            return rest[:i].strip()
        i += 1
    return rest.strip()
