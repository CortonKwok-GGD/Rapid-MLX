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

import json
import re
from pathlib import Path

from ._nodeid_reporter import RERUN_LABEL, SESSION_LABEL, unescape_field

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


def rerun_detected(path: Path) -> bool:
    """True iff the reporter logged any RERUN record — a gating pytest run
    reran a test, which it must NEVER do.

    Name-based ``-p no:<name>`` blocking is bypassable — a conftest can
    register pytest-rerunfailures under an arbitrary name (verified, codex
    #1222 r23) — so this OUTCOME-based signal is the name-independent
    backstop shared by EVERY gating pytest step (``full_unit``,
    ``targeted_tests``): a real failure could have been retried into a pass,
    so any RERUN in a gating run makes it untrustworthy (codex #1222 r24).
    Kept here so no gating step reimplements the check and drifts."""
    return bool(report_log_node_ids(path, RERUN_LABEL))


# The three ``str.splitlines()`` line boundaries that ``json.dumps`` leaves
# LITERAL under ``ensure_ascii=False``: it escapes every control char < U+0020
# (``\n``/``\r``/``\t``/``\v``/``\f``/``\x1c``-``\x1e``) but NOT U+0085 (NEL),
# U+2028 (LINE SEP), or U+2029 (PARA SEP), which are all ≥ U+0020 (codex r31).
_JSON_UNESCAPED_LINE_SEPARATORS = ("\x85", "\u2028", "\u2029")


def render_fence_safe(node_id: str) -> str:
    """JSON-encode a node id for safe interpolation into a Markdown ``` code
    fence in the scorecard.

    ``report.nodeid`` is not trustworthy display text — a hostile
    ``pytest_make_parametrize_id`` can embed a newline (and backticks) in it
    (codex #1222 r24). Interpolated raw into a fenced block, a newline could
    start a fence-closing line and let the rest of the id spoof scorecard
    Markdown. JSON escaping collapses the value to a single quoted line
    (newline → ``\\n``, tab → ``\\t``, backslash/quote escaped), so no
    embedded newline can begin a fence-closer; ``ensure_ascii=False`` keeps
    a legitimate non-ASCII id readable. A normal id is unchanged except for
    the surrounding quotes.

    ``json.dumps`` escapes only control chars < U+0020, so it would leave
    U+0085/U+2028/U+2029 LITERAL — yet ``str.splitlines()`` (and some Markdown
    renderers) treat those as line boundaries too, so a hostile id embedding
    one could still begin a fence-closer on a second physical line. Escape
    them explicitly as ``\\uXXXX`` (valid JSON, so ``json.loads`` still
    round-trips the id) to keep the single-line guarantee under splitlines
    (codex #1222 r31, mirroring the escape_field hardening in r28)."""
    out = json.dumps(node_id, ensure_ascii=False)
    for sep in _JSON_UNESCAPED_LINE_SEPARATORS:
        out = out.replace(sep, f"\\u{ord(sep):04x}")
    return out


# ``errors?`` — pytest pluralizes the error outcome ("1 error" vs "2 errors"),
# so an error-only run summary ("2 errors in 0.10s") must still match or it
# degrades to a bare "exit 2" (codex #1222 r29).
_SUMMARY_COUNTS_RE = re.compile(
    r"\b\d+ (passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b"
)


def last_summary_line(stdout: str) -> str:
    """pytest's final counts line, WITH or WITHOUT the ``====`` bars.

    In verbose mode the line is fenced (``==== 3 passed in 1s ====``); once the
    inherited ``-v`` is dropped (``-o addopts=`` neutralizes the repo addopts)
    pytest prints it bar-less (``3 passed in 1s``). A ``startswith("=")`` rule
    would miss the bar-less form and degrade the step summary to a bare
    "exit N" (codex #1222 r27/r28). Anchor on the ``<n> <outcome> … in <t>s``
    shape so both forms are recognized. Shared by ``targeted_tests`` and
    ``full_unit`` so the two can't drift. Returns "" when there is no counts
    line (the caller falls back to the exit code)."""
    for line in reversed((stdout or "").splitlines()):
        stripped = line.strip().strip("=").strip()
        if _SUMMARY_COUNTS_RE.search(stripped) and " in " in stripped:
            return stripped
    return ""


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
    positive collected count and ``ran == collected``. Anything else —
    missing, malformed, duplicated, or ``ran != collected`` — withholds trust,
    the safe direction. ``ran > collected`` is rejected as MALFORMED, not
    accepted as a benign over-count (codex #1222 r36): the reporter counts
    DISTINCT terminal-teardown ids (``--reruns`` retries can't inflate it), so
    a faithful record can never complete more distinct tests than it collected
    — an over-count is an inconsistent/forged record, and treating it as
    "complete" would let an inflated ``ran`` paper over unexecuted collected
    tests.

    SCOPE (codex #1222 r27): this proves every SELECTED item finished; it does
    NOT prove the selection itself was the whole real suite. ``collected`` is
    pytest's count under candidate-controlled collection — a committed
    ``pytest.ini`` (``python_files`` / ``norecursedirs``) or a ``conftest.py``
    collection hook (``pytest_collection_modifyitems`` / ``collect_ignore`` /
    ``pytest_ignore_collect``) could reduce ``tests/`` to a handful of passing
    tests and still yield a truthful ``ran == collected``. A base-derived
    count/manifest cross-check does NOT close this soundly: the SAME in-process
    access that shrinks the set can SURGICALLY deselect the one regression test
    (``collected`` drops by 1, under any ratio floor) or reparametrize its id,
    so a count guard would only catch the crudest variant while advertising a
    completeness it can't deliver. This is the documented in-process-hostile-
    author residual — a candidate with commit access to shared test
    infrastructure has strictly easier sabotage routes (a fixture that
    monkeypatches the code under test, an ``atexit`` forging this very record);
    the quarantine downgrade is scoped to mistakes and flakes, not such an
    author, and the PR body states this plainly rather than shipping a guard we
    can't keep.

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
    return collected > 0 and ran == collected


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

    Ambiguity is resolved by failing SAFE, never by guessing. Four tells,
    all making a split return the WHOLE line instead of a truncated id:

      * ``r6``: a node id whose EXPLICIT param id holds an *unmatched*
        bracket (``ids=["["]`` → ``test_x[[]``) leaves the depth counter
        unbalanced, so the message is absorbed into the returned id.
      * ``r9``: a param id that itself contains ``"] - ["`` (e.g.
        ``test_x[a] - [b]``) makes the inner ``]`` close the depth
        prematurely, so the FIRST depth-0 ``" - "`` lands *inside* the
        param sequence. Splitting there would truncate to ``test_x[a]`` —
        which could wrongly match a shorter quarantine entry and DOWNGRADE
        a real failure. Rejected when the message begins with ``"["``.
      * ``r40``: a custom param id like ``ids=["a] - b"]`` yields
        ``test_x[a] - b]``; the inner ``]`` drops depth to 0 so the first
        depth-0 ``" - "`` would COLLAPSE it to ``test_x[a]``. Rejected when
        the message holds an UNMATCHED ``"]"`` (that ``"]"`` closes a real
        param bracket, so the id extends past the ``"-"``).
      * ``r41``: the BALANCED variant ``test_x[a] - b[c]`` (param value
        ``a] - b[c``) — the r40 tell misses it because ``b[c]`` balances,
        yet it is still indistinguishable from a single node id. Rejected
        when the id-part ends with ``"]"`` and the message contains ``"["``.

    After these four guards a split is taken ONLY when the id-part is
    unparametrized (no ``"]"``, so no ``" - "`` can lie inside it) or the
    message provably cannot be a bracketed param continuation (no ``"["``,
    no unmatched ``"]"``). That makes the scrape FAIL-CLOSED SOUND: it never
    truncates to a shorter id that could false-green the base negative
    control — at the cost of over-blocking a pre-existing PARAMETRIZED
    failure whose message contains ``"["`` (safe, and rare: a normal
    parametrized failure only trips it when it was ALREADY failing at base).
    A blocked id won't match a normal quarantine entry either, so the
    failure blocks rather than being wrongly waived, and a mistargeted
    advisory re-run just yields an inconclusive result. The r4/r5 common
    cases are unaffected — an UNPARAMETRIZED test's message can hold any
    brackets (``test_x - AssertionError: [1]`` → ``test_x``); only a
    parametrized id-part gates on the message's ``"["``.

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
            id_part = rest[:i]
            msg = rest[i + 3 :]
            # Fail safe (return the WHOLE line, which won't match a shorter
            # id) whenever the split is unreliable:
            #  * message begins with "[" — can't tell a genuine bracket-
            #    leading message from a split taken in the MIDDLE of a
            #    bracketed param sequence ("test_x[a] - [b]") (codex r9).
            #  * message contains an UNMATCHED "]" — that "]" almost
            #    certainly closes a param bracket in the REAL node id (a
            #    custom param id like "a] - b" → node "test_x[a] - b]"), so
            #    the id extends PAST this "-" and splitting here would
            #    COLLAPSE it to a shorter id ("test_x[a]") that could wrongly
            #    match a PR-introduced failure and waive it as pre-existing —
            #    a FALSE GREEN in the base negative control (codex #1222 r40).
            #  * the id-part ends with "]" AND the message contains a "[" —
            #    a BALANCED-bracket variant of the above ("test_x[a] - b[c]",
            #    from a custom param value that embeds "] ... ["). The r40
            #    unmatched-"]" tell misses it because "b[c]" is balanced, yet
            #    the string is still indistinguishable from a single node id
            #    whose param value is "a] - b[c". Splitting COLLAPSES it to
            #    "test_x[a]", which could coincide with a DIFFERENT test's
            #    canonical id and waive a real PR regression as pre-existing —
            #    the same FALSE GREEN r40 closed for the unbalanced case
            #    (codex #1222 r41). This is the LAST ambiguous shape: after
            #    these three guards, a split is only taken when the id-part is
            #    unparametrized (no "]" → no " - " can be inside it) or the
            #    message provably can't be a bracketed param continuation (no
            #    "[", no unmatched "]"), so the scrape is fail-closed sound —
            #    never a false green, at the cost of blocking a pre-existing
            #    PARAMETRIZED failure whose message contains "[" (a safe,
            #    rare over-block that matches this scrape's stated preference:
            #    a false block over an unsound negative control).
            if (
                msg.startswith("[")
                or _has_unmatched_close_bracket(msg)
                or (id_part.endswith("]") and "[" in msg)
            ):
                return rest.strip()
            return id_part.strip()
        i += 1
    return rest.strip()


def _has_unmatched_close_bracket(s: str) -> bool:
    """True if ``s`` holds a ``]`` with no matching earlier ``[`` — the
    tell-tale that a candidate node-id/message split fell INSIDE the id's
    parametrization brackets (codex #1222 r40)."""
    depth = 0
    for c in s:
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth < 0:
                return True
    return False
