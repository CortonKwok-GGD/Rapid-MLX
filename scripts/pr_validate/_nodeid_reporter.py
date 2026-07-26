# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: record the CANONICAL node id of each test outcome.

``full_unit`` (quarantine partition) and ``flake_tracking`` (rerun
classification) both need the exact node id of every failing/passing test.
Parsing them out of pytest's terminal short-summary is IRREDUCIBLY
ambiguous: a summary line is ``<LABEL> <nodeid>[ - <message>]``, but a
parametrized node id can itself contain ``" - "`` and unbalanced ``[]``
(``test_x[a] - [b]``, ``test_x[a] - b]``, ``test_x[a] - b[c]`` …), so no
text rule can split id-from-message without a hole — five successive
patches each grew a new edge case (codex #1222 r4/r5/r6/r9/r10). This
plugin sidesteps the parse entirely: ``report.nodeid`` is pytest's ground
truth, the same string a ``quarantine.yaml`` entry is written as.

It writes tab-separated ``<LABEL>\t<nodeid>`` lines to the file named by
``$PR_VALIDATE_NODEID_LOG`` (a plain, append-only log the consumers read
back):

  * call-phase failure  → ``FAILED``
  * strict-xfail XPASS   → ``ERROR`` — a call-phase "failed" that pytest makes
    FATAL because the xfail marker is stale; recorded as ERROR (unconditional
    block, never a quarantinable flake) rather than a downgradeable FAILED
    (codex #1222 r31). Detected by its string ``"[XPASS(strict)]"`` longrepr.
  * setup/teardown fail  → ``ERROR``  (mirrors pytest's own -rfE split)
  * collection failure   → ``ERROR``
  * call-phase pass      → ``PASSED`` — ONLY when
    ``$PR_VALIDATE_NODEID_LOG_PASSES=1``. The rerun classifier needs
    positive PASSED signals; the full suite (14k passes) does not, and
    logging them all would bloat the gate's artifact for no benefit.
  * session finished      → ``SESSIONFINISH\t<ran> <collected>`` — one
    line written from ``pytest_sessionfinish``, proving the run reached a
    GRACEFUL end and recording how many DISTINCT items pytest attempted vs
    collected (distinct so ``--reruns`` retries can't inflate the count —
    codex #1222 r16). The quarantine downgrade in ``full_unit`` is only
    sound on a COMPLETE run, and this record is how the consumer proves it
    STRUCTURALLY instead of scraping stdout banners (codex #1222 r15):
      - the hook fires only on a graceful end, so its ABSENCE flags a hard
        truncation — ``os._exit`` / a crash / SIGKILL kills the process
        mid-suite and this line is simply never written;
      - ``ran < collected`` flags an EARLY stop — ``-x`` / a hit
        ``--maxfail`` / ``--stepwise`` run fewer items than they collected.
    This replaces a fragile ``"stopping after" in stdout`` heuristic that
    false-positived when those words appeared in a test's own traceback.

  * retried attempt        → ``RERUN`` — logged whenever a report's
    ``outcome`` is ``"rerun"`` (pytest-rerunfailures). A GATING run
    DISABLES reruns, so any RERUN record proves the plugin was smuggled
    back in — under an ARBITRARY name that a ``-p no:<name>`` block can't
    reach (``config.pluginmanager.register(mod, name="…")`` + a
    ``@pytest.mark.flaky`` marker still reruns, verified). ``full_unit``
    blocks on this record NAME-INDEPENDENTLY (codex #1222 r23), the
    backstop the by-name guard alone can't provide.

Reads/skips are never logged: a test absent from the log positively
"didn't fail (or pass, if passes are logged)", which is exactly the
"inconclusive, not a flake" signal ``flake_tracking`` wants.

Node ids are NOT trustworthy text. A hostile
``pytest_make_parametrize_id`` can return a param id containing a literal
``\t`` or ``\n`` (verified — pytest preserves both), which written raw
would forge extra log records — a fake ``SESSIONFINISH`` masking a
truncated run, a fake ``PASSED`` turning a real failure into a
"recovered" flake in the advisory step. So every value field is
``escape_field``-encoded on write (backslash, tab, newline, CR) and
``unescape_field``-decoded on read, guaranteeing one record per physical
line with no injectable delimiter (codex #1222 r23).

This is NOT tamper-proof against a hostile in-process author (an
``atexit``/``conftest`` could rewrite the file with real newlines) —
nothing in-process is, junit-xml included; that residual is scoped in
``_pytest_summary`` and defended by the other gates. What escaping buys
is closing the node-id INJECTION channel (the realistic vector), on top
of exact, unambiguous node ids for every realistic run.
"""

from __future__ import annotations

import importlib.util
import os

_ENV_LOG = "PR_VALIDATE_NODEID_LOG"
_ENV_PASSES = "PR_VALIDATE_NODEID_LOG_PASSES"

PLUGIN_MODULE = "scripts.pr_validate._nodeid_reporter"

# Label of the one-per-session completion record (see module docstring and
# ``pytest_sessionfinish``). The consumer keys on this to prove a COMPLETE
# run before granting any quarantine downgrade (codex #1222 r15).
SESSION_LABEL = "SESSIONFINISH"

# Label of a retried-attempt record. Logged on any ``outcome == "rerun"``
# report so ``full_unit`` can detect a smuggled pytest-rerunfailures
# NAME-INDEPENDENTLY: a gating run disables reruns, so any RERUN record
# means the plugin was re-registered under a name the ``-p no:<name>``
# block can't reach, and a rerun could retry a real failure into a pass
# (codex #1222 r23).
RERUN_LABEL = "RERUN"

# SUPPLEMENTARY label for a COLLECTION failure, written ALONGSIDE the ERROR
# record (never instead of it), so ``full_unit``'s gate — which reads ERROR
# — is untouched and can't be weakened. The advisory classifier reads this
# to recognize a collection target whose id CONTAINS ``::`` (a class
# collector: ``tests/test_x.py::TestClass``), which the "``::``-absence"
# heuristic alone misses (codex #1222 r24).
COLLECTERROR_LABEL = "COLLECTERROR"

# The DISTINCT node ids pytest ran to COMPLETION this session — counted on
# the ``teardown`` report, which fires only AFTER an item's full lifecycle
# (setup + call + teardown) finishes. Compared against
# ``session.testscollected`` at sessionfinish to detect an early stop (``-x``
# / ``--maxfail`` / ``--stepwise`` tear down fewer items than they collected).
#
# Counting the ``teardown`` (terminal) report, NOT the ``setup`` report, is
# deliberate (codex #1222 r20): a ``setup`` report is emitted the moment an
# item STARTS, so a test that calls ``pytest.exit(returncode=0)`` from its
# CALL body — after its own setup report but before completing — would leave
# ``ran == collected`` and be accepted as a complete green run even though its
# body never finished and every later item was skipped. An item that exits
# mid-call emits NO teardown report (verified empirically), so teardown-
# counting withholds on exactly that truncation. skip / skipif / xfail /
# xpass / runtime-``pytest.skip`` / setup-ERROR / teardown-ERROR items all DO
# emit a teardown report, so a legitimately complete run still counts
# item-for-item.
#
# A SET, not a counter: ``pytest-rerunfailures`` retries can emit multiple
# reports for the SAME id, so a raw count could inflate past ``collected`` and
# mask a truncated run — deduping by node id keeps the comparison
# item-for-item (codex #1222 r16). Reset at ``pytest_sessionstart`` so a
# reused module never carries a stale set.
_completed_ids: set[str] = set()


def available() -> bool:
    """True iff this plugin module can be imported by a pytest subprocess.

    It is normally always importable (the validator itself runs as
    ``scripts.pr_validate``), but the check lets a caller degrade to
    terminal-summary parsing instead of passing an unloadable ``-p`` that
    would make pytest exit with a usage error."""
    return importlib.util.find_spec(PLUGIN_MODULE) is not None


def build_invocation(
    log_path,
    repo_root,
    *,
    log_passes: bool = False,
    base_env: dict | None = None,
) -> tuple[list[str], dict]:
    """Build the ``(extra_pytest_args, env)`` that make a pytest subprocess
    emit the structured node-id log at ``log_path``.

    NOTE: this helper must NOT be named ``pytest_*`` — pytest's plugin
    manager treats every ``pytest_``-prefixed callable in a loaded plugin
    module as a hook and raises ``PluginValidationError: unknown hook`` at
    collection when the name isn't a real hook (this module IS loaded as a
    ``-p`` plugin, so the two genuine hooks below are the only ``pytest_``
    names allowed).

    ``repo_root`` is prepended to ``PYTHONPATH`` so the ``-p`` plugin
    resolves in the subprocess (``scripts`` is a namespace package, so it
    must be importable from the repo root). Pass ``log_passes=True`` for the
    rerun classifier, which needs positive PASSED signals."""
    args = ["-p", PLUGIN_MODULE]
    env = dict(os.environ if base_env is None else base_env)
    env[_ENV_LOG] = str(log_path)
    if log_passes:
        env[_ENV_PASSES] = "1"
    else:
        # Set the flag DEFINITIVELY, don't just skip it: an inherited
        # PR_VALIDATE_NODEID_LOG_PASSES=1 from the parent env would
        # otherwise make full_unit (log_passes=False) log all ~14k passes
        # and bloat the artifact (codex #1222 r12). Pop it so False means
        # off regardless of what was inherited.
        env.pop(_ENV_PASSES, None)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    return args, env


# Every character ``str.splitlines()`` treats as a line boundary BEYOND
# ``\n`` / ``\r`` (handled by their own named escapes). A consumer that reads
# the log with ``splitlines()`` — the readers in ``_pytest_summary`` do —
# would break a record on ANY of these, so a hostile node id embedding one
# would still forge a second record (a fake ``SESSIONFINISH`` / ``PASSED``),
# the r23 injection channel via a separator the original escape missed (codex
# #1222 r28). None has a legitimate place in a pytest node id; each is encoded
# as a ``\uXXXX`` escape so the written value is a single "line" under BOTH
# ``splitlines()`` and ``split("\n")``.
_EXTRA_LINE_SEPARATORS = (
    "\v",  # U+000B LINE TABULATION
    "\f",  # U+000C FORM FEED
    "\x1c",  # U+001C FILE SEPARATOR
    "\x1d",  # U+001D GROUP SEPARATOR
    "\x1e",  # U+001E RECORD SEPARATOR
    "\x85",  # U+0085 NEXT LINE
    "\u2028",  # U+2028 LINE SEPARATOR
    "\u2029",  # U+2029 PARAGRAPH SEPARATOR
)

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def escape_field(value: str) -> str:
    """Encode a value field so it round-trips through the one-record-per-line
    log even when it embeds a delimiter or a line-boundary character.

    A node id is untrusted text (a hostile ``pytest_make_parametrize_id``
    can smuggle a literal ``\\t``/``\\n`` — or a Unicode line separator like
    ``\\u2028`` — into it), so an unescaped write would let it forge additional
    log records. Escape the BACKSLASH first, then tab / newline / CR, then
    every OTHER character ``str.splitlines()`` recognizes (as ``\\uXXXX``), so
    the written value occupies exactly one physical line under any line-split
    definition and decodes back losslessly (codex #1222 r23/r28)."""
    value = value.replace("\\", "\\\\")
    value = value.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    for sep in _EXTRA_LINE_SEPARATORS:
        value = value.replace(sep, f"\\u{ord(sep):04x}")
    return value


def unescape_field(value: str) -> str:
    """Inverse of ``escape_field``. Scans left-to-right so an escaped
    backslash (``\\\\``) is consumed as one literal ``\\`` and can't pair
    with a following ``t``/``n``/``r``/``uXXXX`` to fabricate a control char
    (``\\\\t`` decodes to ``\\`` + ``t``, NOT a tab). An unknown escape, a
    malformed ``\\u`` (not four hex digits), or a trailing lone backslash is
    preserved verbatim — the writer never emits those, so this only keeps a
    hand-written/raw record intact rather than corrupting it."""
    if "\\" not in value:
        return value
    out: list[str] = []
    i = 0
    n = len(value)
    while i < n:
        c = value[i]
        if c == "\\" and i + 1 < n:
            nxt = value[i + 1]
            decoded = {"\\": "\\", "t": "\t", "n": "\n", "r": "\r"}.get(nxt)
            if decoded is not None:
                out.append(decoded)
                i += 2
                continue
            # ``\uXXXX`` (exactly four hex digits) → the encoded separator.
            if nxt == "u" and i + 6 <= n:
                hexdigits = value[i + 2 : i + 6]
                if all(ch in _HEX_DIGITS for ch in hexdigits):
                    out.append(chr(int(hexdigits, 16)))
                    i += 6
                    continue
        out.append(c)
        i += 1
    return "".join(out)


def _append(label: str, nodeid: str) -> None:
    path = os.environ.get(_ENV_LOG)
    if not path:
        return
    # Append so setup/call/teardown reports for the same test each add
    # their own line; open per-call to stay robust to xdist workers each
    # writing (append is atomic for the small line sizes here). The value
    # is escape_field-encoded so an injected tab/newline in a hostile node
    # id can't forge a second record (codex #1222 r23).
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{label}\t{escape_field(nodeid)}\n")


def pytest_sessionstart(session) -> None:  # noqa: ANN001, ARG001 — pytest hook
    # Reset the completion set so a reused module (e.g. a harness running
    # multiple sessions in one process) never carries stale ids into the
    # next session's completeness check (codex #1222 r15).
    _completed_ids.clear()


def _is_strict_xpass(report) -> bool:  # noqa: ANN001
    """True iff ``report`` is a strict-xfail test that XPASSED.

    pytest makes a ``@pytest.mark.xfail(strict=True)`` that unexpectedly
    passes FATAL (``outcome == "failed"``) — the xfail marker is now stale.
    Its call report carries a STRING ``longrepr`` of the form
    ``"[XPASS(strict)] <reason>"``, whereas a genuine call failure carries an
    ``ExceptionRepr`` OBJECT — the only clean, attribute-stable discriminator
    (``report.wasxfail`` is UNSET for a strict xpass; verified codex
    #1222 r27/r31)."""
    longrepr = getattr(report, "longrepr", None)
    return isinstance(longrepr, str) and longrepr.startswith("[XPASS(strict)]")


def pytest_runtest_logreport(report) -> None:  # noqa: ANN001 — pytest hook
    if report.outcome == "rerun":
        # pytest-rerunfailures emitted a retry. Record it NAME-INDEPENDENTLY
        # (this fires on the OUTCOME, not the plugin's registered name) so
        # full_unit can detect a plugin smuggled in under an arbitrary name
        # that a ``-p no:<name>`` block can't reach (codex #1222 r23). A
        # gating run disables reruns, so any RERUN record is a tamper signal.
        _append(RERUN_LABEL, report.nodeid)
    if report.when == "teardown":
        # One teardown report per item pytest ran to COMPLETION; record the
        # DISTINCT node id so pytest_sessionfinish can prove ran == collected.
        # Teardown (not setup) so a ``pytest.exit()`` mid-call can't leave a
        # phantom "ran" for an item whose body never finished (codex #1222
        # r20). Deduping matters under --reruns, where the same id can emit
        # several reports (codex #1222 r16). A run truncated by -x / --maxfail
        # / --stepwise / a mid-call pytest.exit tears down strictly fewer
        # distinct items than it collected.
        _completed_ids.add(report.nodeid)
    if report.when == "call":
        if report.outcome == "failed":
            if _is_strict_xpass(report):
                # A strict-xfail XPASS is pytest's INTENTIONAL fatal signal
                # (the xfail marker is stale). Record it as ERROR, not FAILED,
                # so it BLOCKS unconditionally in every gate: ERROR is never a
                # quarantinable flake in full_unit and always makes targeted
                # untrusted — a FAILED strict-xpass on a (mistakenly)
                # quarantined node would otherwise be downgraded to a
                # non-blocking flake, masking the fatal (codex #1222 r31).
                _append("ERROR", report.nodeid)
            else:
                _append("FAILED", report.nodeid)
        elif report.outcome == "passed" and os.environ.get(_ENV_PASSES) == "1":
            _append("PASSED", report.nodeid)
    elif report.failed:
        # A failure in setup or teardown — pytest reports these as ERROR.
        _append("ERROR", report.nodeid)


def pytest_collectreport(report) -> None:  # noqa: ANN001 — pytest hook
    # A collection failure (import error, bad conftest) — ERROR, and the
    # node id is the module/dir/class that failed to collect.
    if report.failed:
        _append("ERROR", report.nodeid)
        # ALSO tag it as a COLLECTION error (in ADDITION to ERROR, never
        # instead) so the advisory classifier can recognize a collection
        # target even when its id contains "::" — a class collector reports
        # `tests/test_x.py::TestClass`, which the "::"-absence heuristic
        # misses. full_unit reads ERROR for its gate, so this extra label
        # cannot weaken it (codex #1222 r24).
        _append(COLLECTERROR_LABEL, report.nodeid)


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001, ARG001 — pytest hook
    # Write the STRUCTURED completion record. This hook fires only on a
    # graceful session end, so the record's mere presence proves the
    # process did not die mid-suite (os._exit / crash / SIGKILL), and the
    # ran-vs-collected counts prove no early stop (-x / --maxfail /
    # --stepwise). The consumer requires this record with ran >= collected
    # before any quarantine downgrade — a structural replacement for the
    # old stdout-banner heuristic (codex #1222 r15). ``_append`` no-ops
    # when no log path is configured.
    collected = int(getattr(session, "testscollected", 0) or 0)
    _append(SESSION_LABEL, f"{len(_completed_ids)} {collected}")
