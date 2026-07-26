# SPDX-License-Identifier: Apache-2.0
"""Flake tracking (dev-flow proposal item ③).

ADVISORY ONLY — this step never blocks a PR. It surfaces flake
*candidates* for the quarantine from real PR runs (so a human can promote
a confirmed flake instead of hand-hunting for it), and it loudly flags the
inverse — a *quarantined* test that stopped flaking and now reproduces —
for a human to DE-quarantine. It never turns either signal into an
automated gate, and it never lets its own crash block a PR: the base
``execute`` would turn an uncaught exception into a blocking ``error``, so
``run`` catches everything and downgrades to ``skip``.

Why not auto-gate the "quarantine graveyard" case (codex #1222 r7 → r8):
r7 added a hard block when a quarantined test reproduced on every re-run,
reasoning it must be a deterministic regression the allowlist masks. r8
correctly rebutted that: back-to-back, in-process re-runs are
*correlated*, so this cannot soundly decide the quarantine question in
EITHER direction. A genuine environmental flake — e.g. the seeded
GPU-contention case — fails the whole retry batch under sustained
contention and *looks* deterministic; and an isolated re-run's order /
fixture scope / resource state differs from the full suite, so a failure
that is deterministic only under full-suite conditions *passes* here and
looks flaky. Four correlated re-runs are evidence for a human, not a
verdict — deciding it needs independent runs / history (slice 2+). So we
report it LOUDLY (a ``pass`` whose summary leads with a REVIEW flag and a
populated ``reproduced_quarantined`` bucket) and leave the call to the
human, exactly like the *adding*-to-quarantine direction. This keeps the
"never automate the quarantine decision" contract symmetric.

Mechanism: after ``full_unit`` runs, read its log for the tests that
failed, then re-run exactly those node ids in isolation with
``pytest-rerunfailures``. A test that FAILED in the full suite but PASSES
here (first try or after a rerun) is non-deterministic — a flake
candidate. One that reproduces its failure is likely a real bug, and is
reported as such (``full_unit`` already blocked on it).

Classification is on POSITIVE outcomes (``-rA`` gives every test's final
verdict): a node id that PASSED on the isolated re-run is the flake
candidate; one that FAILED/ERRORed again reproduces; anything that merely
went missing from the summary (skipped, deselected, never collected) is
reported as inconclusive, never called a flake.

Soundness guards (codex #1222):
  * Classification only runs when the re-run exits with a code we can
    reason about (0 = all passed, 1 = some failed). Anything else
    (collection error, crash, usage error) is inconclusive → we skip
    rather than mislabel a real failure as a flake.
  * The re-run is launched in its own session; on timeout the whole
    process group is SIGKILLed (race-free — the leader is unreaped at
    that point, so its pgid is still reserved), sweeping in-group
    descendants, and the post-kill drain is time-bounded so it can't
    WEDGE the gate.

Containment is best-effort, not absolute (codex #1222 r3, honestly
scoped): a descendant that escapes its group via its own ``setsid()``,
or a detached-stdio survivor of a *normally*-exiting re-run, is not
portably reapable on macOS — there is no cgroup / job object, and once
the group leader is reaped pgid recycling makes a post-reap ``killpg``
unsafe (it could SIGKILL an unrelated, recycled group). We do NOT sweep
after normal completion for that reason; this is the same trade-off
accepted in #1220. In practice a pytest re-run does neither of these,
and this step is advisory (never gates), so the residual leak is
tolerated rather than papered over with an unsafe kill.

If ``pytest-rerunfailures`` is not installed, the step skips cleanly —
the plugin is an optional [test]/[dev] extra, not a required one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import traceback

from .. import _nodeid_reporter
from .._pytest_summary import report_log_node_ids, summary_node_ids
from ..base import Step, StepResult
from ..context import Context
from ..quarantine import QuarantineError, is_quarantined, load_quarantine_from_ref

# Cap on how many failed ids we re-run. If main is broken with hundreds
# of reds, re-running them all (× reruns) would blow the budget for no
# advisory benefit — the signal is the same from a sample.
_MAX_RERUN_IDS = 25
_RERUNS = 3
_RERUN_TIMEOUT_S = 900
# Upper bound on the post-kill output drain. If a descendant escaped the
# process group (e.g. via its own setsid) and still holds the pipe open,
# we abandon the drain rather than let it hang the whole gate.
_DRAIN_TIMEOUT_S = 10
# pytest exit codes we can classify: 0 = all passed, 1 = some failed.
# 2 (interrupted), 3 (internal), 4 (usage), 5 (no tests) are abnormal —
# treating an unlisted id as "passed" would then mislabel real failures.
_CLASSIFIABLE_EXITS = (0, 1)


class FlakeTrackingStep(Step):
    name = "flake_tracking"
    description = (
        "classify full_unit failures — advisory (flake candidates + graveyard review)"
    )
    # Advisory: this step never emits fail/error — every path returns
    # pass/skip, and run() catches everything and downgrades to skip so a
    # crash can't turn into a blocking error. continue_on_error stays True
    # as belt-and-suspenders; the verdict cannot depend on this step.
    continue_on_error = True

    def should_run(self, ctx: Context) -> bool:
        # Nothing to classify unless full_unit actually ran (same blast
        # gate) and left a log to read.
        if ctx.blast_radius == "low":
            return False
        return ctx.artifact_path("full-unit.log").exists()

    def run(self, ctx: Context) -> StepResult:
        try:
            return self._run(ctx)
        except Exception:  # noqa: BLE001 — advisory: never block a PR
            return StepResult(
                name=self.name,
                status="skip",
                summary="advisory step errored; skipped (non-blocking)",
                details=f"```\n{traceback.format_exc()}\n```",
            )

    def _run(self, ctx: Context) -> StepResult:
        log_path = ctx.artifact_path("full-unit.log")
        if not log_path.exists():
            return StepResult(
                name=self.name,
                status="skip",
                summary="no full-unit.log to classify",
            )

        # Prefer full_unit's STRUCTURED node-id log (exact ids); fall back
        # to scraping its text log only if the plugin didn't run there.
        # Read BOTH FAILED and ERROR: a setup / teardown / collection flake
        # is recorded as ERROR, not FAILED, and it's just as much a flake
        # candidate as a call-phase failure — reading only FAILED reported
        # "no failures" and never re-ran those errored tests (codex #1222
        # r15 NIT).
        nodeid_log = ctx.artifact_path("full-unit-nodeids.tsv")
        if nodeid_log.exists():
            failed_only = report_log_node_ids(nodeid_log, "FAILED")
            error_only = report_log_node_ids(nodeid_log, "ERROR")
        else:
            text = log_path.read_text()
            failed_only = summary_node_ids(text, "FAILED")
            error_only = summary_node_ids(text, "ERROR")
        failed_set = set(failed_only)
        # Ids that logged an ERROR in ANY phase (setup / call / teardown /
        # collection). full_unit blocks any run containing an ERROR
        # UNCONDITIONALLY (it never quarantines what it couldn't even
        # complete), so promoting one of these to quarantine.yaml would be
        # INEFFECTIVE — a quarantine entry can't waive an ERROR run. This
        # deliberately includes an id that ALSO logged a FAILED — a call
        # failure followed by a teardown ERROR: the ERROR still makes the
        # whole run un-waivable, so it is error-origin regardless of the
        # FAILED (codex #1222 r19 — excluding failed_set here would mislabel
        # such an id as a quarantinable flake). Track their origin so a
        # recovered ERROR flake is reported in its own "investigate, not
        # quarantinable" bucket rather than recommended for a promotion that
        # wouldn't work.
        error_origin = set(error_only)
        original_failed = failed_only + [i for i in error_only if i not in failed_set]
        if not original_failed:
            return StepResult(
                name=self.name,
                status="skip",
                summary="full_unit had no failures or errors — nothing to classify",
            )

        if importlib.util.find_spec("pytest_rerunfailures") is None:
            return StepResult(
                name=self.name,
                status="skip",
                summary=(
                    "pytest-rerunfailures not installed — skipping flake "
                    "classification (add the [test] extra to enable)"
                ),
            )

        # Report against the ACTIVE quarantine, read from the IMMUTABLE base
        # SHA — same protected source full_unit's downgrade uses, and never
        # the mutable branch ref a candidate could rewrite (codex #1222 r10).
        # Fail-safe: no SHA / a broken registry just means "nothing
        # known-flaky yet" for the advisory report.
        entries = []
        if ctx.base_sha:
            try:
                entries = load_quarantine_from_ref(ctx.base_sha, ctx.repo_root)
            except QuarantineError:
                entries = []

        # Check EVERY quarantined failure, then fill the rest of the cap
        # with non-quarantined candidates (codex #1222 r8→r9 nit): a
        # graveyard candidate — a quarantined test that may have stopped
        # flaking — is the highest-value review signal, so it must NEVER be
        # dropped by the cap. The cap bounds only the advisory
        # non-quarantined remainder; if quarantined failures alone exceed
        # _MAX_RERUN_IDS we re-run all of them anyway (they're the whole
        # point). In practice the human-curated registry is tiny, so this
        # is a handful at most.
        q_failed = [i for i in original_failed if is_quarantined(i, entries)]
        other_failed = [i for i in original_failed if not is_quarantined(i, entries)]
        remaining = max(0, _MAX_RERUN_IDS - len(q_failed))
        sample = q_failed + other_failed[:remaining]
        truncated = len(original_failed) - len(sample)

        rerun_log = ctx.artifact_path("flake-rerun.log")
        rerun_nodeids = ctx.artifact_path("flake-rerun-nodeids.tsv")
        if rerun_nodeids.exists():
            rerun_nodeids.unlink()
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            # Neutralize candidate-controlled config for the advisory re-run
            # too (codex #1222 r16): -o addopts= drops the pytest.ini /
            # pyproject addopts wholesale so a planted --collect-only / marker
            # filter / -p no:<reporter> can't suppress classification, and
            # PYTEST_ADDOPTS is stripped from the env below (it bites through
            # -o addopts=). We re-run EXPLICIT node ids, so no marker filter
            # needs re-applying — the sample already comes from full_unit's
            # filtered run.
            "-o",
            "addopts=",
            *sample,
            f"--reruns={_RERUNS}",
            "-q",
            "--no-header",
            # Force color off so ANSI escapes can't break outcome parsing
            # (codex #1222 r6) — same reason as full_unit.
            "--color=no",
            # -rA => list every test's final outcome (PASSED/FAILED/…) in
            # the short summary. Used as the FALLBACK when the structured
            # node-id plugin isn't available; the plugin path classifies on
            # exact ``report.nodeid`` instead. Either way we classify on a
            # *positive* PASSED, not on mere absence from FAILED — a test
            # SKIPPED / deselected / never run is inconclusive, not a false
            # flake (codex #1222 r2).
            "-rA",
            "-p",
            "no:cacheprovider",
        ]
        # Strip PYTEST_ADDOPTS from the subprocess env — it bites THROUGH
        # ``-o addopts=`` (that only overrides the ini file), so a candidate
        # env var could still inject options into the advisory re-run
        # otherwise (codex #1222 r16). Done regardless of plugin availability.
        rerun_env = dict(os.environ)
        rerun_env.pop("PYTEST_ADDOPTS", None)
        # pytest-rerunfailures provides ``--reruns``. It autoloads by entry
        # point (registered under the name "rerunfailures") UNLESS
        # PYTEST_DISABLE_PLUGIN_AUTOLOAD is set. Add an explicit ``-p`` ONLY
        # when autoload is off: a redundant ``-p`` on an already-autoloaded
        # plugin is NOT a no-op — pluggy raises "Plugin already registered
        # under a different name" because the entry-point name (rerunfailures)
        # differs from the module name (pytest_rerunfailures), which crashes
        # the whole advisory re-run (codex #1222 r18 regression, fixed r19).
        # find_spec confirmed it's importable above, so the -p can't
        # ImportError when we do add it.
        if rerun_env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
            cmd += ["-p", "pytest_rerunfailures"]
        # Emit the structured node-id log (with PASSED, which the classifier
        # needs) so flake-vs-real is decided on exact ids, not summary text.
        if _nodeid_reporter.available():
            plugin_args, rerun_env = _nodeid_reporter.build_invocation(
                rerun_nodeids, ctx.repo_root, log_passes=True, base_env=rerun_env
            )
            cmd += plugin_args
        try:
            proc = _run_session(
                cmd, cwd=str(ctx.repo_root), timeout=_RERUN_TIMEOUT_S, env=rerun_env
            )
        except subprocess.TimeoutExpired as e:
            # Persist whatever we drained before the kill so a human can
            # see which re-run hung, instead of discarding it (codex
            # #1222 r3). _run_session attaches the partial output/stderr.
            rerun_log.write_text((e.output or "") + (e.stderr or ""))
            return StepResult(
                name=self.name,
                status="skip",
                summary=(
                    f"flake re-run exceeded {_RERUN_TIMEOUT_S}s on "
                    f"{len(sample)} test(s) — skipped (non-blocking)"
                ),
                artifacts=[str(rerun_log)],
            )
        rerun_log.write_text((proc.stdout or "") + (proc.stderr or ""))

        # Guard: only classify when the re-run exit code is one we can
        # interpret. On an abnormal exit, "absent from FAILED/ERROR" no
        # longer implies "passed" — bail rather than mislabel real
        # failures as flakes (codex #1222).
        if proc.returncode not in _CLASSIFIABLE_EXITS:
            return StepResult(
                name=self.name,
                status="skip",
                summary=(
                    f"flake re-run exited abnormally (code {proc.returncode}) "
                    f"— classification inconclusive, skipped"
                ),
                artifacts=[str(rerun_log)],
            )

        # Classify on POSITIVE outcomes, not on absence (codex #1222 r2):
        #   * PASSED on isolated re-run → was red in the suite, green now
        #     → non-deterministic → flake candidate.
        #   * FAILED/ERROR again → reproduces → likely a real bug.
        #   * anything else (SKIPPED / deselected / never collected) →
        #     INCONCLUSIVE — reported, but NOT called a flake.
        # Prefer the structured node-id log (exact ids); fall back to
        # scraping the summary only if the plugin didn't run (codex r10).
        if rerun_nodeids.exists():
            passed = set(report_log_node_ids(rerun_nodeids, "PASSED"))
            still_bad = set(report_log_node_ids(rerun_nodeids, "FAILED", "ERROR"))
        else:
            passed = set(summary_node_ids(proc.stdout, "PASSED"))
            still_bad = set(summary_node_ids(proc.stdout, "FAILED", "ERROR"))

        # FAILED/ERROR takes PRECEDENCE over PASSED for the same node id
        # (codex #1222 r12): a test whose call PASSES but whose teardown
        # ERRORs emits BOTH a PASSED and an ERROR line (they're different
        # pytest phases), and a test can likewise fail then pass across
        # phases. Such a run is NOT "green on re-run" — it ended badly — so
        # it must be a reproduction, never a flake candidate. Subtracting
        # ``still_bad`` from the passed set makes the buckets disjoint and
        # gives badness the last word.
        #
        # Collection-error targets need special handling (codex #1222 r17): a
        # collection ERROR id is a MODULE or DIR path (no "::"), and on a clean
        # re-run its CONTAINED tests log PASSED under their own "::"-bearing
        # ids — the target id itself never logs a PASSED. So a recovered
        # collection flake would fall through to "inconclusive" forever.
        # Treat it as recovered ONLY on POSITIVE evidence — a contained test
        # actually PASSED — not merely on the absence of a re-error (codex
        # #1222 r18): a module that collected nothing or only skipped is
        # inconclusive, not a proven flake, same as any other id.
        #
        # A MODULE target (``tests/test_x.py``) contains tests at the ``::``
        # boundary (``tests/test_x.py::test_y``); a DIRECTORY target
        # (``tests/subdir``) contains them at the ``/`` boundary
        # (``tests/subdir/test_x.py::test_y``) and would NEVER satisfy a
        # ``::`` check, leaving genuine directory-collection flakes forever
        # inconclusive (codex #1222 r19). Accept a passed descendant at
        # EITHER boundary — both are true containment, and the required
        # separator (``::`` or ``/``) prevents a sibling-prefix false match
        # (``tests/foo.py`` is not a descendant of dir ``tests/foo``).
        def _recovered(i: str) -> bool:
            if i in passed:
                return True
            return _is_collection_target(i) and any(
                p.startswith(i + "::") or p.startswith(i + "/") for p in passed
            )

        candidates = [i for i in sample if i not in still_bad and _recovered(i)]
        reproduced = [i for i in sample if i in still_bad]
        cand_set = set(candidates)
        inconclusive = [i for i in sample if i not in still_bad and i not in cand_set]

        # Split candidates three ways (codex #1222 r18): an ERROR-origin flake
        # can NOT be quarantined — full_unit blocks any run containing an
        # ERROR unconditionally, so a quarantine entry would never waive it.
        # Report those separately as "investigate / fix the root cause"
        # instead of recommending an ineffective promotion. The remaining
        # (call-phase FAILED) candidates split by whether they're already
        # covered, as before.
        error_candidates = [i for i in candidates if i in error_origin]
        new_candidates = [
            i
            for i in candidates
            if i not in error_origin and not is_quarantined(i, entries)
        ]
        known_flakes = [
            i
            for i in candidates
            if i not in error_origin and is_quarantined(i, entries)
        ]

        # Split reproductions the same way: a NON-quarantined test that
        # reproduces is a real failure full_unit blocked on. A QUARANTINED
        # one reproduced deterministically here — full_unit PASSED it as
        # non-blocking, so it's NOT something the gate blocked on; flag it
        # separately as "quarantine may be masking a now-deterministic
        # bug" rather than mislabeling it (codex #1222 r5).
        reproduced_new = [i for i in reproduced if not is_quarantined(i, entries)]
        reproduced_known = [i for i in reproduced if is_quarantined(i, entries)]

        payload = {
            "original_failed": original_failed,
            "sampled": sample,
            "truncated": truncated,
            "flake_candidates_new": new_candidates,
            "flake_candidates_known": known_flakes,
            "error_flake_candidates": error_candidates,
            "reproduced_likely_real": reproduced_new,
            "reproduced_quarantined": reproduced_known,
            "inconclusive": inconclusive,
        }
        cand_path = ctx.artifact_path("flake-candidates.json")
        cand_path.write_text(json.dumps(payload, indent=2))

        details = _render(
            new_candidates,
            known_flakes,
            error_candidates,
            reproduced_new,
            reproduced_known,
            inconclusive,
            sampled=len(sample),
            total=len(original_failed),
        )

        # The "quarantine graveyard" signal (codex #1222 r7 → r8): a
        # quarantined test that full_unit downgraded, yet reproduced on
        # every isolated re-run, MAY be a deterministic regression the
        # allowlist is masking — but correlated in-process re-runs can't
        # prove that (a sustained-contention flake looks identical). So we
        # surface it LOUDLY for a human to de-quarantine rather than
        # auto-blocking on weak evidence (see module docstring). The
        # summary leads with the review flag; status stays advisory.
        review = (
            f"⚠ REVIEW: {len(reproduced_known)} quarantined test(s) "
            f"reproduced on all {_RERUNS} re-run(s) — may be a regression, "
            f"not a flake; a human should re-verify and de-quarantine. "
            if reproduced_known
            else ""
        )
        error_note = (
            f"{len(error_candidates)} error-flake(s) [not quarantinable], "
            if error_candidates
            else ""
        )
        summary = (
            f"{review}"
            f"{len(new_candidates)} new flake candidate(s), "
            f"{error_note}"
            f"{len(reproduced_new)} reproduced, "
            f"{len(known_flakes)} known-flaky, "
            f"{len(inconclusive)} inconclusive — advisory"
        )
        return StepResult(
            name=self.name,
            status="pass",
            summary=summary,
            details=details,
            artifacts=[str(cand_path), str(rerun_log)],
        )


def _is_collection_target(node_id: str) -> bool:
    """True for a collection-error id — a module / directory path with no
    ``::`` test component. pytest reports a collection failure against the
    module/dir, whose contained tests re-run under their own ``::`` ids, so
    the target itself never logs a PASSED and needs recovery handled
    explicitly (codex #1222 r17)."""
    return "::" not in node_id


def _run_session(
    cmd: list[str], cwd: str, timeout: int, env: dict | None = None
) -> subprocess.CompletedProcess:
    """Run ``cmd`` in its own session and, on timeout, SIGKILL the whole
    process group so in-group test-spawned descendants (inference
    servers, GPU workers) are swept before later validation steps
    (codex #1222). Best-effort: a setsid-escaping descendant is not
    portably containable on macOS — see the module docstring.

    ``env`` (when given) is the subprocess environment — used to carry the
    ``_nodeid_reporter`` plugin's log path + PYTHONPATH; ``None`` inherits.

    ``start_new_session=True`` makes the child a session/group leader
    (pgid == pid). On timeout the child has not been reaped, so its pgid
    is still reserved — killpg is race-free (the safe pre-reap case). We
    kill BEFORE the final ``communicate`` so a descendant holding the
    pipe open can't hang the drain, and the drain itself is time-bounded
    so even a group-escaping descendant can't wedge the gate (codex
    #1222 r2).
    """
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        start_new_session=True,
        env=env,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        out, err = _drain_bounded(proc)
        raise subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def _drain_bounded(proc: subprocess.Popen) -> tuple[str, str]:
    """Drain the child's pipes after a kill, but never block forever.

    The normal case returns immediately (the group is dead) with the full
    captured output. If a descendant escaped the process group and still
    holds the write end open, the bounded ``communicate`` raises again —
    we then re-kill, force the pipes closed, and REAP the direct child so
    it can't be left a zombie (the group leader is already SIGKILLed, so
    the bounded ``wait`` returns promptly — codex #1222 r4). The tail of
    output is unrecoverable on this rare double-timeout path: CPython does
    not expose the bytes buffered inside a timed-out ``communicate``, so
    we return empties and rely on the module docstring's honest note that
    a setsid-escaping descendant is not portably containable on macOS."""
    try:
        return proc.communicate(timeout=_DRAIN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        # Reap the (already-killed) direct child; bounded so a wedged
        # waitpid can't hang the gate either.
        try:
            proc.wait(timeout=_DRAIN_TIMEOUT_S)
        except (subprocess.TimeoutExpired, OSError):
            pass
        return "", ""


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        # Group already gone, or getpgid raced the exit — best-effort
        # kill the direct child and move on.
        try:
            proc.kill()
        except OSError:
            pass


def _render(
    new_candidates: list[str],
    known_flakes: list[str],
    error_candidates: list[str],
    reproduced_new: list[str],
    reproduced_known: list[str],
    inconclusive: list[str],
    *,
    sampled: int,
    total: int,
) -> str:
    parts: list[str] = []
    if new_candidates:
        parts.append(
            "**New flake candidates** (failed the full suite, passed on "
            "isolated re-run — consider promoting to `quarantine.yaml` "
            "after confirming):\n```\n" + "\n".join(new_candidates) + "\n```"
        )
    if error_candidates:
        parts.append(
            "**Error flakes — investigate, NOT quarantinable** (errored "
            "(setup / teardown / collection) in the full suite, recovered on "
            "isolated re-run — so non-deterministic, but `full_unit` blocks "
            "ANY run containing an ERROR unconditionally, so a `quarantine.yaml`"
            " entry would NOT waive these; fix the fixture / collection or the "
            "code):\n```\n" + "\n".join(error_candidates) + "\n```"
        )
    if reproduced_new:
        parts.append(
            "**Reproduced on re-run** (failure is deterministic — likely a "
            "real bug, NOT a flake; `full_unit` blocked on these):\n```\n"
            + "\n".join(reproduced_new)
            + "\n```"
        )
    if reproduced_known:
        parts.append(
            "**Quarantined but reproduced deterministically** (`full_unit` "
            "PASSED these as quarantined, but they failed every re-run here "
            "— the quarantine may be masking a now-deterministic bug; "
            "re-check whether the entry still belongs in `quarantine.yaml`):"
            "\n```\n" + "\n".join(reproduced_known) + "\n```"
        )
    if known_flakes:
        parts.append(
            "**Already quarantined** (confirmed still flaky):\n```\n"
            + "\n".join(known_flakes)
            + "\n```"
        )
    if inconclusive:
        parts.append(
            "**Inconclusive** (didn't positively pass or fail on isolated "
            "re-run — skipped, deselected, or not collected; NOT classified "
            "as a flake):\n```\n" + "\n".join(inconclusive) + "\n```"
        )
    if total > sampled:
        parts.append(
            f"_Re-ran {sampled} of {total} failures (quarantined first, then "
            f"non-quarantined up to the cap) — {total - sampled} not re-run "
            f"this pass._"
        )
    return "\n\n".join(parts) if parts else "no classification produced"
