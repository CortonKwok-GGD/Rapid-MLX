# SPDX-License-Identifier: Apache-2.0
"""Advisory step — flake tracking (dev-flow proposal item ③).

MEASURE-ONLY. This step never blocks a PR: every path returns ``pass``
or ``skip`` (the base ``execute`` would turn an uncaught exception into a
blocking ``error``, so ``run`` catches everything and downgrades to
``skip``). It is the reporting half of the flake system whose gating half
lives in ``full_unit`` (quarantine-aware) — here we surface *candidates*
for that quarantine from real PR runs, so a human can promote a confirmed
flake instead of hand-hunting for it.

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

from .._pytest_summary import summary_node_ids
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
    description = "advisory — classify full_unit failures as flake vs real"
    # Advisory: an error here must not sink the PR. run() also catches
    # everything itself; this is belt-and-suspenders with the runner.
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

        original_failed = summary_node_ids(log_path.read_text(), "FAILED")
        if not original_failed:
            return StepResult(
                name=self.name,
                status="skip",
                summary="full_unit had no failures — nothing to classify",
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

        # Report against the ACTIVE (protected base) quarantine, same
        # source the gate uses. Fail-safe: a broken/absent registry just
        # means "nothing known-flaky yet" for the report.
        try:
            entries = load_quarantine_from_ref(
                ctx.base_sha or ctx.base_branch, ctx.repo_root
            )
        except QuarantineError:
            entries = []

        sample = original_failed[:_MAX_RERUN_IDS]
        truncated = len(original_failed) - len(sample)

        rerun_log = ctx.artifact_path("flake-rerun.log")
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            *sample,
            f"--reruns={_RERUNS}",
            "-q",
            "--no-header",
            # Force color off so ANSI escapes can't break outcome parsing
            # (codex #1222 r6) — same reason as full_unit.
            "--color=no",
            # -rA => list every test's final outcome (PASSED/FAILED/…) in
            # the short summary. We classify a flake candidate on a
            # *positive* PASSED, not on mere absence from FAILED — so a
            # test that got SKIPPED / deselected / never ran on the
            # isolated re-run is inconclusive, not a false flake
            # (codex #1222 r2).
            "-rA",
            "-p",
            "no:cacheprovider",
        ]
        try:
            proc = _run_session(cmd, cwd=str(ctx.repo_root), timeout=_RERUN_TIMEOUT_S)
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
        passed = set(summary_node_ids(proc.stdout, "PASSED"))
        still_bad = set(summary_node_ids(proc.stdout, "FAILED", "ERROR"))
        candidates = [i for i in sample if i in passed]
        reproduced = [i for i in sample if i in still_bad]
        inconclusive = [i for i in sample if i not in passed and i not in still_bad]

        # Split candidates by whether they're already covered.
        new_candidates = [i for i in candidates if not is_quarantined(i, entries)]
        known_flakes = [i for i in candidates if is_quarantined(i, entries)]

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
            "reproduced_likely_real": reproduced_new,
            "reproduced_quarantined": reproduced_known,
            "inconclusive": inconclusive,
        }
        cand_path = ctx.artifact_path("flake-candidates.json")
        cand_path.write_text(json.dumps(payload, indent=2))

        details = _render(
            new_candidates,
            known_flakes,
            reproduced_new,
            reproduced_known,
            inconclusive,
            truncated,
        )
        summary = (
            f"{len(new_candidates)} new flake candidate(s), "
            f"{len(reproduced_new)} reproduced, "
            f"{len(known_flakes)} known-flaky, "
            f"{len(reproduced_known)} quarantined-reproduced, "
            f"{len(inconclusive)} inconclusive — advisory"
        )
        return StepResult(
            name=self.name,
            status="pass",
            summary=summary,
            details=details,
            artifacts=[str(cand_path), str(rerun_log)],
        )


def _run_session(cmd: list[str], cwd: str, timeout: int) -> subprocess.CompletedProcess:
    """Run ``cmd`` in its own session and, on timeout, SIGKILL the whole
    process group so in-group test-spawned descendants (inference
    servers, GPU workers) are swept before later validation steps
    (codex #1222). Best-effort: a setsid-escaping descendant is not
    portably containable on macOS — see the module docstring.

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
    reproduced_new: list[str],
    reproduced_known: list[str],
    inconclusive: list[str],
    truncated: int,
) -> str:
    parts: list[str] = []
    if new_candidates:
        parts.append(
            "**New flake candidates** (failed the full suite, passed on "
            "isolated re-run — consider promoting to `quarantine.yaml` "
            "after confirming):\n```\n" + "\n".join(new_candidates) + "\n```"
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
    if truncated > 0:
        parts.append(
            f"_Sampled the first {_MAX_RERUN_IDS} of "
            f"{_MAX_RERUN_IDS + truncated} failures — {truncated} not "
            f"re-run this pass._"
        )
    return "\n\n".join(parts) if parts else "no classification produced"
