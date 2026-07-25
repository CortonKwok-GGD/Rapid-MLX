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

The re-run set is tiny (only the failures — usually zero), so a plain
``subprocess.run`` with a timeout is enough; none of the bounded-tail /
process-group machinery the full-suite ``diff_coverage`` run needs. If
``pytest-rerunfailures`` is not installed, the step skips cleanly — the
plugin is an optional [test]/[dev] extra, not a required one.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import traceback

from ..base import Step, StepResult
from ..context import Context
from ..quarantine import QuarantineError, is_quarantined, load_quarantine

# Cap on how many failed ids we re-run. If main is broken with hundreds
# of reds, re-running them all (× reruns) would blow the budget for no
# advisory benefit — the signal is the same from a sample.
_MAX_RERUN_IDS = 25
_RERUNS = 3
_RERUN_TIMEOUT_S = 900


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

        original_failed = _failed_ids(log_path.read_text())
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

        # Fail-safe: advisory, so a broken registry just means "nothing
        # is known-flaky yet" for the purpose of the report.
        try:
            entries = load_quarantine()
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
            "-p",
            "no:cacheprovider",
        ]
        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                cwd=str(ctx.repo_root),
                timeout=_RERUN_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return StepResult(
                name=self.name,
                status="skip",
                summary=(
                    f"flake re-run exceeded {_RERUN_TIMEOUT_S}s on "
                    f"{len(sample)} test(s) — skipped (non-blocking)"
                ),
            )
        rerun_log.write_text((proc.stdout or "") + (proc.stderr or ""))

        # A test that still failed/errored on isolated re-run (with
        # reruns) reproduces → likely a real bug. Everything else in the
        # sample passed this time → non-deterministic → flake candidate.
        still_bad = set(_bad_ids(proc.stdout))
        candidates = [i for i in sample if i not in still_bad]
        reproduced = [i for i in sample if i in still_bad]

        # Split candidates by whether they're already covered.
        new_candidates = [i for i in candidates if not is_quarantined(i, entries)]
        known_flakes = [i for i in candidates if is_quarantined(i, entries)]

        payload = {
            "original_failed": original_failed,
            "sampled": sample,
            "truncated": truncated,
            "flake_candidates_new": new_candidates,
            "flake_candidates_known": known_flakes,
            "reproduced_likely_real": reproduced,
        }
        cand_path = ctx.artifact_path("flake-candidates.json")
        cand_path.write_text(json.dumps(payload, indent=2))

        details = _render(new_candidates, known_flakes, reproduced, truncated)
        summary = (
            f"{len(new_candidates)} new flake candidate(s), "
            f"{len(reproduced)} reproduced, "
            f"{len(known_flakes)} known-flaky — advisory"
        )
        return StepResult(
            name=self.name,
            status="pass",
            summary=summary,
            details=details,
            artifacts=[str(cand_path), str(rerun_log)],
        )


def _render(
    new_candidates: list[str],
    known_flakes: list[str],
    reproduced: list[str],
    truncated: int,
) -> str:
    parts: list[str] = []
    if new_candidates:
        parts.append(
            "**New flake candidates** (failed the full suite, passed on "
            "isolated re-run — consider promoting to `quarantine.yaml` "
            "after confirming):\n```\n" + "\n".join(new_candidates) + "\n```"
        )
    if reproduced:
        parts.append(
            "**Reproduced on re-run** (failure is deterministic — likely a "
            "real bug, NOT a flake; `full_unit` blocked on these):\n```\n"
            + "\n".join(reproduced)
            + "\n```"
        )
    if known_flakes:
        parts.append(
            "**Already quarantined** (confirmed still flaky):\n```\n"
            + "\n".join(known_flakes)
            + "\n```"
        )
    if truncated > 0:
        parts.append(
            f"_Sampled the first {_MAX_RERUN_IDS} of "
            f"{_MAX_RERUN_IDS + truncated} failures — {truncated} not "
            f"re-run this pass._"
        )
    return "\n\n".join(parts) if parts else "no classification produced"


def _failed_ids(text: str) -> list[str]:
    """Node ids pytest labels FAILED in its short-summary section."""
    return _summary_ids(text, prefixes=("FAILED",))


def _bad_ids(text: str) -> list[str]:
    """Node ids pytest labels FAILED or ERROR — an ERROR on re-run is
    not a pass, so it counts as 'reproduced', not a flake candidate."""
    return _summary_ids(text, prefixes=("FAILED", "ERROR"))


def _summary_ids(text: str, prefixes: tuple[str, ...]) -> list[str]:
    """Parse node ids out of pytest's short test summary block.

    A summary line is ``<LABEL> <nodeid>[ - <message>]``; the node id is
    everything between the label and the ``" - "`` message separator
    (parametrized ids can hold spaces inside ``[...]``, so we split on
    ``" - "`` rather than whitespace).
    """
    out: list[str] = []
    in_summary = False
    for line in (text or "").splitlines():
        if "short test summary" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        if line.startswith("="):
            break
        for prefix in prefixes:
            if line.startswith(prefix + " "):
                rest = line[len(prefix) + 1 :]
                node_id = rest.split(" - ", 1)[0].strip()
                if node_id:
                    out.append(node_id)
                break
    return out
