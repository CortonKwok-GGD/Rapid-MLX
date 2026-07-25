# SPDX-License-Identifier: Apache-2.0
"""Step 4 — full unit test suite.

Skipped for low-blast PRs (docs, examples) — they can't break behavior.
For medium and high blast PRs, runs the same set we use locally:
``tests/`` minus integrations (those need a running server) and
``test_event_loop.py`` (long-running, separate gate).

Pre-existing failures on main are NOT filtered here — that's step 3's
job. This step validates "the suite as-is is still green"; if main is
broken that's a separate problem and we want to surface it loudly.

Quarantine (dev-flow ③): a failure whose node id is listed in
``quarantine.yaml`` is reported but does NOT block the PR. This exists
so a single CONFIRMED flake can't force the "rerun until green" ritual
that quietly destroys gate credibility. The split is fail-safe — an
unreadable registry falls back to an empty quarantine, so a broken file
can only make the gate *stricter*, never pass something it otherwise
wouldn't. A non-quarantined failure still blocks, exactly as before.
"""

from __future__ import annotations

import subprocess
import sys

from ..base import Step, StepResult
from ..context import Context
from ..quarantine import QuarantineError, load_quarantine, partition_failures


class FullUnitStep(Step):
    name = "full_unit"
    description = "full pytest suite (gated on blast radius)"

    def should_run(self, ctx: Context) -> bool:
        # Low-blast PRs get a skip — docs/examples can't break runtime.
        return ctx.blast_radius != "low"

    def run(self, ctx: Context) -> StepResult:
        log_path = ctx.artifact_path("full-unit.log")

        # Mirror what we run by hand. Two ignores: integrations needs a
        # live server (covered in step 5), and test_event_loop is the
        # long-running soak — separate budget.
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--ignore=tests/integrations",
            "--ignore=tests/test_event_loop.py",
            "-q",
            "--no-header",
            # Don't stop on first failure — we want the full count for
            # the scorecard ("3 failed, 2080 passed" is more actionable
            # than "1 failed, ???? passed").
        ]
        proc = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, cwd=str(ctx.repo_root)
        )
        log_path.write_text((proc.stdout or "") + (proc.stderr or ""))

        # Pull the summary line: pytest ends with a line like
        # "==== 3 failed, 2080 passed, 17 skipped in 25.56s ===="
        summary_line = _last_summary_line(proc.stdout)

        if proc.returncode == 0:
            return StepResult(
                name=self.name,
                status="pass",
                summary=summary_line or "all tests passed",
                artifacts=[str(log_path)],
            )

        # Non-zero exit. Extract the per-test node ids so we can consult
        # the quarantine registry.
        failed_ids = _failed_node_ids(proc.stdout)

        # If pytest exited non-zero but we parsed NO per-test failures,
        # this is a collection/internal/xdist-level error we can't name —
        # never quarantine what we can't identify. Block on the raw exit.
        if not failed_ids:
            raw = _extract_failed_lines(proc.stdout)
            body = raw[:30] if raw else ["(no FAILED lines — see log)"]
            details = [
                "**pytest failed with no parseable test ids:**\n```",
                *body,
                "```",
            ]
            return StepResult(
                name=self.name,
                status="fail",
                summary=summary_line or f"pytest exited {proc.returncode}",
                details="\n".join(details),
                artifacts=[str(log_path)],
            )

        # Fail-safe: an unreadable registry means "no quarantine", i.e.
        # every failure blocks. A broken file can only make us stricter.
        registry_note = ""
        try:
            entries = load_quarantine()
        except QuarantineError as e:
            entries = []
            registry_note = (
                f"\n\n⚠️ quarantine registry unreadable — treating every "
                f"failure as blocking: {e}"
            )
        blocking, quarantined = partition_failures(failed_ids, entries)

        details = _render_details(blocking, quarantined, log_path) + registry_note

        if blocking:
            # Real failures present → still a red, regardless of any
            # quarantined flakes riding along.
            return StepResult(
                name=self.name,
                status="fail",
                summary=summary_line or f"pytest exited {proc.returncode}",
                details=details,
                artifacts=[str(log_path)],
            )

        # Every failure was a known flake → do NOT block, but say so
        # loudly so a quarantined red is never mistaken for a clean run.
        return StepResult(
            name=self.name,
            status="pass",
            summary=(
                f"{len(quarantined)} known-flaky test(s) failed "
                f"(quarantined, non-blocking) — {summary_line}"
                if summary_line
                else f"{len(quarantined)} known-flaky test(s) failed "
                f"(quarantined, non-blocking)"
            ),
            details=details,
            artifacts=[str(log_path)],
        )


def _render_details(blocking: list[str], quarantined: list[str], log_path) -> str:
    """Scorecard detail block — blocking failures first (the actionable
    part), then any quarantined flakes that rode along (visible, but
    flagged non-blocking)."""
    parts: list[str] = []
    if blocking:
        shown = blocking[:30]
        parts.append("**Failed tests (blocking):**\n```\n" + "\n".join(shown) + "\n```")
        if len(blocking) > 30:
            parts.append(f"…and {len(blocking) - 30} more — see {log_path}")
    if quarantined:
        shown = quarantined[:30]
        parts.append(
            "**Quarantined flakes that failed (non-blocking):**\n```\n"
            + "\n".join(shown)
            + "\n```"
        )
        if len(quarantined) > 30:
            parts.append(f"…and {len(quarantined) - 30} more — see {log_path}")
    return "\n\n".join(parts)


def _last_summary_line(stdout: str) -> str:
    """Pytest writes its overall summary as the very last non-empty
    line wrapped in '====' decorations. Return without decorations."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("=") and ("passed" in line or "failed" in line):
            return line.strip("= ").strip()
    return ""


def _extract_failed_lines(stdout: str) -> list[str]:
    """Return the lines pytest's short summary section labels FAILED."""
    out = []
    in_summary = False
    for line in (stdout or "").splitlines():
        if "short test summary" in line:
            in_summary = True
            continue
        if in_summary:
            if line.startswith("="):
                break
            if line.startswith("FAILED"):
                out.append(line)
    return out


def _failed_node_ids(stdout: str) -> list[str]:
    """Extract pytest node ids from the short-summary FAILED lines.

    A FAILED line looks like ``FAILED tests/foo.py::test_bar - AssertionError``
    or bare ``FAILED tests/foo.py::test_bar``. The node id is the token
    after ``FAILED `` and before the ``  - <message>`` separator pytest
    inserts (space-dash-space). Parametrized ids can contain spaces
    inside ``[...]``, so we split on the first ``" - "`` rather than on
    whitespace.
    """
    ids: list[str] = []
    for line in _extract_failed_lines(stdout):
        rest = line[len("FAILED ") :] if line.startswith("FAILED ") else line
        # Strip the trailing " - <error message>" if present.
        node_id = rest.split(" - ", 1)[0].strip()
        if node_id:
            ids.append(node_id)
    return ids
