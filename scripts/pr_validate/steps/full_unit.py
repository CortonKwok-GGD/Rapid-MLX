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

from .._pytest_summary import summary_node_ids
from ..base import Step, StepResult
from ..context import Context
from ..quarantine import (
    QuarantineError,
    load_quarantine_from_ref,
    partition_failures,
)

# pytest exit code 1 is the ordinary "some tests failed" outcome — the
# only one where the FAILED list fully explains the red. 2 (interrupted),
# 3 (internal error), 4 (usage), 5 (no tests) all mean something the
# FAILED list can't account for, so the quarantine downgrade must NOT
# apply. See https://docs.pytest.org/en/stable/reference/exit-codes.html
_PYTEST_TESTS_FAILED = 1


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
            # Force color OFF so the summary/label parsing can't be broken
            # by ANSI escapes when color is forced via PY_COLORS=1 or an
            # inherited --color=yes (codex #1222 r6). --color=no overrides
            # PY_COLORS on the command line.
            "--color=no",
            # PIN the short-summary contents: always list FAILED and ERROR
            # node ids regardless of any future default change. The
            # quarantine downgrade's soundness rests on being able to SEE
            # every error (a fixture/collection error must never be
            # silently absent from the summary and mistaken for "only
            # ordinary test failures" — codex #1222 r3). -rfE is explicit
            # so the gate never depends on pytest's implicit default.
            "-rfE",
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

        # Non-zero exit. Extract the per-test node ids (FAILED) and any
        # ERROR entries (collection / fixture failures) separately.
        failed_ids = summary_node_ids(proc.stdout, "FAILED")
        error_ids = summary_node_ids(proc.stdout, "ERROR")

        # The quarantine downgrade is only sound when the ONLY thing that
        # went wrong is ordinary, named test failures we can reason about:
        #   * exit code exactly 1 (not interrupted / internal / usage), and
        #   * at least one parseable FAILED id, and
        #   * NO ERROR entries (an errored test isn't a quarantinable flake
        #     — the suite couldn't even complete it).
        # Anything else blocks unconditionally — we never quarantine what
        # we can't fully account for (codex #1222).
        only_test_failures = (
            proc.returncode == _PYTEST_TESTS_FAILED
            and bool(failed_ids)
            and not error_ids
        )
        if not only_test_failures:
            return StepResult(
                name=self.name,
                status="fail",
                summary=summary_line or f"pytest exited {proc.returncode}",
                details=_render_unaccounted(
                    proc.returncode, failed_ids, error_ids, log_path
                ),
                artifacts=[str(log_path)],
            )

        # Read the registry from the PROTECTED base revision, not the
        # candidate checkout — a PR must not be able to quarantine its own
        # failing tests (codex #1222). Fail-safe: any read error → empty
        # quarantine → every failure blocks (stricter, never looser).
        registry_note = ""
        try:
            entries = load_quarantine_from_ref(
                ctx.base_sha or ctx.base_branch, ctx.repo_root
            )
        except QuarantineError as e:
            entries = []
            registry_note = (
                f"\n\n⚠️ base quarantine registry unreadable — treating "
                f"every failure as blocking: {e}"
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


def _render_unaccounted(
    returncode: int,
    failed_ids: list[str],
    error_ids: list[str],
    log_path,
) -> str:
    """Detail block for a red the quarantine logic must NOT relax — an
    abnormal exit code, an ERROR entry, or no parseable failures. Shows
    whatever we could name plus why it's blocking."""
    parts: list[str] = [
        f"**Blocking — not eligible for quarantine** (pytest exit "
        f"{returncode}; quarantine only applies to exit "
        f"{_PYTEST_TESTS_FAILED} with no errors)."
    ]
    if failed_ids:
        parts.append("**FAILED:**\n```\n" + "\n".join(failed_ids[:30]) + "\n```")
    if error_ids:
        parts.append("**ERROR:**\n```\n" + "\n".join(error_ids[:30]) + "\n```")
    if not failed_ids and not error_ids:
        parts.append(f"(no parseable FAILED/ERROR node ids — see {log_path})")
    return "\n\n".join(parts)


def _last_summary_line(stdout: str) -> str:
    """Pytest writes its overall summary as the very last non-empty
    line wrapped in '====' decorations. Return without decorations."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("=") and ("passed" in line or "failed" in line):
            return line.strip("= ").strip()
    return ""
