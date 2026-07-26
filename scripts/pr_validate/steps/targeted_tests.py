# SPDX-License-Identifier: Apache-2.0
"""Step 3 — diff-aware targeted test selection with negative control.

Goal: run the tests most likely to catch a regression from this PR
without running the entire ~25s suite. Plus the negative-control move
we used in PR #187: any failure on the PR branch is re-checked on main
to filter pre-existing flakes (so we don't block merge on something
that was already broken).

Selection heuristic — deliberately simple, grep-able:

1. For each Python file in the diff, derive the candidate test file
   name(s):
   - ``vllm_mlx/foo.py`` → ``tests/test_foo.py``
   - ``vllm_mlx/bar/baz.py`` → ``tests/test_baz.py``
2. For each non-test Python file, also include any test file whose
   name contains the module's stem.
3. If the diff hits a test file directly, include it.
4. If the heuristic matches nothing (e.g. PR only touches docs), skip
   the step.

We don't import-graph trace because pytest's collection cost dominates
and the heuristic catches >90% of the cases that matter. The full unit
suite (step 4) covers the rest for medium/high blast PRs.

Negative control: when targeted tests fail on the PR branch, re-run
the same set on a fresh ``git worktree`` of ``main``. Tests that fail
on both → pre-existing → don't block. Tests that pass on main but fail
on PR → real regression → BLOCK.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

from .. import _nodeid_reporter
from .._pytest_summary import (
    last_summary_line,
    raw_session_records,
    render_fence_safe,
    report_log_node_ids,
    rerun_detected,
    session_completed,
    summary_node_ids,
)
from ..base import GATING_PYTEST_GUARD, Step, StepResult
from ..context import Context


class TargetedTestsStep(Step):
    name = "targeted_tests"
    description = "diff-aware tests + negative control"

    def should_run(self, ctx: Context) -> bool:
        # Skip if no python in the diff — nothing to target.
        return any(p.endswith(".py") for p in ctx.files_changed)

    def run(self, ctx: Context) -> StepResult:
        targets = _select_test_files(ctx)
        if not targets:
            return StepResult(
                name=self.name,
                status="skip",
                summary="diff has no python files mapping to tests",
            )

        # Cap the targeted set — if the diff is huge, prefer falling
        # through to step 4 (full unit) rather than re-implementing it
        # here. Otherwise we'd spend 30s collecting tests that step 4
        # will then re-run from scratch.
        if len(targets) > 25:
            return StepResult(
                name=self.name,
                status="skip",
                summary=(
                    f"too many test targets ({len(targets)}) — "
                    f"deferring to full_unit step"
                ),
            )

        ctx.run_log(f"running {len(targets)} targeted test file(s) on PR branch")

        # Run on the PR branch (current working tree should be the PR's
        # head — we don't enforce that here, but the caller setup should
        # ensure it).  TODO if we ever auto-checkout PRs: switch to the
        # PR head here too.
        pr_log = ctx.artifact_path("targeted-pr.log")
        pr_nodeids = ctx.artifact_path("targeted-pr-nodeids.tsv")
        pr = _run_pytest(targets, pr_log, ctx.repo_root, pr_nodeids)

        # A GATING run must never rerun. targeted_tests is the FIRST — and,
        # for a low-blast PR where full_unit is skipped, the ONLY — gating
        # step that runs candidate tests, so the RERUN backstop has to live
        # here too: the by-name GATING_PYTEST_GUARD is bypassable by a
        # conftest registering pytest-rerunfailures under an arbitrary name,
        # and if full_unit never runs, nothing else would catch the rerun
        # (codex #1222 r24). Any RERUN record → a real failure may have been
        # retried to green → block, regardless of the scraped FAILED set.
        if pr.reran:
            return StepResult(
                name=self.name,
                status="fail",
                summary=f"{pr.summary} (gating run reran a test)",
                details="⚠️ the targeted gating run reran at least one test — "
                "pytest-rerunfailures was active despite the by-name block, so "
                "it was smuggled in under a different name. A rerun can retry a "
                "real failure into a pass, so this run is not trusted; reruns "
                "must be OFF for the gate.",
                artifacts=[str(pr_log)],
            )

        # The gate-owned ``-m "not slow and not integration and not needle"``
        # filter can legitimately deselect EVERY test in the targeted files (a
        # PR touching only slow/integration/needle tests), which makes pytest
        # exit 5 ("no tests collected"). That is NOT a tamper signal — nothing
        # gate-relevant remained to run — so SKIP rather than block on the exit
        # code (codex #1222 r28, a regression from the r27 ``-m`` re-apply). A
        # no-collection run WITHOUT a deselection (an empty/renamed file, or a
        # candidate ``python_files`` matching nothing) has no "deselected" in
        # pytest's summary and no structured ERROR, so it stays BLOCKED by
        # ``_untrusted_run_reason`` below. Candidate ``addopts`` is neutralized
        # (``-o addopts=``) and the ``-m`` filter is gate-owned.
        #
        # BOUNDARY (codex #1222 r30/r31) — this SKIP is the in-process
        # collection-control residual (r27), NOT a closed hole, and the earlier
        # "full_unit runs the whole suite regardless" reassurance was WRONG on a
        # low-blast PR (full_unit is skipped, so targeted is the only
        # candidate-test gate). Two routes reach an all-deselected exit 5:
        #   (a) an author newly marks a *changed* test slow/integration/needle
        #       to move it out of this fast gate; and
        #   (b) a candidate ``conftest.py`` deselects from
        #       ``pytest_collection_modifyitems`` (``-o addopts=`` doesn't stop
        #       conftest loading), forging the "deselected" summary.
        # Neither is soundly closeable here. This gate DELIBERATELY cannot run
        # slow/integration/needle tests (they need real models / a live server —
        # the reason they're excluded), so it can't tell a broken new slow test
        # from a legitimately-slow passing one. codex's "block unless the base
        # proves the exclusion pre-existing" is BACKWARDS AND incomplete: a
        # *new* broken slow test is absent on base (so a "did base run it?"
        # check misses exactly the dangerous case) while an existing test
        # legitimately reclassified WOULD trip it (a false block of routine
        # maintenance) — and either attacker can instead deselect JUST the one
        # failing test (exit 0, not 5), sailing past any exit-5 rule. That
        # surgical exit-0 deselection is the fundamental r27 residual: no
        # in-process count/manifest floor closes it. The real defenses are the
        # separate slow-lane CI (``-m slow``) and diff review (which sees the
        # added marker or conftest hook). We therefore SKIP (never false-block
        # legitimate reclassification) but make it LOUD so a reviewer is
        # prompted to confirm the deselection is intentional.
        #
        # GATE (codex #1222 r34): grant this non-blocking skip ONLY on
        # ``all_deselected_clean`` — the reporter's structured proof of a benign
        # zero-selection run (exactly ``SESSIONFINISH 0 0`` with NO
        # FAILED/ERROR/RERUN). Exit 5 + "deselected" ALONE is not enough: a
        # candidate hook could normalize a run that actually FAILED (or was
        # truncated) to exit 5 with a "deselected" summary and steal the skip.
        # Requiring positive structured proof that nothing failed and nothing
        # ran closes that; if the reporter didn't run, the proof is absent →
        # skip denied → falls through to ``_untrusted_run_reason`` (blocks).
        if (
            pr.returncode == 5
            and "deselected" in pr.summary
            and pr.all_deselected_clean
        ):
            return StepResult(
                name=self.name,
                status="skip",
                summary=(
                    f"⚠ REVIEW: {pr.summary} — every targeted test was "
                    f"deselected by the slow/integration/needle filter. If this "
                    f"PR newly marked a changed test with one of those markers, "
                    f"its failures run only in the slow-lane CI, not here; "
                    f"confirm the reclassification is intentional."
                ),
                artifacts=[str(pr_log)],
            )

        # An empty scraped FAILED list is only a genuine PASS when the run
        # actually finished as an ORDINARY pass (exit 0) or fail (exit 1).
        # ``_run_pytest`` scrapes only ``FAILED`` summary lines, so a
        # collection error, a usage error, "no tests collected", or an early
        # ``pytest.exit`` would otherwise slip through as a false green with
        # an empty FAILED set — the same class of hole full_unit closes with
        # its exit-code / ERROR / session-completeness checks (codex #1222
        # r25). Gate on those signals BEFORE trusting an empty FAILED list.
        untrusted = _untrusted_run_reason(pr)
        if untrusted:
            return StepResult(
                name=self.name,
                status="fail",
                summary=f"{pr.summary} (untrusted targeted run)",
                details=(
                    f"⚠️ the targeted run cannot be trusted as green: {untrusted}. "
                    "An empty FAILED list is not accepted as a pass on an "
                    "abnormal/truncated run — blocking fail-safe."
                ),
                artifacts=[str(pr_log)],
            )

        if not pr.failed:
            return StepResult(
                name=self.name,
                status="pass",
                summary=f"{pr.summary} (in {len(targets)} target file(s))",
                artifacts=[str(pr_log)],
            )

        pr_failed = pr.failed

        # Failures on PR branch — run negative control on main.
        ctx.run_log(
            f"{len(pr_failed)} fail on PR branch — running same tests "
            f"on main to filter pre-existing flakes"
        )
        main_log = ctx.artifact_path("targeted-main.log")
        try:
            # Use the PR's exact base SHA, not the branch tip — main may
            # have advanced since the PR was opened, and a moving
            # negative-control would misclassify regressions caused by
            # newer main commits as PR-introduced.
            main_failed = _run_on_main(
                targets,
                main_log,
                ctx.repo_root,
                ctx.base_sha or ctx.base_branch,
            )
        except Exception as e:  # noqa: BLE001
            # Worktree setup failed — surface it but don't lose the PR
            # failures we already have. Treat as fail-safe (block).
            details = (
                f"**negative control unavailable** ({type(e).__name__}: {e}). "
                "Cannot distinguish regressions from pre-existing flakes — "
                "treating all PR failures as regressions.\n\n"
                f"```\n{_failed_block(pr_failed)}\n```"
            )
            return StepResult(
                name=self.name,
                status="fail",
                summary=f"{len(pr_failed)} fail (neg control unavailable)",
                details=details,
                artifacts=[str(pr_log)],
            )

        # Classify each PR failure: regression vs pre-existing.
        regressions = sorted(set(pr_failed) - set(main_failed))
        pre_existing = sorted(set(pr_failed) & set(main_failed))
        if not regressions:
            return StepResult(
                name=self.name,
                status="pass",
                summary=(
                    f"{len(pr_failed)} fail on PR — all also fail on main "
                    f"(pre-existing, not regressions)"
                ),
                details=(
                    "**Pre-existing failures (also fail on main, ignored):**\n```\n"
                    + _failed_block(pre_existing)
                    + "\n```"
                ),
                artifacts=[str(pr_log), str(main_log)],
            )

        details = [
            "**Regressions (fail on PR, pass on main):**",
            "```",
            _failed_block(regressions),
            "```",
        ]
        if pre_existing:
            details.append("\n**Pre-existing (also fail on main, not blocking):**\n```")
            details.append(_failed_block(pre_existing))
            details.append("```")
        return StepResult(
            name=self.name,
            status="fail",
            summary=f"{len(regressions)} regression(s), "
            f"{len(pre_existing)} pre-existing",
            details="\n".join(details),
            artifacts=[str(pr_log), str(main_log)],
        )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _select_test_files(ctx: Context) -> list[str]:
    """Return tests/ paths to run, deduped, sorted."""
    candidates: set[str] = set()
    tests_dir = ctx.repo_root / "tests"
    if not tests_dir.exists():
        return []

    for path in ctx.files_changed:
        if not path.endswith(".py"):
            continue
        # Direct hit on a test file.
        if path.startswith("tests/"):
            if (ctx.repo_root / path).exists():
                candidates.add(path)
            continue

        stem = Path(path).stem  # e.g. "scheduler"

        # Try the obvious test_<stem>.py first.
        direct = tests_dir / f"test_{stem}.py"
        if direct.exists():
            candidates.add(f"tests/test_{stem}.py")

        # Plus any test files whose name contains the stem (covers
        # e.g. test_prefix_cache_persistence.py for prefix_cache.py).
        for tf in tests_dir.glob("test_*.py"):
            if stem in tf.stem:
                candidates.add(f"tests/{tf.name}")

    return sorted(candidates)


# ---------------------------------------------------------------------------
# Pytest runner
# ---------------------------------------------------------------------------


# Use ``sys.executable`` so pytest runs in the *same* interpreter as
# pr_validate itself. The test_env_check step verifies pytest-asyncio
# (etc.) are importable in this Python; if we hardcoded ``python3.12``
# we could end up handing pytest to a sibling Python where the
# self-check never ran. Closes #185 — the prior ``python3.12`` literal
# meant a venv-installed pr_validate could collect 124 async-test
# failures because the system python3.12 didn't have the plugin.
_PYTEST_CMD = [
    sys.executable,
    "-m",
    "pytest",
    # Neutralize candidate-controlled config: ``-o addopts=`` drops the
    # pytest.ini / pyproject ``addopts`` wholesale, so a planted ``--deselect``
    # / weaponized ``-m`` filter / ``-x`` / ``-rN`` can't steer this GATING run
    # — a subset that omits the changed failing test would otherwise exit 0 and
    # false-pass (codex #1222 r26). Every option we rely on is spelled out
    # explicitly below (never inherited from addopts), and ``PYTEST_ADDOPTS``
    # is popped from the subprocess env in ``_run_pytest`` because it bites
    # THROUGH ``-o addopts=`` (that only overrides the ini file). Shared by the
    # PR run AND the negative control so both stay hermetic and symmetric — the
    # repo's own ``addopts`` is empty, so this is a no-op on the trusted base.
    "-o",
    "addopts=",
    # Re-apply the marker exclusion the repo's addopts carried (slow /
    # integration / needle need real models or a live server). ``-o addopts=``
    # above dropped it, so re-specify it HARDCODED — exactly as full_unit does
    # — so the selection can't be widened/narrowed by candidate config and
    # targeted keeps the same exclusion it inherited before the addopts was
    # neutralized (codex #1222 r26).
    "-m",
    "not slow and not integration and not needle",
    "-q",
    "--no-header",
    "--tb=no",  # we don't render tracebacks here; the artifact has them
    # Force color OFF and PIN the short summary to list FAILED + ERROR. The
    # negative control (`_run_on_main`) runs on the pre-plugin base and so
    # CANNOT use the structured reporter — it must scrape stdout, and that
    # scrape has to be a stable, ANSI-free summary regardless of an inherited
    # PY_COLORS or a future pytest default (mirrors full_unit, codex #1222 r6).
    "--color=no",
    "-rfE",
    # No early stop — a candidate ``-x`` / ``--maxfail`` must not truncate the
    # failure set the negative control compares (a command-line ``--maxfail=0``
    # overrides any ini one); an early stop would also trip the completeness
    # guard in ``_untrusted_run_reason``.
    "--maxfail=0",
    # Block pytest-rerunfailures so a candidate ``@pytest.mark.flaky`` on a
    # changed test can't rerun-and-pass a real failure past this gate — same
    # reason as full_unit (codex #1222 r21). Used by both _run_pytest (the
    # candidate run) and _run_on_main (the negative control).
    *GATING_PYTEST_GUARD,
]


class _PytestRun(NamedTuple):
    """Result of a targeted pytest run.

    ``failed`` is the authoritative FAILED node-id list: read from the
    structured reporter when it was injected, else scraped from stdout.
    ``reran`` / ``errored`` / ``complete`` / ``structured`` also come from the
    reporter; without it (``nodeid_log`` absent / plugin unimportable) they
    degrade to the trusting defaults ``False`` / ``False`` / ``True`` /
    ``False`` so a legacy no-reporter run behaves as before. ``returncode`` is
    always pytest's real exit code; ``structured`` records whether the reporter
    ran, so the exit-code / completeness / unaccounted-exit checks know when
    they may trust the reporter-derived fields.

    ``all_deselected_clean`` is the STRUCTURED proof of a benign zero-selection
    run — the reporter recorded exactly ``SESSIONFINISH 0 0`` (nothing was
    collected/run) with NO FAILED/ERROR/RERUN. It gates the exit-5
    all-deselected skip so a candidate can't force an exit-5-with-``deselected``
    summary over a run that actually FAILED and steal a non-blocking skip
    (codex #1222 r34). Without the reporter it is the trusting default False,
    so a fallback exit-5 is not granted the skip (fails safe → blocks)."""

    summary: str
    failed: list[str]
    reran: bool
    returncode: int
    errored: bool
    complete: bool
    structured: bool
    all_deselected_clean: bool = False


def _untrusted_run_reason(run: _PytestRun) -> str | None:
    """Why a targeted run can't be trusted as a clean pass/fail, or ``None``.

    An empty ``FAILED`` list only means "clean" when the run finished as an
    ordinary pass (exit 0) or fail (exit 1). Any OTHER exit — collection error
    (2), internal error (3), usage error (4), no tests collected (5) — or a
    structured ``ERROR`` record (a collection / setup / teardown failure, which
    NEVER appears as a ``FAILED`` line so it can't be negative-control
    filtered), or a truncated session (an early ``pytest.exit`` / crash before
    the session-finish record) would otherwise slip through as a false green
    with an empty FAILED set (codex #1222 r25). And an exit of 1 with NO
    structured FAILED/RERUN record accounting for it means the failure summary
    was suppressed (``-rN`` / a summary-blanking addopts) — the empty list is
    not trusted (codex #1222 r26). The reporter-derived checks apply only when
    ``run.structured`` (the reporter ran); otherwise the caller passed the
    trusting defaults and only the exit-code and exit-1-accounting checks are
    meaningful."""
    if run.returncode not in (0, 1):
        return (
            f"pytest exited {run.returncode} (not a plain pass=0 / fail=1 — a "
            "collection/usage/internal error, or no tests were collected)"
        )
    if run.errored:
        return (
            "the structured log recorded an ERROR (collection or "
            "setup/teardown failure), which never scrapes as a FAILED line"
        )
    if not run.complete:
        return (
            "pytest did not run to completion (an early pytest.exit / crash "
            "before the session-finish record) — a truncated run is not green"
        )
    if run.returncode == 1 and not run.failed and not run.reran:
        # Exit 1 means tests were collected and some failed/errored — there is
        # no green exit-1. An empty failure list therefore can't be trusted:
        # in the STRUCTURED path a suppressed summary (``-rN`` / a
        # summary-blanking addopts) blanks the scrape while the reporter's
        # FAILED record would normally account for it; in the NO-REPORTER
        # FALLBACK path a setup/teardown ERROR exits 1 yet scrapes NO ``FAILED``
        # line at all (it prints ``ERROR``), so ``run.errored`` is the trusting
        # default False and the failure would slip through as a false green
        # (codex #1222 r33). Either way, an unaccounted exit 1 is untrusted.
        return (
            "pytest exited 1 but no FAILED/RERUN record (or scraped FAILED "
            "line) accounts for it — the failure summary was suppressed or a "
            "setup/teardown ERROR produced no FAILED line; an empty FAILED "
            "list is not accepted as green"
        )
    return None


def _run_pytest(
    targets: list[str], log_path: Path, cwd: Path, nodeid_log: Path | None = None
) -> _PytestRun:
    """Run pytest against ``targets``. Returns a ``_PytestRun`` — the one-line
    summary, scraped FAILED node IDs, and the structured-reporter signals
    (``reran`` / ``errored`` / ``complete``) plus the real exit code. A gating
    run must never rerun; and an empty FAILED list is only trusted as a pass
    when the exit code + reporter signals confirm an ordinary complete run
    (see ``_untrusted_run_reason``).

    When ``nodeid_log`` is given and the reporter plugin is importable, the
    run emits the structured RERUN / ERROR / SESSIONFINISH records those
    signals read from. ``cwd`` (the PR head worktree) must contain the
    reporter module — it does, since this PR adds it; the negative-control run
    on the protected base does NOT inject it (the base predates the plugin)."""
    cmd = [*_PYTEST_CMD, *targets]
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    inject = nodeid_log is not None and _nodeid_reporter.available()
    if inject:
        plugin_args, env = _nodeid_reporter.build_invocation(
            nodeid_log, cwd, base_env=env
        )
        cmd += plugin_args
        # Start-sentinel so an empty log is distinguishable from "plugin
        # never ran" (mirrors full_unit); the signals below read it back.
        nodeid_log.write_text("", encoding="utf-8")
    proc = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
    )
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""))
    summary = last_summary_line(proc.stdout) or f"exit {proc.returncode}"
    # Reporter-derived signals. Without the reporter we can't prove
    # completeness / distinguish ERROR from a clean run, so degrade to the
    # trusting defaults (structured=False, reran/errored=False, complete=True)
    # and fall back to scraping stdout for FAILED — the exit-code check in
    # _untrusted_run_reason still applies either way.
    structured = inject and nodeid_log.exists()
    if structured:
        reran = rerun_detected(nodeid_log)
        errored = bool(report_log_node_ids(nodeid_log, "ERROR"))
        complete = session_completed(nodeid_log)
        # Take FAILED from the reporter's records, NOT scraped stdout: a
        # candidate that suppresses the short summary (``-rN`` / a
        # summary-blanking addopts, though addopts is already neutralized)
        # would blank the scrape and false-pass, but the structured FAILED
        # record is written per failing outcome and can't be turned off from
        # candidate config (codex #1222 r26).
        failed = report_log_node_ids(nodeid_log, "FAILED")
        # Structured proof of a BENIGN zero-selection run: exactly one
        # ``SESSIONFINISH 0 0`` record (nothing collected/run to completion)
        # AND no failing outcomes. This gates the exit-5 all-deselected skip so
        # a forced exit-5-with-``deselected`` summary over a run that actually
        # failed can't steal a non-blocking skip (codex #1222 r34).
        all_deselected_clean = (
            raw_session_records(nodeid_log) == ["0 0"]
            and not failed
            and not errored
            and not reran
        )
    else:
        reran = errored = False
        complete = True
        failed = summary_node_ids(proc.stdout, "FAILED")
        # No reporter → we can't prove a clean zero-selection; the exit-5 skip
        # is therefore denied on the fallback path (fails safe → blocks).
        all_deselected_clean = False
    return _PytestRun(
        summary=summary,
        failed=failed,
        reran=reran,
        returncode=proc.returncode,
        errored=errored,
        complete=complete,
        structured=structured,
        all_deselected_clean=all_deselected_clean,
    )


def _run_on_main(
    targets: list[str], log_path: Path, repo_root: Path, base_ref: str
) -> list[str]:
    """Run the same targets on a fresh worktree of ``base_ref``.

    ``base_ref`` should be the PR's base SHA (preferred) — using a
    branch name lets ``main`` move under us. Falls back to a branch
    name if the SHA isn't available.

    Uses a temp directory; the caller's repo root + working tree stay
    untouched. We re-resolve targets relative to the worktree (some
    files may not exist on main, e.g. tests added by this PR — those
    are dropped from the negative control).
    """
    tmp = Path(tempfile.mkdtemp(prefix="pr_validate_main_"))
    try:
        # Create a worktree pointing at base_ref (sha-pinned when given).
        subprocess.run(  # noqa: S603
            ["git", "worktree", "add", "--detach", str(tmp), base_ref],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        try:
            existing_targets = [t for t in targets if (tmp / t).exists()]
            if not existing_targets:
                # Every target test file is new in this PR. By
                # construction they don't exist on main → no failures
                # to filter; treat as no pre-existing fails.
                log_path.write_text(
                    "(no targeted test files exist on main — all are new in PR)\n"
                )
                return []

            # Strip PYTEST_ADDOPTS from the base subprocess env too — the
            # PR-side run pops it (it bites THROUGH ``-o addopts=``), so the
            # negative control MUST do the same or the comparison is asymmetric:
            # an inherited ``PYTEST_ADDOPTS=--runxfail`` (etc.) applied only on
            # base could make base report a node as FAILED that the PR side does
            # not, so a real PR regression gets misclassified as pre-existing
            # and waived — a FALSE GREEN (codex #1222 r34). ``_PYTEST_CMD``
            # already makes the two runs command-symmetric; this makes them
            # env-symmetric.
            base_env = dict(os.environ)
            base_env.pop("PYTEST_ADDOPTS", None)
            proc = subprocess.run(  # noqa: S603
                [*_PYTEST_CMD, *existing_targets],
                capture_output=True,
                text=True,
                cwd=str(tmp),
                env=base_env,
            )
            log_path.write_text((proc.stdout or "") + (proc.stderr or ""))
            # Space-preserving scrape (NOT a ``\S+`` grab): the PR side records
            # canonical ``report.nodeid`` via the reporter, so a pre-existing
            # failure whose param id contains spaces — ``test_x[param with
            # spaces]`` — must be parsed whole here too, or the truncated base
            # id wouldn't match the PR id and the pre-existing failure would be
            # misclassified as a PR-only regression (codex #1222 r29).
            #
            # RESIDUAL (codex #1222 r30, fail-safe): ``summary_node_ids`` is
            # exact for normal ids but IRREDUCIBLY ambiguous for an id that
            # itself contains ``" - "`` with unbalanced brackets
            # (``test_x[a] - b]``) — the same wall that motivated the structured
            # reporter plugin (see its module docstring; codex r4/r5/r6/r9/r10).
            # The base predates that plugin, so it cannot get canonical ids here,
            # and injecting the PR checkout's plugin would drag the PR's library
            # onto the base run's ``PYTHONPATH`` — contaminating the negative
            # control (base TEST files run against PR code), a worse and unsound
            # failure than the one it fixes. So a ``" - "``-laden base id may
            # truncate and NOT match the PR-side canonical id — which fails
            # CLOSED: the pre-existing failure is treated as a regression and
            # BLOCKS (never a false green — a real regression is never masked;
            # see the fail-safe test). We accept a rare false block of a
            # pathological pre-existing id over an unsound negative control.
            return summary_node_ids(proc.stdout, "FAILED")
        finally:
            # Remove the worktree even if pytest crashed.
            subprocess.run(  # noqa: S603
                ["git", "worktree", "remove", "--force", str(tmp)],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
            )
    finally:
        # In case `git worktree remove` failed, nuke the dir.
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def _failed_block(items: list[str]) -> str:
    # Node ids reach here from the reporter's structured records (r26), which
    # capture pytest's exact ``report.nodeid`` — and a hostile
    # ``pytest_make_parametrize_id`` can embed a newline + triple backticks in
    # that id (r23). Emitted verbatim into the scorecard's ``` fence, such an id
    # could close the fence and spoof review content, the same sink r24 hardened
    # for full_unit/flake_tracking — reachable in targeted only once it switched
    # to canonical ids. ``render_fence_safe`` renders each id as a single quoted
    # line so no embedded newline can begin a fence-closer (codex #1222 r30).
    return "\n".join(render_fence_safe(i) for i in items) if items else "(none)"
