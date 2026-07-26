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

import os
import subprocess
import sys

from .. import _nodeid_reporter
from .._pytest_summary import (
    last_summary_line,
    render_fence_safe,
    report_log_node_ids,
    summary_node_ids,
)
from .._pytest_summary import (
    rerun_detected as _rerun_detected,
)
from .._pytest_summary import (
    session_completed as _session_completed,
)
from ..base import GATING_PYTEST_GUARD, Step, StepResult
from ..context import Context
from ..quarantine import (
    QuarantineError,
    effective_quarantine,
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
            # Neutralize candidate-controlled config: -o addopts= drops the
            # pytest.ini / pyproject addopts wholesale so a planted -x /
            # --maxfail / --stepwise / -p no:<reporter> / weaponized -m
            # filter can't steer the gate (--maxfail=0 alone does NOT stop
            # --stepwise — codex #1222 r14). We then RE-SPECIFY the marker
            # filter we actually want below; PYTEST_ADDOPTS is cleared from
            # the subprocess env for the same reason.
            "-o",
            "addopts=",
            # Re-apply the same exclusions the repo's addopts carried
            # (slow / integration / needle need real models or a live
            # server). Hardcoded here so the SELECTION can't be widened or
            # narrowed by candidate config.
            "-m",
            "not slow and not integration and not needle",
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
            # Don't stop on first failure — we want the full count for the
            # scorecard, and, critically, the quarantine downgrade is only
            # SOUND on a COMPLETE run. --maxfail=0 (= no limit) overrides any
            # -x / --maxfail a candidate planted in pytest.ini / pyproject
            # addopts (a command-line maxfail wins over the prepended ini
            # one); without it, a candidate could stop the suite after one
            # quarantined failure and have the partial, regression-hiding run
            # accepted as green (codex #1222 r13).
            "--maxfail=0",
            # Block pytest-rerunfailures in this GATING run so a candidate
            # `@pytest.mark.flaky(reruns=N)` can't rerun-and-pass a real
            # failure without consulting the quarantine (codex #1222 r20).
            # Shared with targeted_tests via base.GATING_PYTEST_GUARD so no
            # gating step can silently forget it (codex #1222 r21). See the
            # constant's docstring for the full rationale.
            *GATING_PYTEST_GUARD,
        ]

        # Emit a STRUCTURED node-id log via the _nodeid_reporter plugin so
        # the quarantine partition uses pytest's exact ``report.nodeid``
        # instead of scraping the ambiguous terminal summary (codex #1222
        # r4→r10). A stale log from a prior run would poison the read, so
        # start clean; fall back to summary parsing if the plugin can't be
        # loaded (never pass an unloadable ``-p`` — that's a usage error).
        nodeid_log = ctx.artifact_path("full-unit-nodeids.tsv")
        if nodeid_log.exists():
            nodeid_log.unlink()
        # Build the subprocess env with PYTEST_ADDOPTS stripped — it bites
        # THROUGH ``-o addopts=`` (that only overrides the ini file), so a
        # candidate env var could still inject --stepwise / -x otherwise
        # (codex #1222 r14). Done regardless of plugin availability.
        run_env = dict(os.environ)
        run_env.pop("PYTEST_ADDOPTS", None)
        # ``structured`` means the reporter was INJECTED, decided HERE — not
        # inferred later from ``nodeid_log.exists()``. The reporter creates
        # the file only when it appends the first outcome / SESSIONFINISH, so
        # a hard ``os._exit(0)`` before the first test leaves NO file, which
        # ``exists()`` would misread as "plugin unavailable" and trust the
        # zero exit on a fully-truncated run (codex #1222 r17). We instead
        # pre-create the log as a START SENTINEL right after injecting, so its
        # existence is decoupled from whether any record was written: an
        # injected run with an empty/record-less log → no completion record →
        # withhold, never "unavailable → trust exit code".
        structured = _nodeid_reporter.available()
        if structured:
            plugin_args, run_env = _nodeid_reporter.build_invocation(
                nodeid_log, ctx.repo_root, base_env=run_env
            )
            cmd += plugin_args
            nodeid_log.write_text("", encoding="utf-8")

        proc = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, cwd=str(ctx.repo_root), env=run_env
        )
        log_path.write_text((proc.stdout or "") + (proc.stderr or ""))

        # Pull the summary line: pytest ends with a line like
        # "==== 3 failed, 2080 passed, 17 skipped in 25.56s ===="
        summary_line = last_summary_line(proc.stdout)

        # A GATING run must NEVER rerun a test. GATING_PYTEST_GUARD disables
        # pytest-rerunfailures by name, but name-based ``-p no:<name>``
        # blocking is bypassable — a conftest can re-register the plugin
        # under an arbitrary name (codex #1222 r23). The reporter's RERUN
        # record is name-independent: any RERUN means the plugin ran anyway
        # and a real failure may have been retried into a pass, so the run's
        # exit code and FAILED set are both untrustworthy. Block regardless
        # of exit code, before either the clean-exit or the failure path
        # can trust this run.
        if structured and _rerun_detected(nodeid_log):
            return StepResult(
                name=self.name,
                status="fail",
                summary=summary_line or "gating run reran a test",
                details="⚠️ the gating pytest run reran at least one test — "
                "pytest-rerunfailures was active despite the by-name block, so "
                "it was smuggled in under a different name. A rerun can retry a "
                "real failure into a pass, so this run is not trusted; reruns "
                "must be OFF for the gate.",
                artifacts=[str(log_path)],
            )

        if proc.returncode == 0:
            # A clean exit is trustworthy only on a COMPLETE run that recorded
            # NO failures. When the reporter ran, the STRUCTURED log — not the
            # exit code — is the source of truth (the exit code is candidate-
            # influenceable and never trusted over the records):
            #
            #   * Completeness: os._exit(0) can terminate pytest with code 0
            #     after skipping the rest of the suite, hiding a regression in
            #     the un-run tail (codex #1222 r16 B1). Require the completion
            #     record — an absent record / a ran<collected gap means the
            #     session was truncated.
            #   * No masked failures: a conftest / exit-code plugin can
            #     normalize a FAILING run to code 0 (e.g. pytest_sessionfinish
            #     setting session.exitstatus = 0), so a clean exit whose
            #     structured log carries FAILED / ERROR ids is tampering or a
            #     bug, not green (codex #1222 r21). Block and surface those
            #     ids; grant NO quarantine downgrade — a forged clean exit is
            #     exactly the signal we must not reward. A legitimately green
            #     run records zero FAILED/ERROR (xfail logs as skipped;
            #     xfail_strict xpass exits non-zero), so this never fires on a
            #     real pass.
            #
            # If the plugin couldn't be injected at all (no structured log),
            # there's no signal to check, so fall back to the exit code (the
            # degraded, non-candidate path; the plugin is always available in
            # production).
            if structured:
                if not _session_completed(nodeid_log):
                    return StepResult(
                        name=self.name,
                        status="fail",
                        summary=summary_line or "pytest exited 0 but did not complete",
                        details="⚠️ pytest exited 0 but the session did not run to "
                        "completion — no session-finish record (os._exit / crash / "
                        "SIGKILL), or fewer tests ran than were collected. A later "
                        "regression may not have run, so a truncated run is not "
                        "accepted as green.",
                        artifacts=[str(log_path)],
                    )
                clean_failed = report_log_node_ids(nodeid_log, "FAILED")
                clean_errors = report_log_node_ids(nodeid_log, "ERROR")
                if clean_failed or clean_errors:
                    return StepResult(
                        name=self.name,
                        status="fail",
                        summary=summary_line or "pytest exited 0 but recorded failures",
                        details="⚠️ pytest exited 0 but the structured log recorded "
                        f"{len(clean_failed)} FAILED / {len(clean_errors)} ERROR "
                        "test(s) — the exit code was normalized away from the real "
                        "outcome (a conftest / plugin can force exit 0). A forged "
                        "clean exit is not accepted as green and grants no "
                        "quarantine downgrade.\n\n"
                        + _render_details(clean_failed + clean_errors, [], log_path),
                        artifacts=[str(log_path)],
                    )
            return StepResult(
                name=self.name,
                status="pass",
                summary=summary_line or "all tests passed",
                artifacts=[str(log_path)],
            )

        # Non-zero exit. Extract the per-test node ids (FAILED) and any
        # ERROR entries (collection / fixture failures) separately. The
        # STRUCTURED plugin log carries pytest's exact node ids; the
        # terminal-summary fallback is ambiguous (its id/message split has
        # documented edge cases — ``test_x[a] - b]`` …). The quarantine
        # downgrade below is only granted on the exact source (codex r13).
        if structured:
            failed_ids = report_log_node_ids(nodeid_log, "FAILED")
            error_ids = report_log_node_ids(nodeid_log, "ERROR")
        else:
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

        # The quarantine downgrade turns a red green, so it MUST rest on
        # pytest's exact ``report.nodeid`` — never the ambiguous summary
        # fallback, whose id/message split could truncate a param id and
        # wrongly match a family quarantine entry, waiving a real failure.
        # If the structured log is absent — the plugin couldn't load, or a
        # candidate disabled it via pytest plugin config — grant NO
        # downgrade: every failure blocks, with the fallback ids used only
        # to REPORT which tests failed. This keeps the ambiguous parser out
        # of the gating path entirely (it survives only in the advisory
        # flake_tracking step, where a mis-split is a mislabeled advisory,
        # not a wrongly-waived gate) (codex #1222 r13).
        if not structured:
            return StepResult(
                name=self.name,
                status="fail",
                summary=summary_line or f"pytest exited {proc.returncode}",
                details=_render_details(failed_ids, [], log_path)
                + "\n\n⚠️ structured node-id log absent (reporter plugin did "
                "not run) — quarantine downgrade withheld; every failure "
                "blocks.",
                artifacts=[str(log_path)],
            )

        # A quarantine downgrade is only sound on a COMPLETE run. Prove
        # completeness STRUCTURALLY from the reporter's session-finish
        # record (codex #1222 r15), not by scraping stdout: the record is
        # ABSENT when the process died mid-suite (os._exit / crash / SIGKILL
        # — pytest_sessionfinish never fired), and shows ran < collected
        # when the session stopped early (-x / --maxfail / --stepwise, which
        # a conftest can still set programmatically even with addopts
        # cleared). Either way a later real regression may not have run, so
        # the FAILED set can't account for the whole suite — withhold the
        # downgrade and block. This replaces a fragile stdout heuristic that
        # false-positived when "stopping after" / "Interrupted:" appeared in
        # a quarantined test's own traceback (codex #1222 r15 B2).
        if not _session_completed(nodeid_log):
            return StepResult(
                name=self.name,
                status="fail",
                summary=summary_line or "pytest did not run to completion",
                details=_render_details(failed_ids, [], log_path)
                + "\n\n⚠️ the pytest session did not run to completion — no "
                "session-finish record (os._exit / crash / SIGKILL), or fewer "
                "tests ran than were collected (--stepwise / -x / --maxfail). "
                "A later regression may not have run, so the quarantine "
                "downgrade is withheld; every failure blocks.",
                artifacts=[str(log_path)],
            )

        # Read the registry from the PROTECTED base revision, not the
        # candidate checkout — a PR must not be able to quarantine its own
        # failing tests (codex #1222). Pin to the IMMUTABLE base SHA
        # (``baseRefOid``, resolved at fetch before any candidate code
        # ran); never fall back to the mutable ``base_branch`` ref, which a
        # candidate test could rewrite (``git branch -f main …``) between
        # the pytest run above and this load, then serve its own allowlist
        # (codex #1222 r10). No SHA → fail CLOSED to an empty quarantine.
        registry_note = ""
        if not ctx.base_sha:
            entries = []
            registry_note = (
                "\n\n⚠️ no immutable base SHA available — treating every "
                "failure as blocking (quarantine requires a protected base "
                "commit; the mutable branch ref is not trusted)."
            )
        else:
            try:
                base_entries = load_quarantine_from_ref(ctx.base_sha, ctx.repo_root)
            except QuarantineError as e:
                entries = []
                # The exception message can echo candidate-controlled
                # registry text (a bad node id / reason from the PR's own
                # quarantine.yaml), so render it as a single JSON-quoted line
                # — a raw newline/backtick would otherwise spoof a scorecard
                # heading or fence out of this note (codex #1222 r25).
                registry_note = (
                    f"\n\n⚠️ base quarantine registry unreadable — treating "
                    f"every failure as blocking: {render_fence_safe(str(e))}"
                )
            else:
                # Intersect base with the CANDIDATE registry so a
                # de-quarantine (removal) in THIS PR is honored: an entry the
                # PR removed drops out of the effective set, so a still-red
                # de-quarantined test blocks instead of riding on the stale
                # base list (codex #1222 r21). Read the candidate from the
                # pinned PR HEAD sha — symmetric with base, and immune to a
                # candidate test rewriting the working-tree file at runtime.
                # The intersection is ALWAYS ⊆ base, so a hostile candidate
                # registry can only TIGHTEN, never waive more than the base
                # approved (a PR that DELETES quarantine.yaml → candidate empty
                # → nothing waived, the correct strict de-quarantine-all).
                #
                # If the candidate revision can't be read/validated, fail
                # CLOSED to an EMPTY effective quarantine — NEVER back to the
                # base list (codex #1222 r22). A candidate CONTROLS its own
                # committed quarantine.yaml, so falling back to base on a
                # malformed/corrupt registry would let a PR keep a
                # base-quarantined but now-failing test green by simply
                # breaking its own registry. Empty → every failure blocks,
                # the same fail-closed posture as a missing base sha /
                # unreadable base registry.
                if not ctx.head_sha:
                    entries = []
                    registry_note = (
                        "\n\n⚠️ no PR head SHA — cannot confirm the candidate "
                        "registry; failing closed to an empty quarantine (every "
                        "failure blocks)."
                    )
                else:
                    try:
                        cand_entries = load_quarantine_from_ref(
                            ctx.head_sha, ctx.repo_root
                        )
                    except QuarantineError as e:
                        entries = []
                        # Single-line JSON render — the candidate CONTROLS
                        # this registry, so its error text is fully attacker-
                        # chosen and must not reach the scorecard raw (codex
                        # #1222 r25).
                        registry_note = (
                            f"\n\n⚠️ candidate quarantine registry unreadable — "
                            f"failing closed to an empty quarantine (every "
                            f"failure blocks): {render_fence_safe(str(e))}"
                        )
                    else:
                        entries = effective_quarantine(base_entries, cand_entries)
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
        shown = [render_fence_safe(i) for i in blocking[:30]]
        parts.append("**Failed tests (blocking):**\n```\n" + "\n".join(shown) + "\n```")
        if len(blocking) > 30:
            parts.append(f"…and {len(blocking) - 30} more — see {log_path}")
    if quarantined:
        shown = [render_fence_safe(i) for i in quarantined[:30]]
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
        shown = [render_fence_safe(i) for i in failed_ids[:30]]
        parts.append("**FAILED:**\n```\n" + "\n".join(shown) + "\n```")
    if error_ids:
        shown = [render_fence_safe(i) for i in error_ids[:30]]
        parts.append("**ERROR:**\n```\n" + "\n".join(shown) + "\n```")
    if not failed_ids and not error_ids:
        parts.append(f"(no parseable FAILED/ERROR node ids — see {log_path})")
    return "\n\n".join(parts)
