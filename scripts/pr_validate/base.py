# SPDX-License-Identifier: Apache-2.0
"""Base classes for pipeline steps.

Each step is a small class with three contract points: a name, a
``should_run(ctx)`` predicate (lets us gate expensive steps on blast
radius), and a ``run(ctx) -> StepResult``. Keep step modules
self-contained — the runner doesn't know what each does, only that it
returns a uniform result object the scorecard can render.

We intentionally avoid a plugin registry / entrypoints mechanism. The
runner explicitly imports + orders steps so the pipeline is grep-able
and review-time obvious. Adding a step = one import + one list entry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .context import Context


# Status meanings:
#   pass    — step ran, nothing wrong → does not block merge
#   fail    — step ran, found a real problem → BLOCKS merge (strict mode)
#   skip    — step decided not to run (gating) → does not block, neutral
#   error   — step crashed before it could decide → BLOCKS merge (treat
#             unknown like failure; never let a broken validator silently
#             approve a PR)
StepStatus = Literal["pass", "fail", "skip", "error"]


# Args every GATING pytest subprocess must carry so a candidate can't use
# an autoloaded plugin to fake a green run (codex #1222 r20/r21). This PR
# installs pytest-rerunfailures (for the advisory flake_tracking step), and
# it autoloads by entry point — so WITHOUT this block a PR could tag a
# genuinely failing test ``@pytest.mark.flaky(reruns=N)`` and have ANY gating
# pytest run (targeted_tests, full_unit) re-run and pass it, bypassing the
# quarantine registry. ``-o addopts=`` / PYTEST_ADDOPTS-stripping can't stop
# an autoloaded plugin, and disabling autoload globally would also drop
# plugins the gates rely on (pytest-asyncio, etc.), so we block it by name on
# every gating invocation. We block BOTH the entry-point name
# (``rerunfailures``) and the module name (``pytest_rerunfailures``), since a
# conftest can register it under either.
#
# By-name blocking is the cheap first line, NOT the whole defense: a hostile
# conftest can register the plugin under an ARBITRARY name that no ``-p
# no:<name>`` can enumerate (``pluginmanager.register(mod, name="x")`` + a
# ``@pytest.mark.flaky`` marker still reruns — verified, codex #1222 r23).
# The NAME-INDEPENDENT backstop lives in ``full_unit``: the _nodeid_reporter
# logs a RERUN record on any rerun OUTCOME, and full_unit blocks if one
# appears (a gating run must never rerun). This constant just stops the
# common/accidental autoload path without paying the reporter round-trip.
#
# Harmless no-op when the plugin isn't installed. flake_tracking deliberately
# does NOT use this — its post-gating advisory re-run WANTS reruns. A shared
# constant (not per-step literals) so a new gating step can't silently forget
# it; pinned by ``test_every_gating_pytest_cmd_blocks_rerunfailures``.
GATING_PYTEST_GUARD: tuple[str, ...] = (
    "-p",
    "no:rerunfailures",
    "-p",
    "no:pytest_rerunfailures",
)


@dataclass
class StepResult:
    """Per-step output the scorecard renders.

    ``summary`` is a one-liner shown in the verdict table; ``details``
    is the multiline markdown shown when the step failed (or when the
    user asked for ``--verbose``). ``artifacts`` are paths to log files
    in the working dir — preserved so the user can inspect after.
    """

    name: str
    status: StepStatus
    summary: str
    details: str = ""
    duration_seconds: float = 0.0
    artifacts: list[str] = field(default_factory=list)
    # Free-form findings list; populated by the adversarial-review step
    # so the scorecard can render them inline. Each is one short string.
    findings: list[str] = field(default_factory=list)


class Step:
    """Pipeline step. Subclasses override ``run``; everything else is
    handled by the runner."""

    name: str = "unnamed"
    description: str = ""
    # If True, an ``error`` from this step still lets later steps run
    # (best-effort gating). Default is False — most steps are blocking.
    continue_on_error: bool = False

    def should_run(self, ctx: Context) -> bool:
        """Return False to skip — e.g. blast radius too low. Default
        runs every time."""
        return True

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError

    def execute(self, ctx: Context) -> StepResult:
        """Wrapper called by the runner. Times the step, catches
        exceptions, normalizes errors so a step bug never silently
        passes a bad PR."""
        if not self.should_run(ctx):
            return StepResult(
                name=self.name,
                status="skip",
                summary="skipped (gating predicate returned False)",
                duration_seconds=0.0,
            )
        t0 = time.monotonic()
        try:
            result = self.run(ctx)
        except Exception as e:  # noqa: BLE001 — we want to catch ANY crash
            import traceback

            return StepResult(
                name=self.name,
                status="error",
                summary=f"step crashed: {type(e).__name__}: {e}",
                details=f"```\n{traceback.format_exc()}\n```",
                duration_seconds=time.monotonic() - t0,
            )
        result.duration_seconds = time.monotonic() - t0
        if result.name == "unnamed":
            result.name = self.name
        return result
