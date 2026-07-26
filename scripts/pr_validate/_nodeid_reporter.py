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
  * setup/teardown fail  → ``ERROR``  (mirrors pytest's own -rfE split)
  * collection failure   → ``ERROR``
  * call-phase pass      → ``PASSED`` — ONLY when
    ``$PR_VALIDATE_NODEID_LOG_PASSES=1``. The rerun classifier needs
    positive PASSED signals; the full suite (14k passes) does not, and
    logging them all would bloat the gate's artifact for no benefit.

Reads/skips are never logged: a test absent from the log positively
"didn't fail (or pass, if passes are logged)", which is exactly the
"inconclusive, not a flake" signal ``flake_tracking`` wants.

This is NOT tamper-proof against a hostile in-process author (an
``atexit``/``conftest`` could rewrite the file) — nothing in-process is,
junit-xml included; that residual is scoped in ``_pytest_summary`` and
defended by the other gates. What it DOES buy is exact, unambiguous node
ids for every realistic run, closing the parse-ambiguity class for good.
"""

from __future__ import annotations

import importlib.util
import os

_ENV_LOG = "PR_VALIDATE_NODEID_LOG"
_ENV_PASSES = "PR_VALIDATE_NODEID_LOG_PASSES"

PLUGIN_MODULE = "scripts.pr_validate._nodeid_reporter"


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


def _append(label: str, nodeid: str) -> None:
    path = os.environ.get(_ENV_LOG)
    if not path:
        return
    # Append so setup/call/teardown reports for the same test each add
    # their own line; open per-call to stay robust to xdist workers each
    # writing (append is atomic for the small line sizes here).
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{label}\t{nodeid}\n")


def pytest_runtest_logreport(report) -> None:  # noqa: ANN001 — pytest hook
    if report.when == "call":
        if report.outcome == "failed":
            _append("FAILED", report.nodeid)
        elif report.outcome == "passed" and os.environ.get(_ENV_PASSES) == "1":
            _append("PASSED", report.nodeid)
    elif report.failed:
        # A failure in setup or teardown — pytest reports these as ERROR.
        _append("ERROR", report.nodeid)


def pytest_collectreport(report) -> None:  # noqa: ANN001 — pytest hook
    # A collection failure (import error, bad conftest) — ERROR, and the
    # node id is the module/dir that failed to collect.
    if report.failed:
        _append("ERROR", report.nodeid)
