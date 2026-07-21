# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the #558 default-on gate comparator (``scripts/default_on_gate``).

The comparator's job is to REFUSE a GREEN verdict for any run that did not
actually validate the constrained path. Before the #558-PR5 codex fix a
two-arm-IDENTICAL failure was classified ``PARITY`` and a missing arm ``GONE`` /
``NEW`` — both slipped past a CLI that only blocked on ``REGRESSED*``, so a
crashed or absent run could show GREEN having validated nothing. These tests pin
the fix: a missing arm, or any hard-fail status (``fail``/``error``/``xpass``)
in either arm, is FATAL and turns the gate RED.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.default_on_gate import (
    CellResult,
    _fatal_reason,
    _verdict,
    cmd_compare,
    cmd_compare_modes,
)


class _Args:
    """Minimal argparse.Namespace stand-in for the cmd_* entrypoints."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _cell(nodeid: str, mode: str, status: str, *, dur: float = 1.0, leak: bool = False):
    return CellResult(
        nodeid=nodeid,
        family="qwen36",
        mode=mode,
        status=status,
        duration_s=dur,
        leak=leak,
    )


def _write_results(dir_path: Path, cells: list[CellResult]) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "results.json").write_text(
        json.dumps(
            {
                "ref_label": dir_path.name,
                "families": ["qwen36"],
                "port": 8123,
                "cells": [
                    {
                        "nodeid": c.nodeid,
                        "family": c.family,
                        "mode": c.mode,
                        "status": c.status,
                        "duration_s": c.duration_s,
                        "leak": c.leak,
                        "message": c.message,
                    }
                    for c in cells
                ],
            }
        )
    )
    return dir_path


# --------------------------------------------------------------------------- #
# _fatal_reason unit behavior
# --------------------------------------------------------------------------- #


def test_fatal_reason_missing_arm_is_fatal():
    ok = _cell("t::a", "on", "pass")
    assert _fatal_reason(None, ok) is not None  # base/off missing
    assert _fatal_reason(ok, None) is not None  # head/on missing
    assert _fatal_reason(None, None) is not None  # both missing


def test_fatal_reason_hard_fail_status_is_fatal_in_either_arm():
    ok = _cell("t::a", "off", "pass")
    for bad in ("fail", "error", "xpass"):
        assert _fatal_reason(_cell("t::a", "off", bad), ok) is not None, bad
        assert _fatal_reason(ok, _cell("t::a", "on", bad)) is not None, bad


def test_fatal_reason_two_arm_identical_fail_is_fatal():
    # THE regression codex flagged: two identically-failing arms used to be
    # classified PARITY and slip through. Both arms hard-fail -> fatal.
    off = _cell("t::a", "off", "fail")
    on = _cell("t::a", "on", "fail")
    assert _verdict(off, on) == "PARITY"  # _verdict still says PARITY...
    assert _fatal_reason(off, on) is not None  # ...but the fatal gate catches it


def test_fatal_reason_benign_statuses_are_not_fatal():
    # pass / skip / xfail are all non-red; none is fatal.
    for good in ("pass", "skip", "xfail"):
        a = _cell("t::a", "off", good)
        b = _cell("t::a", "on", good)
        assert _fatal_reason(a, b) is None, good


# --------------------------------------------------------------------------- #
# cmd_compare_modes end-to-end exit code (off-vs-on within one run)
# --------------------------------------------------------------------------- #


def test_compare_modes_all_pass_is_green(tmp_path):
    run = _write_results(
        tmp_path / "HEAD",
        [_cell("t::a", "off", "pass"), _cell("t::a", "on", "pass")],
    )
    assert cmd_compare_modes(_Args(run=str(run))) == 0


def test_compare_modes_two_arm_identical_fail_is_not_green(tmp_path):
    run = _write_results(
        tmp_path / "HEAD",
        [_cell("t::a", "off", "fail"), _cell("t::a", "on", "fail")],
    )
    assert cmd_compare_modes(_Args(run=str(run))) == 1


def test_compare_modes_missing_on_arm_is_fatal(tmp_path):
    # off ran, on crashed/absent -> only the off cell present. Must be RED.
    run = _write_results(tmp_path / "HEAD", [_cell("t::a", "off", "pass")])
    assert cmd_compare_modes(_Args(run=str(run))) == 1


def test_compare_modes_missing_off_arm_is_fatal(tmp_path):
    run = _write_results(tmp_path / "HEAD", [_cell("t::a", "on", "pass")])
    assert cmd_compare_modes(_Args(run=str(run))) == 1


def test_compare_modes_xpass_on_arm_is_fatal(tmp_path):
    # A strict-xfail cell that unexpectedly PASSED is a tripwire, not a pass.
    run = _write_results(
        tmp_path / "HEAD",
        [_cell("t::a", "off", "pass"), _cell("t::a", "on", "xpass")],
    )
    assert cmd_compare_modes(_Args(run=str(run))) == 1


def test_compare_modes_skip_both_arms_is_green(tmp_path):
    # A benign skip (family-guard / optional-dep) in both arms is not fatal.
    run = _write_results(
        tmp_path / "HEAD",
        [_cell("t::a", "off", "skip"), _cell("t::a", "on", "skip")],
    )
    assert cmd_compare_modes(_Args(run=str(run))) == 0


def test_compare_modes_on_arm_regression_is_red(tmp_path):
    # off passes, on fails -> REGRESSED verdict AND fatal; either way RED.
    run = _write_results(
        tmp_path / "HEAD",
        [_cell("t::a", "off", "pass"), _cell("t::a", "on", "fail")],
    )
    assert cmd_compare_modes(_Args(run=str(run))) == 1


# --------------------------------------------------------------------------- #
# cmd_compare end-to-end exit code (baseline-ref vs head-ref, same mode)
# --------------------------------------------------------------------------- #


def test_compare_all_pass_is_green(tmp_path):
    base = _write_results(tmp_path / "base", [_cell("t::a", "on", "pass")])
    head = _write_results(tmp_path / "head", [_cell("t::a", "on", "pass")])
    assert cmd_compare(_Args(baseline=str(base), head=str(head))) == 0


def test_compare_missing_head_arm_is_fatal(tmp_path):
    # Head ref's run crashed for this cell (present in base, absent in head).
    # Historically classified GONE and shown GREEN; now fatal.
    base = _write_results(tmp_path / "base", [_cell("t::a", "on", "pass")])
    head = _write_results(tmp_path / "head", [])
    assert cmd_compare(_Args(baseline=str(base), head=str(head))) == 1


def test_compare_two_arm_identical_fail_is_not_green(tmp_path):
    base = _write_results(tmp_path / "base", [_cell("t::a", "on", "fail")])
    head = _write_results(tmp_path / "head", [_cell("t::a", "on", "fail")])
    assert cmd_compare(_Args(baseline=str(base), head=str(head))) == 1
