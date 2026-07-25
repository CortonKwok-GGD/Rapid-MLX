# SPDX-License-Identifier: Apache-2.0
"""Tests for the flake-tracking system (dev-flow proposal item ③).

Three units under test:
  * ``quarantine.py`` — registry loader + node-id matcher (pure).
  * ``full_unit.py`` — quarantine-aware pass/fail split of a failure set.
  * ``flake_tracking.py`` — advisory classification of full_unit failures
    (flake candidate vs reproduced-real), driven against a real pytest
    subprocess so the classification is proven, not mocked.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.pr_validate.context import Context
from scripts.pr_validate.quarantine import (
    QuarantineEntry,
    QuarantineError,
    load_quarantine,
    node_id_matches,
    partition_failures,
)
from scripts.pr_validate.steps import flake_tracking as flake_mod
from scripts.pr_validate.steps import full_unit as full_unit_mod
from scripts.pr_validate.steps.flake_tracking import FlakeTrackingStep
from scripts.pr_validate.steps.full_unit import FullUnitStep, _failed_node_ids

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _summary(*failed_lines: str) -> str:
    """A minimal pytest tail with a short-summary section."""
    body = "\n".join(failed_lines)
    return (
        "==== short test summary info ====\n"
        f"{body}\n"
        "==== 1 failed, 10 passed in 1.23s ====\n"
    )


def _passed(n: int = 10) -> str:
    return f"==== {n} passed in 1.00s ====\n"


@pytest.fixture
def ctx_factory(tmp_path, monkeypatch):
    """Context.__post_init__ insists on a repo-root cwd (a pyproject)."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
    monkeypatch.chdir(tmp_path)

    def _make(files_changed: list[str] | None = None) -> Context:
        ctx = Context(pr_number=1)
        # Default to a high-blast production file so full_unit / flake
        # steps' should_run predicate is satisfied.
        ctx.files_changed = files_changed or ["vllm_mlx/scheduler.py"]
        ctx.work_dir = tmp_path / "work"
        return ctx

    return _make


# --------------------------------------------------------------------------
# quarantine.py — loader
# --------------------------------------------------------------------------


class TestLoadQuarantine:
    def test_loads_shipped_registry(self):
        # Default path = the quarantine.yaml we ship. It must parse and
        # contain the seeded G8 signal flake.
        entries = load_quarantine()
        ids = {e.id for e in entries}
        assert (
            "tests/test_signal_observability.py::"
            "test_subprocess_sighup_default_disposition_dumps_and_stays_alive" in ids
        )
        # Every shipped entry carries a justification.
        assert all(e.reason for e in entries)

    def test_missing_file_is_empty(self, tmp_path):
        assert load_quarantine(tmp_path / "nope.yaml") == []

    def test_malformed_yaml_raises(self, tmp_path):
        p = tmp_path / "q.yaml"
        p.write_text("tests: [unclosed\n")
        with pytest.raises(QuarantineError):
            load_quarantine(p)

    def test_root_must_be_mapping(self, tmp_path):
        p = tmp_path / "q.yaml"
        p.write_text("- a\n- b\n")
        with pytest.raises(QuarantineError):
            load_quarantine(p)

    def test_tests_must_be_list(self, tmp_path):
        p = tmp_path / "q.yaml"
        p.write_text("tests: not-a-list\n")
        with pytest.raises(QuarantineError):
            load_quarantine(p)

    def test_entry_must_be_mapping(self, tmp_path):
        p = tmp_path / "q.yaml"
        p.write_text("tests:\n  - just-a-string\n")
        with pytest.raises(QuarantineError):
            load_quarantine(p)

    def test_entry_needs_nonempty_id(self, tmp_path):
        p = tmp_path / "q.yaml"
        p.write_text("tests:\n  - reason: no id here\n")
        with pytest.raises(QuarantineError):
            load_quarantine(p)

    def test_empty_tests_ok(self, tmp_path):
        p = tmp_path / "q.yaml"
        p.write_text("tests: []\n")
        assert load_quarantine(p) == []

    def test_null_tests_ok(self, tmp_path):
        p = tmp_path / "q.yaml"
        p.write_text("tests:\n")
        assert load_quarantine(p) == []

    def test_optional_fields_default_empty(self, tmp_path):
        p = tmp_path / "q.yaml"
        p.write_text("tests:\n  - id: a.py::t\n")
        (entry,) = load_quarantine(p)
        assert entry.id == "a.py::t"
        assert entry.reason == "" and entry.added == "" and entry.issue == ""


# --------------------------------------------------------------------------
# quarantine.py — matcher / partition
# --------------------------------------------------------------------------


class TestNodeIdMatch:
    def test_exact(self):
        assert node_id_matches("a.py::t", "a.py::t")

    def test_param_family(self):
        assert node_id_matches("a.py::t[x-1]", "a.py::t")

    def test_base_entry_does_not_match_prefix_sibling(self):
        # "a.py::t" must NOT swallow "a.py::t_extra" — a real bug risk if
        # we matched on bare startswith without the '[' guard.
        assert not node_id_matches("a.py::t_extra", "a.py::t")

    def test_specific_param_entry_is_exact_only(self):
        assert node_id_matches("a.py::t[x]", "a.py::t[x]")
        assert not node_id_matches("a.py::t[y]", "a.py::t[x]")

    def test_unrelated(self):
        assert not node_id_matches("a.py::t", "b.py::t")


class TestPartition:
    def test_splits_preserving_order(self):
        entries = [QuarantineEntry(id="a.py::flaky")]
        blocking, quarantined = partition_failures(
            ["a.py::real", "a.py::flaky", "a.py::real2"], entries
        )
        assert blocking == ["a.py::real", "a.py::real2"]
        assert quarantined == ["a.py::flaky"]

    def test_empty_registry_all_block(self):
        blocking, quarantined = partition_failures(["a.py::x"], [])
        assert blocking == ["a.py::x"]
        assert quarantined == []

    def test_param_family_all_quarantined(self):
        entries = [QuarantineEntry(id="a.py::t")]
        blocking, quarantined = partition_failures(
            ["a.py::t[1]", "a.py::t[2]"], entries
        )
        assert blocking == []
        assert quarantined == ["a.py::t[1]", "a.py::t[2]"]


# --------------------------------------------------------------------------
# full_unit.py — FAILED-line node id parsing
# --------------------------------------------------------------------------


class TestFailedNodeIds:
    def test_plain(self):
        assert _failed_node_ids(_summary("FAILED tests/a.py::test_x")) == [
            "tests/a.py::test_x"
        ]

    def test_with_message(self):
        out = _failed_node_ids(
            _summary("FAILED tests/a.py::test_x - AssertionError: nope")
        )
        assert out == ["tests/a.py::test_x"]

    def test_parametrized_with_spaces(self):
        out = _failed_node_ids(
            _summary("FAILED tests/a.py::test_x[a b c] - ValueError")
        )
        assert out == ["tests/a.py::test_x[a b c]"]

    def test_ignores_failed_outside_summary_section(self):
        text = (
            "tests/a.py::test_x FAILED in call setup\n"  # pre-summary noise
            + _summary("FAILED tests/a.py::test_x")
        )
        assert _failed_node_ids(text) == ["tests/a.py::test_x"]


# --------------------------------------------------------------------------
# full_unit.py — quarantine-aware verdict (subprocess mocked)
# --------------------------------------------------------------------------


class TestFullUnitQuarantineAware:
    def _patch_pytest(self, monkeypatch, returncode, stdout):
        def fake_run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, returncode, stdout, "")

        monkeypatch.setattr(full_unit_mod.subprocess, "run", fake_run)

    def test_clean_pass(self, ctx_factory, monkeypatch):
        self._patch_pytest(monkeypatch, 0, _passed())
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "pass"

    def test_blocking_failure_fails(self, ctx_factory, monkeypatch):
        monkeypatch.setattr(full_unit_mod, "load_quarantine", lambda: [])
        self._patch_pytest(monkeypatch, 1, _summary("FAILED tests/a.py::test_real - X"))
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"
        assert "tests/a.py::test_real" in res.details

    def test_all_quarantined_passes_but_loud(self, ctx_factory, monkeypatch):
        monkeypatch.setattr(
            full_unit_mod,
            "load_quarantine",
            lambda: [QuarantineEntry(id="tests/a.py::test_flaky")],
        )
        self._patch_pytest(
            monkeypatch, 1, _summary("FAILED tests/a.py::test_flaky - X")
        )
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "pass"
        assert "known-flaky" in res.summary
        assert "tests/a.py::test_flaky" in res.details

    def test_mixed_blocks_and_shows_both(self, ctx_factory, monkeypatch):
        monkeypatch.setattr(
            full_unit_mod,
            "load_quarantine",
            lambda: [QuarantineEntry(id="tests/a.py::test_flaky")],
        )
        self._patch_pytest(
            monkeypatch,
            1,
            _summary(
                "FAILED tests/a.py::test_real - X",
                "FAILED tests/a.py::test_flaky - Y",
            ),
        )
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"
        assert "test_real" in res.details
        assert "test_flaky" in res.details  # quarantined still surfaced

    def test_unreadable_registry_fails_safe(self, ctx_factory, monkeypatch):
        def boom():
            raise QuarantineError("broken registry")

        monkeypatch.setattr(full_unit_mod, "load_quarantine", boom)
        self._patch_pytest(
            monkeypatch, 1, _summary("FAILED tests/a.py::test_flaky - X")
        )
        res = FullUnitStep().run(ctx_factory())
        # Even though test_flaky *would* be quarantined, a broken registry
        # collapses to an empty quarantine → strict gate → it blocks.
        assert res.status == "fail"
        assert "unreadable" in res.details

    def test_nonzero_exit_without_node_ids_blocks(self, ctx_factory, monkeypatch):
        # Collection/internal error: exit 1 but no FAILED short-summary.
        self._patch_pytest(monkeypatch, 1, "==== 1 error in 0.50s ====\n")
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"
        assert "no parseable test ids" in res.details


# --------------------------------------------------------------------------
# flake_tracking.py — advisory step
# --------------------------------------------------------------------------


class TestFlakeTrackingContract:
    def test_should_run_false_without_log(self, ctx_factory):
        assert FlakeTrackingStep().should_run(ctx_factory()) is False

    def test_should_run_false_low_blast(self, ctx_factory):
        ctx = ctx_factory(["docs/x.md"])
        ctx.artifact_path("full-unit.log").write_text("x")
        assert FlakeTrackingStep().should_run(ctx) is False

    def test_should_run_true_with_log(self, ctx_factory):
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text("x")
        assert FlakeTrackingStep().should_run(ctx) is True

    def test_skip_when_no_failures(self, ctx_factory):
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text(_passed())
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "skip"

    def test_skip_when_plugin_absent(self, ctx_factory, monkeypatch):
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text(
            _summary("FAILED tests/a.py::t - X")
        )
        monkeypatch.setattr(flake_mod.importlib.util, "find_spec", lambda name: None)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "skip"
        assert "rerunfailures" in res.summary

    def test_advisory_never_errors(self, ctx_factory, monkeypatch):
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text(
            _summary("FAILED tests/a.py::t - X")
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )

        def boom(*a, **k):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(flake_mod.subprocess, "run", boom)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "skip"  # downgraded, NOT a blocking error

    def test_timeout_skips(self, ctx_factory, monkeypatch):
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text(
            _summary("FAILED tests/a.py::t - X")
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )

        def timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="pytest", timeout=1)

        monkeypatch.setattr(flake_mod.subprocess, "run", timeout)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "skip"
        assert "exceeded" in res.summary


class TestFlakeTrackingClassification:
    """Drive a real pytest subprocess so the flake-vs-real split is
    proven end-to-end, not mocked."""

    def test_classifies_flake_vs_real(self, ctx_factory, tmp_path):
        pytest.importorskip("pytest_rerunfailures")
        # A genuinely flaky test (fails the first attempt, passes on the
        # rerun via a filesystem counter) alongside a deterministic
        # failure.
        (tmp_path / "test_sample.py").write_text(
            "import pathlib\n"
            "_c = pathlib.Path(__file__).with_name('.count')\n"
            "def test_flaky():\n"
            "    n = int(_c.read_text()) if _c.exists() else 0\n"
            "    _c.write_text(str(n + 1))\n"
            "    assert n >= 1\n"
            "def test_real():\n"
            "    assert False\n"
        )
        ctx = ctx_factory()
        # full_unit's log: both failed the full suite.
        ctx.artifact_path("full-unit.log").write_text(
            _summary(
                "FAILED test_sample.py::test_flaky - AssertionError",
                "FAILED test_sample.py::test_real - AssertionError",
            )
        )
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"  # advisory always pass/skip
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert "test_sample.py::test_flaky" in data["flake_candidates_new"]
        assert "test_sample.py::test_real" in data["reproduced_likely_real"]


class TestRegistration:
    def test_registered_after_full_unit(self):
        from scripts.pr_validate.runner import STEPS

        names = [s.name for s in STEPS]
        assert "flake_tracking" in names
        assert names.index("flake_tracking") > names.index("full_unit")

    def test_is_advisory(self):
        assert FlakeTrackingStep().continue_on_error is True
