# SPDX-License-Identifier: Apache-2.0
"""Tests for the flake-tracking system (dev-flow proposal item ③).

Units under test:
  * ``_pytest_summary.summary_node_ids`` — shared FAILED/ERROR parser.
  * ``quarantine.py`` — registry loader (file + git base ref) + matcher.
  * ``full_unit.py`` — quarantine-aware pass/fail split, exit-code and
    ERROR guards, base-revision registry source.
  * ``flake_tracking.py`` — advisory classification (flake vs reproduced),
    exit-code guard, and process-group timeout kill.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from scripts.pr_validate._pytest_summary import summary_node_ids
from scripts.pr_validate.context import Context
from scripts.pr_validate.quarantine import (
    QuarantineEntry,
    QuarantineError,
    load_quarantine,
    load_quarantine_from_ref,
    node_id_matches,
    partition_failures,
)
from scripts.pr_validate.steps import flake_tracking as flake_mod
from scripts.pr_validate.steps import full_unit as full_unit_mod
from scripts.pr_validate.steps.flake_tracking import FlakeTrackingStep, _run_session
from scripts.pr_validate.steps.full_unit import FullUnitStep

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


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(["pytest"], returncode, stdout, stderr)


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
# _pytest_summary.summary_node_ids
# --------------------------------------------------------------------------


class TestSummaryNodeIds:
    def test_plain_failed(self):
        assert summary_node_ids(_summary("FAILED tests/a.py::test_x"), "FAILED") == [
            "tests/a.py::test_x"
        ]

    def test_strips_message(self):
        out = summary_node_ids(
            _summary("FAILED tests/a.py::test_x - AssertionError: nope"), "FAILED"
        )
        assert out == ["tests/a.py::test_x"]

    def test_parametrized_with_spaces(self):
        out = summary_node_ids(
            _summary("FAILED tests/a.py::test_x[a b c] - ValueError"), "FAILED"
        )
        assert out == ["tests/a.py::test_x[a b c]"]

    def test_param_id_with_dash_space_not_truncated(self):
        # A param value containing " - " must NOT be cut at the first
        # occurrence — the separator is the one after the final ']'
        # (codex #1222 r4).
        out = summary_node_ids(
            _summary("FAILED tests/a.py::test_x[a - b] - AssertionError"), "FAILED"
        )
        assert out == ["tests/a.py::test_x[a - b]"]

    def test_param_id_with_dash_space_no_message(self):
        out = summary_node_ids(_summary("FAILED tests/a.py::test_x[a - b]"), "FAILED")
        assert out == ["tests/a.py::test_x[a - b]"]

    def test_ignores_failed_outside_summary(self):
        text = (
            "tests/a.py::test_x FAILED in call setup\n"  # pre-summary noise
            + _summary("FAILED tests/a.py::test_x")
        )
        assert summary_node_ids(text, "FAILED") == ["tests/a.py::test_x"]

    def test_multiple_labels(self):
        text = _summary(
            "FAILED tests/a.py::test_x - X",
            "ERROR tests/a.py::test_y - Y",
        )
        assert summary_node_ids(text, "FAILED") == ["tests/a.py::test_x"]
        assert summary_node_ids(text, "ERROR") == ["tests/a.py::test_y"]
        assert summary_node_ids(text, "FAILED", "ERROR") == [
            "tests/a.py::test_x",
            "tests/a.py::test_y",
        ]

    def test_default_label_is_failed(self):
        assert summary_node_ids(_summary("FAILED tests/a.py::t")) == ["tests/a.py::t"]

    def test_requires_padded_banner_not_bare_substring(self):
        # A bare "short test summary" line (no '=' padding) is NOT the real
        # pytest banner and must not open a summary block.
        text = (
            "some test printed: short test summary\n"
            "FAILED tests/evil.py::forged - injected\n"
        )
        assert summary_node_ids(text, "FAILED") == []

    def test_forged_banner_in_captured_output_is_ignored(self):
        # A malicious/careless test prints a fully-padded fake banner in its
        # captured stdout (which renders ABOVE the real summary). Taking the
        # LAST banner means the genuine final summary wins, so the real
        # failure can't be masked by a forged quarantined id (codex r2).
        text = (
            "==== FAILURES ====\n"
            "____ test_evil ____\n"
            "----- Captured stdout call -----\n"
            "==================== short test summary info ====================\n"
            "FAILED tests/quarantined.py::known_flake - forged\n"
            "==================== short test summary info ====================\n"
            "FAILED tests/real.py::genuine - AssertionError\n"
            "==== 1 failed in 0.10s ====\n"
        )
        assert summary_node_ids(text, "FAILED") == ["tests/real.py::genuine"]


# --------------------------------------------------------------------------
# quarantine.py — file loader
# --------------------------------------------------------------------------


class TestLoadQuarantine:
    def test_loads_shipped_registry(self):
        entries = load_quarantine()
        ids = {e.id for e in entries}
        assert (
            "tests/test_signal_observability.py::"
            "test_subprocess_sighup_default_disposition_dumps_and_stays_alive" in ids
        )
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

    @pytest.mark.parametrize("body", ["[]\n", "false\n", "0\n"])
    def test_falsey_non_mapping_root_raises(self, tmp_path, body):
        # A present-but-falsey root ([] / false / 0) is a schema violation,
        # NOT an empty registry — it must raise, not be swallowed (codex r2).
        p = tmp_path / "q.yaml"
        p.write_text(body)
        with pytest.raises(QuarantineError):
            load_quarantine(p)

    def test_empty_file_is_empty_registry(self, tmp_path):
        # An absent document (empty / all-comments → None root) IS a valid
        # empty registry.
        p = tmp_path / "q.yaml"
        p.write_text("# just a comment\n")
        assert load_quarantine(p) == []

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
# quarantine.py — git base-ref loader (a PR must not quarantine itself)
# --------------------------------------------------------------------------


class TestLoadQuarantineFromRef:
    def test_returns_entries_from_git_show(self, tmp_path, monkeypatch):
        yaml_text = "tests:\n  - id: a.py::t\n    reason: r\n"
        monkeypatch.setattr(
            "scripts.pr_validate.quarantine.subprocess.run",
            lambda *a, **k: _completed(0, yaml_text),
        )
        (entry,) = load_quarantine_from_ref("BASE", tmp_path)
        assert entry.id == "a.py::t"

    def test_absent_at_ref_is_empty(self, tmp_path, monkeypatch):
        # git show returns 128 when the path doesn't exist at that rev.
        monkeypatch.setattr(
            "scripts.pr_validate.quarantine.subprocess.run",
            lambda *a, **k: _completed(128, "", "fatal: path does not exist"),
        )
        assert load_quarantine_from_ref("BASE", tmp_path) == []

    def test_git_missing_is_empty_not_raise(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr("scripts.pr_validate.quarantine.subprocess.run", boom)
        # Fail-safe: infra hiccup → empty quarantine (stricter gate).
        assert load_quarantine_from_ref("BASE", tmp_path) == []

    def test_malformed_at_ref_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.pr_validate.quarantine.subprocess.run",
            lambda *a, **k: _completed(0, "tests: [unclosed\n"),
        )
        with pytest.raises(QuarantineError):
            load_quarantine_from_ref("BASE", tmp_path)

    def test_real_git_show_roundtrip(self, tmp_path):
        # End-to-end proof the actual `git show <ref>:<path>` path works.
        import os

        def git(*args):
            subprocess.run(
                ["git", *args],
                cwd=tmp_path,
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "GIT_AUTHOR_NAME": "t",
                    "GIT_AUTHOR_EMAIL": "t@t",
                    "GIT_COMMITTER_NAME": "t",
                    "GIT_COMMITTER_EMAIL": "t@t",
                },
            )

        git("init", "-q")
        rel = "scripts/pr_validate/quarantine.yaml"
        (tmp_path / "scripts" / "pr_validate").mkdir(parents=True)
        (tmp_path / rel).write_text("tests:\n  - id: x.py::t\n    reason: r\n")
        git("add", "-A")
        git("commit", "-q", "-m", "seed")
        (entry,) = load_quarantine_from_ref("HEAD", tmp_path)
        assert entry.id == "x.py::t"


# --------------------------------------------------------------------------
# quarantine.py — matcher / partition
# --------------------------------------------------------------------------


class TestNodeIdMatch:
    def test_exact(self):
        assert node_id_matches("a.py::t", "a.py::t")

    def test_param_family(self):
        assert node_id_matches("a.py::t[x-1]", "a.py::t")

    def test_base_entry_does_not_match_prefix_sibling(self):
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
# full_unit.py — quarantine-aware verdict (subprocess mocked)
# --------------------------------------------------------------------------


class TestFullUnitQuarantineAware:
    def _patch_pytest(self, monkeypatch, returncode, stdout):
        def fake_run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, returncode, stdout, "")

        monkeypatch.setattr(full_unit_mod.subprocess, "run", fake_run)

    def _patch_quarantine(self, monkeypatch, entries):
        monkeypatch.setattr(
            full_unit_mod, "load_quarantine_from_ref", lambda *a, **k: entries
        )

    def test_clean_pass(self, ctx_factory, monkeypatch):
        self._patch_pytest(monkeypatch, 0, _passed())
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "pass"

    def test_blocking_failure_fails(self, ctx_factory, monkeypatch):
        self._patch_quarantine(monkeypatch, [])
        self._patch_pytest(monkeypatch, 1, _summary("FAILED tests/a.py::test_real - X"))
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"
        assert "tests/a.py::test_real" in res.details

    def test_all_quarantined_passes_but_loud(self, ctx_factory, monkeypatch):
        self._patch_quarantine(
            monkeypatch, [QuarantineEntry(id="tests/a.py::test_flaky")]
        )
        self._patch_pytest(
            monkeypatch, 1, _summary("FAILED tests/a.py::test_flaky - X")
        )
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "pass"
        assert "known-flaky" in res.summary
        assert "tests/a.py::test_flaky" in res.details

    def test_mixed_blocks_and_shows_both(self, ctx_factory, monkeypatch):
        self._patch_quarantine(
            monkeypatch, [QuarantineEntry(id="tests/a.py::test_flaky")]
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
        assert "test_flaky" in res.details

    def test_error_entry_blocks_even_if_failed_quarantined(
        self, ctx_factory, monkeypatch
    ):
        # A quarantined FAILED riding along with an ERROR (collection /
        # fixture failure) must still BLOCK — an errored test is not a
        # quarantinable flake.
        self._patch_quarantine(
            monkeypatch, [QuarantineEntry(id="tests/a.py::test_flaky")]
        )
        self._patch_pytest(
            monkeypatch,
            1,
            _summary(
                "FAILED tests/a.py::test_flaky - X",
                "ERROR tests/a.py::test_boom - fixture blew up",
            ),
        )
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"
        assert "test_boom" in res.details

    def test_gate_pins_error_reporting(self, ctx_factory, monkeypatch):
        # The quarantine downgrade's soundness rests on ERROR lines always
        # appearing in the summary. The command must pin -rfE so it never
        # depends on pytest's implicit default (codex #1222 r3).
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, _passed(), "")

        monkeypatch.setattr(full_unit_mod.subprocess, "run", fake_run)
        FullUnitStep().run(ctx_factory())
        assert "-rfE" in captured["cmd"]

    def test_abnormal_exit_blocks(self, ctx_factory, monkeypatch):
        # Exit 2 (interrupted) with an otherwise-quarantined failure must
        # still block — the exit code says something the FAILED list
        # can't account for.
        self._patch_quarantine(
            monkeypatch, [QuarantineEntry(id="tests/a.py::test_flaky")]
        )
        self._patch_pytest(
            monkeypatch, 2, _summary("FAILED tests/a.py::test_flaky - X")
        )
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"

    def test_unreadable_registry_fails_safe(self, ctx_factory, monkeypatch):
        def boom(*a, **k):
            raise QuarantineError("broken registry")

        monkeypatch.setattr(full_unit_mod, "load_quarantine_from_ref", boom)
        self._patch_pytest(
            monkeypatch, 1, _summary("FAILED tests/a.py::test_flaky - X")
        )
        res = FullUnitStep().run(ctx_factory())
        # Even though test_flaky *would* be quarantined, a broken registry
        # collapses to an empty quarantine → strict gate → it blocks.
        assert res.status == "fail"
        assert "unreadable" in res.details

    def test_nonzero_exit_without_node_ids_blocks(self, ctx_factory, monkeypatch):
        self._patch_pytest(monkeypatch, 1, "==== 1 error in 0.50s ====\n")
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"

    def test_registry_sourced_from_base_revision(self, ctx_factory, monkeypatch):
        # The gate must read the registry from the PROTECTED base, not the
        # candidate checkout — assert full_unit passes ctx.base_sha to the
        # loader (so a PR can't quarantine its own failing tests).
        captured = {}

        def fake_loader(ref, repo_root, *a, **k):
            captured["ref"] = ref
            return [QuarantineEntry(id="tests/a.py::test_flaky")]

        monkeypatch.setattr(full_unit_mod, "load_quarantine_from_ref", fake_loader)
        self._patch_pytest(
            monkeypatch, 1, _summary("FAILED tests/a.py::test_flaky - X")
        )
        ctx = ctx_factory()
        ctx.base_sha = "deadbeefbase"
        res = FullUnitStep().run(ctx)
        assert captured["ref"] == "deadbeefbase"
        assert res.status == "pass"  # quarantined → non-blocking


# --------------------------------------------------------------------------
# flake_tracking.py — advisory step contract
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

    def _prime(self, ctx, monkeypatch):
        """full-unit.log with a failure + plugin present."""
        ctx.artifact_path("full-unit.log").write_text(
            _summary("FAILED tests/a.py::t - X")
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )

    def test_advisory_never_errors(self, ctx_factory, monkeypatch):
        ctx = ctx_factory()
        self._prime(ctx, monkeypatch)

        def boom(*a, **k):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(flake_mod, "_run_session", boom)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "skip"  # downgraded, NOT a blocking error

    def test_timeout_skips(self, ctx_factory, monkeypatch):
        ctx = ctx_factory()
        self._prime(ctx, monkeypatch)

        def timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="pytest", timeout=1)

        monkeypatch.setattr(flake_mod, "_run_session", timeout)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "skip"
        assert "exceeded" in res.summary

    def test_timeout_persists_drained_output(self, ctx_factory, monkeypatch):
        # On timeout the partial output must be written to flake-rerun.log
        # and attached, so a human can see which re-run hung (codex r3).
        ctx = ctx_factory()
        self._prime(ctx, monkeypatch)

        def timeout(*a, **k):
            raise subprocess.TimeoutExpired(
                cmd="pytest", timeout=1, output="partial stdout here", stderr="err tail"
            )

        monkeypatch.setattr(flake_mod, "_run_session", timeout)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "skip"
        log = ctx.artifact_path("flake-rerun.log")
        assert log.exists()
        body = log.read_text()
        assert "partial stdout here" in body and "err tail" in body
        assert str(log) in res.artifacts

    def test_abnormal_rerun_exit_skips(self, ctx_factory, monkeypatch):
        ctx = ctx_factory()
        self._prime(ctx, monkeypatch)
        # Re-run exits 2 (interrupted/collection) — cannot infer "passed"
        # from absence in FAILED, so classification must bail.
        monkeypatch.setattr(
            flake_mod, "_run_session", lambda *a, **k: _completed(2, "boom")
        )
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "skip"
        assert "abnormally" in res.summary

    def test_classifies_on_positive_outcomes(self, ctx_factory, monkeypatch):
        # Three sampled failures; the -rA re-run reports one PASSED (flake
        # candidate), one FAILED again (reproduced), and one that only went
        # SKIPPED — which must land in `inconclusive`, NOT be called a flake
        # just because it's absent from FAILED (codex r2).
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text(
            _summary(
                "FAILED tests/x.py::a - X",
                "FAILED tests/x.py::b - X",
                "FAILED tests/x.py::c - X",
            )
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])
        rerun = (
            "==================== short test summary info ====================\n"
            "PASSED tests/x.py::a\n"
            "FAILED tests/x.py::b - AssertionError\n"
            "SKIPPED [1] tests/x.py::c:1: conditionally skipped\n"
            "==== 1 failed, 1 passed, 1 skipped in 0.20s ====\n"
        )
        monkeypatch.setattr(
            flake_mod, "_run_session", lambda *a, **k: _completed(1, rerun)
        )
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert data["flake_candidates_new"] == ["tests/x.py::a"]
        assert data["reproduced_likely_real"] == ["tests/x.py::b"]
        assert data["inconclusive"] == ["tests/x.py::c"]


class TestFlakeTrackingClassification:
    """Drive a real pytest subprocess so the flake-vs-real split is
    proven end-to-end, not mocked."""

    def test_classifies_flake_vs_real(self, ctx_factory, tmp_path):
        pytest.importorskip("pytest_rerunfailures")
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


# --------------------------------------------------------------------------
# flake_tracking._run_session — process-group timeout kill
# --------------------------------------------------------------------------


class TestRunSession:
    def test_normal_completion(self, tmp_path):
        proc = _run_session(
            [sys.executable, "-c", "print('hi')"], cwd=str(tmp_path), timeout=30
        )
        assert proc.returncode == 0
        assert "hi" in proc.stdout

    def test_timeout_raises_bounded(self, tmp_path):
        # A child that would sleep far past the timeout must raise
        # TimeoutExpired quickly (not hang), and the kill path must run.
        t0 = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            _run_session(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=str(tmp_path),
                timeout=0.5,
            )
        assert time.monotonic() - t0 < 10.0

    def test_timeout_kills_grandchild_in_group(self, tmp_path):
        # Prove it's a process-GROUP kill, not just proc.kill(): the child
        # spawns a long-lived grandchild in the SAME group (no setsid) and
        # records its pid. On timeout, killpg must reap the grandchild too —
        # a bare proc.kill() on the direct child would leave it alive
        # (codex #1222 r2).
        pidfile = tmp_path / "grandchild.pid"
        child_prog = textwrap.dedent(
            f"""
            import subprocess, sys, time
            gc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)"]
            )
            with open({str(pidfile)!r}, "w") as f:
                f.write(str(gc.pid))
                f.flush()
            time.sleep(120)
            """
        )
        t0 = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            _run_session(
                [sys.executable, "-c", child_prog], cwd=str(tmp_path), timeout=3
            )
        assert time.monotonic() - t0 < 20.0

        gc_pid = int(pidfile.read_text().strip())
        # killpg SIGKILL is async; poll briefly for the grandchild to die.
        deadline = time.monotonic() + 5.0
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(gc_pid, 0)
            except ProcessLookupError:
                alive = False
                break
            time.sleep(0.1)
        if alive:
            # Clean up before failing so we don't leak a 120s sleeper.
            try:
                os.kill(gc_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            pytest.fail(f"grandchild {gc_pid} survived the process-group kill")

    def test_double_timeout_reaps_direct_child(self, monkeypatch):
        # If the post-kill drain ALSO times out (a group-escaping
        # descendant holds the pipe), _drain_bounded must still reap the
        # direct child so it isn't left a zombie (codex #1222 r4). Driven
        # with a stub so it's deterministic and fast.
        class _FakeProc:
            def __init__(self):
                self.stdout = None
                self.stderr = None
                self.pid = 999999
                self.wait_called = False

            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)

            def wait(self, timeout=None):
                self.wait_called = True
                return -9

        monkeypatch.setattr(flake_mod.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(flake_mod.os, "killpg", lambda pgid, sig: None)
        fake = _FakeProc()
        out, err = flake_mod._drain_bounded(fake)
        assert (out, err) == ("", "")
        assert fake.wait_called  # direct child was reaped, not orphaned


class TestRegistration:
    def test_registered_after_full_unit(self):
        from scripts.pr_validate.runner import STEPS

        names = [s.name for s in STEPS]
        assert "flake_tracking" in names
        assert names.index("flake_tracking") > names.index("full_unit")

    def test_is_advisory(self):
        assert FlakeTrackingStep().continue_on_error is True
