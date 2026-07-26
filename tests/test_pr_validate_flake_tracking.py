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
from pathlib import Path

import pytest

from scripts.pr_validate import _nodeid_reporter
from scripts.pr_validate._pytest_summary import (
    last_summary_line,
    raw_session_records,
    render_fence_safe,
    report_log_node_ids,
    summary_node_ids,
)
from scripts.pr_validate.base import GATING_PYTEST_GUARD
from scripts.pr_validate.context import Context
from scripts.pr_validate.quarantine import (
    QuarantineEntry,
    QuarantineError,
    _has_param_suffix,
    effective_quarantine,
    load_quarantine,
    load_quarantine_from_ref,
    node_id_matches,
    partition_failures,
)
from scripts.pr_validate.steps import flake_tracking as flake_mod
from scripts.pr_validate.steps import full_unit as full_unit_mod
from scripts.pr_validate.steps import targeted_tests as targeted_mod
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


# A complete-session marker appended to a simulated rerun node-id log.
# flake_tracking now requires the advisory rerun to have run to completion
# before it trusts the structured classification (codex #1222 r23); a real
# rerun always writes this record, so the fakes must too. The exact counts
# don't matter to session_completed beyond ``collected > 0 and ran >=
# collected``.
_RERUN_DONE = f"{_nodeid_reporter.SESSION_LABEL}\t99 99\n"


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
        # Truthy immutable base + head SHAs so the quarantine load path runs
        # the base ∩ candidate intersection (codex r10/r21). full_unit reads
        # the base registry from base_sha and the candidate from head_sha;
        # `_patch_quarantine` returns the same list for both, so the effective
        # set equals the patched registry. Tests exercising a fail-closed path
        # (no base/head sha, unreadable candidate) clear / diverge these.
        ctx.base_sha = "basesha0000000000000000000000000000000000"
        ctx.head_sha = "headsha0000000000000000000000000000000000"
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

    def test_message_with_bracket_not_absorbed_into_id(self):
        # A ']' inside the MESSAGE must not be mistaken for the id's
        # parametrization bracket — the separator is the first " - "
        # outside brackets (codex #1222 r5 regression guard).
        out = summary_node_ids(
            _summary("FAILED tests/a.py::test_x - AssertionError: [1]"), "FAILED"
        )
        assert out == ["tests/a.py::test_x"]

    def test_param_id_and_message_both_bracketed(self):
        out = summary_node_ids(
            _summary("FAILED tests/a.py::test_x[a - b] - ValueError: got [2]"),
            "FAILED",
        )
        assert out == ["tests/a.py::test_x[a - b]"]

    def test_nested_balanced_brackets_ok(self):
        # A param id with nested, BALANCED brackets and inner " - " is
        # parsed whole (depth returns to 0 only at the real separator).
        out = summary_node_ids(
            _summary("FAILED tests/a.py::test_x[[1, 2] - [3, 4]] - ValueError"),
            "FAILED",
        )
        assert out == ["tests/a.py::test_x[[1, 2] - [3, 4]]"]

    def test_unmatched_bracket_param_id_is_known_safe_limitation(self):
        # DOCUMENTED r6 limitation: an unmatched '[' in the param id leaves
        # the depth counter unbalanced, so the message is absorbed. Fails
        # SAFE — the mangled id won't match a normal quarantine entry, so
        # the failure BLOCKS. Locked here so it's an intentional trade-off,
        # not a silent regression.
        out = summary_node_ids(
            _summary("FAILED tests/a.py::test_x[[] - AssertionError"), "FAILED"
        )
        assert out == ["tests/a.py::test_x[[] - AssertionError"]

    def test_param_id_with_bracket_dash_bracket_no_message(self):
        # A param id containing "] - [" (e.g. ids=["a] - [b"]) makes the
        # inner ']' close depth early. With no message, failing safe returns
        # the WHOLE line — which is exactly the correct node id here.
        out = summary_node_ids(_summary("FAILED tests/a.py::test_x[a] - [b]"), "FAILED")
        assert out == ["tests/a.py::test_x[a] - [b]"]

    def test_param_id_with_bracket_dash_bracket_and_message_fails_safe(self):
        # DOCUMENTED r9 limitation: the same "] - [" param WITH a message is
        # irreducibly ambiguous. We must NOT truncate to "test_x[a]" (which
        # could wrongly match a shorter quarantine entry and downgrade a real
        # failure), so we fail SAFE and return the whole line — it won't match
        # a normal entry, so the failure BLOCKS. Locked as an intentional
        # trade-off, not a silent regression.
        out = summary_node_ids(
            _summary("FAILED tests/a.py::test_x[a] - [b] - AssertionError"), "FAILED"
        )
        assert out == ["tests/a.py::test_x[a] - [b] - AssertionError"]

    def test_bracket_leading_message_fails_safe_not_mismatched(self):
        # A genuine bracket-LEADING message ("[Errno 2] …") on a plain test
        # is equally indistinguishable from a mid-param split, so it also
        # fails safe (returns the whole line). This never mis-waives; at
        # worst a quarantined flake with such a message blocks instead of
        # being downgraded — the safe direction (codex #1222 r9).
        out = summary_node_ids(
            _summary("FAILED tests/a.py::test_x - [Errno 2] no such file"), "FAILED"
        )
        assert out == ["tests/a.py::test_x - [Errno 2] no such file"]

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

    def test_permission_error_raises_quarantine_error(self, tmp_path, monkeypatch):
        # codex #1222 r24: the old `if not path.exists(): return []` pre-check
        # ran OUTSIDE the try/except, and `Path.exists()` re-raises a
        # PermissionError (inaccessible parent dir) rather than swallowing it —
        # so a raw OSError escaped the documented QuarantineError fail-safe.
        # The read now goes direct; only FileNotFoundError → [], every other
        # OSError → QuarantineError.
        p = tmp_path / "q.yaml"

        def boom(*_a, **_k):
            raise PermissionError("permission denied")

        monkeypatch.setattr(Path, "read_text", boom)
        with pytest.raises(QuarantineError):
            load_quarantine(p)

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

    def test_non_utf8_file_raises(self, tmp_path):
        # A non-UTF-8 blob is malformed, not absent — must surface as
        # QuarantineError (not an unhandled UnicodeDecodeError) so the gate
        # fails safe to empty (codex #1222 r5).
        p = tmp_path / "q.yaml"
        p.write_bytes(b"tests:\n  - id: \xff\xfe not utf8\n")
        with pytest.raises(QuarantineError):
            load_quarantine(p)

    def test_unreadable_file_raises_quarantine_error(self, tmp_path, monkeypatch):
        # A present-but-unreadable registry (permissions / I/O error) is a
        # read failure, not "absent" — the loader must re-raise it as the
        # documented QuarantineError so a caller catching only that exception
        # doesn't crash on a raw OSError (codex #1222 r18).
        p = tmp_path / "q.yaml"
        p.write_text("tests: []\n")

        def boom(*a, **k):
            raise PermissionError("permission denied")

        monkeypatch.setattr(type(p), "read_text", boom)
        with pytest.raises(QuarantineError, match="could not be read"):
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

    def test_issue_optional_reason_added_present(self, tmp_path):
        # reason + added are mandatory audit data; issue stays optional
        # (not every flake has a tracker yet) — codex #1222 r10.
        p = tmp_path / "q.yaml"
        p.write_text(
            "tests:\n  - id: a.py::t\n    reason: flaky\n    added: 2026-01-01\n"
        )
        (entry,) = load_quarantine(p)
        assert entry.id == "a.py::t"
        assert entry.reason == "flaky" and entry.added == "2026-01-01"
        assert entry.issue == ""

    def test_missing_reason_raises(self, tmp_path):
        p = tmp_path / "q.yaml"
        p.write_text("tests:\n  - id: a.py::t\n    added: 2026-01-01\n")
        with pytest.raises(QuarantineError, match="reason"):
            load_quarantine(p)

    def test_duplicate_entry_key_rejected(self, tmp_path):
        # codex #1222 r21: yaml.safe_load silently keeps the LAST value for a
        # duplicated key, so a reviewer could approve `reason: real` while a
        # duplicate `reason:` invisibly overrides it. The strict loader rejects
        # duplicate mapping keys.
        p = tmp_path / "q.yaml"
        p.write_text(
            "tests:\n"
            "  - id: a.py::t\n"
            "    reason: real\n"
            "    added: 2026-01-01\n"
            "    reason: attacker-override\n"
        )
        with pytest.raises(QuarantineError, match="[Dd]uplicate"):
            load_quarantine(p)

    def test_duplicate_top_level_key_rejected(self, tmp_path):
        # A duplicated top-level `tests:` key would silently replace the whole
        # reviewed list — also rejected.
        p = tmp_path / "q.yaml"
        p.write_text(
            "tests:\n"
            "  - id: a.py::t\n"
            "    reason: r\n"
            "    added: 2026-01-01\n"
            "tests:\n"
            "  - id: b.py::t\n"
            "    reason: r\n"
            "    added: 2026-01-01\n"
        )
        with pytest.raises(QuarantineError, match="[Dd]uplicate"):
            load_quarantine(p)

    def test_unhashable_mapping_key_raises_quarantine_error(self, tmp_path):
        # codex #1222 r22: a complex YAML key (`? [a, b]` → a list) is
        # unhashable, so the duplicate-key check's `key in seen` would raise a
        # raw TypeError instead of the documented QuarantineError. The loader
        # must convert it to a QuarantineError so the gating caller fails safe.
        p = tmp_path / "q.yaml"
        p.write_text("tests:\n  - ? [a, b]\n    : value\n")
        with pytest.raises(QuarantineError):
            load_quarantine(p)

    def test_missing_added_raises(self, tmp_path):
        p = tmp_path / "q.yaml"
        p.write_text("tests:\n  - id: a.py::t\n    reason: flaky\n")
        with pytest.raises(QuarantineError, match="added"):
            load_quarantine(p)

    def test_non_string_reason_raises(self, tmp_path):
        # codex #1222 r15: a non-string reason (e.g. a YAML list) must raise,
        # not be str()-coerced into a bogus "['foo', 'bar']" rationale that
        # then satisfies the mandatory non-empty check.
        p = tmp_path / "q.yaml"
        p.write_text(
            "tests:\n  - id: a.py::t\n    reason: [foo, bar]\n    added: 2026-01-01\n"
        )
        with pytest.raises(QuarantineError, match="reason"):
            load_quarantine(p)

    def test_non_string_issue_raises(self, tmp_path):
        # Same guard on the optional ``issue`` audit field (codex #1222 r15).
        p = tmp_path / "q.yaml"
        p.write_text(
            "tests:\n  - id: a.py::t\n    reason: flaky\n    added: 2026-01-01\n"
            "    issue: [123]\n"
        )
        with pytest.raises(QuarantineError, match="issue"):
            load_quarantine(p)

    def test_non_date_added_raises(self, tmp_path):
        # `added` claims YYYY-MM-DD, so a non-empty-but-garbage value is
        # audit rot and must be rejected, not silently accepted (codex r13).
        p = tmp_path / "q.yaml"
        p.write_text(
            'tests:\n  - id: a.py::t\n    reason: flaky\n    added: "someday"\n'
        )
        with pytest.raises(QuarantineError, match="added"):
            load_quarantine(p)

    def test_noncanonical_added_date_raises(self, tmp_path):
        # A date that doesn't round-trip to canonical YYYY-MM-DD (non-padded,
        # out-of-range month/day, a timestamp) is rejected so the audit trail
        # stays uniform (codex #1222 r13).
        p = tmp_path / "q.yaml"
        p.write_text(
            'tests:\n  - id: a.py::t\n    reason: flaky\n    added: "2026-13-40"\n'
        )
        with pytest.raises(QuarantineError, match="added"):
            load_quarantine(p)

    def test_family_field_defaults_false_and_parses_true(self, tmp_path):
        # codex #1222 r14: family is an optional boolean, default False.
        p = tmp_path / "q.yaml"
        p.write_text(
            "tests:\n"
            "  - id: a.py::t\n    reason: flaky\n    added: 2026-01-01\n"
            "  - id: b.py::t\n    reason: flaky\n    added: 2026-01-01\n"
            "    family: true\n"
        )
        a, b = load_quarantine(p)
        assert a.family is False
        assert b.family is True

    def test_non_boolean_family_raises(self, tmp_path):
        # A truthy string must not silently widen the allowlist — family
        # must be a real boolean (codex #1222 r14).
        p = tmp_path / "q.yaml"
        p.write_text(
            "tests:\n  - id: a.py::t\n    reason: flaky\n    added: 2026-01-01\n"
            '    family: "yes"\n'
        )
        with pytest.raises(QuarantineError, match="family"):
            load_quarantine(p)


# --------------------------------------------------------------------------
# quarantine.py — git base-ref loader (a PR must not quarantine itself)
# --------------------------------------------------------------------------


class TestLoadQuarantineFromRef:
    def test_returns_entries_from_git_show(self, tmp_path, monkeypatch):
        yaml_text = "tests:\n  - id: a.py::t\n    reason: r\n    added: 2026-01-01\n"
        monkeypatch.setattr(
            "scripts.pr_validate.quarantine.subprocess.run",
            lambda *a, **k: _completed(0, yaml_text),
        )
        (entry,) = load_quarantine_from_ref("BASE", tmp_path)
        assert entry.id == "a.py::t"

    def test_absent_at_ref_is_empty(self, tmp_path, monkeypatch):
        # git show returns 128 with a "does not exist in" message when the
        # PATH isn't in that rev's tree — a legitimate empty registry (the
        # base predates the file), NOT an infra failure (codex #1222 r10).
        monkeypatch.setattr(
            "scripts.pr_validate.quarantine.subprocess.run",
            lambda *a, **k: _completed(
                128, "", "fatal: path 'q.yaml' does not exist in 'BASE'"
            ),
        )
        assert load_quarantine_from_ref("BASE", tmp_path) == []

    def test_git_missing_raises(self, tmp_path, monkeypatch):
        # git not on PATH is an INFRA failure, not "no registry" — it must
        # raise so full_unit reports the fail-safe fallback loudly instead
        # of silently behaving as if nothing is quarantined (codex r10).
        def boom(*a, **k):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr("scripts.pr_validate.quarantine.subprocess.run", boom)
        with pytest.raises(QuarantineError):
            load_quarantine_from_ref("BASE", tmp_path)

    def test_git_timeout_raises(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd="git", timeout=30)

        monkeypatch.setattr("scripts.pr_validate.quarantine.subprocess.run", boom)
        with pytest.raises(QuarantineError):
            load_quarantine_from_ref("BASE", tmp_path)

    def test_bad_ref_git_error_raises(self, tmp_path, monkeypatch):
        # A non-"missing path" git error (bad ref, not a repo) must surface,
        # not be swallowed as an empty registry (codex #1222 r10).
        monkeypatch.setattr(
            "scripts.pr_validate.quarantine.subprocess.run",
            lambda *a, **k: _completed(128, "", "fatal: invalid object name 'BASE'"),
        )
        with pytest.raises(QuarantineError):
            load_quarantine_from_ref("BASE", tmp_path)

    def test_malformed_at_ref_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.pr_validate.quarantine.subprocess.run",
            lambda *a, **k: _completed(0, "tests: [unclosed\n"),
        )
        with pytest.raises(QuarantineError):
            load_quarantine_from_ref("BASE", tmp_path)

    def test_non_utf8_at_ref_raises(self, tmp_path, monkeypatch):
        # A non-UTF-8 blob at the ref makes `text=True` subprocess.run
        # raise UnicodeDecodeError — it must be wrapped as QuarantineError,
        # not escape as an unhandled ValueError (codex #1222 r5).
        def boom(*a, **k):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        monkeypatch.setattr("scripts.pr_validate.quarantine.subprocess.run", boom)
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
        (tmp_path / rel).write_text(
            "tests:\n  - id: x.py::t\n    reason: r\n    added: 2026-01-01\n"
        )
        git("add", "-A")
        git("commit", "-q", "-m", "seed")
        (entry,) = load_quarantine_from_ref("HEAD", tmp_path)
        assert entry.id == "x.py::t"

    def test_real_git_show_ignores_replace_ref(self, tmp_path):
        # codex #1222 r12 (security): pinning to an immutable base SHA is not
        # enough on its own — `git show` honors local refs/replace/*, so a
        # candidate test that ran earlier could `git replace <base_sha>
        # <attacker-commit>` and make `git show <base_sha>:…` serve its own
        # allowlist. The loader passes --no-replace-objects, so the TRUE
        # content-addressed base blob wins regardless of any planted replace.
        import os

        def git(*args):
            return subprocess.run(
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
        (tmp_path / rel).write_text(
            "tests:\n  - id: GOOD::t\n    reason: r\n    added: 2026-01-01\n"
        )
        git("add", "-A")
        git("commit", "-q", "-m", "base")
        base_sha = git("rev-parse", "HEAD").stdout.strip()
        # An attacker commit carrying a DIFFERENT allowlist.
        (tmp_path / rel).write_text(
            "tests:\n  - id: EVIL::t\n    reason: r\n    added: 2026-01-01\n"
        )
        git("add", "-A")
        git("commit", "-q", "-m", "evil")
        evil_sha = git("rev-parse", "HEAD").stdout.strip()
        git("replace", base_sha, evil_sha)
        # Sanity: a naive `git show <base_sha>:…` IS fooled by the replace.
        assert "EVIL::t" in git("show", f"{base_sha}:{rel}").stdout
        # The loader must NOT be — it reads the true base blob.
        (entry,) = load_quarantine_from_ref(base_sha, tmp_path)
        assert entry.id == "GOOD::t"


# --------------------------------------------------------------------------
# quarantine.py — matcher / partition
# --------------------------------------------------------------------------


class TestNodeIdMatch:
    def test_exact(self):
        assert node_id_matches("a.py::t", "a.py::t")

    def test_param_family_requires_opt_in(self):
        # codex #1222 r14: a base entry covers parametrizations ONLY with
        # family=True — exact by default so a new failing param isn't
        # silently waived.
        assert node_id_matches("a.py::t[x-1]", "a.py::t", family=True)

    def test_exact_by_default_does_not_match_params(self):
        assert not node_id_matches("a.py::t[x-1]", "a.py::t")

    def test_family_only_expands_base_entry(self):
        # family on an already-specific param id doesn't widen it.
        assert not node_id_matches("a.py::t[y]", "a.py::t[x]", family=True)

    def test_base_entry_does_not_match_prefix_sibling(self):
        assert not node_id_matches("a.py::t_extra", "a.py::t", family=True)

    def test_specific_param_entry_is_exact_only(self):
        assert node_id_matches("a.py::t[x]", "a.py::t[x]")
        assert not node_id_matches("a.py::t[y]", "a.py::t[x]")

    def test_unrelated(self):
        assert not node_id_matches("a.py::t", "b.py::t")

    def test_has_param_suffix_only_inspects_final_component(self):
        # codex #1222 r27: the parametrization bracket is only in the last
        # ``::`` component — a "[" in the directory/class path is not a param.
        assert _has_param_suffix("a.py::t[x]")
        assert _has_param_suffix("tests/[legacy]/a.py::TestC::t[x-1]")
        assert not _has_param_suffix("tests/[legacy]/a.py::t")
        assert not _has_param_suffix("tests/models/[v2]/a.py::TestC::t")

    def test_family_matches_under_bracketed_directory(self):
        # A test beneath a bracket-named directory must still get family
        # coverage — the "[" is in the path, not a param suffix (codex r27).
        entry = "tests/models/[legacy]/test_x.py::test_y"
        assert node_id_matches(entry + "[a-1]", entry, family=True)
        assert not node_id_matches(entry + "[a-1]", entry)  # exact by default

    def test_family_disabled_when_final_component_parametrized(self):
        # Even under a bracket dir, an entry whose FINAL component is already
        # parametrized is exact-only (family can't widen a specific param).
        entry = "tests/[legacy]/test_x.py::test_y[a]"
        assert not node_id_matches(
            "tests/[legacy]/test_x.py::test_y[b]", entry, family=True
        )


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

    def test_param_family_all_quarantined_with_opt_in(self):
        entries = [QuarantineEntry(id="a.py::t", family=True)]
        blocking, quarantined = partition_failures(
            ["a.py::t[1]", "a.py::t[2]"], entries
        )
        assert blocking == []
        assert quarantined == ["a.py::t[1]", "a.py::t[2]"]

    def test_base_entry_without_family_blocks_new_params(self):
        # codex #1222 r14: without family=True, a base entry does NOT waive
        # parametrizations — a newly-added failing case still blocks.
        entries = [QuarantineEntry(id="a.py::t")]
        blocking, quarantined = partition_failures(
            ["a.py::t[1]", "a.py::t[2]"], entries
        )
        assert blocking == ["a.py::t[1]", "a.py::t[2]"]
        assert quarantined == []


class TestEffectiveQuarantine:
    # codex #1222 r21: effective registry = base ∩ candidate, so a
    # de-quarantine REMOVAL is honored while a candidate can never widen or
    # self-add coverage.
    def _e(self, i, fam=False):
        return QuarantineEntry(id=i, reason="r", added="2026-07-25", family=fam)

    def test_kept_when_in_both(self):
        eff = effective_quarantine([self._e("t::a")], [self._e("t::a")])
        assert [e.id for e in eff] == ["t::a"]

    def test_removal_dropped(self):
        # base lists it, candidate removed it → not effective.
        eff = effective_quarantine([self._e("t::a")], [])
        assert eff == []

    def test_candidate_addition_ignored(self):
        # candidate lists it, base doesn't → not effective (no self-quarantine).
        eff = effective_quarantine([], [self._e("t::a")])
        assert eff == []

    def test_candidate_cannot_widen_family(self):
        # base exact, candidate widens to family → stays exact (narrower wins).
        eff = effective_quarantine(
            [self._e("t::a", fam=False)], [self._e("t::a", fam=True)]
        )
        assert eff[0].family is False

    def test_candidate_may_tighten_family(self):
        # base family, candidate tightens to exact → effective exact (allowed).
        eff = effective_quarantine(
            [self._e("t::a", fam=True)], [self._e("t::a", fam=False)]
        )
        assert eff[0].family is False

    def test_family_kept_when_both_family(self):
        eff = effective_quarantine(
            [self._e("t::a", fam=True)], [self._e("t::a", fam=True)]
        )
        assert eff[0].family is True

    def test_result_is_always_subset_of_base(self):
        base = [self._e("t::a"), self._e("t::b", fam=True)]
        # A maximally-permissive candidate cannot add ids nor widen breadth.
        candidate = [
            self._e("t::a", fam=True),
            self._e("t::c"),
            self._e("t::b", fam=True),
        ]
        eff = effective_quarantine(base, candidate)
        assert {e.id for e in eff} <= {b.id for b in base}
        by_id = {e.id: e.family for e in eff}
        assert by_id["t::a"] is False  # base was exact → stays exact
        assert by_id["t::b"] is True
        assert "t::c" not in by_id  # candidate-added ignored

    def test_base_audit_metadata_preserved(self):
        base = [QuarantineEntry(id="t::a", reason="base-why", added="2026-01-01")]
        cand = [QuarantineEntry(id="t::a", reason="cand-why", added="2026-09-09")]
        (eff,) = effective_quarantine(base, cand)
        assert eff.reason == "base-why" and eff.added == "2026-01-01"


# --------------------------------------------------------------------------
# full_unit.py — quarantine-aware verdict (subprocess mocked)
# --------------------------------------------------------------------------


class TestFullUnitQuarantineAware:
    def _patch_pytest(
        self,
        monkeypatch,
        returncode,
        stdout,
        *,
        structured=True,
        session=True,
        ran=100,
        collected=100,
        session_value=None,
        write_log=True,
        rerun_ids=None,
    ):
        def fake_run(cmd, **kw):
            # Simulate the reporter plugin: when structured logging is on,
            # mirror the summary's FAILED/ERROR lines into the node-id TSV
            # the real plugin would have written. full_unit reads THAT for
            # the quarantine decision (the summary is only a fallback, and a
            # downgrade is withheld entirely when the structured log is
            # absent — codex #1222 r13). ``write_log=False`` simulates a hard
            # os._exit before the first append: the plugin writes NOTHING, so
            # only full_unit's own start sentinel remains (codex #1222 r17).
            env = kw.get("env") or {}
            log = env.get("PR_VALIDATE_NODEID_LOG")
            if structured and log and write_log:
                rows = [
                    f"{label}\t{nid}"
                    for label in ("FAILED", "ERROR")
                    for nid in summary_node_ids(stdout, label)
                ]
                # A RERUN record simulates pytest-rerunfailures having run in a
                # gating invocation despite the by-name block (smuggled in under
                # an arbitrary name) — full_unit must block on it name-
                # independently (codex #1222 r23).
                rows += [
                    f"{_nodeid_reporter.RERUN_LABEL}\t{nid}" for nid in rerun_ids or []
                ]
                # The reporter's session-finish completion record (codex
                # #1222 r15). ``ran >= collected`` marks a COMPLETE run — the
                # precondition for a downgrade. Truncation tests pass
                # ``session=False`` (hard exit / crash → the hook never fires,
                # no record) or ``ran < collected`` (early stop); a malformed
                # record is driven via ``session_value``.
                if session:
                    value = (
                        session_value
                        if session_value is not None
                        else f"{ran} {collected}"
                    )
                    rows.append(f"{_nodeid_reporter.SESSION_LABEL}\t{value}")
                with open(log, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(rows) + ("\n" if rows else ""))
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

    def test_gate_pins_error_reporting_and_no_color(self, ctx_factory, monkeypatch):
        # The quarantine downgrade's soundness rests on ERROR lines always
        # appearing (-rfE, codex r3) in ANSI-free text (--color=no, codex
        # r6) — otherwise forced color would break summary parsing.
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, _passed(), "")

        monkeypatch.setattr(full_unit_mod.subprocess, "run", fake_run)
        FullUnitStep().run(ctx_factory())
        assert "-rfE" in captured["cmd"]
        assert "--color=no" in captured["cmd"]

    def test_gate_disables_rerunfailures(self, ctx_factory, monkeypatch):
        # codex #1222 r20: the gating run must block pytest-rerunfailures so a
        # candidate `@pytest.mark.flaky(reruns=N)` can't re-run and pass a
        # genuinely failing test to green without consulting the quarantine.
        # It's an AUTOLOADED plugin, so it must be disabled by its registered
        # name (`-p no:rerunfailures`); -o addopts= / env-strip can't stop it.
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, _passed(), "")

        monkeypatch.setattr(full_unit_mod.subprocess, "run", fake_run)
        FullUnitStep().run(ctx_factory())
        cmd = captured["cmd"]
        assert "no:rerunfailures" in cmd
        assert cmd[cmd.index("no:rerunfailures") - 1] == "-p"
        # codex #1222 r23: block BOTH the entry-point name (rerunfailures) and
        # the module name (pytest_rerunfailures) — a conftest can register
        # under either. (Name-blocking is only the cheap first line; the
        # name-independent RERUN backstop is asserted separately.)
        assert "no:pytest_rerunfailures" in cmd
        assert cmd[cmd.index("no:pytest_rerunfailures") - 1] == "-p"

    def test_gate_blocks_on_rerun_record(self, ctx_factory, monkeypatch):
        # codex #1222 r23: name-based -p blocking is bypassable — a conftest
        # can register pytest-rerunfailures under an ARBITRARY name, then a
        # @pytest.mark.flaky marker reruns a real failure to green. The
        # reporter logs a RERUN record on any rerun OUTCOME (name-independent),
        # and full_unit must BLOCK on it regardless of a green exit code, since
        # a gating run must never rerun. Here the exit code is 0 with a passing
        # summary, yet a RERUN record is present → block.
        self._patch_pytest(monkeypatch, 0, _passed(), rerun_ids=["tests/a.py::test_x"])
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"
        assert "rerun" in res.details.lower()

    def test_nonzero_exit_with_rerun_record_blocks(self, ctx_factory, monkeypatch):
        # The RERUN backstop fires on the failure path too: a run that reran
        # (and passed some tests) but still exited non-zero is equally
        # untrusted — the rerun could have flipped OTHER real failures green.
        self._patch_quarantine(monkeypatch, [])
        self._patch_pytest(
            monkeypatch,
            1,
            _summary("FAILED tests/a.py::test_real - X"),
            rerun_ids=["tests/a.py::test_real"],
        )
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"
        assert "rerun" in res.details.lower()

    def test_every_gating_pytest_cmd_blocks_rerunfailures(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r21: r20 disabled rerunfailures only in full_unit, but
        # targeted_tests ALSO runs candidate tests — a `@pytest.mark.flaky` on
        # a changed file could rerun-and-pass there. Both gating pytest
        # invocations must carry the shared guard so no step can silently
        # forget it. full_unit is captured live; targeted_tests exposes its
        # base command as a module constant.
        guard = list(GATING_PYTEST_GUARD)

        def _contains_subseq(seq, sub):
            return any(
                seq[i : i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1)
            )

        # targeted_tests base cmd
        assert _contains_subseq(list(targeted_mod._PYTEST_CMD), guard)

        # full_unit cmd (captured)
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, _passed(), "")

        monkeypatch.setattr(full_unit_mod.subprocess, "run", fake_run)
        FullUnitStep().run(ctx_factory())
        assert _contains_subseq(list(captured["cmd"]), guard)

    def test_exit_zero_with_recorded_failures_blocks(self, ctx_factory, monkeypatch):
        # codex #1222 r21: a conftest / exit-code plugin can normalize a
        # FAILING run to exit 0 (e.g. pytest_sessionfinish setting
        # session.exitstatus = 0). The exit-0 fast path must NOT trust the
        # code over the structured records — a clean exit whose log carries
        # FAILED ids is tampering, not green. Block, and grant no downgrade.
        self._patch_pytest(monkeypatch, 0, _summary("FAILED tests/a.py::test_real - X"))
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"
        assert "exited 0" in res.details
        assert "tests/a.py::test_real" in res.details

    def _patch_quarantine_by_ref(self, monkeypatch, *, base, candidate):
        # Return DIFFERENT registries for the base_sha vs head_sha reads so
        # the base ∩ candidate intersection can be exercised (codex #1222 r21).
        def fake(ref, *a, **k):
            return candidate if ref == "headsha1111" else base

        monkeypatch.setattr(full_unit_mod, "load_quarantine_from_ref", fake)

    def test_dequarantine_removal_blocks(self, ctx_factory, monkeypatch):
        # A PR that REMOVES an entry (base lists it, candidate doesn't) while
        # the test still fails must BLOCK — the removal drops it from the
        # effective (base ∩ candidate) set, so it's no longer waived.
        self._patch_quarantine_by_ref(
            monkeypatch,
            base=[QuarantineEntry(id="tests/a.py::test_flaky")],
            candidate=[],  # de-quarantined in this PR
        )
        self._patch_pytest(
            monkeypatch, 1, _summary("FAILED tests/a.py::test_flaky - X")
        )
        ctx = ctx_factory()
        ctx.head_sha = "headsha1111"
        res = FullUnitStep().run(ctx)
        assert res.status == "fail"
        assert "tests/a.py::test_flaky" in res.details

    def test_quarantine_in_both_still_waived(self, ctx_factory, monkeypatch):
        # An entry present in BOTH base and candidate is still effective → the
        # flaky failure is waived (non-blocking), as before.
        self._patch_quarantine_by_ref(
            monkeypatch,
            base=[QuarantineEntry(id="tests/a.py::test_flaky")],
            candidate=[QuarantineEntry(id="tests/a.py::test_flaky")],
        )
        self._patch_pytest(
            monkeypatch, 1, _summary("FAILED tests/a.py::test_flaky - X")
        )
        ctx = ctx_factory()
        ctx.head_sha = "headsha1111"
        res = FullUnitStep().run(ctx)
        assert res.status == "pass"
        assert "known-flaky" in res.summary

    def test_candidate_added_quarantine_ignored(self, ctx_factory, monkeypatch):
        # A candidate ADDITION (id in candidate but not base) is ignored — a
        # PR still can't self-quarantine its own failing test.
        self._patch_quarantine_by_ref(
            monkeypatch,
            base=[],
            candidate=[QuarantineEntry(id="tests/a.py::test_flaky")],
        )
        self._patch_pytest(
            monkeypatch, 1, _summary("FAILED tests/a.py::test_flaky - X")
        )
        ctx = ctx_factory()
        ctx.head_sha = "headsha1111"
        res = FullUnitStep().run(ctx)
        assert res.status == "fail"
        assert "tests/a.py::test_flaky" in res.details

    def test_malformed_candidate_registry_fails_closed(self, ctx_factory, monkeypatch):
        # codex #1222 r22: a candidate CONTROLS its own committed
        # quarantine.yaml. If it's malformed/unreadable, we must NOT fall back
        # to the stale base waiver (that would let a PR keep a base-quarantined
        # but now-failing test green by breaking its own registry). Fail CLOSED
        # to an empty quarantine → the failure blocks.
        def fake(ref, *a, **k):
            if ref == "headsha1111":  # candidate read
                raise QuarantineError("candidate registry is malformed")
            return [QuarantineEntry(id="tests/a.py::test_flaky")]  # base

        monkeypatch.setattr(full_unit_mod, "load_quarantine_from_ref", fake)
        self._patch_pytest(
            monkeypatch, 1, _summary("FAILED tests/a.py::test_flaky - X")
        )
        ctx = ctx_factory()
        ctx.head_sha = "headsha1111"
        res = FullUnitStep().run(ctx)
        assert res.status == "fail"
        assert "failing closed" in res.details
        assert "tests/a.py::test_flaky" in res.details

    def test_missing_head_sha_fails_closed(self, ctx_factory, monkeypatch):
        # No pinned head sha → the candidate registry can't be confirmed →
        # fail closed to empty (every failure blocks), not the base waiver.
        self._patch_quarantine(
            monkeypatch, [QuarantineEntry(id="tests/a.py::test_flaky")]
        )
        self._patch_pytest(
            monkeypatch, 1, _summary("FAILED tests/a.py::test_flaky - X")
        )
        ctx = ctx_factory()
        ctx.head_sha = ""  # unavailable
        res = FullUnitStep().run(ctx)
        assert res.status == "fail"
        assert "failing closed" in res.details

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

    def test_registry_error_message_is_single_line_sanitized(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r25: a QuarantineError message can echo candidate-
        # controlled registry text (a bad node id / reason from the PR's own
        # quarantine.yaml). Interpolated RAW into the scorecard note, a newline
        # + "## heading" would forge a scorecard section; render_fence_safe
        # collapses it to a single escaped JSON line, so the hostile text
        # survives for the reader but can NEVER inject a raw newline/heading.
        hostile = "bad id\n## FORGED HEADING\n```\nnot a real fence"

        def boom(*a, **k):
            raise QuarantineError(hostile)

        monkeypatch.setattr(full_unit_mod, "load_quarantine_from_ref", boom)
        self._patch_pytest(
            monkeypatch, 1, _summary("FAILED tests/a.py::test_flaky - X")
        )
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"
        # No raw newline-injected heading anywhere in the rendered note …
        assert "\n## FORGED HEADING" not in res.details
        # … but the text is preserved as an escaped single-line JSON string.
        assert "\\n## FORGED HEADING" in res.details

    def test_nonzero_exit_without_node_ids_blocks(self, ctx_factory, monkeypatch):
        self._patch_pytest(monkeypatch, 1, "==== 1 error in 0.50s ====\n")
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"

    def test_registry_sourced_from_base_and_head_revisions(
        self, ctx_factory, monkeypatch
    ):
        # The gate reads the BASE registry from the protected base SHA (so a
        # PR can't quarantine its own failing tests) and the CANDIDATE from
        # the pinned head SHA (so a de-quarantine removal is honored) — both
        # from immutable revisions, never the working tree (codex r10/r21).
        captured = {"refs": []}

        def fake_loader(ref, repo_root, *a, **k):
            captured["refs"].append(ref)
            return [QuarantineEntry(id="tests/a.py::test_flaky")]

        monkeypatch.setattr(full_unit_mod, "load_quarantine_from_ref", fake_loader)
        self._patch_pytest(
            monkeypatch, 1, _summary("FAILED tests/a.py::test_flaky - X")
        )
        ctx = ctx_factory()
        ctx.base_sha = "deadbeefbase"
        ctx.head_sha = "deadbeefhead"
        res = FullUnitStep().run(ctx)
        # Base read first (base_sha), then candidate (head_sha) — both pinned.
        assert captured["refs"] == ["deadbeefbase", "deadbeefhead"]
        assert res.status == "pass"  # in both → quarantined → non-blocking

    def test_downgrade_withheld_without_structured_log(self, ctx_factory, monkeypatch):
        # codex #1222 r13: the quarantine downgrade must rest on the exact
        # structured node ids, never the ambiguous summary fallback. If the
        # reporter can't be injected (available() False), grant NO downgrade —
        # a would-be quarantined failure BLOCKS, so a mis-split fallback id can
        # never wrongly match a family entry and waive a real red. (r17: "no
        # structured log" now means available() False, since an injected
        # plugin always leaves a start sentinel.)
        monkeypatch.setattr(full_unit_mod._nodeid_reporter, "available", lambda: False)
        self._patch_quarantine(
            monkeypatch, [QuarantineEntry(id="tests/a.py::test_flaky")]
        )
        self._patch_pytest(
            monkeypatch,
            1,
            _summary("FAILED tests/a.py::test_flaky - X"),
            structured=False,  # plugin produced no log
        )
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"  # withheld → blocks despite being listed
        assert "structured node-id log absent" in res.details

    def test_cmd_overrides_candidate_maxfail(self, ctx_factory, monkeypatch):
        # codex #1222 r13: a candidate could plant -x / --maxfail=1 in
        # pytest.ini addopts to stop the suite after one quarantined failure
        # and have the partial, regression-hiding run accepted. full_unit
        # appends --maxfail=0 (no limit) so a trailing command-line maxfail
        # overrides the prepended ini one and the WHOLE suite always runs.
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, _passed(), "")

        monkeypatch.setattr(full_unit_mod.subprocess, "run", fake_run)
        FullUnitStep().run(ctx_factory())
        assert "--maxfail=0" in captured["cmd"]

    def test_neutralizes_candidate_addopts_and_env(self, ctx_factory, monkeypatch):
        # codex #1222 r14: --maxfail=0 alone does NOT stop --stepwise, so the
        # gate clears the candidate's ini addopts (-o addopts=) AND strips
        # PYTEST_ADDOPTS from the subprocess env, re-applying its OWN marker
        # filter so selection can't be steered by candidate config.
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            captured["env"] = kw.get("env")
            return subprocess.CompletedProcess(cmd, 0, _passed(), "")

        monkeypatch.setattr(full_unit_mod.subprocess, "run", fake_run)
        monkeypatch.setenv("PYTEST_ADDOPTS", "--stepwise")
        FullUnitStep().run(ctx_factory())
        assert "-o" in captured["cmd"]
        assert "addopts=" in captured["cmd"]
        assert "not slow and not integration and not needle" in captured["cmd"]
        assert "PYTEST_ADDOPTS" not in (captured["env"] or {})

    def test_early_stop_withholds_downgrade(self, ctx_factory, monkeypatch):
        # codex #1222 r15: completeness is proven STRUCTURALLY from the
        # reporter's session-finish record, not a stdout banner (which
        # false-positived when "Interrupted:" appeared in a test's own
        # traceback — r15 B2). An early stop (-x / --maxfail / --stepwise,
        # which a conftest can set even with addopts cleared) attempts fewer
        # items than it collected: ran < collected → the suite is incomplete,
        # a later regression may not have run, so the downgrade is withheld
        # and every failure blocks.
        self._patch_quarantine(
            monkeypatch, [QuarantineEntry(id="tests/a.py::test_flaky")]
        )
        self._patch_pytest(
            monkeypatch,
            1,
            _summary("FAILED tests/a.py::test_flaky - X"),
            ran=1,
            collected=50,  # 49 collected tests never ran → truncated
        )
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"  # withheld despite being listed
        assert "did not run to completion" in res.details

    def test_hard_truncation_withholds_downgrade(self, ctx_factory, monkeypatch):
        # codex #1222 r15 B1: a test calling os._exit(1) (or a crash /
        # SIGKILL) kills the process mid-suite, so pytest_sessionfinish never
        # fires and NO completion record is written — even though the exit
        # code is 1 and the only logged failure is quarantined. The record's
        # absence proves the run was truncated, so the downgrade is withheld
        # and the failure blocks. This is the case the old stdout heuristic
        # could not see (os._exit prints no banner).
        self._patch_quarantine(
            monkeypatch, [QuarantineEntry(id="tests/a.py::test_flaky")]
        )
        self._patch_pytest(
            monkeypatch,
            1,
            _summary("FAILED tests/a.py::test_flaky - X"),
            session=False,  # process died before sessionfinish → no record
        )
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"  # withheld → blocks despite being listed
        assert "did not run to completion" in res.details

    def test_complete_run_allows_downgrade(self, ctx_factory, monkeypatch):
        # The positive control for r15: a well-formed completion record with
        # ran >= collected proves the whole suite ran, so a solely-quarantined
        # red is correctly downgraded to a (loud) pass.
        self._patch_quarantine(
            monkeypatch, [QuarantineEntry(id="tests/a.py::test_flaky")]
        )
        self._patch_pytest(
            monkeypatch,
            1,
            _summary("FAILED tests/a.py::test_flaky - X"),
            ran=50,
            collected=50,
        )
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "pass"
        assert "known-flaky" in res.summary

    def test_malformed_session_record_withholds_downgrade(
        self, ctx_factory, monkeypatch
    ):
        # A present-but-unparseable completion record (a tampered / unexpected
        # shape) is not proof of completeness — withhold the downgrade, the
        # safe direction (codex #1222 r15).
        self._patch_quarantine(
            monkeypatch, [QuarantineEntry(id="tests/a.py::test_flaky")]
        )
        self._patch_pytest(
            monkeypatch,
            1,
            _summary("FAILED tests/a.py::test_flaky - X"),
            session_value="garbage-not-two-ints",
        )
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"
        assert "did not run to completion" in res.details

    def test_clean_exit_without_completion_blocks(self, ctx_factory, monkeypatch):
        # codex #1222 r16 B1: os._exit(0) can exit pytest with code 0 after
        # skipping the rest of the suite, hiding a regression in the un-run
        # tail. When the reporter ran, even a GREEN exit requires the
        # completion record — its absence proves the run was truncated.
        self._patch_pytest(monkeypatch, 0, _passed(), session=False)
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"
        assert "did not run to completion" in res.details

    def test_clean_exit_early_stop_record_blocks(self, ctx_factory, monkeypatch):
        # Exit 0 with a ran<collected record is still a truncated run (a
        # conftest could stop early yet exit 0) — not accepted as green.
        self._patch_pytest(monkeypatch, 0, _passed(), ran=1, collected=50)
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"
        assert "did not run to completion" in res.details

    def test_clean_exit_with_completion_passes(self, ctx_factory, monkeypatch):
        # Positive control: a green exit WITH a complete record is a clean
        # pass (the plugin ran and the whole suite completed).
        self._patch_pytest(monkeypatch, 0, _passed(), ran=50, collected=50)
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "pass"

    def test_clean_exit_without_plugin_trusts_exit_code(self, ctx_factory, monkeypatch):
        # If the reporter genuinely can't be injected (available() False),
        # there's no completeness signal to check — a green exit falls back to
        # the exit code (the degraded, non-candidate path).
        monkeypatch.setattr(full_unit_mod._nodeid_reporter, "available", lambda: False)
        self._patch_pytest(monkeypatch, 0, _passed(), structured=False)
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "pass"

    def test_clean_exit_injected_but_no_output_blocks(self, ctx_factory, monkeypatch):
        # codex #1222 r17 B1: a hard os._exit(0) before the first report
        # leaves the log UNwritten (not even empty). full_unit pre-creates a
        # start sentinel when it injects the plugin, so injection is known
        # independent of any append — the empty sentinel + exit 0 blocks,
        # instead of being misread as "plugin unavailable → trust exit code".
        # write_log=False simulates the plugin writing nothing at all.
        self._patch_pytest(monkeypatch, 0, _passed(), write_log=False)
        res = FullUnitStep().run(ctx_factory())
        assert res.status == "fail"
        assert "did not run to completion" in res.details


# --------------------------------------------------------------------------
# full_unit._session_completed — structured completeness proof (codex r15)
# --------------------------------------------------------------------------


class TestSessionCompleted:
    """The quarantine downgrade's completeness precondition is proven from
    the reporter's SESSIONFINISH record, replacing the fragile stdout-banner
    heuristic (codex #1222 r15)."""

    def _tsv(self, tmp_path, text):
        p = tmp_path / "nodeids.tsv"
        p.write_text(text, encoding="utf-8")
        return p

    def test_complete_run(self, tmp_path):
        assert full_unit_mod._session_completed(
            self._tsv(tmp_path, "SESSIONFINISH\t50 50\n")
        )

    def test_ran_exceeds_collected_is_complete(self, tmp_path):
        # A benign over-count (e.g. a reran item logs an extra setup) still
        # ran the whole suite — completeness is ran >= collected, not ==.
        assert full_unit_mod._session_completed(
            self._tsv(tmp_path, "SESSIONFINISH\t51 50\n")
        )

    def test_early_stop_is_incomplete(self, tmp_path):
        # ran < collected → -x / --maxfail / --stepwise truncated the suite.
        assert not full_unit_mod._session_completed(
            self._tsv(tmp_path, "SESSIONFINISH\t1 50\n")
        )

    def test_absent_record_is_incomplete(self, tmp_path):
        # os._exit / crash / SIGKILL → sessionfinish never fired → no record.
        assert not full_unit_mod._session_completed(
            self._tsv(tmp_path, "FAILED\ttests/a.py::t\n")
        )

    def test_duplicate_record_is_incomplete(self, tmp_path):
        # An unexpected multi-record shape (xdist / tamper) → withhold, the
        # safe direction.
        assert not full_unit_mod._session_completed(
            self._tsv(tmp_path, "SESSIONFINISH\t50 50\nSESSIONFINISH\t1 50\n")
        )

    def test_identical_duplicate_records_incomplete(self, tmp_path):
        # codex #1222 r16: two IDENTICAL completion records must NOT collapse
        # — report_log_node_ids would dedup them by value and pass the
        # "exactly one" check; the raw line count rejects the duplicate.
        assert not full_unit_mod._session_completed(
            self._tsv(tmp_path, "SESSIONFINISH\t50 50\nSESSIONFINISH\t50 50\n")
        )

    def test_malformed_value_is_incomplete(self, tmp_path):
        assert not full_unit_mod._session_completed(
            self._tsv(tmp_path, "SESSIONFINISH\tgarbage\n")
        )

    def test_zero_collected_is_incomplete(self, tmp_path):
        assert not full_unit_mod._session_completed(
            self._tsv(tmp_path, "SESSIONFINISH\t0 0\n")
        )


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

    def test_reruns_error_nodeids_structured(self, ctx_factory, monkeypatch):
        # codex #1222 r15 NIT: a setup / teardown / collection flake is
        # recorded as ERROR, not FAILED. The advisory tracker must re-run
        # those too (reading only FAILED reported "nothing to classify" and
        # skipped them). Structured-log path: an ERROR id must reach the
        # re-run command.
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text("boom")
        ctx.artifact_path("full-unit-nodeids.tsv").write_text(
            "ERROR\ttests/x.py::errored\n"
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])
        captured = {}

        def _capture(cmd, cwd, timeout, env=None):
            captured["cmd"] = cmd
            return _completed(0, _passed())

        monkeypatch.setattr(flake_mod, "_run_session", _capture)
        FlakeTrackingStep().run(ctx)
        assert "tests/x.py::errored" in captured["cmd"]

    def test_incomplete_rerun_session_skips_classification(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r23: the structured rerun classification is only trusted
        # on a COMPLETE rerun. A truncated rerun (a sampled test's pytest.exit
        # / os._exit / crash) leaves a structured log with NO session-finish
        # record, so its PASSED/FAILED set is partial — an id whose real
        # reproduction was in the un-run tail could be mislabeled a recovered
        # flake and wrongly recommended for a quarantine promotion. When the
        # rerun didn't run to completion the step SKIPS instead.
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text("boom")
        ctx.artifact_path("full-unit-nodeids.tsv").write_text(
            "FAILED\ttests/x.py::test_real\n"
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])

        def fake(cmd, cwd, timeout, env=None):
            # A truncated rerun: test_real logged a PASSED before a mid-session
            # exit, but NO session-finish record was written (deliberately no
            # _RERUN_DONE) — the completeness proof is absent.
            ctx.artifact_path("flake-rerun-nodeids.tsv").write_text(
                "PASSED\ttests/x.py::test_real\n"
            )
            return _completed(0, "")

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "skip"
        assert "did not run to completion" in res.summary
        # No misleading candidate artifact is emitted on the truncated path.
        assert not ctx.artifact_path("flake-candidates.json").exists()

    def test_recovered_collection_error_is_error_candidate(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r17: a collection ERROR id is a MODULE / DIR path (no
        # "::"); on a clean re-run its CONTAINED tests log PASSED under their
        # own "::" ids, but the module id itself never logs a PASSED — so a
        # recovered collection flake would wrongly land in `inconclusive`
        # forever. With POSITIVE evidence (a contained test passed) it is a
        # recovered flake — but as an ERROR-origin one it is NOT quarantinable
        # (full_unit blocks any ERROR run), so it lands in
        # `error_flake_candidates`, not `flake_candidates_new` (codex r18).
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text("boom")
        ctx.artifact_path("full-unit-nodeids.tsv").write_text("ERROR\ttests/mod.py\n")
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])

        def fake(cmd, cwd, timeout, env=None):
            # Clean re-run: the module collected fine and its test passed; the
            # module id itself logs nothing.
            ctx.artifact_path("flake-rerun-nodeids.tsv").write_text(
                "PASSED\ttests/mod.py::test_a\n" + _RERUN_DONE
            )
            return _completed(0, "")

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert "tests/mod.py" in data["error_flake_candidates"]
        assert "tests/mod.py" not in data["flake_candidates_new"]
        assert "tests/mod.py" not in data["inconclusive"]

    def test_collection_target_no_positive_outcome_is_inconclusive(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r18: a collection target that stopped erroring but
        # produced NO positive outcome on re-run (collected nothing, or only
        # skipped) is NOT a proven flake — same positive-outcome rule as any
        # other id. It must land in `inconclusive`, not be falsely recovered.
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text("boom")
        ctx.artifact_path("full-unit-nodeids.tsv").write_text("ERROR\ttests/mod.py\n")
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])

        def fake(cmd, cwd, timeout, env=None):
            # Re-run collected but nothing passed (all skipped) — the session
            # still completed, so only the completion marker is present.
            ctx.artifact_path("flake-rerun-nodeids.tsv").write_text(_RERUN_DONE)
            return _completed(0, "")

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert "tests/mod.py" in data["inconclusive"]
        assert "tests/mod.py" not in data["error_flake_candidates"]
        assert "tests/mod.py" not in data["flake_candidates_new"]

    def test_recovered_setup_error_is_not_quarantinable(self, ctx_factory, monkeypatch):
        # codex #1222 r18: a setup/teardown ERROR flake that recovers is real
        # (non-deterministic) but NOT quarantinable — full_unit blocks any
        # ERROR run unconditionally, so a quarantine entry would never waive
        # it. It lands in error_flake_candidates ("investigate"), never in
        # flake_candidates_new (which recommends quarantine promotion).
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text("boom")
        ctx.artifact_path("full-unit-nodeids.tsv").write_text(
            "ERROR\ttests/x.py::test_boom\n"
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])

        def fake(cmd, cwd, timeout, env=None):
            ctx.artifact_path("flake-rerun-nodeids.tsv").write_text(
                "PASSED\ttests/x.py::test_boom\n" + _RERUN_DONE
            )
            return _completed(0, "")

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"
        assert "not quarantinable" in res.summary
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert data["error_flake_candidates"] == ["tests/x.py::test_boom"]
        assert data["flake_candidates_new"] == []

    def test_reproduced_collection_error_not_candidate(self, ctx_factory, monkeypatch):
        # The inverse: a collection target that RE-ERRORS on re-run reproduces
        # (still_bad) and must NOT be a candidate — badness takes precedence
        # over the collection-recovery rule.
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text("boom")
        ctx.artifact_path("full-unit-nodeids.tsv").write_text("ERROR\ttests/mod.py\n")
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])

        def fake(cmd, cwd, timeout, env=None):
            ctx.artifact_path("flake-rerun-nodeids.tsv").write_text(
                "ERROR\ttests/mod.py\n" + _RERUN_DONE
            )
            return _completed(1, "")

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert "tests/mod.py" not in data["flake_candidates_new"]
        assert "tests/mod.py" in data["reproduced_likely_real"]

    def test_call_fail_plus_teardown_error_is_not_quarantinable(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r19: an id that logs BOTH a FAILED (call) and an ERROR
        # (teardown) in the full suite is still un-waivable — full_unit blocks
        # ANY run containing an ERROR unconditionally. Even though it also
        # appears under FAILED, it must be treated as error-origin and land in
        # error_flake_candidates ("investigate"), NOT flake_candidates_new
        # (which recommends an ineffective quarantine promotion). The r18 code
        # excluded FAILED ids from error_origin and mislabeled this case.
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text("boom")
        ctx.artifact_path("full-unit-nodeids.tsv").write_text(
            "FAILED\ttests/x.py::test_flaky\nERROR\ttests/x.py::test_flaky\n"
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])

        def fake(cmd, cwd, timeout, env=None):
            ctx.artifact_path("flake-rerun-nodeids.tsv").write_text(
                "PASSED\ttests/x.py::test_flaky\n" + _RERUN_DONE
            )
            return _completed(0, "")

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert data["error_flake_candidates"] == ["tests/x.py::test_flaky"]
        assert data["flake_candidates_new"] == []

    def test_recovered_directory_collection_error_is_error_candidate(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r19: a DIRECTORY collection target ("tests/subdir", no
        # ".py") contains its tests at the "/" boundary
        # ("tests/subdir/test_x.py::test_y"), which never satisfies a "::"
        # check — so the r18 code left genuine directory-collection flakes
        # permanently inconclusive. A passed descendant at the "/" boundary is
        # positive recovery evidence; as an ERROR-origin id it is not
        # quarantinable → error_flake_candidates.
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text("boom")
        ctx.artifact_path("full-unit-nodeids.tsv").write_text("ERROR\ttests/subdir\n")
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])

        def fake(cmd, cwd, timeout, env=None):
            ctx.artifact_path("flake-rerun-nodeids.tsv").write_text(
                "PASSED\ttests/subdir/test_x.py::test_y\n" + _RERUN_DONE
            )
            return _completed(0, "")

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert "tests/subdir" in data["error_flake_candidates"]
        assert "tests/subdir" not in data["inconclusive"]
        assert "tests/subdir" not in data["flake_candidates_new"]

    def test_recovered_class_collection_error_is_error_candidate(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r24: a CLASS collection error id CONTAINS "::"
        # ("tests/x.py::TestC"), so the "::"-absence heuristic misses it and
        # the r19 code left it permanently inconclusive. The structured
        # COLLECTERROR label (written alongside ERROR, so the gate is
        # untouched) lets the classifier recognize it; a passed descendant
        # ("tests/x.py::TestC::test_m") is positive recovery evidence. As an
        # ERROR-origin id it is not quarantinable → error_flake_candidates.
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text("boom")
        ctx.artifact_path("full-unit-nodeids.tsv").write_text(
            "ERROR\ttests/x.py::TestC\nCOLLECTERROR\ttests/x.py::TestC\n"
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])

        def fake(cmd, cwd, timeout, env=None):
            ctx.artifact_path("flake-rerun-nodeids.tsv").write_text(
                "PASSED\ttests/x.py::TestC::test_m\n" + _RERUN_DONE
            )
            return _completed(0, "")

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert "tests/x.py::TestC" in data["error_flake_candidates"]
        assert "tests/x.py::TestC" not in data["inconclusive"]
        assert "tests/x.py::TestC" not in data["flake_candidates_new"]

    def test_error_origin_reproduced_is_not_reproduced_quarantined(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r24: an ERROR-origin id that is ALSO quarantined and
        # reproduces must NOT be labeled reproduced_quarantined ("full_unit
        # PASSED as quarantined") — full_unit BLOCKS any ERROR run regardless
        # of the registry, so the quarantine never waived it. Route it to
        # reproduced_likely_real (the gate DID block it).
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text("boom")
        ctx.artifact_path("full-unit-nodeids.tsv").write_text(
            "ERROR\ttests/x.py::test_e\n"
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(
            flake_mod,
            "load_quarantine_from_ref",
            lambda *a, **k: [QuarantineEntry(id="tests/x.py::test_e")],
        )

        def fake(cmd, cwd, timeout, env=None):
            ctx.artifact_path("flake-rerun-nodeids.tsv").write_text(
                "ERROR\ttests/x.py::test_e\n" + _RERUN_DONE
            )
            return _completed(1, "")

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        FlakeTrackingStep().run(ctx)
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert "tests/x.py::test_e" in data["reproduced_likely_real"]
        assert "tests/x.py::test_e" not in data["reproduced_quarantined"]

    def test_dir_collection_target_not_recovered_by_sibling_prefix(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r19 boundary guard: a passed id that merely shares a
        # string prefix with a dir target but sits at NEITHER a "::" nor a "/"
        # boundary ("tests/foo.py::test" vs dir "tests/foo") is NOT a
        # descendant and must not count as recovery — the id stays
        # inconclusive, not a false error-flake.
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text("boom")
        ctx.artifact_path("full-unit-nodeids.tsv").write_text("ERROR\ttests/foo\n")
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])

        def fake(cmd, cwd, timeout, env=None):
            ctx.artifact_path("flake-rerun-nodeids.tsv").write_text(
                "PASSED\ttests/foo.py::test_sibling\n" + _RERUN_DONE
            )
            return _completed(0, "")

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert "tests/foo" in data["inconclusive"]
        assert "tests/foo" not in data["error_flake_candidates"]

    def test_reruns_error_nodeids_fallback(self, ctx_factory, monkeypatch):
        # Same NIT via the terminal-summary FALLBACK (no structured log): an
        # ERROR line in full-unit.log must be picked up for the advisory
        # re-run, not dropped.
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text(
            _summary("ERROR tests/x.py::errored - fixture boom")
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])
        captured = {}

        def _capture(cmd, cwd, timeout, env=None):
            captured["cmd"] = cmd
            return _completed(0, _passed())

        monkeypatch.setattr(flake_mod, "_run_session", _capture)
        FlakeTrackingStep().run(ctx)
        assert "tests/x.py::errored" in captured["cmd"]

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

    def test_rerun_forces_color_off(self, ctx_factory, monkeypatch):
        # The advisory re-run must also disable color so ANSI escapes can't
        # break its outcome parsing (codex #1222 r6).
        ctx = ctx_factory()
        self._prime(ctx, monkeypatch)
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])
        captured = {}

        def fake(cmd, cwd, timeout, env=None):
            captured["cmd"] = cmd
            return _completed(0, _passed())

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        FlakeTrackingStep().run(ctx)
        assert "--color=no" in captured["cmd"]
        assert "-rA" in captured["cmd"]

    def test_rerun_loads_rerunfailures_explicitly_when_autoload_disabled(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r18: find_spec confirms rerunfailures is installed but
        # not that it's ACTIVE — under PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 an
        # installed plugin won't register, so --reruns would be unknown and
        # the advisory would silently skip. In THAT case only, load it
        # explicitly with -p so --reruns is recognized.
        monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        ctx = ctx_factory()
        self._prime(ctx, monkeypatch)
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])
        captured = {}

        def fake(cmd, cwd, timeout, env=None):
            captured["cmd"] = cmd
            return _completed(0, _passed())

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        FlakeTrackingStep().run(ctx)
        cmd = captured["cmd"]
        # -p pytest_rerunfailures appears as adjacent argv entries.
        assert "pytest_rerunfailures" in cmd
        assert cmd[cmd.index("pytest_rerunfailures") - 1] == "-p"

    def test_rerun_omits_explicit_rerunfailures_when_autoload_on(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r18 REGRESSION (fixed r19): with autoload ON,
        # rerunfailures self-registers under the entry-point name
        # "rerunfailures". Passing an extra ``-p pytest_rerunfailures`` then
        # makes pluggy raise "Plugin already registered under a different
        # name" and crashes the advisory re-run entirely. So the explicit
        # ``-p`` MUST be absent unless autoload is disabled.
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        ctx = ctx_factory()
        self._prime(ctx, monkeypatch)
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])
        captured = {}

        def fake(cmd, cwd, timeout, env=None):
            captured["cmd"] = cmd
            return _completed(0, _passed())

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        FlakeTrackingStep().run(ctx)
        assert "pytest_rerunfailures" not in captured["cmd"]

    def test_rerun_neutralizes_candidate_addopts_and_env(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r16: the advisory re-run must be hermetic like full_unit
        # — clear ini addopts (-o addopts=) and strip PYTEST_ADDOPTS so a
        # candidate can't suppress classification via --collect-only / a
        # marker filter / -p no:<reporter>.
        ctx = ctx_factory()
        self._prime(ctx, monkeypatch)
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])
        monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
        captured = {}

        def fake(cmd, cwd, timeout, env=None):
            captured["cmd"] = cmd
            captured["env"] = env
            return _completed(0, _passed())

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        FlakeTrackingStep().run(ctx)
        assert "-o" in captured["cmd"]
        assert "addopts=" in captured["cmd"]
        assert "PYTEST_ADDOPTS" not in (captured["env"] or {})

    def test_quarantined_reproduction_is_advisory_loud(self, ctx_factory, monkeypatch):
        # A quarantined test that reproduces on EVERY re-run MAY be a
        # deterministic regression the quarantine is masking — but
        # correlated in-process re-runs can't prove that (a sustained-
        # contention flake looks identical), so flake_tracking does NOT
        # auto-block (codex #1222 r8, reverting r7's hard gate). It surfaces
        # the case LOUDLY for human de-quarantine: status stays advisory
        # (pass), the summary leads with a REVIEW flag, and the id lands in
        # `reproduced_quarantined`. A non-quarantined reproduction still
        # lands in `reproduced_likely_real` (full_unit already blocked it).
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text(
            _summary(
                "FAILED tests/x.py::flaky - X",
                "FAILED tests/x.py::real - X",
            )
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(
            flake_mod,
            "load_quarantine_from_ref",
            lambda *a, **k: [QuarantineEntry(id="tests/x.py::flaky")],
        )
        rerun = (
            "==================== short test summary info ====================\n"
            "FAILED tests/x.py::flaky - AssertionError\n"
            "FAILED tests/x.py::real - AssertionError\n"
            "==== 2 failed in 0.20s ====\n"
        )
        monkeypatch.setattr(
            flake_mod, "_run_session", lambda *a, **k: _completed(1, rerun)
        )
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"  # advisory — never auto-blocks (r8)
        assert "REVIEW" in res.summary
        assert "de-quarantine" in res.summary
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert data["reproduced_likely_real"] == ["tests/x.py::real"]
        assert data["reproduced_quarantined"] == ["tests/x.py::flaky"]

    def test_quarantined_flake_confirmed_does_not_block(self, ctx_factory, monkeypatch):
        # The inverse: a quarantined test that PASSES on re-run is a genuine
        # flake — the quarantine downgrade was correct, so we do NOT block;
        # it's reported as a confirmed known-flake (advisory pass).
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text(
            _summary("FAILED tests/x.py::flaky - X")
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(
            flake_mod,
            "load_quarantine_from_ref",
            lambda *a, **k: [QuarantineEntry(id="tests/x.py::flaky")],
        )
        rerun = (
            "==================== short test summary info ====================\n"
            "PASSED tests/x.py::flaky\n"
            "==== 1 passed in 0.20s ====\n"
        )
        monkeypatch.setattr(
            flake_mod, "_run_session", lambda *a, **k: _completed(0, rerun)
        )
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"  # confirmed flake → downgrade was right
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert data["flake_candidates_known"] == ["tests/x.py::flaky"]
        assert data["reproduced_quarantined"] == []

    def test_dequarantined_flake_classified_against_effective_intersection(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r22: the advisory must classify against the SAME
        # effective quarantine (base ∩ candidate) full_unit enforces. A test
        # the PR DE-QUARANTINED (in base, removed in candidate) that reproduces
        # must NOT be mislabeled "known-flaky" / reproduced_quarantined —
        # full_unit actually BLOCKED it, so the advisory reports it as a real
        # reproduction, matching the gate instead of contradicting it.
        ctx = ctx_factory()
        ctx.head_sha = "headsha1111"
        ctx.artifact_path("full-unit.log").write_text(
            _summary("FAILED tests/x.py::flaky - X")
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )

        def fake_loader(ref, *a, **k):
            # base still lists it; candidate (head) removed it → de-quarantined.
            if ref == "headsha1111":
                return []
            return [QuarantineEntry(id="tests/x.py::flaky")]

        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", fake_loader)
        rerun = (
            "==================== short test summary info ====================\n"
            "FAILED tests/x.py::flaky\n"
            "==== 1 failed in 0.20s ====\n"
        )
        monkeypatch.setattr(
            flake_mod, "_run_session", lambda *a, **k: _completed(1, rerun)
        )
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"  # advisory never blocks
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        # De-quarantined → NOT known / not reproduced_quarantined; it's a real
        # reproduction, same as full_unit treated it.
        assert data["flake_candidates_known"] == []
        assert data["reproduced_quarantined"] == []
        assert "tests/x.py::flaky" in data["reproduced_likely_real"]

    def test_teardown_error_after_pass_is_reproduced_not_flake(
        self, ctx_factory, monkeypatch
    ):
        # codex #1222 r12 BLOCKING: the structured reporter logs one line per
        # pytest PHASE, so a test whose call PASSES but whose teardown ERRORs
        # emits BOTH a PASSED and an ERROR line for the same node id. That run
        # ended badly — it must classify as a reproduction, never a flake
        # candidate. FAILED/ERROR takes precedence over PASSED, keeping the
        # buckets disjoint.
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text(
            _summary("FAILED tests/x.py::td - X")
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(flake_mod, "load_quarantine_from_ref", lambda *a, **k: [])

        def fake(cmd, cwd, timeout, env=None):
            # Emulate the plugin writing PASSED (call) + ERROR (teardown) for
            # the same id — the exact double-log codex flagged.
            ctx.artifact_path("flake-rerun-nodeids.tsv").write_text(
                "PASSED\ttests/x.py::td\nERROR\ttests/x.py::td\n" + _RERUN_DONE
            )
            return _completed(1, "")

        monkeypatch.setattr(flake_mod, "_run_session", fake)
        res = FlakeTrackingStep().run(ctx)
        assert res.status == "pass"
        data = json.loads(ctx.artifact_path("flake-candidates.json").read_text())
        assert "tests/x.py::td" not in data["flake_candidates_new"]
        assert "tests/x.py::td" in data["reproduced_likely_real"]

    def test_quarantined_failure_always_sampled_past_cap(
        self, ctx_factory, monkeypatch
    ):
        # A quarantined failure is the highest-value review signal, so it
        # must be re-run even when it sorts past the _MAX_RERUN_IDS cap —
        # otherwise the graveyard-review report silently drops it (codex
        # #1222 r8 nit). Put the quarantined id LAST among 40 failures and
        # assert it still reaches the re-run command.
        ids = [f"tests/x.py::t{i}" for i in range(40)]
        q_id = ids[-1]
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text(
            _summary(*[f"FAILED {i} - X" for i in ids])
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(
            flake_mod,
            "load_quarantine_from_ref",
            lambda *a, **k: [QuarantineEntry(id=q_id)],
        )
        captured = {}

        def _capture(cmd, cwd, timeout, env=None):
            captured["cmd"] = cmd
            return _completed(0, _passed())

        monkeypatch.setattr(flake_mod, "_run_session", _capture)
        FlakeTrackingStep().run(ctx)
        assert q_id in captured["cmd"]  # quarantined id survived the cap
        # ...and the cap still bounds the total re-run set.
        rerun_ids = [c for c in captured["cmd"] if c.startswith("tests/x.py::")]
        assert len(rerun_ids) == flake_mod._MAX_RERUN_IDS

    def test_all_quarantined_sampled_even_past_cap(self, ctx_factory, monkeypatch):
        # When quarantined failures ALONE exceed the cap, every one is still
        # re-run — the cap bounds only the non-quarantined remainder (codex
        # #1222 r9 nit): dropping a quarantined failure would silently skip
        # the graveyard check the step promises.
        n_q = flake_mod._MAX_RERUN_IDS + 5
        q_ids = [f"tests/x.py::q{i}" for i in range(n_q)]
        other_ids = [f"tests/x.py::o{i}" for i in range(10)]
        ctx = ctx_factory()
        ctx.artifact_path("full-unit.log").write_text(
            _summary(*[f"FAILED {i} - X" for i in q_ids + other_ids])
        )
        monkeypatch.setattr(
            flake_mod.importlib.util, "find_spec", lambda name: object()
        )
        monkeypatch.setattr(
            flake_mod,
            "load_quarantine_from_ref",
            lambda *a, **k: [QuarantineEntry(id=i) for i in q_ids],
        )
        captured = {}

        def _capture(cmd, cwd, timeout, env=None):
            captured["cmd"] = cmd
            return _completed(0, _passed())

        monkeypatch.setattr(flake_mod, "_run_session", _capture)
        FlakeTrackingStep().run(ctx)
        rerun = set(captured["cmd"])
        assert all(q in rerun for q in q_ids)  # every quarantined id re-run
        assert not any(o in rerun for o in other_ids)  # cap spent on quarantined


class TestFlakeTrackingClassification:
    """Drive a real pytest subprocess so the flake-vs-real split is
    proven end-to-end, not mocked."""

    def test_classifies_flake_vs_real_fallback(
        self, ctx_factory, tmp_path, monkeypatch
    ):
        # Exercises the terminal-summary FALLBACK classification path with a
        # real subprocess. The structured-plugin path can't load here (the
        # fake repo root has no `scripts` package), so force the fallback;
        # the dedicated plugin round-trip is proven in TestNodeidReporter.
        pytest.importorskip("pytest_rerunfailures")
        monkeypatch.setattr(flake_mod._nodeid_reporter, "available", lambda: False)
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
# _nodeid_reporter — structured node-id log (the parse-ambiguity fix)
# --------------------------------------------------------------------------


class TestFieldEscaping:
    """The log's value fields are escape_field-encoded so a hostile node id
    can't inject a delimiter and forge a record (codex #1222 r23)."""

    @pytest.mark.parametrize(
        "value",
        [
            "tests/a.py::test_x",
            "tests/a.py::test_x[a - b]",
            "evil\nSESSIONFINISH\t9999 1",
            "tab\there",
            "cr\rhere",
            "back\\slash",
            "escaped\\tnot_a_tab",
            "trailing\\",
            "50 50",
            "",
        ],
    )
    def test_round_trip(self, value):
        enc = _nodeid_reporter.escape_field(value)
        # No raw delimiter survives encoding — the whole point.
        assert "\n" not in enc and "\t" not in enc
        assert _nodeid_reporter.unescape_field(enc) == value

    def test_escaped_backslash_does_not_pair_with_following_letter(self):
        # "\\t" (a literal backslash then a 't') must decode to backslash+'t',
        # NOT a tab — otherwise a node id containing a literal backslash could
        # still smuggle a control char through.
        assert _nodeid_reporter.unescape_field("\\\\t") == "\\t"
        assert _nodeid_reporter.unescape_field("\\\\n") == "\\n"

    def test_unknown_escape_preserved_verbatim(self):
        # The writer never emits "\z"; a raw/hand-written record keeps it
        # intact rather than being corrupted.
        assert _nodeid_reporter.unescape_field("a\\zb") == "a\\zb"

    def test_no_backslash_is_identity_fast_path(self):
        assert _nodeid_reporter.unescape_field("plain::id") == "plain::id"

    def test_reader_not_fooled_by_injected_value(self, tmp_path, monkeypatch):
        # End-to-end at the writer/reader boundary: append a malicious id via
        # the REAL _append, then prove the readers see one FAILED id and ZERO
        # session records (the injected SESSIONFINISH is inert).
        log = tmp_path / "n.tsv"
        monkeypatch.setenv("PR_VALIDATE_NODEID_LOG", str(log))
        _nodeid_reporter._append("FAILED", "evil\nSESSIONFINISH\t9999 1")
        assert raw_session_records(log) == []
        assert report_log_node_ids(log, "FAILED") == ["evil\nSESSIONFINISH\t9999 1"]

    def test_all_splitlines_separators_neutralized(self):
        # codex #1222 r28: str.splitlines() splits on more than \n/\r — a
        # hostile id embedding U+000B/000C/001C-E/0085/2028/2029 would forge a
        # record for a reader using splitlines() (the log readers do). Each such
        # separator must encode to a single "line" under BOTH splitlines() AND
        # split("\n"), and round-trip losslessly.
        seps = [
            "\v",
            "\f",
            "\x1c",
            "\x1d",
            "\x1e",
            "\x85",
            chr(0x2028),
            chr(0x2029),
        ]
        for sep in seps:
            payload = f"evil{sep}SESSIONFINISH\t9999 1"
            enc = _nodeid_reporter.escape_field(payload)
            assert len(enc.splitlines()) == 1, (hex(ord(sep)), enc)
            assert len(enc.split("\n")) == 1
            assert _nodeid_reporter.unescape_field(enc) == payload

    def test_unicode_separator_injection_inert_end_to_end(self, tmp_path, monkeypatch):
        # A U+2028-bearing id written via the REAL _append yields exactly one
        # FAILED record and ZERO forged SESSIONFINISH when read back through the
        # splitlines()-based readers (codex #1222 r28).
        log = tmp_path / "n.tsv"
        monkeypatch.setenv("PR_VALIDATE_NODEID_LOG", str(log))
        evil = f"evil{chr(0x2028)}SESSIONFINISH\t9999 1"
        _nodeid_reporter._append("FAILED", evil)
        assert raw_session_records(log) == []
        assert report_log_node_ids(log, "FAILED") == [evil]

    def test_malformed_unicode_escape_preserved(self):
        # "\u12" (fewer than four hex digits) is not a valid encoded separator;
        # a hand-written record keeps it verbatim rather than mis-decoding.
        assert _nodeid_reporter.unescape_field("a\\u12b") == "a\\u12b"
        assert _nodeid_reporter.unescape_field("x\\uZZZZy") == "x\\uZZZZy"


class TestRenderFenceSafe:
    """A node id interpolated into a Markdown ``` code fence must not be able
    to break out of it and spoof scorecard content (codex #1222 r24)."""

    def test_collapses_newlines_to_single_line(self):
        evil = "tests/a.py::t[a\n```\n# spoof]"
        out = render_fence_safe(evil)
        # No embedded newline survives → no line can start a fence-closer.
        assert "\n" not in out
        assert out.startswith('"') and out.endswith('"')
        # Data fidelity: the original id is recoverable via JSON.
        assert json.loads(out) == evil

    def test_preserves_unicode(self):
        assert render_fence_safe("tests/tûv.py::tîme") == '"tests/tûv.py::tîme"'

    def test_full_unit_render_details_is_fence_safe(self):
        evil = "tests/a.py::test_x[a\n```\n# spoof]"
        out = full_unit_mod._render_details([evil], [], "/tmp/log")
        assert json.dumps(evil, ensure_ascii=False) in out
        # The raw fence-breaking newline sequence must NOT leak into the block.
        assert "a\n```\n# spoof" not in out

    def test_flake_render_is_fence_safe(self):
        evil = "tests/a.py::test_y[b\n```\n# spoof]"
        out = flake_mod._render([evil], [], [], [], [], [], sampled=1, total=1)
        assert json.dumps(evil, ensure_ascii=False) in out
        assert "b\n```\n# spoof" not in out


class TestNodeidReporter:
    """Prove the plugin round-trips real ``report.nodeid`` values through a
    genuine pytest subprocess, using the ACTUAL repo root on PYTHONPATH so
    the risky ``-p scripts.pr_validate._nodeid_reporter`` import is exercised
    end-to-end (not mocked). This is what lets full_unit/flake_tracking stop
    scraping the ambiguous terminal summary."""

    def _repo_root(self):
        # _nodeid_reporter.py lives at <root>/scripts/pr_validate/ — climb two.
        import pathlib

        return pathlib.Path(flake_mod._nodeid_reporter.__file__).resolve().parents[2]

    def test_plugin_writes_structured_log(self, tmp_path):
        (tmp_path / "test_outcomes.py").write_text(
            textwrap.dedent(
                """
                import pytest

                def test_ok():
                    assert True

                # A node id that would defeat any terminal-summary text rule:
                # it contains ' - ' and unbalanced brackets. report.nodeid is
                # exact, so the structured log must capture it verbatim.
                @pytest.mark.parametrize("x", ["a - b]"])
                def test_param(x):
                    assert False

                @pytest.fixture
                def broken():
                    raise RuntimeError("setup boom")

                def test_setup_error(broken):
                    assert True
                """
            )
        )
        # A second module that fails to even collect → ERROR (collection).
        (tmp_path / "test_uncollectable.py").write_text(
            "import this_module_does_not_exist_xyz  # noqa: F401\n"
        )

        log_path = tmp_path / "nodeids.tsv"
        repo_root = self._repo_root()
        assert flake_mod._nodeid_reporter.available()
        plugin_args, env = flake_mod._nodeid_reporter.build_invocation(
            log_path, repo_root, log_passes=True
        )

        subprocess.run(
            # --continue-on-collection-errors so the collection ERROR and the
            # call/setup outcomes are all recorded in ONE run (pytest would
            # otherwise abort the whole session on the first collection error).
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:randomly",
                "--continue-on-collection-errors",
                *plugin_args,
                "test_outcomes.py",
                "test_uncollectable.py",
            ],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )

        assert log_path.exists(), "plugin never wrote the structured log"
        lines = log_path.read_text(encoding="utf-8").splitlines()
        rows = [tuple(ln.split("\t", 1)) for ln in lines if "\t" in ln]

        passed = {nid for label, nid in rows if label == "PASSED"}
        failed = {nid for label, nid in rows if label == "FAILED"}
        errored = {nid for label, nid in rows if label == "ERROR"}

        assert "test_outcomes.py::test_ok" in passed
        # The bracket/dash-laden parametrization is captured EXACTLY — the
        # whole point of the structured reporter over summary scraping.
        assert "test_outcomes.py::test_param[a - b]]" in failed
        # Setup-phase failure is an ERROR, not a FAILED (mirrors -rfE).
        assert any("test_setup_error" in nid for nid in errored)
        # Collection failure is an ERROR keyed on the uncollectable module.
        assert any("test_uncollectable.py" in nid for nid in errored)
        # A pass is never mis-logged as a failure.
        assert "test_outcomes.py::test_ok" not in failed

    def test_hostile_param_id_cannot_forge_records(self, tmp_path):
        # codex #1222 r23: a hostile pytest_make_parametrize_id can embed a
        # literal newline/tab in report.nodeid (verified — pytest preserves
        # both). Written RAW, the id would forge a second log record — a fake
        # SESSIONFINISH masking a truncated run, or a fake PASSED faking a
        # recovery. escape_field keeps the value on ONE physical line, so the
        # reader sees the exact id and NO injected record.
        (tmp_path / "conftest.py").write_text(
            textwrap.dedent(
                """
                def pytest_make_parametrize_id(config, val, argname):
                    return val
                """
            )
        )
        (tmp_path / "test_inj.py").write_text(
            textwrap.dedent(
                """
                import pytest

                # A param id carrying a newline + tab that spells a forged
                # completion record: "SESSIONFINISH\\t9999 1".
                INJECT = "evil\\nSESSIONFINISH\\t9999 1"

                @pytest.mark.parametrize("x", [INJECT])
                def test_p(x):
                    assert False
                """
            )
        )
        log_path = tmp_path / "nodeids.tsv"
        plugin_args, env = flake_mod._nodeid_reporter.build_invocation(
            log_path, self._repo_root(), log_passes=True
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:randomly",
                *plugin_args,
                "test_inj.py",
            ],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        assert log_path.exists()
        # Exactly ONE genuine SESSIONFINISH record — the injected
        # "SESSIONFINISH\t9999 1" is escaped inside the FAILED value, not a
        # record of its own, so it can't forge a second completion line.
        assert len(raw_session_records(log_path)) == 1
        # The real record (1 item collected, 1 ran) proves completeness; the
        # injection did NOT fabricate an early-stop/garbage record.
        assert full_unit_mod._session_completed(log_path)
        failed = report_log_node_ids(log_path, "FAILED")
        # The malicious id round-trips EXACTLY — newline + tab intact, as ONE
        # id, not two records.
        assert any("evil\nSESSIONFINISH\t9999 1" in nid for nid in failed)

    def test_passes_not_logged_without_opt_in(self, tmp_path):
        (tmp_path / "test_only_pass.py").write_text("def test_ok():\n    assert True\n")
        log_path = tmp_path / "nodeids.tsv"
        repo_root = self._repo_root()
        # log_passes defaults False → the 14k-pass full suite stays lean.
        plugin_args, env = flake_mod._nodeid_reporter.build_invocation(
            log_path, repo_root
        )
        subprocess.run(
            [sys.executable, "-m", "pytest", *plugin_args, "test_only_pass.py"],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        # No failures/errors and passes opted out → nothing to write. The
        # file may be absent or empty; either way, no PASSED rows.
        if log_path.exists():
            assert "PASSED" not in log_path.read_text(encoding="utf-8")

    def test_log_passes_false_clears_inherited_flag(self, tmp_path):
        # codex #1222 r12 nit: log_passes=False must POP an inherited
        # PR_VALIDATE_NODEID_LOG_PASSES=1, else full_unit (which passes
        # False) would log all ~14k passes and bloat the artifact.
        _, env_off = flake_mod._nodeid_reporter.build_invocation(
            tmp_path / "n.tsv",
            tmp_path,
            base_env={"PR_VALIDATE_NODEID_LOG_PASSES": "1", "PATH": "/usr/bin"},
        )
        assert "PR_VALIDATE_NODEID_LOG_PASSES" not in env_off
        # log_passes=True still turns it on.
        _, env_on = flake_mod._nodeid_reporter.build_invocation(
            tmp_path / "n.tsv", tmp_path, log_passes=True, base_env={}
        )
        assert env_on["PR_VALIDATE_NODEID_LOG_PASSES"] == "1"

    def test_reruns_log_only_terminal_outcome(self, tmp_path):
        # The reporter is consumed downstream WITH pytest-rerunfailures, so a
        # flaky test's INTERMEDIATE attempts must never leak into the log: a
        # fail-then-pass test has to appear ONLY as PASSED (never also FAILED)
        # and a fail-then-pass-in-SETUP test likewise, else flake_tracking
        # would classify one node id as flake AND reproduction at once. This
        # locks that behavior against a future rerunfailures change — pytest
        # marks retried attempts outcome=="rerun", which the plugin's
        # failed/passed matchers exclude by construction.
        pytest.importorskip("pytest_rerunfailures")
        (tmp_path / "test_flaky.py").write_text(
            textwrap.dedent(
                """
                import pathlib

                _call = pathlib.Path(__file__).with_name('.call')
                _setup = pathlib.Path(__file__).with_name('.setup')

                def _bump(p):
                    n = int(p.read_text()) if p.exists() else 0
                    p.write_text(str(n + 1))
                    return n

                def test_call_flake():
                    assert _bump(_call) >= 2          # fails attempts 0,1,2, passes on 3rd

                import pytest

                @pytest.fixture
                def flaky_setup():
                    if _bump(_setup) < 1:             # setup fails once, then passes
                        raise RuntimeError("setup flake")

                def test_setup_flake(flaky_setup):
                    assert True

                def test_always_bad():
                    assert False                       # exhausts reruns → terminal FAILED
                """
            )
        )
        log_path = tmp_path / "nodeids.tsv"
        repo_root = self._repo_root()
        plugin_args, env = flake_mod._nodeid_reporter.build_invocation(
            log_path, repo_root, log_passes=True
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:randomly",
                "--reruns=3",
                *plugin_args,
                "test_flaky.py",
            ],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        rows = [
            tuple(ln.split("\t", 1))
            for ln in log_path.read_text(encoding="utf-8").splitlines()
            if "\t" in ln
        ]
        passed = [nid for label, nid in rows if label == "PASSED"]
        failed = [nid for label, nid in rows if label == "FAILED"]
        errored = [nid for label, nid in rows if label == "ERROR"]
        rerun = [nid for label, nid in rows if label == "RERUN"]

        # Eventually-passing flakes: PASSED exactly once, never FAILED/ERROR.
        for nid in (
            "test_flaky.py::test_call_flake",
            "test_flaky.py::test_setup_flake",
        ):
            assert passed.count(nid) == 1, (nid, rows)
            assert nid not in failed
            assert nid not in errored
        # The always-failing test: terminal FAILED exactly once, never PASSED.
        assert failed.count("test_flaky.py::test_always_bad") == 1
        assert "test_flaky.py::test_always_bad" not in passed
        # codex #1222 r23: every retried attempt logs a RERUN record so
        # full_unit can detect a smuggled pytest-rerunfailures NAME-
        # INDEPENDENTLY (a gating run disables reruns, so any RERUN is a
        # tamper signal). All three tests here are retried, so RERUN records
        # exist — including the always-bad one that exhausts its reruns.
        assert rerun, ("expected RERUN records under --reruns", rows)
        assert "test_flaky.py::test_always_bad" in rerun

    def test_strict_xpass_logged_failed_and_carries_no_wasxfail(self, tmp_path):
        # codex #1222 r27 (finding 2) REBUTTAL, locked as committed evidence.
        # Claim: a strict-xfail test that unexpectedly PASSES is logged FAILED
        # and, if quarantined, wrongly waived — "fix: detect report.wasxfail".
        # EMPIRICALLY a strict XPASS report has outcome=='failed' but NO
        # wasxfail (only the NON-strict xpass and the as-expected xfail carry
        # it), so the proposed detection can't fire — the premise is false. The
        # scenario is also not reachable: the allowlist is base-protected
        # (candidates can't add), and a flaky (quarantined) test that is ALSO
        # xfail(strict=True) is self-contradictory. A conftest captures the
        # report to prove the wasxfail attribute is unset.
        (tmp_path / "conftest.py").write_text(
            textwrap.dedent(
                """
                import pathlib
                _obs = pathlib.Path(__file__).with_name('.obs')

                def pytest_runtest_logreport(report):
                    if (report.when == 'call'
                            and report.nodeid.endswith('::test_strict_xpass')):
                        wx = getattr(report, 'wasxfail', '<UNSET>')
                        _obs.write_text(f'{report.outcome}|{wx!r}')
                """
            )
        )
        (tmp_path / "test_xp.py").write_text(
            textwrap.dedent(
                """
                import pytest

                @pytest.mark.xfail(strict=True, reason='exp')
                def test_strict_xpass():
                    assert True  # unexpectedly passes -> strict -> XPASS-as-fail
                """
            )
        )
        log_path = tmp_path / "nodeids.tsv"
        repo_root = self._repo_root()
        plugin_args, env = flake_mod._nodeid_reporter.build_invocation(
            log_path, repo_root
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:randomly",
                "-o",
                "addopts=",
                *plugin_args,
                "test_xp.py",
            ],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        rows = [
            tuple(ln.split("\t", 1))
            for ln in log_path.read_text(encoding="utf-8").splitlines()
            if "\t" in ln
        ]
        failed = [nid for label, nid in rows if label == "FAILED"]
        # (a) The strict XPASS is recorded as a plain FAILED …
        assert "test_xp.py::test_strict_xpass" in failed
        # (b) … and the report carried NO wasxfail, so codex's wasxfail-based
        # detection is inapplicable — the finding's premise is empirically false.
        outcome, wasxfail = (tmp_path / ".obs").read_text().split("|", 1)
        assert outcome == "failed"
        assert wasxfail == "'<UNSET>'"

    def _session_records(self, log_path):
        if not log_path.exists():
            return []
        return [
            ln.split("\t", 1)[1]
            for ln in log_path.read_text(encoding="utf-8").splitlines()
            if ln.startswith(_nodeid_reporter.SESSION_LABEL + "\t")
        ]

    def test_session_finish_record_on_complete_run(self, tmp_path):
        # codex #1222 r15: a graceful, complete run writes exactly ONE
        # SESSIONFINISH record with ran == collected — the structural proof
        # full_unit requires before granting a quarantine downgrade.
        (tmp_path / "test_three.py").write_text(
            "def test_a(): assert True\n"
            "def test_b(): assert True\n"
            "def test_c(): assert True\n"
        )
        log_path = tmp_path / "nodeids.tsv"
        plugin_args, env = flake_mod._nodeid_reporter.build_invocation(
            log_path, self._repo_root()
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:randomly",
                *plugin_args,
                "test_three.py",
            ],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        assert self._session_records(log_path) == ["3 3"]

    def test_session_finish_absent_on_hard_exit(self, tmp_path):
        # codex #1222 r15 B1: a test calling os._exit kills the process
        # before pytest_sessionfinish, so the record is never written — this
        # ABSENCE is how full_unit detects a truncated run.
        (tmp_path / "test_hardexit.py").write_text(
            "import os\n"
            "def test_a(): assert True\n"
            "def test_boom(): os._exit(1)\n"
            "def test_c(): assert True\n"
        )
        log_path = tmp_path / "nodeids.tsv"
        plugin_args, env = flake_mod._nodeid_reporter.build_invocation(
            log_path, self._repo_root()
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:randomly",
                "-p",
                "no:cacheprovider",
                *plugin_args,
                "test_hardexit.py",
            ],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        assert self._session_records(log_path) == []  # hook never fired

    def test_session_finish_early_stop_records_gap(self, tmp_path):
        # codex #1222 r15: -x stops after the first failure, so the record
        # carries ran < collected — full_unit reads that gap as incomplete.
        (tmp_path / "test_earlystop.py").write_text(
            "def test_a(): assert False\n"
            "def test_b(): assert True\n"
            "def test_c(): assert True\n"
        )
        log_path = tmp_path / "nodeids.tsv"
        plugin_args, env = flake_mod._nodeid_reporter.build_invocation(
            log_path, self._repo_root()
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:randomly",
                "-x",
                *plugin_args,
                "test_earlystop.py",
            ],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        assert self._session_records(log_path) == ["1 3"]

    def test_session_finish_mid_call_exit_records_gap(self, tmp_path):
        # codex #1222 r20: the LAST collected item calls pytest.exit(0) from
        # its CALL body. It emits a setup report but NO teardown, and pytest
        # ends gracefully with exitstatus 0. Counting on SETUP would record
        # ran == collected — a phantom "ran" for an item whose body never
        # finished — and full_unit would accept a truncated run as green.
        # Counting on TEARDOWN (terminal) records ran < collected, so the gap
        # is visible and the quarantine downgrade is withheld. This is the
        # exact hole that setup-counting missed.
        (tmp_path / "test_midexit.py").write_text(
            "import pytest\n"
            "def test_a(): assert True\n"
            "def test_b(): assert True\n"
            "def test_c(): pytest.exit('bail', returncode=0)\n"
        )
        log_path = tmp_path / "nodeids.tsv"
        plugin_args, env = flake_mod._nodeid_reporter.build_invocation(
            log_path, self._repo_root()
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:randomly",
                "-p",
                "no:cacheprovider",
                *plugin_args,
                "test_midexit.py",
            ],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0  # pytest.exit(returncode=0) → graceful zero
        # test_c exited mid-call → no teardown for it → ran(2) < collected(3).
        assert self._session_records(log_path) == ["2 3"]

    def test_session_finish_reruns_do_not_inflate(self, tmp_path):
        # codex #1222 r16 B2: under --reruns a flaky test emits SEVERAL setup
        # reports for the same id, so a raw setup COUNT would inflate ran past
        # collected and mask a truncated run. The record counts DISTINCT node
        # ids, so a complete run stays ran == collected despite the retries.
        pytest.importorskip("pytest_rerunfailures")
        (tmp_path / "test_reruns.py").write_text(
            "import pathlib\n"
            "_c = pathlib.Path(__file__).with_name('.n')\n"
            "def test_flaky():\n"
            "    n = int(_c.read_text()) if _c.exists() else 0\n"
            "    _c.write_text(str(n + 1))\n"
            "    assert n >= 2\n"  # fails attempts 0,1,2 → passes on the 3rd
            "def test_a(): assert True\n"
            "def test_b(): assert True\n"
        )
        log_path = tmp_path / "nodeids.tsv"
        plugin_args, env = flake_mod._nodeid_reporter.build_invocation(
            log_path, self._repo_root(), log_passes=True
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:randomly",
                "--reruns=3",
                *plugin_args,
                "test_reruns.py",
            ],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        # 3 distinct tests all ran to completion → ran == collected == 3,
        # even though the flaky one produced ~5 setup reports.
        assert self._session_records(log_path) == ["3 3"]

    def test_no_log_written_on_hard_exit_zero_before_tests(self, tmp_path):
        # codex #1222 r17 B1: os._exit(0) at collection time (here from a
        # conftest imported before any test runs) kills pytest before any
        # setup report or sessionfinish, so the reporter writes NOTHING — the
        # log file is never even created. This is exactly why full_unit
        # pre-creates a start sentinel: without it, "no file" would be
        # misread as "plugin unavailable" and the zero exit trusted on a
        # fully-truncated run. (The plugin itself cannot defend this; the
        # sentinel in full_unit does.)
        (tmp_path / "conftest.py").write_text("import os\nos._exit(0)\n")
        (tmp_path / "test_never.py").write_text("def test_a(): assert True\n")
        log_path = tmp_path / "nodeids.tsv"
        plugin_args, env = flake_mod._nodeid_reporter.build_invocation(
            log_path, self._repo_root()
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:randomly",
                "-p",
                "no:cacheprovider",
                *plugin_args,
                "test_never.py",
            ],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        assert not log_path.exists()  # plugin wrote nothing → sentinel needed


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
        # spawns a long-lived grandchild in the SAME group (no setsid). On
        # timeout, killpg must kill the grandchild too — a bare proc.kill()
        # on the direct child would leave it running (codex #1222 r2).
        #
        # Liveness is checked via a HEARTBEAT, not os.kill(pid, 0): a
        # SIGKILLed-but-not-yet-reaped ZOMBIE still answers signal 0, so a
        # PID-disappearance poll can flake when the reaper is slow (codex
        # #1222 r7). A killed process — zombie or not — cannot tick the
        # heartbeat file, so a frozen counter is unambiguous proof of death.
        pidfile = tmp_path / "grandchild.pid"
        heartbeat = tmp_path / "heartbeat"
        grandchild_prog = textwrap.dedent(
            f"""
            import time
            n = 0
            while True:
                with open({str(heartbeat)!r}, "w") as f:
                    f.write(str(n))
                    f.flush()
                n += 1
                time.sleep(0.05)
            """
        )
        child_prog = textwrap.dedent(
            f"""
            import subprocess, sys, time
            gc = subprocess.Popen([sys.executable, "-c", {grandchild_prog!r}])
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
        try:
            # Let the SIGKILL settle, then confirm the heartbeat froze.
            time.sleep(0.3)
            assert heartbeat.exists(), "grandchild never started"
            beat1 = heartbeat.read_text()
            time.sleep(1.0)  # >> 0.05s tick — an ALIVE grandchild ticks ~20×
            beat2 = heartbeat.read_text()
            assert beat1 == beat2, (
                f"grandchild heartbeat advanced {beat1!r}->{beat2!r} — it "
                f"survived the process-group kill"
            )
        finally:
            # best-effort cleanup so a survivor (on failure) doesn't leak.
            try:
                os.kill(gc_pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

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
    def test_registered_after_all_gating_steps(self):
        # Must read full-unit.log (so after full_unit) AND run after every
        # step that spawns model servers / GPU workers, so its possible
        # descendant leak on macOS can't contaminate a later gate (codex
        # #1222 r7): after full_unit and stress_e2e_bench.
        from scripts.pr_validate.runner import STEPS

        names = [s.name for s in STEPS]
        assert "flake_tracking" in names
        assert names.index("flake_tracking") > names.index("full_unit")
        assert names.index("flake_tracking") > names.index("stress_e2e_bench")

    def test_crash_never_blocks(self):
        # Advisory: this step has NO blocking path — every outcome is
        # pass/skip, and run() downgrades any crash to skip so an uncaught
        # exception can't become a blocking error (codex #1222 r8).
        assert FlakeTrackingStep().continue_on_error is True


class TestTargetedTestsRerunBackstop:
    """targeted_tests is a gating run — and, for a low-blast PR where
    full_unit is skipped, the ONLY gating run of candidate tests — so the
    name-independent RERUN backstop must live here too (codex #1222 r24)."""

    def test_run_pytest_detects_rerun_record(self, tmp_path, monkeypatch):
        # _run_pytest injects the reporter and flags reran=True when the
        # structured log carries a RERUN record (rerunfailures smuggled in
        # under a non-blocked name).
        nodeid_log = tmp_path / "n.tsv"

        def fake_run(cmd, **kw):
            env = kw.get("env") or {}
            log = env.get("PR_VALIDATE_NODEID_LOG")
            assert log, "reporter was not injected"
            with open(log, "w", encoding="utf-8") as fh:
                fh.write("RERUN\ttests/a.py::t\nFAILED\ttests/a.py::t\n")
            return subprocess.CompletedProcess(cmd, 1, "==== 1 failed in 1s ====\n", "")

        monkeypatch.setattr(targeted_mod.subprocess, "run", fake_run)
        result = targeted_mod._run_pytest(
            ["tests/a.py"], tmp_path / "log.txt", tmp_path, nodeid_log
        )
        assert result.reran is True

    def test_run_pytest_no_rerun_is_false(self, tmp_path, monkeypatch):
        nodeid_log = tmp_path / "n.tsv"

        def fake_run(cmd, **kw):
            env = kw.get("env") or {}
            log = env.get("PR_VALIDATE_NODEID_LOG")
            with open(log, "w", encoding="utf-8") as fh:
                fh.write("FAILED\ttests/a.py::t\nSESSIONFINISH\t1 1\n")
            return subprocess.CompletedProcess(cmd, 1, "==== 1 failed in 1s ====\n", "")

        monkeypatch.setattr(targeted_mod.subprocess, "run", fake_run)
        result = targeted_mod._run_pytest(
            ["tests/a.py"], tmp_path / "log.txt", tmp_path, nodeid_log
        )
        assert result.reran is False

    def test_run_blocks_when_pr_run_reran(self, ctx_factory, monkeypatch):
        ctx = ctx_factory(["vllm_mlx/scheduler.py"])
        monkeypatch.setattr(
            targeted_mod, "_select_test_files", lambda c: ["tests/test_scheduler.py"]
        )
        monkeypatch.setattr(
            targeted_mod,
            "_run_pytest",
            lambda *a, **k: targeted_mod._PytestRun(
                summary="1 failed, 2 passed",
                failed=["tests/test_scheduler.py::t"],
                reran=True,
                returncode=1,
                errored=False,
                complete=True,
                structured=True,
            ),
        )
        res = targeted_mod.TargetedTestsStep().run(ctx)
        assert res.status == "fail"
        assert "rerun" in res.details.lower()

    def test_negative_control_run_does_not_inject_reporter(self):
        # The base predates this PR's reporter module, so the negative-control
        # run on main must NOT try to inject `-p …_nodeid_reporter` (it would
        # be an unloadable plugin → usage error). _run_on_main uses the plain
        # _PYTEST_CMD, never the reporter.
        import inspect

        src = inspect.getsource(targeted_mod._run_on_main)
        assert "build_invocation" not in src
        assert "_nodeid_reporter" not in src


class TestTargetedTestsUntrustedRun:
    """An empty FAILED list is only a genuine pass when the run finished as an
    ordinary pass (0) or fail (1). A collection error, usage error, "no tests
    collected", a structured ERROR (which never scrapes as a FAILED line), an
    early ``pytest.exit``, or an exit-1 with a suppressed failure summary must
    NOT slip through as a false green (codex #1222 r25/r26)."""

    @staticmethod
    def _run(
        returncode,
        *,
        errored=False,
        complete=True,
        failed=(),
        reran=False,
        structured=True,
    ):
        return targeted_mod._PytestRun(
            summary="s",
            failed=list(failed),
            reran=reran,
            returncode=returncode,
            errored=errored,
            complete=complete,
            structured=structured,
        )

    def test_reason_none_for_ordinary_pass_or_fail(self):
        assert targeted_mod._untrusted_run_reason(self._run(0)) is None
        # exit 1 is trusted only when a concrete failure accounts for it.
        assert (
            targeted_mod._untrusted_run_reason(self._run(1, failed=["tests/a.py::t"]))
            is None
        )

    def test_reason_flags_abnormal_exit(self):
        for code in (2, 3, 4, 5):
            reason = targeted_mod._untrusted_run_reason(self._run(code))
            assert reason and str(code) in reason

    def test_reason_flags_error_record(self):
        reason = targeted_mod._untrusted_run_reason(self._run(0, errored=True))
        assert reason and "ERROR" in reason

    def test_reason_flags_incomplete_session(self):
        reason = targeted_mod._untrusted_run_reason(self._run(0, complete=False))
        assert reason and "completion" in reason

    def test_reason_flags_unaccounted_exit_one(self):
        # exit 1 but the structured log has NO FAILED/RERUN record → the
        # summary was suppressed; the empty FAILED list must not pass (r26).
        reason = targeted_mod._untrusted_run_reason(
            self._run(1, failed=[], reran=False, structured=True)
        )
        assert reason and "exited 1" in reason

    def test_reason_unaccounted_exit_one_only_when_structured(self):
        # Without the reporter we can't prove the record set, so we don't
        # second-guess an exit-1 (the degraded, non-candidate path).
        assert (
            targeted_mod._untrusted_run_reason(
                self._run(1, failed=[], structured=False)
            )
            is None
        )

    def _patch_run_pytest(self, monkeypatch, ctx_factory, run):
        ctx = ctx_factory(["vllm_mlx/scheduler.py"])
        monkeypatch.setattr(
            targeted_mod, "_select_test_files", lambda c: ["tests/test_scheduler.py"]
        )
        monkeypatch.setattr(targeted_mod, "_run_pytest", lambda *a, **k: run)
        return ctx

    def test_run_blocks_on_collection_error_empty_failed(
        self, ctx_factory, monkeypatch
    ):
        # Exit 2 + ERROR record + EMPTY FAILED (a collection failure scrapes no
        # FAILED summary line). Pre-r25 this passed as a false green because the
        # gate only checked the empty FAILED list.
        ctx = self._patch_run_pytest(
            monkeypatch,
            ctx_factory,
            self._run(2, errored=True, complete=False),
        )
        res = targeted_mod.TargetedTestsStep().run(ctx)
        assert res.status == "fail"
        assert "untrusted" in res.summary

    def test_run_blocks_on_error_record_with_exit_one(self, ctx_factory, monkeypatch):
        # Exit 1 but only a setup/teardown ERROR (no scrapable FAILED) — still
        # must block, not pass on the empty FAILED list.
        ctx = self._patch_run_pytest(
            monkeypatch, ctx_factory, self._run(1, errored=True)
        )
        res = targeted_mod.TargetedTestsStep().run(ctx)
        assert res.status == "fail"

    def test_run_blocks_on_unaccounted_exit_one(self, ctx_factory, monkeypatch):
        # Exit 1, structured, but empty FAILED + no RERUN — the failure summary
        # was suppressed. Must block, not pass (codex #1222 r26).
        ctx = self._patch_run_pytest(
            monkeypatch, ctx_factory, self._run(1, failed=[], reran=False)
        )
        res = targeted_mod.TargetedTestsStep().run(ctx)
        assert res.status == "fail"
        assert "untrusted" in res.summary

    def test_run_blocks_on_incomplete_session(self, ctx_factory, monkeypatch):
        # Exit 0 but a truncated session (early pytest.exit before the
        # session-finish record) → not accepted as green.
        ctx = self._patch_run_pytest(
            monkeypatch, ctx_factory, self._run(0, complete=False)
        )
        res = targeted_mod.TargetedTestsStep().run(ctx)
        assert res.status == "fail"

    def test_run_passes_on_clean_exit_zero(self, ctx_factory, monkeypatch):
        # The hardening must not break the happy path: exit 0, complete, no
        # errors, empty FAILED → pass.
        ctx = self._patch_run_pytest(monkeypatch, ctx_factory, self._run(0))
        res = targeted_mod.TargetedTestsStep().run(ctx)
        assert res.status == "pass"

    def test_run_pytest_populates_error_and_incomplete(self, tmp_path, monkeypatch):
        # _run_pytest reads ERROR + session-completeness from the structured
        # log, not just RERUN, and always carries the real exit code.
        nodeid_log = tmp_path / "n.tsv"

        def fake_run(cmd, **kw):
            env = kw.get("env") or {}
            log = env.get("PR_VALIDATE_NODEID_LOG")
            with open(log, "w", encoding="utf-8") as fh:
                fh.write("ERROR\ttests/a.py::t\n")  # no SESSIONFINISH → incomplete
            return subprocess.CompletedProcess(cmd, 2, "", "")

        monkeypatch.setattr(targeted_mod.subprocess, "run", fake_run)
        result = targeted_mod._run_pytest(
            ["tests/a.py"], tmp_path / "log.txt", tmp_path, nodeid_log
        )
        assert result.errored is True
        assert result.complete is False
        assert result.returncode == 2
        assert result.structured is True

    def test_pytest_cmd_neutralizes_candidate_addopts(self):
        # "-o" immediately followed by "addopts=" must be on the shared gating
        # command so a candidate pytest.ini / pyproject addopts (a planted
        # --deselect / -m filter / -x / -rN) can't steer this gating run
        # (codex #1222 r26).
        cmd = list(targeted_mod._PYTEST_CMD)
        assert any(
            cmd[i] == "-o" and cmd[i + 1] == "addopts=" for i in range(len(cmd) - 1)
        )

    def test_run_pytest_uses_structured_failed_not_scrape(self, tmp_path, monkeypatch):
        # The reporter's FAILED record is authoritative even when the stdout
        # short summary is blank (summary suppressed): the scrape would return
        # [] but the structured read must still surface the failure (r26).
        nodeid_log = tmp_path / "n.tsv"

        def fake_run(cmd, **kw):
            env = kw.get("env") or {}
            log = env.get("PR_VALIDATE_NODEID_LOG")
            with open(log, "w", encoding="utf-8") as fh:
                fh.write("FAILED\ttests/a.py::t\nSESSIONFINISH\t1 1\n")
            # stdout carries NO "short test summary" section → scrape sees [].
            return subprocess.CompletedProcess(cmd, 1, "1 failed in 0.1s\n", "")

        monkeypatch.setattr(targeted_mod.subprocess, "run", fake_run)
        result = targeted_mod._run_pytest(
            ["tests/a.py"], tmp_path / "log.txt", tmp_path, nodeid_log
        )
        assert result.failed == ["tests/a.py::t"]
        assert result.structured is True

    def test_pytest_cmd_reapplies_marker_exclusion(self):
        # `-o addopts=` drops the repo's addopts, which carried
        # `-m "not slow and not integration and not needle"`. That exclusion
        # must be RE-SPECIFIED on the gating command or targeted would start
        # running slow/integration/needle tests it used to inherit-exclude
        # (codex #1222 r26 regression, caught in r27). Mirrors full_unit.
        cmd = list(targeted_mod._PYTEST_CMD)
        mexpr = "not slow and not integration and not needle"
        # present as a value, and immediately preceded by the `-m` flag (not
        # the leading `python -m pytest`).
        assert mexpr in cmd
        assert cmd[cmd.index(mexpr) - 1] == "-m"

    def test_last_summary_line_parses_barless_quiet_mode(self):
        # Once `-v` is dropped (via `-o addopts=`) pytest prints the counts
        # line WITHOUT the ==== bars; the parser must still find it rather than
        # fall back to a bare "exit N" (codex #1222 r27). Shared parser lives in
        # _pytest_summary so full_unit and targeted can't drift (codex r28).
        stdout = "........\n261 passed, 2 skipped in 10.86s\n"
        assert last_summary_line(stdout) == "261 passed, 2 skipped in 10.86s"

    def test_last_summary_line_parses_fenced_and_failures(self):
        assert (
            last_summary_line("==== 5 failed, 3 passed in 2.0s ====")
            == "5 failed, 3 passed in 2.0s"
        )
        # No summary at all → empty (caller falls back to the exit code).
        assert last_summary_line("collecting...\n") == ""
        # A bar-less "all deselected" summary is recognized (drives the r28
        # exit-5 skip path).
        assert last_summary_line("2 deselected in 0.05s") == "2 deselected in 0.05s"

    def test_run_skips_when_all_targets_deselected_by_marker(
        self, ctx_factory, monkeypatch
    ):
        # The gate-owned `-m "not slow…"` filter can deselect every test in the
        # targeted files (a PR touching only slow/integration/needle tests) →
        # pytest exit 5. That's not a tamper signal, so SKIP not block (codex
        # #1222 r28 — a regression from the r27 `-m` re-apply).
        ctx = self._patch_run_pytest(
            monkeypatch,
            ctx_factory,
            targeted_mod._PytestRun(
                summary="2 deselected in 0.05s",
                failed=[],
                reran=False,
                returncode=5,
                errored=False,
                complete=False,
                structured=True,
            ),
        )
        res = targeted_mod.TargetedTestsStep().run(ctx)
        assert res.status == "skip"
        assert "deselected" in res.summary

    def test_last_summary_line_recognizes_plural_errors(self):
        # pytest pluralizes the error outcome ("1 error" vs "2 errors"), so an
        # error-only run summary must still be recognized or the step summary
        # degrades to a bare "exit 2" (codex #1222 r29).
        assert last_summary_line("2 errors in 0.10s") == "2 errors in 0.10s"
        assert last_summary_line("==== 1 error in 0.1s ====") == "1 error in 0.1s"

    def test_run_blocks_exit5_without_deselection(self, ctx_factory, monkeypatch):
        # exit 5 WITHOUT a deselection (empty/renamed file, or a candidate
        # python_files matching nothing) is NOT the benign case — no
        # "deselected" in the summary → stays blocked (codex #1222 r28).
        ctx = self._patch_run_pytest(
            monkeypatch,
            ctx_factory,
            targeted_mod._PytestRun(
                summary="no tests ran in 0.01s",
                failed=[],
                reran=False,
                returncode=5,
                errored=False,
                complete=False,
                structured=True,
            ),
        )
        res = targeted_mod.TargetedTestsStep().run(ctx)
        assert res.status == "fail"
        assert "untrusted" in res.summary

    def test_run_pytest_fallback_scrape_preserves_spaced_param_ids(
        self, tmp_path, monkeypatch
    ):
        # Without the reporter, _run_pytest scrapes stdout — and the scrape must
        # be space-preserving (summary_node_ids), NOT a `\S+` grab, so a failing
        # param id containing spaces is captured whole. Otherwise it couldn't
        # match the base negative control's parse of the same id and a
        # pre-existing failure would be misclassified as a PR-only regression
        # (codex #1222 r29).
        stdout = (
            "==== short test summary info ====\n"
            "FAILED tests/a.py::test_x[param with spaces] - AssertionError\n"
            "==== 1 failed in 0.1s ====\n"
        )

        def fake_run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 1, stdout, "")

        monkeypatch.setattr(targeted_mod.subprocess, "run", fake_run)
        result = targeted_mod._run_pytest(
            ["tests/a.py"], tmp_path / "log.txt", tmp_path
        )
        assert result.structured is False
        assert result.failed == ["tests/a.py::test_x[param with spaces]"]

    def test_negative_control_scrape_matches_canonical_reporter_id(self, tmp_path):
        # The base negative control scrapes stdout (summary_node_ids); the PR
        # side records report.nodeid via the reporter. To classify a shared
        # failure as pre-existing (not a PR-only regression), the two parsers
        # must yield the SAME id — even when the param id contains spaces. Run a
        # real pytest and prove parity end-to-end (codex #1222 r29). Pre-fix the
        # base used `^FAILED\s+(\S+)`, truncating `test_x[a b c]` to `test_x[a`.
        import pathlib

        (tmp_path / "test_spaced.py").write_text(
            textwrap.dedent(
                """
                import pytest

                @pytest.mark.parametrize("x", ["a b c"])
                def test_x(x):
                    assert False
                """
            )
        )
        repo_root = (
            pathlib.Path(targeted_mod._nodeid_reporter.__file__).resolve().parents[2]
        )
        log_path = tmp_path / "nodeids.tsv"
        plugin_args, env = targeted_mod._nodeid_reporter.build_invocation(
            log_path, repo_root
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:randomly",
                "-rfE",
                *plugin_args,
                "test_spaced.py",
            ],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        canonical = report_log_node_ids(log_path, "FAILED")
        scraped = summary_node_ids(proc.stdout, "FAILED")
        assert canonical == ["test_spaced.py::test_x[a b c]"]
        # Parity: base scrape == PR canonical id → shared failure classified
        # pre-existing, not a false regression.
        assert scraped == canonical
