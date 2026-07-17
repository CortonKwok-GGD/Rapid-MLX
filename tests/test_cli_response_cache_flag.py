# SPDX-License-Identifier: Apache-2.0
"""``--response-cache-entries`` must be honored by BOTH ``serve`` and
``bench`` — mirroring the #1103 lesson where ``--hybrid-cache-entries``
was registered only on ``serve_parser`` and codex flagged the bench-parser
omission as BLOCKING.

These tests drive the real ``main()`` parser + command plumbing in-process,
mocking only the model-loading / disk / network boundaries, and assert the
parsed value ACTUALLY arrives at ``SchedulerConfig(response_cache_entries=N)``.
An argparse-only check could not prove the plumbing line was wired at the
construction site — deleting it would leave an argparse-only test green.
Fully offline: no engine boot, no network.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest

import vllm_mlx.cli as cli

# Pre-import so any lazy ``from .engine_core import ...`` binds the REAL
# class before a test patches ``vllm_mlx.scheduler.SchedulerConfig``.
import vllm_mlx.engine_core as _engine_core  # noqa: E402,F401


class _StopError(Exception):
    """Raised right after SchedulerConfig is built to short-circuit the
    command before any engine boot."""


def _capture_bench_scheduler_config(argv: list[str]) -> dict:
    captured: dict = {}

    def _fake_scheduler_config(*args, **kwargs):
        captured.update(kwargs)
        raise _StopError

    with (
        mock.patch.object(cli, "_check_disk_space", lambda *a, **k: None),
        mock.patch.object(cli, "_check_memory_capacity", lambda *a, **k: None),
        mock.patch.object(cli, "_ensure_model_downloaded", lambda *a, **k: None),
        mock.patch("mlx_lm.load", return_value=(object(), object())),
        mock.patch("vllm_mlx.scheduler.SchedulerConfig", _fake_scheduler_config),
        mock.patch.object(sys, "argv", ["rapid-mlx", *argv]),
        mock.patch.object(sys.stdin, "isatty", return_value=False),
        pytest.raises((_StopError, SystemExit)),
    ):
        cli.main()

    assert captured, (
        "SchedulerConfig was never constructed — the flow died before the "
        "response-cache plumbing under test"
    )
    return captured


# ── bench parser ──────────────────────────────────────────────────────


def test_bench_response_cache_entries_reaches_scheduler_config():
    captured = _capture_bench_scheduler_config(
        [
            "bench",
            "does-not-exist/definitely-not-a-real-model",
            "--response-cache-entries",
            "32",
            "--num-prompts",
            "1",
            "--max-tokens",
            "1",
        ]
    )
    assert captured.get("response_cache_entries") == 32


def test_bench_response_cache_entries_defaults_to_zero():
    captured = _capture_bench_scheduler_config(
        [
            "bench",
            "does-not-exist/definitely-not-a-real-model",
            "--num-prompts",
            "1",
            "--max-tokens",
            "1",
        ]
    )
    assert captured.get("response_cache_entries") == 0


def test_bench_negative_response_cache_entries_rejected_by_scheduler_config():
    """argparse does not clamp ``type=int``, so ``-1`` flows through the
    bench plumbing unchanged; the REAL ``SchedulerConfig.__post_init__``
    is what rejects it. Prove (1) it arrives unclamped, then (2) the real
    config construction raises."""
    captured = _capture_bench_scheduler_config(
        [
            "bench",
            "does-not-exist/definitely-not-a-real-model",
            "--response-cache-entries",
            "-1",
            "--num-prompts",
            "1",
            "--max-tokens",
            "1",
        ]
    )
    assert captured.get("response_cache_entries") == -1

    from vllm_mlx.scheduler import SchedulerConfig

    with pytest.raises(ValueError, match=r"response_cache_entries must be >= 0"):
        SchedulerConfig(response_cache_entries=captured["response_cache_entries"])


# ── serve parser registration ─────────────────────────────────────────


def _parse_serve_args(argv: list[str]):
    """Drive the REAL ``main()`` argument parser for a ``serve`` invocation
    and capture the parsed ``args`` namespace by intercepting the dispatch
    to ``serve_command`` — so we exercise the actual serve_parser
    registration, not a reconstructed stand-in. Nothing past parsing runs
    (no model download, no engine boot)."""
    captured = {}

    def _capture(args):
        captured["args"] = args

    with (
        mock.patch.object(cli, "serve_command", _capture),
        mock.patch.object(sys, "argv", ["rapid-mlx", *argv]),
    ):
        cli.main()
    assert "args" in captured, "serve_command dispatch was never reached"
    return captured["args"]


def test_serve_parser_registers_response_cache_entries():
    """The flag must be registered on serve_parser too (not bench-only) —
    the #1103 BLOCKING miss was a serve/bench asymmetry."""
    args = _parse_serve_args(
        ["serve", "qwen3.5-4b-4bit", "--response-cache-entries", "16"]
    )
    assert getattr(args, "response_cache_entries", None) == 16


def test_serve_parser_response_cache_entries_defaults_to_zero():
    args = _parse_serve_args(["serve", "qwen3.5-4b-4bit"])
    assert getattr(args, "response_cache_entries", None) == 0
