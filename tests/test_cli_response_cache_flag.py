# SPDX-License-Identifier: Apache-2.0
"""``--response-cache-entries`` end-to-end CLI wiring for ``serve``.

The response cache is a chat/serve feature: its lookup/store logic lives
only in the chat route, and ``bench`` never consumes it — so the flag is
registered on ``serve_parser`` ONLY (codex #1123 BLOCKING-2 removed the
advertised-no-op bench flag).

This test drives the real ``main()`` parser + ``serve_command`` far enough
to intercept the ACTUAL ``SchedulerConfig`` the serve path constructs, and
asserts the parsed ``--response-cache-entries`` value arrives at
``SchedulerConfig(response_cache_entries=N)`` (cli.py serve wiring). An
argparse-only check could not prove the plumbing line was wired at the
construction site — deleting the ``response_cache_entries=...`` kwarg would
leave an argparse-only test green. Fully offline: model load / disk /
network / version-check boundaries are mocked, and construction raises
``_StopError`` before any engine boot.
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


def _capture_serve_scheduler_config(argv: list[str]) -> dict:
    """Drive the REAL ``main()`` → ``serve_command`` for a ``serve``
    invocation and capture the kwargs the serve path passes to
    ``SchedulerConfig(...)``.

    Patching ``vllm_mlx.scheduler.SchedulerConfig`` is the correct
    interception point: ``serve_command`` imports it locally via
    ``from .scheduler import SchedulerConfig`` at each call, so the patch
    is picked up at the real construction site (cli.py serve wiring). The
    fake records the kwargs and raises ``_StopError`` so nothing past
    construction (model load, engine boot, uvicorn) runs. Only the
    minimal I/O boundaries the serve path hits BEFORE construction are
    mocked out — everything between parse and ``SchedulerConfig(...)`` is
    the real code.
    """
    captured: dict = {}

    def _fake_scheduler_config(*args, **kwargs):
        captured.update(kwargs)
        raise _StopError

    with (
        mock.patch.object(cli, "_check_disk_space", lambda *a, **k: None),
        mock.patch.object(cli, "_check_memory_capacity", lambda *a, **k: None),
        mock.patch.object(cli, "_ensure_model_downloaded", lambda *a, **k: None),
        mock.patch.object(
            cli, "_gather_kv_cache_dtype_inputs", lambda *a, **k: ({}, None)
        ),
        mock.patch(
            "vllm_mlx._version_check.prompt_upgrade_if_available",
            return_value=False,
        ),
        mock.patch("mlx_lm.load", return_value=(object(), object())),
        mock.patch("vllm_mlx.scheduler.SchedulerConfig", _fake_scheduler_config),
        mock.patch.object(sys, "argv", ["rapid-mlx", *argv]),
        mock.patch.object(sys.stdin, "isatty", return_value=False),
        pytest.raises((_StopError, SystemExit)),
    ):
        cli.main()

    assert captured, (
        "SchedulerConfig was never constructed by the serve path — the flow "
        "died before the response-cache plumbing under test. If this fires "
        "after unrelated serve-path changes, extend the boundary mocks above."
    )
    return captured


# ── serve wiring: the parsed value must reach SchedulerConfig ──────────


def test_serve_response_cache_entries_reaches_scheduler_config():
    """MUTATION-KILL: deleting ``response_cache_entries=...`` at the serve
    ``SchedulerConfig(...)`` construction (cli.py) makes this FAIL —
    ``captured`` would then lack the key and the assertion trips."""
    captured = _capture_serve_scheduler_config(
        ["serve", "qwen3.5-4b-4bit", "--response-cache-entries", "16"]
    )
    assert captured.get("response_cache_entries") == 16


def test_serve_response_cache_entries_defaults_to_zero():
    captured = _capture_serve_scheduler_config(["serve", "qwen3.5-4b-4bit"])
    assert captured.get("response_cache_entries") == 0


# ── serve parser registration (argparse surface) ──────────────────────


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
    args = _parse_serve_args(
        ["serve", "qwen3.5-4b-4bit", "--response-cache-entries", "16"]
    )
    assert getattr(args, "response_cache_entries", None) == 16


def test_serve_parser_response_cache_entries_defaults_to_zero():
    args = _parse_serve_args(["serve", "qwen3.5-4b-4bit"])
    assert getattr(args, "response_cache_entries", None) == 0


def test_serve_negative_response_cache_entries_rejected_by_scheduler_config():
    """argparse does not clamp ``type=int``, so ``-1`` flows through the
    serve plumbing unchanged; the REAL ``SchedulerConfig.__post_init__``
    is what rejects it. Prove (1) it arrives unclamped at the serve
    construction site, then (2) the real config construction raises."""
    captured = _capture_serve_scheduler_config(
        ["serve", "qwen3.5-4b-4bit", "--response-cache-entries", "-1"]
    )
    assert captured.get("response_cache_entries") == -1

    from vllm_mlx.scheduler import SchedulerConfig

    with pytest.raises(ValueError, match=r"response_cache_entries must be >= 0"):
        SchedulerConfig(response_cache_entries=captured["response_cache_entries"])


# ── bench must NOT advertise the flag (codex #1123 BLOCKING-2) ─────────


def test_bench_does_not_register_response_cache_entries():
    """The response cache is serve-only; ``bench --response-cache-entries``
    must be REJECTED by argparse (the flag was an advertised no-op and was
    removed). A SystemExit(2) from argparse proves the flag is gone."""
    with (
        mock.patch.object(
            sys,
            "argv",
            [
                "rapid-mlx",
                "bench",
                "does-not-exist/definitely-not-a-real-model",
                "--response-cache-entries",
                "8",
            ],
        ),
        pytest.raises(SystemExit) as excinfo,
    ):
        cli.main()
    # argparse exits 2 on an unrecognized argument.
    assert excinfo.value.code == 2
