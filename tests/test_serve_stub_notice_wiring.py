# SPDX-License-Identifier: Apache-2.0
"""0.10.16 dogfood finding ⑥ — serve_command wiring for the weightless-stub
notice.

The unit tests in ``test_download_gate.py`` prove ``weightless_stub_notice``
returns the right string, but they don't prove ``serve_command`` actually
emits it — before this test, deleting the ``print(_stub_notice, ...)`` line
left every test green. These integration tests drive the real
``cli.serve_command`` prologue and pin:

* a stub-cache notice reaches **stderr**, and does so **before**
  ``_ensure_model_downloaded`` runs (so the operator sees it up front, not
  buffered behind a multi-GB download), and
* nothing is emitted on the warm-cache / non-stub path.

We stop execution at the ``_ensure_model_downloaded`` call (raise a sentinel)
so the heavy server boot never runs — the notice line sits immediately above
it, so ordering is exercised faithfully.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from vllm_mlx import _download_gate as gate
from vllm_mlx import cli


class _StopServeError(Exception):
    """Sentinel to abort serve_command right at the download step."""


def _serve_ns():
    """Resolve a real serve Namespace for a plain text alias via argparse,
    mirroring the established ``_minimal_serve_ns`` pattern."""
    captured: list = []
    argv = ["rapid-mlx", "serve", "qwen3.5-4b-4bit"]
    with (
        patch.object(sys, "argv", argv),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    return captured[0]


@pytest.fixture
def _quiet_version_check(monkeypatch):
    """Stub the interactive upgrade / staleness prompts so the prologue
    reaches the notice block deterministically without touching the network
    or stdin."""
    from vllm_mlx import _version_check

    monkeypatch.setattr(_version_check, "prompt_upgrade_if_available", lambda: False)
    monkeypatch.setattr(
        _version_check, "print_staleness_warning_if_any", lambda: None, raising=False
    )
    return monkeypatch


def test_serve_emits_stub_notice_to_stderr_before_download(
    _quiet_version_check, capsys
):
    """A weightless-stub cache → the notice lands on stderr BEFORE
    ``_ensure_model_downloaded`` is invoked."""
    monkeypatch = _quiet_version_check
    notice = (
        "  Note: some/model config cached but weights missing — download will start."
    )
    # serve_command imports this name locally from _download_gate at call
    # time, so patch it on the source module (not on ``cli``).
    monkeypatch.setattr(gate, "weightless_stub_notice", lambda _m: notice)

    order: list = []

    def _fake_download(model):
        # Snapshot stderr at the moment the download step begins: the notice
        # must already be present (it's the line immediately above this call).
        order.append(capsys.readouterr().err)
        raise _StopServeError()

    monkeypatch.setattr(cli, "_ensure_model_downloaded", _fake_download)

    ns = _serve_ns()
    with pytest.raises(_StopServeError):
        cli.serve_command(ns)

    assert len(order) == 1, "download step must run exactly once"
    err_at_download = order[0]
    assert notice in err_at_download, (
        "weightless-stub notice must reach stderr BEFORE _ensure_model_downloaded; "
        f"stderr at download start was: {err_at_download!r}"
    )


def test_serve_emits_nothing_when_not_a_stub(_quiet_version_check, capsys):
    """Warm-cache / non-stub path → ``weightless_stub_notice`` returns None,
    so no notice is emitted, but the download step still runs (wiring stays
    on the normal path)."""
    monkeypatch = _quiet_version_check
    monkeypatch.setattr(gate, "weightless_stub_notice", lambda _m: None)

    order: list = []

    def _fake_download(model):
        order.append(capsys.readouterr().err)
        raise _StopServeError()

    monkeypatch.setattr(cli, "_ensure_model_downloaded", _fake_download)

    ns = _serve_ns()
    with pytest.raises(_StopServeError):
        cli.serve_command(ns)

    assert len(order) == 1
    assert "config cached but" not in order[0]
    assert "weights are missing" not in order[0]
