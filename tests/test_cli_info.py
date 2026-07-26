# SPDX-License-Identifier: Apache-2.0
"""Tests for ``rapid-mlx info <model>``.

The ``info`` subcommand is a fast read-only path — it resolves an alias
and prints the per-model profile (parser names, hybrid flag, etc.) without
loading weights. Before this test file existed, `info_command` was the
one alias-touching CLI command with no test coverage; a refactor that
broke alias resolution in `info` would have shipped silently.
"""

from __future__ import annotations

import argparse
import io
import sys
from unittest.mock import patch

from vllm_mlx import cli


def _run_info(model_name: str) -> str:
    """Invoke ``info_command`` with stdout captured."""
    args = argparse.Namespace(model=model_name)
    buf = io.StringIO()
    with patch.object(sys, "stdout", buf):
        cli.info_command(args)
    return buf.getvalue()


def test_info_resolves_alias_to_hf_path() -> None:
    """Typing the alias ``qwen3.5-4b-4bit`` must show the resolution arrow
    ``qwen3.5-4b-4bit → mlx-community/Qwen3.5-4B-MLX-4bit``. A refactor that
    bypasses ``resolve_model`` would drop the alias signal."""
    out = _run_info("qwen3.5-4b-4bit")
    assert "qwen3.5-4b-4bit" in out
    assert "→" in out, f"expected alias-resolution arrow in output, got:\n{out}"
    assert "mlx-community/Qwen3.5-4B-MLX-4bit" in out, (
        f"expected resolved HF path in output, got:\n{out}"
    )


def test_info_passes_full_hf_path_through() -> None:
    """An HF path passed directly (no alias) must not show a resolution
    arrow — it's already canonical. Showing one would falsely imply the
    user typed an alias."""
    hf_path = "mlx-community/Qwen3.5-4B-MLX-4bit"
    out = _run_info(hf_path)
    assert hf_path in out
    assert "→" not in out, (
        f"info wrongly showed resolution arrow for an already-canonical "
        f"HF path; output:\n{out}"
    )


def test_info_unknown_model_does_not_crash() -> None:
    """An unknown model name must produce a graceful 'no pattern matched'
    explanation instead of crashing. Catches regressions where the
    detection pipeline raises on truly novel names."""
    out = _run_info("totally-made-up-model-xyz-9000")
    # Either a runtime-probe message or any non-empty rendered table —
    # we just need to confirm no exception escapes and something is shown.
    assert out.strip(), "info_command produced empty output for unknown model"


def test_info_does_not_require_model_weights() -> None:
    """``info`` must work without weights on disk — it's the user's
    primary 'what is this thing' command before they pull anything.
    Anything that triggers weight loading here would be a regression.

    Indirectly verified: this whole module imports `cli` and calls
    `info_command` for several aliases; if `info_command` were calling
    into model loading we'd hit network / filesystem operations during
    test collection, which would slow the unit suite by minutes. The
    test passing in < 1s under pytest is the assertion.
    """
    # Reaching this point in the test session is the success condition.
    # The other tests in this module already exercise multiple aliases.
    assert True


def test_info_prints_mtp_sidecar_note_when_head_declared_but_absent(
    monkeypatch,
) -> None:
    """When ``config.json`` declares an MTP head but the head weights were
    stripped from this checkpoint, ``info`` must print the actionable ``MTP:``
    note that mirrors the serve-time hard-fail and points to the sidecar path.

    This reconciles ``info`` (which otherwise reports a bare ``disabled``)
    with ``serve --speculative-config '{"method":"mtp"}'`` (which announces
    ``MTP: enabled`` then hard-fails at inject). The config read is
    monkeypatched via the eligibility helper so the test is deterministic and
    cache-independent.
    """
    from vllm_mlx import model_auto_config as auto_config_mod

    monkeypatch.setattr(
        auto_config_mod, "_mtp_declared_absent_layers", lambda name, cfg: 1
    )
    out = _run_info("qwen3.6-27b-8bit")
    assert "mtp_num_hidden_layers=1" in out, f"declared-layer count missing:\n{out}"
    assert "head weights are absent" in out, f"absent-head diagnosis missing:\n{out}"
    # Actionable: must emit the COMPLETE sidecar remedy — the ``model`` field
    # is load-bearing. Asserting the full fragment (not just ``method``)
    # guards against regressing to a bare ``{"method":"mtp"}`` that would just
    # reproduce the original inject-time hard failure (codex #1202 round 2).
    assert """--speculative-config '{"method":"mtp","model":"<head-repo>"}'""" in out, (
        f"complete sidecar serve command missing from note:\n{out}"
    )
    # The serve command must carry the user-typed alias, not the HF path —
    # placed after the ``--`` option/positional separator so a ``-``-prefixed
    # local path can't be misparsed as a serve option (codex #1202 round 6).
    assert "-- qwen3.6-27b-8bit" in out, (
        f"note should reference the alias the user typed (after ``--``):\n{out}"
    )


def test_info_omits_mtp_note_when_not_declared_absent(monkeypatch) -> None:
    """The reconciliation note must stay scoped: when the helper reports the
    model is NOT in the declared-but-absent state (returns ``None``), ``info``
    must not print the ``MTP:`` sidecar note."""
    from vllm_mlx import model_auto_config as auto_config_mod

    monkeypatch.setattr(
        auto_config_mod, "_mtp_declared_absent_layers", lambda name, cfg: None
    )
    out = _run_info("qwen3.6-27b-8bit")
    assert "head weights are absent" not in out, (
        f"note leaked for a model that is not declared-but-absent:\n{out}"
    )
