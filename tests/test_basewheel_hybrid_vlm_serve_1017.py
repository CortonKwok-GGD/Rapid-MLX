# SPDX-License-Identifier: Apache-2.0
"""0.10.16 dogfood P1 (④) — a base-wheel serve of a HYBRID-backbone VLM
alias must boot text-only WITHOUT demanding the ``[vision]`` extra.

Context
-------
#1178 added ``mllm_backbone_is_hybrid`` + ``resolve_serving_lane``: a
multimodal alias whose LANGUAGE backbone is hybrid/linear-attention
(Qwen3.6 GatedDeltaNet — ``mlx-community/Qwen3.6-27B-4bit``) auto-downgrades
to the text-only mlx-lm lane at load time and never touches mlx-vlm.

But the CLI boot guard (``serve_command``) hard-required the ``[vision]``
extra whenever ``is_mllm_model(args.model)`` was True — which fires for a
hybrid VLM checkpoint (its config declares ``vision_config``) BEFORE the
auto-downgrade decision. Result: a base-wheel user was pushed into a ~1 GB
``[vision]`` install for a model that then serves text-only.

The fix makes the guard consult the SAME resolved-lane signal the engine
uses (``resolve_serving_lane``) via the ``_serve_will_run_on_mllm_lane``
helper: require ``[vision]`` ONLY when the model will actually run on the
MLLM lane. A genuine VLM (non-hybrid backbone, e.g. qwen3-vl) still requires
it; ``--mllm`` / ``--no-mllm`` are honoured.

These tests pin:
  * the helper decision for hybrid / genuine / forced / text-only / non-VLM,
  * the end-to-end guard behaviour with mlx-vlm mocked ABSENT (base wheel):
    hybrid VLM boots past the guard, genuine VLM still exits rc=2.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _args(model: str = "some/model", *, mllm: bool = False, no_mllm: bool = False):
    return SimpleNamespace(model=model, mllm=mllm, no_mllm=no_mllm)


# ---------------------------------------------------------------------------
# Unit: the decision helper ``_serve_will_run_on_mllm_lane``.
# ---------------------------------------------------------------------------


def _patch_probes(monkeypatch, *, is_mllm: bool, hybrid: bool):
    """Stub the two offline probes ``resolve_serving_lane`` consults so the
    lane decision is exercised without a materialized checkpoint config."""
    from vllm_mlx.api import utils as api_utils

    monkeypatch.setattr(api_utils, "is_mllm_model", lambda name: is_mllm)
    monkeypatch.setattr(api_utils, "mllm_backbone_is_hybrid", lambda name: hybrid)


def test_helper_hybrid_vlm_does_not_run_on_mllm_lane(monkeypatch):
    """A multimodal alias with a hybrid backbone auto-downgrades to text —
    the helper reports it will NOT run on the MLLM lane, so the guard skips
    the ``[vision]`` requirement (the base-wheel fix)."""
    from vllm_mlx import cli

    _patch_probes(monkeypatch, is_mllm=True, hybrid=True)
    assert cli._serve_will_run_on_mllm_lane(_args()) is False


def test_helper_genuine_vlm_runs_on_mllm_lane(monkeypatch):
    """A genuine VLM (non-hybrid backbone) stays on the MLLM lane, so the
    helper reports True and the guard STILL requires ``[vision]``."""
    from vllm_mlx import cli

    _patch_probes(monkeypatch, is_mllm=True, hybrid=False)
    assert cli._serve_will_run_on_mllm_lane(_args()) is True


def test_helper_force_mllm_on_hybrid_still_requires_vision(monkeypatch):
    """Explicit ``--mllm`` wins: even a hybrid backbone is reported as the
    MLLM lane so the operator gets the flag they asked for (and its
    ``[vision]`` requirement)."""
    from vllm_mlx import cli

    _patch_probes(monkeypatch, is_mllm=True, hybrid=True)
    assert cli._serve_will_run_on_mllm_lane(_args(mllm=True)) is True


def test_helper_no_mllm_never_runs_on_mllm_lane(monkeypatch):
    """``--no-mllm`` forces the text lane regardless of the checkpoint — the
    guard must never require ``[vision]`` for it."""
    from vllm_mlx import cli

    _patch_probes(monkeypatch, is_mllm=True, hybrid=False)
    assert cli._serve_will_run_on_mllm_lane(_args(no_mllm=True)) is False


def test_helper_non_vlm_does_not_run_on_mllm_lane(monkeypatch):
    """A plain text model is never on the MLLM lane."""
    from vllm_mlx import cli

    _patch_probes(monkeypatch, is_mllm=False, hybrid=False)
    assert cli._serve_will_run_on_mllm_lane(_args()) is False


# ---------------------------------------------------------------------------
# Integration: drive the real ``serve_command`` boot guard with mlx-vlm
# mocked ABSENT (the fresh ``pip install rapid-mlx`` state, no [vision]).
# ---------------------------------------------------------------------------


class _ReachedPastVisionGuardError(Exception):
    """Sentinel raised by the stubbed audio probe that immediately follows
    the vision guard — proves the vision guard did NOT sys.exit(2)."""


def _mock_mllm_absent(monkeypatch):
    from vllm_mlx.models import mllm as mllm_mod

    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
    )


def _stub_post_guard_sentinel(monkeypatch):
    """Make the audio boot guard (the very next step after the vision guard)
    raise a sentinel so ``serve_command`` stops right there — we only care
    whether the vision guard let us through."""
    import vllm_mlx.audio.probe as audio_probe

    def _raise(_name):
        raise _ReachedPastVisionGuardError()

    monkeypatch.setattr(audio_probe, "is_audio_model_alias", _raise)


def test_serve_guard_hybrid_vlm_boots_without_vision_extra(monkeypatch, capsys):
    """Base wheel (mlx-vlm ABSENT): a hybrid-backbone VLM must pass the boot
    guard WITHOUT the ``[vision]``-required ``sys.exit(2)`` — it will
    auto-downgrade to the text lane."""
    from vllm_mlx import cli

    _patch_probes(monkeypatch, is_mllm=True, hybrid=True)
    _mock_mllm_absent(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)

    args = _args("mlx-community/Qwen3.6-27B-4bit")
    # Reaching the sentinel means the vision guard did NOT exit — the model
    # is allowed to boot text-only from the base wheel.
    with pytest.raises(_ReachedPastVisionGuardError):
        cli.serve_command(args)

    err = capsys.readouterr().err
    assert "[vision]" not in err, (
        "hybrid-backbone VLM must not be pushed into the [vision] install on "
        f"a base wheel; stderr was: {err!r}"
    )


def test_serve_guard_genuine_vlm_still_requires_vision_extra(monkeypatch, capsys):
    """Base wheel (mlx-vlm ABSENT): a genuine VLM (non-hybrid backbone) must
    STILL fail fast with the ``[vision]``-required guard — the fix must not
    weaken real vision aliases."""
    from vllm_mlx import cli

    _patch_probes(monkeypatch, is_mllm=True, hybrid=False)
    _mock_mllm_absent(monkeypatch)
    # If the guard wrongly let this through, the sentinel would surface
    # instead of SystemExit — making the test fail loudly rather than pass.
    _stub_post_guard_sentinel(monkeypatch)

    args = _args("mlx-community/Qwen3-VL-2B-Instruct-4bit")
    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(args)
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert "Qwen3-VL-2B-Instruct-4bit" in err
    assert "[vision]" in err
    # The guard message also surfaces --no-mllm as the text-only escape hatch.
    assert "--no-mllm" in err


def test_serve_guard_no_mllm_skips_vision_extra(monkeypatch):
    """``--no-mllm`` on a genuine VLM must bypass the vision guard entirely
    even on a base wheel (the pre-existing escape hatch, preserved)."""
    from vllm_mlx import cli

    _patch_probes(monkeypatch, is_mllm=True, hybrid=False)
    _mock_mllm_absent(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)

    args = _args("mlx-community/Qwen3-VL-2B-Instruct-4bit", no_mllm=True)
    with pytest.raises(_ReachedPastVisionGuardError):
        cli.serve_command(args)
