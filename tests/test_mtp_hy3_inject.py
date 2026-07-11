# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the HY3 (Tencent Hunyuan 3) native MTP inject module.

Mirrors ``tests/test_mtp_inject_and_install.py`` +
``tests/test_mtp_gemma4_assistant_inject.py`` (the qwen3_5 / gemma4 paths)
for ``model_type == "hy_v3"``. Everything runs against a tiny synthetic
HY3 model — end-to-end forward through the real 155 GB target is out of
scope for CI (covered by the operator smoke + the weekly Golden Path job).

Coverage
--------

1. **Detection** — ``detect_mtp_eligibility`` reads HY3's
   ``num_nextn_predict_layers`` (root + ``text_config``) and does NOT
   cross-contaminate the Qwen3.5 ``mtp_num_hidden_layers`` key.
2. **Head build** — :func:`build_hy3_mtp_module` produces the
   DeepSeek-V3-shaped param tree (``enorm`` / ``hnorm`` / ``eh_proj`` /
   a single MoE ``DecoderLayer`` / ``norm``).
3. **Wiring probe** — ``inject_hy3_mtp_support(..., allow_random_init=True)``
   attaches the four MTP contract surfaces, and ``validate_hy3_mtp_support``
   returns True; a forward through them produces the right shapes.
4. **Sidecar refusal (fail-closed default)** — no sidecar + no
   ``allow_random_init`` would resolve the default HF repo; with a bogus
   sidecar path it returns False and leaves the model unmodified.
5. **Missing-tensor refusal** — a sidecar that omits a required tensor
   is rejected (no partial-random head).
6. **Architecture guard** — a non-``hy_v3`` model (lacking ``num_nextn``)
   builds no head.
7. **Dispatcher routing** — ``model_type == "hy_v3"`` routes to
   ``inject_hy3_mtp_support`` / ``validate_hy3_mtp_support`` in the
   dispatch tables.
8. **Default sidecar** — the module exposes the published-repo default so
   a bare ``--speculative-config '{"method":"mtp"}'`` boot resolves it.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _tiny_hy3_args(**overrides):
    """Minimal ``hy_v3.ModelArgs`` for a fake HY3 target."""
    from vllm_mlx.models.hy_v3 import ModelArgs

    defaults = dict(
        model_type="hy_v3",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        expert_hidden_dim=32,
        first_k_dense_replace=1,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000.0},
        num_nextn_predict_layers=1,
        tie_word_embeddings=False,
    )
    defaults.update(overrides)
    return ModelArgs(**defaults)


def _build_tiny_hy3_model(**overrides):
    from vllm_mlx.models.hy_v3 import Model

    return Model(_tiny_hy3_args(**overrides))


# ---------------------------------------------------------------------------
# 1. Detection — num_nextn_predict_layers vs mtp_num_hidden_layers
# ---------------------------------------------------------------------------


def test_detect_hy3_num_nextn_chain():
    """HY3 with ``num_nextn_predict_layers == 1`` → CHAIN."""
    from vllm_mlx.spec_decode.mtp import MTPEligibility, detect_mtp_eligibility

    config = {"model_type": "hy_v3", "num_nextn_predict_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.CHAIN


def test_detect_hy3_num_nextn_under_text_config():
    """HY3 nextn count nested under ``text_config`` still resolves CHAIN."""
    from vllm_mlx.spec_decode.mtp import MTPEligibility, detect_mtp_eligibility

    config = {"model_type": "hy_v3", "text_config": {"num_nextn_predict_layers": 1}}
    assert detect_mtp_eligibility(config) is MTPEligibility.CHAIN


def test_detect_hy3_zero_nextn_rejected():
    """HY3 with ``num_nextn_predict_layers == 0`` (stripped) → NONE."""
    from vllm_mlx.spec_decode.mtp import MTPEligibility, detect_mtp_eligibility

    config = {"model_type": "hy_v3", "num_nextn_predict_layers": 0}
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE


def test_detect_hy3_mtp_key_fallback():
    """HY3 also accepts the Qwen-style ``mtp_num_hidden_layers`` fallback."""
    from vllm_mlx.spec_decode.mtp import MTPEligibility, detect_mtp_eligibility

    config = {"model_type": "hy_v3", "mtp_num_hidden_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.CHAIN


def test_detect_qwen35_does_not_read_num_nextn():
    """Cross-contamination guard: Qwen3.5 must NOT pick up HY3's
    ``num_nextn_predict_layers`` key — its head lives under
    ``mtp_num_hidden_layers`` only."""
    from vllm_mlx.spec_decode.mtp import MTPEligibility, detect_mtp_eligibility

    config = {"model_type": "qwen3_5", "num_nextn_predict_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE


def test_detect_deepseek_v3_num_nextn_still_off_allowlist():
    """``deepseek_v3`` uses the same nextn convention but is NOT on the
    MTP allowlist — must stay NONE (no accidental promotion)."""
    from vllm_mlx.spec_decode.mtp import MTPEligibility, detect_mtp_eligibility

    config = {"model_type": "deepseek_v3", "num_nextn_predict_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE


# ---------------------------------------------------------------------------
# 2. Head build — DeepSeek-V3-shaped param tree
# ---------------------------------------------------------------------------


def test_build_hy3_mtp_module_param_tree():
    """The head exposes ``enorm`` / ``hnorm`` / ``eh_proj`` / ``norm`` plus
    a single MoE ``DecoderLayer`` whose param names match a quantized
    backbone MoE layer (so the sidecar tree lines up 1:1)."""
    from mlx.utils import tree_flatten

    from vllm_mlx.spec_decode.mtp.hy3_head import build_hy3_mtp_module

    args = _tiny_hy3_args()
    mtp = build_hy3_mtp_module(args, 1)
    keys = {k for k, _ in tree_flatten(mtp.parameters())}
    for want in (
        "enorm.weight",
        "hnorm.weight",
        "eh_proj.weight",
        "norm.weight",
        "layers.0.mlp.router.gate.weight",
        "layers.0.mlp.router.expert_bias",
        "layers.0.mlp.switch_mlp.gate_proj.weight",
        "layers.0.mlp.switch_mlp.down_proj.weight",
        "layers.0.mlp.switch_mlp.up_proj.weight",
        "layers.0.mlp.shared_mlp.gate_proj.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.k_norm.weight",
        "layers.0.self_attn.q_proj.weight",
    ):
        assert want in keys, f"MTP head missing expected tensor {want!r}"


def test_build_hy3_mtp_module_rejects_multi_layer():
    """HY3 ships exactly one next-n layer; >1 must refuse loudly."""
    from vllm_mlx.spec_decode.mtp.hy3_head import build_hy3_mtp_module

    args = _tiny_hy3_args()
    with pytest.raises(ValueError):
        build_hy3_mtp_module(args, 2)
    with pytest.raises(ValueError):
        build_hy3_mtp_module(args, 0)


# ---------------------------------------------------------------------------
# 3. Wiring probe — four surfaces attach under random init
# ---------------------------------------------------------------------------


def test_inject_attaches_four_surfaces_random_init():
    """``allow_random_init=True`` attaches ``mtp`` / ``mtp_forward`` /
    ``make_mtp_cache`` / ``__call__(return_hidden, n_confirmed)`` and a
    forward through them produces the right shapes."""
    from vllm_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    model = _build_tiny_hy3_model()
    assert inject_hy3_mtp_support(model, allow_random_init=True) is True
    assert validate_hy3_mtp_support(model) is True

    ids = mx.array([[1, 2, 3, 4]])
    out, hidden = model(ids, return_hidden=True)
    assert out.shape == (1, 4, 128)
    assert hidden.shape == (1, 4, 64)

    mtp_cache = model.make_mtp_cache()
    mtp_logits = model.mtp_forward(hidden, ids, mtp_cache)
    mx.eval(mtp_logits)
    assert mtp_logits.shape == (1, 4, 128)


def test_validate_false_before_inject():
    """A vanilla HY3 model has none of the MTP surfaces."""
    from vllm_mlx.spec_decode.mtp.hy3_inject import validate_hy3_mtp_support

    model = _build_tiny_hy3_model()
    assert validate_hy3_mtp_support(model) is False


# ---------------------------------------------------------------------------
# 4. Sidecar refusal / missing-tensor refusal
# ---------------------------------------------------------------------------


def test_inject_refuses_unresolvable_sidecar():
    """A sidecar path that resolves to nothing → False, model unmodified."""
    from vllm_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    model = _build_tiny_hy3_model()
    ok = inject_hy3_mtp_support(model, mtp_sidecar="/nonexistent/path/nowhere")
    assert ok is False
    assert validate_hy3_mtp_support(model) is False


def test_inject_refuses_sidecar_missing_tensor(tmp_path):
    """A sidecar missing a required tensor is rejected (no partial-random
    head), mirroring the qwen3_5 coverage check."""
    from mlx.utils import tree_flatten

    from vllm_mlx.spec_decode.mtp.hy3_head import build_hy3_mtp_module
    from vllm_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    # Build the (unquantized) param tree, then drop one tensor.
    args = _tiny_hy3_args()
    template = build_hy3_mtp_module(args, 1)
    weights = dict(tree_flatten(template.parameters()))
    weights.pop("eh_proj.weight")  # required — must trip the coverage gate
    for k in list(weights):
        mx.eval(weights[k])
    side_dir = tmp_path / "sidecar"
    side_dir.mkdir()
    mx.save_safetensors(str(side_dir / "model-mtp.safetensors"), weights)

    model = _build_tiny_hy3_model()
    # Base is unquantized here, so the injected head is unquantized too;
    # the coverage gate compares against that same tree.
    ok = inject_hy3_mtp_support(model, mtp_sidecar=str(side_dir))
    assert ok is False
    assert validate_hy3_mtp_support(model) is False


def test_inject_loads_synthetic_full_sidecar(tmp_path):
    """A complete sidecar (all required tensors) round-trips through
    inject → validate → True."""
    from mlx.utils import tree_flatten

    from vllm_mlx.spec_decode.mtp.hy3_head import build_hy3_mtp_module
    from vllm_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    args = _tiny_hy3_args()
    template = build_hy3_mtp_module(args, 1)
    weights = dict(tree_flatten(template.parameters()))
    for k in list(weights):
        mx.eval(weights[k])
    side_dir = tmp_path / "sidecar"
    side_dir.mkdir()
    mx.save_safetensors(str(side_dir / "model-mtp.safetensors"), weights)

    model = _build_tiny_hy3_model()
    assert inject_hy3_mtp_support(model, mtp_sidecar=str(side_dir)) is True
    assert validate_hy3_mtp_support(model) is True


# ---------------------------------------------------------------------------
# 5. Architecture guard
# ---------------------------------------------------------------------------


def test_inject_refuses_model_without_num_nextn():
    """A HY3-shaped model whose config advertises no next-n layer builds
    no head (num_nextn_predict_layers=0)."""
    from vllm_mlx.spec_decode.mtp.hy3_inject import (
        inject_hy3_mtp_support,
        validate_hy3_mtp_support,
    )

    model = _build_tiny_hy3_model(num_nextn_predict_layers=0)
    ok = inject_hy3_mtp_support(model, allow_random_init=True)
    assert ok is False
    assert validate_hy3_mtp_support(model) is False


# ---------------------------------------------------------------------------
# 6. Dispatcher routing
# ---------------------------------------------------------------------------


def test_dispatch_tables_route_hy_v3():
    """``hy_v3`` is registered in both dispatch tables → the family
    inject / validate entry points."""
    from vllm_mlx.spec_decode.mtp.dispatch import (
        _MTP_INJECT_DISPATCH,
        _MTP_VALIDATE_DISPATCH,
    )

    assert _MTP_INJECT_DISPATCH["hy_v3"] == (
        "vllm_mlx.spec_decode.mtp.hy3_inject",
        "inject_hy3_mtp_support",
    )
    assert _MTP_VALIDATE_DISPATCH["hy_v3"] == (
        "vllm_mlx.spec_decode.mtp.hy3_inject",
        "validate_hy3_mtp_support",
    )


def test_dispatch_inject_routes_and_attaches():
    """End-to-end through the dispatcher: ``dispatch_mtp_inject(model,
    'hy_v3', allow_random_init=True)`` attaches the surfaces, and
    ``dispatch_mtp_validate`` confirms them."""
    from vllm_mlx.spec_decode.mtp import (
        dispatch_mtp_inject,
        dispatch_mtp_validate,
    )

    model = _build_tiny_hy3_model()
    assert dispatch_mtp_inject(model, "hy_v3", allow_random_init=True) is True
    assert dispatch_mtp_validate(model, "hy_v3") is True


# ---------------------------------------------------------------------------
# 7. Default sidecar repo
# ---------------------------------------------------------------------------


def test_default_sidecar_repo_constant():
    """The module exposes the published-sidecar default so a bare
    ``--speculative-config '{"method":"mtp"}'`` boot auto-resolves it."""
    from vllm_mlx.spec_decode.mtp import hy3_inject

    assert hy3_inject.DEFAULT_HY3_MTP_SIDECAR == "mlx-community/Hy3-preview-MTP-4bit"
