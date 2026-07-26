# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MTP norm double-shift guard.

These exercise the PURE decision helper ``_norms_already_shifted`` with
synthetic means only -- no network, no model downloads, no MLX tensors.
"""

from scripts.extract_mtp_weights import _find_backbone_reference, _norms_already_shifted


def test_unshifted_norms_get_shifted():
    # HF ``(1 + w)`` convention: the MTP norm mean sits ~1.0 below the
    # +1-shifted backbone reference. Guard must say NOT already shifted so the
    # +1.0 shift is applied.
    backbone_mean = 1.02
    mtp_mean = backbone_mean - 1.0  # 0.02
    assert _norms_already_shifted(mtp_mean, backbone_mean) is False


def test_already_shifted_norms_are_skipped():
    # Source already stores MTP norms in MLX's shifted convention: the MTP mean
    # already matches the backbone. Guard must detect this so the shift is
    # skipped (avoids a silent w+2 double-shift).
    backbone_mean = 1.02
    mtp_mean = 1.03
    assert _norms_already_shifted(mtp_mean, backbone_mean) is True


def test_boundary_midpoint_prefers_shift():
    # Exactly halfway between the shifted and unshifted references: equidistant,
    # so the strict ``<`` resolves to "not already shifted" -> shift applied.
    backbone_mean = 1.0
    mtp_mean = backbone_mean - 0.5  # 0.5, equidistant from 1.0 and 0.0
    assert _norms_already_shifted(mtp_mean, backbone_mean) is False


def test_boundary_just_past_midpoint_detects_shift():
    # Just closer to the backbone than to backbone-1.0 -> already shifted.
    backbone_mean = 1.0
    mtp_mean = backbone_mean - 0.4  # 0.6, closer to 1.0 than to 0.0
    assert _norms_already_shifted(mtp_mean, backbone_mean) is True


def test_find_backbone_reference_matches_same_norm_type():
    mtp_norm_keys = [
        "mtp.pre_fc_norm_embedding.weight",  # MTP-only: no backbone counterpart
        "mtp.layers.0.input_layernorm.weight",
    ]
    backbone_keys = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
    ]
    mtp_key, backbone_key = _find_backbone_reference(mtp_norm_keys, backbone_keys)
    assert mtp_key == "mtp.layers.0.input_layernorm.weight"
    assert backbone_key == "model.layers.0.input_layernorm.weight"


def test_find_backbone_reference_skips_mtp_keys_and_returns_none_when_absent():
    # Only an MTP-specific norm with no backbone counterpart present.
    mtp_norm_keys = ["mtp.pre_fc_norm_hidden.weight"]
    backbone_keys = ["model.layers.0.input_layernorm.weight", "mtp.something.weight"]
    assert _find_backbone_reference(mtp_norm_keys, backbone_keys) == (None, None)
