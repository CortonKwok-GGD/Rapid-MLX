# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MTP norm double-shift guard.

These exercise the PURE decision helpers (``_norms_already_shifted`` and
``_classify_norm``) with synthetic means only -- no network, no model
downloads, no MLX tensors.
"""

from scripts.extract_mtp_weights import (
    NORM_AMBIGUITY_BAND,
    _classify_norm,
    _norms_already_shifted,
)


def test_unshifted_norms_get_shifted():
    # HF ``(1 + w)`` convention: the MTP norm mean sits ~1.0 below the
    # +1-shifted backbone reference. Guard must say NOT already shifted so the
    # +1.0 shift is applied.
    backbone_mean = 1.02
    mtp_mean = backbone_mean - 1.0  # 0.02
    assert _norms_already_shifted(mtp_mean, backbone_mean) is False


def test_already_shifted_norms_are_detected():
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


def test_classify_unshifted():
    # Mean hugging the unshifted reference (backbone - 1.0).
    assert _classify_norm(0.02, 1.02) == "unshifted"


def test_classify_already_shifted():
    # Mean hugging the shifted backbone reference.
    assert _classify_norm(1.01, 1.02) == "already_shifted"


def test_classify_ambiguous_far_from_both():
    # backbone 1.0 -> references at 1.0 (shifted) and 0.0 (unshifted). A mean of
    # 0.4 is >0.3 from both, so it must NOT be force-classified: an independently
    # trained norm is excluded from the consensus vote rather than misread as
    # unshifted and double-shifted (the codex-flagged failure mode).
    assert min(abs(0.4 - 1.0), abs(0.4 - 0.0)) > NORM_AMBIGUITY_BAND
    assert _classify_norm(0.4, 1.0) == "ambiguous"


def test_classify_band_edge_is_not_ambiguous():
    # Exactly at the band edge from the unshifted reference stays classified.
    assert _classify_norm(0.0 + NORM_AMBIGUITY_BAND, 1.0) == "unshifted"
