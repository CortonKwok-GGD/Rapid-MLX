# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the architecture-aware KV footprint estimator.

These pin the contract that the D-METAL-CAP admission gate depends on
(``scheduler._resolve_kv_bytes_per_token`` → ``estimate_kv_footprint``):

* Dense / unknown configs are BYTE-IDENTICAL to the uniform formula (no
  spurious change to a codex-hardened, over-estimate-safe admission path).
* Recognized hybrids (sliding-window, KV-sharing, recurrent) get the smaller,
  accurate per-token growth with the window / recurrent footprint moved into a
  per-sequence fixed baseline.
* The estimate never under-counts the counted layers (a sliding layer with no
  readable window, or an unknown layer type, is charged as full-growth).

Hermetic: pure config dataclasses (``SimpleNamespace``) and dicts — no model
load, no network.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm_mlx.kv_estimation import KVFootprintEstimate, estimate_kv_footprint


def _uniform(num_layers: int, kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    return 2 * num_layers * kv_heads * head_dim * dtype_bytes


def _est(cfg, *, num_layers, kv_heads, head_dim, dtype_bytes=2) -> KVFootprintEstimate:
    return estimate_kv_footprint(
        cfg,
        dtype_bytes=dtype_bytes,
        uniform_per_token_bytes=_uniform(num_layers, kv_heads, head_dim, dtype_bytes),
        base_num_layers=num_layers,
        base_kv_heads=kv_heads,
        base_head_dim=head_dim,
    )


class TestDenseFallback:
    """Anything not a recognized hybrid stays byte-identical to uniform."""

    def test_dense_config_is_byte_identical(self):
        # No layer_types, no sliding_window, no num_kv_shared_layers.
        cfg = SimpleNamespace(num_hidden_layers=32, num_key_value_heads=8, head_dim=128)
        est = _est(cfg, num_layers=32, kv_heads=8, head_dim=128)
        assert est.per_token_growth_bytes == _uniform(32, 8, 128, 2)
        assert est.fixed_baseline_bytes == 0

    def test_sliding_window_without_layer_types_falls_back(self):
        # Mistral-like: a global ``sliding_window`` field but NO per-layer
        # ``layer_types``. We must NOT treat every layer as sliding (that would
        # zero the per-token growth) — ambiguous → uniform.
        cfg = SimpleNamespace(
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
            sliding_window=4096,
        )
        est = _est(cfg, num_layers=32, kv_heads=8, head_dim=128)
        assert est.per_token_growth_bytes == _uniform(32, 8, 128, 2)
        assert est.fixed_baseline_bytes == 0

    def test_wrong_length_layer_types_falls_back(self):
        cfg = SimpleNamespace(
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
            layer_types=["full_attention"] * 30,  # len != num_layers
        )
        est = _est(cfg, num_layers=32, kv_heads=8, head_dim=128)
        assert est.per_token_growth_bytes == _uniform(32, 8, 128, 2)
        assert est.fixed_baseline_bytes == 0

    def test_non_string_layer_types_falls_back(self):
        cfg = SimpleNamespace(
            num_hidden_layers=4,
            num_key_value_heads=8,
            head_dim=128,
            layer_types=[0, 1, 2, 3],  # not strings
        )
        est = _est(cfg, num_layers=4, kv_heads=8, head_dim=128)
        assert est.per_token_growth_bytes == _uniform(4, 8, 128, 2)
        assert est.fixed_baseline_bytes == 0

    def test_zero_base_dims_returns_uniform(self):
        # Defensive: the scheduler never calls us with 0 uniform, but the
        # estimator must not crash / invent an estimate on degenerate input.
        cfg = SimpleNamespace(num_hidden_layers=0)
        est = estimate_kv_footprint(
            cfg,
            dtype_bytes=2,
            uniform_per_token_bytes=0,
            base_num_layers=0,
            base_kv_heads=0,
            base_head_dim=0,
        )
        assert est.per_token_growth_bytes == 0
        assert est.fixed_baseline_bytes == 0


class TestKVSharing:
    """Gemma-4 / Gemma-3n ``num_kv_shared_layers`` borrower layers = 0 KV."""

    def test_shared_layers_reduce_growth_exactly(self):
        # 35 layers, all full-attention, last 20 borrow → 15 producers grow.
        n, n_shared, kv, hd = 35, 20, 1, 128
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=["full_attention"] * n,
            num_kv_shared_layers=n_shared,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == 2 * (n - n_shared) * kv * hd * 2
        assert est.fixed_baseline_bytes == 0

    def test_shared_layers_without_layer_types_falls_back(self):
        # num_kv_shared_layers alone (no per-layer map) is NOT enough to zero
        # borrowers — without layer_types we cannot verify the last-N contract,
        # so we stay on the uniform estimate (codex round 1 BLOCKING #3).
        n, n_shared, kv, hd = 40, 10, 4, 64
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            num_kv_shared_layers=n_shared,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.fixed_baseline_bytes == 0

    def test_shared_layers_with_unverifiable_split_not_zeroed(self):
        # num_kv_shared_layers present WITH layer_types, but the last-N borrower
        # types have no same-type producer below the split (a full-attention
        # borrower with only sliding producers) → the last-N contract is
        # unverified, so no layer is zeroed (over-count, never under-count).
        n, n_shared, kv, hd = 10, 3, 4, 64
        # First 7 (producers) all sliding; last 3 (borrowers) full → a full
        # borrower has no full producer below the split.
        layer_types = ["sliding_attention"] * 7 + ["full_attention"] * 3
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            num_kv_shared_layers=n_shared,
            sliding_window=128,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        # Nothing zeroed: 3 full layers grow, 7 sliding layers -> window
        # baseline (all 10 layers counted).
        assert est.per_token_growth_bytes == 2 * 3 * kv * hd * 2
        assert est.fixed_baseline_bytes == 2 * 7 * 128 * kv * hd * 2

    def test_zero_shared_is_not_a_hybrid_signal(self):
        # num_kv_shared_layers=0 (dense Gemma-4 12B/26B/31B) → no reduction,
        # uniform fallback.
        cfg = SimpleNamespace(
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
            num_kv_shared_layers=0,
        )
        est = _est(cfg, num_layers=32, kv_heads=8, head_dim=128)
        assert est.per_token_growth_bytes == _uniform(32, 8, 128, 2)
        assert est.fixed_baseline_bytes == 0


class TestSlidingWindow:
    """GPT-OSS ~50% sliding-window layers are window-bounded, not per-token."""

    def test_half_sliding_halves_growth_and_adds_window_baseline(self):
        n, kv, hd, window = 24, 8, 64, 128
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            sliding_window=window,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        n_full = n // 2
        n_sliding = n // 2
        assert est.per_token_growth_bytes == 2 * n_full * kv * hd * 2
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2) // 2  # halved
        assert est.fixed_baseline_bytes == 2 * n_sliding * window * kv * hd * 2

    def test_sliding_without_window_charged_as_full_growth(self):
        # A sliding layer whose window is unreadable cannot be bounded → it is
        # charged as full-growth so we never under-count.
        n, kv, hd = 24, 8, 64
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            # sliding_window intentionally absent
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.fixed_baseline_bytes == 0


class TestRecurrent:
    """Qwen3.5/3.6 GatedDeltaNet linear-attention layers have zero growth.

    Recurrent layers keep a FIXED state that our configs do not expose the
    fields to size, so we model zero per-token growth (the correctness win) and
    do NOT invent a fixed-state term (codex round 1 BLOCKING #2). Only the
    full-attention layers show up in the estimate.
    """

    def test_linear_layers_contribute_zero_growth_and_zero_baseline(self):
        # 48 layers, 3:1 linear:full → only the 12 full layers grow; the 36
        # recurrent layers contribute nothing to either term.
        n, kv, hd = 48, 2, 128
        pattern = [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        layer_types = pattern * (n // 4)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        n_full = n // 4
        assert est.per_token_growth_bytes == 2 * n_full * kv * hd * 2
        assert est.fixed_baseline_bytes == 0

    def test_mamba_marker_is_recurrent(self):
        n, kv, hd = 8, 4, 64
        layer_types = ["mamba"] * (n - 1) + ["full_attention"]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == 2 * 1 * kv * hd * 2
        assert est.fixed_baseline_bytes == 0

    def test_full_attention_name_containing_linear_is_not_recurrent(self):
        # Exact-match allowlist (codex round 1 BLOCKING #4): an unfamiliar
        # full-attention type that merely CONTAINS "linear"/"gated"/"local" must
        # NOT be misread as zero-growth — it is charged as full-growth.
        n, kv, hd = 6, 4, 64
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=["gated_full_attention", "linear_global_attention"] * (n // 2),
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.fixed_baseline_bytes == 0


class TestHybridDimsAndNesting:
    def test_global_head_dim_used_for_full_layers(self):
        # Realistic Gemma-4 text: 35 layers, 4:1 sliding:full pattern, last 20
        # borrow, full layers use the wider global_head_dim/global kv heads.
        n, n_shared, kv, hd = 35, 20, 1, 256
        global_hd, global_kv, window = 512, 1, 512
        layer_types = (["sliding_attention"] * 4 + ["full_attention"]) * (n // 5)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            global_head_dim=global_hd,
            num_global_key_value_heads=global_kv,
            num_kv_shared_layers=n_shared,
            sliding_window=window,
            layer_types=layer_types,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        # Producers = first 15 layers. Full at indices 4, 9, 14 → 3 full,
        # 12 sliding. Borrowers (last 20) contribute nothing.
        assert est.per_token_growth_bytes == 2 * 3 * global_kv * global_hd * 2
        assert est.fixed_baseline_bytes == 12 * 2 * window * kv * hd * 2

    def test_layer_structure_read_from_text_config(self):
        # Multimodal configs nest the language layer structure under
        # ``text_config``; the estimator must find it there.
        n, kv, hd, window = 24, 8, 64, 128
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = SimpleNamespace(
            text_config=SimpleNamespace(
                num_key_value_heads=kv,
                head_dim=hd,
                layer_types=layer_types,
                sliding_window=window,
            )
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == 2 * (n // 2) * kv * hd * 2
        assert est.fixed_baseline_bytes == 2 * (n // 2) * window * kv * hd * 2

    def test_dict_config_supported(self):
        # The offline hybrid probe reads parsed config.json dicts.
        n, kv, hd, window = 24, 8, 64, 128
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = {
            "num_hidden_layers": n,
            "num_key_value_heads": kv,
            "head_dim": hd,
            "layer_types": layer_types,
            "sliding_window": window,
        }
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == 2 * (n // 2) * kv * hd * 2
        assert est.fixed_baseline_bytes == 2 * (n // 2) * window * kv * hd * 2

    def test_unknown_layer_type_charged_as_full_growth(self):
        # An unrecognized attention-type string must never under-count → full.
        n, kv, hd = 4, 8, 128
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=["mystery_attention"] * n,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.fixed_baseline_bytes == 0


class TestNeverUnderCounts:
    """The projected footprint is always >= the true counted-layer footprint."""

    def test_projection_upper_bounds_true_footprint(self):
        # GPT-OSS-like hybrid. For any token count, fixed_baseline +
        # tokens*growth must be >= the true footprint of the counted layers:
        #   full layers: exact tokens*per-layer
        #   sliding layers: min(tokens, window)*per-layer  (<= window*per-layer)
        n, kv, hd, window, db = 24, 8, 64, 128, 2
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            sliding_window=window,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd, dtype_bytes=db)
        n_full = n // 2
        n_sliding = n // 2
        full_per_layer = 2 * kv * hd * db
        for tokens in (1, 100, window, window + 1, 4096, 32768):
            projected = est.fixed_baseline_bytes + tokens * est.per_token_growth_bytes
            true_full = n_full * tokens * full_per_layer
            true_sliding = n_sliding * min(tokens, window) * full_per_layer
            assert projected >= true_full + true_sliding
