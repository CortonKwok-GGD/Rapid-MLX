# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the architecture-aware KV footprint estimator.

These pin the contract that the D-METAL-CAP admission gate depends on
(``scheduler._resolve_kv_bytes_per_token`` → ``estimate_kv_footprint``):

* Dense / unknown configs are BYTE-IDENTICAL to the uniform formula (no
  spurious change to a codex-hardened, over-estimate-safe admission path).
* Recognized hybrids (sliding-window, KV-sharing, recurrent) get the smaller,
  accurate per-token growth: only full-attention layers grow unbounded per
  token; sliding-window layers reserve their WHOLE window buffer once per
  sequence, and recurrent layers reserve a token-independent fixed state — both
  in the fixed baseline.
* The estimate is over-count-safe throughout — it never under-counts the counted
  layers. A sliding layer with no readable window (or an unknown layer type) is
  charged as full unbounded growth; a sliding layer WITH a window reserves the
  whole window (an upper bound on a block-rounded rotating cache).

Hermetic: pure config dataclasses (``SimpleNamespace``) and dicts — no model
load, no network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
        # Nothing zeroed: 3 full layers grow unbounded, 7 sliding layers reserve
        # their whole window buffer (all 10 layers counted, none borrowed).
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

    def test_half_sliding_halves_growth_and_reserves_whole_window(self):
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
        # Only the full layers grow unbounded per token — halved vs uniform.
        assert est.per_token_growth_bytes == 2 * n_full * kv * hd * 2
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2) // 2  # halved
        # Sliding layers reserve their WHOLE window buffer once per sequence —
        # the over-count-safe upper bound (codex round 9 BLOCKING).
        assert est.fixed_baseline_bytes == 2 * n_sliding * window * kv * hd * 2

    def test_projection_below_uniform_for_long_generation(self):
        # The reduction: for a long generation the arch-aware projection (halved
        # per-token growth + a bounded window baseline) stays well below the
        # uniform per-token over-count that keeps multiplying by every token.
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
        tokens = 8192  # >> window
        projected = est.fixed_baseline_bytes + est.per_token_growth_bytes * tokens
        assert projected < _uniform(n, kv, hd, 2) * tokens

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
    """Qwen3.5/3.6 GatedDeltaNet layers have zero PER-TOKEN growth.

    Their fixed recurrent state does not grow with generated tokens, so it lands
    in the per-sequence baseline (sized from GatedDeltaNet's real config fields)
    rather than the per-token term. Any recurrent family we do NOT size exactly
    (Mamba/Mamba2, RWKV, generic) falls back to full per-token growth so it is
    never left unreserved (codex round 2 BLOCKING #1 + round 7 BLOCKING #1).
    """

    # GatedDeltaNet state, sized from the same fields as
    # ``mlx_lm.models.qwen3_next.Qwen3NextGatedDeltaNet``, at fp32/element.
    _GDN_FIELDS = dict(
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )

    @staticmethod
    def _gdn_state_bytes():
        num_v, num_k, hk, hv, conv = 32, 16, 128, 128, 4
        key_dim = num_k * hk
        value_dim = num_v * hv
        recurrent = num_v * hk * hv
        conv_dim = 2 * key_dim + value_dim
        conv_state = (conv - 1) * conv_dim
        return 4 * (recurrent + conv_state)

    def test_gateddeltanet_state_reserved_in_baseline_zero_growth(self):
        # 48 layers, 3:1 linear:full → only the 12 full layers grow per token;
        # the 36 GatedDeltaNet layers reserve their fixed state in the baseline.
        n, kv, hd = 48, 2, 128
        pattern = ["linear_attention"] * 3 + ["full_attention"]
        layer_types = pattern * (n // 4)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            **self._GDN_FIELDS,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        n_full = n // 4
        n_recurrent = n - n_full
        assert est.per_token_growth_bytes == 2 * n_full * kv * hd * 2
        assert est.fixed_baseline_bytes == n_recurrent * self._gdn_state_bytes()

    @pytest.mark.parametrize(
        "gdn_type",
        ["linear_attention", "gated_delta_net", "gated_deltanet"],
    )
    def test_gateddeltanet_aliases_all_sized(self, gdn_type):
        # Every GatedDeltaNet alias the classifier treats as recurrent must ALSO
        # route to the GatedDeltaNet sizer — otherwise the alias classifies as
        # zero-growth but its state cannot be sized, reverting to full per-token
        # growth and losing the reduction this module exists to deliver (codex
        # round 5 BLOCKING). All three spellings must yield the identical zero
        # per-token growth + baseline-reserved fixed state.
        n, kv, hd = 8, 4, 64
        layer_types = [gdn_type] * (n - 1) + ["full_attention"]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            **self._GDN_FIELDS,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        # Only the single full layer grows per token; the recurrent layers reserve
        # their fixed state in the baseline (never full per-token growth).
        assert est.per_token_growth_bytes == 2 * 1 * kv * hd * 2
        assert est.fixed_baseline_bytes == (n - 1) * self._gdn_state_bytes()

    def test_unsizeable_recurrent_falls_back_to_full_growth(self):
        # linear_attention layers WITHOUT any GatedDeltaNet/Mamba sizing fields
        # → cannot size the state → charged the uniform per-token estimate
        # (full growth), never left unreserved.
        n, kv, hd = 8, 4, 64
        layer_types = ["linear_attention"] * (n - 1) + ["full_attention"]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        # All 8 layers charged full growth (uniform) — no reduction, no baseline.
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.fixed_baseline_bytes == 0

    def test_mamba_not_sized_falls_back_to_full_growth(self):
        # We deliberately do NOT size a Mamba/Mamba2 state: the conv buffer
        # channel count differs between Mamba1 (d_inner) and Mamba2 (d_inner +
        # grouped B/C channels), so a single formula would under-reserve Mamba2
        # (codex round 7 BLOCKING #1). This engine ships no Mamba model, so its
        # layers conservatively fall back to full per-token growth — over-count
        # safe, never unreserved — even when Mamba-shaped fields are present.
        n, kv, hd = 8, 4, 64
        layer_types = ["mamba"] * (n - 1) + ["full_attention"]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            mamba_d_state=128,
            mamba_d_conv=4,
            mamba_n_heads=24,
            mamba_d_head=64,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        # All 8 layers charged full growth — no baseline, no under-count.
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.fixed_baseline_bytes == 0

    def test_gateddeltanet_missing_conv_kernel_falls_back(self):
        # GatedDeltaNet head fields present but linear_conv_kernel_dim ABSENT →
        # cannot size the conv buffer → the whole layer falls back to full
        # growth rather than silently omitting the conv state (codex round 3
        # BLOCKING #1).
        n, kv, hd = 8, 4, 64
        layer_types = ["linear_attention"] * (n - 1) + ["full_attention"]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            linear_num_value_heads=32,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            # linear_conv_kernel_dim intentionally absent
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
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

    def test_generic_recurrent_family_not_mis_sized(self):
        # An RWKV (or other generic recurrent) layer that incidentally exposes
        # Mamba-ish generic fields must NOT be stamped with a bogus fixed state —
        # only GatedDeltaNet is sized; every other recurrent family falls back to
        # full growth (codex round 4 BLOCKING #2 + round 7 BLOCKING #1).
        n, kv, hd = 8, 4, 64
        layer_types = ["rwkv"] * (n - 1) + ["full_attention"]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            # generic fields a naive recurrent sizer might otherwise consume:
            state_size=128,
            intermediate_size=1536,
            conv_kernel_size=4,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        # No baseline reserved (only GatedDeltaNet is sized); all layers full.
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
        # Sliding layers use local dims; whole window reserved in the baseline.
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
        # GPT-OSS-like hybrid. The scheduler projection
        #   fixed_baseline + tokens * per_token_growth
        # must be >= the true footprint of the counted layers for ANY token
        # count, since that is the D-METAL-CAP never-under-count guarantee:
        #   full layers: exact tokens*per-layer
        #   sliding layers: min(tokens, window)*per-layer  (<= window*per-layer,
        #                   the whole-window buffer reserved in the baseline)
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
            # Over-count-safe: the whole-window baseline is >= any decode/prefill
            # peak of the sliding layers, so the projection never under-counts.
            assert projected >= true_full + true_sliding

    def test_full_layer_growth_never_below_uniform_base(self):
        # A pathological config where global_head_dim / num_global_key_value_heads
        # are SMALLER than the base dims must not shrink full-layer per-token
        # growth below the uniform base-layer charge (codex round 4 BLOCKING #1).
        n, kv, hd, db = 8, 8, 128, 2
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=["full_attention"] * n,
            global_head_dim=16,  # << base head_dim
            num_global_key_value_heads=1,  # << base kv heads
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd, dtype_bytes=db)
        # Clamped to the uniform base-layer size, not the tiny global dims.
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, db)
        assert est.fixed_baseline_bytes == 0

    def test_sliding_rate_never_below_uniform_base(self):
        # Symmetric to the full-layer clamp: a nested/malformed config whose
        # local (sliding) dims read SMALLER than the base dims the scheduler used
        # must not shrink the sliding per-layer rate below the uniform per-layer
        # charge (codex round 7 BLOCKING #2). Here the struct exposes a tiny
        # local head_dim; the base head_dim passed by the caller is larger, so
        # the whole-window baseline is computed at the clamped (base) rate.
        n, kv, hd, db, window = 8, 8, 128, 2, 256
        layer_types = ["sliding_attention"] * n
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=8,  # << base head_dim (128) — malformed / non-authoritative
            layer_types=layer_types,
            sliding_window=window,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd, dtype_bytes=db)
        # Whole-window baseline at the clamped base per-layer rate, not the tiny
        # local dims (which would under-count).
        assert est.fixed_baseline_bytes == n * window * 2 * kv * hd * db
        assert est.per_token_growth_bytes == 0

    def test_projection_covers_recurrent_fixed_state(self):
        # A GatedDeltaNet hybrid: the projection must reserve the recurrent
        # layers' fixed state (in the baseline) on top of the full layers'
        # growth — it is never omitted (codex round 2 BLOCKING #1 + #3).
        n, kv, hd, db = 16, 2, 128, 2
        gdn = dict(
            linear_num_value_heads=32,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        layer_types = (["linear_attention"] * 3 + ["full_attention"]) * (n // 4)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            **gdn,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd, dtype_bytes=db)
        n_full = n // 4
        n_recurrent = n - n_full
        # Per-layer GatedDeltaNet state (fp32/element), independently recomputed.
        key_dim = 16 * 128
        value_dim = 32 * 128
        state = 4 * (32 * 128 * 128 + (4 - 1) * (2 * key_dim + value_dim))
        assert est.fixed_baseline_bytes == n_recurrent * state
        full_per_layer = 2 * kv * hd * db
        for tokens in (1, 100, 4096, 32768):
            projected = est.fixed_baseline_bytes + tokens * est.per_token_growth_bytes
            # >= full-attention growth + the reserved recurrent fixed state.
            assert projected >= n_full * tokens * full_per_layer + n_recurrent * state
