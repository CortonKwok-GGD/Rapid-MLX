# SPDX-License-Identifier: Apache-2.0
"""Multi-slot GatedDeltaNet rollback for chain-of-K MTP.

Locks the ``cache_patch`` contract that the verify chunk-split records a
``(conv, ssm)`` snapshot at EVERY draft boundary (keyed by
``n_from_end = 1..K``), and that each snapshot equals the recurrent state
after processing exactly ``S - n_from_end`` tokens — so
``_rollback_draft(n)`` can restore to any accepted depth without leaving
rejected draft state in the cache.

Regression for the chain-of-K-on-hybrid work (PR feat/mtp-hybrid-chain-of-k):
before the multi-slot rewrite only a single boundary was snapshotted, so
partial accept at depth < K was unrepresentable.
"""

import mlx.core as mx
import pytest


def _build_gated_delta_net():
    from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs

    args = TextModelArgs(
        model_type="qwen3_5_moe_text",
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        vocab_size=100,
    )
    layer = GatedDeltaNet(args)
    # Inference mode: ``gated_delta_update(use_kernel=not self.training)`` — so
    # eval() exercises the SAME Metal-kernel segmented path production uses
    # (training mode would silently fall to the pure-ops reference instead).
    layer.eval()
    mx.eval(layer.parameters())
    return layer, args


@pytest.mark.parametrize("K", [1, 2, 3])
def test_multislot_chunksplit_snapshots_every_boundary(K):  # noqa: N803  (K = draft width)
    """Every ``n_to_drop`` in 1..K has a snapshot equal to the true
    ``(S - n)``-token recurrent state."""
    mx.random.seed(0)
    from mlx_lm.models.cache import ArraysCache

    from vllm_mlx.spec_decode.mtp import cache_patch

    # Install the chunk-split patch; keep a handle to the unpatched forward
    # for reference (single-call, no snapshots).
    cache_patch.patch_gated_delta_net_for_mtp()
    orig_call = cache_patch._orig_gated_delta_call
    assert orig_call is not None

    layer, args = _build_gated_delta_net()

    # A FIXED prefix + verify window so every cache starts from the same state.
    prefix = mx.random.normal((1, 3, args.hidden_size))
    verify = mx.random.normal((1, K + 1, args.hidden_size))
    mx.eval(prefix, verify)

    def fresh_cache():
        c = ArraysCache(size=2)
        orig_call(layer, prefix, mask=None, cache=c)
        mx.eval(c[0], c[1])
        return c

    # Patched multi-slot run over the (K+1)-token verify window.
    c_pat = fresh_cache()
    c_pat.snapshot_offsets = list(range(1, K + 1))
    layer(verify, mask=None, cache=c_pat)
    mx.eval(c_pat[0], c_pat[1])

    assert c_pat.rollback_states is not None, "chunk-split must record snapshots"
    assert set(c_pat.rollback_states) == set(range(1, K + 1)), (
        "chunk-split must snapshot every 1..K draft boundary "
        f"(got {sorted(c_pat.rollback_states)})"
    )

    # Each snapshot at n_from_end == the state after processing the first
    # (S - n) verify tokens, computed independently via the original forward.
    # This compares two DIFFERENT computation paths (a mid-sequence boundary in
    # one long conv1d vs a fresh short forward), so a tight float tolerance —
    # not exact equality — is the correct bar; observed divergence is ~1e-7.
    S = K + 1
    for n in range(1, K + 1):
        c_ref = fresh_cache()
        orig_call(layer, verify[:, : S - n], mask=None, cache=c_ref)
        mx.eval(c_ref[0], c_ref[1])
        conv_snap, ssm_snap = c_pat.rollback_states[n]
        assert float(mx.max(mx.abs(conv_snap - c_ref[0]))) < 1e-5, (
            f"conv snapshot at n_from_end={n} diverges from the true "
            f"{S - n}-token state"
        )
        assert float(mx.max(mx.abs(ssm_snap - c_ref[1]))) < 1e-5, (
            f"ssm snapshot at n_from_end={n} diverges from the true {S - n}-token state"
        )


def test_patched_forward_output_matches_unsplit():
    """The chunk-split forward is byte-equal to the unsplit forward for the
    confirmed tokens (the split only exposes intermediate state; it must not
    change the output or the final cache state)."""
    mx.random.seed(1)
    from mlx_lm.models.cache import ArraysCache

    from vllm_mlx.spec_decode.mtp import cache_patch

    cache_patch.patch_gated_delta_net_for_mtp()
    orig_call = cache_patch._orig_gated_delta_call
    layer, args = _build_gated_delta_net()

    prefix = mx.random.normal((1, 3, args.hidden_size))
    verify = mx.random.normal((1, 3, args.hidden_size))  # K=2 verify window
    mx.eval(prefix, verify)

    def fresh_cache():
        c = ArraysCache(size=2)
        orig_call(layer, prefix, mask=None, cache=c)
        mx.eval(c[0], c[1])
        return c

    c_orig = fresh_cache()
    out_orig = orig_call(layer, verify, mask=None, cache=c_orig)
    mx.eval(out_orig, c_orig[0], c_orig[1])

    c_pat = fresh_cache()
    c_pat.snapshot_offsets = [1, 2]
    out_pat = layer(verify, mask=None, cache=c_pat)
    mx.eval(out_pat, c_pat[0], c_pat[1])

    # EXACT equality — the chunk-split is a pure sequential scan over the same
    # tokens in the same order, so per-token logits and the final (conv, ssm)
    # state are bit-identical to the unsplit forward. A tolerance here would let
    # a logit-changing regression slip through and quietly break the lossless
    # speculative-decoding contract.
    assert bool(mx.array_equal(out_orig, out_pat)), "chunk-split changed the output"
    assert bool(mx.array_equal(c_orig[0], c_pat[0])), "chunk-split changed conv state"
    assert bool(mx.array_equal(c_orig[1], c_pat[1])), "chunk-split changed ssm state"
