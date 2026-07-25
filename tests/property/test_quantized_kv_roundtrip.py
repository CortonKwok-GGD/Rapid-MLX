# SPDX-License-Identifier: Apache-2.0
"""Property-based invariants for the quantized live KV cache (#1208-tied).

The quantized continuous-batching KV cache
(``vllm_mlx/quantized_batch_cache.py``) is the exact code path behind the
``--kv-cache-dtype int8/int4`` flag. Bug #1208 was a *dimension-probe*
gap in that path — the class of failure where a group size is chosen that
does not actually divide the head dim, or a round-trip silently corrupts
the stored KV. Example tests pin one point each; these properties pin the
*invariant over the whole input space* — the only deterministic guard for
numeric round-trip behavior.

Three pure functions are under test:

* ``supported_group_size(head_dim, requested)`` — the divisor-selection
  logic a mis-probe (#1208) would trip.
* ``_quantize`` / ``_dequantize`` — the affine ``mx.quantize`` round-trip
  the cache uses on every read.

All tensors are tiny and in-memory: the suite is fully hermetic.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vllm_mlx.quantized_batch_cache import (
    _dequantize,
    _quantize,
    supported_group_size,
)

from .strategies import QUANT_GROUP_SIZES, mlx_kv_tensors

pytestmark = pytest.mark.property

# Extra examples for the pure-integer ``supported_group_size`` properties:
# they carry no MLX cost, so a wider sweep is essentially free and buys
# denser coverage of the divisibility lattice.
_INT_SWEEP = settings(max_examples=600)

# Head dims include the pathological non-divisible cases called out in the
# module docstring of the cache (80, 96, 100, 256, ...). The wide range
# guarantees Hypothesis hits both "no supported size divides" and
# "several do, must pick the largest".
_head_dims = st.integers(min_value=1, max_value=1024)
_requested = st.integers(min_value=1, max_value=512)


# ---------------------------------------------------------------------------
# supported_group_size — the divisor-selection invariant (#1208 root cause)
# ---------------------------------------------------------------------------


@given(head_dim=_head_dims, requested=_requested)
@_INT_SWEEP
def test_supported_group_size_is_the_largest_valid_divisor(head_dim, requested):
    """The result is ``None`` or the LARGEST of {32,64,128} that is both
    ``<= requested`` and divides ``head_dim``."""
    gs = supported_group_size(head_dim, requested)

    # A candidate is "valid" iff it is affordable (<= requested) AND it
    # actually divides the head dim (the #1208 correctness condition).
    valid = [s for s in QUANT_GROUP_SIZES if s <= requested and head_dim % s == 0]

    if gs is None:
        # None must mean there genuinely is no valid divisor — never a
        # false negative that would push a quantizable cache to bf16.
        assert valid == [], (
            f"supported_group_size({head_dim}, {requested}) returned None "
            f"but these sizes are valid: {valid}"
        )
    else:
        assert gs in QUANT_GROUP_SIZES
        assert gs <= requested
        assert head_dim % gs == 0
        # It is the maximum of the valid set — no larger valid divisor
        # exists (a larger one would be the sound choice #1208 needs).
        assert gs == max(valid)
        assert not any(s > gs for s in valid)


@given(head_dim=_head_dims, r1=_requested, r2=_requested)
@_INT_SWEEP
def test_supported_group_size_monotonic_in_requested(head_dim, r1, r2):
    """Raising ``requested`` never LOWERS the chosen group size.

    A larger budget can only enlarge the eligible-divisor set, and the
    function returns the max of that set, so the result is non-decreasing
    in ``requested``. ``None`` (quantization disabled) is treated as the
    bottom of the order.
    """
    lo, hi = sorted((r1, r2))
    g_lo = supported_group_size(head_dim, lo)
    g_hi = supported_group_size(head_dim, hi)
    # gs values are all truthy (32/64/128); only None maps to 0.
    assert (g_lo or 0) <= (g_hi or 0)


# ---------------------------------------------------------------------------
# _quantize / _dequantize — round-trip invariants
# ---------------------------------------------------------------------------


@given(t=mlx_kv_tensors())
def test_roundtrip_preserves_shape(t):
    """Dequantizing a quantized tensor yields the original shape — the
    cache stores 3 packed tensors but the model must read back exactly
    ``(rows, head_dim)``."""
    x, gs, bits = t
    dq = _dequantize(_quantize(x, gs, bits), gs, bits)
    assert dq.shape == x.shape


@given(t=mlx_kv_tensors())
def test_roundtrip_error_bounded_by_group_step(t):
    """Reconstruction error is bounded by each group's OWN quantization
    step — the differential, data-derived bound, not a magic epsilon.

    For every group of ``group_size`` consecutive elements along the head
    axis, affine quantization spaces the reconstruction levels by
    ``step = (max - min) / (2**bits - 1)``. Any value in the group
    therefore dequantizes to within one step of its true value. (MLX's
    affine quantizer is *not* round-to-nearest, so the tight bound is a
    full step, not step/2 — verified empirically: the observed worst case
    is ~0.99 step.) The tolerance's only non-data term is ``rel`` — a
    float32-rounding slack on the ``scale*q + bias`` recombination — and
    it is applied *relative to the group's magnitude*, never as a fixed
    absolute number.
    """
    x, gs, bits = t
    dq = _dequantize(_quantize(x, gs, bits), gs, bits)

    xn = np.array(x, dtype=np.float32)
    dn = np.array(dq, dtype=np.float32)
    rows, head_dim = xn.shape
    n_groups = head_dim // gs
    xg = xn.reshape(rows, n_groups, gs)
    dg = dn.reshape(rows, n_groups, gs)

    gmin = xg.min(axis=-1)
    gmax = xg.max(axis=-1)
    step = (gmax - gmin) / (2**bits - 1)  # per-group quantization step
    err = np.abs(dg - xg).max(axis=-1)  # per-group worst reconstruction err

    rel = 1e-4  # float32 recombination slack (data-relative, below)
    magnitude = np.maximum(np.abs(gmin), np.abs(gmax))
    tol = step * (1.0 + rel) + rel * magnitude

    worst = float(np.max(err - tol))
    assert np.all(err <= tol), (
        f"round-trip error exceeded per-group step bound by {worst:.3e} "
        f"(gs={gs}, bits={bits}); max err/step="
        f"{float(np.max(err / np.maximum(step, 1e-30))):.4f}"
    )


@given(t=mlx_kv_tensors())
def test_quantize_is_deterministic(t):
    """Quantizing the same tensor twice is byte-identical across all three
    stored tensors (packed / scales / biases) — the cache's ``state``
    serialization relies on this to round-trip losslessly."""
    x, gs, bits = t
    a = _quantize(x, gs, bits)
    b = _quantize(x, gs, bits)
    assert len(a) == len(b) == 3
    for name, ma, mb in zip(("packed", "scales", "biases"), a, b):
        assert bool(mx.array_equal(ma, mb).item()), f"{name} tensor not deterministic"


@given(t=mlx_kv_tensors())
def test_requantization_reaches_a_byte_exact_fixed_point(t):
    """Iterated quantization converges to a byte-exact fixed point.

    IMPORTANT — a naive ``dequantize(quantize(y)) == y`` would be WRONG:
    MLX affine quantization is NOT idempotent from an arbitrary
    grid-aligned point. Re-deriving ``(scale, bias)`` from ``y``'s own
    group extrema can shift the grid, so the first re-quantization
    ``|z - y|`` can be a *full* quantization step (empirically up to ~1
    step). The sound metamorphic invariant is CONVERGENCE: after the
    second round-trip the tensor is a genuine fixed point of
    quant->dequant, and a third round-trip reproduces it byte-for-byte.
    This guards against unbounded drift under repeated cache
    save/restore cycles.
    """
    x, gs, bits = t
    y = _dequantize(_quantize(x, gs, bits), gs, bits)
    z = _dequantize(_quantize(y, gs, bits), gs, bits)
    w = _dequantize(_quantize(z, gs, bits), gs, bits)

    # (a) the first drift is bounded by one of y's own quantization steps
    #     — re-quantization never amplifies error beyond a single step.
    yn = np.array(y, dtype=np.float32)
    zn = np.array(z, dtype=np.float32)
    rows, head_dim = yn.shape
    n_groups = head_dim // gs
    yg = yn.reshape(rows, n_groups, gs)
    zg = zn.reshape(rows, n_groups, gs)
    step_y = (yg.max(axis=-1) - yg.min(axis=-1)) / (2**bits - 1)
    drift = np.abs(zg - yg).max(axis=-1)
    rel = 1e-4
    magnitude = np.maximum(np.abs(yg.min(axis=-1)), np.abs(yg.max(axis=-1)))
    assert np.all(drift <= step_y * (1.0 + rel) + rel * magnitude)

    # (b) the second round-trip is a byte-exact fixed point: z is stable.
    assert bool(mx.array_equal(w, z).item()), (
        f"quant->dequant did not converge to a fixed point (gs={gs}, "
        f"bits={bits}): max|w - z|={float(np.max(np.abs(np.array(w) - zn))):.3e}"
    )
