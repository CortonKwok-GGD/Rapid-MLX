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


def _raw(a) -> tuple:
    """``(dtype, shape, raw-bytes)`` of an MLX array via a contiguous NumPy
    copy. Compares the underlying BIT pattern — unlike ``mx.array_equal``,
    this distinguishes ``+0.0`` from ``-0.0`` and any other byte-level
    difference, which is exactly what the "byte-exact" claim requires."""
    n = np.ascontiguousarray(np.array(a))
    return (n.dtype.str, n.shape, n.tobytes())


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

    Two MLX-specific facts make the bound what it is (both verified
    empirically, not assumed):

    * MLX affine quantization is ZERO-POINT-INCLUSIVE: the grid it lays
      down always spans zero, so the effective range is
      ``[min(gmin, 0), max(gmax, 0)]`` — NOT ``[gmin, gmax]``. A group of
      all-negative (or all-positive) values at a nonzero offset therefore
      gets a step set by its *distance from zero*, and every value can
      land a full such step away. Using ``gmax - gmin`` here would be
      wrong (it under-bounds these offset groups) — the ``constant`` /
      ``narrow`` strategy distributions exercise exactly that case.
    * The quantizer is *not* round-to-nearest, so the tight bound is a
      full step, not step/2 (observed worst case ~0.98 step).

    ``step`` below is the zero-inclusive step; the only non-data term is
    ``rel`` — a float32 recombination slack applied *relative to the
    group magnitude*, never a fixed absolute number. For a constant group
    the step is 0 and the reconstruction is exact.
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
    # Zero-inclusive effective range — the range MLX actually quantizes
    # over (see docstring). Never divides by step, so step == 0 (constant
    # group) is safe and demands exact reconstruction.
    eff_step = (np.maximum(gmax, 0.0) - np.minimum(gmin, 0.0)) / (2**bits - 1)
    err = np.abs(dg - xg).max(axis=-1)  # per-group worst reconstruction err

    rel = 1e-4  # float32 recombination slack (data-relative, below)
    magnitude = np.maximum(np.abs(gmin), np.abs(gmax))
    tol = eff_step * (1.0 + rel) + rel * magnitude

    worst = float(np.max(err - tol))
    assert np.all(err <= tol), (
        f"round-trip error exceeded per-group step bound by {worst:.3e} "
        f"(gs={gs}, bits={bits}); max err/step="
        f"{float(np.max(err / np.maximum(eff_step, 1e-30))):.4f}"
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
    second round-trip ``z`` is a genuine fixed point of quant->dequant.

    The "byte-exact" wording is PATH (a) — verified empirically to hold
    across the generated space (constant + narrow-range groups included):
    at the fixed point ``z`` BOTH

      * the dequantized tensor (``w == z`` at the raw-byte level — signed
        zero and all, not just ``mx.array_equal`` which folds ``+0.0`` and
        ``-0.0``), AND
      * the full quantized STATE — the ``[packed, scales, biases]`` triple
        (dtype, shape, and raw bytes)

    reproduce byte-for-byte. This guards against unbounded drift under
    repeated cache save/restore cycles (``state`` serializes the triple).
    """
    x, gs, bits = t
    y = _dequantize(_quantize(x, gs, bits), gs, bits)
    z = _dequantize(_quantize(y, gs, bits), gs, bits)

    # (a) the first drift is bounded by one of y's own quantization steps
    #     — re-quantization never amplifies error beyond a single step.
    #     Same zero-inclusive step as test_roundtrip_error_bounded_by_group_step.
    yn = np.array(y, dtype=np.float32)
    zn = np.array(z, dtype=np.float32)
    rows, head_dim = yn.shape
    n_groups = head_dim // gs
    yg = yn.reshape(rows, n_groups, gs)
    zg = zn.reshape(rows, n_groups, gs)
    y_min = yg.min(axis=-1)
    y_max = yg.max(axis=-1)
    eff_step_y = (np.maximum(y_max, 0.0) - np.minimum(y_min, 0.0)) / (2**bits - 1)
    drift = np.abs(zg - yg).max(axis=-1)
    rel = 1e-4
    magnitude = np.maximum(np.abs(y_min), np.abs(y_max))
    assert np.all(drift <= eff_step_y * (1.0 + rel) + rel * magnitude)

    # (b) z is a byte-exact fixed point. Quantize z (its stored state),
    #     read it back to w, then quantize w — both the dequantized tensor
    #     and the quantized triple must reproduce byte-for-byte.
    q_z = _quantize(z, gs, bits)  # quantized STATE of z
    w = _dequantize(q_z, gs, bits)
    q_w = _quantize(w, gs, bits)  # quantized STATE of w

    # (b1) dequantized tensor is byte-identical (catches signed zero).
    assert _raw(w) == _raw(z), (
        f"dequant fixed point not byte-exact (gs={gs}, bits={bits}): "
        f"max|w - z|={float(np.max(np.abs(np.array(w) - zn))):.3e}"
    )
    # (b2) quantized state (packed/scales/biases) is byte-identical.
    for name, mz, mw in zip(("packed", "scales", "biases"), q_z, q_w):
        assert _raw(mz) == _raw(mw), (
            f"quantized {name} not a byte-exact fixed point (gs={gs}, bits={bits})"
        )
