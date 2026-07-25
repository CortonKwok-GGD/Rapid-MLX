# SPDX-License-Identifier: Apache-2.0
"""Shared Hypothesis strategies for the hermetic property-based suite.

Every strategy here builds a *small in-memory value* — no model load, no
booted server. That is a hard constraint, not an accident: property
fuzzing multiplies the work by ``max_examples``, so it can only stay fast
enough to run on every commit if each example is cheap. Model-requiring
metamorphic properties (streaming == non-streaming, temp=0 determinism)
are deliberately out of scope and belong in the integration suite.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
from hypothesis import strategies as st

# The live quantized KV cache only ever uses these (group_size, bits)
# pairs — see ``vllm_mlx/quantized_batch_cache.py``. ``mx.quantize``
# requires the quantized (last) dim to be an exact multiple of group_size.
QUANT_BITS: tuple[int, ...] = (4, 8)
QUANT_GROUP_SIZES: tuple[int, ...] = (32, 64, 128)

# Per-tensor magnitude scales so the round-trip invariants are exercised
# across both tiny (~1e-2) and large (~5e1) activations. The affine
# quantization step scales with the data range, so any correct error
# bound has to be magnitude-invariant — hence the spread.
_MAGNITUDE_SCALES: tuple[float, ...] = (1e-2, 1.0, 7.0, 5e1)


@st.composite
def mlx_kv_tensors(draw, *, max_rows: int = 6, max_groups: int = 4):
    """Draw ``(x, group_size, bits)`` for the KV round-trip invariants.

    ``x`` is a finite ``mx.array`` of shape ``(rows, head_dim)`` where
    ``head_dim = group_size * n_groups`` — the exact-divisibility
    ``mx.quantize`` needs along its last (head) axis. Values span
    negative + positive and small + large magnitudes; NaN/inf are out of
    scope (they never reach the KV cache — attention/SDPA would already
    have produced NaN upstream). ``bits`` covers {4, 8} and
    ``group_size`` covers {32, 64, 128}, always with a divisible head_dim.

    Values are generated from a Hypothesis-drawn seed via NumPy rather
    than element-by-element: shrinking a 3072-float list adds no signal
    for these whole-tensor numeric invariants, and the seed keeps every
    failing example perfectly reproducible while staying fast.
    """
    bits = draw(st.sampled_from(QUANT_BITS))
    group_size = draw(st.sampled_from(QUANT_GROUP_SIZES))
    n_groups = draw(st.integers(min_value=1, max_value=max_groups))
    rows = draw(st.integers(min_value=1, max_value=max_rows))
    scale = draw(st.sampled_from(_MAGNITUDE_SCALES))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    dist = draw(st.sampled_from(("normal", "uniform", "bimodal")))

    head_dim = group_size * n_groups
    rng = np.random.default_rng(seed)
    if dist == "normal":
        base = rng.standard_normal((rows, head_dim))
    elif dist == "uniform":
        base = rng.uniform(-1.0, 1.0, size=(rows, head_dim))
    else:  # bimodal — pushes mass toward the group extrema, the worst
        # case for affine min/max quantization.
        base = rng.choice((-1.0, 1.0), size=(rows, head_dim))
        base = base + 0.05 * rng.standard_normal((rows, head_dim))
    x = mx.array((base * scale).astype(np.float32))
    return x, group_size, bits


# ---- sampling-parameter float strategies -------------------------------


def nonfinite_floats() -> st.SearchStrategy:
    """NaN, +inf, -inf — the non-finite forms every sampling-param
    validator must reject (the H-10 fix)."""
    return st.sampled_from((float("nan"), float("inf"), float("-inf")))


def in_range_floats(
    lo: float,
    hi: float,
    *,
    lo_inclusive: bool = True,
    hi_inclusive: bool = True,
) -> st.SearchStrategy:
    """Finite floats inside ``[lo, hi]``, honoring per-bound inclusivity."""
    return st.floats(
        min_value=lo,
        max_value=hi,
        exclude_min=not lo_inclusive,
        exclude_max=not hi_inclusive,
        allow_nan=False,
        allow_infinity=False,
    )


def out_of_range_finite_floats(
    lo: float,
    hi: float,
    *,
    span: float = 1e6,
) -> st.SearchStrategy:
    """Finite floats strictly below ``lo`` or strictly above ``hi``.

    Hypothesis realizes ``exclude_min`` / ``exclude_max`` with
    ``math.nextafter``, so every drawn value is a representable float
    *distinct* from the bound — no float-rounding value can silently land
    ON the (accepted) boundary and turn a rejection property flaky.
    """
    below = st.floats(
        min_value=lo - span,
        max_value=lo,
        exclude_max=True,
        allow_nan=False,
        allow_infinity=False,
    )
    above = st.floats(
        min_value=hi,
        max_value=hi + span,
        exclude_min=True,
        allow_nan=False,
        allow_infinity=False,
    )
    return st.one_of(below, above)
