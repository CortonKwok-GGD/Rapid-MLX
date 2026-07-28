# SPDX-License-Identifier: Apache-2.0
"""Architecture-aware per-token KV-cache footprint estimation.

The D-METAL-CAP admission gate (``scheduler._enforce_metal_cap_at_admission``)
projects how much KV-cache a new request would allocate and rejects it before
the allocation pushes Metal active past the operator cap. That projection is
driven by a *per-token* byte figure. The historical figure was a UNIFORM
formula::

    per_token = 2 (K+V) * num_hidden_layers * num_key_value_heads
                  * head_dim * dtype_bytes

which treats EVERY layer as a full-growth attention layer. That is wrong for
the hybrid architectures this engine ships:

* **Sliding-window layers** (GPT-OSS ~50% sliding; Gemma-4 local layers) hold a
  fixed rotating window of KV. Their footprint is bounded by the window size —
  it does NOT grow per generated token.
* **KV-sharing / borrower layers** (Gemma-4 / Gemma-3n ``num_kv_shared_layers``)
  compute no K/V of their own; they reuse an earlier producer layer's cache and
  allocate ZERO KV.
* **Recurrent / linear-attention layers** (Qwen3.5/3.6 GatedDeltaNet, Mamba)
  carry a fixed-size recurrent state. Zero per-token growth.

Counting all of those as full-growth over-estimates the per-token figure by up
to ~2x for Gemma-4 / GPT-OSS / Qwen3.5, which makes the admission gate reject
requests that would actually fit — a spurious rejection on memory-constrained
Macs.

This module computes an accurate footprint by classifying each layer:

``per_token_growth_bytes``
    ``2 * dtype_bytes * Σ (kv_heads_L * head_dim_L)`` over ONLY the
    full-attention (unbounded per-token-growing) layers, honoring per-layer /
    global dims when the architecture is hybrid.

``sliding_per_token_bytes`` / ``sliding_window_tokens``
    ``2 * dtype_bytes * Σ_sliding (kv_heads_L * head_dim_L)`` — the per-token
    KV rate of the sliding-window layers — together with the rotating window
    length. The caller charges ``sliding_per_token * min(tokens, window)``: a
    rotating cache holds at most ``window`` positions, but a request shorter
    than the window holds only ``tokens``, so short requests stay byte-identical
    to the historical uniform figure and only long requests get the window cap
    (codex round 6 BLOCKING). Flattening the whole window into a fixed baseline
    would over-charge — and spuriously reject — short requests.

``fixed_baseline_bytes``
    The conservative fixed recurrent state of each recurrent layer we can size
    from its config. Allocated once per sequence, independent of token count
    (an SSM/conv state is materialized in full at cache init). Sliding windows
    are NOT flattened here — see above.

Borrower (KV-sharing) layers contribute 0 to all terms. Recurrent /
linear-attention layers contribute 0 to per-token growth; their fixed state is
reserved in the baseline instead (see safety point 2).

Safety contract (this feeds a codex-hardened, over-estimate-safe admission
path — see ``scheduler._resolve_kv_bytes_per_token``):

1. **Over-estimate-safe fallback.** A recognized hybrid must expose a per-layer
   ``layer_types`` list whose length matches ``num_hidden_layers``. When that is
   absent, or anything is ambiguous, the estimator returns the caller's uniform
   figure unchanged (``per_token_growth_bytes == uniform_per_token_bytes`` and
   ``fixed_baseline_bytes == 0``) so dense models (Llama/Qwen dense) and
   unknown/stub configs stay BYTE-IDENTICAL to the historical behavior. Only a
   recognized hybrid gets the smaller accurate number. ``num_kv_shared_layers``
   is NOT sufficient on its own — the per-layer map must confirm the borrower
   split (see point 3).
2. **Never under-count the counted layers, never over-count short requests.**
   The load-bearing guarantee for the D-METAL-CAP cliff is that the projected
   footprint is >= the true footprint of every counted layer. Full-attention
   layers are charged exactly per token; a sliding layer with no readable window
   is charged as full unbounded growth; an unrecognized layer-type string is
   charged as full growth (exact-match allowlists, no substring guessing).
   Sliding-window layers with a readable window report a per-token ``rate`` and
   the window; the caller charges ``rate * min(tokens, window)`` for decode (an
   upper bound on the rotating cache, equal to the historical uniform figure for
   any request shorter than the window) floored at the full prompt to cover the
   transient prefill peak — so the reduction never comes at the cost of
   over-charging a short request nor under-charging a large prefill. Recurrent
   layers genuinely have zero per-token growth; a GatedDeltaNet layer's fixed
   state is sized from its real config fields (linear head dims + conv kernel)
   and charged at fp32/element to stay conservative. Every other recurrent family
   (Mamba/Mamba2, RWKV, generic) is NOT sized — its exact cache layout is not
   modelled here — and falls back to full per-token growth rather than a possibly
   wrong fixed state, so it is never under-reserved. The fixed baseline is
   charged once per sequence and never scales with generated tokens.
3. **KV-sharing only zeroes verified borrowers.** ``num_kv_shared_layers``
   declares that the LAST N layers borrow (Gemma-4 / Gemma-3n contract). We zero
   those borrowers only when the per-layer ``layer_types`` map confirms it —
   every borrower attention type has a same-type producer below the split
   (mirroring ``gemma4_text._check_kv_share_config``). If the map does not
   confirm the last-N contract, no layer is zeroed (over-count, never
   under-count).

The estimator is a pure function: it reads only plain fields off a config
object (or its ``text_config``), performs no I/O, loads no weights, and is unit
testable without a live model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Per-layer attention-type classification uses EXACT (case-folded) matches, not
# substrings: a substring rule would misclassify an unfamiliar full-attention
# name that merely contains ``"linear"`` / ``"gated"`` / ``"local"`` as
# zero-growth or window-bounded and silently UNDER-count it (codex round 1
# BLOCKING #4). Any value not on these allowlists falls through to full-growth —
# the conservative, never-under-count default. Values are the ``layer_types``
# strings the engine actually ships (Gemma-4 / GPT-OSS ``sliding_attention`` +
# ``full_attention``; Qwen3.5/3.6 GatedDeltaNet ``linear_attention``; Mamba
# stacks) plus their common upstream aliases.
# GatedDeltaNet (Qwen3-Next / Qwen3.5 / Qwen3.6) ships its linear-attention
# layers under the canonical ``linear_attention`` type plus these upstream
# aliases. ALL of them must route to ``_gateddeltanet_state_bytes`` — a name
# that classifies as recurrent (zero per-token growth) but whose state we cannot
# size would fall back to full-growth and lose the reduction this module exists
# to deliver (codex round 5 BLOCKING). Keeping the sizer's dispatch keyed on the
# same set the classifier uses prevents that drift.
_GATEDDELTANET_TYPES: frozenset[str] = frozenset(
    {
        "linear_attention",
        "gated_delta_net",
        "gated_deltanet",
    }
)
_RECURRENT_TYPES: frozenset[str] = _GATEDDELTANET_TYPES | frozenset(
    {
        "mamba",
        "mamba2",
        "recurrent",
        "rwkv",
    }
)
_SLIDING_TYPES: frozenset[str] = frozenset(
    {
        "sliding_attention",
        "sliding_window_attention",
        "local_attention",
        "local_sliding_attention",
        "swa",
    }
)


@dataclass(frozen=True)
class KVFootprintEstimate:
    """Architecture-aware KV footprint of a single sequence.

    The D-METAL-CAP projection reconstructs a request's peak KV footprint as::

        fixed_baseline_bytes
          + per_token_growth_bytes * (prompt_tokens + max_tokens)
          + sliding_per_token_bytes * min(prompt_tokens + max_tokens,
                                          sliding_window_tokens)

    Splitting the sliding-window term out (rather than flattening the whole
    window into ``fixed_baseline``) keeps SHORT requests byte-identical to the
    historical uniform estimate: a rotating-window layer only holds
    ``min(tokens, window)`` positions, so a request whose token budget is below
    the window is charged exactly ``tokens`` worth — never the full window
    (codex round 6 BLOCKING). Long requests still get the reduction: the sliding
    term is capped at ``window`` while the old uniform figure kept growing.

    Attributes:
        per_token_growth_bytes: Bytes of KV cache allocated per additional
            token, summed over the full-attention (unbounded-growth) layers
            only.
        fixed_baseline_bytes: Bytes allocated once per sequence regardless of
            token count — the conservative fixed recurrent state of each
            sizeable recurrent layer. (Sliding windows are NOT flattened here;
            see ``sliding_per_token_bytes``.) Zero for a dense model and for a
            hybrid with no sizeable recurrent layers.
        sliding_per_token_bytes: Bytes of KV allocated per token by the
            sliding-window (local-attention) layers, summed over those layers.
            Capped in the projection at ``sliding_window_tokens`` positions.
            Zero when there are no readable-window sliding layers.
        sliding_window_tokens: The rotating window length (tokens) that caps the
            sliding term. Zero when ``sliding_per_token_bytes`` is zero.
    """

    per_token_growth_bytes: int
    fixed_baseline_bytes: int
    sliding_per_token_bytes: int = 0
    sliding_window_tokens: int = 0


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` off a config that may be a dict or an attribute object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _pos_int(value: Any) -> int:
    """Return ``value`` when it is a strictly-positive, non-bool int, else 0.

    Mirrors the scheduler's ``isinstance(..., int)`` MagicMock guard: a
    ``MagicMock`` attribute is not an ``int`` (and ``bool`` is excluded because
    it is an ``int`` subclass that never denotes a real dimension).
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value > 0:
        return value
    return 0


def _valid_layer_types(value: Any, expected_len: int) -> list[str] | None:
    """Return ``value`` as a list of type strings iff it is well-formed.

    Well-formed = a list/tuple of exactly ``expected_len`` string entries. Any
    other shape (missing, MagicMock, wrong length, non-string entries) returns
    ``None`` so the caller falls back to the uniform formula rather than
    guessing a per-layer layout it cannot trust.
    """
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) != expected_len or expected_len <= 0:
        return None
    if not all(isinstance(entry, str) for entry in value):
        return None
    return [entry.lower() for entry in value]


def _pick_structural_config(model_config: Any, base_num_layers: int) -> Any | None:
    """Select the config object that carries the hybrid layer structure.

    Layer-structure fields live either directly on ``model_config`` (text-lane
    configs) or nested under ``model_config.text_config`` (multimodal configs).
    Returns the config whose ``layer_types`` list matches ``base_num_layers``,
    or ``None`` when neither carries one (the byte-identical dense/unknown
    fallback).

    A per-layer ``layer_types`` list is REQUIRED to engage the hybrid path.
    ``num_kv_shared_layers`` on its own is not enough: without the per-layer map
    we cannot verify which layers actually borrow (codex round 1 BLOCKING #3),
    so a config exposing only a share count stays on the uniform estimate.
    """
    candidates = [model_config]
    text_config = _cfg_get(model_config, "text_config")
    if text_config is not None:
        candidates.append(text_config)

    for cfg in candidates:
        if _valid_layer_types(_cfg_get(cfg, "layer_types"), base_num_layers):
            return cfg
    return None


def _classify_layer(layer_type: str) -> str:
    """Map a case-folded ``layer_types`` entry to a footprint class.

    Returns one of ``"recurrent"``, ``"sliding"``, ``"full"``. Matching is
    EXACT against the zero-growth / window-bounded allowlists; every other
    value — a known full-attention type or an unrecognized string — resolves to
    ``"full"``, the conservative (over-counting) default that can never
    under-count.
    """
    if layer_type in _RECURRENT_TYPES:
        return "recurrent"
    if layer_type in _SLIDING_TYPES:
        return "sliding"
    return "full"


# mlx-lm materializes the recurrent conv/SSM state buffers with ``mx.zeros``,
# which defaults to float32 — so we size the recurrent per-sequence state at 4
# bytes/element regardless of the (possibly smaller) KV dtype. Over-charging a
# small fixed term is the safe direction and guarantees we never under-reserve
# it (codex round 2 BLOCKING #1).
_RECURRENT_STATE_BYTES_PER_ELEM = 4


def _gateddeltanet_state_bytes(struct: Any) -> int | None:
    """GatedDeltaNet (Qwen3-Next / Qwen3.5 / Qwen3.6) per-sequence state bytes.

    State = a per-value-head ``head_k_dim × head_v_dim`` matrix plus the
    causal-conv ring buffer over ``conv_dim = 2·key_dim + value_dim`` channels.
    Shapes match ``mlx_lm.models.qwen3_next.Qwen3NextGatedDeltaNet``. ALL of the
    state + conv dimensions must be present and positive — including the conv
    kernel — else return ``None`` so the caller charges full growth rather than
    silently omitting the conv buffer and under-reserving (codex round 3
    BLOCKING #1).
    """
    num_v = _pos_int(_cfg_get(struct, "linear_num_value_heads"))
    num_k = _pos_int(_cfg_get(struct, "linear_num_key_heads"))
    head_k = _pos_int(_cfg_get(struct, "linear_key_head_dim"))
    head_v = _pos_int(_cfg_get(struct, "linear_value_head_dim"))
    conv_kernel = _pos_int(_cfg_get(struct, "linear_conv_kernel_dim"))
    if not (num_v and num_k and head_k and head_v and conv_kernel):
        return None
    key_dim = num_k * head_k
    value_dim = num_v * head_v
    recurrent_elems = num_v * head_k * head_v
    conv_dim = 2 * key_dim + value_dim
    conv_elems = (conv_kernel - 1) * conv_dim
    return _RECURRENT_STATE_BYTES_PER_ELEM * (recurrent_elems + conv_elems)


def _recurrent_state_bytes(struct: Any, layer_type: str) -> int | None:
    """Conservative per-sequence FIXED state of one recurrent layer, in bytes.

    Sizes the state only for the recurrent family whose EXACT state + conv
    layout we have validated against the shipped implementation — GatedDeltaNet
    (Qwen3-Next / Qwen3.5 / Qwen3.6), keyed on its family-specific ``linear_*``
    fields. Every GatedDeltaNet alias the classifier treats as recurrent
    (``_GATEDDELTANET_TYPES``: ``linear_attention`` + the ``gated_delta_net`` /
    ``gated_deltanet`` upstream spellings) routes to the same sizer, so a config
    that names its linear layers with an alias still gets the reduction rather
    than silently reverting to full per-token growth (codex round 5 BLOCKING).

    Every OTHER recurrent family — Mamba / Mamba2, RWKV, generic ``recurrent`` —
    returns ``None`` so the caller charges the layer full per-token growth (the
    safe, never-under-count fallback). We deliberately do NOT attempt to size a
    Mamba state: the conv buffer channel count differs between Mamba1
    (``d_inner``) and Mamba2 (``d_inner`` + grouped ``B``/``C`` channels =
    ``d_inner + 2·n_groups·d_state``), so a single generic formula would
    UNDER-reserve Mamba2 (codex round 7 BLOCKING) — and this engine ships no
    Mamba model, so over-counting those layers as full growth costs nothing while
    keeping the OOM-safety guarantee intact. When a real Mamba family is shipped,
    add a family-specific sizer here mirroring its exact cache shapes.

    The returned term is a per-sequence constant (it does not grow with
    generated tokens), so it lands in the fixed baseline, never in per-token
    growth.
    """
    if layer_type in _GATEDDELTANET_TYPES:
        return _gateddeltanet_state_bytes(struct)
    # Mamba/Mamba2/RWKV/other recurrent: state layout not modelled → full growth.
    return None


def estimate_kv_footprint(
    model_config: Any,
    *,
    dtype_bytes: int,
    uniform_per_token_bytes: int,
    base_num_layers: int,
    base_kv_heads: int,
    base_head_dim: int,
) -> KVFootprintEstimate:
    """Compute the architecture-aware KV footprint for one sequence.

    Args:
        model_config: The live model config (attribute object or dict). Only
            plain fields are read; no I/O, no weight load.
        dtype_bytes: KV element size in bytes, as already resolved by the
            caller (``scheduler._infer_kv_dtype_bytes``).
        uniform_per_token_bytes: The caller's historical uniform per-token
            figure. Returned unchanged when the config is not a recognized
            hybrid, guaranteeing byte-identical behavior for dense/unknown
            models.
        base_num_layers: ``num_hidden_layers`` as read by the caller. Used to
            validate the ``layer_types`` length and to bound the borrower split.
        base_kv_heads: ``num_key_value_heads`` (or attention heads) as read by
            the caller — the local/sliding-layer KV-head count.
        base_head_dim: ``head_dim`` as read by the caller — the local/sliding
            head dimension.

    Returns:
        A :class:`KVFootprintEstimate`. For a non-hybrid config this is exactly
        ``(uniform_per_token_bytes, 0)``.
    """
    uniform = KVFootprintEstimate(
        per_token_growth_bytes=uniform_per_token_bytes,
        fixed_baseline_bytes=0,
    )

    # Guard the primitives. If the caller could not read a positive uniform
    # figure (MagicMock model / missing config), stay at the byte-identical
    # fallback — the scheduler never calls us in that case, but be defensive.
    if (
        _pos_int(dtype_bytes) == 0
        or _pos_int(base_num_layers) == 0
        or _pos_int(base_kv_heads) == 0
        or _pos_int(base_head_dim) == 0
        or uniform_per_token_bytes <= 0
    ):
        return uniform

    struct = _pick_structural_config(model_config, base_num_layers)
    if struct is None:
        # Not a recognized hybrid — dense (Llama/Qwen dense) or an
        # unknown/stub config. Byte-identical to the uniform formula.
        return uniform

    # Local (sliding / default) dims — prefer the structural config, fall back
    # to the caller's base values so a config that only splits full vs sliding
    # via ``layer_types`` (no distinct local fields) still works.
    local_kv_heads = (
        _pos_int(_cfg_get(struct, "num_key_value_heads"))
        or _pos_int(_cfg_get(struct, "num_attention_heads"))
        or base_kv_heads
    )
    local_head_dim = _pos_int(_cfg_get(struct, "head_dim")) or base_head_dim

    # Global (full-attention) dims. Gemma-4 full layers use a wider
    # ``global_head_dim`` / ``num_global_key_value_heads`` than the local
    # sliding layers; when absent (GPT-OSS and most hybrids) they fall back to
    # the local dims. Using the (larger-or-equal) global dim for full layers is
    # both accurate and never-under-count.
    global_kv_heads = (
        _pos_int(_cfg_get(struct, "num_global_key_value_heads")) or local_kv_heads
    )
    global_head_dim = _pos_int(_cfg_get(struct, "global_head_dim")) or local_head_dim

    # Sliding window size. When a layer is sliding but the window is unreadable
    # we cannot bound its footprint, so it is charged as full-growth below.
    window = _pos_int(_cfg_get(struct, "sliding_window"))

    # ``layer_types`` is guaranteed present + length-checked by
    # ``_pick_structural_config``; the ``is None`` branch is unreachable but
    # kept as a defensive fallback (avoids an assert that ``python -O`` strips).
    layer_types = _valid_layer_types(_cfg_get(struct, "layer_types"), base_num_layers)
    if layer_types is None:
        return uniform

    # Borrower split: Gemma-4 / Gemma-3n reuse the LAST ``num_kv_shared_layers``
    # decoder layers' K/V from an earlier same-type producer (split at
    # ``num_hidden_layers - num_kv_shared_layers`` — see
    # ``models/gemma4_text._check_kv_share_config``). We only zero those
    # borrowers when the ACTUAL per-layer map confirms the last-N contract:
    # every borrower attention type must have a producer of the same type below
    # the split. If it does not (interleaved / differently-anchored sharing we
    # do not model), we keep every layer as a producer — an over-count, never an
    # under-count (codex round 1 BLOCKING #3).
    #
    # For FOOTPRINT purposes the set-of-types check is sufficient (and mirrors
    # the shipped ``_check_kv_share_config`` orphan check exactly): a borrower
    # whose attention type has ANY same-type producer below the split allocates
    # zero KV — it reuses that producer's cache. WHICH producer index it borrows
    # from does not change the borrower's own footprint (still zero), so a
    # per-index source mapping would not alter this estimate.
    num_shared = _cfg_get(struct, "num_kv_shared_layers")
    if isinstance(num_shared, int) and not isinstance(num_shared, bool):
        num_shared = num_shared if 0 < num_shared < base_num_layers else 0
    else:
        num_shared = 0
    if num_shared > 0:
        split = base_num_layers - num_shared
        producer_types = set(layer_types[:split])
        borrower_types = set(layer_types[split:])
        if not borrower_types <= producer_types:
            num_shared = 0  # last-N contract unverified → do not zero anything
    first_borrower = base_num_layers - num_shared

    # The uniform per-layer charge (base dims). It is the never-under-count
    # floor: no per-token-growing layer is ever charged LESS than this.
    uniform_per_layer = 2 * dtype_bytes * base_kv_heads * base_head_dim
    # Full-attention layers use the (wider) global dims when present, but never
    # below the uniform base-layer size — a config whose ``global_head_dim`` /
    # ``num_global_key_value_heads`` are smaller than (or unrelated to) the base
    # dims must not shrink a full layer's per-token growth below uniform and
    # open an under-count (codex round 4 BLOCKING #1).
    full_per_token = max(
        2 * dtype_bytes * global_kv_heads * global_head_dim, uniform_per_layer
    )
    # Per-token KV rate of ONE sliding-window layer (local dims). The projection
    # multiplies this by ``min(tokens, window)`` — the true peak occupancy of a
    # rotating cache — rather than flattening the whole window into the baseline,
    # so a request shorter than the window is charged only for the tokens it
    # actually holds (codex round 6 BLOCKING). Clamped to the uniform base-layer
    # floor for the same reason full layers are (codex round 7 BLOCKING #2): if a
    # malformed or semantically-different nested config exposes smaller local
    # dims than the base the scheduler read, the sliding term must not drop below
    # the uniform per-layer charge and open an under-count.
    sliding_per_layer = max(
        2 * dtype_bytes * local_kv_heads * local_head_dim, uniform_per_layer
    )

    per_token_growth = 0
    fixed_baseline = 0
    sliding_per_token = 0
    for idx in range(base_num_layers):
        if idx >= first_borrower:
            # Borrower (KV-sharing) layer — reuses a producer's cache, allocates
            # nothing.
            continue
        layer_type = layer_types[idx]
        layer_class = _classify_layer(layer_type)
        if layer_class == "recurrent":
            # Recurrent / linear-attention layers keep a FIXED state that does
            # not grow per token. When we can size that state from the family's
            # real config fields we reserve it in the per-sequence baseline (so
            # concurrent sequences don't silently allocate past the cap — codex
            # round 2 BLOCKING #1) while keeping per-token growth at zero — the
            # load-bearing correctness win for Qwen3.5/3.6/Mamba. When we CANNOT
            # size it, we fall back to charging the layer the uniform per-token
            # estimate (full growth), which never under-counts.
            state = _recurrent_state_bytes(struct, layer_type)
            if state is not None:
                fixed_baseline += state
            else:
                per_token_growth += uniform_per_layer
            continue
        if layer_class == "sliding":
            if window > 0:
                # Window-capped per-token growth — accumulated separately so the
                # projection can cap it at ``window`` positions.
                sliding_per_token += sliding_per_layer
            else:
                # No readable window → cannot bound it → charge the uniform
                # per-layer growth (never under-count).
                per_token_growth += uniform_per_layer
        else:  # "full"
            per_token_growth += full_per_token

    return KVFootprintEstimate(
        per_token_growth_bytes=int(per_token_growth),
        fixed_baseline_bytes=int(fixed_baseline),
        sliding_per_token_bytes=int(sliding_per_token),
        sliding_window_tokens=int(window) if sliding_per_token > 0 else 0,
    )
