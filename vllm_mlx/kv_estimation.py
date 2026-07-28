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
    full-attention (per-token-growing) layers, honoring per-layer / global dims
    when the architecture is hybrid.

``fixed_baseline_bytes``
    ``2 * dtype_bytes * Σ_sliding (window_L * kv_heads_L * head_dim_L)`` plus a
    conservative fixed term per recurrent layer. Allocated once per sequence,
    not per token.

Borrower (KV-sharing) layers contribute 0 to both.

Safety contract (this feeds a codex-hardened, over-estimate-safe admission
path — see ``scheduler._resolve_kv_bytes_per_token``):

1. **Over-estimate-safe fallback.** When the config lacks the layer-structure
   fields (no per-layer ``layer_types`` and no ``num_kv_shared_layers``), or
   anything is ambiguous, the estimator returns the caller's uniform figure
   unchanged (``per_token_growth_bytes == uniform_per_token_bytes`` and
   ``fixed_baseline_bytes == 0``) so dense models (Llama/Qwen dense) and
   unknown/stub configs stay BYTE-IDENTICAL to the historical behavior. Only a
   recognized hybrid — one that exposes a per-layer ``layer_types`` list of the
   right length or a valid ``num_kv_shared_layers`` — gets the smaller accurate
   number.
2. **Never under-count the counted layers.** ``fixed_baseline + tokens *
   per_token_growth`` is always >= the true architecture-aware footprint of the
   counted layers: full-attention layers are charged exactly; sliding layers
   are charged their whole window (an upper bound for any token count); a
   sliding layer with no readable window is charged as full-growth; an
   unrecognized layer type is charged as full-growth. So the estimate can only
   over-count, never under-count — it can never re-introduce the D-METAL-CAP
   OOM cliff.

The estimator is a pure function: it reads only plain fields off a config
object (or its ``text_config``), performs no I/O, loads no weights, and is unit
testable without a live model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Per-layer attention-type classification tokens. Ordered checks in
# ``_classify_layer`` matter: ``"linear_attention"`` and ``"sliding_attention"``
# both contain ``"attention"``, so recurrent and sliding are tested BEFORE the
# generic full-attention tokens.
_RECURRENT_TOKENS: tuple[str, ...] = (
    "linear",
    "mamba",
    "recurrent",
    "rwkv",
    "ssm",
    "gated",
    "delta",
)
_SLIDING_TOKENS: tuple[str, ...] = ("slid", "swa", "local", "window")
_FULL_TOKENS: tuple[str, ...] = ("full", "global", "attention", "sdpa")


@dataclass(frozen=True)
class KVFootprintEstimate:
    """Architecture-aware KV footprint of a single sequence.

    Attributes:
        per_token_growth_bytes: Bytes of KV cache allocated per additional
            token, summed over the full-attention (per-token-growing) layers
            only. This is the value the D-METAL-CAP projection multiplies by
            ``(prompt_tokens + max_tokens)``.
        fixed_baseline_bytes: Bytes allocated once per sequence regardless of
            token count — the sliding-window rotating buffers plus a
            conservative recurrent-state term. Zero for a dense model.
    """

    per_token_growth_bytes: int
    fixed_baseline_bytes: int


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
    or — absent that — the one exposing a valid ``num_kv_shared_layers`` in
    ``(0, base_num_layers)``. Returns ``None`` when neither carries a
    recognized hybrid signal (the byte-identical dense/unknown fallback).
    """
    candidates = [model_config]
    text_config = _cfg_get(model_config, "text_config")
    if text_config is not None:
        candidates.append(text_config)

    for cfg in candidates:
        if _valid_layer_types(_cfg_get(cfg, "layer_types"), base_num_layers):
            return cfg
    for cfg in candidates:
        shared = _cfg_get(cfg, "num_kv_shared_layers")
        if isinstance(shared, int) and not isinstance(shared, bool):
            if 0 < shared < base_num_layers:
                return cfg
    return None


def _classify_layer(layer_type: str) -> str:
    """Map a lower-cased ``layer_types`` entry to a footprint class.

    Returns one of ``"recurrent"``, ``"sliding"``, ``"full"``. A recognized
    full-attention token and an UNrecognized string both resolve to ``"full"``
    — the conservative (over-counting) default that can never under-count.
    """
    if any(tok in layer_type for tok in _RECURRENT_TOKENS):
        return "recurrent"
    if any(tok in layer_type for tok in _SLIDING_TOKENS):
        return "sliding"
    if any(tok in layer_type for tok in _FULL_TOKENS):
        return "full"
    # Unrecognized attention-type string → charge as full growth (never
    # under-count).
    return "full"


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

    # Borrower split: the last ``num_kv_shared_layers`` layers reuse an earlier
    # producer's K/V and allocate nothing. ``first_borrower`` is the index at
    # which borrowing begins.
    num_shared = _cfg_get(struct, "num_kv_shared_layers")
    if isinstance(num_shared, int) and not isinstance(num_shared, bool):
        num_shared = num_shared if 0 <= num_shared < base_num_layers else 0
    else:
        num_shared = 0
    first_borrower = base_num_layers - num_shared

    layer_types = _valid_layer_types(_cfg_get(struct, "layer_types"), base_num_layers)

    full_per_token = 2 * dtype_bytes * global_kv_heads * global_head_dim
    sliding_full_per_token = 2 * dtype_bytes * local_kv_heads * local_head_dim
    sliding_window_bytes = 2 * dtype_bytes * window * local_kv_heads * local_head_dim
    # Recurrent state is a fixed per-head state matrix (~head_dim x head_dim per
    # KV head for GatedDeltaNet / Mamba). It never grows per token, so it lands
    # entirely in the fixed baseline. A conservative single-matrix term (no K+V
    # doubling) derived from the local dims — see module docstring.
    recurrent_state_bytes = (
        dtype_bytes * local_kv_heads * local_head_dim * local_head_dim
    )

    per_token_growth = 0
    fixed_baseline = 0
    for idx in range(base_num_layers):
        if idx >= first_borrower:
            # Borrower (KV-sharing) layer — allocates nothing.
            continue
        layer_class = _classify_layer(layer_types[idx]) if layer_types else "full"
        if layer_class == "recurrent":
            fixed_baseline += recurrent_state_bytes
        elif layer_class == "sliding":
            if window > 0:
                fixed_baseline += sliding_window_bytes
            else:
                # No readable window → cannot bound it → charge as full growth
                # (never under-count).
                per_token_growth += sliding_full_per_token
        else:  # "full"
            per_token_growth += full_per_token

    return KVFootprintEstimate(
        per_token_growth_bytes=int(per_token_growth),
        fixed_baseline_bytes=int(fixed_baseline),
    )
