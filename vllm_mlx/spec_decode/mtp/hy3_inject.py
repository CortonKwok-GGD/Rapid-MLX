# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP injection for HY3 (Tencent Hunyuan 3, ``model_type=hy_v3``).

Forked from :mod:`vllm_mlx.spec_decode.mtp.qwen3_5_inject`. HY3's native MTP head
is DeepSeek-V3-shaped (``enorm``/``hnorm``/``eh_proj``), not Qwen3.5-shaped
(``pre_fc_norm_*``/``fc``), so we build the head from
:func:`vllm_mlx.spec_decode.mtp.hy3_head.build_hy3_mtp_module` and inject the same
four contract surfaces the generator needs:

* ``__call__(inputs, cache=None, input_embeddings=None, return_hidden=False,
  n_confirmed=0)`` — inlines ``hy_v3.HYV3Model``'s single-node backbone loop and
  returns ``(logits, pre_final_norm_hidden)`` when ``return_hidden=True``. HY3 is
  pure-attention (no GatedDeltaNet), so ``n_confirmed`` is a no-op.
* ``mtp_forward(hidden, next_token_ids, mtp_cache)`` -> logits, via the shared
  ``lm_head`` (HY3 has ``tie_word_embeddings=false``).
* ``make_mtp_cache()`` -> ``[KVCache()]`` (1 full-attention MTP layer).

The loaded HY3 model is ``vllm_mlx.models.hy_v3.Model`` (no VLM wrapper): it holds
``.model`` (the ``HYV3Model`` backbone with ``embed_tokens`` / ``layers`` /
``norm``), ``.lm_head``, and ``.args`` (a ``hy_v3.ModelArgs``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The 4-bit HY3 MLX checkpoint (mlx-community/Hy3-preview-4bit) STRIPS the
# native MTP head at convert time (it keeps only backbone layers 0..79). The
# head lives in this published sidecar, extracted from the full-precision
# tencent/Hy3-preview layer 80 (see scripts/extract_hy3_mtp.py). Unlike
# Qwen3.5 — whose MTP head ships inside the base checkpoint — HY3 must resolve
# this sidecar by default so a bare
# ``--speculative-config '{"method":"mtp"}'`` boot works with no extra flag.
DEFAULT_HY3_MTP_SIDECAR = "mlx-community/Hy3-preview-MTP-4bit"


def _resolve_inner_model(model: Any) -> Any:
    """Return the HY3 ``Model`` instance to patch (holds ``.model`` + ``.args``)."""
    # HY3 has no VLM wrapper — the loaded object already exposes .model + .args.
    if hasattr(model, "model") and hasattr(model, "args"):
        return model
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "args") and hasattr(lm, "model"):
        return lm
    return None


def _detect_base_quantization(inner: Any) -> dict | None:
    """Detect ``bits`` / ``group_size`` from a backbone QuantizedLinear."""
    try:
        from mlx.nn import QuantizedEmbedding, QuantizedLinear
    except ImportError:  # pragma: no cover
        return None

    backbone = getattr(inner, "model", None)
    if backbone is None:
        return None
    for layer in getattr(backbone, "layers", []):
        if layer is None:
            continue
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
            qp = layer.self_attn.q_proj
            if isinstance(qp, QuantizedLinear):
                return {"bits": int(qp.bits), "group_size": int(qp.group_size)}
    embed = getattr(backbone, "embed_tokens", None)
    if isinstance(embed, QuantizedEmbedding):
        return {"bits": int(embed.bits), "group_size": int(embed.group_size)}
    return None


def _find_mtp_weights_file(sidecar_dir: Path) -> Path | None:
    for c in (sidecar_dir / "model-mtp.safetensors", sidecar_dir / "model.safetensors"):
        if c.exists():
            return c
    return None


def _resolve_sidecar_file(mtp_sidecar: str | Path) -> Path | None:
    if mtp_sidecar is None:
        return None
    path = Path(mtp_sidecar)
    if path.is_file():
        return path
    if path.is_dir():
        return _find_mtp_weights_file(path)
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id=str(mtp_sidecar))
        return _find_mtp_weights_file(Path(local))
    except Exception as exc:  # pragma: no cover — network failure path
        logger.warning(
            "[mtp.inject.hy3] could not resolve sidecar %r: %s", mtp_sidecar, exc
        )
        return None


def _resolve_num_mtp_layers(inner: Any, model: Any) -> int:
    """HY3 config uses ``num_nextn_predict_layers`` (not ``mtp_num_hidden_layers``)."""
    args = inner.args
    n = int(getattr(args, "num_nextn_predict_layers", 0) or 0)
    if n >= 1:
        return n
    # Fall back to a wrapper text_config dict (defensive; HY3 has no wrapper).
    outer_args = getattr(model, "args", None)
    text_config = getattr(outer_args, "text_config", None) or {}
    if isinstance(text_config, dict):
        n = int(text_config.get("num_nextn_predict_layers", 0) or 0)
    return n


def inject_hy3_mtp_support(
    model: Any,
    mtp_sidecar: str | Path | None = None,
    *,
    allow_random_init: bool = False,
) -> bool:
    """Inject the four MTP contract surfaces onto a loaded HY3 model.

    Args:
        model: Loaded ``vllm_mlx.models.hy_v3.Model`` (no VLM wrapper).
        mtp_sidecar: Path / dir / HF repo id holding the extracted MTP head.
            When ``None`` (the common ``--speculative-config '{"method":"mtp"}'``
            path with no explicit ``--mtp-sidecar``), it defaults to
            :data:`DEFAULT_HY3_MTP_SIDECAR` and is downloaded from the Hub on
            first use — HY3's 4-bit backbone ships with the head stripped, so a
            sidecar is always required (there is no baked-in head to fall back
            to). Pass ``allow_random_init=True`` (test-only) to skip resolution.
        allow_random_init: Test-only escape hatch — permits a random-init head
            with no sidecar. Never enable in production (~0% accept rate).

    Returns:
        ``True`` when the four MTP surfaces are attached; ``False`` on any
        refusal (unresolvable sidecar, missing tensors, wrong arch).
    """
    import mlx.core as mx
    import mlx.nn as nn

    # HY3's base checkpoint has no baked-in MTP head — always resolve a sidecar
    # (default to the published repo) unless the caller opted into random init.
    if mtp_sidecar is None and not allow_random_init:
        mtp_sidecar = DEFAULT_HY3_MTP_SIDECAR
        logger.info(
            "[mtp.inject.hy3] no explicit sidecar; defaulting to %s",
            DEFAULT_HY3_MTP_SIDECAR,
        )

    inner = _resolve_inner_model(model)
    if inner is None:
        logger.warning(
            "[mtp.inject.hy3] model %s lacks (model + args); skipping.",
            type(model).__name__,
        )
        return False

    args = inner.args
    num_mtp_layers = _resolve_num_mtp_layers(inner, model)
    if num_mtp_layers < 1:
        logger.info(
            "[mtp.inject.hy3] config has no num_nextn_predict_layers >= 1; skipping."
        )
        return False

    # --- Step 1: Build the HY3 MTP head. ---
    from .hy3_head import build_hy3_mtp_module

    mtp = build_hy3_mtp_module(args, num_mtp_layers)
    logger.info(
        "[mtp.inject.hy3] Built HY3 MTP head (%d layer(s), hidden_size=%d).",
        num_mtp_layers,
        getattr(args, "hidden_size", -1),
    )

    # --- Step 2: Quantize to match base (4-bit Linear; 8-bit router.gate). ---
    quant_info = _detect_base_quantization(inner)
    if quant_info is not None:

        def _class_predicate(path: str, module) -> Any:
            # Only touch modules that actually support quantization (Linear /
            # Embedding / SwitchLinear expose ``to_quantized``); never norms.
            # 8-bit for the MoE router gate; 4-bit for the rest. Mirrors
            # hy_v3.Model.quant_predicate + the default nn.quantize gate.
            if not hasattr(module, "to_quantized"):
                return False
            if path.endswith("mlp.router.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        nn.quantize(
            mtp,
            group_size=quant_info["group_size"],
            bits=quant_info["bits"],
            class_predicate=_class_predicate,
        )
        logger.info(
            "[mtp.inject.hy3] Quantized MTP head: %d-bit gs=%d (router.gate 8-bit)",
            quant_info["bits"],
            quant_info["group_size"],
        )

    # --- Step 3: Load sidecar weights with strict coverage check. ---
    if mtp_sidecar is not None:
        weights_file = _resolve_sidecar_file(mtp_sidecar)
        if weights_file is None:
            logger.warning(
                "[mtp.inject.hy3] sidecar %r could not be resolved; skipping.",
                mtp_sidecar,
            )
            return False
        raw = mx.load(str(weights_file))
        mtp_weights = {
            (k.removeprefix("mtp.") if k.startswith("mtp.") else k): v
            for k, v in raw.items()
        }
        from mlx.utils import tree_flatten

        expected_keys = {k for k, _ in tree_flatten(mtp.parameters())}
        loaded_keys = set(mtp_weights.keys())
        missing = expected_keys - loaded_keys
        if missing:
            logger.warning(
                "[mtp.inject.hy3] sidecar %s missing %d required tensor(s); "
                "refusing partial-random head. Missing (first 8): %s",
                weights_file.name,
                len(missing),
                sorted(missing)[:8],
            )
            return False
        mtp.load_weights(list(mtp_weights.items()), strict=False)
        mx.eval(mtp.parameters())
        extra = loaded_keys - expected_keys
        logger.info(
            "[mtp.inject.hy3] Loaded %d/%d MTP tensors from %s%s",
            len(expected_keys),
            len(expected_keys),
            weights_file.name,
            f" (+{len(extra)} extra ignored)" if extra else "",
        )
    else:
        if not allow_random_init:
            logger.warning(
                "[mtp.inject.hy3] no mtp_sidecar and allow_random_init=False; "
                "refusing random-init head."
            )
            return False
        mx.eval(mtp.parameters())
        logger.warning(
            "[mtp.inject.hy3] allow_random_init=True — RANDOM init head (test-only)."
        )

    # --- Step 4: Attach + monkey-patch the HY3 Model class. ---
    inner.mtp = mtp
    original_class = type(inner)

    class _HY3WithMTP(original_class):  # type: ignore[valid-type, misc]
        """HY3 ``Model`` + the four MTP surfaces the generator drives."""

        def __call__(  # type: ignore[override]
            self,
            inputs,
            cache=None,
            input_embeddings=None,
            return_hidden: bool = False,
            n_confirmed: int = 0,
        ):
            from mlx_lm.models.base import create_attention_mask

            backbone = self.model
            if input_embeddings is not None:
                h = input_embeddings
            else:
                h = backbone.embed_tokens(inputs)
            if cache is None:
                cache = [None] * len(backbone.layers)

            # Single-node inline of hy_v3.HYV3Model.__call__ (pipeline_size=1,
            # rank=0 => no distributed branches). n_confirmed is a no-op:
            # HY3 is pure-attention, no SSM/conv state to snapshot.
            mask = create_attention_mask(h, cache[0])
            for layer, c in zip(backbone.layers, cache):
                h = layer(h, mask, cache=c)

            pre_norm_hidden = h
            normed = backbone.norm(h)
            if self.args.enable_lm_head_fp32:
                normed = normed.astype(mx.float32)
            if self.args.tie_word_embeddings:
                out = backbone.embed_tokens.as_linear(normed)
            else:
                out = self.lm_head(normed)

            if return_hidden:
                return out, pre_norm_hidden
            return out

        def mtp_forward(self, hidden_states, next_token_ids, mtp_cache):
            mtp_out = self.mtp(
                hidden_states,
                next_token_ids,
                self.model.embed_tokens,
                mtp_cache,
            )
            if self.args.enable_lm_head_fp32:
                mtp_out = mtp_out.astype(mx.float32)
            if self.args.tie_word_embeddings:
                return self.model.embed_tokens.as_linear(mtp_out)
            return self.lm_head(mtp_out)

        def make_mtp_cache(self):
            from mlx_lm.models.cache import KVCache

            return [KVCache() for _ in self.mtp.layers]

    inner.__class__ = _HY3WithMTP
    logger.info(
        "[mtp.inject.hy3] Patched %s with MTP surfaces "
        "(return_hidden, n_confirmed, mtp_forward, make_mtp_cache).",
        original_class.__name__,
    )
    return True


def validate_hy3_mtp_support(model: Any) -> bool:
    """Verify inject_hy3_mtp_support attached the four surfaces."""
    import inspect

    inner = _resolve_inner_model(model)
    if inner is None:
        return False
    if getattr(inner, "mtp", None) is None:
        logger.warning("[mtp.validate.hy3] model.mtp is missing.")
        return False
    if not callable(getattr(inner, "mtp_forward", None)):
        logger.warning("[mtp.validate.hy3] model.mtp_forward is missing.")
        return False
    if not callable(getattr(inner, "make_mtp_cache", None)):
        logger.warning("[mtp.validate.hy3] model.make_mtp_cache is missing.")
        return False
    sig = inspect.signature(type(inner).__call__)
    if "return_hidden" not in sig.parameters:
        logger.warning("[mtp.validate.hy3] __call__ lacks return_hidden.")
        return False
    if "n_confirmed" not in sig.parameters:
        logger.warning("[mtp.validate.hy3] __call__ lacks n_confirmed.")
        return False
    return True
