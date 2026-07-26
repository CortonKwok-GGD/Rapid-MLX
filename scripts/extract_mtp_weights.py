#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Extract MTP weights from a HuggingFace model and save as a quantized sidecar.

mlx-lm's convert/quantize pipeline strips mtp.* weights during sanitize().
This script extracts them from the original bf16 weights, quantizes them
to match the target MLX model's quantization config, and saves them as
model-mtp.safetensors in the MLX model directory.

Usage:
    python3.12 scripts/extract_mtp_weights.py \
        --hf-model Qwen/Qwen3.5-27B \
        --mlx-model /path/to/quantized-mlx-model

The script will:
1. Download only the safetensors shard(s) containing mtp.* weights
2. Quantize them to match the MLX model's quantization config
3. Save as model-mtp.safetensors in the MLX model directory
"""

import argparse
import json
import logging
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _quantize_weight(w, group_size, bits):
    """Quantize a single weight tensor."""
    q_w, q_s, q_b = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(q_w, q_s, q_b)
    return q_w, q_s, q_b


# A norm whose mean is farther than this from BOTH the shifted reference and
# the unshifted reference (which are exactly 1.0 apart) is treated as an
# unreliable vote: its distribution is too far from either convention to trust
# individually (an independently trained MTP norm), so it is excluded from the
# consensus rather than force-classified.
NORM_AMBIGUITY_BAND = 0.3


class BackboneReferenceError(Exception):
    """The backbone norm reference exists but could not be read/parsed.

    Distinct from *no compatible reference* (an empty result): this signals an
    I/O or corruption failure so the caller can fail closed instead of silently
    double-shifting.
    """


def _norm_type(key: str) -> str:
    """Norm-type tail used to pair an MTP norm with its backbone counterpart.

    ``mtp.layers.0.input_layernorm.weight`` -> ``input_layernorm.weight``.
    """
    return ".".join(key.split(".")[-2:])


def _norms_already_shifted(mtp_mean: float, backbone_mean: float) -> bool:
    """Return True when an MTP norm is ALREADY +1-shifted.

    The backbone norm (post mlx-lm sanitize) is the shifted reference. An
    unshifted MTP norm sits ~1.0 below it (HF ``(1 + w)`` convention), so its
    mean is closer to ``backbone_mean - 1.0``. If instead the MTP mean is
    closer to ``backbone_mean``, the norm was already shifted at the source and
    applying +1.0 again would silently double-shift it (w+2).
    """
    return abs(mtp_mean - backbone_mean) < abs(mtp_mean - (backbone_mean - 1.0))


def _classify_norm(
    mtp_mean: float, backbone_mean: float, band: float = NORM_AMBIGUITY_BAND
) -> str:
    """Classify one MTP norm against its same-type backbone reference.

    Returns ``"already_shifted"``, ``"unshifted"``, or ``"ambiguous"``. The
    shifted reference is ``backbone_mean`` and the unshifted reference is
    ``backbone_mean - 1.0``. If the MTP mean is farther than ``band`` from
    whichever reference is nearest, the result is ``"ambiguous"`` (the norm's
    distribution matches neither convention closely enough to trust).
    """
    if min(abs(mtp_mean - backbone_mean), abs(mtp_mean - (backbone_mean - 1.0))) > band:
        return "ambiguous"
    return (
        "already_shifted"
        if _norms_already_shifted(mtp_mean, backbone_mean)
        else "unshifted"
    )


def _backbone_norm_means_by_type(mlx_dir: Path) -> dict[str, float]:
    """Map each backbone norm *type* to a representative +1-shifted mean.

    Reads the target MLX model's backbone (non-``mtp.``) 1-D norms and, per
    norm type, picks one deterministic representative (the sorted-first key, so
    the result is independent of JSON/shard ordering) and returns its mean.
    Returns ``{}`` when the model has no readable norm reference (unusual
    layout) — distinct from an I/O/corruption failure, which raises
    ``BackboneReferenceError`` so the caller can fail closed.
    """
    index_path = mlx_dir / "model.safetensors.index.json"
    single_path = mlx_dir / "model.safetensors"

    def _is_backbone_norm(k: str) -> bool:
        return not k.startswith("mtp.") and "norm" in k and k.endswith(".weight")

    try:
        if index_path.exists():
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            key_to_shard = {k: v for k, v in weight_map.items() if _is_backbone_norm(k)}
            if not key_to_shard:
                return {}
            # One representative key per type (deterministic: sorted-first).
            rep_by_type: dict[str, str] = {}
            for k in sorted(key_to_shard):
                rep_by_type.setdefault(_norm_type(k), k)
            means: dict[str, float] = {}
            for shard_name in sorted({key_to_shard[k] for k in rep_by_type.values()}):
                shard = mx.load(str(mlx_dir / shard_name))
                for ntype, k in rep_by_type.items():
                    if ntype not in means and key_to_shard[k] == shard_name:
                        t = shard.get(k)
                        if t is not None and t.ndim == 1:
                            means[ntype] = float(mx.mean(t).item())
            return means
        if single_path.exists():
            shard = mx.load(str(single_path))
            rep_by_type = {}
            for k in sorted(shard):
                if _is_backbone_norm(k) and shard[k].ndim == 1:
                    rep_by_type.setdefault(_norm_type(k), k)
            return {
                nt: float(mx.mean(shard[k]).item()) for nt, k in rep_by_type.items()
            }
    except Exception as e:  # parse / I/O / corruption on an existing reference
        raise BackboneReferenceError(str(e)) from e
    return {}  # no index and no single-file weights: no reference available


def main():
    parser = argparse.ArgumentParser(description="Extract MTP weights from HF model")
    parser.add_argument(
        "--hf-model", required=True, help="HuggingFace model ID (e.g. Qwen/Qwen3.5-27B)"
    )
    parser.add_argument(
        "--mlx-model", required=True, help="Path to quantized MLX model directory"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="Override quantization bits (default: from MLX model config)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Override group size (default: from MLX model config)",
    )
    parser.add_argument(
        "--force-norm-shift",
        action="store_true",
        help="Force the +1.0 norm shift even if the double-shift guard detects "
        "already-shifted MTP norms (escape hatch).",
    )
    args = parser.parse_args()

    mlx_dir = Path(args.mlx_model)
    if not mlx_dir.exists():
        logger.error(f"MLX model directory not found: {mlx_dir}")
        return 1

    # Read quantization config from MLX model
    config_path = mlx_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    quant_config = config.get("quantization", {})
    bits = args.bits or quant_config.get("bits", 4)
    group_size = args.group_size or quant_config.get("group_size", 64)
    logger.info(f"Target quantization: {bits}-bit, group_size={group_size}")

    # Find which shard files contain MTP weights
    from huggingface_hub import hf_hub_download

    logger.info(f"Downloading weight index from {args.hf_model}...")
    idx_path = hf_hub_download(args.hf_model, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)

    weight_map = idx.get("weight_map", {})
    mtp_keys = {k: v for k, v in weight_map.items() if k.startswith("mtp.")}

    if not mtp_keys:
        logger.error("No mtp.* weights found in model index!")
        return 1

    logger.info(f"Found {len(mtp_keys)} MTP weight keys")

    # Get unique shard files needed
    shard_files = sorted(set(mtp_keys.values()))
    logger.info(f"Need to download {len(shard_files)} shard file(s): {shard_files}")

    # Download and extract MTP weights
    all_mtp_weights = {}
    for shard_file in shard_files:
        logger.info(f"Downloading {shard_file}...")
        shard_path = hf_hub_download(args.hf_model, shard_file)
        shard_weights = mx.load(shard_path)
        for k in mtp_keys:
            if mtp_keys[k] == shard_file and k in shard_weights:
                all_mtp_weights[k] = shard_weights[k]
        del shard_weights

    logger.info(f"Extracted {len(all_mtp_weights)} MTP weight tensors")

    # mlx-lm's sanitize shifts norm weights by +1.0 when mtp weights are present.
    # Since we're extracting post-sanitize, the main model norms are already shifted.
    # We need to apply the same shift to MTP norm weights for consistency.
    # EVERY RMSNorm weight uses HF's ``(1 + w)`` convention; MLX's nn.RMSNorm
    # applies ``x * w`` directly, so each norm weight needs +1.0. This MUST
    # include the MTP-specific ``pre_fc_norm_embedding`` / ``pre_fc_norm_hidden``
    # (the "norm" is mid-name, so an ``endswith('norm.weight')`` list silently
    # missed them — leaving the MTP head's fc-input normalization inverted and
    # producing ~0% draft acceptance). Match any 1-D norm weight instead.
    #
    # DOUBLE-SHIFT GUARD: the +1.0 shift is correct ONLY if the HF source stores
    # its MTP norms unshifted (HF convention). If a source already stores them in
    # MLX's shifted convention, applying +1.0 again silently double-shifts (w+2),
    # with the SAME ~0% acceptance symptom and no error. The shifted/unshifted
    # convention is a per-checkpoint property, so decide it by CONSENSUS: classify
    # every MTP norm that has a same-type backbone counterpart in the target MLX
    # model (the +1-shifted reference) as already-shifted / unshifted / ambiguous,
    # then apply the majority verdict uniformly. Using many norms (not one
    # arbitrary reference) and excluding means that match neither convention
    # avoids order-dependent or noisy misclassification.
    #   --force-norm-shift            : shift unconditionally (escape hatch).
    #   no backbone counterpart / tie : fall back to shifting (historical behavior).
    #   reference unreadable (I/O)    : fail closed (do not risk a double-shift).
    mtp_norm_keys = [
        k
        for k in all_mtp_weights
        if "norm" in k and k.endswith(".weight") and all_mtp_weights[k].ndim == 1
    ]

    apply_shift = True
    if not mtp_norm_keys:
        apply_shift = False
    elif args.force_norm_shift:
        logger.info("--force-norm-shift set: applying +1.0 norm shift unconditionally.")
    else:
        try:
            backbone_means = _backbone_norm_means_by_type(mlx_dir)
        except BackboneReferenceError as e:
            logger.error(
                f"Double-shift guard could not read the backbone norm reference in "
                f"{mlx_dir}: {e}. Refusing to shift blindly (an already-shifted "
                "source would be double-shifted). Verify the source convention and "
                "re-run with --force-norm-shift."
            )
            return 1

        votes = []  # True = this norm looks already-shifted
        for k in mtp_norm_keys:
            bmean = backbone_means.get(_norm_type(k))
            if bmean is None:
                continue  # MTP-only norm (e.g. pre_fc_norm_*): no counterpart
            mmean = float(mx.mean(all_mtp_weights[k]).item())
            verdict = _classify_norm(mmean, bmean)
            if verdict == "ambiguous":
                logger.info(
                    f"  guard: '{k}' mean {mmean:.4f} matches neither the shifted "
                    f"({bmean:.4f}) nor unshifted ({bmean - 1.0:.4f}) reference; "
                    "excluded from the convention vote."
                )
                continue
            votes.append(verdict == "already_shifted")

        n_shifted, n_total = sum(votes), len(votes)
        if n_total == 0:
            logger.warning(
                "Double-shift guard found no reliable backbone counterpart for the "
                f"MTP norms in {mlx_dir}; falling back to applying the +1.0 shift. "
                "Pass --force-norm-shift to silence this."
            )
        elif n_shifted * 2 > n_total:  # strict majority already-shifted
            apply_shift = False
            logger.warning(
                f"MTP norms appear ALREADY shifted ({n_shifted}/{n_total} same-type "
                "norms match the backbone convention); skipping +1.0 shift to avoid a "
                "double-shift. Pass --force-norm-shift to override."
            )
        elif n_shifted * 2 == n_total:  # tie: undecided -> historical behavior
            logger.warning(
                f"Double-shift guard is undecided ({n_shifted}/{n_total} norms look "
                "already-shifted); falling back to applying the +1.0 shift. Pass "
                "--force-norm-shift to override."
            )
        else:
            logger.info(
                f"Double-shift guard: MTP norms match the unshifted convention "
                f"({n_total - n_shifted}/{n_total} sit ~1.0 below the backbone); "
                "applying +1.0 shift."
            )

    if apply_shift:
        for k in list(all_mtp_weights.keys()):
            if "norm" in k and k.endswith(".weight") and all_mtp_weights[k].ndim == 1:
                all_mtp_weights[k] = all_mtp_weights[k] + 1.0
                logger.info(f"  Shifted norm: {k}")

    # Convert fused MoE experts (``experts.gate_up_proj`` / ``experts.down_proj``,
    # 3-D stacked, no ``.weight`` suffix) into the ``switch_mlp.{gate,up,down}_proj``
    # split layout the MLX qwen3_5_moe model + MTP injector expect. Mirrors
    # ``mlx_lm.models.qwen3_5_moe.Model.sanitize`` so quantization below sees the
    # canonical names. Dense-MTP models have no such keys and are unaffected.
    for gate_up_key in [
        k for k in list(all_mtp_weights) if k.endswith(".experts.gate_up_proj")
    ]:
        prefix = gate_up_key[: -len(".experts.gate_up_proj")]  # e.g. mtp.layers.0.mlp
        gate_up = all_mtp_weights.pop(gate_up_key)
        if gate_up.shape[-2] % 2 != 0:
            raise ValueError(
                f"{gate_up_key}: fused gate_up_proj has odd intermediate dim "
                f"{gate_up.shape[-2]}; cannot split into equal gate/up halves. "
                "The checkpoint's expert layout is not the expected "
                "[num_experts, 2*intermediate, hidden]."
            )
        mid = gate_up.shape[-2] // 2
        all_mtp_weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
        all_mtp_weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
        down_key = f"{prefix}.experts.down_proj"
        if down_key not in all_mtp_weights:
            raise ValueError(
                f"{gate_up_key} present but paired {down_key} is missing; the "
                "extracted sidecar would be incomplete. Check that the MTP shard "
                "carries the full expert set."
            )
        all_mtp_weights[f"{prefix}.switch_mlp.down_proj.weight"] = all_mtp_weights.pop(
            down_key
        )
        logger.info(
            f"  Converted MoE experts: {prefix}.experts.* -> {prefix}.switch_mlp.*"
        )

    # Quantize MTP weights
    quantized = {}
    for k, v in sorted(all_mtp_weights.items()):
        if not k.endswith(".weight"):
            continue

        # Only 1-D norm/bias vectors stay in FP. Every 2-D Linear weight —
        # including the router ``mlp.gate`` and the ``shared_expert_gate``
        # (out_features=1) — MUST be quantized, because the MTP injector
        # (``qwen3_5_inject``) quantizes the whole MTP module uniformly and
        # then loads this sidecar into it: an FP tensor where the module
        # expects a quantized (weight/scales/biases) triple is rejected as a
        # "missing required MTP tensor". Mirror the target model's scheme.
        is_norm = "norm" in k or "layernorm" in k
        if is_norm or v.ndim == 1:
            quantized[k] = v
            logger.info(f"  FP: {k} {v.shape}")
            continue

        q_bits, q_gs = bits, group_size
        q_w, q_s, q_b = _quantize_weight(v, q_gs, q_bits)
        quantized[k] = q_w
        quantized[k.replace(".weight", ".scales")] = q_s
        quantized[k.replace(".weight", ".biases")] = q_b
        logger.info(f"  Quantized {q_bits}-bit: {k} {v.shape} -> {q_w.shape}")

    # Save
    output_file = mlx_dir / "model-mtp.safetensors"
    logger.info(f"\nSaving {len(quantized)} tensors to {output_file}")
    mx.save_safetensors(str(output_file), quantized)

    total_bytes = sum(v.nbytes for v in quantized.values())
    logger.info(f"MTP weights size: {total_bytes / 1e6:.1f} MB")

    # Ensure config has mtp_num_hidden_layers (for our patch to detect)
    text_config = config.get("text_config", config)
    if text_config.get("mtp_num_hidden_layers") is None:
        # Read from HF config
        hf_cfg_path = hf_hub_download(args.hf_model, "config.json")
        with open(hf_cfg_path) as f:
            hf_cfg = json.load(f)
        hf_tc = hf_cfg.get("text_config", hf_cfg)
        mtp_layers = hf_tc.get("mtp_num_hidden_layers", 0)
        if mtp_layers:
            if "text_config" in config:
                config["text_config"]["mtp_num_hidden_layers"] = mtp_layers
            else:
                config["mtp_num_hidden_layers"] = mtp_layers
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Updated config: mtp_num_hidden_layers={mtp_layers}")

    logger.info("\nDone! MTP weights extracted and quantized.")
    logger.info("Start server with: --enable-mtp")


if __name__ == "__main__":
    exit(main() or 0)
