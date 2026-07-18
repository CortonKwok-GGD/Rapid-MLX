# SPDX-License-Identifier: Apache-2.0
"""Offline model metadata inspection shared by routing decisions.

Model names are mutable packaging labels, while a downloaded checkpoint's
``config.json`` and chat template declare the architecture and wire protocol
the runtime actually needs to support.  This module reads that metadata from
either a local model directory or the local Hugging Face cache.  It never
contacts Hugging Face, so callers may safely use it on every server start.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# These files are configuration, not model weights.  Keep the reads bounded
# anyway: a corrupt cache entry must not turn startup classification into an
# unbounded allocation.
MAX_METADATA_FILE_BYTES = 1 * 1024 * 1024


@dataclass(frozen=True)
class ModelMetadata:
    """Offline metadata available for a local directory or HF snapshot."""

    config: dict[str, Any] | None
    chat_template: str | None
    snapshot_dir: Path | None


def _read_json(path: Path | None) -> dict[str, Any] | None:
    """Read one bounded JSON object, returning ``None`` on cache failures."""
    if path is None or not path.is_file():
        return None
    try:
        if path.stat().st_size > MAX_METADATA_FILE_BYTES:
            return None
        with path.open(encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _read_text(path: Path | None) -> str | None:
    """Read one bounded template, returning ``None`` on cache failures."""
    if path is None or not path.is_file():
        return None
    try:
        if path.stat().st_size > MAX_METADATA_FILE_BYTES:
            return None
        with path.open(encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def _chat_template(snapshot_dir: Path | None) -> str | None:
    """Load the template using Transformers' standalone-file precedence."""
    if snapshot_dir is None:
        return None
    standalone = _read_text(snapshot_dir / "chat_template.jinja")
    if standalone is not None:
        return standalone
    tokenizer_config = _read_json(snapshot_dir / "tokenizer_config.json")
    template = tokenizer_config.get("chat_template") if tokenizer_config else None
    return template if isinstance(template, str) else None


def read_local_model_metadata(model_path: str) -> ModelMetadata | None:
    """Read metadata from a local model directory without interpreting IDs."""
    try:
        snapshot_dir = Path(model_path)
    except (TypeError, ValueError):
        return None
    if not snapshot_dir.is_dir():
        return None
    return ModelMetadata(
        config=_read_json(snapshot_dir / "config.json"),
        chat_template=_chat_template(snapshot_dir),
        snapshot_dir=snapshot_dir,
    )


def _looks_like_hub_repo_id(model_name: str) -> bool:
    """Accept HF ``owner/repository`` IDs, never filesystem lookalikes."""
    return (
        isinstance(model_name, str)
        and "/" in model_name
        and not model_name.startswith(("/", "./", "../", "~"))
    )


def _cached_file(model_name: str, filename: str) -> Path | None:
    """Resolve a cache file through huggingface_hub without network access."""
    if not _looks_like_hub_repo_id(model_name):
        return None
    try:
        from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache
    except ImportError:
        return None
    try:
        cached = try_to_load_from_cache(model_name, filename)
    except Exception:
        return None
    if cached is None or cached is _CACHED_NO_EXIST or not isinstance(cached, (str, os.PathLike)):
        return None
    return Path(cached)


def read_cached_model_metadata(model_name: str) -> ModelMetadata | None:
    """Read metadata for a cached HF repository, never triggering download."""
    config_path = _cached_file(model_name, "config.json")
    template_path = _cached_file(model_name, "chat_template.jinja")
    tokenizer_path = _cached_file(model_name, "tokenizer_config.json")
    snapshot_dir = next(
        (
            path.parent
            for path in (config_path, template_path, tokenizer_path)
            if path is not None
        ),
        None,
    )
    if snapshot_dir is None:
        return None
    template = _read_text(template_path)
    if template is None:
        tokenizer_config = _read_json(tokenizer_path)
        candidate = tokenizer_config.get("chat_template") if tokenizer_config else None
        template = candidate if isinstance(candidate, str) else None
    return ModelMetadata(
        config=_read_json(config_path),
        chat_template=template,
        snapshot_dir=snapshot_dir,
    )


def read_model_metadata(model_name: str) -> ModelMetadata | None:
    """Read local-directory metadata first, then the local HF-cache entry."""
    local = read_local_model_metadata(model_name)
    return local if local is not None else read_cached_model_metadata(model_name)


# Config keys / architecture fragments shared by MLLM routing and tests.
VLM_CONFIG_KEYS = (
    "vision_config",
    "audio_config",
    "vision_tower",
    "visual_config",
    "mm_vision_tower",
    "image_token_id",
    "image_token_index",
    "audio_token_id",
    "audio_token_index",
)
VLM_ARCHITECTURE_KEYWORDS = (
    "VLForCondition",
    "VLForCausal",
    "VisionForCondition",
    "VisionForCausal",
    "MultiModalityCausalLM",
    "Llava",
    "Idefics",
    "PaliGemma",
    "Pixtral",
    "Molmo",
    "Phi3V",
    "Phi4V",
    "CogVLM",
    "InternVL",
    "DeepseekVL",
    "Mllama",
    "Gemma3ForConditional",
    "Gemma4ForConditional",
)
MULTIMODAL_TENSOR_PREFIXES = (
    "vision_tower",
    "vision_model",
    "visual.",
    "audio_tower",
    "audio_model",
    "mm_projector",
    "patch_embed.",
)


def config_indicates_multimodal(config: dict[str, Any]) -> bool:
    """Return whether a model config declares a vision or audio modality."""
    architectures = config.get("architectures") or []
    if isinstance(architectures, list):
        for architecture in architectures:
            if isinstance(architecture, str) and any(
                keyword.lower() in architecture.lower()
                for keyword in VLM_ARCHITECTURE_KEYWORDS
            ):
                return True
    return any(key in config for key in VLM_CONFIG_KEYS)


def checkpoint_has_multimodal_weights(snapshot_dir: Path | None) -> bool | None:
    """Inspect a sharded checkpoint index for actual vision/audio tensors."""
    if snapshot_dir is None:
        return None
    index = _read_json(snapshot_dir / "model.safetensors.index.json")
    if index is None:
        return None
    weights = index.get("weight_map")
    if not isinstance(weights, dict):
        return None
    return any(
        isinstance(name, str)
        and any(prefix in name for prefix in MULTIMODAL_TENSOR_PREFIXES)
        for name in weights
    )
