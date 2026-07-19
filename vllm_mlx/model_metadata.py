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
# Checkpoint indexes enumerate every tensor in large sharded models, so they
# legitimately exceed the config/template cap.  Keep their own bounded budget
# rather than silently discarding modality evidence for production checkpoints.
MAX_WEIGHT_INDEX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ModelMetadata:
    """Offline metadata available for a local directory or HF snapshot."""

    config: dict[str, Any] | None
    chat_template: str | None
    snapshot_dir: Path | None
    is_local: bool = False


def _read_json(
    path: Path | None, *, max_bytes: int = MAX_METADATA_FILE_BYTES
) -> dict[str, Any] | None:
    """Read one bounded JSON object, returning ``None`` on cache failures."""
    if path is None or not path.is_file():
        return None
    try:
        if path.stat().st_size > max_bytes:
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


def _select_chat_template(tokenizer_config: dict[str, Any] | None) -> str | None:
    """Select the template Transformers uses when tools are present."""
    candidate = tokenizer_config.get("chat_template") if tokenizer_config else None
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, list):
        candidate = {
            item["name"]: item["template"]
            for item in candidate
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("template"), str)
        }
    if not isinstance(candidate, dict):
        return None
    for name in ("tool_use", "default"):
        template = candidate.get(name)
        if isinstance(template, str):
            return template
    templates = [
        template for template in candidate.values() if isinstance(template, str)
    ]
    return templates[0] if len(templates) == 1 else None


# Transformers stores extra (named) chat templates under this directory, one
# ``<name>.jinja`` per template, with ``chat_template.jinja`` acting as the
# ``default``.  Mirrors ``transformers.utils.hub.CHAT_TEMPLATE_DIR``.
CHAT_TEMPLATE_DIR = "additional_chat_templates"


def _read_named_chat_templates(snapshot_dir: Path) -> dict[str, str]:
    """Read named templates from the ``additional_chat_templates/`` directory.

    Mirrors ``PreTrainedTokenizerBase._from_pretrained`` (tokenization_utils_base):
    each ``<name>.jinja`` under ``additional_chat_templates/`` becomes a
    ``chat_templates[name]`` entry, with ``name = filename.removesuffix(".jinja")``.
    """
    template_dir = snapshot_dir / CHAT_TEMPLATE_DIR
    if not template_dir.is_dir():
        return {}
    named: dict[str, str] = {}
    try:
        template_files = sorted(template_dir.glob("*.jinja"))
    except OSError:
        return {}
    for template_file in template_files:
        template = _read_text(template_file)
        if template is not None:
            named[template_file.name.removesuffix(".jinja")] = template
    return named


def _chat_template(snapshot_dir: Path | None) -> str | None:
    """Load the template using Transformers' standalone-file precedence.

    Mirrors ``PreTrainedTokenizerBase._from_pretrained`` (tokenization_utils_base
    ~lines 1665-1808): independent chat-template files take priority over the
    ``tokenizer_config.json`` entry.  ``chat_template.jinja`` supplies the
    ``default`` template, while every ``<name>.jinja`` under
    ``additional_chat_templates/`` contributes a named template.  Transformers
    collapses a lone ``default`` to a single string and otherwise keeps a
    ``{name: template}`` dict; we then apply the SAME ``tool_use`` → ``default``
    selection used for the tokenizer-config form via ``_select_chat_template``.
    """
    if snapshot_dir is None:
        return None
    templates = _read_named_chat_templates(snapshot_dir)
    standalone = _read_text(snapshot_dir / "chat_template.jinja")
    if standalone is not None:
        templates["default"] = standalone
    if templates:
        # Transformers flattens a lone ``default`` to a bare string.
        if len(templates) == 1 and "default" in templates:
            return templates["default"]
        return _select_chat_template({"chat_template": templates})
    return _select_chat_template(_read_json(snapshot_dir / "tokenizer_config.json"))


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
        is_local=True,
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
    if (
        cached is None
        or cached is _CACHED_NO_EXIST
        or not isinstance(cached, (str, os.PathLike))
    ):
        return None
    return Path(cached)


def read_cached_model_metadata(model_name: str) -> ModelMetadata | None:
    """Read metadata for a cached HF repository, never triggering download."""
    config_path = _cached_file(model_name, "config.json")
    snapshot_dir = config_path.parent if config_path is not None else None
    if snapshot_dir is None:
        template_path = _cached_file(model_name, "chat_template.jinja")
        snapshot_dir = template_path.parent if template_path is not None else None
    if snapshot_dir is None:
        tokenizer_path = _cached_file(model_name, "tokenizer_config.json")
        snapshot_dir = tokenizer_path.parent if tokenizer_path is not None else None
    if snapshot_dir is None:
        return None
    return ModelMetadata(
        config=_read_json(snapshot_dir / "config.json"),
        chat_template=_chat_template(snapshot_dir),
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
_QWEN3_5_MOE_ARCHITECTURE = "qwen3_5moeforconditionalgeneration"
_QWEN3_5_MOE_TEXT_TENSOR_PREFIX = "language_model."


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


def _contains_multimodal_weight_names(weight_names) -> bool:
    """Return whether an iterable of safetensors names has modality weights."""
    return any(
        isinstance(name, str)
        and any(prefix in name for prefix in MULTIMODAL_TENSOR_PREFIXES)
        for name in weight_names
    )


def _known_text_only_weight_layout(weight_names, config: dict[str, Any] | None) -> bool:
    """Recognise only an exhaustive architecture-specific text-only layout."""
    architectures = config.get("architectures") if config else None
    if not isinstance(architectures, list) or not any(
        isinstance(architecture, str)
        and architecture.lower() == _QWEN3_5_MOE_ARCHITECTURE
        for architecture in architectures
    ):
        return False
    names = tuple(name for name in weight_names if isinstance(name, str))
    return bool(names) and all(
        name.startswith(_QWEN3_5_MOE_TEXT_TENSOR_PREFIX) for name in names
    )


def _single_safetensors_has_multimodal_weights(snapshot_dir: Path) -> bool | None:
    """Inspect one safetensors header without loading model tensor data.

    The header is useful positive evidence: a known vision/audio prefix proves
    that the checkpoint is multimodal.  It is *not* exhaustive negative
    evidence, though.  Repackagers may name a vision encoder differently, and
    the header does not declare an architecture-wide tensor schema.  Preserve
    that uncertainty as ``None`` instead of routing such a VLM to the text
    loader.
    """
    files = tuple(snapshot_dir.glob("*.safetensors"))
    if len(files) != 1:
        return None
    try:
        with files[0].open("rb") as f:
            size_bytes = f.read(8)
            if len(size_bytes) != 8:
                return None
            header_size = int.from_bytes(size_bytes, "little")
            if header_size > MAX_METADATA_FILE_BYTES:
                return None
            header = f.read(header_size)
        if len(header) != header_size:
            return None
        parsed = json.loads(header.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return True if _contains_multimodal_weight_names(parsed) else None


def checkpoint_has_multimodal_weights(
    snapshot_dir: Path | None, config: dict[str, Any] | None = None
) -> bool | None:
    """Return positive modality proof or a schema-backed text-only verdict.

    Unrecognised tensor namespaces are deliberately inconclusive.  A generic
    prefix list cannot prove a checkpoint has no vision encoder; only a known
    architecture's complete language-only layout may return ``False``.
    """
    if snapshot_dir is None:
        return None
    index = _read_json(
        snapshot_dir / "model.safetensors.index.json",
        max_bytes=MAX_WEIGHT_INDEX_BYTES,
    )
    if index is None:
        return _single_safetensors_has_multimodal_weights(snapshot_dir)
    weights = index.get("weight_map")
    if not isinstance(weights, dict):
        return None
    if _contains_multimodal_weight_names(weights):
        return True
    return False if _known_text_only_weight_layout(weights, config) else None


def checkpoint_evidence_is_available(snapshot_dir: Path | None) -> bool:
    """Return whether checkpoint metadata was available for inspection."""
    if snapshot_dir is None:
        return False
    return (snapshot_dir / "model.safetensors.index.json").is_file() or any(
        snapshot_dir.glob("*.safetensors")
    )
