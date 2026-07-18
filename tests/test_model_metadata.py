# SPDX-License-Identifier: Apache-2.0
"""Hermetic coverage for offline model metadata inspection."""

import json
import sys
import types
from pathlib import Path

import pytest

from vllm_mlx import model_metadata as metadata


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_safetensors_header(path: Path, tensor_names) -> None:
    header = json.dumps({name: {} for name in tensor_names}).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header)


def test_readers_reject_missing_malformed_non_object_and_oversized_files(tmp_path):
    missing = tmp_path / "missing.json"
    assert metadata._read_json(None) is None
    assert metadata._read_json(missing) is None
    assert metadata._read_text(None) is None
    assert metadata._read_text(missing) is None

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    assert metadata._read_json(malformed) is None

    list_json = tmp_path / "list.json"
    _write_json(list_json, ["not", "an", "object"])
    assert metadata._read_json(list_json) is None

    huge = tmp_path / "huge.txt"
    huge.write_text("x" * (metadata.MAX_METADATA_FILE_BYTES + 1), encoding="utf-8")
    assert metadata._read_json(huge) is None
    assert metadata._read_text(huge) is None

    invalid_utf8 = tmp_path / "invalid-utf8.jinja"
    invalid_utf8.write_bytes(b"\xff")
    assert metadata._read_text(invalid_utf8) is None


def test_local_metadata_prefers_standalone_template_then_tokenizer_fallback(tmp_path):
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    _write_json(standalone / "config.json", {"model_type": "qwen3"})
    _write_json(standalone / "tokenizer_config.json", {"chat_template": "tokenizer"})
    (standalone / "chat_template.jinja").write_text("standalone", encoding="utf-8")

    result = metadata.read_local_model_metadata(str(standalone))

    assert result == metadata.ModelMetadata(
        config={"model_type": "qwen3"},
        chat_template="standalone",
        snapshot_dir=standalone,
        is_local=True,
    )

    fallback = tmp_path / "fallback"
    fallback.mkdir()
    _write_json(fallback / "tokenizer_config.json", {"chat_template": "tokenizer"})
    fallback_result = metadata.read_local_model_metadata(str(fallback))

    assert fallback_result == metadata.ModelMetadata(
        config=None,
        chat_template="tokenizer",
        snapshot_dir=fallback,
        is_local=True,
    )
    assert metadata._chat_template(None) is None
    assert metadata.read_local_model_metadata(object()) is None


def test_named_tokenizer_templates_prefer_tool_use_then_default():
    assert (
        metadata._select_chat_template(
            {"chat_template": {"default": "default", "tool_use": "tool-use"}}
        )
        == "tool-use"
    )
    assert (
        metadata._select_chat_template({"chat_template": {"default": "default"}})
        == "default"
    )
    assert (
        metadata._select_chat_template({"chat_template": {"a": "one", "b": "two"}})
        is None
    )


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("publisher/model", True),
        ("publisher/nested/model", True),
        ("plain-name", False),
        ("/absolute/model", False),
        ("./relative/model", False),
        ("../parent/model", False),
        ("~/cache/model", False),
    ],
)
def test_hub_repo_id_validation_rejects_path_lookalikes(model_name, expected):
    assert metadata._looks_like_hub_repo_id(model_name) is expected


def test_cached_file_handles_cache_hit_missing_and_lookup_error(monkeypatch, tmp_path):
    hit = tmp_path / "config.json"
    _write_json(hit, {"ok": True})
    no_exist = object()
    responses = {
        "config.json": str(hit),
        "missing.json": None,
        "no-exist.json": no_exist,
    }

    def lookup(repo_id, filename):
        if filename == "explode.json":
            raise RuntimeError("cache unavailable")
        return responses[filename]

    hub = types.ModuleType("huggingface_hub")
    hub._CACHED_NO_EXIST = no_exist
    hub.try_to_load_from_cache = lookup
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert metadata._cached_file("publisher/model", "config.json") == hit
    assert metadata._cached_file("publisher/model", "missing.json") is None
    assert metadata._cached_file("publisher/model", "no-exist.json") is None
    assert metadata._cached_file("publisher/model", "explode.json") is None
    assert metadata._cached_file("not-a-repo", "config.json") is None


def test_cached_file_handles_missing_huggingface_hub_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    assert metadata._cached_file("publisher/model", "config.json") is None


def test_cached_metadata_reads_standalone_template_and_tokenizer_fallback(
    monkeypatch, tmp_path
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = snapshot / "config.json"
    standalone = snapshot / "chat_template.jinja"
    tokenizer = snapshot / "tokenizer_config.json"
    _write_json(config, {"model_type": "qwen3_5"})
    standalone.write_text("standalone", encoding="utf-8")
    _write_json(tokenizer, {"chat_template": "tokenizer"})
    paths = {
        "config.json": config,
        "chat_template.jinja": standalone,
        "tokenizer_config.json": tokenizer,
    }
    monkeypatch.setattr(
        metadata, "_cached_file", lambda name, filename: paths[filename]
    )

    result = metadata.read_cached_model_metadata("publisher/model")

    assert result == metadata.ModelMetadata(
        config={"model_type": "qwen3_5"},
        chat_template="standalone",
        snapshot_dir=snapshot,
    )

    standalone.unlink()
    paths["chat_template.jinja"] = None
    result = metadata.read_cached_model_metadata("publisher/model")
    assert result is not None
    assert result.chat_template == "tokenizer"


def test_cached_metadata_reads_one_snapshot_when_cache_refs_change(
    monkeypatch, tmp_path
):
    snapshot = tmp_path / "snapshot"
    stale = tmp_path / "stale"
    snapshot.mkdir()
    stale.mkdir()
    _write_json(snapshot / "config.json", {"model_type": "qwen3"})
    (snapshot / "chat_template.jinja").write_text("current", encoding="utf-8")
    (stale / "chat_template.jinja").write_text("stale", encoding="utf-8")
    calls = []

    def cached_file(name, filename):
        calls.append(filename)
        if filename == "config.json":
            return snapshot / filename
        return stale / filename

    monkeypatch.setattr(metadata, "_cached_file", cached_file)

    result = metadata.read_cached_model_metadata("publisher/model")

    assert result is not None
    assert result.snapshot_dir == snapshot
    assert result.chat_template == "current"
    assert calls == ["config.json"]


def test_cached_metadata_returns_none_without_any_cached_metadata(monkeypatch):
    monkeypatch.setattr(metadata, "_cached_file", lambda name, filename: None)

    assert metadata.read_cached_model_metadata("publisher/model") is None


def test_read_model_metadata_prefers_local_directory_then_cache(monkeypatch, tmp_path):
    local = metadata.ModelMetadata({}, "local", tmp_path)
    cached = metadata.ModelMetadata({}, "cached", tmp_path)
    monkeypatch.setattr(metadata, "read_local_model_metadata", lambda name: local)
    monkeypatch.setattr(metadata, "read_cached_model_metadata", lambda name: cached)
    assert metadata.read_model_metadata("anything") is local

    monkeypatch.setattr(metadata, "read_local_model_metadata", lambda name: None)
    assert metadata.read_model_metadata("anything") is cached


def test_multimodal_config_and_sharded_weight_detection(tmp_path):
    assert metadata.config_indicates_multimodal(
        {"architectures": ["LlavaForConditionalGeneration"]}
    )
    assert metadata.config_indicates_multimodal({"audio_config": {}})
    assert not metadata.config_indicates_multimodal({"architectures": "not-a-list"})

    assert metadata.checkpoint_has_multimodal_weights(None) is None
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is None

    _write_json(
        tmp_path / "model.safetensors.index.json",
        {"weight_map": {"language_model.layers.0.weight": "model.safetensors"}},
    )
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is None
    assert (
        metadata.checkpoint_has_multimodal_weights(
            tmp_path,
            {"architectures": ["Qwen3_5MoeForConditionalGeneration"]},
        )
        is False
    )

    _write_json(tmp_path / "model.safetensors.index.json", {"weight_map": []})
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is None

    _write_json(
        tmp_path / "model.safetensors.index.json",
        {"weight_map": {"vision_tower.blocks.0.weight": "model.safetensors"}},
    )
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is True

    (tmp_path / "model.safetensors.index.json").unlink()
    safetensors = tmp_path / "model.safetensors"
    _write_safetensors_header(safetensors, ["language_model.layers.0.weight"])
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is None

    _write_safetensors_header(safetensors, ["vision_tower.blocks.0.weight"])
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is True

    _write_safetensors_header(safetensors, ["vision_encoder.blocks.0.weight"])
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is None


def test_weight_index_has_independent_production_size_bound(tmp_path):
    weight_map = {
        "language_model." + "x" * metadata.MAX_METADATA_FILE_BYTES: "model.safetensors",
        "vision_tower.blocks.0.weight": "model.safetensors",
    }
    _write_json(tmp_path / "model.safetensors.index.json", {"weight_map": weight_map})

    assert (
        tmp_path / "model.safetensors.index.json"
    ).stat().st_size > metadata.MAX_METADATA_FILE_BYTES
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is True


def test_single_safetensors_header_rejects_corrupt_or_unsupported_shapes(tmp_path):
    model = tmp_path / "model.safetensors"

    model.write_bytes(b"tiny")
    assert metadata._single_safetensors_has_multimodal_weights(tmp_path) is None

    model.write_bytes((metadata.MAX_METADATA_FILE_BYTES + 1).to_bytes(8, "little"))
    assert metadata._single_safetensors_has_multimodal_weights(tmp_path) is None

    model.write_bytes((8).to_bytes(8, "little") + b"{}")
    assert metadata._single_safetensors_has_multimodal_weights(tmp_path) is None

    model.write_bytes((1).to_bytes(8, "little") + b"[")
    assert metadata._single_safetensors_has_multimodal_weights(tmp_path) is None

    header = json.dumps(["not", "an", "object"]).encode("utf-8")
    model.write_bytes(len(header).to_bytes(8, "little") + header)
    assert metadata._single_safetensors_has_multimodal_weights(tmp_path) is None
