# SPDX-License-Identifier: Apache-2.0
"""Wire-level tests for the KV cache export/import HTTP API (#476 stub).

The engine integration is the follow-up PR's job — these tests cover
the protocol surface that the stub freezes: auth, path sandbox,
manifest validation, and the explicit 501s on the engine-touching paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.cache.protocol import (
    PROTOCOL_VERSION,
    InvalidExportPathError,
    MalformedManifestError,
    Manifest,
    read_manifest,
    resolve_cache_dir,
    write_manifest,
)


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Point the export sandbox at an isolated tmp dir for the test."""
    export_root = tmp_path / "exports"
    monkeypatch.setenv("RAPID_MLX_CACHE_EXPORT_DIR", str(export_root))
    return export_root


class _FakeCache:
    """Minimal stand-in for ``MemoryAwarePrefixCache`` — just the ledger
    fields the routes read (``_entries`` / ``_current_memory``) plus a
    ``clear()`` that the ``merge_strategy="replace"`` path calls."""

    def __init__(self, entries: int = 0, current_memory: int = 0):
        # The routes read ``len(cache._entries)`` and ``cache._current_memory``.
        self._entries = {f"k{i}": i for i in range(entries)}
        self._current_memory = current_memory
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1
        self._entries = {}
        self._current_memory = 0


class _FakeEngine:
    """Faithful fake of the engine surface the cache routes touch.

    Mirrors ``EngineCore``: ``scheduler.memory_aware_cache`` (the prefix
    cache), ``scheduler.config`` (a real-ish ``SchedulerConfig`` shape) plus
    ``save_cache_to_disk`` / ``load_cache_from_disk``. The save/load bodies
    are stubs that write/read a sentinel file so tests can assert the route
    plumbing without a real model; the round-trip suite below exercises the
    REAL ``MemoryAwarePrefixCache.save_to_disk`` / ``load_from_disk`` instead.
    """

    def __init__(
        self,
        *,
        entries: int = 0,
        current_memory: int = 0,
        prefix_cache: bool = True,
        kv_cache_dtype: str = "bf16",
        use_paged_cache: bool = False,
        kv_cache_turboquant: bool = False,
        load_returns: int = 0,
    ):
        cache = (
            _FakeCache(entries=entries, current_memory=current_memory)
            if prefix_cache
            else None
        )
        self.scheduler = SimpleNamespace(
            memory_aware_cache=cache,
            config=SimpleNamespace(
                kv_cache_dtype=kv_cache_dtype,
                use_paged_cache=use_paged_cache,
                kv_cache_turboquant=kv_cache_turboquant,
            ),
        )
        self.config = SimpleNamespace(model_name="test-model", model_path=None)
        self._load_returns = load_returns
        self.saved_to: str | None = None
        self.loaded_from: str | None = None

    def save_cache_to_disk(self, cache_dir: str, should_abort=None) -> bool:
        self.saved_to = cache_dir
        return bool(self.scheduler.memory_aware_cache and self._entry_count())

    def load_cache_from_disk(self, cache_dir: str) -> int:
        self.loaded_from = cache_dir
        return self._load_returns

    def _entry_count(self) -> int:
        cache = self.scheduler.memory_aware_cache
        return len(cache._entries) if cache is not None else 0


@pytest.fixture
def cache_client(monkeypatch, sandbox):
    """FastAPI TestClient with the cache router + auth + a fake engine.

    The engine now has to be faithful (unlike the pre-#476 stub, where the
    handlers never touched it): the export/import handlers read
    ``scheduler.memory_aware_cache`` + ``scheduler.config`` and call
    ``save_cache_to_disk`` / ``load_cache_from_disk``. ``_FakeEngine``
    provides exactly that surface. Tests that need a specific engine shape
    (no prefix cache, non-empty cache, a load that skips entries) install
    their own via ``get_config().engine = _FakeEngine(...)``.
    """
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.cache import router

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.engine = _FakeEngine()
    cfg.model_name = "test-model"

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(
        client=TestClient(app), sandbox=sandbox, cfg=cfg, FakeEngine=_FakeEngine
    )

    reset_config()


def _auth() -> dict:
    return {"Authorization": "Bearer test-secret"}


# ---------------------------------------------------------------------------
# protocol.resolve_cache_dir — unit tests at the helper level
# ---------------------------------------------------------------------------


def test_resolve_cache_dir_returns_sandbox_root_for_none(sandbox):
    """``None`` resolves to the sandbox root itself, which is created."""
    resolved = resolve_cache_dir(None)
    assert resolved == Path(sandbox).resolve()
    assert resolved.is_dir()


def test_resolve_cache_dir_relative_path_is_joined(sandbox):
    """Relative paths resolve under the sandbox root."""
    resolved = resolve_cache_dir("session-a")
    assert resolved == (Path(sandbox).resolve() / "session-a")


def test_resolve_cache_dir_rejects_dotdot_segment(sandbox):
    """``..`` in any segment is rejected before realpath even runs."""
    with pytest.raises(InvalidExportPathError, match="not allowed"):
        resolve_cache_dir("../etc/passwd")


def test_resolve_cache_dir_rejects_absolute_outside(sandbox):
    """An absolute path outside the sandbox is rejected by commonpath."""
    with pytest.raises(InvalidExportPathError, match="outside sandbox"):
        resolve_cache_dir("/tmp/anywhere-else")


def test_resolve_cache_dir_rejects_symlink_escape(sandbox):
    """A symlink whose realpath leaves the sandbox is rejected.

    ``os.path.realpath`` follows the link to ``outside_dir``, and the
    subsequent ``commonpath`` check sees the result is no longer a
    descendant of the sandbox root. Without realpath the literal path
    ``sandbox/escape/anything`` would look safe — this is the case
    that justifies the realpath step.
    """
    sandbox.mkdir(parents=True, exist_ok=True)
    outside = sandbox.parent / "outside_dir"
    outside.mkdir()
    link = sandbox / "escape"
    link.symlink_to(outside)

    with pytest.raises(InvalidExportPathError, match="outside sandbox"):
        resolve_cache_dir("escape/anything")


# ---------------------------------------------------------------------------
# protocol.Manifest — roundtrip + additive evolution
# ---------------------------------------------------------------------------


def test_manifest_roundtrip(tmp_path):
    """``write_manifest`` then ``read_manifest`` recovers every field."""
    original = Manifest(
        protocol_version=PROTOCOL_VERSION,
        model_id="mlx-community/Qwen3.5-9B-4bit",
        quantization="4bit",
        paged_cache=True,
        turboquant_kv=False,
        index_format_version=2,
        entries=42,
        total_bytes=12_345_678,
        rapid_mlx_version="0.7.29",
        created_at="2026-06-18T00:00:00Z",
    )
    write_manifest(tmp_path, original)
    recovered = read_manifest(tmp_path)
    assert recovered == original


def test_write_manifest_failed_rename_preserves_prior_manifest(tmp_path, monkeypatch):
    """A crash mid-rename must not corrupt the prior manifest.

    Atomic write idiom: write tmp → fsync → ``os.replace``. If ``replace``
    fails (here we monkeypatch it to ValueError), the prior manifest.json
    must be untouched and the tmp file cleaned up. Without this the
    next ``read_manifest`` would 400 against a truncated file even
    though a valid one existed before.
    """
    original = Manifest(model_id="qwen3.5-9b-4bit", entries=18)
    write_manifest(tmp_path, original)
    assert (tmp_path / "manifest.json").is_file()

    import vllm_mlx.cache.protocol as protocol_mod

    def _boom(*args, **kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(protocol_mod.os, "replace", _boom)

    with pytest.raises(OSError, match="simulated rename failure"):
        write_manifest(tmp_path, Manifest(model_id="will-not-land", entries=99))

    # Prior manifest intact: same fields, no truncation.
    recovered = read_manifest(tmp_path)
    assert recovered.model_id == "qwen3.5-9b-4bit"
    assert recovered.entries == 18

    # No temp file left behind.
    leaked = list(tmp_path.glob(".manifest-*.json"))
    assert leaked == [], f"leaked temp files: {leaked}"


def test_read_manifest_rejects_invalid_json(tmp_path):
    """Malformed JSON at the manifest path surfaces a typed exception.

    Without this branch the JSONDecodeError would propagate as a 500 in
    the routes — a caller-controlled bug masquerading as a server fault.
    """
    (tmp_path / "manifest.json").write_text("not even close to JSON {")
    with pytest.raises(MalformedManifestError, match="not valid JSON"):
        read_manifest(tmp_path)


def test_read_manifest_rejects_non_object_payload(tmp_path):
    """A JSON list at the manifest path is structurally malformed."""
    (tmp_path / "manifest.json").write_text('["this", "is", "a", "list"]')
    with pytest.raises(MalformedManifestError, match="JSON object"):
        read_manifest(tmp_path)


def test_manifest_from_dict_rejects_wrong_type(tmp_path):
    """A known field with the wrong JSON type → MalformedManifestError.

    Codex round-3 BLOCKING: ``"entries": "not-an-int"`` previously
    constructed the dataclass blindly, so a peer could serve a manifest
    that violated its own advertised schema and the route would return
    200 anyway. Now each known field's value is checked against its
    expected Python type at read time.
    """
    (tmp_path / "manifest.json").write_text(
        json.dumps({"protocol_version": "1", "entries": "not-an-int"})
    )
    with pytest.raises(MalformedManifestError, match="entries"):
        read_manifest(tmp_path)


def test_manifest_from_dict_rejects_bool_for_int_field(tmp_path):
    """``isinstance(True, int)`` is True in Python — but JSON ``true`` is
    clearly not the integer 1. The strict check rejects this."""
    (tmp_path / "manifest.json").write_text(
        json.dumps({"protocol_version": "1", "entries": True})
    )
    with pytest.raises(MalformedManifestError, match="entries"):
        read_manifest(tmp_path)


def test_manifest_from_dict_rejects_string_for_bool_field(tmp_path):
    """``"paged_cache": "yes"`` is structurally wrong even if intuitive."""
    (tmp_path / "manifest.json").write_text(
        json.dumps({"protocol_version": "1", "paged_cache": "yes"})
    )
    with pytest.raises(MalformedManifestError, match="paged_cache"):
        read_manifest(tmp_path)


def test_manifest_from_dict_drops_unknown_fields(tmp_path):
    """An older reader handling a newer writer's extra fields just ignores them."""
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": "x",
        "future_field_v2": "something the current reader doesn't know about",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    m = read_manifest(tmp_path)
    assert m.model_id == "x"
    assert m.protocol_version == PROTOCOL_VERSION


# ---------------------------------------------------------------------------
# auth — every route requires the bearer when --api-key is set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/v1/cache/export", {}),
        ("post", "/v1/cache/import", {"source": "anywhere"}),
        ("get", "/v1/cache/info", None),
    ],
)
def test_routes_require_auth(cache_client, method, path, body):
    """No bearer → 401 on every route."""
    client = cache_client.client
    if method == "post":
        resp = client.post(path, json=body)
    else:
        resp = client.get(path)
    assert resp.status_code == 401, resp.text


def test_info_requires_auth_even_with_valid_manifest(cache_client):
    """An unauthenticated ``GET /v1/cache/info`` against a path with a
    valid manifest must still return 401, not 200.

    Codex round-3 NIT: the parametrized auth check uses an empty default
    path, where auth-fires-before-handler is indistinguishable from
    auth-fires-after-handler by 404 vs 401 ordering. With a real manifest
    in place, a bypassed auth dependency would surface as a 200 — this
    test catches that exact regression.
    """
    _write_export_root(
        cache_client.sandbox,
        "valid",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="x", entries=1),
    )
    resp = cache_client.client.get("/v1/cache/info?path=valid")
    assert resp.status_code == 401


def test_routes_reject_wrong_bearer(cache_client):
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /v1/cache/export — engine-backed 200 after passing the sandbox check
# ---------------------------------------------------------------------------


def test_export_default_destination_returns_200(cache_client):
    """No destination → uses sandbox root, calls the engine, returns a
    caller-oriented summary. Resolved destination must NOT appear in the
    response body (sibling concern to F-180 — no operator-home-dir leak)."""
    # Fake engine with a small non-empty cache so bytes/entries are non-zero.
    cache_client.cfg.engine = cache_client.FakeEngine(
        entries=3, current_memory=2048, kv_cache_dtype="int8"
    )
    resp = cache_client.client.post("/v1/cache/export", json={}, headers=_auth())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["protocol_version"] == PROTOCOL_VERSION
    assert body["entries_exported"] == 3
    assert body["bytes_written"] == 2048
    assert body["quantization"] == "int8"
    assert body["manifest_path"] == "manifest.json"
    # The engine's save was actually invoked with the resolved destination.
    assert cache_client.cfg.engine.saved_to is not None

    # H-02/H-12: the resolved sandbox path must NOT ride the wire. Only the
    # caller-relative ``manifest.json`` filename is echoed.
    serialized = resp.text
    sandbox_real = str(Path(cache_client.sandbox).resolve())
    assert sandbox_real not in serialized


def test_export_empty_cache_returns_200_with_zero_entries(cache_client):
    """A prefix cache that's disabled (None) still exports a valid empty
    snapshot — 200 with entries_exported=0 and a manifest on disk."""
    cache_client.cfg.engine = cache_client.FakeEngine(prefix_cache=False)
    resp = cache_client.client.post(
        "/v1/cache/export", json={"destination": "empty-snap"}, headers=_auth()
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entries_exported"] == 0
    assert body["bytes_written"] == 0
    # Manifest was written under the resolved destination.
    manifest_root = Path(cache_client.sandbox).resolve() / "empty-snap"
    assert (manifest_root / "manifest.json").is_file()


def test_export_engine_not_loaded_returns_503(cache_client):
    """No model loaded (``cfg.engine is None``) → 503, matching the
    ``/v1/cache/clear`` idiom in routes.health."""
    cache_client.cfg.engine = None
    resp = cache_client.client.post("/v1/cache/export", json={}, headers=_auth())
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"] == "engine not loaded"


def test_export_over_max_bytes_returns_413(cache_client):
    """A cache whose in-memory footprint exceeds ``max_bytes`` is rejected
    with 413 BEFORE any write — the engine's save is never called."""
    engine = cache_client.FakeEngine(entries=5, current_memory=10_000)
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "too-big", "max_bytes": 4096},
        headers=_auth(),
    )
    assert resp.status_code == 413, resp.text
    # save_cache_to_disk must NOT have run (no write started).
    assert engine.saved_to is None
    # Nothing written to disk either.
    assert not (Path(cache_client.sandbox).resolve() / "too-big").exists()


def test_export_under_max_bytes_returns_200(cache_client):
    """Footprint at/under the cap exports normally."""
    engine = cache_client.FakeEngine(entries=2, current_memory=4096)
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "fits", "max_bytes": 4096},
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    assert engine.saved_to is not None


def test_export_rejects_path_traversal(cache_client):
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "../../../etc"},
        headers=_auth(),
    )
    assert resp.status_code == 403
    # H-02: 403 body is sanitized — error.code identifies the failure
    # mode, the caller-supplied path stays in server logs only.
    detail = resp.json()["detail"]
    assert detail["error"]["code"] == "sandbox_escape"
    assert detail["error"]["type"] == "invalid_request_error"


def test_export_rejects_absolute_outside(cache_client):
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "/tmp/escape-target"},
        headers=_auth(),
    )
    assert resp.status_code == 403


def test_export_rejects_invalid_max_bytes(cache_client):
    """pydantic catches the ge=1 violation as 422 before the handler runs."""
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"max_bytes": 0},
        headers=_auth(),
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /v1/cache/import — manifest mismatches surface as 409 before engine work
# ---------------------------------------------------------------------------


def _write_export_root(sandbox: Path, name: str, manifest: Manifest) -> Path:
    root = sandbox / name
    root.mkdir(parents=True, exist_ok=True)
    write_manifest(root, manifest)
    return root


def test_import_malformed_manifest_returns_400(cache_client):
    """Corrupt manifest.json at the source → 400, not 500.

    Without the dedicated mapping in ``_read_manifest_or_http``, the
    underlying ``json.JSONDecodeError`` would escape and FastAPI would
    surface it as an opaque 500 — hiding a caller-supplied bad blob
    inside a server-fault status. Codex blocking-finding regression.
    """
    bad = cache_client.sandbox / "corrupt"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{ not valid json")
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "corrupt"},
        headers=_auth(),
    )
    assert resp.status_code == 400
    assert "not valid JSON" in resp.json()["detail"]


def test_info_malformed_manifest_returns_400(cache_client):
    bad = cache_client.sandbox / "corrupt-info"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text('"a bare JSON string is not an object"')
    resp = cache_client.client.get(
        "/v1/cache/info?path=corrupt-info",
        headers=_auth(),
    )
    assert resp.status_code == 400
    assert "JSON object" in resp.json()["detail"]


def test_info_400_detail_does_not_leak_resolved_path(cache_client):
    """The 400 body must not include the server's resolved cache root.

    Codex round-3 NIT: leaking ``/Users/raullen/.cache/rapid-mlx/...`` to
    any bearer-token holder is unnecessary information disclosure.
    """
    bad = cache_client.sandbox / "leak-probe"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{ syntax error here")
    resp = cache_client.client.get(
        "/v1/cache/info?path=leak-probe",
        headers=_auth(),
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert str(cache_client.sandbox) not in detail
    assert "/" not in detail or "JSON" in detail  # may mention syntax but no path


def test_info_404_detail_does_not_leak_resolved_path(cache_client):
    """The 404 body must not include the server's resolved cache root."""
    (cache_client.sandbox / "no-such").mkdir(parents=True)
    resp = cache_client.client.get(
        "/v1/cache/info?path=no-such",
        headers=_auth(),
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert str(cache_client.sandbox) not in detail
    assert detail == "no manifest.json at the requested cache path"


def test_import_missing_manifest_returns_404(cache_client):
    """Source path exists but has no manifest.json."""
    (cache_client.sandbox / "no-manifest").mkdir(parents=True)
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "no-manifest"},
        headers=_auth(),
    )
    assert resp.status_code == 404


def test_import_protocol_version_mismatch_returns_409(cache_client):
    _write_export_root(
        cache_client.sandbox,
        "v999",
        Manifest(protocol_version="999", model_id="any"),
    )
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "v999", "expected_protocol_version": PROTOCOL_VERSION},
        headers=_auth(),
    )
    assert resp.status_code == 409
    assert "protocol_version" in resp.json()["detail"]


def test_import_model_id_mismatch_returns_409(cache_client):
    _write_export_root(
        cache_client.sandbox,
        "qwen",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="qwen3.5-9b-4bit"),
    )
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={
            "source": "qwen",
            "expected_model_id": "gpt-oss-20b-mxfp4-q8",
        },
        headers=_auth(),
    )
    assert resp.status_code == 409
    assert "model_id" in resp.json()["detail"]


def test_import_validated_request_returns_200(cache_client):
    """All wire checks pass → engine.load_cache_from_disk runs, 200 summary.

    ``entries_skipped`` derives from ``manifest.entries - entries_loaded``.
    Here the manifest claims 18 entries and the fake loader returns 15, so
    3 were dropped (per-entry validation) → entries_skipped == 3."""
    manifest = Manifest(
        protocol_version=PROTOCOL_VERSION,
        model_id="qwen3.5-9b-4bit",
        entries=18,
        total_bytes=4_096_000,
    )
    _write_export_root(cache_client.sandbox, "ready", manifest)
    engine = cache_client.FakeEngine(entries=2, current_memory=99, load_returns=15)
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={
            "source": "ready",
            "expected_model_id": "qwen3.5-9b-4bit",
            "merge_strategy": "replace",
        },
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["protocol_version"] == PROTOCOL_VERSION
    assert body["entries_loaded"] == 15
    assert body["entries_skipped"] == 3  # 18 claimed − 15 loaded
    assert body["bytes_loaded"] == 4_096_000
    # The engine's load actually ran, with the resolved source dir.
    assert engine.loaded_from is not None
    # merge_strategy="replace" cleared the in-memory cache before loading.
    assert engine.scheduler.memory_aware_cache.clear_calls == 1


def test_import_merge_does_not_clear_cache(cache_client):
    """``merge_strategy="merge"`` (default) must NOT clear the in-memory
    cache — existing entries are kept and the new blob is layered on top."""
    _write_export_root(
        cache_client.sandbox,
        "ready-merge",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="test-model", entries=4),
    )
    engine = cache_client.FakeEngine(entries=2, load_returns=4)
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "ready-merge", "merge_strategy": "merge"},
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    assert engine.scheduler.memory_aware_cache.clear_calls == 0
    assert resp.json()["entries_loaded"] == 4


def test_import_entries_skipped_floored_at_zero(cache_client):
    """If the loader returns MORE entries than the manifest claimed (a
    merge that hydrated pre-existing on-disk entries too), entries_skipped
    floors at 0 rather than going negative."""
    _write_export_root(
        cache_client.sandbox,
        "ready-over",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="test-model", entries=3),
    )
    engine = cache_client.FakeEngine(load_returns=5)
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "ready-over"},
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["entries_skipped"] == 0


def test_import_engine_not_loaded_returns_503(cache_client):
    """No model loaded → 503, AFTER the manifest checks pass (so a bad
    manifest still 409s even without an engine — order matters)."""
    _write_export_root(
        cache_client.sandbox,
        "ready-noeng",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="test-model"),
    )
    cache_client.cfg.engine = None
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "ready-noeng"},
        headers=_auth(),
    )
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"] == "engine not loaded"


def test_import_409_precedes_503(cache_client):
    """A protocol mismatch must 409 even when no engine is loaded — the
    manifest gate runs before the engine gate."""
    _write_export_root(
        cache_client.sandbox,
        "v999-noeng",
        Manifest(protocol_version="999", model_id="test-model"),
    )
    cache_client.cfg.engine = None
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "v999-noeng"},
        headers=_auth(),
    )
    assert resp.status_code == 409, resp.text


def test_import_rejects_path_traversal(cache_client):
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "../etc"},
        headers=_auth(),
    )
    assert resp.status_code == 403


def test_import_missing_source_returns_422(cache_client):
    """``source`` is required — pydantic rejects the missing field."""
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={},
        headers=_auth(),
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /v1/cache/info — fully implemented (the only non-stub endpoint)
# ---------------------------------------------------------------------------


def test_info_returns_manifest(cache_client):
    manifest = Manifest(
        protocol_version=PROTOCOL_VERSION,
        model_id="qwen3.5-9b-4bit",
        quantization="4bit",
        entries=18,
    )
    _write_export_root(cache_client.sandbox, "ready", manifest)
    resp = cache_client.client.get(
        "/v1/cache/info?path=ready",
        headers=_auth(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["protocol_version"] == PROTOCOL_VERSION
    assert body["manifest"]["model_id"] == "qwen3.5-9b-4bit"
    assert body["manifest"]["entries"] == 18


def test_info_missing_manifest_returns_404(cache_client):
    (cache_client.sandbox / "empty").mkdir(parents=True)
    resp = cache_client.client.get(
        "/v1/cache/info?path=empty",
        headers=_auth(),
    )
    assert resp.status_code == 404


def test_info_rejects_path_traversal(cache_client):
    resp = cache_client.client.get(
        "/v1/cache/info?path=../etc",
        headers=_auth(),
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# build_manifest_from_engine_state — reads model id + cache-config knobs
# ---------------------------------------------------------------------------


def test_build_manifest_from_engine_reads_config(monkeypatch):
    """The manifest builder pulls quantization/paged/turboquant off
    ``scheduler.config`` and entries/bytes off the live prefix cache."""
    from vllm_mlx.cache.protocol import build_manifest_from_engine_state
    from vllm_mlx.config import reset_config
    from vllm_mlx.memory_cache import _TOKENS_FORMAT_VERSION_IN_INDEX

    cfg = reset_config()
    cfg.model_name = "mlx-community/Qwen3.5-9B-4bit"
    try:
        engine = _FakeEngine(
            entries=7,
            current_memory=8192,
            kv_cache_dtype="int4",
            use_paged_cache=True,
            kv_cache_turboquant=True,
        )
        m = build_manifest_from_engine_state(engine)
        assert m.protocol_version == PROTOCOL_VERSION
        assert m.model_id == "mlx-community/Qwen3.5-9B-4bit"  # from server config
        assert m.quantization == "int4"
        assert m.paged_cache is True
        assert m.turboquant_kv is True
        assert m.index_format_version == _TOKENS_FORMAT_VERSION_IN_INDEX
        assert m.entries == 7
        assert m.total_bytes == 8192
        assert m.created_at.endswith("Z")  # ISO-8601 UTC
    finally:
        reset_config()


def test_build_manifest_prefix_cache_none_is_empty(monkeypatch):
    """A disabled prefix cache (None) yields entries/bytes = 0, no raise."""
    from vllm_mlx.cache.protocol import build_manifest_from_engine_state
    from vllm_mlx.config import reset_config

    cfg = reset_config()
    cfg.model_name = "test-model"
    try:
        engine = _FakeEngine(prefix_cache=False)
        m = build_manifest_from_engine_state(engine)
        assert m.entries == 0
        assert m.total_bytes == 0
        assert m.model_id == "test-model"
    finally:
        reset_config()


def test_build_manifest_falls_back_to_engine_model_id(monkeypatch):
    """When the server singleton has no ``model_name`` (embedded engine),
    fall back to the engine's own ``config.model_name``."""
    from vllm_mlx.cache.protocol import build_manifest_from_engine_state
    from vllm_mlx.config import reset_config

    reset_config()  # model_name stays None
    try:
        engine = _FakeEngine()
        engine.config.model_name = "embedded/model-id"
        m = build_manifest_from_engine_state(engine)
        assert m.model_id == "embedded/model-id"
    finally:
        reset_config()


# ---------------------------------------------------------------------------
# REAL round-trip — MemoryAwarePrefixCache.save_to_disk → load_from_disk
#
# This is the decisive test the #476 follow-up owes: a hand-built cache with
# BOTH a plain KVCache layer AND a recurrent-state ``ArraysCache`` layer must
# survive save→load. ``_cache_classes_compatible`` returns loadable=True for
# ArraysCache under a no-quant config, so the entry is NOT dropped as
# incompatible — this asserts that end to end against the real engine cache.
# ---------------------------------------------------------------------------


def _build_cache_with_arrays_layer():
    """A real ``MemoryAwarePrefixCache`` holding one entry whose layers are
    ``[KVCache, ArraysCache]``. The ArraysCache is injected directly into
    ``_entries`` (bypassing ``store()``'s non-trimmable-drop gate) because
    the point here is the SAVE/LOAD persistence path, not the reuse gate."""
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    from vllm_mlx.memory_cache import (
        MemoryAwarePrefixCache,
        MemoryCacheConfig,
        _CacheEntry,
    )

    cfg = MemoryCacheConfig(max_memory_percent=0.5)
    cache = MemoryAwarePrefixCache(model=object(), config=cfg)

    kv = KVCache()
    kv.update_and_fetch(mx.zeros((1, 2, 4, 8)), mx.ones((1, 2, 4, 8)))
    arrays = ArraysCache(size=1)
    arrays[0] = mx.ones((1, 3, 5))

    tokens = (10, 11, 12, 13)
    entry = _CacheEntry.create(list(tokens), [kv, arrays])
    with cache._lock:
        cache._entries[tokens] = entry
        cache._current_memory += entry.memory_bytes
    return cache, MemoryCacheConfig(max_memory_percent=0.5)


def test_arrays_cache_layer_is_loadable_under_no_quant():
    """Unit-level proof of the frozen-design claim: ArraysCache round-trips
    (the QuantizedKVCache/TurboQuantKVCache gates do NOT apply to it)."""
    from vllm_mlx.memory_cache import MemoryCacheConfig, _cache_classes_compatible

    ok, reason = _cache_classes_compatible(
        ["KVCache", "ArraysCache"], MemoryCacheConfig(max_memory_percent=0.5)
    )
    assert ok is True, reason
    assert reason == ""


def test_real_cache_roundtrip_survives_arrays_layer(tmp_path):
    """Save a real cache with a KVCache + ArraysCache entry, load it into a
    fresh cache, and assert the ArraysCache layer survived (not dropped as
    incompatible). This is the correctness backbone the HTTP handlers ride."""
    import json

    cache, load_cfg = _build_cache_with_arrays_layer()
    target = str(tmp_path / "snap")

    assert cache.save_to_disk(target) is True
    assert len(cache._entries) == 1

    # The on-disk index records the REAL per-layer class names — proving
    # the recurrent-state layer is persisted, not silently converted.
    with open(Path(target) / "index.json") as f:
        index = json.load(f)
    persisted_types = index["entries"][0]["cache_types"]
    assert "ArraysCache" in persisted_types, persisted_types
    assert "KVCache" in persisted_types, persisted_types

    # Load into a FRESH cache under a default (no-quant) config.
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache

    fresh = MemoryAwarePrefixCache(model=object(), config=load_cfg)
    loaded = fresh.load_from_disk(target)
    assert loaded == 1, "the ArraysCache entry must NOT be dropped on load"
    assert len(fresh._entries) == 1

    # The reconstructed entry still carries the ArraysCache layer.
    (entry,) = fresh._entries.values()
    layer_classes = [type(layer).__name__ for layer in entry.cache]
    assert "ArraysCache" in layer_classes, layer_classes
    assert "KVCache" in layer_classes, layer_classes


def test_http_export_import_roundtrip_end_to_end(monkeypatch, sandbox):
    """Full HTTP round-trip against the REAL engine cache primitives.

    Wires a live ``MemoryAwarePrefixCache`` (with a KVCache + ArraysCache
    entry) into a fake engine whose ``save_cache_to_disk`` /
    ``load_cache_from_disk`` delegate to the real cache methods, then drives
    ``POST /v1/cache/export`` followed by ``POST /v1/cache/import`` over the
    TestClient. Asserts the ArraysCache entry survives the full wire path."""
    from vllm_mlx.config import reset_config
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache
    from vllm_mlx.routes.cache import router

    export_root = sandbox
    monkeypatch.setenv("RAPID_MLX_CACHE_EXPORT_DIR", str(export_root))

    src_cache, load_cfg = _build_cache_with_arrays_layer()
    dst_cache = MemoryAwarePrefixCache(model=object(), config=load_cfg)

    # The route runs save/load in an ``anyio.to_thread`` worker. MLX arrays
    # are stream-bound, so a worker thread that touches main-thread arrays
    # raises "There is no Stream(gpu, 0) in current thread" — exactly why the
    # REAL ``EngineCore`` routes these through its mlx-step thread which owns
    # the generation_stream (engine_core.py:73 binds
    # ``mx.default_stream(mx.default_device())``). We mimic that here: eval
    # the arrays on the main thread, then wrap the worker-side cache call in
    # the shared device default-stream context. This is test-harness plumbing
    # standing in for the step thread — production doesn't need it because
    # ``_run_on_step_thread`` already provides the stream.
    import mlx.core as mx

    for entry in src_cache._entries.values():
        for layer in entry.cache:
            state = getattr(layer, "state", None)
            if state is not None:
                mx.eval(state)

    def _on_default_stream(fn, *args):
        with mx.stream(mx.default_stream(mx.default_device())):
            return fn(*args)

    class _RealEngine:
        """Engine whose save/load delegate to real cache methods. The
        exporter uses ``src_cache``; the importer hydrates ``dst_cache``."""

        def __init__(self):
            self.scheduler = SimpleNamespace(
                memory_aware_cache=src_cache,
                config=SimpleNamespace(
                    kv_cache_dtype="bf16",
                    use_paged_cache=False,
                    kv_cache_turboquant=False,
                ),
            )
            self.config = SimpleNamespace(model_name="test-model", model_path=None)

        def save_cache_to_disk(self, cache_dir, should_abort=None):
            return _on_default_stream(src_cache.save_to_disk, cache_dir)

        def load_cache_from_disk(self, cache_dir):
            # Importer loads into the DESTINATION cache — the real
            # load_from_disk path (compat gate included).
            return _on_default_stream(dst_cache.load_from_disk, cache_dir)

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.model_name = "test-model"
    cfg.engine = _RealEngine()

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    try:
        # Export → 200, entries_exported == 1, ArraysCache persisted.
        exp = client.post(
            "/v1/cache/export",
            json={"destination": "rt"},
            headers=_auth(),
        )
        assert exp.status_code == 200, exp.text
        assert exp.json()["entries_exported"] == 1

        # Import the same blob back → 200, entries_loaded == 1.
        imp = client.post(
            "/v1/cache/import",
            json={"source": "rt", "expected_model_id": "test-model"},
            headers=_auth(),
        )
        assert imp.status_code == 200, imp.text
        assert imp.json()["entries_loaded"] == 1
        assert imp.json()["entries_skipped"] == 0

        # The destination cache now holds the entry, ArraysCache intact.
        assert len(dst_cache._entries) == 1
        (entry,) = dst_cache._entries.values()
        layer_classes = [type(layer).__name__ for layer in entry.cache]
        assert "ArraysCache" in layer_classes, layer_classes
    finally:
        reset_config()
