# SPDX-License-Identifier: Apache-2.0
"""KV cache export/import HTTP API (issue #476).

This module defines the wire surface (request/response shapes, auth,
path-whitelist, manifest validation) AND the engine-level save/load body:

* ``POST /v1/cache/export`` — validates the request, resolves the
  destination under the sandbox, then calls ``EngineCore.save_cache_to_disk``
  to snapshot the in-memory prefix cache and writes a ``manifest.json``
  alongside it. Returns 200 with a byte/entry summary.
* ``POST /v1/cache/import`` — validates the request, resolves the source
  under the sandbox, reads + checks the manifest against caller
  expectations (protocol_version / model_id mismatch → 409), then calls
  ``EngineCore.load_cache_from_disk`` to hydrate the prefix cache. With
  ``merge_strategy="replace"`` the in-memory cache is cleared first.
* ``GET /v1/cache/info`` — reads the manifest at a whitelisted path and
  returns it. Lets a peer instance (or oai-mlx) GC / inspect an export
  root without round-tripping a full import. H-12: response carries
  ``protocol_version`` + ``manifest`` only — the resolved sandbox root
  stays in the server log, never on the wire.

Both engine-touching handlers 503 when no model is loaded (``cfg.engine
is None``), matching ``vllm_mlx.routes.health``'s ``clear_cache`` idiom.

Quiesce (out of MVP): the issue's ``wait_for_quiesce_seconds`` — draining
in-flight decode before snapshotting — is deliberately NOT implemented
here. ``EngineCore.save_cache_to_disk`` already serializes on the mlx-step
worker thread (that's why the KV arrays are materializable at all), so a
snapshot taken while a request is mid-decode captures a consistent
per-entry view; a torn cross-entry snapshot is possible only under
concurrent store()/evict, which is acceptable for the MVP (the importer
validates each entry independently and drops any that fail). A real
quiesce controller is tracked in #476's follow-up list.

Auth follows ``vllm_mlx.routes.health``'s ``router``: the bearer key is
enforced when ``--api-key`` is set, no new header is invented.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import anyio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..cache.protocol import (
    PROTOCOL_VERSION,
    InvalidExportPathError,  # noqa: F401 — re-exported for _resolve_or_400 callers
    MalformedManifestError,  # noqa: F401 — used via _read_manifest_or_http
    ManifestMismatchError,
    ManifestNotFoundError,  # noqa: F401 — used via _read_manifest_or_http
    build_manifest_from_engine_state,
    read_manifest,
    resolve_cache_dir,
    write_manifest,
)
from ..config import get_config
from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

# H-02/H-12: the resolved sandbox path never rides the wire. On the
# engine-touching paths we log the destination/source server-side and
# return only caller-oriented counters. ``_ENGINE_NOT_LOADED_DETAIL``
# keeps the same sanitized-envelope discipline the stub established for
# its 403 body — no path, no manifest excerpt.
_ENGINE_NOT_LOADED_DETAIL = "engine not loaded"

# H-02: sandbox-escape 403 envelope. The underlying
# ``InvalidExportPathError`` carries the caller-supplied path AND the
# fully resolved sandbox root (``/Users/<username>/.cache/rapid-mlx/
# cache_exports``). Echoing either to an unauthenticated caller leaks
# the operator's home dir + username. Same treatment as the #756 501
# envelope: generic wire message, full detail goes to the server log.
_SANDBOX_ESCAPE_MSG = "destination must resolve under the cache-export sandbox"
_SANDBOX_ESCAPE_DETAIL = {
    "error": {
        "message": _SANDBOX_ESCAPE_MSG,
        "type": "invalid_request_error",
        "code": "sandbox_escape",
    }
}


router = APIRouter(
    prefix="/v1/cache",
    tags=["cache"],
    dependencies=[Depends(verify_api_key)],
)


class ExportRequest(BaseModel):
    """Request body for ``POST /v1/cache/export``."""

    destination: str | None = Field(
        default=None,
        description=(
            "Path under RAPID_MLX_CACHE_EXPORT_DIR (default "
            "~/.cache/rapid-mlx/cache_exports/). May be relative (resolved "
            "against the sandbox root) or absolute (must resolve inside "
            "the sandbox). Omit to use the sandbox root itself."
        ),
    )
    max_bytes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional cap on the exported blob size. Checked BEFORE any "
            "write: if the prefix cache's current in-memory footprint "
            "(``_current_memory``) exceeds this, the export is rejected "
            "with 413 and nothing is written. MVP is all-or-nothing — no "
            "partial-entry eviction to fit under the cap."
        ),
    )


class ImportRequest(BaseModel):
    """Request body for ``POST /v1/cache/import``."""

    source: str = Field(
        ...,
        description=(
            "Path to an export root containing manifest.json + index.json. "
            "Resolved under the export sandbox (see ExportRequest.destination)."
        ),
    )
    expected_protocol_version: str = Field(
        default=PROTOCOL_VERSION,
        description=(
            "Manifest protocol version the caller expects. Mismatch → 409. "
            f"Current: {PROTOCOL_VERSION!r}."
        ),
    )
    expected_model_id: str | None = Field(
        default=None,
        description=(
            "If set, manifest.model_id must match exactly. Mismatch → 409. "
            "Omit to skip the model-identity check (importer accepts any "
            "model — only use when you're sure the engine matches)."
        ),
    )
    merge_strategy: Literal["replace", "merge"] = Field(
        default="merge",
        description=(
            "'merge' keeps existing entries and adds new ones (token-tuple "
            "key collisions resolved by the engine's ``store``). 'replace' "
            "calls ``MemoryAwarePrefixCache.clear()`` before loading so the "
            "imported blob is the only thing in the cache."
        ),
    )


class ExportResponse(BaseModel):
    """200 body for ``POST /v1/cache/export`` — caller-oriented summary.

    H-02/H-12: NO resolved sandbox path is echoed. ``manifest_path`` is the
    caller-relative filename (``manifest.json``), not the absolute on-disk
    location — a peer that wrote to ``destination="session-a"`` already
    knows where it asked to write; it does not need (and must not learn)
    the operator's ``$HOME``-rooted expansion.
    """

    protocol_version: str
    entries_exported: int
    bytes_written: int
    model_id: str
    quantization: str
    paged_cache: bool
    turboquant_kv: bool
    manifest_path: str


class ImportResponse(BaseModel):
    """200 body for ``POST /v1/cache/import`` — caller-oriented summary."""

    protocol_version: str
    entries_loaded: int
    entries_skipped: int
    bytes_loaded: int


def _resolve_or_400(caller_path: str | None) -> Path:
    """Wrap ``resolve_cache_dir`` so path violations surface as 403.

    H-02: ``InvalidExportPathError`` carries the caller-supplied path AND
    the resolved sandbox root (which expands to ``/Users/<USERNAME>/.cache
    /rapid-mlx/cache_exports`` on macOS). Both stay in the server log via
    ``logger.warning`` — only the sanitized envelope reaches the wire.
    """
    try:
        return resolve_cache_dir(caller_path)
    except InvalidExportPathError as exc:
        # 403 (not 400) because the request is well-formed JSON — what's
        # rejected is the *authorization* to write/read outside the sandbox.
        logger.warning(
            "cache: sandbox-escape rejected (caller_path=%r): %s",
            caller_path,
            exc,
        )
        raise HTTPException(status_code=403, detail=_SANDBOX_ESCAPE_DETAIL) from exc


def _read_manifest_or_http(root: Path):
    """Wrap ``read_manifest`` so missing/malformed surface as 404/400.

    Without this, a peer-written corrupt ``manifest.json`` would escape
    as a JSONDecodeError → FastAPI 500, hiding a caller-controlled bug
    inside an opaque server error. Mapping the three failure modes
    distinctly is what makes the contract usable from a client.

    Response details are caller-oriented — the fully resolved local
    filesystem path stays in the server log only, not in the HTTP body
    where a bearer-token holder could harvest the export-root layout.
    """
    try:
        return read_manifest(root)
    except ManifestNotFoundError as exc:
        logger.info("cache: manifest not found at %s", root)
        raise HTTPException(
            status_code=404,
            detail="no manifest.json at the requested cache path",
        ) from exc
    except MalformedManifestError as exc:
        # ``str(exc)`` is already path-free (see protocol.read_manifest).
        # It carries the structural reason — "not valid JSON: ...",
        # "must decode to a JSON object, got list", "manifest field
        # 'entries': expected int, got str" — which the client needs to
        # fix its own payload. The resolved path only lands in server logs.
        logger.warning("cache: malformed manifest at %s: %s", root, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _engine_or_503():
    """Return the loaded engine or raise 503 (matches ``health.clear_cache``).

    The engine-touching routes are meaningless before a model is loaded —
    there is no prefix cache to snapshot / hydrate. Same 503 idiom the rest
    of the cache-management surface uses so operators get one consistent
    "engine not loaded" signal across ``/v1/cache/clear``, ``/v1/cache/stats``
    and this pair.
    """
    engine = get_config().engine
    if engine is None:
        raise HTTPException(status_code=503, detail=_ENGINE_NOT_LOADED_DETAIL)
    return engine


def _prefix_cache(engine):
    """The engine's in-memory ``MemoryAwarePrefixCache`` or ``None``.

    ``None`` when the prefix cache is disabled (``--disable-prefix-cache``)
    or the scheduler hasn't finished building it. Callers treat ``None`` as
    "empty snapshot" (0 entries / 0 bytes) rather than an error — an export
    of an empty cache is a valid no-op that still writes a manifest.
    """
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is None:
        return None
    return getattr(scheduler, "memory_aware_cache", None)


@router.post("/export", response_model=ExportResponse)
async def export_cache(req: ExportRequest):
    """Export the engine's KV prefix cache to disk under the sandbox root.

    Flow: resolve the destination against the path whitelist (403 on
    escape) → require a loaded engine (503) → enforce ``max_bytes`` from the
    cheap in-memory footprint BEFORE writing (413) → snapshot via
    ``EngineCore.save_cache_to_disk`` → write ``manifest.json`` alongside.

    An empty prefix cache (disabled or 0 entries) is a valid export: it
    returns 200 with ``entries_exported=0`` and still writes a manifest so
    a peer can inspect the (empty) blob's provenance.

    H-02/H-12: the resolved ``destination`` is logged server-side only; the
    200 body carries counters + the caller-relative manifest filename, never
    the ``$HOME``-rooted absolute path.
    """
    destination = _resolve_or_400(req.destination)
    engine = _engine_or_503()
    cache = _prefix_cache(engine)

    # Cheap pre-write size gate. ``_current_memory`` is the live ledger the
    # cache maintains on every store/evict — reading it costs nothing and
    # lets us reject an over-cap export before touching the disk (the issue's
    # H-04 "never start a write you can't afford" concern).
    current_bytes = 0
    if cache is not None:
        current_bytes = int(getattr(cache, "_current_memory", 0))
    if req.max_bytes is not None and current_bytes > req.max_bytes:
        logger.info(
            "cache/export: rejected — cache footprint %d B exceeds max_bytes %d B "
            "(destination=%s)",
            current_bytes,
            req.max_bytes,
            destination,
        )
        raise HTTPException(
            status_code=413,
            detail=(
                f"cache footprint {current_bytes} bytes exceeds max_bytes "
                f"{req.max_bytes}"
            ),
        )

    # Snapshot to disk. Routed through the mlx-step worker thread inside the
    # engine (that's where the KV arrays are materializable). Returns True if
    # at least one entry committed; False for an empty / no-op cache — either
    # way the manifest below records the outcome. Run in a threadpool so the
    # asyncio loop isn't blocked by the (potentially multi-GB) write.
    saved = await anyio.to_thread.run_sync(engine.save_cache_to_disk, str(destination))

    manifest = build_manifest_from_engine_state(engine)
    write_manifest(destination, manifest)

    logger.info(
        "cache/export: wrote %d entries (%d B, saved=%s) to destination=%s",
        manifest.entries,
        manifest.total_bytes,
        saved,
        destination,
    )
    return ExportResponse(
        protocol_version=PROTOCOL_VERSION,
        entries_exported=manifest.entries,
        bytes_written=manifest.total_bytes,
        model_id=manifest.model_id,
        quantization=manifest.quantization,
        paged_cache=manifest.paged_cache,
        turboquant_kv=manifest.turboquant_kv,
        manifest_path="manifest.json",
    )


@router.post("/import", response_model=ImportResponse)
async def import_cache(req: ImportRequest):
    """Import a peer instance's export into the local engine.

    Flow: resolve the source under the sandbox (403 on escape) → read the
    manifest (404 missing / 400 malformed) → reject protocol-version or
    model-id mismatch (409) → require a loaded engine (503) → optionally
    clear the in-memory cache (``merge_strategy="replace"``) → hydrate via
    ``EngineCore.load_cache_from_disk``.

    ``entries_skipped`` is ``manifest.entries - entries_loaded`` floored at
    0: the loader drops entries that fail per-entry validation (truncated
    safetensors, cache-type incompatible under the current quant config —
    see ``MemoryAwarePrefixCache.load_from_disk``), so a caller can tell a
    partial import from a clean one without re-reading the blob.
    """
    source = _resolve_or_400(req.source)
    manifest = _read_manifest_or_http(source)

    if manifest.protocol_version != req.expected_protocol_version:
        raise HTTPException(
            status_code=409,
            detail=str(
                ManifestMismatchError(
                    "protocol_version",
                    req.expected_protocol_version,
                    manifest.protocol_version,
                )
            ),
        )

    if req.expected_model_id is not None and manifest.model_id != req.expected_model_id:
        raise HTTPException(
            status_code=409,
            detail=str(
                ManifestMismatchError(
                    "model_id", req.expected_model_id, manifest.model_id
                )
            ),
        )

    engine = _engine_or_503()

    # merge_strategy="replace" — drop the current in-memory prefix cache so
    # the imported blob is the ONLY thing present. ``MemoryAwarePrefixCache
    # .clear()`` is the smallest correct primitive: it empties ``_entries`` /
    # ``_sorted_keys`` / the radix index and zeroes ``_current_memory`` while
    # carrying the monotonic Prometheus counters over (load_skipped /
    # save_drift_drops) — exactly what a "replace the contents" reset wants.
    # We deliberately do NOT call ``scheduler.deep_reset()``: that also tears
    # down the BatchGenerator and paged manager, which would abort in-flight
    # requests — far heavier than clearing the prefix cache.
    if req.merge_strategy == "replace":
        cache = _prefix_cache(engine)
        if cache is not None:
            cache.clear()
            logger.info("cache/import: cleared in-memory prefix cache (replace)")

    # Hydrate from disk. Also routed through the mlx-step worker thread so
    # the loaded arrays are tagged with the right generation_stream. Run in
    # a threadpool for the same reason as export.
    entries_loaded = await anyio.to_thread.run_sync(
        engine.load_cache_from_disk, str(source)
    )

    entries_skipped = max(0, manifest.entries - entries_loaded)
    logger.info(
        "cache/import: loaded %d/%d entries (skipped=%s, merge=%s) from source=%s",
        entries_loaded,
        manifest.entries,
        entries_skipped,
        req.merge_strategy,
        source,
    )
    return ImportResponse(
        protocol_version=PROTOCOL_VERSION,
        entries_loaded=entries_loaded,
        entries_skipped=entries_skipped,
        bytes_loaded=manifest.total_bytes,
    )


@router.get("/info")
async def cache_info(path: str | None = None):
    """Read the manifest at a whitelisted export root.

    Returns the manifest dict so callers (peer instances, oai-mlx, ops
    tooling) can GC / route / version-gate without paying a full import.
    Path resolution follows the same sandbox rules as export/import.

    H-12: pre-fix this handler echoed the resolved sandbox root back to
    the caller in a top-level ``"path"`` field. ``str(root)`` expands to
    ``/Users/<USERNAME>/.cache/rapid-mlx/cache_exports/<sub>`` on macOS
    — same operator home-dir / username disclosure that H-02 fixed on
    the 403 envelope. Same treatment here: keep the resolved root in
    the server log only, omit it from the wire envelope. Callers that
    need to dedupe by location already have the request-side ``path``
    they supplied.
    """
    root = _resolve_or_400(path)
    manifest = _read_manifest_or_http(root)

    # Codex r1 follow-up: log at DEBUG (not INFO) so the resolved root
    # only lands in operator logs when the operator explicitly opts in
    # (RAPID_MLX_LOG_LEVEL=DEBUG or equivalent). Routine 200 traffic
    # carries no path on the wire AND no path in the default log stream
    # — but the breadcrumb is still there for ops who need to debug a
    # peer-sync issue. Sibling concern: H-02's logger.warning on the
    # 403 path is fine because that's an anomaly worth recording at
    # default verbosity, whereas every successful info read shouldn't
    # rewrite the sandbox path into the rolling log.
    logger.debug(
        "cache/info: resolved root=%s model_id=%s entries=%s",
        root,
        manifest.model_id,
        manifest.entries,
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "manifest": manifest.to_dict(),
    }
