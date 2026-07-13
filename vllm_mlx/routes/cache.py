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

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Literal

import anyio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..cache.protocol import (
    MANIFEST_FILENAME,
    PROTOCOL_VERSION,
    InvalidExportPathError,  # noqa: F401 — re-exported for _resolve_or_400 callers
    MalformedManifestError,  # noqa: F401 — used via _read_manifest_or_http
    ManifestMismatchError,
    ManifestNotFoundError,  # noqa: F401 — used via _read_manifest_or_http
    build_manifest_from_engine_state,
    default_export_root,
    read_manifest,
    resolve_cache_dir,
    resolve_engine_model_id,
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


# #1100 codex round 3 (findings #2/#5): serialize the whole export
# transaction (save → manifest → committed-size gate → publish/discard) PER
# RESOLVED DESTINATION. Two concurrent exports to the same path could
# otherwise interleave so one request writes a manifest for another's
# snapshot, or the over-cap cleanup of one deletes the newer snapshot the
# other just committed. A per-destination asyncio.Lock makes the destination
# a single-writer resource. Keyed on the resolved absolute path string so
# distinct destinations still run concurrently. (The engine already
# serializes the KV materialization on its mlx-step thread; this guards the
# route-side filesystem transaction around it.)
_export_dest_locks: dict[str, asyncio.Lock] = {}
_export_locks_guard = asyncio.Lock()


async def _acquire_dest_lock(destination: Path) -> asyncio.Lock:
    """Return (creating if needed) the per-destination export lock."""
    key = str(destination)
    async with _export_locks_guard:
        lock = _export_dest_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _export_dest_locks[key] = lock
        return lock


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
            "Optional cap on the exported blob's COMMITTED ON-DISK size "
            "(the sum of real file sizes an importing peer will read: "
            "entry_*.safetensors + entry_*_tokens.bin + index.json + "
            "manifest.json, including safetensors headers / alignment / "
            "serialization overhead — NOT just the logical KV byte count). "
            "Enforced with 413 in TWO stages: a cheap pre-write check "
            "against the live in-memory footprint (rejects before touching "
            "disk), AND a precise post-write check against the actual "
            "committed on-disk directory size (catches cache growth that "
            "raced the snapshot AND the on-disk overhead the logical count "
            "excludes — the over-cap blob is then discarded from disk). MVP "
            "is all-or-nothing — no partial-entry eviction to fit under the "
            "cap."
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
            "Optional ADDITIONAL caller-side assertion on manifest.model_id "
            "(exact match; mismatch → 409). NOTE: the server ALWAYS rejects a "
            "manifest whose model_id differs from the model it loaded — "
            "omitting this does NOT disable identity checking, it only skips "
            "the extra caller-side assertion. Set it to pin a specific model "
            "id independently of the server's loaded model."
        ),
    )
    merge_strategy: Literal["replace", "merge"] = Field(
        default="merge",
        description=(
            "'merge' keeps existing entries and adds new ones (token-tuple "
            "key collisions resolved by the engine's ``store``). 'replace' "
            "clears the in-memory cache ATOMICALLY inside the step-thread "
            "load — only after the source index is validated — so the "
            "imported blob is the only thing in the cache, and a corrupt/"
            "missing source leaves the existing cache intact."
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

    Delegates to ``runtime.cache._resolve_memory_aware_cache`` so BOTH engine
    shapes resolve: a bare ``EngineCore`` (``engine.scheduler``) AND the
    production ``BatchedEngine`` (``engine._engine.engine.scheduler``). A
    naive ``getattr(engine, "scheduler")`` here returned None under
    BatchedEngine, which silently disarmed the ``max_bytes`` 413 gate (read
    ``_current_memory``=0) and the ``merge_strategy="replace"`` clear (cache
    None) — the real-hardware smoke bug this fix closes.

    ``None`` still means "no prefix cache" (disabled via
    ``--disable-prefix-cache`` or a genuinely foreign engine); callers treat
    it as an empty snapshot (0 entries / 0 bytes), not an error.
    """
    from ..runtime.cache import _resolve_memory_aware_cache

    return _resolve_memory_aware_cache(engine)


def _committed_dir_size(destination: Path) -> int:
    """Sum the actual on-disk byte size of the committed export blob.

    #1100 BLOCKING-4: ``max_bytes`` is documented as capping the committed
    ON-DISK blob size, but the old post-write gate summed only logical
    ``memory_bytes`` (``manifest.total_bytes``) — excluding the tokens.bin
    sidecars, index.json, manifest.json, and every safetensors header /
    alignment / serialization overhead. An accepted export could therefore
    exceed the cap on disk. This walks the export directory and sums real
    file sizes (``os.stat``) so the cap is enforced against what a peer
    actually reads.

    When ``destination`` IS the sandbox root (the ``destination=None``
    export shape), only the known ``save_to_disk`` + manifest artifacts are
    counted — unrelated files that happen to share the root are not this
    export's footprint and must not be charged against its cap.
    """
    root = default_export_root()
    total = 0
    try:
        if destination != root:
            # Dedicated export sub-dir: every file under it is this export.
            for dirpath, _dirnames, filenames in os.walk(destination):
                for name in filenames:
                    try:
                        total += os.stat(os.path.join(dirpath, name)).st_size
                    except OSError:  # pragma: no cover — racing unlink
                        pass
            return total
        # destination is the sandbox root — count only this export's blobs.
        for name in os.listdir(destination):
            is_blob = name in ("index.json", MANIFEST_FILENAME) or (
                name.startswith("entry_")
                and (name.endswith(".safetensors") or name.endswith("_tokens.bin"))
            )
            if is_blob:
                try:
                    total += os.stat(destination / name).st_size
                except OSError:  # pragma: no cover — racing unlink
                    pass
    except OSError:  # pragma: no cover — destination vanished
        pass
    return total


# The two files whose PRESENCE makes an export importable: ``manifest.json``
# (the import route reads it first — no manifest → 404) and ``index.json``
# (``load_from_disk`` needs it — no index → nothing loads). If cleanup can't
# delete a blob, quarantining THESE two (rename to ``.rejected``) is enough
# to guarantee the import path refuses it. The entry_* payload files alone
# are inert without an index pointing at them.
_IMPORT_CRITICAL_NAMES = (MANIFEST_FILENAME, "index.json")


def _sweep_staging_dirs(destination: Path) -> None:
    """Remove the ``.new``/``.old`` staging dirs a FAILED save may have left.

    #1100 codex round 3 (#2): ``save_to_disk`` writes into ``<dest>.new`` and
    only atomically renames it onto ``<dest>`` on success. A non-committing
    save leaves those sibling staging dirs behind but does NOT touch the
    published ``<dest>`` (which may hold a previously-valid snapshot). So on a
    failed save we sweep ONLY the staging siblings — never ``destination``
    itself — to avoid destroying a prior good snapshot the failure never
    modified. Best-effort: a leftover staging dir is an operator-log concern.
    """
    base = str(destination).rstrip(os.sep)
    for suffix in (".new", ".old"):
        staging = Path(base + suffix)
        if staging.exists():
            try:
                shutil.rmtree(staging, ignore_errors=True)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "cache/export: could not sweep staging dir %s: %s",
                    staging,
                    exc,
                )


class _ExportDiscardError(RuntimeError):
    """Raised when a rejected export blob could NOT be made non-importable.

    #1100 BLOCKING-5: a 413/500 reject MUST guarantee the blob is not
    importable. If deletion fails AND quarantine (rename the import-critical
    files to ``.rejected``) also fails, we cannot honor that invariant —
    surface a 500 rather than falsely claim the blob was discarded.
    """


def _quarantine_import_critical(destination: Path) -> bool:
    """Rename import-critical files to ``.rejected`` so import refuses them.

    Returns True if, after the attempt, NEITHER ``manifest.json`` nor
    ``index.json`` remains at ``destination`` (the blob can no longer be
    imported). Returns False if either still exists (quarantine failed).
    """
    for name in _IMPORT_CRITICAL_NAMES:
        src = destination / name
        if not src.exists():
            continue
        try:
            os.replace(src, destination / f"{name}.rejected")
        except OSError as exc:  # pragma: no cover — exercised via monkeypatch
            logger.error("cache/export: quarantine rename failed for %s: %s", src, exc)
    # Verify the invariant: no import-critical file remains.
    return not any((destination / n).exists() for n in _IMPORT_CRITICAL_NAMES)


def _blob_artifacts(destination: Path) -> list[str]:
    """Names under ``destination`` that belong to a save_to_disk export."""
    names = []
    for name in os.listdir(destination):
        is_blob = name in ("index.json", MANIFEST_FILENAME) or (
            name.startswith("entry_")
            and (name.endswith(".safetensors") or name.endswith("_tokens.bin"))
        )
        if is_blob:
            names.append(name)
    return names


def _discard_export(destination: Path) -> None:
    """Delete a just-written export blob after a post-write reject, and VERIFY
    it is actually gone (#1100 BLOCKING-2 max_bytes race + BLOCKING-5
    cleanup-verification).

    ``save_to_disk`` commits the blob (index.json + entry_* files) into
    ``destination`` via an atomic rename before the route can enforce the
    exact committed size / detect a failed save. When we reject we must not
    leave the blob on disk for a peer to import.

    The round-1 helper used ``shutil.rmtree(..., ignore_errors=True)`` and
    returned immediately, so a permission / filesystem failure silently left
    the rejected blob on disk while the API claimed it was discarded — a peer
    could then import it. This version:

    1. Attempts wholesale removal (dedicated sub-dir) or per-artifact unlink
       (sandbox-root export shape).
    2. VERIFIES the import-critical files (manifest.json + index.json) are
       gone. If any survives, QUARANTINES it (rename to ``.rejected`` so the
       import path — which reads ``manifest.json`` / ``index.json`` by exact
       name — refuses it).
    3. If even quarantine can't remove them, raises ``_ExportDiscardError``
       so the caller returns a 500 instead of falsely claiming the blob was
       discarded. The invariant a 413/500 reject upholds: the blob is NOT
       importable.

    When ``destination`` IS the sandbox root itself (the ``destination=None``
    export shape), only the known ``save_to_disk`` artifacts are touched —
    the root directory and anything else in it are preserved.
    """
    root = default_export_root()
    try:
        if destination != root:
            shutil.rmtree(destination, ignore_errors=True)
        else:
            # destination is the sandbox root — unlink only the blob artifacts.
            for name in _blob_artifacts(destination):
                try:
                    (destination / name).unlink()
                except OSError:  # pragma: no cover — racing unlink
                    pass
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "cache/export: rmtree/unlink during discard raised (destination=%s): %s",
            destination,
            exc,
        )

    # VERIFY the import-critical files are gone. On a dedicated sub-dir a
    # successful rmtree removed the whole tree (nothing remains); on the
    # sandbox-root shape the unlinks above should have cleared them.
    survivors = [n for n in _IMPORT_CRITICAL_NAMES if (destination / n).exists()]
    if not survivors:
        return  # clean — blob is not importable.

    # Deletion left import-critical files behind (EACCES, immutable flag, a
    # racing reopen). Quarantine them so import refuses the blob.
    logger.warning(
        "cache/export: discard could not delete %s at %s — quarantining",
        survivors,
        destination,
    )
    if _quarantine_import_critical(destination):
        logger.warning(
            "cache/export: quarantined rejected blob at %s (renamed %s → .rejected)",
            destination,
            survivors,
        )
        return

    # Could neither delete nor quarantine — the blob may still be importable.
    # Do NOT claim success.
    raise _ExportDiscardError(
        f"rejected export blob at {destination} could not be made "
        f"non-importable (survivors: {survivors})"
    )


@router.post("/export", response_model=ExportResponse)
async def export_cache(req: ExportRequest):
    """Export the engine's KV prefix cache to disk under the sandbox root.

    Flow: resolve the destination against the path whitelist (403 on
    escape) → require a loaded engine (503) → pre-write ``max_bytes`` gate
    on the cheap in-memory footprint (413) → snapshot via
    ``EngineCore.save_cache_to_disk`` → build the manifest from the
    committed on-disk index → post-write ``max_bytes`` gate on the exact
    committed size, discarding the blob if the cache raced over the cap
    (413) → write ``manifest.json`` alongside.

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
    #
    # #1100 BLOCKING-3: also snapshot whether the cache had ANY entries to
    # save here, BEFORE the write. ``save_cache_to_disk`` returns False for
    # BOTH a legitimately-empty cache (nothing to commit — 200 with zero
    # counts) AND a genuine save FAILURE of a non-empty cache (rename never
    # committed / all entries dropped by the post-write verify). We can't
    # tell those apart from the boolean alone, so we remember the pre-write
    # entry count as the discriminator: empty-before → False is legit;
    # non-empty-before → False is a failed save.
    current_bytes = 0
    had_entries_before = False
    if cache is not None:
        current_bytes = int(getattr(cache, "_current_memory", 0))
        try:
            had_entries_before = len(cache) > 0
        except Exception:  # pragma: no cover — defensive against partial cache
            had_entries_before = current_bytes > 0
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

    # #1100 codex round 3 (#2/#5): serialize the ENTIRE save→manifest→size-
    # gate→publish transaction per destination. Without this, two concurrent
    # exports to the same path interleave: one writes a manifest describing
    # the other's snapshot, or the over-cap cleanup of one deletes the newer
    # snapshot the other just committed. The lock makes ``destination`` a
    # single-writer resource; distinct destinations still run concurrently.
    dest_lock = await _acquire_dest_lock(destination)
    async with dest_lock:
        # Snapshot to disk. Routed through the mlx-step worker thread inside
        # the engine (that's where the KV arrays are materializable). Returns
        # True if at least one entry committed; False for an empty / no-op
        # cache. Run in a threadpool so the asyncio loop isn't blocked by the
        # (potentially multi-GB) write.
        saved = await anyio.to_thread.run_sync(
            engine.save_cache_to_disk, str(destination)
        )

        # #1100 BLOCKING-3: a non-empty cache whose save committed NOTHING is a
        # FAILURE, not an empty export. ``save_cache_to_disk`` returns False
        # for a genuinely-empty cache (legit — 200 with zero counts) AND for a
        # non-committing save of a non-empty cache (staging dir vanished, all
        # entries dropped by the post-write verify, or the atomic rename never
        # landed). Distinguished by the pre-write entry snapshot: if the cache
        # HAD entries but nothing committed, surface a 500.
        #
        # #1100 codex round 3 (#2): the failing save uses ``save_to_disk``'s
        # own ``.new``/``.old`` staging and does NOT touch the published
        # destination unless its atomic rename succeeds — so on a
        # non-committing save we DELIBERATELY do NOT ``_discard_export`` the
        # destination (that would destroy a previously-valid snapshot that
        # this failed save never modified). We only sweep the orphaned
        # ``.new``/``.old`` staging dirs the failed save may have left.
        if not saved and had_entries_before:
            _sweep_staging_dirs(destination)
            logger.error(
                "cache/export: save FAILED — cache had entries but nothing "
                "committed to disk (destination=%s); staging dirs swept, "
                "any prior published snapshot left intact",
                destination,
            )
            raise HTTPException(
                status_code=500,
                detail="cache export failed to commit any entry to disk",
            )

        # #1100 codex round 3 (#1): an EMPTY export (no entries committed) to a
        # destination that already holds a STALE prior snapshot must not
        # silently re-export the old entries. ``build_manifest_from_engine_
        # state`` reads counts from the committed ``index.json`` — a leftover
        # one would make an "empty" export report the old blob. Clear the
        # destination's blob artifacts first so the manifest honestly reports
        # 0/0 and no importable stale entry files remain. (Only reached when
        # the save committed nothing AND the cache was legitimately empty.)
        if not saved:
            try:
                _discard_export(destination)
            except _ExportDiscardError as exc:
                logger.error("cache/export: %s", exc)
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "empty cache export could not clear a stale prior "
                        "snapshot at the destination"
                    ),
                ) from exc

        # Everything from here to publish is wrapped so that if manifest write
        # or the size gate raises (e.g. ENOSPC after a multi-GB commit — codex
        # round 3 #3), the just-committed blob is discarded before the error
        # propagates, rather than orphaned to worsen disk exhaustion on retry.
        try:
            # Build the manifest from the COMMITTED on-disk index (#1100
            # BLOCKING-3), not the live ledger — ``cache_dir=destination``
            # makes the counters match what a peer will actually load.
            manifest = build_manifest_from_engine_state(engine, cache_dir=destination)

            # Write the manifest BEFORE the size gate so the committed-blob
            # measure below includes manifest.json — it's part of what a peer
            # imports, so it must count against the cap (#1100 BLOCKING-4).
            write_manifest(destination, manifest)

            # #1100 BLOCKING-2 + BLOCKING-4: precise post-write size
            # enforcement against the ACTUAL committed on-disk footprint. The
            # pre-write gate reads the live logical ledger, which (a) can
            # undercount if the cache grows between that check and the snapshot
            # the step thread takes, and (b) counts only logical
            # ``memory_bytes`` — excluding tokens.bin, index.json,
            # manifest.json, and safetensors serialization overhead.
            # ``_committed_dir_size`` sums the real file sizes now on disk.
            committed_bytes = _committed_dir_size(destination)
        except HTTPException:
            raise
        except Exception as exc:
            # Post-save processing failed (manifest write ENOSPC, etc.). Do NOT
            # leave the just-committed blob orphaned on disk — discard it, then
            # surface a 500. Discard is best-effort here (already in an error
            # path); a quarantine failure is logged, not re-raised over the
            # original fault.
            logger.error(
                "cache/export: post-save processing failed (destination=%s): "
                "%s; discarding the just-committed blob",
                destination,
                exc,
            )
            try:
                _discard_export(destination)
            except _ExportDiscardError as discard_exc:
                logger.error(
                    "cache/export: could not discard blob after post-save failure: %s",
                    discard_exc,
                )
            raise HTTPException(
                status_code=500,
                detail="cache export failed during manifest/size finalization",
            ) from exc

        if req.max_bytes is not None and committed_bytes > req.max_bytes:
            try:
                _discard_export(destination)
            except _ExportDiscardError as exc:
                # #1100 BLOCKING-5: a 413 MUST guarantee the blob is not
                # importable. If we couldn't delete OR quarantine it, we can't
                # honor that invariant — 500 instead of a 413 that lies.
                logger.error("cache/export: %s", exc)
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "cache export exceeded max_bytes but the oversized "
                        "blob could not be discarded"
                    ),
                ) from exc
            logger.info(
                "cache/export: post-write reject — committed on-disk %d B "
                "exceeds max_bytes %d B; blob discarded (destination=%s)",
                committed_bytes,
                req.max_bytes,
                destination,
            )
            raise HTTPException(
                status_code=413,
                detail=(
                    f"cache footprint {committed_bytes} bytes exceeds max_bytes "
                    f"{req.max_bytes}"
                ),
            )

        logger.info(
            "cache/export: wrote %d entries (%d logical B, %d on-disk B, "
            "saved=%s) to destination=%s",
            manifest.entries,
            manifest.total_bytes,
            committed_bytes,
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
    manifest (404 missing / 400 malformed) → reject a protocol-version
    mismatch, a manifest model_id that differs from the loaded model
    (unconditional), or a caller ``expected_model_id`` mismatch (409) →
    require a loaded engine (503) → hydrate via
    ``EngineCore.load_cache_from_disk`` (``merge_strategy="replace"`` clears
    the cache atomically inside that step-thread load).

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

    # Resolve the loaded engine BEFORE the identity gate so we can derive
    # its model id the SAME way the manifest builder does (#1100
    # BLOCKING-2). The gate previously read ``get_config().model_name``,
    # which is empty for an embedded / unit-test engine that never
    # populated ServerConfig — but ``build_manifest_from_engine_state``
    # falls back to ``engine.config.model_name`` (populated), so an
    # embedded engine's manifest carried a real id while the gate compared
    # against "" and skipped the check, letting it import ANOTHER model's
    # KV state. ``resolve_engine_model_id`` is the shared source of truth
    # both call sites use so they can't drift again.
    engine = _engine_or_503()

    # #1100 BLOCKING-1: unconditionally reject a manifest whose model_id
    # does not match the model THIS server loaded. KV cache is model-
    # specific (layer/head/dim geometry, quant layout) — loading another
    # model's blob corrupts inference or crashes the fetch. The caller's
    # ``expected_model_id`` below is an ADDITIONAL, caller-side assertion;
    # omitting it must NOT disable server-side identity checking. The 409
    # detail names NEITHER the manifest's nor the server's model id — a
    # bearer-token holder shouldn't be able to probe what this server runs
    # by diffing mismatch messages; the caller can read its own manifest
    # (or GET /v1/cache/info) to see the id it shipped.
    server_model_id = resolve_engine_model_id(engine)
    if server_model_id:
        if manifest.model_id != server_model_id:
            raise HTTPException(
                status_code=409,
                detail="manifest model_id does not match the loaded engine model",
            )
    else:
        # #1100 codex round 3 (#4): FAIL CLOSED when the loaded engine's model
        # id cannot be resolved. The old ``if server_model_id and ...`` gate
        # fell OPEN here — an id-less engine skipped the comparison and would
        # accept ANY model's KV geometry, the exact corruption the gate
        # exists to prevent. Now such an engine may import ONLY when the
        # caller EXPLICITLY pins a matching ``expected_model_id`` (taking
        # responsibility for the identity assertion the server can't make);
        # otherwise reject. The 422 (not 409) distinguishes "server can't
        # verify identity" from "identities mismatch".
        if req.expected_model_id is None or manifest.model_id != req.expected_model_id:
            logger.warning(
                "cache/import: rejected — loaded engine model id is "
                "unresolvable and caller did not pin a matching "
                "expected_model_id (source=%s)",
                source,
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    "cannot verify model identity: the loaded engine has no "
                    "resolvable model id; supply expected_model_id matching "
                    "the manifest to import explicitly"
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

    # merge_strategy="replace": the in-memory cache is cleared ATOMICALLY
    # inside the step-thread load — ``engine.load_cache_from_disk(...,
    # replace=True)`` forwards to ``MemoryAwarePrefixCache.load_from_disk``,
    # which clears only AFTER index.json is read + validated, on the same
    # mlx-step thread that runs the entry-load loop. #1100 BLOCKING-4: the
    # old code cleared here on the ASYNCIO thread before the step-thread
    # load, so (a) a corrupt/missing source destroyed the existing cache
    # before we knew loading would fail, and (b) a concurrent request could
    # ``store`` into the cache in the gap between clear and load. Pushing the
    # clear into the load call closes both. We still deliberately do NOT use
    # ``scheduler.deep_reset()`` (it would abort in-flight requests). clear()
    # carries the monotonic Prometheus counters over, so replace preserves
    # load_skipped / save_drift_drops.
    replace = req.merge_strategy == "replace"

    # #1100 BLOCKING-5: report the bytes THIS import actually loaded, not the
    # manifest's full ``total_bytes`` (which overstates when entries are
    # skipped). Snapshot the cache footprint around the load: for a committed
    # "replace" the post-load footprint IS the loaded bytes (the cache was
    # emptied first); for "merge" the positive delta is what got added.
    cache = _prefix_cache(engine)
    before_bytes = int(getattr(cache, "_current_memory", 0)) if cache is not None else 0

    # Hydrate from disk. Routed through the mlx-step worker thread so the
    # loaded arrays are tagged with the right generation_stream. ``replace``
    # is passed positionally to fit ``anyio.to_thread.run_sync``'s *args.
    entries_loaded = await anyio.to_thread.run_sync(
        engine.load_cache_from_disk, str(source), replace
    )

    after_bytes = int(getattr(cache, "_current_memory", 0)) if cache is not None else 0
    if entries_loaded == 0:
        # #1100 BLOCKING-1 interaction: a "replace" that ABORTED on a corrupt
        # entry blob (stage-then-swap) returns 0 WITHOUT clearing, so
        # ``after_bytes`` still reflects the PRESERVED existing cache — NOT
        # anything this import loaded. Reporting ``after_bytes`` there would
        # claim the untouched cache as loaded bytes. Nothing loaded → 0 bytes.
        bytes_loaded = 0
    elif replace:
        # Committed replace: cache was cleared then filled with the staged
        # set, so the post-load footprint is exactly the loaded bytes.
        bytes_loaded = after_bytes
    else:
        # Merge: the positive delta over the pre-existing footprint.
        bytes_loaded = max(0, after_bytes - before_bytes)

    entries_skipped = max(0, manifest.entries - entries_loaded)
    logger.info(
        "cache/import: loaded %d/%d entries (skipped=%s, %d B, merge=%s) "
        "from source=%s",
        entries_loaded,
        manifest.entries,
        entries_skipped,
        bytes_loaded,
        req.merge_strategy,
        source,
    )
    return ImportResponse(
        protocol_version=PROTOCOL_VERSION,
        entries_loaded=entries_loaded,
        entries_skipped=entries_skipped,
        bytes_loaded=bytes_loaded,
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
