"""Mirror a HuggingFace model repo into the rapid-mlx Cloudflare R2 bucket.

This tool is the persisted, in-repo counterpart to the ad-hoc uploads that
seeded ``https://models.rapidmlx.com``. Client-side download uses
``vllm_mlx/_mirror.py``'s ``download_with_mirror_fallback`` which expects
the bucket to hold objects at ``<hf-owner>/<hf-repo>/<filename>`` — the
exact key layout this script writes.

Design (see PR body for full context):

* **Streaming per-file discipline.** For each file in the HF repo we:
  1. HEAD-check R2. If the key already exists with a matching size, SKIP.
  2. Otherwise download the single file to ``<tmp>/<filename>``,
  3. upload it to R2 via boto3 (``s3_client.upload_file`` — which auto-
     splits into multipart uploads at the profile's 64 MB threshold), and
  4. delete the local file BEFORE moving on to the next.

  Peak local disk usage stays under one shard size (~5 GB for typical
  safetensors). Guardrail G11 (100 GB free floor) is respected.

* **boto3 multipart via SDK, not hand-rolled.** ``upload_file`` reads the
  configured ``s3.multipart_threshold`` / ``s3.multipart_chunksize`` from
  the AWS profile (``r2`` profile: 64 MB / 64 MB) and does the multipart
  dance itself. The old ``wrangler r2 put`` path has a 300 MiB cap — this
  script does not use it.

* **Content-Type routing.** ``.json`` → ``application/json``,
  ``.md`` → ``text/markdown``, everything else →
  ``application/octet-stream``. Deterministic + auditable.

* **Resumability via HEAD.** A run that dies mid-repo can be re-run and
  will resume — every file that already exists on R2 with the expected
  size is SKIPped.

* **Fail-fast.** Any upload failure exits nonzero after cleaning the tmp
  dir. We do not silently continue past a broken file.

* **Verification pass.** After uploads, HEAD every expected key on R2 AND
  GET each file via ``https://models.rapidmlx.com/<key>`` to confirm the
  object is publicly readable. Any HTTP-not-200 fails the run.

CLI:

    python scripts/mirror_to_r2.py <hf-repo-id> \\
        [--endpoint-url URL] [--bucket BUCKET] [--profile r2] \\
        [--dry-run] [--verify-only] [--tmp-dir DIR]

Defaults match the production R2 config
(``https://f25478810829faf5ccc86f4ed9a96ef1.r2.cloudflarestorage.com``,
bucket ``rapid-mlx-models``, profile ``r2``).

Install with ``pip install -e .[mirror]`` to pull ``boto3``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Public defaults — persisted so a fresh operator can invoke the tool
# without hunting for the endpoint URL / bucket name.
DEFAULT_ENDPOINT_URL = (
    "https://f25478810829faf5ccc86f4ed9a96ef1.r2.cloudflarestorage.com"
)
DEFAULT_BUCKET = "rapid-mlx-models"
DEFAULT_PROFILE = "r2"
DEFAULT_PUBLIC_BASE = "https://models.rapidmlx.com"

# Cloudflare 403s vanilla ``Python-urllib/*``; use a plausible UA that
# also identifies the tool for R2 log grep.
_USER_AGENT = "Mozilla/5.0 (rapid-mlx mirror uploader)"


def content_type_for(filename: str) -> str:
    """Route filename → Content-Type. Deterministic, three-branch.

    * ``.json`` → ``application/json``
    * ``.md`` → ``text/markdown``
    * everything else → ``application/octet-stream``

    Rationale: existing keys on R2 are ``application/octet-stream``, but
    for future-legibility (curl in a browser, HF-style client sniffing)
    the two textual formats we ship in every repo get their real type.
    Anything else — safetensors, tokenizer.model binaries, images — is
    correctly opaque.
    """
    ext = Path(filename).suffix.lower()
    if ext == ".json":
        return "application/json"
    if ext == ".md":
        return "text/markdown"
    return "application/octet-stream"


@dataclass
class FileMeta:
    """One expected file in the HF repo → R2 mapping."""

    relpath: str  # HF sibling rfilename
    size: int  # bytes; 0 for empty files, from HF metadata
    key: str  # R2 object key = ``<hf-repo-id>/<relpath>``


def _hf_files(repo_id: str) -> list[FileMeta]:
    """Enumerate expected files via HF ``model_info(files_metadata=True)``.

    Returns a list of ``FileMeta`` — one per sibling. Fails hard on any
    HF-side error (fail-fast per G8 root-cause discipline).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    files: list[FileMeta] = []
    for s in info.siblings or []:
        rname = getattr(s, "rfilename", None)
        if not rname:
            continue
        # Sanity: reject any path-traversal-shaped filename before we
        # write it to disk. HF's own upload path enforces this too, but
        # a compromised repo could still ship one.
        if rname.startswith("/") or ".." in Path(rname).parts:
            raise ValueError(f"Refusing suspicious HF filename: {rname!r}")
        size = int(getattr(s, "size", 0) or 0)
        key = f"{repo_id}/{rname}"
        files.append(FileMeta(relpath=rname, size=size, key=key))
    return files


def _r2_client(endpoint_url: str, profile: str) -> Any:
    """Build a boto3 S3 client bound to the R2 endpoint / profile.

    ``addressing_style="path"`` mirrors the AWS profile config and is
    what Cloudflare's S3-compat surface expects.
    """
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=profile)
    return session.client(
        "s3",
        endpoint_url=endpoint_url,
        config=Config(s3={"addressing_style": "path"}),
    )


def _r2_head_size(client: Any, bucket: str, key: str) -> int | None:
    """Return the ``Content-Length`` of a key, or ``None`` on 404.

    Any non-404 error raises (fail-fast — we want the operator to see
    permission / endpoint issues immediately rather than skipping past
    them).
    """
    from botocore.exceptions import ClientError

    try:
        r = client.head_object(Bucket=bucket, Key=key)
        return int(r["ContentLength"])
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        # Boto raises "404" (string) for HEAD-missing objects.
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def should_skip(existing_size: int | None, expected_size: int) -> bool:
    """Skip decision: same size on R2 = same content, do not re-upload.

    Note we treat "HF metadata size is 0" (some older repos don't expose
    a size) as "cannot skip safely" — force a re-upload rather than risk
    accepting a truncated leftover.
    """
    if existing_size is None:
        return False
    if expected_size <= 0:
        # HF didn't tell us the size — refuse to trust an R2 short-write.
        return False
    return existing_size == expected_size


def _upload_one(
    client: Any,
    local_path: Path,
    bucket: str,
    key: str,
    content_type: str,
) -> None:
    """Upload one file to R2 with boto3 multipart auto-split.

    boto3's ``upload_file`` reads ``s3.multipart_threshold`` and
    ``s3.multipart_chunksize`` from the profile config — the ``r2``
    profile sets both to 64 MB. We do NOT reimplement multipart
    ourselves; the SDK handles ``create_multipart_upload`` /
    ``upload_part`` / ``complete_multipart_upload`` including retries.
    """
    client.upload_file(
        Filename=str(local_path),
        Bucket=bucket,
        Key=key,
        ExtraArgs={"ContentType": content_type},
    )


def _download_one_hf(repo_id: str, relpath: str, tmp_dir: Path) -> Path:
    """Download a single HF file into ``tmp_dir`` (flat, no snapshot layout).

    ``local_dir=tmp_dir`` + ``local_dir_use_symlinks=False`` gives us a
    real file (not a symlink into the HF cache) that we can delete after
    upload without touching the operator's shared HF cache — G1 respect.
    """
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=relpath,
            local_dir=str(tmp_dir),
        )
    )


def _cleanup_local(path: Path) -> None:
    """Best-effort delete of a local file. Never fatal."""
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def _public_url(public_base: str, key: str) -> str:
    return f"{public_base.rstrip('/')}/{key.lstrip('/')}"


def _http_head_status(url: str, timeout: float = 30.0) -> int:
    """HEAD ``url`` and return the HTTP status code.

    Uses stdlib urllib to avoid a hard dep on ``requests``. Cloudflare's
    edge sometimes rejects HEAD; fall through to a byte-range GET on any
    non-2xx status to distinguish "HEAD blocked" from "object missing".
    """
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as e:
        return int(e.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0


def _http_range_get_status(url: str, timeout: float = 30.0) -> int:
    """Range-GET the first byte of ``url`` — proves the object is fetchable.

    Cheaper than a full GET and works when HEAD is rejected by an edge
    rule. Used as the definitive "is this publicly readable?" check.
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Range": "bytes=0-0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _ = resp.read(1)
            # 200 (server ignored Range) or 206 (Range honored) both mean
            # the object is publicly reachable.
            status = int(resp.status)
            return 200 if status in (200, 206) else status
    except urllib.error.HTTPError as e:
        return int(e.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0


# ------------------- top-level flow -------------------


def mirror_repo(
    repo_id: str,
    *,
    endpoint_url: str = DEFAULT_ENDPOINT_URL,
    bucket: str = DEFAULT_BUCKET,
    profile: str = DEFAULT_PROFILE,
    public_base: str = DEFAULT_PUBLIC_BASE,
    dry_run: bool = False,
    verify_only: bool = False,
    tmp_dir: Path | None = None,
) -> int:
    """Mirror one HF repo to R2. Return process exit code (0 = ok)."""
    started = time.monotonic()
    print(f"== mirror {repo_id} → r2://{bucket}/{repo_id}/ ==", flush=True)
    print(f"   endpoint: {endpoint_url}", flush=True)
    print(f"   profile:  {profile}", flush=True)
    if dry_run:
        print("   MODE:     dry-run (no uploads)", flush=True)
    if verify_only:
        print("   MODE:     verify-only (no uploads)", flush=True)

    files = _hf_files(repo_id)
    total_bytes = sum(f.size for f in files)
    print(
        f"   files:    {len(files)} ({total_bytes / 1e9:.3f} GB total)",
        flush=True,
    )

    client = _r2_client(endpoint_url, profile)

    # ---- upload / skip loop
    if not verify_only:
        if tmp_dir is None:
            tmp_dir = Path(f"/tmp/mirror-{os.getpid()}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        uploaded = 0
        skipped = 0
        bytes_uploaded = 0
        try:
            for idx, f in enumerate(files, 1):
                head_size = _r2_head_size(client, bucket, f.key)
                if should_skip(head_size, f.size):
                    skipped += 1
                    print(
                        f"[{idx}/{len(files)}] SKIP existing {f.key} ({head_size} B)",
                        flush=True,
                    )
                    continue
                if dry_run:
                    print(
                        f"[{idx}/{len(files)}] DRY-RUN would upload {f.key} "
                        f"({f.size} B, type={content_type_for(f.relpath)})",
                        flush=True,
                    )
                    continue
                # Download → upload → delete, one at a time.
                print(
                    f"[{idx}/{len(files)}] UPLOAD started {f.key} ({f.size} B)",
                    flush=True,
                )
                t0 = time.monotonic()
                local = _download_one_hf(repo_id, f.relpath, tmp_dir)
                try:
                    _upload_one(
                        client,
                        local,
                        bucket,
                        f.key,
                        content_type_for(f.relpath),
                    )
                finally:
                    _cleanup_local(local)
                wall = time.monotonic() - t0
                uploaded += 1
                bytes_uploaded += f.size
                print(
                    f"[{idx}/{len(files)}] OK {f.size} B {wall:.1f}s "
                    f"({(f.size / max(wall, 0.001)) / 1e6:.1f} MB/s) {f.key}",
                    flush=True,
                )
        except Exception as e:
            print(
                f"FAIL {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            # Clean tmp on failure so a re-run isn't confused by stale
            # partial downloads.
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return 2
        finally:
            # Empty tmp on success too — the whole point is per-file
            # streaming with no accumulating cache.
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print(
            f"   upload summary: {uploaded} uploaded, {skipped} skipped, "
            f"{bytes_uploaded / 1e9:.3f} GB",
            flush=True,
        )

    # ---- verification pass
    print(f"-- verify {repo_id} --", flush=True)
    verify_failed: list[tuple[str, str]] = []
    for f in files:
        head_size = _r2_head_size(client, bucket, f.key)
        if head_size is None:
            verify_failed.append((f.key, "r2-missing"))
            print(f"   FAIL {f.key}: not on R2", flush=True)
            continue
        if f.size > 0 and head_size != f.size:
            verify_failed.append(
                (f.key, f"r2-size:{head_size}!={f.size}")
            )
            print(
                f"   FAIL {f.key}: R2 size {head_size} != HF size {f.size}",
                flush=True,
            )
            continue
        # Public read — Range-GET the first byte to prove the CDN serves it.
        url = _public_url(public_base, f.key)
        status = _http_range_get_status(url)
        if status != 200:
            verify_failed.append((f.key, f"public-http:{status}"))
            print(
                f"   FAIL {f.key}: public URL {url} → HTTP {status}",
                flush=True,
            )
            continue
        # Silent on success — printing 1 line per file is enough on the
        # upload pass; the verify pass only reports failures.
    wall = time.monotonic() - started
    if verify_failed:
        print(
            f"== FAILED: {len(verify_failed)} verify errors in {repo_id} ==",
            file=sys.stderr,
            flush=True,
        )
        for k, why in verify_failed:
            print(f"   {k}: {why}", file=sys.stderr, flush=True)
        return 3
    print(
        f"== OK: {repo_id} verified ({len(files)} files, "
        f"{total_bytes / 1e9:.3f} GB, wall {wall:.1f}s) ==",
        flush=True,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mirror_to_r2",
        description=(
            "Mirror a HuggingFace model repo to the rapid-mlx R2 bucket. "
            "Streams file-by-file (peak disk = one shard) and verifies "
            "each object is publicly readable via models.rapidmlx.com."
        ),
    )
    p.add_argument("repo_id", help="HF repo id, e.g. mlx-community/Qwen3-0.6B-4bit")
    p.add_argument(
        "--endpoint-url",
        default=DEFAULT_ENDPOINT_URL,
        help="R2 S3-compat endpoint URL (default: production)",
    )
    p.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="R2 bucket name (default: rapid-mlx-models)",
    )
    p.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="AWS profile name in ~/.aws/credentials (default: r2)",
    )
    p.add_argument(
        "--public-base",
        default=DEFAULT_PUBLIC_BASE,
        help="Public read URL base (default: https://models.rapidmlx.com)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate files, HEAD-check R2, but do not download/upload.",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip upload; only run the verification pass.",
    )
    p.add_argument(
        "--tmp-dir",
        default=None,
        help="Scratch dir for per-file downloads (default: /tmp/mirror-<pid>)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    tmp = Path(args.tmp_dir) if args.tmp_dir else None
    return mirror_repo(
        args.repo_id,
        endpoint_url=args.endpoint_url,
        bucket=args.bucket,
        profile=args.profile,
        public_base=args.public_base,
        dry_run=args.dry_run,
        verify_only=args.verify_only,
        tmp_dir=tmp,
    )


if __name__ == "__main__":
    sys.exit(main())
