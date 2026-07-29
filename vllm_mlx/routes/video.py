# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible video jobs backed by MLX-native LTX-2.3."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from ..middleware.auth import verify_api_key

router = APIRouter()
logger = logging.getLogger(__name__)

_MAX_REFERENCE_BYTES = 20 * 1024 * 1024
_MAX_JOBS = 100
_MAX_PIXEL_FRAMES = 768 * 512 * 97
_jobs_lock = threading.Lock()
_jobs_root = Path(tempfile.mkdtemp(prefix="rapid-mlx-videos-"))


@dataclass
class _VideoJob:
    id: str
    model: str
    prompt: str
    seconds: str
    size: str
    status: str = "queued"
    progress: int = 0
    created_at: int = 0
    completed_at: int | None = None
    error: dict[str, str] | None = None
    output_path: str | None = None

    def public(self) -> dict:
        value = asdict(self)
        value.pop("output_path")
        value["object"] = "video"
        return value


_jobs: dict[str, _VideoJob] = {}
_tasks: dict[str, asyncio.Task] = {}
_cleanup_tasks: set[asyncio.Task] = set()
_generation_gate = asyncio.Lock()


def _cleanup_jobs() -> None:
    shutil.rmtree(_jobs_root, ignore_errors=True)


atexit.register(_cleanup_jobs)


def _video_engine():
    from ..config import get_config

    engine = get_config().engine
    if engine is None or not getattr(engine, "is_video_gen", False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "message": (
                        "This server is not running a video model. Start it with "
                        "`rapid-mlx serve ltx-2.3-mlx-q4`."
                    ),
                    "type": "invalid_request_error",
                    "code": "video_model_not_loaded",
                    "param": "model",
                }
            },
        )
    return engine


def _parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="size must be WIDTHxHEIGHT"
        ) from exc
    if (
        width < 256
        or height < 256
        or width > 1920
        or height > 1920
        or width % 64
        or height % 64
    ):
        raise HTTPException(
            status_code=400,
            detail="video width/height must be 256..1920 and divisible by 64",
        )
    return width, height


def _frame_count(seconds: int, fps: int = 24) -> int:
    requested = seconds * fps
    return max(9, round((requested - 1) / 8) * 8 + 1)


async def _run_job(
    job: _VideoJob,
    *,
    engine,
    width: int,
    height: int,
    seconds: int,
    seed: int,
    image_path: Path | None,
) -> None:
    started = False
    output = _jobs_root / job.id / "output.mp4"

    async def generate_under_gate() -> None:
        nonlocal started
        # This inner task owns the gate, so cancellation of the request-facing
        # job never releases it while an uncancellable MLX thread is running.
        async with _generation_gate:
            started = True
            with _jobs_lock:
                job.status = "in_progress"
                job.progress = 1
            await asyncio.to_thread(
                engine.generate,
                prompt=job.prompt,
                output_path=output,
                width=width,
                height=height,
                num_frames=_frame_count(seconds),
                fps=24,
                seed=seed,
                image=image_path,
            )

    runner = asyncio.create_task(generate_under_gate())
    try:
        await asyncio.shield(runner)
        with _jobs_lock:
            job.status = "completed"
            job.progress = 100
            job.completed_at = int(time.time())
            job.output_path = str(output)
    except asyncio.CancelledError:
        with _jobs_lock:
            job.status = "failed"
            job.error = {
                "code": "video_generation_cancelled",
                "message": "Video generation was cancelled.",
            }
        if not started:
            runner.cancel()

        async def reap_cancelled_job() -> None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await runner
            await asyncio.to_thread(
                shutil.rmtree, _jobs_root / job.id, ignore_errors=True
            )

        cleanup = asyncio.create_task(reap_cancelled_job())
        _cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(_cleanup_tasks.discard)
        raise
    except Exception as exc:  # noqa: BLE001
        from ..runtime.video_lane import VideoRuntimeError

        logger.exception("LTX-2.3 video job %s failed", job.id)
        message = (
            str(exc)
            if isinstance(exc, VideoRuntimeError)
            else "Video generation failed; check the server logs for details."
        )
        with _jobs_lock:
            job.status = "failed"
            job.error = {"code": "video_generation_failed", "message": message}
        await asyncio.to_thread(shutil.rmtree, _jobs_root / job.id, ignore_errors=True)


@router.post("/v1/videos", dependencies=[Depends(verify_api_key)])
async def create_video(
    prompt: str = Form(..., min_length=1, max_length=4096),
    model: str = Form("ltx-2.3-mlx-q4"),
    seconds: str = Form("4"),
    size: str = Form("768x512"),
    seed: int = Form(42),
    input_reference: UploadFile | None = File(None),
):
    engine = _video_engine()
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must not be blank")
    if model not in {"ltx-2.3-mlx-q4", engine.model_name}:
        raise HTTPException(status_code=400, detail="model must be ltx-2.3-mlx-q4")
    try:
        seconds_int = int(seconds)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="seconds must be an integer"
        ) from exc
    if not 1 <= seconds_int <= 20:
        raise HTTPException(status_code=400, detail="seconds must be between 1 and 20")
    width, height = _parse_size(size)
    if width * height * _frame_count(seconds_int) > _MAX_PIXEL_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=(
                "requested video exceeds the safe LTX-2.3 Q4 workload limit "
                "(768x512 at 4 seconds); reduce size or duration"
            ),
        )

    job_id = f"video_{uuid.uuid4().hex}"
    job_dir = _jobs_root / job_id
    job_dir.mkdir(mode=0o700)
    image_path = None
    enqueued = False
    evicted_id: str | None = None
    try:
        if input_reference is not None:
            image_path = job_dir / "reference.img"
            total = 0
            with image_path.open("xb") as target:
                while chunk := await input_reference.read(1024 * 1024):
                    total += len(chunk)
                    if total > _MAX_REFERENCE_BYTES:
                        raise HTTPException(
                            status_code=413, detail="input_reference exceeds 20 MB"
                        )
                    target.write(chunk)

        job = _VideoJob(
            id=job_id,
            model="ltx-2.3-mlx-q4",
            prompt=prompt,
            seconds=str(seconds_int),
            size=f"{width}x{height}",
            created_at=int(time.time()),
        )
        with _jobs_lock:
            if len(_jobs) >= _MAX_JOBS:
                finished = [
                    item
                    for item in _jobs.values()
                    if item.status in {"completed", "failed"}
                ]
                if not finished:
                    raise HTTPException(
                        status_code=429, detail="video job queue is full"
                    )
                oldest = min(finished, key=lambda item: item.created_at)
                _jobs.pop(oldest.id, None)
                evicted_id = oldest.id
            _jobs[job.id] = job
            enqueued = True
    finally:
        if not enqueued:
            await asyncio.to_thread(shutil.rmtree, job_dir, ignore_errors=True)
    task = asyncio.create_task(
        _run_job(
            job,
            engine=engine,
            width=width,
            height=height,
            seconds=seconds_int,
            seed=seed,
            image_path=image_path,
        )
    )
    _tasks[job.id] = task

    def discard_task(done: asyncio.Task) -> None:
        if _tasks.get(job.id) is done:
            _tasks.pop(job.id, None)

    task.add_done_callback(discard_task)
    if evicted_id is not None:
        await asyncio.to_thread(
            shutil.rmtree, _jobs_root / evicted_id, ignore_errors=True
        )
    return job.public()


def _get_job(video_id: str) -> _VideoJob:
    with _jobs_lock:
        job = _jobs.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="video job not found")
    return job


@router.get("/v1/videos/{video_id}", dependencies=[Depends(verify_api_key)])
async def retrieve_video(video_id: str):
    return _get_job(video_id).public()


@router.get("/v1/videos/{video_id}/content", dependencies=[Depends(verify_api_key)])
async def retrieve_video_content(video_id: str):
    job = _get_job(video_id)
    if job.status != "completed" or job.output_path is None:
        raise HTTPException(status_code=409, detail="video is not completed")
    # Open before releasing control. A concurrent delete/eviction may unlink
    # the path, but the already-open descriptor remains streamable on macOS.
    try:
        source = open(job.output_path, "rb")  # noqa: SIM115
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=410, detail="video content has expired"
        ) from exc

    def chunks():
        try:
            while data := source.read(1024 * 1024):
                yield data
        finally:
            source.close()

    return StreamingResponse(
        chunks(),
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{job.id}.mp4"'},
    )


@router.get("/v1/videos", dependencies=[Depends(verify_api_key)])
async def list_videos(limit: int = Query(20, ge=1, le=100)):
    with _jobs_lock:
        data = sorted(_jobs.values(), key=lambda item: item.created_at, reverse=True)[
            :limit
        ]
    return {"object": "list", "data": [job.public() for job in data]}


@router.delete("/v1/videos/{video_id}", dependencies=[Depends(verify_api_key)])
async def delete_video(video_id: str):
    with _jobs_lock:
        job = _jobs.get(video_id)
        if job is not None and job.status == "in_progress":
            raise HTTPException(
                status_code=409, detail="video generation is in progress"
            )
        task = _tasks.get(video_id) if job is not None else None
        if job is not None and job.status != "queued":
            _jobs.pop(video_id, None)
    if job is None:
        raise HTTPException(status_code=404, detail="video job not found")
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        with _jobs_lock:
            _jobs.pop(video_id, None)
    await asyncio.to_thread(shutil.rmtree, _jobs_root / video_id, ignore_errors=True)
    response = job.public()
    response["deleted"] = True
    return response
