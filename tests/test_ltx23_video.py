"""LTX-2.3 MLX-native lane and OpenAI video API contract tests."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest

from vllm_mlx.model_aliases import resolve_profile
from vllm_mlx.routes import video
from vllm_mlx.runtime.video_lane import VideoEngine


def test_ltx23_alias_routes_to_video_lane() -> None:
    profile = resolve_profile("ltx-2.3-mlx-q4")
    assert profile is not None
    assert profile.hf_path == "notapalindrome/ltx23-mlx-av-q4"
    assert profile.modality == "video-gen"
    assert profile.min_memory_gb == 24
    assert profile.supports_spec_decode is False


def test_video_engine_calls_mlx_native_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}
    fake = ModuleType("mlx_video")

    def generate_video_with_audio(**kwargs) -> None:
        captured.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    fake.generate_video_with_audio = generate_video_with_audio
    monkeypatch.setitem(sys.modules, "mlx_video", fake)
    monkeypatch.setattr("shutil.which", lambda _: "/opt/homebrew/bin/ffmpeg")

    output = tmp_path / "result.mp4"
    engine = VideoEngine("notapalindrome/ltx23-mlx-av-q4")
    engine.generate(
        prompt="A fox runs through snow",
        output_path=output,
        width=768,
        height=512,
        num_frames=97,
        fps=24,
        seed=7,
        image=None,
    )

    assert output.read_bytes() == b"mp4"
    assert captured["model_repo"] == "notapalindrome/ltx23-mlx-av-q4"
    assert captured["num_frames"] == 97
    assert captured["image"] is None


def test_video_parameter_validation() -> None:
    assert video._parse_size("768x512") == (768, 512)
    assert video._frame_count(4) == 97
    with pytest.raises(Exception, match="divisible by 64"):
        video._parse_size("700x512")


@pytest.mark.asyncio
async def test_failed_reference_upload_is_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

    class BrokenUpload:
        async def read(self, size: int) -> bytes:
            raise OSError("upload interrupted")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    before = set(video._jobs_root.iterdir())
    with pytest.raises(OSError, match="upload interrupted"):
        await video.create_video(
            prompt="test",
            model="ltx-2.3-mlx-q4",
            seconds="1",
            size="512x512",
            seed=1,
            input_reference=BrokenUpload(),
        )
    assert set(video._jobs_root.iterdir()) == before


@pytest.mark.asyncio
async def test_video_job_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            output_path.write_bytes(b"generated-mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    created = await video.create_video(
        prompt="Ocean waves at sunset",
        model="ltx-2.3-mlx-q4",
        seconds="2",
        size="768x512",
        seed=42,
        input_reference=None,
    )
    video_id = created["id"]
    for _ in range(100):
        current = await video.retrieve_video(video_id)
        if current["status"] != "queued":
            if current["status"] == "completed":
                break
        await asyncio.sleep(0.01)

    assert current["status"] == "completed"
    assert current["progress"] == 100
    response = await video.retrieve_video_content(video_id)
    chunks = [chunk async for chunk in response.body_iterator]
    assert b"".join(chunks) == b"generated-mp4"
    deleted = await video.delete_video(video_id)
    assert deleted["deleted"] is True


@pytest.mark.asyncio
async def test_video_jobs_stay_queued_until_worker_is_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    calls = 0

    class BlockingEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                assert release.wait(timeout=5)
            output_path.write_bytes(b"mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: BlockingEngine())
    first = await video.create_video(
        prompt="first",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=1,
        input_reference=None,
    )
    assert await asyncio.to_thread(started.wait, 2)
    second = await video.create_video(
        prompt="second",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=2,
        input_reference=None,
    )
    await asyncio.sleep(0.05)

    assert (await video.retrieve_video(first["id"]))["status"] == "in_progress"
    assert (await video.retrieve_video(second["id"]))["status"] == "queued"
    release.set()
    second_status = None
    for _ in range(200):
        second_status = (await video.retrieve_video(second["id"]))["status"]
        if second_status == "completed":
            break
        await asyncio.sleep(0.01)
    assert calls == 2
    assert second_status == "completed"
    response = await video.retrieve_video_content(second["id"])
    assert b"".join([chunk async for chunk in response.body_iterator]) == b"mp4"
    await video.delete_video(first["id"])
    await video.delete_video(second["id"])


@pytest.mark.asyncio
async def test_cancelled_job_reaches_terminal_state_and_cleans_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            started.set()
            assert release.wait(timeout=5)
            output_path.write_bytes(b"late-output")

    monkeypatch.setattr(video, "_video_engine", lambda: BlockingEngine())
    created = await video.create_video(
        prompt="cancel me",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=3,
        input_reference=None,
    )
    assert await asyncio.to_thread(started.wait, 2)
    task = next(task for task in video._tasks if not task.done())
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    current = await video.retrieve_video(created["id"])
    assert current["status"] == "failed"
    assert current["error"]["code"] == "video_generation_cancelled"
    assert not (video._jobs_root / created["id"]).exists()
    await video.delete_video(created["id"])
