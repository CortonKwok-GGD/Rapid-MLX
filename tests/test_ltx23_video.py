"""LTX-2.3 MLX-native lane and OpenAI video API contract tests."""

from __future__ import annotations

import asyncio
import sys
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
    assert Path(response.path).read_bytes() == b"generated-mp4"
    deleted = await video.delete_video(video_id)
    assert deleted["deleted"] is True
