# SPDX-License-Identifier: Apache-2.0
"""MLX-native LTX-2.3 video-generation lane."""

from __future__ import annotations

import shutil
import threading
from pathlib import Path


class VideoRuntimeError(RuntimeError):
    """Safe, actionable generation error suitable for the public API."""


class VideoEngine:
    """Thin adapter over ``mlx-video-with-audio``'s LTX-2.3 pipeline.

    The upstream function owns model loading and generation. Rapid-MLX owns
    request validation, job lifecycle, concurrency and the OpenAI-compatible
    transport surface.
    """

    is_video_gen = True
    _loaded = True

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._generation_lock = threading.Lock()

    def generate(
        self,
        *,
        prompt: str,
        output_path: Path,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int,
        image: Path | None,
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise VideoRuntimeError(
                "LTX-2.3 video generation requires ffmpeg. "
                "Install it with `brew install ffmpeg`."
            )
        try:
            from mlx_video import generate_video_with_audio
        except ImportError as exc:
            raise VideoRuntimeError(
                "LTX-2.3 support is not installed. "
                "Run `pip install 'rapid-mlx[video]'`."
            ) from exc

        # The 22B pipeline is not re-entrant and a second concurrent graph can
        # exhaust unified memory. Serialize jobs per served model.
        with self._generation_lock:
            generate_video_with_audio(
                model_repo=self.model_name,
                text_encoder_repo=None,
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
                fps=fps,
                output_path=str(output_path),
                image=str(image) if image is not None else None,
                verbose=False,
                enhance_prompt=False,
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise VideoRuntimeError(
                "LTX-2.3 generation completed without an MP4 output."
            )

    def generate_warmup(self) -> None:
        """Video weights load lazily; startup must not trigger a 40+ GB pull."""

    async def stop(self) -> None:
        """No persistent worker is owned by this thin adapter."""
