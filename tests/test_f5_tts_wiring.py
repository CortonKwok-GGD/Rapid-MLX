"""Regression tests for the optional F5-TTS integration."""

from __future__ import annotations

import io
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm_mlx.audio.tts import TTSEngine
from vllm_mlx.routes.audio import _allowed_voices_for


def test_f5_family_is_detected() -> None:
    assert TTSEngine("lucasnewman/f5-tts-mlx")._model_family == "f5"
    assert _allowed_voices_for("lucasnewman/f5-tts-mlx") == ["clone"]


def test_f5_clone_requires_audio_and_transcript_together() -> None:
    engine = TTSEngine("lucasnewman/f5-tts-mlx")
    with pytest.raises(ValueError, match="both ref_audio and ref_text"):
        engine._generate_f5("hello", "/tmp/unused.wav", None, 1.0)


def test_f5_generation_uses_safe_in_memory_default_and_speed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mx = SimpleNamespace(
        array=lambda value: np.asarray(value),
        sqrt=np.sqrt,
        mean=np.mean,
        square=np.square,
        eval=lambda value: None,
        expand_dims=np.expand_dims,
    )
    mlx_module = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    for name, value in vars(fake_mx).items():
        setattr(mlx_core, name, value)
    mlx_module.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    generate_module = ModuleType("f5_tts_mlx.generate")
    generate_module.FRAMES_PER_SEC = 100
    generate_module.SAMPLE_RATE = 24_000
    generate_module.TARGET_RMS = 0.1
    generate_module.convert_char_to_pinyin = lambda texts: texts
    duration = MagicMock(return_value=2.0)
    generate_module.estimated_duration = duration
    package = ModuleType("f5_tts_mlx")
    monkeypatch.setitem(sys.modules, "f5_tts_mlx", package)
    monkeypatch.setitem(sys.modules, "f5_tts_mlx.generate", generate_module)

    monkeypatch.setattr("pkgutil.get_data", lambda *_: b"packaged-wave", raising=True)
    read = MagicMock(return_value=(np.ones(240, dtype=np.float32), 24_000))
    monkeypatch.setattr("soundfile.read", read)

    model = MagicMock()
    model.sample.return_value = (np.ones(480, dtype=np.float32), None)
    engine = TTSEngine("lucasnewman/f5-tts-mlx")
    engine.model = model

    output = engine._generate_f5("你好", None, None, 1.5)

    assert isinstance(read.call_args.args[0], io.BytesIO)
    duration.assert_called_once()
    assert duration.call_args.args[-1] == 1.5
    assert model.sample.call_args.kwargs["speed"] == 1.5
    assert output.sample_rate == 24_000
    assert output.audio.shape == (240,)


def test_f5_rejects_stereo_and_silent_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mx = ModuleType("mlx.core")
    fake_mx.array = np.asarray
    fake_mx.sqrt = np.sqrt
    fake_mx.mean = np.mean
    fake_mx.square = np.square
    fake_mx.eval = lambda value: None
    mlx_module = ModuleType("mlx")
    mlx_module.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    generate_module = ModuleType("f5_tts_mlx.generate")
    generate_module.FRAMES_PER_SEC = 100
    generate_module.SAMPLE_RATE = 24_000
    generate_module.TARGET_RMS = 0.1
    generate_module.convert_char_to_pinyin = lambda texts: texts
    generate_module.estimated_duration = lambda *_: 1.0
    monkeypatch.setitem(sys.modules, "f5_tts_mlx", ModuleType("f5_tts_mlx"))
    monkeypatch.setitem(sys.modules, "f5_tts_mlx.generate", generate_module)

    engine = TTSEngine("lucasnewman/f5-tts-mlx")
    monkeypatch.setattr(
        "soundfile.read",
        lambda *_: (np.ones((240, 2), dtype=np.float32), 24_000),
    )
    with pytest.raises(ValueError, match="mono"):
        engine._generate_f5("hello", "ref.wav", "reference", 1.0)

    monkeypatch.setattr(
        "soundfile.read", lambda *_: (np.zeros(240, dtype=np.float32), 24_000)
    )
    with pytest.raises(ValueError, match="non-silent"):
        engine._generate_f5("hello", "ref.wav", "reference", 1.0)
