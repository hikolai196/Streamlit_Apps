"""Unit tests for ASR model loading and transcription (mocked Whisper)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modules.asr as asr


@pytest.fixture(autouse=True)
def clear_model_cache():
    asr._model_cache.clear()
    yield
    asr._model_cache.clear()


def test_load_whisper_model_cpu_uses_int8(monkeypatch):
    captured: dict[str, object] = {}

    def fake_whisper(model_name, device, compute_type):
        captured.update(
            {
                "model_name": model_name,
                "device": device,
                "compute_type": compute_type,
            }
        )
        return MagicMock(name="whisper")

    monkeypatch.setattr(asr, "WhisperModel", fake_whisper)

    model = asr.load_whisper_model("small", device="cpu")
    assert model is not None
    assert captured["model_name"] == "small"
    assert captured["device"] == "cpu"
    assert captured["compute_type"] == "int8"


def test_load_whisper_model_auto_cuda_uses_float16(monkeypatch):
    captured: dict[str, object] = {}

    def fake_whisper(model_name, device, compute_type):
        captured.update(model_name=model_name, device=device, compute_type=compute_type)
        return MagicMock(name="whisper-cuda")

    torch_mod = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_mod)
    monkeypatch.setattr(asr, "WhisperModel", fake_whisper)

    asr.load_whisper_model("medium", device="auto")
    assert captured["compute_type"] == "float16"
    assert captured["device"] == "cuda"


def test_load_whisper_model_reuses_cache(monkeypatch):
    calls = {"count": 0}

    def fake_whisper(model_name, device, compute_type):
        calls["count"] += 1
        return MagicMock(name=f"{model_name}-{calls['count']}")

    monkeypatch.setattr(asr, "WhisperModel", fake_whisper)
    first = asr.load_whisper_model("small", device="cpu")
    second = asr.load_whisper_model("small", device="cpu")
    assert first is second
    assert calls["count"] == 1


def test_transcribe_audio_passes_language_vad_and_beam(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake")

    seg = SimpleNamespace(start=0.123, end=1.987, text=" hello ")
    info = SimpleNamespace(duration=2.0)
    model = MagicMock()
    model.transcribe.return_value = (iter([seg]), info)

    progress_calls: list[tuple[float, float]] = []

    segments = asr.transcribe_audio(
        audio,
        model_name="small",
        language="zh",
        model=model,
        on_segment_progress=lambda cur, total: progress_calls.append((cur, total)),
    )

    model.transcribe.assert_called_once()
    args, kwargs = model.transcribe.call_args
    assert args[0] == str(audio)
    assert kwargs["beam_size"] == 5
    assert kwargs["vad_filter"] is True
    assert kwargs["language"] == "zh"

    assert segments == [{"start": 0.12, "end": 1.99, "text": " hello "}]
    assert progress_calls == [(1.987, 2.0)]


def test_transcribe_audio_omits_language_when_none(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake")

    model = MagicMock()
    model.transcribe.return_value = (iter([]), SimpleNamespace(duration=0.0))

    asr.transcribe_audio(audio, language=None, model=model)
    _, kwargs = model.transcribe.call_args
    assert "language" not in kwargs


def test_transcribe_audio_passes_prompt_and_hotwords(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake")

    model = MagicMock()
    model.transcribe.return_value = (iter([]), SimpleNamespace(duration=0.0))

    asr.transcribe_audio(
        audio,
        language="en",
        model=model,
        initial_prompt="Quarterly planning meeting.",
        hotwords="AILM, Taipei\nWhisper",
    )

    _, kwargs = model.transcribe.call_args
    assert "Quarterly planning meeting." in kwargs["initial_prompt"]
    assert "AILM" in kwargs["initial_prompt"]
    assert kwargs["hotwords"] == "AILM, Taipei, Whisper"


def test_resolve_device_falls_back_from_cuda_without_gpu(monkeypatch):
    monkeypatch.setattr(asr, "cuda_available", lambda: False)
    device, warning = asr.resolve_device("cuda")
    assert device == "cpu"
    assert warning is not None
    assert "CUDA" in warning


def test_resolve_device_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(asr, "cuda_available", lambda: True)
    device, warning = asr.resolve_device("auto")
    assert device == "cuda"
    assert warning is None


def test_build_prompt_bias():
    assert asr.build_prompt_bias() is None
    assert asr.build_prompt_bias(hotwords="  a, b \n c ") == "a, b, c"
    assert asr.build_prompt_bias(initial_prompt="Hello", hotwords="AILM") == "Hello AILM"


def test_transcribe_audio_hotwords_fallback_without_kwarg(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake")

    model = MagicMock()

    def fake_transcribe(path, **kwargs):
        if "hotwords" in kwargs:
            raise TypeError("unexpected keyword argument 'hotwords'")
        return iter([]), SimpleNamespace(duration=0.0)

    model.transcribe.side_effect = fake_transcribe
    asr.transcribe_audio(audio, model=model, hotwords="AILM")
    assert model.transcribe.call_count == 2
    _, kwargs = model.transcribe.call_args
    assert "hotwords" not in kwargs
    assert "initial_prompt" in kwargs


def test_transcribe_audio_wraps_failures(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake")
    model = MagicMock()
    model.transcribe.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="Transcription failed"):
        asr.transcribe_audio(audio, model=model)
