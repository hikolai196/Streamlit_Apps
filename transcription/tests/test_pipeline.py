"""Integration tests for the transcription pipeline (real preprocess, mocked ASR)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from modules.pipeline import transcribe_file


def _write_stereo_fixture(path: Path, *, seconds: float = 0.5, sr: int = 44100) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    stereo = np.column_stack([0.2 * np.sin(2 * np.pi * 440 * t)] * 2)
    sf.write(str(path), stereo.astype(np.float32), sr)


def test_pipeline_end_to_end_with_mocked_asr(tmp_path, monkeypatch):
    source = tmp_path / "meeting.wav"
    _write_stereo_fixture(source)

    monkeypatch.setattr(
        "modules.pipeline.transcribe_audio",
        lambda *args, **kwargs: [
            {"start": 0.0, "end": 0.4, "text": "你好"},
            {"start": 0.4, "end": 0.8, "text": "世界"},
        ],
    )

    progress: list[tuple[float, str]] = []

    result = transcribe_file(
        source,
        model_name="small",
        language="zh",
        convert_traditional=True,
        model=object(),
        progress=lambda fraction, message: progress.append((fraction, message)),
    )

    assert result["segment_count"] == 2
    assert "你好" in result["full_text"] or "世界" in result["full_text"]
    # OpenCC s2t keeps these characters; ensure postprocess ran and trimmed.
    assert all("text" in seg for seg in result["segments"])
    assert progress[0][0] == 0.0
    assert progress[-1] == (1.0, "Done")
    assert source.exists()


def test_pipeline_reports_empty_speech(tmp_path, monkeypatch):
    source = tmp_path / "silent.wav"
    _write_stereo_fixture(source, seconds=0.2)

    monkeypatch.setattr("modules.pipeline.transcribe_audio", lambda *args, **kwargs: [])

    result = transcribe_file(
        source,
        model_name="small",
        language=None,
        convert_traditional=False,
        model=object(),
    )
    assert result["segments"] == []
    assert result["full_text"] == ""
    assert result["segment_count"] == 0


def test_pipeline_cleans_preprocessed_even_on_asr_failure(tmp_path, monkeypatch):
    source = tmp_path / "bad.wav"
    _write_stereo_fixture(source)

    created: dict[str, Path] = {}

    real_preprocess = __import__("modules.preprocess", fromlist=["preprocess_audio"]).preprocess_audio

    def tracking_preprocess(path: Path) -> Path:
        out = real_preprocess(path)
        created["path"] = out
        return out

    monkeypatch.setattr("modules.pipeline.preprocess_audio", tracking_preprocess)

    def boom(*args, **kwargs):
        raise RuntimeError("asr failed")

    monkeypatch.setattr("modules.pipeline.transcribe_audio", boom)

    try:
        transcribe_file(
            source,
            model_name="small",
            language=None,
            convert_traditional=False,
            model=object(),
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "asr failed" in str(exc)

    assert "path" in created
    assert not created["path"].exists()
