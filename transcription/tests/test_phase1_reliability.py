"""Phase 1 reliability tests: unique names, cleanup, limits, empty exports."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf

from config import MAX_DURATION_SEC, MAX_UPLOAD_SIZE_MB
from modules.export_bundle import download_filename, write_exports, write_exports_if_any
from modules.pipeline import transcribe_file
from modules.upload import process_upload
from utils.file_utils import (
    cleanup_temp_file,
    make_unique_filename,
    save_uploaded_bytes,
    validate_audio_duration,
    validate_upload_size,
)


SAMPLE_SEGMENTS = [
    {"start": 0.0, "end": 1.0, "text": "Hello"},
]


def test_make_unique_filename_preserves_stem_and_suffix():
    name_a = make_unique_filename("meeting.wav")
    name_b = make_unique_filename("meeting.wav")
    assert name_a.startswith("meeting_")
    assert name_a.endswith(".wav")
    assert name_a != name_b


def test_save_uploaded_bytes_uses_unique_names(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.file_utils.TEMP_DIR", tmp_path)
    monkeypatch.setattr("utils.file_utils.ensure_dirs", lambda: None)

    path_a = save_uploaded_bytes(b"aaa", "same.wav")
    path_b = save_uploaded_bytes(b"bbb", "same.wav")

    assert path_a.exists()
    assert path_b.exists()
    assert path_a.name != path_b.name
    assert path_a.read_bytes() == b"aaa"
    assert path_b.read_bytes() == b"bbb"


def test_cleanup_temp_file_removes_file(tmp_path):
    path = tmp_path / "temp.wav"
    path.write_bytes(b"data")
    cleanup_temp_file(path)
    assert not path.exists()
    cleanup_temp_file(path)  # missing file is a no-op
    cleanup_temp_file(None)


def test_validate_upload_size_limits():
    ok, err = validate_upload_size(1024)
    assert ok is True
    assert err == ""

    too_big = (MAX_UPLOAD_SIZE_MB + 1) * 1024 * 1024
    ok, err = validate_upload_size(too_big)
    assert ok is False
    assert "too large" in err.lower()


def test_validate_audio_duration_limits():
    ok, err = validate_audio_duration(60.0)
    assert ok is True
    assert err == ""

    ok, err = validate_audio_duration(None)
    assert ok is True

    ok, err = validate_audio_duration(MAX_DURATION_SEC + 1)
    assert ok is False
    assert "too long" in err.lower()


def test_process_upload_rejects_oversized_file(monkeypatch):
    uploaded = MagicMock()
    uploaded.name = "huge.wav"
    uploaded.type = "audio/wav"
    uploaded.getvalue.return_value = b"x" * 100

    monkeypatch.setattr(
        "modules.upload.validate_upload_size",
        lambda _size: (False, "File is too large (501 MB). Maximum allowed size is 500 MB."),
    )

    path, metadata, error = process_upload(uploaded)
    assert path is None
    assert metadata is None
    assert error is not None
    assert "too large" in error.lower()


def test_process_upload_rejects_long_duration(tmp_path, monkeypatch):
    wav_path = tmp_path / "long.wav"
    sf.write(str(wav_path), np.zeros(16000, dtype=np.float32), 16000)

    uploaded = MagicMock()
    uploaded.name = "long.wav"
    uploaded.type = "audio/wav"
    uploaded.getvalue.return_value = wav_path.read_bytes()

    monkeypatch.setattr("modules.upload.save_uploaded_bytes", lambda data, name: wav_path)
    monkeypatch.setattr(
        "modules.upload.get_file_metadata",
        lambda _path: {
            "filename": "long.wav",
            "duration_sec": MAX_DURATION_SEC + 10,
            "size_mb": 0.1,
        },
    )
    cleaned: list[Path] = []
    monkeypatch.setattr("modules.upload.cleanup_temp_file", cleaned.append)

    path, metadata, error = process_upload(uploaded)
    assert path is None
    assert metadata is None
    assert error is not None
    assert "too long" in error.lower()
    assert cleaned == [wav_path]


def test_write_exports_uses_unique_disk_names(monkeypatch, tmp_path):
    monkeypatch.setattr("modules.export_bundle.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("modules.export_bundle.ensure_dirs", lambda: None)

    paths_a = write_exports(SAMPLE_SEGMENTS, "meeting.wav", include_timestamps=True)
    paths_b = write_exports(SAMPLE_SEGMENTS, "meeting.wav", include_timestamps=True)

    assert paths_a["txt"].name != paths_b["txt"].name
    assert paths_a["txt"].exists()
    assert paths_b["txt"].exists()
    assert download_filename("meeting.wav", ".txt") == "meeting.txt"


def test_write_exports_if_any_skips_empty_segments(monkeypatch, tmp_path):
    monkeypatch.setattr("modules.export_bundle.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("modules.export_bundle.ensure_dirs", lambda: None)

    empty_paths = write_exports_if_any([], "silent.wav", include_timestamps=True)
    assert empty_paths == {}
    assert list(tmp_path.iterdir()) == []

    filled_paths = write_exports_if_any(SAMPLE_SEGMENTS, "speech.wav", include_timestamps=True)
    assert set(filled_paths.keys()) == {"txt", "srt", "vtt", "docx"}
    assert all(path.exists() for path in filled_paths.values())


def test_pipeline_cleans_preprocessed_file(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    sf.write(str(source), np.zeros(1600, dtype=np.float32), 16000)

    preprocessed = tmp_path / "pre.wav"
    sf.write(str(preprocessed), np.zeros(1600, dtype=np.float32), 16000)

    monkeypatch.setattr("modules.pipeline.preprocess_audio", lambda _path: preprocessed)
    monkeypatch.setattr(
        "modules.pipeline.transcribe_audio",
        lambda *args, **kwargs: [{"start": 0.0, "end": 0.5, "text": "hi"}],
    )
    monkeypatch.setattr(
        "modules.pipeline.postprocess_segments",
        lambda segments, convert_to_traditional: segments,
    )
    monkeypatch.setattr(
        "modules.pipeline.segments_to_full_text",
        lambda segments, **kwargs: "hi",
    )

    result = transcribe_file(
        source,
        model_name="small",
        language=None,
        convert_traditional=False,
        model=object(),  # bypass model loading
    )

    assert result["segment_count"] == 1
    assert not preprocessed.exists()
    assert source.exists()


def test_pipeline_cleanup_source_when_requested(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    sf.write(str(source), np.zeros(1600, dtype=np.float32), 16000)
    preprocessed = tmp_path / "pre.wav"
    sf.write(str(preprocessed), np.zeros(1600, dtype=np.float32), 16000)

    monkeypatch.setattr("modules.pipeline.preprocess_audio", lambda _path: preprocessed)
    monkeypatch.setattr("modules.pipeline.transcribe_audio", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "modules.pipeline.postprocess_segments",
        lambda segments, convert_to_traditional: segments,
    )
    monkeypatch.setattr("modules.pipeline.segments_to_full_text", lambda segments, **kwargs: "")

    transcribe_file(
        source,
        model_name="small",
        language=None,
        convert_traditional=False,
        model=object(),
        cleanup_source=True,
    )

    assert not preprocessed.exists()
    assert not source.exists()
