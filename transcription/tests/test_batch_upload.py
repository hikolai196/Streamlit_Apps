"""Tests for batch upload handling."""

from pathlib import Path
from unittest.mock import MagicMock

from modules.upload import process_uploads


def test_process_uploads_handles_multiple_files(tmp_path, monkeypatch):
    def fake_save(data: bytes, filename: str) -> Path:
        path = tmp_path / filename
        path.write_bytes(data)
        return path

    def fake_metadata(path: Path) -> dict:
        return {"filename": path.name, "size_mb": 0.01}

    monkeypatch.setattr("modules.upload.save_uploaded_bytes", fake_save)
    monkeypatch.setattr("modules.upload.get_file_metadata", fake_metadata)

    file_a = MagicMock()
    file_a.name = "a.wav"
    file_a.type = "audio/wav"
    file_a.getvalue.return_value = b"wav-a"

    file_b = MagicMock()
    file_b.name = "b.mp3"
    file_b.type = "audio/mpeg"
    file_b.getvalue.return_value = b"mp3-b"

    results = process_uploads([file_a, file_b])
    assert len(results) == 2
    assert results[0]["filename"] == "a.wav"
    assert results[0]["error"] is None
    assert results[1]["filename"] == "b.mp3"
