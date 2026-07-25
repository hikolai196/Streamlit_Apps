"""Export edge cases: empty segments, odd names, path traversal."""

from __future__ import annotations

from pathlib import Path

from modules.export_bundle import download_filename, write_exports, write_exports_if_any
from modules.export_srt import export_srt
from modules.export_txt import export_txt
from utils.file_utils import make_unique_filename, save_uploaded_bytes, validate_audio_file


SAMPLE = [{"start": 0.0, "end": 1.0, "text": "Hello"}]


def test_export_txt_empty_segments(tmp_path):
    out = tmp_path / "empty.txt"
    export_txt([], out, include_timestamps=True)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == ""


def test_export_srt_empty_segments(tmp_path):
    out = tmp_path / "empty.srt"
    export_srt([], out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert content.strip() == "" or content == ""


def test_write_exports_odd_and_unicode_filenames(monkeypatch, tmp_path):
    monkeypatch.setattr("modules.export_bundle.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("modules.export_bundle.ensure_dirs", lambda: None)

    for base in ["meeting (1).wav", "檔案.mp3", "file.with.dots.m4a"]:
        paths = write_exports(SAMPLE, base, include_timestamps=True)
        assert paths["txt"].exists()
        assert download_filename(base, ".txt").endswith(".txt")
        assert ".." not in paths["txt"].name


def test_path_traversal_stripped_from_unique_and_download_names():
    unique = make_unique_filename("../../etc/passwd.wav")
    assert ".." not in unique
    assert unique.endswith(".wav")
    assert unique.startswith("passwd_")

    download = download_filename("../../secret/meeting.wav", ".txt")
    assert download == "meeting.txt"
    assert ".." not in download
    assert "/" not in download
    assert "\\" not in download


def test_save_uploaded_bytes_strips_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.file_utils.TEMP_DIR", tmp_path)
    monkeypatch.setattr("utils.file_utils.ensure_dirs", lambda: None)

    saved = save_uploaded_bytes(b"audio-bytes", "../outside/meeting.wav")
    assert saved.parent == tmp_path
    assert saved.exists()
    assert ".." not in saved.name
    assert saved.name.startswith("meeting_")


def test_validate_rejects_non_audio_even_with_traversal_name():
    valid, err = validate_audio_file("../../notes.pdf")
    assert valid is False
    assert "Unsupported" in err


def test_write_exports_if_any_empty_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr("modules.export_bundle.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("modules.export_bundle.ensure_dirs", lambda: None)

    assert write_exports_if_any([], "x.wav", True) == {}
    assert list(tmp_path.iterdir()) == []
