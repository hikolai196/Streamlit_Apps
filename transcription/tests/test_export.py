"""Tests for export modules."""

from pathlib import Path

from modules.export_docx import export_docx
from modules.export_srt import export_srt
from modules.export_txt import export_txt


SAMPLE_SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": "Hello world"},
    {"start": 2.5, "end": 5.0, "text": "Second line"},
]


def test_export_txt_with_timestamps(tmp_path: Path):
    out = tmp_path / "test.txt"
    export_txt(SAMPLE_SEGMENTS, out, include_timestamps=True)
    content = out.read_text(encoding="utf-8")
    assert "Hello world" in content
    assert "00:00:00" in content


def test_export_txt_without_timestamps(tmp_path: Path):
    out = tmp_path / "test.txt"
    export_txt(SAMPLE_SEGMENTS, out, include_timestamps=False)
    content = out.read_text(encoding="utf-8")
    assert content.strip() == "Hello world\nSecond line"


def test_export_srt_format(tmp_path: Path):
    out = tmp_path / "test.srt"
    export_srt(SAMPLE_SEGMENTS, out)
    content = out.read_text(encoding="utf-8")
    assert "1\n" in content
    assert "Hello world" in content
    assert "-->" in content


def test_export_docx_creates_file(tmp_path: Path):
    out = tmp_path / "test.docx"
    export_docx(SAMPLE_SEGMENTS, out, include_timestamps=True)
    assert out.exists()
    assert out.stat().st_size > 0
