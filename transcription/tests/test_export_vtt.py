"""Tests for WebVTT export."""

from modules.export_vtt import _seconds_to_vtt_time, export_vtt


SAMPLE_SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": "Hello world"},
    {"start": 2.5, "end": 5.0, "text": "Second line"},
]


def test_seconds_to_vtt_time():
    assert _seconds_to_vtt_time(1.25) == "00:00:01.250"
    assert _seconds_to_vtt_time(3661.5) == "01:01:01.500"


def test_export_vtt_format(tmp_path):
    out = tmp_path / "test.vtt"
    export_vtt(SAMPLE_SEGMENTS, out)
    content = out.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in content
    assert "Hello world" in content
    assert "Second line" in content


def test_export_vtt_skips_empty_text(tmp_path):
    out = tmp_path / "emptyish.vtt"
    export_vtt([{"start": 0.0, "end": 1.0, "text": "  "}, {"start": 1.0, "end": 2.0, "text": "Hi"}], out)
    content = out.read_text(encoding="utf-8")
    assert "Hi" in content
    assert content.count("-->") == 1
