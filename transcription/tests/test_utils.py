"""Tests for file and time utilities."""

from utils.file_utils import validate_audio_file
from utils.time_utils import format_duration, seconds_to_hms, seconds_to_srt_time


def test_validate_audio_file_accepts_wav():
    valid, err = validate_audio_file("meeting.wav")
    assert valid is True
    assert err == ""


def test_validate_audio_file_rejects_unknown():
    valid, err = validate_audio_file("notes.pdf")
    assert valid is False
    assert "Unsupported" in err


def test_seconds_to_hms():
    assert seconds_to_hms(3661.5) == "01:01:01.500"


def test_seconds_to_srt_time():
    assert seconds_to_srt_time(1.25) == "00:00:01,250"


def test_format_duration():
    assert format_duration(None) == "Estimating..."
    assert format_duration(45) == "45s"
    assert format_duration(125) == "2m 5s"
