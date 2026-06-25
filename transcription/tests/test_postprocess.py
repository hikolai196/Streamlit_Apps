"""Tests for post-processing utilities."""

from modules.postprocess import clean_text, postprocess_segments, segments_to_full_text, to_traditional_chinese


def test_clean_text_trims_and_collapses_spaces():
    assert clean_text("  hello   world  ") == "hello world"


def test_postprocess_segments_trims_text():
    segments = [{"start": 0.0, "end": 1.0, "text": "  hello  "}]
    result = postprocess_segments(segments, convert_to_traditional=False)
    assert result[0]["text"] == "hello"


def test_to_traditional_chinese_converts_simplified():
    # 软件 -> 軟件
    result = to_traditional_chinese("软件")
    assert "軟" in result


def test_to_traditional_preserves_english():
    result = to_traditional_chinese("Hello 世界")
    assert "Hello" in result
    assert "界" in result


def test_segments_to_full_text():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Line one"},
        {"start": 1.0, "end": 2.0, "text": "Line two"},
    ]
    assert segments_to_full_text(segments) == "Line one\nLine two"
