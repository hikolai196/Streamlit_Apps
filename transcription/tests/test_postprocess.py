"""Tests for post-processing utilities."""

from modules.postprocess import (
    clean_text,
    group_segments_by_silence,
    postprocess_segments,
    segments_to_full_text,
    to_traditional_chinese,
)


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


def test_group_segments_by_silence():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 1.2, "end": 2.0, "text": "there"},
        {"start": 4.0, "end": 5.0, "text": "Later"},
    ]
    groups = group_segments_by_silence(segments, gap_sec=1.5)
    assert len(groups) == 2
    assert [seg["text"] for seg in groups[0]] == ["Hello", "there"]
    assert [seg["text"] for seg in groups[1]] == ["Later"]


def test_segments_to_full_text_with_paragraphs():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 1.2, "end": 2.0, "text": "there"},
        {"start": 4.0, "end": 5.0, "text": "Later"},
    ]
    text = segments_to_full_text(segments, group_paragraphs=True, paragraph_gap_sec=1.5)
    assert text == "Hello there\n\nLater"
