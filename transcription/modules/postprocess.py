"""Post-processing of transcription segments."""

from __future__ import annotations

import re
from typing import Any

from opencc import OpenCC

from config import DEFAULT_PARAGRAPH_GAP_SEC, OPENCC_CONFIG

_cc: OpenCC | None = None


def _get_opencc() -> OpenCC:
    """Lazy-load OpenCC converter."""
    global _cc
    if _cc is None:
        _cc = OpenCC(OPENCC_CONFIG)
    return _cc


def clean_text(text: str) -> str:
    """Trim whitespace and collapse repeated spaces."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def to_traditional_chinese(text: str) -> str:
    """
    Convert Chinese characters to Traditional Chinese while preserving English.

    OpenCC handles mixed content reasonably; English words are left unchanged.
    """
    if not text:
        return text
    return _get_opencc().convert(text)


def postprocess_segments(
    segments: list[dict[str, Any]],
    convert_to_traditional: bool = False,
) -> list[dict[str, Any]]:
    """
    Post-process transcription segments.

    Args:
        segments: Raw ASR segments with start, end, text keys.
        convert_to_traditional: Whether to convert Chinese to Traditional.

    Returns:
        Cleaned segment list.
    """
    processed: list[dict[str, Any]] = []
    for seg in segments:
        text = clean_text(str(seg.get("text", "")))
        if convert_to_traditional and text:
            text = to_traditional_chinese(text)
        item: dict[str, Any] = {
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": text,
        }
        if seg.get("speaker"):
            item["speaker"] = seg["speaker"]
        processed.append(item)
    return processed


def group_segments_by_silence(
    segments: list[dict[str, Any]],
    gap_sec: float = DEFAULT_PARAGRAPH_GAP_SEC,
) -> list[list[dict[str, Any]]]:
    """
    Split segments into paragraph groups when the silence gap exceeds ``gap_sec``.
    """
    if not segments:
        return []

    groups: list[list[dict[str, Any]]] = [[segments[0]]]
    for seg in segments[1:]:
        prev = groups[-1][-1]
        gap = float(seg.get("start", 0.0)) - float(prev.get("end", 0.0))
        if gap >= gap_sec:
            groups.append([seg])
        else:
            groups[-1].append(seg)
    return groups


def segments_to_full_text(
    segments: list[dict[str, Any]],
    *,
    group_paragraphs: bool = False,
    paragraph_gap_sec: float = DEFAULT_PARAGRAPH_GAP_SEC,
) -> str:
    """
    Join segment texts into a transcript string.

    When ``group_paragraphs`` is True, insert a blank line between groups
    separated by silence gaps of at least ``paragraph_gap_sec``.
    Speaker labels are included when present.
    """
    from modules.diarize import format_segment_line

    if not group_paragraphs:
        parts = [format_segment_line(seg) for seg in segments if seg.get("text")]
        return "\n".join(parts)

    paragraphs: list[str] = []
    for group in group_segments_by_silence(segments, gap_sec=paragraph_gap_sec):
        text = " ".join(format_segment_line(seg) for seg in group if seg.get("text")).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)
