"""Post-processing of transcription segments."""

from __future__ import annotations

import re
from typing import Any

from opencc import OpenCC

from config import OPENCC_CONFIG

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
        processed.append(
            {
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": text,
            }
        )
    return processed


def segments_to_full_text(segments: list[dict[str, Any]]) -> str:
    """Join segment texts into a single transcript string."""
    parts = [seg["text"] for seg in segments if seg.get("text")]
    return "\n".join(parts)
