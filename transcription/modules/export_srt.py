"""Export transcript to SRT subtitle format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import srt
from datetime import timedelta

from modules.diarize import format_segment_line


def _seconds_to_timedelta(seconds: float) -> timedelta:
    """Convert float seconds to timedelta for srt library."""
    millis = int(round(seconds * 1000))
    return timedelta(milliseconds=millis)


def export_srt(segments: list[dict[str, Any]], output_path: Path) -> Path:
    """
    Export segments to an SRT subtitle file.

    Args:
        segments: Transcription segments.
        output_path: Destination file path.

    Returns:
        Path to the written file.
    """
    subtitles: list[srt.Subtitle] = []
    index = 1
    for seg in segments:
        text = format_segment_line(seg).strip()
        if not text:
            continue
        subtitles.append(
            srt.Subtitle(
                index=index,
                start=_seconds_to_timedelta(float(seg["start"])),
                end=_seconds_to_timedelta(float(seg["end"])),
                content=text,
            )
        )
        index += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(srt.compose(subtitles), encoding="utf-8")
    return output_path
