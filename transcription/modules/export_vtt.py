"""Export transcript to WebVTT subtitle format."""

from __future__ import annotations

from pathlib import Path
from typing import Any


from modules.diarize import format_segment_line


def _seconds_to_vtt_time(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp HH:MM:SS.mmm."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def export_vtt(segments: list[dict[str, Any]], output_path: Path) -> Path:
    """
    Export segments to a WebVTT subtitle file.

    Args:
        segments: Transcription segments.
        output_path: Destination file path.

    Returns:
        Path to the written file.
    """
    lines = ["WEBVTT", ""]
    for seg in segments:
        text = format_segment_line(seg).strip()
        if not text:
            continue
        start = _seconds_to_vtt_time(float(seg["start"]))
        end = _seconds_to_vtt_time(float(seg["end"]))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
