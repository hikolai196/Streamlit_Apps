"""Export transcript to plain text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modules.postprocess import segments_to_full_text
from utils.time_utils import seconds_to_hms


def export_txt(
    segments: list[dict[str, Any]],
    output_path: Path,
    include_timestamps: bool = True,
) -> Path:
    """
    Export segments to a TXT file.

    Args:
        segments: Transcription segments.
        output_path: Destination file path.
        include_timestamps: Whether to prefix lines with timestamps.

    Returns:
        Path to the written file.
    """
    lines: list[str] = []
    if include_timestamps:
        for seg in segments:
            if not seg.get("text"):
                continue
            start = seconds_to_hms(float(seg["start"]))
            end = seconds_to_hms(float(seg["end"]))
            lines.append(f"[{start} -> {end}] {seg['text']}")
    else:
        lines.append(segments_to_full_text(segments))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path
