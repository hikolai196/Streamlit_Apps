"""Export transcript to plain text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import DEFAULT_PARAGRAPH_GAP_SEC
from modules.diarize import format_segment_line
from modules.postprocess import segments_to_full_text
from utils.time_utils import seconds_to_hms


def export_txt(
    segments: list[dict[str, Any]],
    output_path: Path,
    include_timestamps: bool = True,
    *,
    group_paragraphs: bool = False,
    paragraph_gap_sec: float = DEFAULT_PARAGRAPH_GAP_SEC,
) -> Path:
    """Export segments to a TXT file."""
    lines: list[str] = []
    if include_timestamps:
        for seg in segments:
            if not seg.get("text"):
                continue
            start = seconds_to_hms(float(seg["start"]))
            end = seconds_to_hms(float(seg["end"]))
            lines.append(f"[{start} -> {end}] {format_segment_line(seg)}")
    else:
        lines.append(
            segments_to_full_text(
                segments,
                group_paragraphs=group_paragraphs,
                paragraph_gap_sec=paragraph_gap_sec,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path
