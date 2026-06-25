"""Export transcript to Word DOCX format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docx import Document

from modules.postprocess import segments_to_full_text
from utils.time_utils import seconds_to_hms


def export_docx(
    segments: list[dict[str, Any]],
    output_path: Path,
    include_timestamps: bool = True,
) -> Path:
    """
    Export segments to a DOCX file.

    Args:
        segments: Transcription segments.
        output_path: Destination file path.
        include_timestamps: Whether to include timestamp prefixes.

    Returns:
        Path to the written file.
    """
    doc = Document()
    doc.add_heading("Meeting Transcript", level=1)

    if include_timestamps:
        for seg in segments:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            start = seconds_to_hms(float(seg["start"]))
            end = seconds_to_hms(float(seg["end"]))
            doc.add_paragraph(f"[{start} -> {end}] {text}")
    else:
        doc.add_paragraph(segments_to_full_text(segments))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    return output_path
