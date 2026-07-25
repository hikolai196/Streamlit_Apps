"""Bundle TXT / SRT / VTT / DOCX exports for a transcription result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import DEFAULT_PARAGRAPH_GAP_SEC, OUTPUT_DIR
from modules.export_docx import export_docx
from modules.export_srt import export_srt
from modules.export_txt import export_txt
from modules.export_vtt import export_vtt
from utils.file_utils import ensure_dirs, make_unique_filename


def write_exports(
    segments: list[dict[str, Any]],
    base_name: str,
    include_timestamps: bool,
    *,
    group_paragraphs: bool = False,
    paragraph_gap_sec: float = DEFAULT_PARAGRAPH_GAP_SEC,
) -> dict[str, Path]:
    """
    Write TXT, SRT, VTT, and DOCX exports to the output directory.

    Disk filenames are uniquified to avoid overwrites; callers can still
    present ``Path(base_name).stem`` as the user-facing download name.
    """
    ensure_dirs()
    stem = Path(base_name).stem or "transcript"
    paths = {
        "txt": OUTPUT_DIR / make_unique_filename(f"{stem}.txt"),
        "srt": OUTPUT_DIR / make_unique_filename(f"{stem}.srt"),
        "vtt": OUTPUT_DIR / make_unique_filename(f"{stem}.vtt"),
        "docx": OUTPUT_DIR / make_unique_filename(f"{stem}.docx"),
    }
    export_txt(
        segments,
        paths["txt"],
        include_timestamps=include_timestamps,
        group_paragraphs=group_paragraphs,
        paragraph_gap_sec=paragraph_gap_sec,
    )
    export_srt(segments, paths["srt"])
    export_vtt(segments, paths["vtt"])
    export_docx(
        segments,
        paths["docx"],
        include_timestamps=include_timestamps,
        group_paragraphs=group_paragraphs,
        paragraph_gap_sec=paragraph_gap_sec,
    )
    return paths


def write_exports_if_any(
    segments: list[dict[str, Any]],
    base_name: str,
    include_timestamps: bool,
    *,
    group_paragraphs: bool = False,
    paragraph_gap_sec: float = DEFAULT_PARAGRAPH_GAP_SEC,
) -> dict[str, Path]:
    """Write exports only when there is at least one segment; otherwise return {}."""
    if not segments:
        return {}
    return write_exports(
        segments,
        base_name,
        include_timestamps,
        group_paragraphs=group_paragraphs,
        paragraph_gap_sec=paragraph_gap_sec,
    )


def download_filename(base_name: str, extension: str) -> str:
    """User-facing download name from the original upload stem (path-safe)."""
    safe_name = Path(base_name).name
    stem = Path(safe_name).stem or "transcript"
    ext = extension if extension.startswith(".") else f".{extension}"
    return f"{stem}{ext}"
