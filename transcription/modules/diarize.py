"""Lightweight speaker labeling helpers (gap-based heuristic)."""

from __future__ import annotations

from typing import Any

from config import DIARIZATION_GAP_SEC


def assign_speakers_by_gap(
    segments: list[dict[str, Any]],
    *,
    gap_sec: float = DIARIZATION_GAP_SEC,
    max_speakers: int = 2,
) -> list[dict[str, Any]]:
    """
    Assign alternating SPEAKER_XX labels when silence gaps suggest turn-taking.

    This is a heuristic stand-in for full neural diarization (e.g. pyannote).
    It helps exports and review without requiring heavy optional dependencies.
    """
    if not segments:
        return []

    labeled: list[dict[str, Any]] = []
    speaker_idx = 0
    prev_end: float | None = None

    for seg in segments:
        item = dict(seg)
        start = float(seg.get("start", 0.0))
        if prev_end is not None and (start - prev_end) >= gap_sec:
            speaker_idx = (speaker_idx + 1) % max(1, max_speakers)
        item["speaker"] = f"SPEAKER_{speaker_idx:02d}"
        labeled.append(item)
        prev_end = float(seg.get("end", start))

    return labeled


def format_segment_line(seg: dict[str, Any], *, include_speaker: bool = True) -> str:
    """Format a segment's display/export text, optionally with speaker tag."""
    text = str(seg.get("text", "")).strip()
    speaker = seg.get("speaker")
    if include_speaker and speaker:
        return f"[{speaker}] {text}"
    return text
