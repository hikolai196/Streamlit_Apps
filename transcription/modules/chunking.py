"""Split long audio into chunks and merge timed segments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from config import (
    CHUNK_DURATION_SEC,
    CHUNK_MIN_DURATION_SEC,
    CHUNK_OVERLAP_SEC,
    TEMP_DIR,
)
from utils.file_utils import ensure_dirs, make_unique_filename
from utils.logger import get_logger

logger = get_logger(__name__)


def audio_duration_seconds(path: Path) -> float:
    """Return duration in seconds for a WAV/audio file."""
    info = sf.info(str(path))
    return float(info.duration)


def should_chunk(duration_sec: float, *, enabled: bool = True) -> bool:
    """Whether the clip is long enough to benefit from chunking."""
    return bool(enabled) and duration_sec >= CHUNK_MIN_DURATION_SEC


def split_audio_chunks(
    audio_path: Path,
    *,
    chunk_duration_sec: float = CHUNK_DURATION_SEC,
    overlap_sec: float = CHUNK_OVERLAP_SEC,
) -> list[tuple[Path, float]]:
    """
    Split audio into overlapping WAV chunks.

    Returns:
        List of (chunk_path, start_offset_seconds).
    """
    ensure_dirs()
    audio, sr = sf.read(str(audio_path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)

    total = len(audio)
    if total == 0:
        return []

    chunk_samples = max(1, int(chunk_duration_sec * sr))
    overlap_samples = max(0, int(overlap_sec * sr))
    hop = max(1, chunk_samples - overlap_samples)

    chunks: list[tuple[Path, float]] = []
    start = 0
    index = 0
    while start < total:
        end = min(total, start + chunk_samples)
        piece = audio[start:end]
        offset_sec = start / float(sr)
        out = TEMP_DIR / make_unique_filename(f"{audio_path.stem}_chunk{index:03d}.wav")
        sf.write(str(out), piece, sr, subtype="PCM_16")
        chunks.append((out, offset_sec))
        index += 1
        if end >= total:
            break
        start += hop

    logger.info("Split %s into %d chunk(s)", audio_path.name, len(chunks))
    return chunks


def shift_segments(segments: list[dict[str, Any]], offset_sec: float) -> list[dict[str, Any]]:
    """Add ``offset_sec`` to each segment's start/end."""
    shifted: list[dict[str, Any]] = []
    for seg in segments:
        item = dict(seg)
        item["start"] = round(float(seg.get("start", 0.0)) + offset_sec, 2)
        item["end"] = round(float(seg.get("end", 0.0)) + offset_sec, 2)
        shifted.append(item)
    return shifted


def merge_chunk_segments(
    chunk_results: list[tuple[float, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """
    Merge per-chunk segments ordered by time.

    Drops near-duplicate cues that overlap heavily from chunk overlaps.
    """
    merged: list[dict[str, Any]] = []
    for offset, segments in chunk_results:
        for seg in shift_segments(segments, offset):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            if merged:
                prev = merged[-1]
                same_text = str(prev.get("text", "")).strip() == text
                overlap = float(seg["start"]) < float(prev["end"]) - 0.25
                if same_text and overlap:
                    prev["end"] = max(float(prev["end"]), float(seg["end"]))
                    continue
            merged.append(seg)
    merged.sort(key=lambda s: float(s.get("start", 0.0)))
    return merged
