"""File validation and metadata helpers."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_DURATION_SEC,
    MAX_UPLOAD_SIZE_MB,
    TEMP_DIR,
)
from utils.audio_backend import configure_audio_backend


def ensure_dirs() -> None:
    """Create temp and output directories if they do not exist."""
    from config import OUTPUT_DIR

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_unique_filename(filename: str) -> str:
    """
    Build a collision-safe filename preserving the original stem and suffix.

    Example: meeting.wav -> meeting_a1b2c3d4.wav
    """
    path = Path(filename).name
    stem = Path(path).stem or "audio"
    suffix = Path(path).suffix.lower()
    return f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"


def validate_audio_file(filename: str, mime_type: str | None = None) -> tuple[bool, str]:
    """
    Validate uploaded audio file by extension and optional MIME type.

    Returns:
        Tuple of (is_valid, error_message).
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"

    if mime_type and mime_type not in ALLOWED_MIME_TYPES:
        # Some browsers send generic MIME; only reject clearly wrong types
        if not mime_type.startswith("audio/") and not mime_type.startswith("video/") and mime_type != "application/octet-stream":
            return False, f"Unsupported MIME type: {mime_type}"

    return True, ""


def validate_upload_size(size_bytes: int) -> tuple[bool, str]:
    """Validate uploaded payload size against MAX_UPLOAD_SIZE_MB."""
    if size_bytes < 0:
        return False, "Invalid file size."
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        size_mb = round(size_bytes / (1024 * 1024), 2)
        return (
            False,
            f"File is too large ({size_mb} MB). Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB.",
        )
    return True, ""


def validate_audio_duration(duration_sec: float | None) -> tuple[bool, str]:
    """Validate audio duration against MAX_DURATION_SEC when known."""
    if duration_sec is None:
        return True, ""
    if duration_sec < 0:
        return False, "Invalid audio duration."
    if duration_sec > MAX_DURATION_SEC:
        max_hours = MAX_DURATION_SEC / 3600
        return (
            False,
            f"Audio is too long ({duration_sec:.1f}s). Maximum allowed duration is {max_hours:g} hours.",
        )
    return True, ""


def get_file_metadata(file_path: Path) -> dict[str, Any]:
    """Extract basic metadata from an audio file."""
    stat = file_path.stat()
    metadata: dict[str, Any] = {
        "filename": file_path.name,
        "original_filename": file_path.name,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "extension": file_path.suffix.lower(),
    }

    try:
        import soundfile as sf

        info = sf.info(str(file_path))
        metadata["duration_sec"] = round(info.duration, 2)
        metadata["sample_rate"] = info.samplerate
        metadata["channels"] = info.channels
    except Exception:
        try:
            from pydub import AudioSegment

            configure_audio_backend()
            audio = AudioSegment.from_file(str(file_path))
            metadata["duration_sec"] = round(len(audio) / 1000.0, 2)
            metadata["sample_rate"] = audio.frame_rate
            metadata["channels"] = audio.channels
        except Exception:
            metadata["duration_sec"] = None
            metadata["sample_rate"] = None
            metadata["channels"] = None

    return metadata


def save_uploaded_bytes(data: bytes, filename: str) -> Path:
    """Save uploaded bytes to the temp directory with a unique name and return the path."""
    ensure_dirs()
    dest = TEMP_DIR / make_unique_filename(filename)
    dest.write_bytes(data)
    return dest


def cleanup_temp_file(path: Path | None) -> None:
    """Remove a temporary file if it exists."""
    if path is None:
        return
    try:
        if path.exists() and path.is_file():
            os.remove(path)
    except OSError:
        pass


def cleanup_temp_files(*paths: Path | None) -> None:
    """Remove multiple temporary files if they exist."""
    for path in paths:
        cleanup_temp_file(path)
