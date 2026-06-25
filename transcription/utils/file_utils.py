"""File validation and metadata helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from config import ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, TEMP_DIR
from utils.audio_backend import configure_audio_backend


def ensure_dirs() -> None:
    """Create temp and output directories if they do not exist."""
    from config import OUTPUT_DIR

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
        if not mime_type.startswith("audio/") and mime_type != "application/octet-stream":
            return False, f"Unsupported MIME type: {mime_type}"

    return True, ""


def get_file_metadata(file_path: Path) -> dict[str, Any]:
    """Extract basic metadata from an audio file."""
    stat = file_path.stat()
    metadata: dict[str, Any] = {
        "filename": file_path.name,
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
    """Save uploaded bytes to the temp directory and return the path."""
    ensure_dirs()
    safe_name = Path(filename).name
    dest = TEMP_DIR / safe_name
    dest.write_bytes(data)
    return dest


def cleanup_temp_file(path: Path) -> None:
    """Remove a temporary file if it exists."""
    try:
        if path.exists():
            os.remove(path)
    except OSError:
        pass
