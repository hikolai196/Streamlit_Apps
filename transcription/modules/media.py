"""Media helpers: video → audio extraction via ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from config import ALLOWED_VIDEO_EXTENSIONS, TARGET_SAMPLE_RATE, TEMP_DIR
from utils.audio_backend import configure_audio_backend, ffmpeg_available
from utils.file_utils import ensure_dirs, make_unique_filename
from utils.logger import get_logger

logger = get_logger(__name__)


def is_video_file(path: Path | str) -> bool:
    """Return True when the path looks like a supported video container."""
    return Path(path).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def extract_audio_from_video(video_path: Path) -> Path:
    """
    Extract mono 16 kHz WAV audio from a video file using ffmpeg.

    Returns:
        Path to the extracted WAV in TEMP_DIR.
    """
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is required to extract audio from video. "
            "Install ffmpeg (winget install Gyan.FFmpeg) and retry."
        )

    ensure_dirs()
    ffmpeg = configure_audio_backend()
    if not ffmpeg:
        raise RuntimeError("ffmpeg binary not found.")

    output_path = TEMP_DIR / make_unique_filename(f"{video_path.stem}_audio.wav")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    logger.info("Extracting audio from video: %s", video_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output_path.exists():
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        tail = detail[-3:] if detail else ["unknown ffmpeg error"]
        raise RuntimeError("Video audio extraction failed: " + " | ".join(tail))
    return output_path
