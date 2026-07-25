"""Configure audio decoding backends (ffmpeg for pydub)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

_configured = False


def _find_ffmpeg() -> str | None:
    """Locate an ffmpeg executable on the system or via imageio-ffmpeg."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).exists():
            return bundled
    except Exception:
        pass

    return None


def configure_audio_backend() -> str | None:
    """
    Configure pydub to use an available ffmpeg binary.

    Returns:
        Path to ffmpeg if found, else None.
    """
    global _configured
    if _configured:
        return _find_ffmpeg()

    ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path:
        try:
            os.environ["FFMPEG_BINARY"] = ffmpeg_path
            from pydub import AudioSegment

            AudioSegment.converter = ffmpeg_path
            ffprobe = shutil.which("ffprobe")
            if ffprobe:
                AudioSegment.ffprobe = ffprobe
            logger.info("Using ffmpeg at: %s", ffmpeg_path)
        except Exception as exc:
            logger.warning("Failed to configure pydub ffmpeg: %s", exc)
    else:
        logger.warning("ffmpeg not found; MP3/M4A decoding may fail")

    _configured = True
    return ffmpeg_path


def ffmpeg_available() -> bool:
    """Return True if ffmpeg is available for compressed audio formats."""
    return configure_audio_backend() is not None
