"""Audio preprocessing before ASR."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from config import TARGET_CHANNELS, TARGET_SAMPLE_RATE, TEMP_DIR
from utils.audio_backend import configure_audio_backend, ffmpeg_available
from utils.file_utils import ensure_dirs
from utils.logger import get_logger

logger = get_logger(__name__)

_COMPRESSED_EXTENSIONS = {".mp3", ".m4a"}


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio if amplitude is below a reasonable threshold."""
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio
    if peak < 0.5:
        return (audio / peak) * 0.95
    return audio


def preprocess_audio(input_path: Path) -> Path:
    """
    Convert audio to mono 16 kHz WAV with light normalization.

    Args:
        input_path: Path to the source audio file.

    Returns:
        Path to the preprocessed WAV file in the temp directory.
    """
    ensure_dirs()
    configure_audio_backend()

    if input_path.suffix.lower() in _COMPRESSED_EXTENSIONS and not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is required to process MP3/M4A files. "
            "Install ffmpeg (winget install Gyan.FFmpeg) or restart your terminal after installing."
        )

    output_path = TEMP_DIR / f"{input_path.stem}_preprocessed.wav"

    try:
        audio, sr = librosa.load(str(input_path), sr=TARGET_SAMPLE_RATE, mono=True)
        audio = _normalize_audio(audio.astype(np.float32))
        sf.write(str(output_path), audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
        logger.info("Preprocessed audio with librosa: %s", output_path)
        return output_path
    except Exception as librosa_err:
        logger.warning("librosa preprocessing failed (%s), falling back to pydub", librosa_err)

    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(str(input_path))
        segment = segment.set_channels(TARGET_CHANNELS).set_frame_rate(TARGET_SAMPLE_RATE)

        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        if segment.sample_width == 1:
            samples /= 128.0
        elif segment.sample_width == 2:
            samples /= 32768.0
        elif segment.sample_width == 4:
            samples /= 2147483648.0

        samples = _normalize_audio(samples)
        sf.write(str(output_path), samples, TARGET_SAMPLE_RATE, subtype="PCM_16")
        logger.info("Preprocessed audio with pydub: %s", output_path)
        return output_path
    except Exception as exc:
        raise RuntimeError(f"Audio preprocessing failed: {exc}") from exc
