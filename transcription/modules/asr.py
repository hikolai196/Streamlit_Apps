"""Automatic speech recognition using faster-whisper."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from config import DEFAULT_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level cache for non-Streamlit contexts
_model_cache: dict[str, WhisperModel] = {}

SegmentProgressCallback = Callable[[float, float], None]


def load_whisper_model(model_name: str = DEFAULT_MODEL, device: str = "auto") -> WhisperModel:
    """
    Load a Whisper model, reusing cached instance when possible.

    Args:
        model_name: Model size (small, medium, large-v3).
        device: Compute device ('auto', 'cpu', 'cuda').

    Returns:
        Loaded WhisperModel instance.
    """
    cache_key = f"{model_name}:{device}"
    if cache_key not in _model_cache:
        logger.info("Loading Whisper model: %s on %s", model_name, device)
        try:
            import torch

            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        if device == "cuda" or (device == "auto" and has_cuda):
            compute_type = "float16"
        else:
            compute_type = "int8"

        _model_cache[cache_key] = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _model_cache[cache_key]


def transcribe_audio(
    audio_path: Path,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    model: WhisperModel | None = None,
    on_segment_progress: SegmentProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """
    Transcribe audio and return segment-level results.

    Args:
        audio_path: Path to preprocessed WAV file.
        model_name: Whisper model name.
        language: Language code ('zh', 'en') or None for auto-detect.
        model: Optional pre-loaded model instance.
        on_segment_progress: Optional callback(current_seconds, total_seconds).

    Returns:
        List of segments: [{"start": float, "end": float, "text": str}, ...]
    """
    whisper_model = model or load_whisper_model(model_name)

    transcribe_kwargs: dict[str, Any] = {
        "beam_size": 5,
        "vad_filter": True,
    }
    if language:
        transcribe_kwargs["language"] = language

    try:
        segments_iter, info = whisper_model.transcribe(str(audio_path), **transcribe_kwargs)
        total_duration = float(info.duration or 0.0)
        segments: list[dict[str, Any]] = []
        for seg in segments_iter:
            segments.append(
                {
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text,
                }
            )
            if on_segment_progress:
                progress_end = seg.end if total_duration <= 0 else min(seg.end, total_duration)
                on_segment_progress(progress_end, total_duration or progress_end or 1.0)
        logger.info("Transcription complete: %d segments", len(segments))
        return segments
    except Exception as exc:
        raise RuntimeError(f"Transcription failed: {exc}") from exc
