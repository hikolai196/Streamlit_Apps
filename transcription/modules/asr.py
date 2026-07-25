"""Automatic speech recognition using faster-whisper."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from config import DEFAULT_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level cache for non-Streamlit contexts (CLI/tests/scripts).
# Streamlit UI should prefer app.get_cached_model (@st.cache_resource), which
# calls load_whisper_model and therefore shares this cache for the same key.
# Prefer passing an already-loaded model into transcribe_audio / transcribe_file
# so device/lifecycle stay under the caller's control.
_model_cache: dict[str, WhisperModel] = {}

SegmentProgressCallback = Callable[[float, float], None]


def cuda_available() -> bool:
    """Return True when PyTorch reports a usable CUDA device."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_device(requested: str = "auto") -> tuple[str, str | None]:
    """
    Resolve a user-facing device choice to a Whisper device string.

    Returns:
        (device, warning_message). warning_message is set when falling back.
    """
    choice = (requested or "auto").strip().lower()
    if choice not in {"auto", "cpu", "cuda"}:
        choice = "auto"

    has_cuda = cuda_available()
    if choice == "cpu":
        return "cpu", None
    if choice == "cuda":
        if has_cuda:
            return "cuda", None
        return "cpu", "CUDA was requested but is unavailable; using CPU instead."
    # auto
    if has_cuda:
        return "cuda", None
    return "cpu", None


def build_prompt_bias(hotwords: str | None = None, initial_prompt: str | None = None) -> str | None:
    """
    Build an initial prompt string from optional free-text prompt and hotwords.

    Hotwords may be comma- or newline-separated.
    """
    parts: list[str] = []
    if initial_prompt and initial_prompt.strip():
        parts.append(initial_prompt.strip())

    if hotwords and hotwords.strip():
        tokens = []
        for chunk in hotwords.replace("\n", ",").split(","):
            token = chunk.strip()
            if token:
                tokens.append(token)
        if tokens:
            parts.append(", ".join(tokens))

    if not parts:
        return None
    return " ".join(parts)


def load_whisper_model(model_name: str = DEFAULT_MODEL, device: str = "auto") -> WhisperModel:
    """
    Load a Whisper model, reusing cached instance when possible.

    Args:
        model_name: Model size (small, medium, large-v3).
        device: Compute device ('auto', 'cpu', 'cuda').

    Returns:
        Loaded WhisperModel instance.
    """
    resolved, warning = resolve_device(device)
    if warning:
        logger.warning(warning)

    cache_key = f"{model_name}:{resolved}"
    if cache_key not in _model_cache:
        logger.info("Loading Whisper model: %s on %s", model_name, resolved)
        compute_type = "float16" if resolved == "cuda" else "int8"
        try:
            _model_cache[cache_key] = WhisperModel(
                model_name,
                device=resolved,
                compute_type=compute_type,
            )
        except Exception as exc:
            if resolved != "cpu":
                logger.warning("Failed to load model on %s (%s); falling back to CPU", resolved, exc)
                cache_key = f"{model_name}:cpu"
                if cache_key not in _model_cache:
                    _model_cache[cache_key] = WhisperModel(model_name, device="cpu", compute_type="int8")
            else:
                raise
    return _model_cache[cache_key]


def transcribe_audio(
    audio_path: Path,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    model: WhisperModel | None = None,
    on_segment_progress: SegmentProgressCallback | None = None,
    device: str = "auto",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
) -> list[dict[str, Any]]:
    """
    Transcribe audio and return segment-level results.

    Args:
        audio_path: Path to preprocessed WAV file.
        model_name: Whisper model name.
        language: Language code ('zh', 'en') or None for auto-detect.
        model: Optional pre-loaded model instance.
        on_segment_progress: Optional callback(current_seconds, total_seconds).
        device: Device used when ``model`` is not provided.
        initial_prompt: Optional prompt bias for the decoder.
        hotwords: Optional comma/newline-separated vocabulary hints.

    Returns:
        List of segments: [{"start": float, "end": float, "text": str}, ...]
    """
    whisper_model = model or load_whisper_model(model_name, device=device)

    transcribe_kwargs: dict[str, Any] = {
        "beam_size": 5,
        "vad_filter": True,
    }
    if language:
        transcribe_kwargs["language"] = language

    prompt = build_prompt_bias(hotwords=hotwords, initial_prompt=initial_prompt)
    if prompt:
        transcribe_kwargs["initial_prompt"] = prompt
        # Newer faster-whisper builds also accept a dedicated hotwords string.
        if hotwords and hotwords.strip():
            transcribe_kwargs["hotwords"] = ", ".join(
                token.strip()
                for token in hotwords.replace("\n", ",").split(",")
                if token.strip()
            )

    try:
        try:
            segments_iter, info = whisper_model.transcribe(str(audio_path), **transcribe_kwargs)
        except TypeError:
            # Older faster-whisper without hotwords= support.
            transcribe_kwargs.pop("hotwords", None)
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
