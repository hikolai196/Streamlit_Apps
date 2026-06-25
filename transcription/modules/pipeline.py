"""End-to-end transcription pipeline for single and batch jobs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from modules.asr import transcribe_audio
from modules.postprocess import postprocess_segments, segments_to_full_text
from modules.preprocess import preprocess_audio

ProgressCallback = Callable[[float, str], None]

# Stage weights for within-file progress (must sum to 1.0)
_STAGE_PREPROCESS = 0.10
_STAGE_MODEL = 0.05
_STAGE_TRANSCRIBE = 0.75
_STAGE_POSTPROCESS = 0.10


def _report(progress: ProgressCallback | None, fraction: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, fraction)), message)


def transcribe_file(
    saved_path: Path,
    *,
    model_name: str,
    language: str | None,
    convert_traditional: bool,
    model: WhisperModel | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Run the full transcription pipeline for one audio file.

    Returns:
        Dict with keys: segments, full_text, segment_count.
    """
    _report(progress, 0.0, "Preprocessing audio...")

    def on_preprocess_done() -> None:
        _report(progress, _STAGE_PREPROCESS, "Loading model...")

    preprocessed_path = preprocess_audio(saved_path)
    on_preprocess_done()

    _report(progress, _STAGE_PREPROCESS, "Loading model...")
    active_model = model
    if active_model is None:
        from modules.asr import load_whisper_model

        active_model = load_whisper_model(model_name)

    transcribe_start = _STAGE_PREPROCESS + _STAGE_MODEL
    _report(progress, transcribe_start, "Transcribing...")

    def on_segment(current: float, total: float) -> None:
        if total <= 0:
            segment_fraction = 0.5
        else:
            segment_fraction = min(1.0, current / total)
        overall = transcribe_start + _STAGE_TRANSCRIBE * segment_fraction
        _report(progress, overall, "Transcribing...")

    raw_segments = transcribe_audio(
        preprocessed_path,
        model_name=model_name,
        language=language,
        model=active_model,
        on_segment_progress=on_segment,
    )

    post_start = transcribe_start + _STAGE_TRANSCRIBE
    _report(progress, post_start, "Post-processing...")
    segments = postprocess_segments(raw_segments, convert_to_traditional=convert_traditional)
    _report(progress, 1.0, "Done")

    return {
        "segments": segments,
        "full_text": segments_to_full_text(segments),
        "segment_count": len(segments),
    }
