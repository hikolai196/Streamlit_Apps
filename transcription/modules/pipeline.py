"""End-to-end transcription pipeline for single and batch jobs."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from config import CHUNK_DURATION_SEC, CHUNK_OVERLAP_SEC, DEFAULT_PARAGRAPH_GAP_SEC, DIARIZATION_GAP_SEC
from modules.asr import transcribe_audio
from modules.chunking import (
    audio_duration_seconds,
    merge_chunk_segments,
    should_chunk,
    split_audio_chunks,
)
from modules.diarize import assign_speakers_by_gap
from modules.media import extract_audio_from_video, is_video_file
from modules.postprocess import postprocess_segments, segments_to_full_text
from modules.preprocess import preprocess_audio
from modules.summary import build_meeting_notes
from utils.file_utils import cleanup_temp_file, cleanup_temp_files

ProgressCallback = Callable[[float, str], None]

# Shared lock when one Whisper model is used from multiple worker threads.
_model_lock = threading.Lock()

# Stage weights for within-file progress (must sum to 1.0).
_STAGE_PREPROCESS = 0.08
_STAGE_MODEL = 0.02
_STAGE_TRANSCRIBE = 0.85
_STAGE_POSTPROCESS = 0.05


def _report(progress: ProgressCallback | None, fraction: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, fraction)), message)


def _transcribe_path(
    audio_path: Path,
    *,
    model_name: str,
    language: str | None,
    model: WhisperModel,
    device: str,
    initial_prompt: str | None,
    hotwords: str | None,
    on_segment_progress: Callable[[float, float], None] | None,
) -> list[dict[str, Any]]:
    with _model_lock:
        return transcribe_audio(
            audio_path,
            model_name=model_name,
            language=language,
            model=model,
            on_segment_progress=on_segment_progress,
            device=device,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )


def transcribe_file(
    saved_path: Path,
    *,
    model_name: str,
    language: str | None,
    convert_traditional: bool,
    model: WhisperModel | None = None,
    progress: ProgressCallback | None = None,
    cleanup_source: bool = False,
    device: str = "auto",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    group_paragraphs: bool = False,
    paragraph_gap_sec: float = DEFAULT_PARAGRAPH_GAP_SEC,
    enable_chunking: bool = True,
    enable_diarization: bool = False,
    enable_summary: bool = False,
    diarization_gap_sec: float = DIARIZATION_GAP_SEC,
) -> dict[str, Any]:
    """
    Run the full transcription pipeline for one media file.

    Supports video inputs (audio extraction), optional long-audio chunking,
    gap-based speaker labels, and extractive meeting notes.
    """
    preprocessed_path: Path | None = None
    extracted_audio: Path | None = None
    chunk_paths: list[Path] = []
    source_path = saved_path

    try:
        _report(progress, 0.0, "Preparing media...")
        if is_video_file(source_path):
            extracted_audio = extract_audio_from_video(source_path)
            source_path = extracted_audio

        _report(progress, 0.03, "Preprocessing audio...")
        preprocessed_path = preprocess_audio(source_path)
        _report(progress, _STAGE_PREPROCESS, "Loading model...")

        active_model = model
        if active_model is None:
            from modules.asr import load_whisper_model

            active_model = load_whisper_model(model_name, device=device)

        transcribe_start = _STAGE_PREPROCESS + _STAGE_MODEL
        _report(progress, transcribe_start, "Transcribing...")

        duration = audio_duration_seconds(preprocessed_path)

        def on_segment(current: float, total: float) -> None:
            if total <= 0:
                segment_fraction = 0.5
            else:
                segment_fraction = min(1.0, current / total)
            overall = transcribe_start + _STAGE_TRANSCRIBE * segment_fraction
            _report(progress, overall, "Transcribing...")

        if should_chunk(duration, enabled=enable_chunking):
            chunks = split_audio_chunks(
                preprocessed_path,
                chunk_duration_sec=CHUNK_DURATION_SEC,
                overlap_sec=CHUNK_OVERLAP_SEC,
            )
            chunk_paths = [path for path, _ in chunks]
            chunk_results: list[tuple[float, list[dict[str, Any]]]] = []
            for index, (chunk_path, offset) in enumerate(chunks):
                _report(
                    progress,
                    transcribe_start + _STAGE_TRANSCRIBE * (index / max(len(chunks), 1)),
                    f"Transcribing chunk {index + 1}/{len(chunks)}...",
                )
                raw = _transcribe_path(
                    chunk_path,
                    model_name=model_name,
                    language=language,
                    model=active_model,
                    device=device,
                    initial_prompt=initial_prompt,
                    hotwords=hotwords,
                    on_segment_progress=None,
                )
                chunk_results.append((offset, raw))
            raw_segments = merge_chunk_segments(chunk_results)
        else:
            raw_segments = _transcribe_path(
                preprocessed_path,
                model_name=model_name,
                language=language,
                model=active_model,
                device=device,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
                on_segment_progress=on_segment,
            )

        post_start = transcribe_start + _STAGE_TRANSCRIBE
        _report(progress, post_start, "Post-processing...")
        segments = postprocess_segments(raw_segments, convert_to_traditional=convert_traditional)
        if enable_diarization:
            segments = assign_speakers_by_gap(segments, gap_sec=diarization_gap_sec)

        full_text = segments_to_full_text(
            segments,
            group_paragraphs=group_paragraphs,
            paragraph_gap_sec=paragraph_gap_sec,
        )
        notes = build_meeting_notes(segments) if enable_summary else {"summary": "", "action_items": ""}
        _report(progress, 1.0, "Done")

        return {
            "segments": segments,
            "full_text": full_text,
            "segment_count": len(segments),
            "summary": notes.get("summary", ""),
            "action_items": notes.get("action_items", ""),
        }
    finally:
        cleanup_temp_files(preprocessed_path, extracted_audio, *chunk_paths)
        if cleanup_source:
            cleanup_temp_file(saved_path)
