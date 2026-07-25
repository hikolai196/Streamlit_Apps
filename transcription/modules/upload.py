"""Upload handling for audio and video files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modules.media import extract_audio_from_video, is_video_file
from utils.file_utils import (
    cleanup_temp_file,
    get_file_metadata,
    save_uploaded_bytes,
    validate_audio_duration,
    validate_audio_file,
    validate_upload_size,
)


def process_upload(
    uploaded_file: Any,
) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    """
    Validate and save an uploaded Streamlit file object.

    Video uploads are converted to WAV via ffmpeg before metadata checks.
    """
    if uploaded_file is None:
        return None, None, None

    is_valid, error = validate_audio_file(uploaded_file.name, uploaded_file.type)
    if not is_valid:
        return None, None, error

    data = uploaded_file.getvalue()
    size_ok, size_error = validate_upload_size(len(data))
    if not size_ok:
        return None, None, size_error

    saved_path: Path | None = None
    try:
        saved_path = save_uploaded_bytes(data, uploaded_file.name)
        working_path = saved_path
        if is_video_file(saved_path):
            working_path = extract_audio_from_video(saved_path)
            cleanup_temp_file(saved_path)
            saved_path = working_path

        metadata = get_file_metadata(saved_path)
        metadata["original_filename"] = Path(uploaded_file.name).name
        metadata["filename"] = metadata["original_filename"]

        duration_ok, duration_error = validate_audio_duration(metadata.get("duration_sec"))
        if not duration_ok:
            cleanup_temp_file(saved_path)
            return None, None, duration_error

        return saved_path, metadata, None
    except Exception as exc:
        if saved_path:
            cleanup_temp_file(saved_path)
        return None, None, f"Failed to save uploaded file: {exc}"


def process_uploads(uploaded_files: list[Any]) -> list[dict[str, Any]]:
    """Validate and save multiple uploaded files."""
    items: list[dict[str, Any]] = []
    for uploaded_file in uploaded_files:
        path, metadata, error = process_upload(uploaded_file)
        items.append(
            {
                "filename": uploaded_file.name,
                "path": path,
                "metadata": metadata,
                "error": error,
            }
        )
    return items
