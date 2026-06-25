"""Upload handling for audio files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.file_utils import get_file_metadata, save_uploaded_bytes, validate_audio_file


def process_upload(
    uploaded_file: Any,
) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    """
    Validate and save an uploaded Streamlit file object.

    Returns:
        Tuple of (saved_path, metadata, error_message).
    """
    if uploaded_file is None:
        return None, None, None

    is_valid, error = validate_audio_file(uploaded_file.name, uploaded_file.type)
    if not is_valid:
        return None, None, error

    try:
        saved_path = save_uploaded_bytes(uploaded_file.getvalue(), uploaded_file.name)
        metadata = get_file_metadata(saved_path)
        return saved_path, metadata, None
    except Exception as exc:
        return None, None, f"Failed to save uploaded file: {exc}"


def process_uploads(uploaded_files: list[Any]) -> list[dict[str, Any]]:
    """
    Validate and save multiple uploaded files.

    Returns:
        List of dicts with filename, path, metadata, and error keys.
    """
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

