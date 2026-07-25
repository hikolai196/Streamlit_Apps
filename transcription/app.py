"""Streamlit meeting transcription app."""

from __future__ import annotations

import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from faster_whisper import WhisperModel

from config import (
    CHUNK_ENABLED_DEFAULT,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE_LABEL,
    DEFAULT_MODEL,
    DEFAULT_PARAGRAPH_GAP_SEC,
    DEVICE_OPTIONS,
    LANGUAGE_OPTIONS,
    MAX_DURATION_SEC,
    MAX_PARALLEL_JOBS_DEFAULT,
    MAX_PARALLEL_JOBS_OPTIONS,
    MAX_UPLOAD_SIZE_MB,
    WHISPER_MODELS,
)
from modules.asr import cuda_available, resolve_device
from modules.export_bundle import write_exports_if_any
from modules.pipeline import transcribe_file
from modules.upload import process_upload, process_uploads
from ui.batch import (
    STATUS_ALL,
    STATUS_FILTER_OPTIONS,
    cancelled_result,
    filter_batch_results,
    merge_batch_results,
    retryable_batch_items,
)
from ui.components import (
    render_metadata,
    render_progress_ui,
    render_results,
    render_theme_toggle,
)
from utils.audio_backend import configure_audio_backend, ffmpeg_available
from utils.auth import password_required, verify_password
from utils.file_utils import cleanup_temp_file, cleanup_temp_files, ensure_dirs
from utils.logger import get_logger
from utils.progress import ProgressTracker
from utils.theme import DEFAULT_THEME, apply_streamlit_theme, normalize_theme

UPLOAD_TYPES = ["wav", "mp3", "m4a", "mp4", "mkv", "webm", "mov"]
UPLOAD_HELP = (
    f"Audio: wav/mp3/m4a · Video: mp4/mkv/webm/mov. "
    f"Max size {MAX_UPLOAD_SIZE_MB} MB, max duration {MAX_DURATION_SEC // 3600} hours."
)

logger = get_logger(__name__)

st.set_page_config(
    page_title="Meeting Transcription",
    page_icon="🎙️",
    layout="wide",
)

configure_audio_backend()


@st.cache_resource(show_spinner="Loading Whisper model...")
def get_cached_model(model_name: str, device: str = "auto") -> WhisperModel:
    """Load and cache Whisper model for Streamlit sessions (keyed by device)."""
    from modules.asr import load_whisper_model

    resolved, _warning = resolve_device(device)
    return load_whisper_model(model_name, device=resolved)


def _friendly_error(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return "An unexpected error occurred. Please try again with a different file or settings."


def _log_error(exc: Exception) -> None:
    logger.error("Pipeline error: %s\n%s", exc, traceback.format_exc())


def _init_session_state() -> None:
    defaults = {
        "segments": [],
        "full_text": "",
        "summary": "",
        "action_items": "",
        "source_name": "transcript",
        "audio_path": None,
        "last_upload_key": None,
        "batch_results": [],
        "selected_batch_file": None,
        "batch_status_filter": STATUS_ALL,
        "cached_single_upload_key": None,
        "cached_single_path": None,
        "cached_single_metadata": None,
        "cached_single_error": None,
        "cached_batch_upload_key": None,
        "cached_batch_items": None,
        "ui_theme": DEFAULT_THEME,
        "ui_theme_applied": False,
        "batch_running": False,
        "batch_queue": [],
        "batch_job": None,
        "cancel_requested": False,
        "authenticated": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_auth_gate() -> bool:
    """Return True when the user may use the app."""
    if not password_required():
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("🔒 Meeting Transcription")
    st.caption("This deployment is password-protected. Set TRANSCRIPTION_APP_PASSWORD to enable.")
    password = st.text_input("Password", type="password")
    if st.button("Sign in", type="primary"):
        if verify_password(password):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Incorrect password.")
    return False


def _clear_single_results() -> None:
    st.session_state["segments"] = []
    st.session_state["full_text"] = ""
    st.session_state["summary"] = ""
    st.session_state["action_items"] = ""
    st.session_state["source_name"] = "transcript"
    st.session_state["audio_path"] = None


def _sync_ui_theme() -> None:
    theme = normalize_theme(st.session_state.get("ui_theme"))
    st.session_state["ui_theme"] = theme
    if not st.session_state.get("ui_theme_applied"):
        apply_streamlit_theme(theme)
        st.session_state["ui_theme_applied"] = True
        st.rerun()


def _clear_batch_results() -> None:
    st.session_state["batch_results"] = []
    st.session_state["selected_batch_file"] = None
    st.session_state["batch_running"] = False
    st.session_state["batch_queue"] = []
    st.session_state["batch_job"] = None
    st.session_state["cancel_requested"] = False


def _cleanup_cached_single_upload() -> None:
    path = st.session_state.get("cached_single_path")
    if path:
        cleanup_temp_file(Path(path))
    st.session_state["cached_single_path"] = None
    st.session_state["cached_single_metadata"] = None
    st.session_state["cached_single_error"] = None
    st.session_state["cached_single_upload_key"] = None


def _cleanup_cached_batch_uploads() -> None:
    items = st.session_state.get("cached_batch_items") or []
    cleanup_temp_files(*(item.get("path") for item in items if item.get("path")))
    st.session_state["cached_batch_items"] = None
    st.session_state["cached_batch_upload_key"] = None


def _upload_key(uploaded: object | list) -> str:
    if isinstance(uploaded, list):
        return "|".join(sorted(f.name for f in uploaded))
    return uploaded.name if uploaded else ""


def _get_or_save_single_upload(uploaded: object, upload_key: str):
    if st.session_state.get("cached_single_upload_key") != upload_key:
        _cleanup_cached_single_upload()
        saved_path, metadata, upload_error = process_upload(uploaded)
        st.session_state["cached_single_upload_key"] = upload_key
        st.session_state["cached_single_path"] = str(saved_path) if saved_path else None
        st.session_state["cached_single_metadata"] = metadata
        st.session_state["cached_single_error"] = upload_error
        return saved_path, metadata, upload_error

    path_str = st.session_state.get("cached_single_path")
    return (
        Path(path_str) if path_str else None,
        st.session_state.get("cached_single_metadata"),
        st.session_state.get("cached_single_error"),
    )


def _get_or_save_batch_uploads(uploaded_files: list, upload_key: str) -> list[dict]:
    if st.session_state.get("cached_batch_upload_key") != upload_key:
        _cleanup_cached_batch_uploads()
        items = process_uploads(uploaded_files)
        st.session_state["cached_batch_upload_key"] = upload_key
        st.session_state["cached_batch_items"] = items
        return items
    return st.session_state.get("cached_batch_items") or []


def _build_batch_zip(results: list[dict]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in results:
            if item.get("status") != "success":
                continue
            for export_path in item.get("export_paths", {}).values():
                path = Path(export_path)
                if path.exists():
                    zf.write(path, arcname=path.name)
    buffer.seek(0)
    return buffer.getvalue()


def _store_single_result(segments: list[dict], full_text: str) -> None:
    st.session_state["segments"] = segments
    st.session_state["full_text"] = full_text


def _store_batch_item_edit(filename: str, segments: list[dict], full_text: str) -> None:
    updated = []
    for item in st.session_state.get("batch_results") or []:
        if item.get("filename") == filename:
            paths = write_exports_if_any(
                segments,
                filename,
                bool((st.session_state.get("batch_job") or {}).get("include_timestamps", True)),
                group_paragraphs=bool((st.session_state.get("batch_job") or {}).get("group_paragraphs", False)),
                paragraph_gap_sec=float(
                    (st.session_state.get("batch_job") or {}).get(
                        "paragraph_gap_sec", DEFAULT_PARAGRAPH_GAP_SEC
                    )
                ),
            )
            updated.append(
                {
                    **item,
                    "segments": segments,
                    "full_text": full_text,
                    "segment_count": len(segments),
                    "status": "success" if segments else item.get("status"),
                    "export_paths": {k: str(v) for k, v in paths.items()},
                    "error": None if segments else item.get("error"),
                }
            )
        else:
            updated.append(item)
    st.session_state["batch_results"] = updated


def run_single_transcription(
    saved_path: Path,
    source_name: str,
    model_name: str,
    language_code: str | None,
    convert_traditional: bool,
    *,
    device: str = "auto",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    group_paragraphs: bool = False,
    paragraph_gap_sec: float = DEFAULT_PARAGRAPH_GAP_SEC,
    duration_sec: float | None = None,
    enable_chunking: bool = True,
    enable_diarization: bool = False,
    enable_summary: bool = False,
) -> bool:
    """Run single-file transcription with progress UI. Returns True on success."""
    st.info(
        "Cancel during a single-file run is limited by Streamlit: stop the app or refresh the page "
        "to abort. Batch mode supports cancel between file batches."
    )
    tracker = ProgressTracker(total_units=1.0, expected_audio_seconds=duration_sec)
    progress_bar = st.progress(0.0)
    status_line = st.empty()
    detail_line = st.empty()

    def on_progress(file_fraction: float, message: str) -> None:
        tracker.set_current(file_fraction)
        render_progress_ui(tracker, progress_bar, status_line, detail_line, message)

    try:
        resolved, device_warning = resolve_device(device)
        if device_warning:
            st.warning(device_warning)
        model = get_cached_model(model_name, resolved)
        result = transcribe_file(
            saved_path,
            model_name=model_name,
            language=language_code,
            convert_traditional=convert_traditional,
            model=model,
            progress=on_progress,
            device=resolved,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            group_paragraphs=group_paragraphs,
            paragraph_gap_sec=paragraph_gap_sec,
            enable_chunking=enable_chunking,
            enable_diarization=enable_diarization,
            enable_summary=enable_summary,
        )
        tracker.complete_unit()
        render_progress_ui(tracker, progress_bar, status_line, detail_line, "Transcription complete")

        if not result["segments"]:
            st.warning("No speech detected in the audio file.")
            _clear_single_results()
            return False

        st.session_state["segments"] = result["segments"]
        st.session_state["full_text"] = result["full_text"]
        st.session_state["summary"] = result.get("summary", "")
        st.session_state["action_items"] = result.get("action_items", "")
        st.session_state["source_name"] = source_name
        st.session_state["audio_path"] = str(saved_path)
        return True
    except Exception as exc:
        _log_error(exc)
        st.error(_friendly_error(exc))
        return False


def _start_batch_job(
    items: list[dict],
    *,
    model_name: str,
    language_code: str | None,
    convert_traditional: bool,
    include_timestamps: bool,
    device: str,
    initial_prompt: str | None,
    hotwords: str | None,
    group_paragraphs: bool,
    paragraph_gap_sec: float,
    replace_existing: bool,
    enable_chunking: bool = True,
    enable_diarization: bool = False,
    enable_summary: bool = False,
    max_parallel_jobs: int = 1,
) -> None:
    valid_items = [item for item in items if item["error"] is None and item["path"] is not None]
    if not valid_items:
        st.error("No valid files to transcribe.")
        return

    invalid_results = []
    for item in items:
        if item["error"]:
            invalid_results.append(
                {
                    "filename": item["filename"],
                    "status": "failed",
                    "segments": [],
                    "full_text": "",
                    "segment_count": 0,
                    "error": item["error"],
                    "export_paths": {},
                    "audio_path": None,
                }
            )

    st.session_state["batch_queue"] = list(valid_items)
    st.session_state["batch_running"] = True
    st.session_state["cancel_requested"] = False
    st.session_state["batch_job"] = {
        "model_name": model_name,
        "language_code": language_code,
        "convert_traditional": convert_traditional,
        "include_timestamps": include_timestamps,
        "device": device,
        "initial_prompt": initial_prompt,
        "hotwords": hotwords,
        "group_paragraphs": group_paragraphs,
        "paragraph_gap_sec": paragraph_gap_sec,
        "enable_chunking": enable_chunking,
        "enable_diarization": enable_diarization,
        "enable_summary": enable_summary,
        "max_parallel_jobs": max(1, int(max_parallel_jobs)),
        "total": len(valid_items),
        "completed": 0,
    }
    if replace_existing:
        st.session_state["batch_results"] = invalid_results
    st.rerun()


def _process_batch_queue_step() -> None:
    """Process up to N queued files per rerun (parallel) so Cancel works between batches."""
    if not st.session_state.get("batch_running"):
        return

    job = st.session_state.get("batch_job") or {}
    queue: list[dict] = list(st.session_state.get("batch_queue") or [])
    workers = max(1, int(job.get("max_parallel_jobs") or 1))

    cancel_col, status_col = st.columns([1, 3])
    with cancel_col:
        if st.button("Cancel batch", type="secondary", use_container_width=True, key="cancel_batch"):
            st.session_state["cancel_requested"] = True
    with status_col:
        done = int(job.get("completed") or 0)
        total = int(job.get("total") or max(len(queue) + done, 1))
        st.caption(
            f"Batch progress: {done}/{total} files finished · "
            f"up to {workers} in parallel · cancel applies before the next batch."
        )

    if st.session_state.get("cancel_requested"):
        cancelled = [cancelled_result(item["filename"]) for item in queue]
        st.session_state["batch_results"] = merge_batch_results(
            st.session_state.get("batch_results") or [],
            cancelled,
        )
        st.session_state["batch_queue"] = []
        st.session_state["batch_running"] = False
        st.session_state["cancel_requested"] = False
        st.warning("Batch cancelled. Remaining files were marked as Cancelled.")
        return

    if not queue:
        st.session_state["batch_running"] = False
        return

    batch_items = queue[:workers]
    st.session_state["batch_queue"] = queue[workers:]

    resolved, device_warning = resolve_device(job.get("device", "auto"))
    if device_warning:
        st.warning(device_warning)
    model = get_cached_model(job["model_name"], resolved)
    status_line = st.empty()
    status_line.info(f"Transcribing {len(batch_items)} file(s)...")

    def _work(item: dict) -> dict:
        filename = item["filename"]
        saved_path = item["path"]
        try:
            result = transcribe_file(
                saved_path,
                model_name=job["model_name"],
                language=job.get("language_code"),
                convert_traditional=bool(job.get("convert_traditional")),
                model=model,
                progress=None,
                device=resolved,
                initial_prompt=job.get("initial_prompt"),
                hotwords=job.get("hotwords"),
                group_paragraphs=bool(job.get("group_paragraphs")),
                paragraph_gap_sec=float(job.get("paragraph_gap_sec", DEFAULT_PARAGRAPH_GAP_SEC)),
                enable_chunking=bool(job.get("enable_chunking", True)),
                enable_diarization=bool(job.get("enable_diarization", False)),
                enable_summary=bool(job.get("enable_summary", False)),
            )
            export_paths = write_exports_if_any(
                result["segments"],
                filename,
                bool(job.get("include_timestamps", True)),
                group_paragraphs=bool(job.get("group_paragraphs")),
                paragraph_gap_sec=float(job.get("paragraph_gap_sec", DEFAULT_PARAGRAPH_GAP_SEC)),
            )
            return {
                "filename": filename,
                "status": "success" if result["segments"] else "empty",
                "segments": result["segments"],
                "full_text": result["full_text"],
                "segment_count": result["segment_count"],
                "summary": result.get("summary", ""),
                "action_items": result.get("action_items", ""),
                "error": None if result["segments"] else "No speech detected",
                "export_paths": {k: str(v) for k, v in export_paths.items()},
                "audio_path": str(saved_path),
            }
        except Exception as exc:
            _log_error(exc)
            return {
                "filename": filename,
                "status": "failed",
                "segments": [],
                "full_text": "",
                "segment_count": 0,
                "summary": "",
                "action_items": "",
                "error": _friendly_error(exc),
                "export_paths": {},
                "audio_path": str(saved_path) if saved_path else None,
            }

    updates: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(batch_items)) as executor:
        futures = [executor.submit(_work, item) for item in batch_items]
        for future in as_completed(futures):
            updates.append(future.result())

    st.session_state["batch_results"] = merge_batch_results(
        st.session_state.get("batch_results") or [],
        updates,
    )
    job["completed"] = int(job.get("completed") or 0) + len(updates)
    st.session_state["batch_job"] = job
    status_line.success(f"Finished batch of {len(updates)} file(s).")

    if st.session_state["batch_queue"]:
        st.rerun()
    else:
        st.session_state["batch_running"] = False


def render_batch_summary(
    results: list[dict],
    include_timestamps: bool,
    *,
    group_paragraphs: bool = False,
    paragraph_gap_sec: float = DEFAULT_PARAGRAPH_GAP_SEC,
    upload_items: list[dict] | None = None,
    shared_kwargs: dict | None = None,
) -> None:
    """Display batch summary with status filters, retry, and per-file details."""
    success_count = sum(1 for r in results if r["status"] == "success")
    empty_count = sum(1 for r in results if r["status"] == "empty")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    cancelled_count = sum(1 for r in results if r["status"] == "cancelled")
    st.success(
        f"Batch complete: {success_count} succeeded, {empty_count} empty, "
        f"{failed_count} failed, {cancelled_count} cancelled."
    )

    status_filter = st.selectbox(
        "Filter by status",
        STATUS_FILTER_OPTIONS,
        index=STATUS_FILTER_OPTIONS.index(st.session_state.get("batch_status_filter", STATUS_ALL)),
        key="batch_status_filter_widget",
    )
    st.session_state["batch_status_filter"] = status_filter
    filtered = filter_batch_results(results, status_filter)

    summary_rows = [
        {
            "File": r["filename"],
            "Status": r["status"].title(),
            "Segments": r.get("segment_count", "—"),
            "Message": r.get("error") or "OK",
        }
        for r in filtered
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    successful = [r for r in results if r["status"] == "success"]
    if successful:
        st.download_button(
            "Download all exports (ZIP)",
            data=_build_batch_zip(results),
            file_name="batch_transcripts.zip",
            mime="application/zip",
            use_container_width=True,
            key="batch_zip_download",
        )

    upload_items = upload_items or []
    by_name = {item["filename"]: item for item in upload_items}
    retry_sources = retryable_batch_items(results, by_name)
    if retry_sources and shared_kwargs is not None and not st.session_state.get("batch_running"):
        if st.button(
            f"Retry {len(retry_sources)} failed/empty/cancelled file(s)",
            use_container_width=True,
            key="batch_retry",
        ):
            _start_batch_job(
                retry_sources,
                model_name=shared_kwargs["model_name"],
                language_code=shared_kwargs["language_code"],
                convert_traditional=shared_kwargs["convert_traditional"],
                include_timestamps=include_timestamps,
                device=shared_kwargs["device"],
                initial_prompt=shared_kwargs["initial_prompt"],
                hotwords=shared_kwargs["hotwords"],
                group_paragraphs=group_paragraphs,
                paragraph_gap_sec=paragraph_gap_sec,
                replace_existing=False,
                enable_chunking=bool(shared_kwargs.get("enable_chunking", True)),
                enable_diarization=bool(shared_kwargs.get("enable_diarization", False)),
                enable_summary=bool(shared_kwargs.get("enable_summary", False)),
                max_parallel_jobs=int(shared_kwargs.get("max_parallel_jobs", 1)),
            )

    st.subheader("Batch Details")
    detail_pool = [r for r in filtered if r["status"] == "success"] or successful
    filenames = [r["filename"] for r in detail_pool]
    if not filenames:
        return

    if st.session_state["selected_batch_file"] not in filenames:
        st.session_state["selected_batch_file"] = filenames[0]

    selected = st.selectbox(
        "View transcript",
        filenames,
        index=filenames.index(st.session_state["selected_batch_file"]),
        key="batch_file_selector",
    )
    st.session_state["selected_batch_file"] = selected

    item = next(r for r in detail_pool if r["filename"] == selected)
    safe_key = selected.replace(" ", "_").replace(".", "_")
    render_results(
        item["segments"],
        item["full_text"],
        item["filename"],
        include_timestamps,
        key_prefix=f"batch_{safe_key}_",
        group_paragraphs=group_paragraphs,
        paragraph_gap_sec=paragraph_gap_sec,
        audio_path=item.get("audio_path"),
        on_segments_updated=lambda segs, text: _store_batch_item_edit(selected, segs, text),
        summary=item.get("summary", ""),
        action_items=item.get("action_items", ""),
    )


def render_single_mode(
    language_code: str | None,
    model_name: str,
    include_timestamps: bool,
    convert_traditional: bool,
    *,
    device: str = "auto",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    group_paragraphs: bool = False,
    paragraph_gap_sec: float = DEFAULT_PARAGRAPH_GAP_SEC,
    enable_chunking: bool = True,
    enable_diarization: bool = False,
    enable_summary: bool = False,
    max_parallel_jobs: int = 1,
) -> None:
    uploaded = st.file_uploader(
        "Upload audio or video file",
        type=UPLOAD_TYPES,
        help=UPLOAD_HELP,
    )

    if uploaded is None:
        st.info("Upload a supported audio or video file to begin.")
        if st.session_state["segments"]:
            st.subheader("Previous Transcript")
            render_results(
                st.session_state["segments"],
                st.session_state["full_text"],
                st.session_state["source_name"],
                include_timestamps,
                group_paragraphs=group_paragraphs,
                paragraph_gap_sec=paragraph_gap_sec,
                audio_path=st.session_state.get("audio_path"),
                on_segments_updated=_store_single_result,
                summary=st.session_state.get("summary", ""),
                action_items=st.session_state.get("action_items", ""),
            )
        return

    upload_key = _upload_key(uploaded)
    if upload_key != st.session_state.get("last_upload_key"):
        _clear_single_results()
        _clear_batch_results()
        _cleanup_cached_batch_uploads()
        st.session_state["last_upload_key"] = upload_key

    saved_path, metadata, upload_error = _get_or_save_single_upload(uploaded, upload_key)
    if upload_error:
        st.error(upload_error)
        return

    if metadata:
        st.subheader("File Info")
        render_metadata(metadata)
        if saved_path:
            from ui.components import render_audio_player

            render_audio_player(saved_path, seek_seconds=0)

    if st.button("Transcribe", type="primary", use_container_width=True, key="single_transcribe"):
        if saved_path is None:
            st.error("Could not process the uploaded file.")
            return
        run_single_transcription(
            saved_path,
            uploaded.name,
            model_name,
            language_code,
            convert_traditional,
            device=device,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            group_paragraphs=group_paragraphs,
            paragraph_gap_sec=paragraph_gap_sec,
            duration_sec=(metadata or {}).get("duration_sec"),
            enable_chunking=enable_chunking,
            enable_diarization=enable_diarization,
            enable_summary=enable_summary,
        )

    if st.session_state["segments"]:
        render_results(
            st.session_state["segments"],
            st.session_state["full_text"],
            st.session_state["source_name"],
            include_timestamps,
            group_paragraphs=group_paragraphs,
            paragraph_gap_sec=paragraph_gap_sec,
            audio_path=st.session_state.get("audio_path") or (str(saved_path) if saved_path else None),
            on_segments_updated=_store_single_result,
            summary=st.session_state.get("summary", ""),
            action_items=st.session_state.get("action_items", ""),
        )


def render_batch_mode(
    language_code: str | None,
    model_name: str,
    include_timestamps: bool,
    convert_traditional: bool,
    *,
    device: str = "auto",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    group_paragraphs: bool = False,
    paragraph_gap_sec: float = DEFAULT_PARAGRAPH_GAP_SEC,
    enable_chunking: bool = True,
    enable_diarization: bool = False,
    enable_summary: bool = False,
    max_parallel_jobs: int = 1,
) -> None:
    uploaded_files = st.file_uploader(
        "Upload audio or video files",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help=UPLOAD_HELP,
    )

    shared_kwargs = {
        "model_name": model_name,
        "language_code": language_code,
        "convert_traditional": convert_traditional,
        "device": device,
        "initial_prompt": initial_prompt,
        "hotwords": hotwords,
        "group_paragraphs": group_paragraphs,
        "paragraph_gap_sec": paragraph_gap_sec,
        "enable_chunking": enable_chunking,
        "enable_diarization": enable_diarization,
        "enable_summary": enable_summary,
        "max_parallel_jobs": max_parallel_jobs,
    }

    if st.session_state.get("batch_running"):
        _process_batch_queue_step()

    if not uploaded_files:
        st.info("Upload one or more audio/video files to begin batch processing.")
        if st.session_state["batch_results"] and not st.session_state.get("batch_running"):
            st.subheader("Previous Batch Results")
            render_batch_summary(
                st.session_state["batch_results"],
                include_timestamps,
                group_paragraphs=group_paragraphs,
                paragraph_gap_sec=paragraph_gap_sec,
                upload_items=st.session_state.get("cached_batch_items") or [],
                shared_kwargs=shared_kwargs,
            )
        return

    upload_key = _upload_key(uploaded_files)
    if upload_key != st.session_state.get("last_upload_key"):
        _clear_single_results()
        _clear_batch_results()
        _cleanup_cached_single_upload()
        st.session_state["last_upload_key"] = upload_key

    items = _get_or_save_batch_uploads(uploaded_files, upload_key)
    st.subheader(f"Queued Files ({len(items)})")

    table_rows = []
    for item in items:
        meta = item["metadata"] or {}
        table_rows.append(
            {
                "Filename": item["filename"],
                "Size (MB)": meta.get("size_mb", "—"),
                "Duration (s)": meta.get("duration_sec", "—"),
                "Status": "Invalid" if item["error"] else "Ready",
                "Note": item["error"] or "",
            }
        )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    valid_count = sum(1 for item in items if not item["error"])
    if valid_count == 0:
        st.error("None of the uploaded files are valid.")
        return

    if not st.session_state.get("batch_running") and st.button(
        f"Transcribe {valid_count} file{'s' if valid_count != 1 else ''}",
        type="primary",
        use_container_width=True,
        key="batch_transcribe",
    ):
        _start_batch_job(
            items,
            model_name=model_name,
            language_code=language_code,
            convert_traditional=convert_traditional,
            include_timestamps=include_timestamps,
            device=device,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            group_paragraphs=group_paragraphs,
            paragraph_gap_sec=paragraph_gap_sec,
            replace_existing=True,
            enable_chunking=enable_chunking,
            enable_diarization=enable_diarization,
            enable_summary=enable_summary,
            max_parallel_jobs=max_parallel_jobs,
        )

    if st.session_state["batch_results"] and not st.session_state.get("batch_running"):
        render_batch_summary(
            st.session_state["batch_results"],
            include_timestamps,
            group_paragraphs=group_paragraphs,
            paragraph_gap_sec=paragraph_gap_sec,
            upload_items=items,
            shared_kwargs=shared_kwargs,
        )


def main() -> None:
    ensure_dirs()
    _init_session_state()
    if not _render_auth_gate():
        return
    _sync_ui_theme()

    st.title("🎙️ Meeting Transcription")
    st.caption(
        "Upload meeting audio or video in Traditional Chinese, English, or mixed speech. "
        "Transcription runs locally with faster-whisper."
    )

    with st.sidebar:
        st.header("Settings")
        render_theme_toggle()
        st.divider()
        processing_mode = st.radio("Processing mode", ["Single file", "Batch files"], index=0)
        language_label = st.selectbox(
            "Language",
            list(LANGUAGE_OPTIONS.keys()),
            index=list(LANGUAGE_OPTIONS.keys()).index(DEFAULT_LANGUAGE_LABEL),
        )
        model_name = st.selectbox("Model", WHISPER_MODELS, index=WHISPER_MODELS.index(DEFAULT_MODEL))
        device = st.selectbox(
            "Device",
            DEVICE_OPTIONS,
            index=DEVICE_OPTIONS.index(DEFAULT_DEVICE),
            help="auto uses CUDA when PyTorch + GPU are available; otherwise CPU.",
        )
        if device == "cuda" and not cuda_available():
            st.caption("CUDA not detected (install a CUDA build of PyTorch to enable GPU).")
        elif device == "auto":
            st.caption("CUDA available" if cuda_available() else "Running on CPU (no CUDA detected)")

        include_timestamps = st.checkbox("Include timestamps", value=True)
        convert_traditional = st.checkbox("Convert Chinese output to Traditional Chinese", value=True)
        group_paragraphs = st.checkbox(
            "Group paragraphs by silence",
            value=False,
            help=f"Insert blank lines in the transcript when silence exceeds {DEFAULT_PARAGRAPH_GAP_SEC:g}s.",
        )
        enable_chunking = st.checkbox(
            "Chunk long audio",
            value=CHUNK_ENABLED_DEFAULT,
            help="Split long recordings into overlapping chunks to reduce memory use.",
        )
        enable_diarization = st.checkbox(
            "Speaker labels (gap-based)",
            value=False,
            help="Heuristic SPEAKER_00/01 labels after longer silences (not full neural diarization).",
        )
        enable_summary = st.checkbox(
            "Meeting summary + action items",
            value=False,
            help="Local extractive notes — no cloud LLM required.",
        )
        max_parallel_jobs = st.selectbox(
            "Batch parallel jobs",
            MAX_PARALLEL_JOBS_OPTIONS,
            index=MAX_PARALLEL_JOBS_OPTIONS.index(MAX_PARALLEL_JOBS_DEFAULT),
            help="How many files to transcribe concurrently in batch mode.",
        )
        hotwords = st.text_area(
            "Hotwords (optional)",
            value="",
            height=68,
            help="Comma or newline separated terms to bias recognition (names, jargon).",
            placeholder="AILM, quarterly review, Taipei",
        )
        initial_prompt = st.text_input(
            "Initial prompt (optional)",
            value="",
            help="Free-text prompt bias for the first decoding window.",
            placeholder="This is a product planning meeting.",
        )

        st.divider()
        if ffmpeg_available():
            st.success("ffmpeg detected")
        else:
            st.warning("ffmpeg not found — MP3/M4A/video may fail. Install with: `winget install Gyan.FFmpeg`")

        st.markdown(
            "**Tips**\n"
            "- Video uploads are converted to audio via ffmpeg.\n"
            "- Batch cancel works between parallel batches.\n"
            "- Set env `TRANSCRIPTION_APP_PASSWORD` to require login.\n"
            f"- Max upload: {MAX_UPLOAD_SIZE_MB} MB / {MAX_DURATION_SEC // 3600}h per file."
        )

    language_code = LANGUAGE_OPTIONS[language_label]
    shared_kwargs = {
        "device": device,
        "initial_prompt": initial_prompt.strip() or None,
        "hotwords": hotwords.strip() or None,
        "group_paragraphs": group_paragraphs,
        "paragraph_gap_sec": DEFAULT_PARAGRAPH_GAP_SEC,
        "enable_chunking": enable_chunking,
        "enable_diarization": enable_diarization,
        "enable_summary": enable_summary,
        "max_parallel_jobs": max_parallel_jobs,
    }

    if processing_mode == "Single file":
        render_single_mode(
            language_code,
            model_name,
            include_timestamps,
            convert_traditional,
            **shared_kwargs,
        )
    else:
        render_batch_mode(
            language_code,
            model_name,
            include_timestamps,
            convert_traditional,
            **shared_kwargs,
        )


if __name__ == "__main__":
    main()