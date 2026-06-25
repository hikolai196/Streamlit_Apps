"""Streamlit meeting transcription app."""

from __future__ import annotations

import traceback
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from faster_whisper import WhisperModel

from config import DEFAULT_MODEL, LANGUAGE_OPTIONS, OUTPUT_DIR, WHISPER_MODELS
from modules.export_docx import export_docx
from modules.export_srt import export_srt
from modules.export_txt import export_txt
from modules.pipeline import transcribe_file
from modules.upload import process_upload, process_uploads
from utils.audio_backend import configure_audio_backend, ffmpeg_available
from utils.file_utils import ensure_dirs
from utils.logger import get_logger
from utils.progress import ProgressTracker
from utils.time_utils import format_duration, seconds_to_hms

logger = get_logger(__name__)

st.set_page_config(
    page_title="Meeting Transcription",
    page_icon="🎙️",
    layout="wide",
)

configure_audio_backend()


@st.cache_resource(show_spinner="Loading Whisper model...")
def get_cached_model(model_name: str) -> WhisperModel:
    """Load and cache Whisper model for Streamlit sessions."""
    from modules.asr import load_whisper_model

    return load_whisper_model(model_name)


def _friendly_error(exc: Exception) -> str:
    """Return a user-friendly error message without stack trace."""
    message = str(exc).strip()
    if message:
        return message
    return "An unexpected error occurred. Please try again with a different file or settings."


def _log_error(exc: Exception) -> None:
    """Log full error details server-side."""
    logger.error("Pipeline error: %s\n%s", exc, traceback.format_exc())


def _init_session_state() -> None:
    """Initialize session state keys used by the app."""
    defaults = {
        "segments": [],
        "full_text": "",
        "source_name": "transcript",
        "last_upload_key": None,
        "batch_results": [],
        "selected_batch_file": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_single_results() -> None:
    """Clear stored single-file transcription results."""
    st.session_state["segments"] = []
    st.session_state["full_text"] = ""
    st.session_state["source_name"] = "transcript"


def _clear_batch_results() -> None:
    """Clear stored batch transcription results."""
    st.session_state["batch_results"] = []
    st.session_state["selected_batch_file"] = None


def _upload_key(uploaded: object | list) -> str:
    """Build a stable key for the current upload selection."""
    if isinstance(uploaded, list):
        return "|".join(sorted(f.name for f in uploaded))
    return uploaded.name if uploaded else ""


def render_progress_ui(
    tracker: ProgressTracker,
    progress_bar: st.delta_generator.DeltaGenerator,
    status_line: st.delta_generator.DeltaGenerator,
    detail_line: st.delta_generator.DeltaGenerator,
    message: str,
) -> None:
    """Update progress bar and ETA display."""
    progress_bar.progress(tracker.fraction)
    eta = format_duration(tracker.eta_seconds())
    status_line.markdown(f"**{message}** — {int(tracker.fraction * 100)}% complete · ~{eta} remaining")
    detail_line.caption(f"Elapsed: {format_duration(tracker.elapsed_seconds())}")


def render_metadata(metadata: dict) -> None:
    """Display uploaded file metadata."""
    cols = st.columns(4)
    cols[0].metric("Filename", metadata.get("filename", "—"))
    cols[1].metric("Size (MB)", metadata.get("size_mb", "—"))
    cols[2].metric("Duration (s)", metadata.get("duration_sec", "—"))
    cols[3].metric("Sample Rate", metadata.get("sample_rate", "—"))


def write_exports(
    segments: list[dict],
    base_name: str,
    include_timestamps: bool,
) -> dict[str, Path]:
    """Write TXT, SRT, and DOCX exports to the output directory."""
    ensure_dirs()
    stem = Path(base_name).stem
    paths = {
        "txt": OUTPUT_DIR / f"{stem}.txt",
        "srt": OUTPUT_DIR / f"{stem}.srt",
        "docx": OUTPUT_DIR / f"{stem}.docx",
    }
    export_txt(segments, paths["txt"], include_timestamps=include_timestamps)
    export_srt(segments, paths["srt"])
    export_docx(segments, paths["docx"], include_timestamps=include_timestamps)
    return paths


def render_exports(segments: list[dict], base_name: str, include_timestamps: bool, key_prefix: str = "") -> None:
    """Render download buttons for TXT, SRT, and DOCX exports."""
    paths = write_exports(segments, base_name, include_timestamps)

    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "Download TXT",
        data=paths["txt"].read_bytes(),
        file_name=paths["txt"].name,
        mime="text/plain",
        use_container_width=True,
        key=f"{key_prefix}download_txt",
    )
    col2.download_button(
        "Download SRT",
        data=paths["srt"].read_bytes(),
        file_name=paths["srt"].name,
        mime="application/x-subrip",
        use_container_width=True,
        key=f"{key_prefix}download_srt",
    )
    col3.download_button(
        "Download DOCX",
        data=paths["docx"].read_bytes(),
        file_name=paths["docx"].name,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
        key=f"{key_prefix}download_docx",
    )


def render_results(
    segments: list[dict],
    full_text: str,
    source_name: str,
    include_timestamps: bool,
    key_prefix: str = "single_",
) -> None:
    """Display transcript, segment table, and export buttons."""
    st.success(f"Transcribed {len(segments)} segments.")

    st.subheader("Full Transcript")
    st.text_area("Transcript", value=full_text, height=200, label_visibility="collapsed")

    st.subheader("Segments")
    table_data = [
        {
            "Start": seconds_to_hms(float(seg["start"])),
            "End": seconds_to_hms(float(seg["end"])),
            "Text": seg["text"],
        }
        for seg in segments
    ]
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.subheader("Export")
    render_exports(segments, source_name, include_timestamps, key_prefix=key_prefix)


def _build_batch_zip(results: list[dict]) -> bytes:
    """Create a ZIP archive containing all successful batch exports."""
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


def render_batch_summary(results: list[dict], include_timestamps: bool) -> None:
    """Display batch job summary, per-file results, and downloads."""
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - success_count
    st.success(f"Batch complete: {success_count} succeeded, {failed_count} failed.")

    summary_rows = [
        {
            "File": r["filename"],
            "Status": r["status"].title(),
            "Segments": r.get("segment_count", "—"),
            "Message": r.get("error") or "OK",
        }
        for r in results
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

    st.subheader("Batch Details")
    filenames = [r["filename"] for r in successful]
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

    item = next(r for r in successful if r["filename"] == selected)
    safe_key = selected.replace(" ", "_").replace(".", "_")
    render_results(
        item["segments"],
        item["full_text"],
        item["filename"],
        include_timestamps,
        key_prefix=f"batch_{safe_key}_",
    )


def run_single_transcription(
    saved_path: Path,
    source_name: str,
    model_name: str,
    language_code: str | None,
    convert_traditional: bool,
) -> bool:
    """Run single-file transcription with progress UI. Returns True on success."""
    tracker = ProgressTracker(total_units=1.0)
    progress_bar = st.progress(0.0)
    status_line = st.empty()
    detail_line = st.empty()

    def on_progress(file_fraction: float, message: str) -> None:
        tracker.set_current(file_fraction)
        render_progress_ui(tracker, progress_bar, status_line, detail_line, message)

    try:
        model = get_cached_model(model_name)
        result = transcribe_file(
            saved_path,
            model_name=model_name,
            language=language_code,
            convert_traditional=convert_traditional,
            model=model,
            progress=on_progress,
        )
        tracker.complete_unit()
        render_progress_ui(tracker, progress_bar, status_line, detail_line, "Transcription complete")

        if not result["segments"]:
            st.warning("No speech detected in the audio file.")
            _clear_single_results()
            return False

        st.session_state["segments"] = result["segments"]
        st.session_state["full_text"] = result["full_text"]
        st.session_state["source_name"] = source_name
        return True
    except Exception as exc:
        _log_error(exc)
        st.error(_friendly_error(exc))
        return False


def run_batch_transcription(
    items: list[dict],
    model_name: str,
    language_code: str | None,
    convert_traditional: bool,
    include_timestamps: bool,
) -> None:
    """Run batch transcription with overall progress and ETA."""
    valid_items = [item for item in items if item["error"] is None and item["path"] is not None]
    if not valid_items:
        st.error("No valid files to transcribe.")
        return

    tracker = ProgressTracker(total_units=float(len(valid_items)))
    progress_bar = st.progress(0.0)
    status_line = st.empty()
    detail_line = st.empty()

    model = get_cached_model(model_name)
    batch_results: list[dict] = []

    for index, item in enumerate(valid_items, start=1):
        filename = item["filename"]
        saved_path = item["path"]

        def on_progress(file_fraction: float, message: str) -> None:
            tracker.set_current(file_fraction)
            render_progress_ui(
                tracker,
                progress_bar,
                status_line,
                detail_line,
                f"File {index}/{len(valid_items)}: {filename} — {message}",
            )

        try:
            result = transcribe_file(
                saved_path,
                model_name=model_name,
                language=language_code,
                convert_traditional=convert_traditional,
                model=model,
                progress=on_progress,
            )
            export_paths = write_exports(result["segments"], filename, include_timestamps)
            batch_results.append(
                {
                    "filename": filename,
                    "status": "success" if result["segments"] else "empty",
                    "segments": result["segments"],
                    "full_text": result["full_text"],
                    "segment_count": result["segment_count"],
                    "error": None if result["segments"] else "No speech detected",
                    "export_paths": {k: str(v) for k, v in export_paths.items()},
                }
            )
        except Exception as exc:
            _log_error(exc)
            batch_results.append(
                {
                    "filename": filename,
                    "status": "failed",
                    "segments": [],
                    "full_text": "",
                    "segment_count": 0,
                    "error": _friendly_error(exc),
                    "export_paths": {},
                }
            )

        tracker.complete_unit()
        render_progress_ui(
            tracker,
            progress_bar,
            status_line,
            detail_line,
            f"Finished {index}/{len(valid_items)} files",
        )

    for item in items:
        if item["error"]:
            batch_results.append(
                {
                    "filename": item["filename"],
                    "status": "failed",
                    "segments": [],
                    "full_text": "",
                    "segment_count": 0,
                    "error": item["error"],
                    "export_paths": {},
                }
            )

    batch_results.sort(key=lambda r: r["filename"])
    st.session_state["batch_results"] = batch_results


def render_single_mode(
    language_code: str | None,
    model_name: str,
    include_timestamps: bool,
    convert_traditional: bool,
) -> None:
    """Render single-file upload and transcription UI."""
    uploaded = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "m4a"],
        help="Supported formats: .wav, .mp3, .m4a",
    )

    if uploaded is None:
        st.info("Upload a .wav, .mp3, or .m4a file to begin.")
        if st.session_state["segments"]:
            st.subheader("Previous Transcript")
            render_results(
                st.session_state["segments"],
                st.session_state["full_text"],
                st.session_state["source_name"],
                include_timestamps,
            )
        return

    upload_key = _upload_key(uploaded)
    if upload_key != st.session_state.get("last_upload_key"):
        _clear_single_results()
        _clear_batch_results()
        st.session_state["last_upload_key"] = upload_key

    saved_path, metadata, upload_error = process_upload(uploaded)
    if upload_error:
        st.error(upload_error)
        return

    if metadata:
        st.subheader("File Info")
        render_metadata(metadata)

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
        )

    if st.session_state["segments"]:
        render_results(
            st.session_state["segments"],
            st.session_state["full_text"],
            st.session_state["source_name"],
            include_timestamps,
        )


def render_batch_mode(
    language_code: str | None,
    model_name: str,
    include_timestamps: bool,
    convert_traditional: bool,
) -> None:
    """Render batch upload and transcription UI."""
    uploaded_files = st.file_uploader(
        "Upload audio files",
        type=["wav", "mp3", "m4a"],
        accept_multiple_files=True,
        help="Select multiple .wav, .mp3, or .m4a files",
    )

    if not uploaded_files:
        st.info("Upload one or more audio files to begin batch processing.")
        if st.session_state["batch_results"]:
            st.subheader("Previous Batch Results")
            render_batch_summary(st.session_state["batch_results"], include_timestamps)
        return

    upload_key = _upload_key(uploaded_files)
    if upload_key != st.session_state.get("last_upload_key"):
        _clear_single_results()
        _clear_batch_results()
        st.session_state["last_upload_key"] = upload_key

    items = process_uploads(uploaded_files)
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

    if st.button(
        f"Transcribe {valid_count} file{'s' if valid_count != 1 else ''}",
        type="primary",
        use_container_width=True,
        key="batch_transcribe",
    ):
        run_batch_transcription(items, model_name, language_code, convert_traditional, include_timestamps)

    if st.session_state["batch_results"]:
        render_batch_summary(st.session_state["batch_results"], include_timestamps)


def main() -> None:
    """Main Streamlit application entry point."""
    ensure_dirs()
    _init_session_state()

    st.title("🎙️ Meeting Transcription")
    st.caption(
        "Upload meeting audio in Traditional Chinese, English, or mixed speech. "
        "Transcription runs locally with faster-whisper."
    )

    with st.sidebar:
        st.header("Settings")
        processing_mode = st.radio("Processing mode", ["Single file", "Batch files"], index=0)
        language_label = st.selectbox("Language", list(LANGUAGE_OPTIONS.keys()), index=0)
        model_name = st.selectbox("Model", WHISPER_MODELS, index=WHISPER_MODELS.index(DEFAULT_MODEL))
        include_timestamps = st.checkbox("Include timestamps", value=True)
        convert_traditional = st.checkbox("Convert Chinese output to Traditional Chinese", value=True)

        st.divider()
        if ffmpeg_available():
            st.success("ffmpeg detected")
        else:
            st.warning("ffmpeg not found — MP3/M4A may fail. Install with: `winget install Gyan.FFmpeg`")

        st.markdown(
            "**Tips**\n"
            "- First run downloads the selected model.\n"
            "- Batch mode exports all files automatically.\n"
            "- Progress bar shows estimated time remaining."
        )

    language_code = LANGUAGE_OPTIONS[language_label]

    if processing_mode == "Single file":
        render_single_mode(language_code, model_name, include_timestamps, convert_traditional)
    else:
        render_batch_mode(language_code, model_name, include_timestamps, convert_traditional)


if __name__ == "__main__":
    main()
