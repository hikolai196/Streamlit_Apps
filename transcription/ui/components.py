"""Streamlit UI building blocks for the transcription app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import DEFAULT_PARAGRAPH_GAP_SEC
from modules.export_bundle import download_filename, write_exports
from ui.transcript import apply_edited_transcript
from utils.progress import ProgressTracker
from utils.theme import apply_streamlit_theme, normalize_theme
from utils.time_utils import format_duration, seconds_to_hms


def render_theme_toggle() -> None:
    """Sidebar control to switch between dark and light themes."""
    current = normalize_theme(st.session_state.get("ui_theme"))
    dark_enabled = st.toggle(
        "Dark theme",
        value=(current == "dark"),
        help="Turn off for light theme. Default is dark.",
        key="dark_theme_toggle",
    )
    desired = "dark" if dark_enabled else "light"
    if desired != current:
        st.session_state["ui_theme"] = desired
        apply_streamlit_theme(desired)
        st.session_state["ui_theme_applied"] = True
        st.rerun()


def render_progress_ui(
    tracker: ProgressTracker,
    progress_bar: Any,
    status_line: Any,
    detail_line: Any,
    message: str,
) -> None:
    """Update progress bar and ETA display."""
    progress_bar.progress(tracker.fraction)
    eta = format_duration(tracker.eta_seconds())
    status_line.markdown(
        f"**{message}** — {int(tracker.fraction * 100)}% complete · ~{eta} remaining"
    )
    detail_line.caption(f"Elapsed: {format_duration(tracker.elapsed_seconds())}")


def render_metadata(metadata: dict) -> None:
    """Display uploaded file metadata."""
    cols = st.columns(4)
    cols[0].metric("Filename", metadata.get("filename", "—"))
    cols[1].metric("Size (MB)", metadata.get("size_mb", "—"))
    cols[2].metric("Duration (s)", metadata.get("duration_sec", "—"))
    cols[3].metric("Sample Rate", metadata.get("sample_rate", "—"))


def render_audio_player(
    audio_path: str | Path | None,
    *,
    seek_seconds: float = 0.0,
) -> None:
    """Render an audio player for the uploaded/source file when available."""
    if not audio_path:
        return
    path = Path(audio_path)
    if not path.exists():
        st.caption("Audio preview unavailable (temp file was cleaned up).")
        return

    st.subheader("Audio")
    start = max(0, int(seek_seconds))
    suffix = path.suffix.lstrip(".").lower() or "wav"
    mime = "audio/mpeg" if suffix == "mp3" else f"audio/{suffix}"
    st.audio(str(path), format=mime, start_time=start)
    if start > 0:
        st.caption(f"Playback starts at {seconds_to_hms(float(start))}.")


def render_exports(
    segments: list[dict],
    base_name: str,
    include_timestamps: bool,
    key_prefix: str = "",
    *,
    group_paragraphs: bool = False,
    paragraph_gap_sec: float = DEFAULT_PARAGRAPH_GAP_SEC,
) -> None:
    """Render download buttons for TXT, SRT, VTT, and DOCX exports."""
    paths = write_exports(
        segments,
        base_name,
        include_timestamps,
        group_paragraphs=group_paragraphs,
        paragraph_gap_sec=paragraph_gap_sec,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.download_button(
        "Download TXT",
        data=paths["txt"].read_bytes(),
        file_name=download_filename(base_name, ".txt"),
        mime="text/plain",
        use_container_width=True,
        key=f"{key_prefix}download_txt",
    )
    col2.download_button(
        "Download SRT",
        data=paths["srt"].read_bytes(),
        file_name=download_filename(base_name, ".srt"),
        mime="application/x-subrip",
        use_container_width=True,
        key=f"{key_prefix}download_srt",
    )
    col3.download_button(
        "Download VTT",
        data=paths["vtt"].read_bytes(),
        file_name=download_filename(base_name, ".vtt"),
        mime="text/vtt",
        use_container_width=True,
        key=f"{key_prefix}download_vtt",
    )
    col4.download_button(
        "Download DOCX",
        data=paths["docx"].read_bytes(),
        file_name=download_filename(base_name, ".docx"),
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
    *,
    group_paragraphs: bool = False,
    paragraph_gap_sec: float = DEFAULT_PARAGRAPH_GAP_SEC,
    audio_path: str | Path | None = None,
    on_segments_updated: Any | None = None,
    summary: str = "",
    action_items: str = "",
) -> None:
    """Display audio, editable transcript, segment table, and export buttons."""
    st.success(f"Transcribed {len(segments)} segments.")

    seek_key = f"{key_prefix}seek_seconds"
    if seek_key not in st.session_state:
        st.session_state[seek_key] = 0.0

    render_audio_player(
        audio_path,
        seek_seconds=float(st.session_state.get(seek_key) or 0.0),
    )

    if summary or action_items:
        st.subheader("Meeting notes")
        if summary:
            st.markdown(summary.replace("\n", "  \n"))
        if action_items:
            st.markdown(action_items.replace("\n", "  \n"))

    st.subheader("Full Transcript")
    edited = st.text_area(
        "Transcript",
        value=full_text,
        height=220,
        label_visibility="collapsed",
        key=f"{key_prefix}transcript_editor",
    )
    apply_col, hint_col = st.columns([1, 3])
    with apply_col:
        apply_clicked = st.button(
            "Apply edits",
            use_container_width=True,
            key=f"{key_prefix}apply_edits",
            help="Update segments from the edited transcript, then re-export.",
        )
    with hint_col:
        st.caption(
            "Edit the text, then click Apply edits. Matching line counts keep original timestamps; "
            "otherwise timing is redistributed across the clip."
        )

    active_segments = segments
    active_text = full_text
    if apply_clicked:
        active_segments, active_text = apply_edited_transcript(segments, edited)
        st.session_state[f"{key_prefix}transcript_editor"] = active_text
        if on_segments_updated:
            on_segments_updated(active_segments, active_text)
        st.rerun()

    st.subheader("Segments")
    if active_segments:
        labels = [
            f"{seconds_to_hms(float(seg['start']))} — "
            f"{('[' + seg['speaker'] + '] ') if seg.get('speaker') else ''}"
            f"{seg['text'][:60]}"
            for seg in active_segments
        ]
        selected = st.selectbox(
            "Jump audio to segment",
            options=list(range(len(active_segments))),
            format_func=lambda idx: labels[idx],
            key=f"{key_prefix}segment_seek",
        )
        jump = st.button("Seek to segment", key=f"{key_prefix}seek_btn")
        if jump:
            st.session_state[seek_key] = float(active_segments[selected]["start"])
            st.rerun()

    table_data = []
    for seg in active_segments:
        row = {
            "Start": seconds_to_hms(float(seg["start"])),
            "End": seconds_to_hms(float(seg["end"])),
            "Text": seg["text"],
        }
        if seg.get("speaker"):
            row["Speaker"] = seg["speaker"]
        table_data.append(row)
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.subheader("Export")
    render_exports(
        active_segments,
        source_name,
        include_timestamps,
        key_prefix=key_prefix,
        group_paragraphs=group_paragraphs,
        paragraph_gap_sec=paragraph_gap_sec,
    )
