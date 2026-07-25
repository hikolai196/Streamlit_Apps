"""Phase 4 tests: editable transcript, batch filters/retry, AppTest smoke."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from modules.export_bundle import write_exports
from ui.batch import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    cancelled_result,
    filter_batch_results,
    merge_batch_results,
    retryable_batch_items,
)
from ui.transcript import apply_edited_transcript, redistribute_segments


def test_apply_edited_transcript_keeps_timestamps_when_line_count_matches():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 1.0, "end": 2.0, "text": "World"},
    ]
    updated, full_text = apply_edited_transcript(segments, "Hi\nThere")
    assert updated[0]["start"] == 0.0
    assert updated[0]["text"] == "Hi"
    assert updated[1]["text"] == "There"
    assert full_text == "Hi\nThere"


def test_apply_edited_transcript_redistributes_when_line_count_changes():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 1.0, "end": 2.0, "text": "World"},
    ]
    updated, full_text = apply_edited_transcript(segments, "One\nTwo\nThree")
    assert len(updated) == 3
    assert updated[0]["start"] == 0.0
    assert updated[-1]["end"] == 2.0
    assert full_text == "One\nTwo\nThree"


def test_redistribute_segments_empty_lines():
    assert redistribute_segments([{"start": 0, "end": 1, "text": "x"}], "") == []
    assert redistribute_segments([], "") == []


def test_reexport_from_edited_segments(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.export_bundle.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("modules.export_bundle.ensure_dirs", lambda: None)

    original = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 1.0, "end": 2.0, "text": "World"},
    ]
    edited, _ = apply_edited_transcript(original, "Edited one\nEdited two")
    paths = write_exports(edited, "meeting.wav", include_timestamps=True)
    txt = paths["txt"].read_text(encoding="utf-8")
    assert "Edited one" in txt
    assert "Edited two" in txt
    assert paths["srt"].exists()
    assert paths["vtt"].exists()
    assert paths["docx"].exists()


def test_filter_batch_results_by_status():
    results = [
        {"filename": "a.wav", "status": "success"},
        {"filename": "b.wav", "status": "failed"},
        {"filename": "c.wav", "status": "empty"},
        {"filename": "d.wav", "status": "cancelled"},
    ]
    assert len(filter_batch_results(results, STATUS_SUCCESS)) == 1
    assert len(filter_batch_results(results, STATUS_FAILED)) == 1
    assert len(filter_batch_results(results, "Empty")) == 1
    assert len(filter_batch_results(results, "Cancelled")) == 1
    assert len(filter_batch_results(results, "All")) == 4


def test_retryable_and_merge_batch_helpers():
    results = [
        {"filename": "ok.wav", "status": "success"},
        {"filename": "bad.wav", "status": "failed"},
        {"filename": "quiet.wav", "status": "empty"},
    ]
    uploads = {
        "ok.wav": {"filename": "ok.wav", "path": Path("ok.wav"), "error": None},
        "bad.wav": {"filename": "bad.wav", "path": Path("bad.wav"), "error": None},
        "quiet.wav": {"filename": "quiet.wav", "path": Path("quiet.wav"), "error": None},
    }
    retryable = retryable_batch_items(results, uploads)
    assert [item["filename"] for item in retryable] == ["bad.wav", "quiet.wav"]

    merged = merge_batch_results(
        results,
        [{"filename": "bad.wav", "status": "success", "segments": [{"text": "fixed"}]}],
    )
    by_name = {row["filename"]: row for row in merged}
    assert by_name["bad.wav"]["status"] == "success"
    assert by_name["ok.wav"]["status"] == "success"


def test_cancelled_result_shape():
    row = cancelled_result("x.wav")
    assert row["status"] == "cancelled"
    assert row["filename"] == "x.wav"
    assert row["export_paths"] == {}


def test_app_smoke_loads_sidebar(monkeypatch, tmp_path):
    temp_dir = tmp_path / "temp"
    output_dir = tmp_path / "output"
    temp_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr("config.TEMP_DIR", temp_dir)
    monkeypatch.setattr("config.OUTPUT_DIR", output_dir)
    monkeypatch.setattr("utils.file_utils.TEMP_DIR", temp_dir)

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.sidebar.radio) >= 1
    assert any("Dark theme" in str(toggle.label) for toggle in at.sidebar.toggle)
