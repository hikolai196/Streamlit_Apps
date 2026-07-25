"""Phase 5 tests: chunking, diarization, summary, video helpers, auth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from modules.chunking import (
    merge_chunk_segments,
    shift_segments,
    should_chunk,
    split_audio_chunks,
)
from modules.diarize import assign_speakers_by_gap, format_segment_line
from modules.media import is_video_file
from modules.pipeline import transcribe_file
from modules.summary import build_meeting_notes, extract_action_items, extractive_summary
from utils.auth import password_required, verify_password
from utils.file_utils import validate_audio_file


def test_validate_accepts_video_extension():
    ok, err = validate_audio_file("meeting.mp4", "video/mp4")
    assert ok is True
    assert err == ""


def test_is_video_file():
    assert is_video_file("a.mp4") is True
    assert is_video_file("a.wav") is False


def test_should_chunk_threshold(monkeypatch):
    monkeypatch.setattr("modules.chunking.CHUNK_MIN_DURATION_SEC", 10)
    assert should_chunk(9, enabled=True) is False
    assert should_chunk(11, enabled=True) is True
    assert should_chunk(100, enabled=False) is False


def test_split_and_merge_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.chunking.TEMP_DIR", tmp_path)
    monkeypatch.setattr("modules.chunking.ensure_dirs", lambda: None)

    wav = tmp_path / "long.wav"
    sr = 16000
    audio = np.zeros(sr * 5, dtype=np.float32)
    sf.write(str(wav), audio, sr)

    chunks = split_audio_chunks(wav, chunk_duration_sec=2.0, overlap_sec=0.5)
    assert len(chunks) >= 2
    assert all(path.exists() for path, _ in chunks)

    merged = merge_chunk_segments(
        [
            (0.0, [{"start": 0.0, "end": 0.5, "text": "hello"}]),
            (1.5, [{"start": 0.0, "end": 0.4, "text": "world"}]),
        ]
    )
    assert merged[0]["text"] == "hello"
    assert merged[1]["start"] == 1.5
    assert merged[1]["text"] == "world"


def test_shift_segments():
    shifted = shift_segments([{"start": 1.0, "end": 2.0, "text": "x"}], 10.0)
    assert shifted[0]["start"] == 11.0
    assert shifted[0]["end"] == 12.0


def test_assign_speakers_by_gap():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "A"},
        {"start": 1.2, "end": 2.0, "text": "A2"},
        {"start": 5.0, "end": 6.0, "text": "B"},
    ]
    labeled = assign_speakers_by_gap(segments, gap_sec=2.0, max_speakers=2)
    assert labeled[0]["speaker"] == "SPEAKER_00"
    assert labeled[1]["speaker"] == "SPEAKER_00"
    assert labeled[2]["speaker"] == "SPEAKER_01"
    assert format_segment_line(labeled[2]).startswith("[SPEAKER_01]")


def test_summary_and_action_items():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "We discussed the quarterly roadmap in detail today."},
        {"start": 2.0, "end": 4.0, "text": "Alice will follow up on the budget by Friday."},
        {"start": 4.0, "end": 5.0, "text": "Ok."},
    ]
    summary = extractive_summary(segments, max_bullets=2)
    assert "Summary" in summary
    actions = extract_action_items(segments)
    assert "Action items" in actions
    assert "follow up" in actions.lower() or "will" in actions.lower()
    notes = build_meeting_notes(segments)
    assert notes["summary"]
    assert notes["action_items"]


def test_pipeline_chunk_diarize_summary(tmp_path, monkeypatch):
    source = tmp_path / "meeting.wav"
    sf.write(str(source), np.zeros(1600, dtype=np.float32), 16000)

    pre = tmp_path / "pre.wav"
    sf.write(str(pre), np.zeros(1600, dtype=np.float32), 16000)

    monkeypatch.setattr("modules.pipeline.is_video_file", lambda _p: False)
    monkeypatch.setattr("modules.pipeline.preprocess_audio", lambda _p: pre)
    monkeypatch.setattr("modules.pipeline.audio_duration_seconds", lambda _p: 5.0)
    monkeypatch.setattr("modules.pipeline.should_chunk", lambda *_a, **_k: False)
    monkeypatch.setattr(
        "modules.pipeline._transcribe_path",
        lambda *args, **kwargs: [
            {"start": 0.0, "end": 1.0, "text": "hello there everyone"},
            {"start": 3.5, "end": 4.5, "text": "we should follow up tomorrow"},
        ],
    )

    result = transcribe_file(
        source,
        model_name="small",
        language=None,
        convert_traditional=False,
        model=object(),
        enable_chunking=False,
        enable_diarization=True,
        enable_summary=True,
        diarization_gap_sec=2.0,
    )
    assert result["segment_count"] == 2
    assert result["segments"][0].get("speaker")
    assert "Summary" in result["summary"]
    assert result["action_items"]


def test_auth_helpers(monkeypatch):
    monkeypatch.setattr("utils.auth.APP_PASSWORD", "")
    assert password_required() is False
    assert verify_password("anything") is True

    monkeypatch.setattr("utils.auth.APP_PASSWORD", "secret")
    assert password_required() is True
    assert verify_password("secret") is True
    assert verify_password("nope") is False
