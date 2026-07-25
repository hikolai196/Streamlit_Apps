"""Tests for progress tracking utilities."""

from utils.progress import ProgressTracker, duration_to_units


def test_progress_tracker_fraction():
    tracker = ProgressTracker(total_units=2.0)
    tracker.set_current(0.5)
    assert tracker.fraction == 0.25

    tracker.complete_unit()
    assert tracker.fraction == 0.5


def test_progress_tracker_weighted_current_unit():
    tracker = ProgressTracker(total_units=4.0)
    tracker.set_current(0.5, unit_weight=2.0)
    assert tracker.fraction == 0.25


def test_progress_tracker_eta_after_progress():
    tracker = ProgressTracker(total_units=1.0)
    tracker.set_current(0.5)
    eta = tracker.eta_seconds()
    assert eta is not None
    assert eta >= 0


def test_progress_tracker_early_eta_uses_duration_hint():
    tracker = ProgressTracker(total_units=1.0, expected_audio_seconds=100.0, realtime_factor=0.5)
    eta = tracker.eta_seconds()
    assert eta is not None
    assert 40 <= eta <= 50


def test_progress_tracker_reset():
    tracker = ProgressTracker(total_units=1.0)
    tracker.complete_unit()
    tracker.reset(total_units=3.0, expected_audio_seconds=30.0)
    assert tracker.fraction == 0.0
    assert tracker.total_units == 3.0
    assert tracker.expected_audio_seconds == 30.0


def test_duration_to_units():
    assert duration_to_units(None) == 1.0
    assert duration_to_units(0) == 1.0
    assert duration_to_units(30) == 1.0
    assert duration_to_units(120) == 2.0
