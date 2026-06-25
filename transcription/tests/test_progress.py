"""Tests for progress tracking utilities."""

from utils.progress import ProgressTracker


def test_progress_tracker_fraction():
    tracker = ProgressTracker(total_units=2.0)
    tracker.set_current(0.5)
    assert tracker.fraction == 0.25

    tracker.complete_unit()
    assert tracker.fraction == 0.5


def test_progress_tracker_eta_after_progress():
    tracker = ProgressTracker(total_units=1.0)
    tracker.set_current(0.5)
    eta = tracker.eta_seconds()
    assert eta is not None
    assert eta >= 0


def test_progress_tracker_reset():
    tracker = ProgressTracker(total_units=1.0)
    tracker.complete_unit()
    tracker.reset(total_units=3.0)
    assert tracker.fraction == 0.0
    assert tracker.total_units == 3.0
