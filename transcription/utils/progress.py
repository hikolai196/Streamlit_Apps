"""Progress tracking with estimated time remaining."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from config import PROGRESS_REALTIME_FACTOR, PROGRESS_SECONDS_PER_UNIT


def duration_to_units(duration_sec: float | None) -> float:
    """
    Convert audio duration into batch work units.

    Longer files contribute more weight so overall ETA stays realistic.
    """
    if duration_sec is None or duration_sec <= 0:
        return 1.0
    return max(1.0, float(duration_sec) / PROGRESS_SECONDS_PER_UNIT)


@dataclass
class ProgressTracker:
    """Track overall progress across weighted work units and estimate ETA."""

    total_units: float = 1.0
    completed_units: float = 0.0
    current_unit_fraction: float = 0.0
    # Optional audio-duration hint (seconds) for early ETA before enough progress.
    expected_audio_seconds: float | None = None
    realtime_factor: float = PROGRESS_REALTIME_FACTOR
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def fraction(self) -> float:
        """Overall completion fraction in [0, 1]."""
        if self.total_units <= 0:
            return 1.0
        return min(1.0, (self.completed_units + self.current_unit_fraction) / self.total_units)

    def set_current(self, fraction: float, unit_weight: float = 1.0) -> None:
        """
        Update progress within the current work unit.

        ``fraction`` is in [0, 1] for the active unit; ``unit_weight`` scales it
        when batch jobs use duration-based units.
        """
        clamped = max(0.0, min(1.0, fraction))
        weight = max(0.0, unit_weight)
        self.current_unit_fraction = clamped * weight

    def complete_unit(self, units: float = 1.0) -> None:
        """Mark one or more work units as finished."""
        self.completed_units += units
        self.current_unit_fraction = 0.0

    def reset(self, total_units: float, expected_audio_seconds: float | None = None) -> None:
        """Reset tracker for a new job."""
        self.total_units = total_units
        self.completed_units = 0.0
        self.current_unit_fraction = 0.0
        self.expected_audio_seconds = expected_audio_seconds
        self._start_time = time.monotonic()

    def elapsed_seconds(self) -> float:
        """Seconds elapsed since tracking started."""
        return time.monotonic() - self._start_time

    def eta_seconds(self) -> float | None:
        """Estimated seconds remaining, or None if not enough data yet."""
        frac = self.fraction
        elapsed = self.elapsed_seconds()

        # Once we have a little progress, prefer measured throughput.
        if frac >= 0.02:
            return elapsed * (1.0 - frac) / frac

        # Early hint from audio duration (assume ~realtime_factor of realtime).
        if self.expected_audio_seconds and self.expected_audio_seconds > 0:
            expected_total = self.expected_audio_seconds * max(0.05, self.realtime_factor)
            remaining = expected_total - elapsed
            return max(0.0, remaining)

        return None
