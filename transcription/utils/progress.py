"""Progress tracking with estimated time remaining."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProgressTracker:
    """Track overall progress across weighted work units and estimate ETA."""

    total_units: float = 1.0
    completed_units: float = 0.0
    current_unit_fraction: float = 0.0
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def fraction(self) -> float:
        """Overall completion fraction in [0, 1]."""
        if self.total_units <= 0:
            return 1.0
        return min(1.0, (self.completed_units + self.current_unit_fraction) / self.total_units)

    def set_current(self, fraction: float) -> None:
        """Update progress within the current work unit."""
        self.current_unit_fraction = max(0.0, min(1.0, fraction))

    def complete_unit(self, units: float = 1.0) -> None:
        """Mark one or more work units as finished."""
        self.completed_units += units
        self.current_unit_fraction = 0.0

    def reset(self, total_units: float) -> None:
        """Reset tracker for a new job."""
        self.total_units = total_units
        self.completed_units = 0.0
        self.current_unit_fraction = 0.0
        self._start_time = time.monotonic()

    def elapsed_seconds(self) -> float:
        """Seconds elapsed since tracking started."""
        return time.monotonic() - self._start_time

    def eta_seconds(self) -> float | None:
        """Estimated seconds remaining, or None if not enough data yet."""
        frac = self.fraction
        if frac < 0.02:
            return None
        elapsed = self.elapsed_seconds()
        return elapsed * (1.0 - frac) / frac
