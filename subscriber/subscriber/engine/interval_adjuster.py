from __future__ import annotations

from collections import deque


class DynamicIntervalAdjuster:
    """Dynamically adjusts a polling interval based on observed diff sizes.

    Uses a rolling window to smooth observations and adjusts the interval
    toward the target diff size using a dampening factor.
    """

    WINDOW_SIZE: int = 30
    DAMPENING_FACTOR: float = 0.3
    ADJUSTMENT_THRESHOLD: float = 0.1
    MIN_SAMPLES: int = 3

    def __init__(
        self,
        target_diff_size: int,
        min_interval_s: float,
        max_interval_s: float,
        initial_interval_s: float,
    ) -> None:
        self._target_diff_size = target_diff_size
        self._min_interval_s = min_interval_s
        self._max_interval_s = max_interval_s
        self._initial_interval_s = initial_interval_s
        self._current_interval_s = initial_interval_s
        self._window: deque[int] = deque(maxlen=self.WINDOW_SIZE)

    @property
    def current_interval_s(self) -> float:
        return self._current_interval_s

    def update(self, diff_size: int) -> None:
        self._window.append(diff_size)

        if len(self._window) < self.MIN_SAMPLES:
            return

        rolling_average = sum(self._window) / len(self._window)
        deviation = (rolling_average - self._target_diff_size) / self._target_diff_size

        if abs(deviation) < self.ADJUSTMENT_THRESHOLD:
            return

        if rolling_average > self._target_diff_size:
            new_interval = self._current_interval_s * (1 - self.DAMPENING_FACTOR)
        else:
            new_interval = self._current_interval_s * (1 + self.DAMPENING_FACTOR)

        self._current_interval_s = max(
            self._min_interval_s, min(self._max_interval_s, new_interval)
        )

    def reset(self) -> None:
        """Clear history on engine restart. Reset interval to initial value."""
        self._window.clear()
        self._current_interval_s = self._initial_interval_s
