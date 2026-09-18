from __future__ import annotations

import pytest

from subscriber.engine.interval_adjuster import DynamicIntervalAdjuster


class TestDynamicIntervalAdjuster:
    def test_initial_interval(self) -> None:
        adjuster = DynamicIntervalAdjuster(
            target_diff_size=100,
            min_interval_s=0.1,
            max_interval_s=10.0,
            initial_interval_s=1.0,
        )
        assert adjuster.current_interval_s == 1.0

    def test_no_adjustment_below_min_samples(self) -> None:
        adjuster = DynamicIntervalAdjuster(
            target_diff_size=100,
            min_interval_s=0.1,
            max_interval_s=10.0,
            initial_interval_s=1.0,
        )
        # First update: only 1 sample, no adjustment
        adjuster.update(500)
        assert adjuster.current_interval_s == 1.0

        # Second update: only 2 samples, still no adjustment
        adjuster.update(500)
        assert adjuster.current_interval_s == 1.0

    def test_no_adjustment_within_threshold(self) -> None:
        adjuster = DynamicIntervalAdjuster(
            target_diff_size=100,
            min_interval_s=0.1,
            max_interval_s=10.0,
            initial_interval_s=1.0,
        )
        # Feed 3 samples that average to exactly target (deviation = 0)
        adjuster.update(100)
        adjuster.update(100)
        adjuster.update(100)
        assert adjuster.current_interval_s == 1.0

        # Feed samples averaging within 10% threshold
        # 3 samples: 95, 100, 105 -> avg = 100, deviation = 0
        adjuster.update(95)
        adjuster.update(100)
        adjuster.update(105)
        assert adjuster.current_interval_s == 1.0

    def test_shorten_interval_when_diff_large(self) -> None:
        adjuster = DynamicIntervalAdjuster(
            target_diff_size=100,
            min_interval_s=0.1,
            max_interval_s=10.0,
            initial_interval_s=1.0,
        )
        # Feed 3 samples well above target so rolling_avg >> target
        # deviation > 10%, rolling_avg > target -> shorten by 30%
        adjuster.update(200)
        adjuster.update(200)
        adjuster.update(200)
        # avg = 200, deviation = (200-100)/100 = 1.0 > 0.1
        # new_interval = 1.0 * (1 - 0.3) = 0.7
        assert adjuster.current_interval_s == pytest.approx(0.7)

    def test_lengthen_interval_when_diff_small(self) -> None:
        adjuster = DynamicIntervalAdjuster(
            target_diff_size=100,
            min_interval_s=0.1,
            max_interval_s=10.0,
            initial_interval_s=1.0,
        )
        # Feed 3 samples well below target so rolling_avg << target
        # deviation < -10%, rolling_avg < target -> lengthen by 30%
        adjuster.update(10)
        adjuster.update(10)
        adjuster.update(10)
        # avg = 10, deviation = (10-100)/100 = -0.9, |deviation| > 0.1
        # new_interval = 1.0 * (1 + 0.3) = 1.3
        assert adjuster.current_interval_s == pytest.approx(1.3)

    def test_clamp_to_min(self) -> None:
        adjuster = DynamicIntervalAdjuster(
            target_diff_size=100,
            min_interval_s=0.5,
            max_interval_s=10.0,
            initial_interval_s=1.0,
        )
        # Repeatedly shorten: 1.0 -> 0.7 -> 0.49 -> clamped to 0.5
        adjuster.update(500)
        adjuster.update(500)
        adjuster.update(500)
        # 1.0 * 0.7 = 0.7
        assert adjuster.current_interval_s == pytest.approx(0.7)

        adjuster.update(500)
        # 0.7 * 0.7 = 0.49, clamped to 0.5
        assert adjuster.current_interval_s == pytest.approx(0.5)

    def test_clamp_to_max(self) -> None:
        adjuster = DynamicIntervalAdjuster(
            target_diff_size=100,
            min_interval_s=0.1,
            max_interval_s=2.0,
            initial_interval_s=1.5,
        )
        # Repeatedly lengthen: 1.5 -> 1.95 -> 2.535 -> clamped to 2.0
        adjuster.update(5)
        adjuster.update(5)
        adjuster.update(5)
        # 1.5 * 1.3 = 1.95
        assert adjuster.current_interval_s == pytest.approx(1.95)

        adjuster.update(5)
        # 1.95 * 1.3 = 2.535, clamped to 2.0
        assert adjuster.current_interval_s == pytest.approx(2.0)

    def test_rolling_average_smooths_spikes(self) -> None:
        adjuster = DynamicIntervalAdjuster(
            target_diff_size=100,
            min_interval_s=0.1,
            max_interval_s=10.0,
            initial_interval_s=1.0,
        )
        # Fill window with target-matching values
        for _ in range(30):
            adjuster.update(100)
        assert adjuster.current_interval_s == 1.0

        # One spike
        adjuster.update(10000)
        # Window: 29 * 100 + 10000 = 12900, avg = 12900/30 = 430
        # deviation = (430 - 100) / 100 = 3.3 > 0.1 -> shorten
        # new_interval = 1.0 * 0.7 = 0.7
        # The spike caused ONE adjustment, not a wild swing
        assert adjuster.current_interval_s == pytest.approx(0.7)

    def test_reset_clears_state(self) -> None:
        adjuster = DynamicIntervalAdjuster(
            target_diff_size=100,
            min_interval_s=0.1,
            max_interval_s=10.0,
            initial_interval_s=1.0,
        )
        # Make some adjustments
        adjuster.update(500)
        adjuster.update(500)
        adjuster.update(500)
        assert adjuster.current_interval_s != 1.0

        # Reset
        adjuster.reset()
        assert adjuster.current_interval_s == 1.0

        # After reset, first 2 updates should not adjust (history cleared)
        adjuster.update(500)
        adjuster.update(500)
        assert adjuster.current_interval_s == 1.0

    def test_convergence(self) -> None:
        adjuster = DynamicIntervalAdjuster(
            target_diff_size=100,
            min_interval_s=0.1,
            max_interval_s=10.0,
            initial_interval_s=5.0,
        )
        # Repeatedly feed a diff_size above target; interval should keep
        # shrinking and eventually clamp to min
        for _ in range(100):
            adjuster.update(300)

        # After many updates with consistently high diff, interval is at min
        assert adjuster.current_interval_s == pytest.approx(0.1)

    def test_window_is_rolling_not_cumulative(self) -> None:
        """After WINDOW_SIZE samples, old values fall out of the window."""
        adjuster = DynamicIntervalAdjuster(
            target_diff_size=100,
            min_interval_s=0.1,
            max_interval_s=10.0,
            initial_interval_s=1.0,
        )
        # Fill 30 samples at target
        for _ in range(30):
            adjuster.update(100)
        interval_after_fill = adjuster.current_interval_s

        # Now feed 30 more samples at a very different value
        for _ in range(30):
            adjuster.update(10)

        # The old 100s have all fallen out; window is all 10s
        # avg = 10, deviation = (10-100)/100 = -0.9 -> lengthen
        # The interval should have been lengthened from whatever it was
        assert adjuster.current_interval_s > interval_after_fill
