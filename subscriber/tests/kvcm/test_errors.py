from __future__ import annotations

import pytest

from subscriber.forwarding import KvcmDropTracker
from subscriber.kvcm.errors import (
    KvcmReportError,
    KvcmReportRejectedError,
    KvcmResponseRejectedError,
    KvcmUnavailableError,
)


class TestKvcmReportErrorHierarchy:
    """The typed KVCM report error hierarchy."""

    def test_base_is_runtime_error(self) -> None:
        assert isinstance(KvcmReportError("base"), RuntimeError)

    def test_unavailable_is_report_error(self) -> None:
        exc = KvcmUnavailableError("transport failure")
        assert isinstance(exc, KvcmReportError)

    def test_rejected_is_report_error(self) -> None:
        exc = KvcmReportRejectedError("rejected")
        assert isinstance(exc, KvcmReportError)

    def test_transport_rejected_is_internal_runtime_error(self) -> None:
        exc = KvcmResponseRejectedError("rejected")
        assert isinstance(exc, RuntimeError)
        assert not isinstance(exc, KvcmReportError)

    def test_subclasses_are_distinct(self) -> None:
        assert not issubclass(KvcmUnavailableError, KvcmReportRejectedError)
        assert not issubclass(KvcmReportRejectedError, KvcmUnavailableError)

    def test_catch_base_catches_both_subclasses(self) -> None:
        for exc in (
            KvcmUnavailableError("unavailable"),
            KvcmReportRejectedError("rejected"),
        ):
            with pytest.raises(KvcmReportError):
                raise exc


class _FakeClock:
    """Deterministic monotonic clock for testing time-based rate limits."""

    def __init__(self, start_s: float = 0.0) -> None:
        self.now_s = start_s

    def __call__(self) -> float:
        return self.now_s

    def advance(self, seconds: float) -> None:
        self.now_s += seconds


class TestKvcmDropTracker:
    """Rate-limited drop warnings: first immediate, then summarized."""

    def test_first_drop_logs_immediately(self, mocker) -> None:
        warning = mocker.patch("subscriber.forwarding.logger.warning")
        tracker = KvcmDropTracker(clock=_FakeClock())

        tracker.record_drop(
            epoch=1,
            batch_count=2,
            event_count=4,
            error=KvcmUnavailableError("down"),
        )

        warning.assert_called_once_with(
            "failed to send kv event batch to kvcm; dropping batch",
            step="kvcm_send",
            tags={
                "pipeline": "incremental",
                "epoch": 1,
                "batch_count": 2,
                "event_count": 4,
                "error": "KvcmUnavailableError",
                "message": "down",
                "reason": "unknown",
                "dropped_batch_total": 2,
                "dropped_event_total": 4,
            },
        )
        assert tracker.dropped_batch_count == 2
        assert tracker.dropped_event_count == 4

    def test_subsequent_drops_within_window_are_silent(self, mocker) -> None:
        warning = mocker.patch("subscriber.forwarding.logger.warning")
        clock = _FakeClock()
        tracker = KvcmDropTracker(
            summary_every=100, summary_interval_s=30.0, clock=clock
        )

        tracker.record_drop(
            epoch=1, batch_count=1, event_count=1, error=KvcmUnavailableError("e")
        )
        warning.reset_mock()

        # Within both the count and time window: no further logging.
        clock.advance(1.0)
        tracker.record_drop(
            epoch=1, batch_count=1, event_count=1, error=KvcmUnavailableError("e")
        )

        warning.assert_not_called()
        assert tracker.dropped_batch_count == 2

    def test_summary_emitted_after_count_threshold(self, mocker) -> None:
        warning = mocker.patch("subscriber.forwarding.logger.warning")
        clock = _FakeClock()
        tracker = KvcmDropTracker(
            summary_every=3, summary_interval_s=9999.0, clock=clock
        )

        # First drop logs immediately.
        tracker.record_drop(
            epoch=1, batch_count=1, event_count=1, error=KvcmUnavailableError("e")
        )
        warning.reset_mock()

        # Drops 2 and 3 accumulate but stay under the count threshold.
        tracker.record_drop(
            epoch=1, batch_count=1, event_count=1, error=KvcmUnavailableError("e")
        )
        tracker.record_drop(
            epoch=1, batch_count=1, event_count=1, error=KvcmUnavailableError("e")
        )
        warning.assert_not_called()

        # Drop 4 crosses summary_every=3 since the first failure.
        tracker.record_drop(
            epoch=1, batch_count=1, event_count=1, error=KvcmUnavailableError("e")
        )
        warning.assert_called_once()
        tags = warning.call_args.kwargs["tags"]
        assert tags["dropped_batch_since_summary"] == 3
        assert tags["dropped_batch_total"] == 4
        assert tags["dropped_event_total"] == 4

    def test_summary_emitted_after_time_threshold(self, mocker) -> None:
        warning = mocker.patch("subscriber.forwarding.logger.warning")
        clock = _FakeClock()
        tracker = KvcmDropTracker(
            summary_every=1000, summary_interval_s=30.0, clock=clock
        )

        tracker.record_drop(
            epoch=1, batch_count=1, event_count=1, error=KvcmUnavailableError("e")
        )
        warning.reset_mock()

        # Advance past the time threshold; a single drop triggers a summary.
        clock.advance(31.0)
        tracker.record_drop(
            epoch=1, batch_count=1, event_count=1, error=KvcmReportRejectedError("r")
        )

        warning.assert_called_once_with(
            "kvcm send failures continue; dropping batches",
            step="kvcm_send",
            tags={
                "pipeline": "incremental",
                "epoch": 1,
                "error": "KvcmReportRejectedError",
                "message": "r",
                "reason": "unknown",
                "dropped_batch_since_summary": 1,
                "dropped_batch_total": 2,
                "dropped_event_total": 2,
            },
        )

    def test_summary_counter_resets_after_emit(self, mocker) -> None:
        warning = mocker.patch("subscriber.forwarding.logger.warning")
        clock = _FakeClock()
        tracker = KvcmDropTracker(
            summary_every=2, summary_interval_s=9999.0, clock=clock
        )

        # Drop 1: first failure, immediate log.
        tracker.record_drop(
            epoch=1, batch_count=1, event_count=1, error=KvcmUnavailableError("e")
        )
        warning.reset_mock()

        # Drop 2: accumulates, still under threshold.
        tracker.record_drop(
            epoch=1, batch_count=1, event_count=1, error=KvcmUnavailableError("e")
        )
        warning.assert_not_called()

        # Drop 3: reaches summary_every=2, emits and resets the counter.
        tracker.record_drop(
            epoch=1, batch_count=1, event_count=1, error=KvcmUnavailableError("e")
        )
        warning.assert_called_once()
        tags = warning.call_args.kwargs["tags"]
        assert tags["dropped_batch_since_summary"] == 2
        assert tags["dropped_batch_total"] == 3
        warning.reset_mock()

        # Drop 4: counter restarted, accumulates silently.
        tracker.record_drop(
            epoch=1, batch_count=1, event_count=1, error=KvcmUnavailableError("e")
        )
        warning.assert_not_called()

        # Drop 5: reaches the threshold again.
        tracker.record_drop(
            epoch=1, batch_count=1, event_count=1, error=KvcmUnavailableError("e")
        )
        warning.assert_called_once()
        tags = warning.call_args.kwargs["tags"]
        assert tags["dropped_batch_since_summary"] == 2
        assert tags["dropped_batch_total"] == 5
