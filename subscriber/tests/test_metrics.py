from __future__ import annotations

import asyncio
import importlib


class _ContinuousQueue:
    async def get(self) -> float:
        await asyncio.sleep(0.001)
        return 0.01


async def test_latency_reporter_warns_from_background_task(mocker) -> None:
    metrics = importlib.import_module("subscriber.metrics")
    warning_logged = asyncio.Event()
    warning = mocker.patch(
        "subscriber.metrics.logger.warning",
        side_effect=lambda *args, **kwargs: warning_logged.set(),
    )
    reporter = metrics.LatencyReporter(summary_interval_s=1)

    await reporter.start()
    reporter.report(0.051)

    try:
        await asyncio.wait_for(warning_logged.wait(), timeout=1)
    finally:
        await reporter.stop()
    warning.assert_called_once_with(
        "kv event forwarding latency exceeded threshold",
        step="kv_event_metrics",
        tags={"latency_ms": 51.0, "threshold_ms": 50.0},
    )


async def test_latency_reporter_logs_minute_average(mocker) -> None:
    metrics = importlib.import_module("subscriber.metrics")
    summary_logged = asyncio.Event()

    def _record_info(*args, **kwargs) -> None:
        if kwargs.get("tags", {}).get("sample_count") == 2:
            summary_logged.set()

    info = mocker.patch("subscriber.metrics.logger.info", side_effect=_record_info)
    reporter = metrics.LatencyReporter(summary_interval_s=0.01)

    await reporter.start()
    reporter.report(0.01)
    reporter.report(0.03)

    try:
        await asyncio.wait_for(summary_logged.wait(), timeout=1)
    finally:
        await reporter.stop()
    info.assert_called_with(
        "kv event forwarding latency average",
        step="kv_event_metrics",
        tags={"average_latency_ms": 20.0, "sample_count": 2},
    )


async def test_latency_reporter_logs_average_during_continuous_traffic(mocker) -> None:
    metrics = importlib.import_module("subscriber.metrics")
    summary_logged = asyncio.Event()
    mocker.patch(
        "subscriber.metrics.logger.info",
        side_effect=lambda *args, **kwargs: summary_logged.set(),
    )
    reporter = metrics.LatencyReporter(summary_interval_s=0.01)
    reporter._queue = _ContinuousQueue()

    await reporter.start()
    try:
        await asyncio.wait_for(summary_logged.wait(), timeout=0.5)
    finally:
        await reporter.stop()


async def test_latency_reporter_offloads_warning_logging(mocker) -> None:
    metrics = importlib.import_module("subscriber.metrics")
    logging_offloaded = asyncio.Event()

    async def _to_thread(*args, **kwargs) -> None:
        del args, kwargs
        logging_offloaded.set()

    mocker.patch("subscriber.metrics.asyncio.to_thread", side_effect=_to_thread)
    reporter = metrics.LatencyReporter(summary_interval_s=1)

    await reporter.start()
    reporter.report(0.051)

    try:
        await asyncio.wait_for(logging_offloaded.wait(), timeout=1)
    finally:
        await reporter.stop()


async def test_latency_reporter_start_is_idempotent() -> None:
    metrics = importlib.import_module("subscriber.metrics")
    reporter = metrics.LatencyReporter()

    await reporter.start()
    first_task = reporter._task
    await reporter.start()

    try:
        assert reporter._task is first_task
    finally:
        await reporter.stop()


def test_latency_reporter_swallows_report_errors(mocker) -> None:
    metrics = importlib.import_module("subscriber.metrics")
    reporter = metrics.LatencyReporter()
    reporter._queue = mocker.Mock()
    reporter._queue.put_nowait.side_effect = RuntimeError("queue failure")

    reporter.report(0.01)
