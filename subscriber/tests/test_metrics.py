from __future__ import annotations

import asyncio
import importlib

import pytest


def _spans(module, *pairs: tuple[str, float]) -> list:
    return [module.Span(name, duration) for name, duration in pairs]


def _sample(module, *pairs: tuple[str, float], counters: dict[str, int] | None = None):
    return module.MetricSample(
        _spans(module, *pairs),
        {} if counters is None else counters,
    )


class _ContinuousSpanQueue:
    async def get(self):
        await asyncio.sleep(0.001)
        metrics = importlib.import_module("subscriber.metrics")
        return _sample(metrics, ("kvcm_send", 0.01))


def test_stage_timer_derives_named_spans_between_marks() -> None:
    metrics = importlib.import_module("subscriber.metrics")
    ticks = iter([100.0, 100.2, 100.5, 101.0])
    timer = metrics.StageTimer(clock=lambda: next(ticks))

    timer.mark("queue_wait")
    timer.mark("gate_wait")
    timer.mark("kvcm_send")

    spans = timer.spans()
    assert [span.name for span in spans] == ["queue_wait", "gate_wait", "kvcm_send"]
    assert [span.duration_s for span in spans] == pytest.approx([0.2, 0.3, 0.5])


def test_stage_timer_without_marks_has_no_spans() -> None:
    metrics = importlib.import_module("subscriber.metrics")
    timer = metrics.StageTimer(clock=lambda: 5.0)

    assert timer.spans() == []


async def test_reporter_warns_with_stage_breakdown(mocker) -> None:
    metrics = importlib.import_module("subscriber.metrics")
    warning_logged = asyncio.Event()
    warning = mocker.patch(
        "subscriber.metrics.logger.warning",
        side_effect=lambda *args, **kwargs: warning_logged.set(),
    )
    reporter = metrics.SpanMetricsReporter(summary_interval_s=1)

    await reporter.start()
    reporter.report(
        _sample(
            metrics,
            ("queue_wait", 0.01),
            ("gate_wait", 0.005),
            ("kvcm_send", 0.05),
        )
    )

    try:
        await asyncio.wait_for(warning_logged.wait(), timeout=1)
    finally:
        await reporter.stop()
    warning.assert_called_once_with(
        "kv event forwarding latency exceeded threshold",
        step="kv_event_metrics",
        tags={
            "queue_wait_ms": 10.0,
            "gate_wait_ms": 5.0,
            "kvcm_send_ms": 50.0,
            "total_ms": 65.0,
            "threshold_ms": 50.0,
        },
    )


async def test_reporter_skips_warning_below_threshold(mocker) -> None:
    metrics = importlib.import_module("subscriber.metrics")
    summary_logged = asyncio.Event()
    warning = mocker.patch("subscriber.metrics.logger.warning")
    mocker.patch(
        "subscriber.metrics.logger.info",
        side_effect=lambda *args, **kwargs: summary_logged.set(),
    )
    reporter = metrics.SpanMetricsReporter(summary_interval_s=0.01)

    await reporter.start()
    reporter.report(_sample(metrics, ("kvcm_send", 0.001)))

    try:
        await asyncio.wait_for(summary_logged.wait(), timeout=1)
    finally:
        await reporter.stop()
    warning.assert_not_called()


async def test_reporter_logs_stage_averages(mocker) -> None:
    metrics = importlib.import_module("subscriber.metrics")
    summary_logged = asyncio.Event()

    def _record_info(*args, **kwargs) -> None:
        if kwargs.get("tags", {}).get("sample_count") == 2:
            summary_logged.set()

    info = mocker.patch("subscriber.metrics.logger.info", side_effect=_record_info)
    reporter = metrics.SpanMetricsReporter(summary_interval_s=0.01)

    await reporter.start()
    reporter.report(
        metrics.MetricSample(
            [metrics.Span("queue_wait", 0.01), metrics.Span("kvcm_send", 0.03)],
            {"stored_block_hash_count": 3, "removed_block_hash_count": 2},
        )
    )
    reporter.report(
        metrics.MetricSample(
            [metrics.Span("queue_wait", 0.03), metrics.Span("kvcm_send", 0.05)],
            {"stored_block_hash_count": 4, "removed_block_hash_count": 1},
        )
    )

    try:
        await asyncio.wait_for(summary_logged.wait(), timeout=1)
    finally:
        await reporter.stop()
    info.assert_called_with(
        "kv event forwarding latency average",
        step="kv_event_metrics",
        tags={
            "queue_wait_avg_ms": 20.0,
            "kvcm_send_avg_ms": 40.0,
            "total_avg_ms": 60.0,
            "sample_count": 2,
            "stored_block_hash_count": 7,
            "removed_block_hash_count": 3,
        },
    )


async def test_reporter_resets_counters_after_summary_interval(mocker) -> None:
    metrics = importlib.import_module("subscriber.metrics")
    first_summary_logged = asyncio.Event()
    second_summary_logged = asyncio.Event()
    summaries: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()

    def _record_info(*args, **kwargs) -> None:
        tags = kwargs["tags"]
        if "stored_block_hash_count" not in tags:
            return
        summaries.append(dict(tags))
        event = first_summary_logged if len(summaries) == 1 else second_summary_logged
        loop.call_soon_threadsafe(event.set)

    mocker.patch("subscriber.metrics.logger.info", side_effect=_record_info)
    reporter = metrics.SpanMetricsReporter(summary_interval_s=0.01)

    await reporter.start()
    reporter.report(
        _sample(
            metrics,
            ("kvcm_send", 0.001),
            counters={"stored_block_hash_count": 3, "removed_block_hash_count": 2},
        )
    )
    try:
        await asyncio.wait_for(first_summary_logged.wait(), timeout=1)
        await asyncio.sleep(0)
        reporter.report(
            _sample(
                metrics,
                ("kvcm_send", 0.001),
                counters={"stored_block_hash_count": 4, "removed_block_hash_count": 1},
            )
        )
        await asyncio.wait_for(second_summary_logged.wait(), timeout=1)
    finally:
        await reporter.stop()

    assert [
        (summary["stored_block_hash_count"], summary["removed_block_hash_count"])
        for summary in summaries
    ] == [(3, 2), (4, 1)]


async def test_reporter_averages_each_stage_independently(mocker) -> None:
    metrics = importlib.import_module("subscriber.metrics")
    summary_logged = asyncio.Event()

    def _record_info(*args, **kwargs) -> None:
        if kwargs.get("tags", {}).get("sample_count") == 2:
            summary_logged.set()

    info = mocker.patch("subscriber.metrics.logger.info", side_effect=_record_info)
    reporter = metrics.SpanMetricsReporter(summary_interval_s=0.01)

    await reporter.start()
    reporter.report(_sample(metrics, ("decode", 0.02), ("queue_wait", 0.01)))
    reporter.report(_sample(metrics, ("replay_fetch", 0.10), ("queue_wait", 0.03)))

    try:
        await asyncio.wait_for(summary_logged.wait(), timeout=1)
    finally:
        await reporter.stop()
    info.assert_called_with(
        "kv event forwarding latency average",
        step="kv_event_metrics",
        tags={
            "decode_avg_ms": 20.0,
            "queue_wait_avg_ms": 20.0,
            "replay_fetch_avg_ms": 100.0,
            "total_avg_ms": 80.0,
            "sample_count": 2,
        },
    )


async def test_reporter_logs_average_during_continuous_traffic(mocker) -> None:
    metrics = importlib.import_module("subscriber.metrics")
    summary_logged = asyncio.Event()
    mocker.patch(
        "subscriber.metrics.logger.info",
        side_effect=lambda *args, **kwargs: summary_logged.set(),
    )
    reporter = metrics.SpanMetricsReporter(summary_interval_s=0.01)
    reporter._queue = _ContinuousSpanQueue()

    await reporter.start()
    try:
        await asyncio.wait_for(summary_logged.wait(), timeout=0.5)
    finally:
        await reporter.stop()


async def test_reporter_offloads_warning_logging(mocker) -> None:
    metrics = importlib.import_module("subscriber.metrics")
    logging_offloaded = asyncio.Event()

    async def _to_thread(*args, **kwargs) -> None:
        del args, kwargs
        logging_offloaded.set()

    mocker.patch("subscriber.metrics.asyncio.to_thread", side_effect=_to_thread)
    reporter = metrics.SpanMetricsReporter(summary_interval_s=1)

    await reporter.start()
    reporter.report(_sample(metrics, ("kvcm_send", 0.051)))

    try:
        await asyncio.wait_for(logging_offloaded.wait(), timeout=1)
    finally:
        await reporter.stop()


async def test_reporter_start_is_idempotent() -> None:
    metrics = importlib.import_module("subscriber.metrics")
    reporter = metrics.SpanMetricsReporter()

    await reporter.start()
    first_task = reporter._task
    await reporter.start()

    try:
        assert reporter._task is first_task
    finally:
        await reporter.stop()


def test_reporter_swallows_report_errors(mocker) -> None:
    metrics = importlib.import_module("subscriber.metrics")
    reporter = metrics.SpanMetricsReporter()
    reporter._queue = mocker.Mock()
    reporter._queue.put_nowait.side_effect = RuntimeError("queue failure")

    reporter.report(_sample(metrics, ("kvcm_send", 0.01)))
