"""Tests for the new metrics reporter architecture (BatchTelemetry + MetricsReporter).

These tests exercise the two public seams introduced by the metrics refactor:

- ``BatchTelemetry``: main-path telemetry container with ``mark`` / ``count`` /
  ``mark_dropped`` verbs; observed via its finalized ``spans`` / ``counters`` /
  ``drop_reason`` / ``pipeline`` snapshot.
- ``MetricsReporter``: the single async output for the main path; observed via
  the dashlog counter/gauge calls it emits when a telemetry is submitted.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest


def _telemetry(module, pipeline: str = "incremental", clock=None):
    if clock is None:
        return module.BatchTelemetry(pipeline=pipeline)
    return module.BatchTelemetry(pipeline=pipeline, clock=clock)


# ---------------------------------------------------------------------------
# BatchTelemetry
# ---------------------------------------------------------------------------


class TestBatchTelemetry:
    def test_mark_produces_named_spans_between_marks(self) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        ticks = iter([100.0, 100.2, 100.5, 101.0])
        telemetry = _telemetry(metrics, clock=lambda: next(ticks))

        telemetry.mark("queue_wait")
        telemetry.mark("gate_wait")
        telemetry.mark("kvcm_send")

        spans = telemetry.spans
        assert [span.name for span in spans] == [
            "queue_wait",
            "gate_wait",
            "kvcm_send",
        ]
        assert [span.duration_s for span in spans] == pytest.approx([0.2, 0.3, 0.5])

    def test_mark_returns_duration_since_previous_mark(self) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        ticks = iter([100.0, 100.2, 100.5])
        telemetry = _telemetry(metrics, clock=lambda: next(ticks))

        first_duration_s = telemetry.mark("snapshot_fetch")
        second_duration_s = telemetry.mark("snapshot_build")

        assert first_duration_s == pytest.approx(0.2)
        assert second_duration_s == pytest.approx(0.3)

    def test_mark_ignores_clock_failures(self) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        ticks = iter([100.0])

        def failing_clock() -> float:
            try:
                return next(ticks)
            except StopIteration as exc:
                raise RuntimeError("metric clock failed") from exc

        telemetry = _telemetry(metrics, clock=failing_clock)

        assert telemetry.mark("kvcm_send") == 0.0
        assert [span.name for span in telemetry.spans] == ["kvcm_send"]
        assert telemetry.spans[0].duration_s == 0.0

    def test_count_accumulates_named_counters(self) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        telemetry = _telemetry(metrics)

        telemetry.count("stored_block_hash_count", 3)
        telemetry.count("stored_block_hash_count", 2, tags={})
        telemetry.count("removed_block_hash_count", 1)

        assert telemetry.counters == {
            "stored_block_hash_count": 5,
            "removed_block_hash_count": 1,
        }

    def test_count_and_gauge_ignore_unconvertible_values(self) -> None:
        metrics = importlib.import_module("subscriber.metrics")

        class UnconvertibleValue:
            def __int__(self) -> int:
                raise RuntimeError("counter conversion failed")

            def __float__(self) -> float:
                raise RuntimeError("gauge conversion failed")

        telemetry = _telemetry(metrics)
        value = UnconvertibleValue()

        telemetry.count("stored_block_hash_count", value)
        telemetry.gauge("full_snapshot_payload_bytes", value)

        assert telemetry.counters == {}
        assert telemetry.gauges == {}

    def test_gauge_records_last_value_per_name(self) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        telemetry = _telemetry(metrics)

        telemetry.gauge("full_snapshot_fetch_latency_ms", 12.5)
        telemetry.gauge("full_snapshot_payload_bytes", 2048)
        # Repeated gauge writes overwrite (gauge semantics, not accumulate).
        telemetry.gauge("full_snapshot_fetch_latency_ms", 20.0, tags={})

        assert telemetry.gauges == {
            "full_snapshot_fetch_latency_ms": 20.0,
            "full_snapshot_payload_bytes": 2048,
        }

    def test_mark_dropped_records_reason_and_preserves_spans(self) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        ticks = iter([100.0, 100.1, 100.3])
        telemetry = _telemetry(metrics, clock=lambda: next(ticks))

        telemetry.mark("queue_wait")
        telemetry.mark_dropped("send_failed")

        assert telemetry.drop_reason == "send_failed"
        # The completed stage before the drop is still visible.
        assert [span.name for span in telemetry.spans] == ["queue_wait"]

    def test_checkpoint_and_elapsed_since_checkpoint(self) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        # 4 ticks: origin, checkpoint, elapsed-read, missing-check
        ticks = iter([100.0, 100.1, 100.5, 101.0])
        telemetry = _telemetry(metrics, clock=lambda: next(ticks))

        telemetry.checkpoint("queue_enqueued")
        elapsed = telemetry.elapsed_since_checkpoint("queue_enqueued")

        assert elapsed == pytest.approx(0.4)
        assert telemetry.elapsed_since_checkpoint("missing") is None

    def test_pipeline_is_exposed(self) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        telemetry = _telemetry(metrics, pipeline="snapshot")

        assert telemetry.pipeline == "snapshot"


# ---------------------------------------------------------------------------
# MetricsReporter
# ---------------------------------------------------------------------------


class TestMetricsReporterSuccess:
    async def test_flushes_stage_spans_e2e_latency_and_counters(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        gauge = mocker.patch("subscriber.metrics.reporter._dashlog_gauge")
        counter = mocker.patch("subscriber.metrics.reporter._dashlog_counter")

        ticks = iter([100.0, 100.002, 100.012, 100.042])
        telemetry = _telemetry(
            metrics, pipeline="incremental", clock=lambda: next(ticks)
        )
        telemetry.mark("decode")
        telemetry.mark("queue_wait")
        telemetry.mark("kvcm_send")
        telemetry.count("stored_block_hash_count", 3)
        telemetry.count("removed_block_hash_count", 2)

        reporter = metrics.MetricsReporter()
        await reporter.start()
        reporter.submit(telemetry)
        await asyncio.sleep(0.05)
        await reporter.stop()

        assert gauge.call_args_list == [
            mocker.call(
                "kv_stage_decode_ms",
                pytest.approx(2.0),
                tags={"pipeline": "incremental"},
            ),
            mocker.call(
                "kv_stage_queue_wait_ms",
                pytest.approx(10.0),
                tags={"pipeline": "incremental"},
            ),
            mocker.call(
                "kv_stage_kvcm_send_ms",
                pytest.approx(30.0),
                tags={"pipeline": "incremental"},
            ),
            mocker.call(
                "kv_event_e2e_latency_ms",
                pytest.approx(42.0),
                tags={"pipeline": "incremental"},
            ),
        ]
        assert counter.call_args_list == [
            mocker.call("stored_block_hash_count", 3, tags={"pipeline": "incremental"}),
            mocker.call(
                "removed_block_hash_count", 2, tags={"pipeline": "incremental"}
            ),
        ]

    async def test_slow_success_emits_warning(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        warning_logged = asyncio.Event()
        warning = mocker.patch(
            "subscriber.metrics.reporter.logger.warning",
            side_effect=lambda *args, **kwargs: warning_logged.set(),
        )

        ticks = iter([0.0, 0.01, 0.015, 0.065])
        telemetry = _telemetry(metrics, pipeline="snapshot", clock=lambda: next(ticks))
        telemetry.mark("queue_wait")
        telemetry.mark("gate_wait")
        telemetry.mark("kvcm_send")

        reporter = metrics.MetricsReporter()
        await reporter.start()
        reporter.submit(telemetry)

        try:
            await asyncio.wait_for(warning_logged.wait(), timeout=1)
        finally:
            await reporter.stop()

        warning.assert_called_once_with(
            "kv event forwarding latency exceeded threshold",
            step="kv_event_metrics",
            tags={
                "pipeline": "snapshot",
                "queue_wait_ms": 10.0,
                "gate_wait_ms": 5.0,
                "kvcm_send_ms": 50.0,
                "total_ms": 65.0,
                "threshold_ms": 50.0,
            },
        )

    async def test_fast_success_skips_warning(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        processed = asyncio.Event()
        warning = mocker.patch("subscriber.metrics.reporter.logger.warning")
        mocker.patch(
            "subscriber.metrics.reporter._dashlog_gauge",
            side_effect=lambda *args, **kwargs: processed.set(),
        )

        ticks = iter([0.0, 0.001])
        telemetry = _telemetry(metrics, clock=lambda: next(ticks))
        telemetry.mark("kvcm_send")

        reporter = metrics.MetricsReporter()
        await reporter.start()
        reporter.submit(telemetry)

        try:
            await asyncio.wait_for(processed.wait(), timeout=1)
        finally:
            await reporter.stop()

        warning.assert_not_called()

    async def test_empty_spans_do_not_emit_e2e_latency(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        gauge = mocker.patch("subscriber.metrics.reporter._dashlog_gauge")

        telemetry = _telemetry(metrics)  # no marks
        reporter = metrics.MetricsReporter()
        await reporter.start()
        reporter.submit(telemetry)
        await asyncio.sleep(0.05)
        await reporter.stop()

        gauge.assert_not_called()

    async def test_flushes_gauges_with_pipeline_tag(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        gauge = mocker.patch("subscriber.metrics.reporter._dashlog_gauge")

        telemetry = _telemetry(metrics, pipeline="snapshot")
        telemetry.gauge("full_snapshot_fetch_latency_ms", 12.5)
        telemetry.gauge("full_snapshot_payload_bytes", 2048)

        reporter = metrics.MetricsReporter()
        await reporter.start()
        reporter.submit(telemetry)
        await asyncio.sleep(0.05)
        await reporter.stop()

        # Gauges are emitted with the pipeline tag, alongside any stage span
        # gauges (there are none here since the telemetry has no marks).
        assert (
            mocker.call(
                "full_snapshot_fetch_latency_ms",
                12.5,
                tags={"pipeline": "snapshot"},
            )
            in gauge.call_args_list
        )
        assert (
            mocker.call(
                "full_snapshot_payload_bytes",
                2048,
                tags={"pipeline": "snapshot"},
            )
            in gauge.call_args_list
        )

    async def test_flushes_tagged_request_observations_asynchronously(
        self, mocker
    ) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        flushed = asyncio.Event()
        emitted: list[object] = []

        def capture(*args: object, **kwargs: object) -> None:
            emitted.append((args, kwargs))
            if len(emitted) == 2:
                flushed.set()

        counter = mocker.patch(
            "subscriber.metrics.reporter._dashlog_counter", side_effect=capture
        )
        gauge = mocker.patch(
            "subscriber.metrics.reporter._dashlog_gauge", side_effect=capture
        )
        telemetry = _telemetry(metrics, pipeline="incremental")
        telemetry.count(
            "kvcm_report_event_request_count",
            1,
            tags={"status_code": "OK"},
        )
        telemetry.gauge(
            "kvcm_report_event_call_ms",
            12.5,
            tags={"status_code": "OK"},
        )

        reporter = metrics.MetricsReporter()
        await reporter.start()
        reporter.submit(telemetry)

        counter.assert_not_called()
        gauge.assert_not_called()
        try:
            await asyncio.wait_for(flushed.wait(), timeout=1)
        finally:
            await reporter.stop()

        assert (
            mocker.call(
                "kvcm_report_event_request_count",
                1,
                tags={"pipeline": "incremental", "status_code": "OK"},
            )
            in counter.call_args_list
        )
        assert (
            mocker.call(
                "kvcm_report_event_call_ms",
                12.5,
                tags={"pipeline": "incremental", "status_code": "OK"},
            )
            in gauge.call_args_list
        )

    async def test_flushes_same_metric_with_distinct_tags_separately(
        self, mocker
    ) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        telemetry = _telemetry(metrics, pipeline="incremental")
        counter = mocker.patch("subscriber.metrics.reporter._dashlog_counter")
        gauge = mocker.patch("subscriber.metrics.reporter._dashlog_gauge")

        telemetry.count("kvcm_report_event_request_count", 1, tags={"status": "OK"})
        telemetry.count(
            "kvcm_report_event_request_count", 1, tags={"status": "REJECTED"}
        )
        telemetry.gauge("kvcm_report_event_call_ms", 1.5, tags={"status": "OK"})
        telemetry.gauge("kvcm_report_event_call_ms", 2.5, tags={"status": "REJECTED"})

        reporter = metrics.MetricsReporter()
        await reporter.start()
        try:
            reporter.submit(telemetry)
            await asyncio.sleep(0.05)
        finally:
            await reporter.stop()

        assert counter.call_args_list == [
            mocker.call(
                "kvcm_report_event_request_count",
                1,
                tags={"pipeline": "incremental", "status": "OK"},
            ),
            mocker.call(
                "kvcm_report_event_request_count",
                1,
                tags={"pipeline": "incremental", "status": "REJECTED"},
            ),
        ]
        assert gauge.call_args_list == [
            mocker.call(
                "kvcm_report_event_call_ms",
                1.5,
                tags={"pipeline": "incremental", "status": "OK"},
            ),
            mocker.call(
                "kvcm_report_event_call_ms",
                2.5,
                tags={"pipeline": "incremental", "status": "REJECTED"},
            ),
        ]


class TestMetricsReporterDrop:
    async def test_drop_emits_drop_counter_and_preserved_spans(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        gauge = mocker.patch("subscriber.metrics.reporter._dashlog_gauge")
        counter = mocker.patch("subscriber.metrics.reporter._dashlog_counter")

        ticks = iter([0.0, 0.01, 0.015])
        telemetry = _telemetry(
            metrics, pipeline="incremental", clock=lambda: next(ticks)
        )
        telemetry.mark("queue_wait")
        telemetry.mark("engine_gate_wait")
        telemetry.mark_dropped("epoch_changed")

        reporter = metrics.MetricsReporter()
        await reporter.start()
        reporter.submit(telemetry)
        await asyncio.sleep(0.05)
        await reporter.stop()

        # Drop counter is emitted with pipeline + reason tags.
        assert (
            mocker.call(
                "kv_batch_drop_count",
                1,
                tags={"pipeline": "incremental", "reason": "epoch_changed"},
            )
            in counter.call_args_list
        )
        # Stage histograms up to the drop point are still emitted, but tagged
        # with the drop reason so dashboards can distinguish them from
        # successful batches.
        assert gauge.call_args_list == [
            mocker.call(
                "kv_stage_queue_wait_ms",
                pytest.approx(10.0),
                tags={
                    "pipeline": "incremental",
                    "drop_reason": "epoch_changed",
                },
            ),
            mocker.call(
                "kv_stage_engine_gate_wait_ms",
                pytest.approx(5.0),
                tags={
                    "pipeline": "incremental",
                    "drop_reason": "epoch_changed",
                },
            ),
        ]

    async def test_drop_emits_gauges_with_drop_reason_tag(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        gauge = mocker.patch("subscriber.metrics.reporter._dashlog_gauge")

        telemetry = _telemetry(metrics, pipeline="snapshot")
        telemetry.gauge("full_snapshot_fetch_latency_ms", 12.5)
        telemetry.mark_dropped("fetch_failed")

        reporter = metrics.MetricsReporter()
        await reporter.start()
        reporter.submit(telemetry)
        await asyncio.sleep(0.05)
        await reporter.stop()

        # On drop, gauges carry both pipeline and drop_reason tags so
        # dashboards can distinguish partial from successful measurements.
        assert (
            mocker.call(
                "full_snapshot_fetch_latency_ms",
                12.5,
                tags={"pipeline": "snapshot", "drop_reason": "fetch_failed"},
            )
            in gauge.call_args_list
        )

    async def test_drop_does_not_emit_slow_warning(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        warning = mocker.patch("subscriber.metrics.reporter.logger.warning")
        processed = asyncio.Event()
        mocker.patch(
            "subscriber.metrics.reporter._dashlog_counter",
            side_effect=lambda *args, **kwargs: processed.set(),
        )

        # A long-running batch that ultimately dropped: total span 200 ms.
        ticks = iter([0.0, 0.1, 0.2])
        telemetry = _telemetry(metrics, clock=lambda: next(ticks))
        telemetry.mark("queue_wait")
        telemetry.mark("engine_gate_wait")
        telemetry.mark_dropped("send_failed")

        reporter = metrics.MetricsReporter()
        await reporter.start()
        reporter.submit(telemetry)

        try:
            await asyncio.wait_for(processed.wait(), timeout=1)
        finally:
            await reporter.stop()

        warning.assert_not_called()


class TestMetricsReporterLifecycle:
    async def test_start_is_idempotent(self) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        reporter = metrics.MetricsReporter()

        await reporter.start()
        first_task = reporter._task
        await reporter.start()

        try:
            assert reporter._task is first_task
        finally:
            await reporter.stop()

    async def test_uses_configured_task_name(self) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        reporter = metrics.MetricsReporter(task_name="snapshot-kv-event-metrics")

        await reporter.start()
        try:
            assert reporter._task is not None
            assert reporter._task.get_name() == "snapshot-kv-event-metrics"
        finally:
            await reporter.stop()

    def test_submit_without_start_does_not_raise(self) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        reporter = metrics.MetricsReporter()
        telemetry = _telemetry(metrics)
        telemetry.mark("kvcm_send")

        # Must not raise even though the flush task never started.
        reporter.submit(telemetry)

    def test_submit_swallows_queue_errors(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        reporter = metrics.MetricsReporter()
        reporter._queue = mocker.Mock()
        reporter._queue.put_nowait.side_effect = RuntimeError("queue failure")

        telemetry = _telemetry(metrics)
        telemetry.mark("kvcm_send")

        reporter.submit(telemetry)  # must not raise

    async def test_dashlog_failures_are_swallowed(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        mocker.patch(
            "subscriber.metrics.reporter._dashlog_gauge",
            side_effect=RuntimeError("dashlog unavailable"),
        )
        telemetry = _telemetry(metrics)
        telemetry.mark("kvcm_send")

        reporter = metrics.MetricsReporter()
        await reporter.start()
        reporter.submit(telemetry)
        await asyncio.sleep(0.05)
        await reporter.stop()  # must not raise
