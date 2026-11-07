"""Canonical-truth inspection: every metric name the code can emit must
appear in ``docs/metrics.json``.

The test exercises both metric emission surfaces:

- **Lifecycle helpers** — each ``report_xxx()`` function in
  ``subscriber.metrics.lifecycle``.
- **Main-path reporter** — ``MetricsReporter`` flush for both the success
  and drop paths, covering dynamically constructed stage names
  (``kv_stage_{span}_ms``), e2e latency, aggregate counters, and gauges.

Every stage name, counter name, and gauge name used in production code is
represented.  The test collects the full prefixed metric names and asserts
membership in the ``docs/metrics.json`` catalog.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# All stage / counter / gauge names used in production code.
#
# Stage names: every ``telemetry.mark("<name>")`` call site in
#   subscriber/engine/**, subscriber/forwarding.py, subscriber/kvcm/client.py.
# Counter names: every ``telemetry.count("<name>", …)`` call site.
# Gauge names: every ``telemetry.gauge("<name>", …)`` call site.
# ---------------------------------------------------------------------------

_PRODUCTION_STAGE_NAMES: list[str] = [
    "decode",  # incremental adapter, snapshot adapter
    "replay_fetch",  # incremental adapter (replay path)
    "snapshot_fetch",  # snapshot adapter (gRPC fetch)
    "snapshot_build",  # snapshot adapter (build batch)
    "queue_wait",  # forwarding (dequeue)
    "engine_gate_wait",  # forwarding (health gate)
    "block_filter",  # forwarding (incremental filter)
    "expand",  # kvcm client (payload serialization)
    "kvcm_send",  # kvcm client (KVCM ReportEvent call)
]

_PRODUCTION_COUNTER_NAMES: list[str] = [
    "stored_block_hash_count",  # forwarding._submit_incremental_success
    "removed_block_hash_count",  # forwarding._submit_incremental_success
    "snapshot_block_count",  # forwarding._submit_snapshot_success
    "zmq_sequence_gap_count",  # incremental adapter (replay)
    "zmq_missed_message_count",  # incremental adapter (replay)
]

_PRODUCTION_TAGGED_COUNTERS: list[tuple[str, dict[str, str]]] = [
    ("kvcm_report_event_request_count", {"status_code": "OK"}),
    (
        "kvcm_report_event_failure_count",
        {"status_code": "GRPC_UNAVAILABLE", "reason": "transport"},
    ),
    ("kvcm_report_event_retry_count", {"reason": "SERVER_NOT_LEADER"}),
]

_PRODUCTION_GAUGE_NAMES: list[str] = [
    "full_snapshot_payload_bytes",  # snapshot adapter
    "kvcm_merged_queue_item_count",  # forwarding incremental merge diagnostics
    "kvcm_source_batch_count",  # forwarding incremental merge diagnostics
    "kvcm_source_event_count",  # forwarding incremental merge diagnostics
    "kvcm_merged_report_event_count",  # forwarding incremental merge diagnostics
    "kvcm_queue_size_before_merge",  # forwarding incremental merge diagnostics
    "kvcm_queue_size_after_merge",  # forwarding incremental merge diagnostics
    "kvcm_oldest_enqueue_to_send_ms",  # forwarding incremental merge diagnostics
    "kvcm_newest_enqueue_to_send_ms",  # forwarding incremental merge diagnostics
    "kvcm_report_event_count",  # KvcmClient incremental report request
]

_PRODUCTION_TAGGED_GAUGES: list[tuple[str, dict[str, str]]] = [
    ("kvcm_report_event_call_ms", {"status_code": "OK"}),
    ("kvcm_report_event_request_bytes", {"status_code": "OK"}),
    ("kvcm_report_event_wire_encode_ms", {"status_code": "OK"}),
    ("kvcm_report_event_grpc_call_ms", {"status_code": "OK"}),
    ("kvcm_snapshot_source_block_count", {"status_code": "OK"}),
    ("kvcm_snapshot_merged_block_count", {"status_code": "OK"}),
]

_METRICS_JSON_PATH = Path(__file__).resolve().parents[2] / "docs" / "metrics.json"


def _load_metrics_catalog() -> set[str]:
    """Return the set of fully-qualified metric names from docs/metrics.json."""
    data = json.loads(_METRICS_JSON_PATH.read_text())
    return {entry["name"] for entry in data["metrics"]}


# ---------------------------------------------------------------------------
# Lifecycle emission
# ---------------------------------------------------------------------------


def _exercise_lifecycle(collector: set[str], mocker: pytest.MockFixture) -> None:
    """Call every lifecycle helper, capturing emitted metric names."""
    metrics = importlib.import_module("subscriber.metrics")

    def _capture(name: str, *args: object, **kwargs: object) -> None:
        collector.add(name)

    mocker.patch("subscriber.metrics.lifecycle._dashlog_counter", side_effect=_capture)
    mocker.patch("subscriber.metrics.lifecycle._dashlog_gauge", side_effect=_capture)

    metrics.report_engine_probe("alive", 1.0)
    metrics.report_engine_probe("dead", 2.0)
    metrics.report_engine_probe("timeout", 3.0)
    metrics.report_engine_probe("rpc_error", 4.0)

    metrics.report_engine_state_transition("starting", "healthy")
    metrics.report_engine_state_transition("healthy", "dead")

    metrics.report_subscriber_phase_transition("starting", "active")
    metrics.report_subscriber_phase_transition("active", "inactive")

    metrics.report_dashserving_heartbeat(success=True)
    metrics.report_dashserving_heartbeat(success=False)

    metrics.report_dashserving_state_report("active", "accepted", 5.0)
    metrics.report_dashserving_state_report("failed", "error", 1.0)

    metrics.report_shutdown("host_down_sent:shutdown")
    metrics.report_shutdown("host_down_failed:timeout")

    metrics.report_heartbeat(success=True)
    metrics.report_heartbeat(success=False)

    metrics.report_registration_recovery()
    metrics.report_registration_transition("unregistered", "registered")

    metrics.report_zmq_message()


# ---------------------------------------------------------------------------
# Reporter emission
# ---------------------------------------------------------------------------


async def _exercise_reporter(collector: set[str], mocker: pytest.MockFixture) -> None:
    """Exercise MetricsReporter success + drop paths with all production names."""
    metrics = importlib.import_module("subscriber.metrics")

    def _capture(name: str, *args: object, **kwargs: object) -> None:
        collector.add(name)

    mocker.patch("subscriber.metrics.reporter._dashlog_counter", side_effect=_capture)
    mocker.patch("subscriber.metrics.reporter._dashlog_gauge", side_effect=_capture)

    # --- Success path: all stages, all counters, all gauges ---
    ticks = iter([0.0] + [float(i) for i in range(1, len(_PRODUCTION_STAGE_NAMES) + 1)])
    telemetry = metrics.BatchTelemetry(
        pipeline="incremental", clock=lambda: next(ticks)
    )
    for stage in _PRODUCTION_STAGE_NAMES:
        telemetry.mark(stage)
    for name in _PRODUCTION_COUNTER_NAMES:
        telemetry.count(name, 1)
    for name in _PRODUCTION_GAUGE_NAMES:
        telemetry.gauge(name, 100.0)
    for name, tags in _PRODUCTION_TAGGED_COUNTERS:
        telemetry.count(name, 1, tags=tags)
    for name, tags in _PRODUCTION_TAGGED_GAUGES:
        telemetry.gauge(name, 100.0, tags=tags)

    reporter = metrics.MetricsReporter()
    await reporter.start()
    reporter.submit(telemetry)
    await asyncio.sleep(0.05)

    # --- Drop path: partial stages then drop ---
    ticks2 = iter([0.0, 0.1, 0.2])
    drop_telemetry = metrics.BatchTelemetry(
        pipeline="snapshot", clock=lambda: next(ticks2)
    )
    drop_telemetry.mark("snapshot_fetch")
    drop_telemetry.mark("decode")
    drop_telemetry.gauge("full_snapshot_payload_bytes", 512)
    drop_telemetry.mark_dropped("fetch_failed")
    reporter.submit(drop_telemetry)
    await asyncio.sleep(0.05)

    await reporter.stop()


# ---------------------------------------------------------------------------
# The canonical-truth test
# ---------------------------------------------------------------------------


class TestMetricsCatalog:
    """docs/metrics.json and emitted metric names must match in both directions."""

    async def test_all_emitted_metrics_are_cataloged(
        self, mocker: pytest.MockFixture
    ) -> None:
        catalog = _load_metrics_catalog()
        prefix = "kvcache_subscriber_"
        raw_names: set[str] = set()

        _exercise_lifecycle(raw_names, mocker)
        await _exercise_reporter(raw_names, mocker)

        # Build the full qualified names the way _base.py does:
        # _dashlog_counter / _dashlog_gauge prepend _METRIC_PREFIX.
        full_names = {prefix + name for name in raw_names}

        missing = sorted(full_names - catalog)
        assert not missing, (
            f"Metric name(s) emitted by code but absent from "
            f"docs/metrics.json: {missing}"
        )

    async def test_no_stale_catalog_entries(self, mocker: pytest.MockFixture) -> None:
        """Reverse drift: every cataloged metric must still be emitted by code."""
        catalog = _load_metrics_catalog()
        prefix = "kvcache_subscriber_"
        raw_names: set[str] = set()

        _exercise_lifecycle(raw_names, mocker)
        await _exercise_reporter(raw_names, mocker)

        full_names = {prefix + name for name in raw_names}

        stale = sorted(catalog - full_names)
        assert not stale, (
            f"Metric name(s) cataloged in docs/metrics.json but no longer "
            f"emitted by code (remove them or exercise them here): {stale}"
        )
