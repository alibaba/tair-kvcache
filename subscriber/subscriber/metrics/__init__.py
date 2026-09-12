"""Subscriber metrics package.

This package exposes two clearly separated surfaces:

- **Main-path metrics**: per-batch telemetry produced by the forwarding and
  snapshot pipelines. Callers construct a :class:`BatchTelemetry` per event,
  ``mark`` / ``count`` / ``mark_dropped`` on it during the pipeline, and
  finally submit it to the pipeline's :class:`MetricsReporter` for
  asynchronous flush to dashlog. This is the single async output for all
  hot-path metric emission.

- **Lifecycle metrics**: sparse, point-event counters and histograms that
  live outside the main path (engine health probes, KVCM heartbeat, ZMQ
  message throughput, subscriber shutdown, ...). Each lifecycle helper is a
  synchronous, best-effort ``report_xxx()`` function that writes to dashlog
  directly.

The ``_base`` module provides the underlying dashlog wrappers and the
``init_dashlog`` bootstrap; callers should prefer the higher-level helpers
above over reaching for the ``_dashlog_*`` primitives directly.
"""

from subscriber.metrics._base import (
    _dashlog_counter,
    _dashlog_gauge,
    init_dashlog,
)
from subscriber.metrics.lifecycle import (
    report_dashserving_heartbeat,
    report_dashserving_state_report,
    report_engine_probe,
    report_engine_state_transition,
    report_heartbeat,
    report_registration_recovery,
    report_registration_transition,
    report_shutdown,
    report_subscriber_phase_transition,
    report_zmq_message,
)
from subscriber.metrics.reporter import BatchTelemetry, MetricsReporter, Span

__all__ = [
    "BatchTelemetry",
    "MetricsReporter",
    "Span",
    "_dashlog_counter",
    "_dashlog_gauge",
    "init_dashlog",
    "report_dashserving_heartbeat",
    "report_dashserving_state_report",
    "report_engine_probe",
    "report_engine_state_transition",
    "report_heartbeat",
    "report_registration_recovery",
    "report_registration_transition",
    "report_shutdown",
    "report_subscriber_phase_transition",
    "report_zmq_message",
]
