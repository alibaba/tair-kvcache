"""Lifecycle metric helpers — synchronous, low-frequency, point-event counters.

These functions cover metrics that are not part of the main forwarding path:

- Engine and subscriber health transitions.
- KVCM registration and heartbeat.
- DashServing heartbeats and state reports.
- Individual ZMQ message receipts (background throughput counter).
- Subscriber shutdown outcomes.

Per-batch main-path metrics (forwarding stages, snapshot fetch/build,
drop counts, etc.) go through :class:`~subscriber.metrics.MetricsReporter`
instead — see :mod:`subscriber.metrics.reporter`.

Each helper wraps the underlying dashlog call in ``try/except`` so metric
reporting never blocks or fails the caller. Every metric preserves the
``kvcache_subscriber_`` prefix and its tags exactly as before the refactor.
"""

from __future__ import annotations

from subscriber.metrics._base import _dashlog_counter, _dashlog_gauge

# ---------------------------------------------------------------------------
# Engine and subscriber health
# ---------------------------------------------------------------------------


def report_engine_probe(result: str, latency_ms: float) -> None:
    try:
        _dashlog_counter("engine_health_probe_count", 1, tags={"result": result})
        _dashlog_gauge(
            "engine_health_probe_latency_ms", latency_ms, tags={"result": result}
        )
    except Exception:
        pass


def report_engine_state_transition(from_state: str, to_state: str) -> None:
    try:
        _dashlog_counter(
            "engine_health_state_transition_count",
            1,
            tags={"from": from_state, "to": to_state},
        )
    except Exception:
        pass


def report_subscriber_phase_transition(from_phase: str, to_phase: str) -> None:
    try:
        _dashlog_counter(
            "subscriber_phase_transition_count",
            1,
            tags={"from": from_phase, "to": to_phase},
        )
    except Exception:
        pass


def report_dashserving_heartbeat(success: bool) -> None:
    try:
        _dashlog_counter(
            "dashserving_heartbeat_count",
            1,
            tags={"status": "success" if success else "failure"},
        )
    except Exception:
        pass


def report_dashserving_state_report(
    state: str,
    result: str,
    latency_ms: float,
) -> None:
    try:
        tags = {"state": state, "result": result}
        _dashlog_counter("dashserving_state_report_count", 1, tags=tags)
        _dashlog_gauge("dashserving_state_report_latency_ms", latency_ms, tags=tags)
    except Exception:
        pass


def report_shutdown(outcome: str) -> None:
    try:
        _dashlog_counter("subscriber_shutdown_count", 1, tags={"outcome": outcome})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# KVCM registration and heartbeat
# ---------------------------------------------------------------------------


def report_heartbeat(success: bool) -> None:
    try:
        _dashlog_counter(
            "kvcm_heartbeat_count",
            1,
            tags={"status": "success" if success else "failure"},
        )
    except Exception:
        pass


def report_registration_recovery() -> None:
    try:
        _dashlog_counter("kvcm_registration_recovery_count", 1)
    except Exception:
        pass


def report_registration_transition(from_state: str, to_state: str) -> None:
    try:
        _dashlog_counter(
            "kvcm_registration_transition_count",
            1,
            tags={"from": from_state, "to": to_state},
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ZMQ background throughput
# ---------------------------------------------------------------------------


def report_zmq_message() -> None:
    """Increment the background ZMQ message counter.

    Emitted once per received ZMQ message. This is a lifecycle-style
    high-frequency counter that reflects background throughput, deliberately
    kept separate from per-batch main-path telemetry (see :class:`BatchTelemetry`).
    """

    try:
        _dashlog_counter("zmq_received_message_count", 1)
    except Exception:
        pass
