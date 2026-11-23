"""Tests for lifecycle metric helpers (synchronous dashlog emitters).

Main-path telemetry (``BatchTelemetry`` + ``MetricsReporter``) is covered by
``tests/metrics/test_reporter.py``. This file only exercises the sparse,
point-event ``report_xxx()`` helpers that live in
``subscriber.metrics.lifecycle`` and route straight to dashlog.
"""

from __future__ import annotations

import importlib


class TestInitDashlog:
    def test_calls_dashlog_init(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        mocker.patch("subscriber.metrics._base._dashlog", mocker.MagicMock())
        init = mocker.patch("subscriber.metrics._base._dashlog_init")

        metrics.init_dashlog("kvcache-subscriber")

        init.assert_called_once_with("kvcache-subscriber")

    def test_swallows_errors(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        mocker.patch("subscriber.metrics._base._dashlog", mocker.MagicMock())
        mocker.patch(
            "subscriber.metrics._base._dashlog_init",
            side_effect=RuntimeError("init failed"),
        )

        metrics.init_dashlog("kvcache-subscriber")

    def test_warns_when_dashlog_unavailable(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        mocker.patch("subscriber.metrics._base._dashlog", None)
        init = mocker.patch("subscriber.metrics._base._dashlog_init")
        warning = mocker.patch("subscriber.logger.warning")

        metrics.init_dashlog("kvcache-subscriber")

        init.assert_not_called()
        warning.assert_called_once_with(
            "dashlog is unavailable; metrics reporting is disabled",
            step="metrics_init",
            tags={"app_name": "kvcache-subscriber"},
        )


class TestKvcmHeartbeat:
    def test_success_increments_counter_with_tag(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        counter = mocker.patch("subscriber.metrics.lifecycle._dashlog_counter")

        metrics.report_heartbeat(success=True)

        counter.assert_called_once_with(
            "kvcm_heartbeat_count", 1, tags={"status": "success"}
        )

    def test_failure_increments_counter_with_tag(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        counter = mocker.patch("subscriber.metrics.lifecycle._dashlog_counter")

        metrics.report_heartbeat(success=False)

        counter.assert_called_once_with(
            "kvcm_heartbeat_count", 1, tags={"status": "failure"}
        )

    def test_swallows_errors(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        mocker.patch(
            "subscriber.metrics.lifecycle._dashlog_counter",
            side_effect=RuntimeError("fail"),
        )

        metrics.report_heartbeat(success=True)


class TestKvcmRegistration:
    def test_recovery_increments_counter(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        counter = mocker.patch("subscriber.metrics.lifecycle._dashlog_counter")

        metrics.report_registration_recovery()

        counter.assert_called_once_with("kvcm_registration_recovery_count", 1)

    def test_transition_is_tagged(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        counter = mocker.patch("subscriber.metrics.lifecycle._dashlog_counter")

        metrics.report_registration_transition("unregistered", "registered")

        counter.assert_called_once_with(
            "kvcm_registration_transition_count",
            1,
            tags={"from": "unregistered", "to": "registered"},
        )


class TestHealthMetrics:
    def test_engine_probe_reports_count_and_latency(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        counter = mocker.patch("subscriber.metrics.lifecycle._dashlog_counter")
        gauge = mocker.patch("subscriber.metrics.lifecycle._dashlog_gauge")

        metrics.report_engine_probe("alive", 12.5)

        counter.assert_called_once_with(
            "engine_health_probe_count", 1, tags={"result": "alive"}
        )
        gauge.assert_called_once_with(
            "engine_health_probe_latency_ms", 12.5, tags={"result": "alive"}
        )

    def test_engine_and_subscriber_transitions_are_tagged(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        counter = mocker.patch("subscriber.metrics.lifecycle._dashlog_counter")

        metrics.report_engine_state_transition("starting", "healthy")
        metrics.report_subscriber_phase_transition("starting", "active")

        assert counter.call_args_list == [
            mocker.call(
                "engine_health_state_transition_count",
                1,
                tags={"from": "starting", "to": "healthy"},
            ),
            mocker.call(
                "subscriber_phase_transition_count",
                1,
                tags={"from": "starting", "to": "active"},
            ),
        ]

    def test_dashserving_heartbeat_and_state_report_are_observable(
        self, mocker
    ) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        counter = mocker.patch("subscriber.metrics.lifecycle._dashlog_counter")
        gauge = mocker.patch("subscriber.metrics.lifecycle._dashlog_gauge")

        metrics.report_dashserving_heartbeat(success=False)
        metrics.report_dashserving_state_report("active", "error", 7.5)

        assert counter.call_args_list == [
            mocker.call("dashserving_heartbeat_count", 1, tags={"status": "failure"}),
            mocker.call(
                "dashserving_state_report_count",
                1,
                tags={"state": "active", "result": "error"},
            ),
        ]
        gauge.assert_called_once_with(
            "dashserving_state_report_latency_ms",
            7.5,
            tags={"state": "active", "result": "error"},
        )

    def test_shutdown_is_tagged(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        counter = mocker.patch("subscriber.metrics.lifecycle._dashlog_counter")

        metrics.report_shutdown("host_down_sent:shutdown")

        counter.assert_called_once_with(
            "subscriber_shutdown_count",
            1,
            tags={"outcome": "host_down_sent:shutdown"},
        )

    def test_helpers_swallow_dashlog_errors(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        mocker.patch(
            "subscriber.metrics.lifecycle._dashlog_counter",
            side_effect=RuntimeError("dashlog unavailable"),
        )

        metrics.report_engine_probe("rpc_error", 1.0)
        metrics.report_dashserving_state_report("failed", "error", 2.0)
        metrics.report_registration_transition("registered", "unregistered")


class TestZmqMessageCounter:
    def test_increments_received_counter(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        counter = mocker.patch("subscriber.metrics.lifecycle._dashlog_counter")

        metrics.report_zmq_message()

        counter.assert_called_once_with("zmq_received_message_count", 1)

    def test_swallows_errors(self, mocker) -> None:
        metrics = importlib.import_module("subscriber.metrics")
        mocker.patch(
            "subscriber.metrics.lifecycle._dashlog_counter",
            side_effect=RuntimeError("dashlog unavailable"),
        )

        metrics.report_zmq_message()
