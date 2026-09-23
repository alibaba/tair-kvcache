from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from subscriber.config import SubscriberConfig, build_parser
from subscriber.kvcm.enum import KvcmStorageType

_KVCM_ARGS = [
    "--kvcm-base-url",
    "http://kvcm-test:8080",
    "--host-port",
    "8080",
]


def test_defaults() -> None:
    config = SubscriberConfig()
    assert config.engine_type == "vllm"
    assert (
        config.engine_kv_event_control_uds_path == "/tmp/dashllm-kv-event-control.sock"
    )
    assert config.engine_kv_event_bootstrap_timeout_ms == 1000.0
    assert config.engine_kv_event_bootstrap_max_retries == 5
    assert config.kvcm_protocol == "grpc"


@pytest.mark.parametrize(
    "environment",
    [
        {"DS_LLM_PROC_ENGINE_RANK": "3", "DS_LLM_PROC_RANK": "7"},
        {"DS_LLM_PROC_RANK": "7"},
    ],
)
def test_control_uds_default_ignores_dashllm_process_ranks(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
) -> None:
    monkeypatch.delenv("DS_LLM_PROC_ENGINE_RANK", raising=False)
    monkeypatch.delenv("DS_LLM_PROC_RANK", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    config = SubscriberConfig()

    assert (
        config.engine_kv_event_control_uds_path == "/tmp/dashllm-kv-event-control.sock"
    )


def test_incremental_kv_event_pipeline_is_enabled_by_default() -> None:
    config = SubscriberConfig()

    assert config.incremental_kv_event_pipeline_enabled is True


def test_incremental_kv_event_pipeline_cli_disable() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [*_KVCM_ARGS, "--no-incremental-kv-event-pipeline-enabled"]
    )

    config = SubscriberConfig.from_args(args)

    assert config.incremental_kv_event_pipeline_enabled is False


def test_incremental_kv_event_pipeline_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("incremental_kv_event_pipeline_enabled: false\n")
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--config", str(yaml_file)])

    config = SubscriberConfig.from_args(args)

    assert config.incremental_kv_event_pipeline_enabled is False


def test_snapshot_kv_event_pipeline_is_disabled_by_default() -> None:
    config = SubscriberConfig()

    assert config.snapshot_kv_event_pipeline_enabled is False


def test_snapshot_kv_event_pipeline_cli_disable() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--no-snapshot-kv-event-pipeline-enabled"])

    config = SubscriberConfig.from_args(args)

    assert config.snapshot_kv_event_pipeline_enabled is False


def test_snapshot_kv_event_pipeline_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("snapshot_kv_event_pipeline_enabled: false\n")
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--config", str(yaml_file)])

    config = SubscriberConfig.from_args(args)

    assert config.snapshot_kv_event_pipeline_enabled is False


def test_engine_type_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--engine-type", "sglang"])
    config = SubscriberConfig.from_args(args)
    assert config.engine_type == "sglang"


def test_extra_attention_types_cli_overrides_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("extra_attention_types:\n  custom_attention: yaml-prefix\n")
    parser = build_parser()
    args = parser.parse_args(
        [
            *_KVCM_ARGS,
            "--config",
            str(yaml_file),
            "--extra-attention-types",
            '{"custom_attention":"cli-prefix"}',
        ]
    )

    config = SubscriberConfig.from_args(args)

    assert config.extra_attention_types == {"custom_attention": "cli-prefix"}


def test_extra_attention_types_cannot_override_builtin_mapping() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            *_KVCM_ARGS,
            "--extra-attention-types",
            '{"full_attention":"custom-prefix"}',
        ]
    )

    with pytest.raises(ValueError, match="cannot override built-in"):
        SubscriberConfig.from_args(args)


def test_storage_type_defaults_to_event_report() -> None:
    config = SubscriberConfig()

    assert config.kvcm_storage_type == KvcmStorageType.ST_EVENT_REPORT_L1P5


def test_removed_storage_type_is_rejected() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--kvcm-storage-type", "ST_EVENT_REPORT"])

    with pytest.raises(ValueError, match="kvcm_storage_type"):
        SubscriberConfig.from_args(args)


def test_protobuf_storage_type_cli_override_is_accepted() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--kvcm-storage-type", "ST_NFS"])

    config = SubscriberConfig.from_args(args)

    assert config.kvcm_storage_type == KvcmStorageType.ST_NFS


def test_unspecified_storage_type_is_rejected() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--kvcm-storage-type", "ST_UNSPECIFIED"])

    with pytest.raises(ValueError, match="must not be ST_UNSPECIFIED"):
        SubscriberConfig.from_args(args)


def test_blank_kvcm_base_url_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KVS_KVCM_ENDPOINT", raising=False)
    parser = build_parser()
    args = parser.parse_args(["--host-port", "8080"])
    with pytest.raises(ValueError, match="kvcm_base_url is required"):
        SubscriberConfig.from_args(args)


def test_kvcm_base_url_falls_back_to_kvs_kvcm_endpoint_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVS_KVCM_ENDPOINT", "spectrum://v-46354ead:6382")
    parser = build_parser()
    args = parser.parse_args(["--host-port", "8080"])
    config = SubscriberConfig.from_args(args)
    assert config.kvcm_base_url == "spectrum://v-46354ead:6382"


def test_kvcm_base_url_cli_takes_precedence_over_kvs_kvcm_endpoint_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVS_KVCM_ENDPOINT", "spectrum://v-46354ead:6382")
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS])
    config = SubscriberConfig.from_args(args)
    assert config.kvcm_base_url == "http://kvcm-test:8080"


def test_kvcm_protocol_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--kvcm-protocol", "http"])
    config = SubscriberConfig.from_args(args)
    assert config.kvcm_protocol == "http"


def test_kvcm_protocol_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("kvcm_protocol: http\n")
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--config", str(yaml_file)])
    config = SubscriberConfig.from_args(args)
    assert config.kvcm_protocol == "http"


def test_invalid_kvcm_protocol_is_rejected() -> None:
    config = SubscriberConfig(
        kvcm_base_url="spectrum://v-46354ead:6381",
        kvcm_protocol="tcp",
        host_port=8080,
    )
    with pytest.raises(ValueError, match="kvcm_protocol must be one of"):
        config.validate()


def test_kv_event_queue_maxsize_defaults_to_4096() -> None:
    config = SubscriberConfig()

    assert config.kv_event_queue_maxsize == 4096


def test_kv_event_queue_maxsize_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--kv-event-queue-maxsize", "7"])

    config = SubscriberConfig.from_args(args)

    assert config.kv_event_queue_maxsize == 7


def test_kv_event_queue_maxsize_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("kv_event_queue_maxsize: 9\n")
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--config", str(yaml_file)])

    config = SubscriberConfig.from_args(args)

    assert config.kv_event_queue_maxsize == 9


def test_kv_event_queue_maxsize_must_be_positive() -> None:
    config = SubscriberConfig(kv_event_queue_maxsize=0, host_port=8080)

    with pytest.raises(ValueError, match="kv_event_queue_maxsize must be >= 1"):
        config.validate()


def test_kv_event_merge_max_report_events_defaults_to_16() -> None:
    config = SubscriberConfig()

    assert config.kv_event_merge_max_report_events == 16


def test_kv_event_merge_max_report_events_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--kv-event-merge-max-report-events", "7"])

    config = SubscriberConfig.from_args(args)

    assert config.kv_event_merge_max_report_events == 7


def test_kv_event_merge_max_report_events_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("kv_event_merge_max_report_events: 9\n")
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--config", str(yaml_file)])

    config = SubscriberConfig.from_args(args)

    assert config.kv_event_merge_max_report_events == 9


def test_kv_event_merge_max_report_events_must_be_positive() -> None:
    config = SubscriberConfig(kv_event_merge_max_report_events=0, host_port=8080)

    with pytest.raises(
        ValueError,
        match="kv_event_merge_max_report_events must be >= 1",
    ):
        config.validate()


def test_kv_event_merge_max_queue_items_defaults_to_4() -> None:
    config = SubscriberConfig()

    assert config.kv_event_merge_max_queue_items == 4


def test_kv_event_merge_max_queue_items_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--kv-event-merge-max-queue-items", "2"])

    config = SubscriberConfig.from_args(args)

    assert config.kv_event_merge_max_queue_items == 2


def test_kv_event_merge_max_queue_items_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("kv_event_merge_max_queue_items: 3\n")
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--config", str(yaml_file)])

    config = SubscriberConfig.from_args(args)

    assert config.kv_event_merge_max_queue_items == 3


def test_kv_event_merge_max_queue_items_must_be_positive() -> None:
    config = SubscriberConfig(kv_event_merge_max_queue_items=0, host_port=8080)

    with pytest.raises(
        ValueError,
        match="kv_event_merge_max_queue_items must be >= 1",
    ):
        config.validate()


def test_kvcm_runtime_config_defaults() -> None:
    config = SubscriberConfig()
    assert config.kvcm_heartbeat_interval_s == 5.0


def test_kvcm_runtime_config_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            *_KVCM_ARGS,
            "--kvcm-heartbeat-interval-s",
            "2.5",
        ]
    )
    config = SubscriberConfig.from_args(args)
    assert config.kvcm_heartbeat_interval_s == 2.5


def test_kvcm_runtime_config_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        textwrap.dedent(
            """\
            kvcm_heartbeat_interval_s: 3.0
            """
        )
    )
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--config", str(yaml_file)])
    config = SubscriberConfig.from_args(args)
    assert config.kvcm_heartbeat_interval_s == 3.0


def test_non_positive_kvcm_heartbeat_interval_raises() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--kvcm-heartbeat-interval-s", "0"])

    with pytest.raises(ValueError, match="^kvcm_heartbeat_interval_s must be > 0$"):
        SubscriberConfig.from_args(args)


def test_host_port_has_no_default() -> None:
    assert SubscriberConfig().host_port is None


def test_host_port_missing_rejected_by_from_args() -> None:
    parser = build_parser()
    args = parser.parse_args(["--kvcm-base-url", "http://kvcm-test:8080"])
    with pytest.raises(ValueError, match="^host_port is required"):
        SubscriberConfig.from_args(args)


def test_host_port_missing_raises_in_validate() -> None:
    config = SubscriberConfig(kvcm_base_url="http://kvcm-test:8080")
    with pytest.raises(ValueError, match="^host_port is required"):
        config.validate()


def test_host_port_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--host-port", "9001"])
    config = SubscriberConfig.from_args(args)
    assert config.host_port == 9001


def test_host_port_out_of_range_raises() -> None:
    parser = build_parser()
    for value in ("0", "65536"):
        args = parser.parse_args([*_KVCM_ARGS, "--host-port", value])
        with pytest.raises(ValueError, match=r"^host_port must be in \[1, 65535\]$"):
            SubscriberConfig.from_args(args)


def test_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        textwrap.dedent("""\
            engine_kv_event_control_uds_path: "/tmp/custom-control.sock"
        """)
    )
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--config", str(yaml_file)])
    config = SubscriberConfig.from_args(args)
    assert config.engine_kv_event_control_uds_path == "/tmp/custom-control.sock"


def test_cli_overrides_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("engine_type: vllm\n")
    parser = build_parser()
    args = parser.parse_args(
        [*_KVCM_ARGS, "--config", str(yaml_file), "--engine-type", "sglang"]
    )
    config = SubscriberConfig.from_args(args)
    assert config.engine_type == "sglang"


def test_health_config_defaults() -> None:
    config = SubscriberConfig()
    assert config.engine_health_interval_s == 5.0
    assert config.engine_kvcache_worker_status_timeout_ms == 1000.0
    assert config.engine_health_failure_threshold == 3


def test_health_config_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            *_KVCM_ARGS,
            "--engine-health-interval-s",
            "0.25",
            "--engine-kvcache-worker-status-timeout-ms",
            "100",
            "--engine-health-failure-threshold",
            "5",
        ]
    )
    config = SubscriberConfig.from_args(args)
    assert config.engine_health_interval_s == 0.25
    assert config.engine_kvcache_worker_status_timeout_ms == 100.0
    assert config.engine_health_failure_threshold == 5


def test_health_config_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        textwrap.dedent(
            """\
            engine_health_interval_s: 2.5
            engine_kvcache_worker_status_timeout_ms: 750
            engine_health_failure_threshold: 7
            """
        )
    )
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--config", str(yaml_file)])
    config = SubscriberConfig.from_args(args)
    assert config.engine_health_interval_s == 2.5
    assert config.engine_kvcache_worker_status_timeout_ms == 750
    assert config.engine_health_failure_threshold == 7


def test_legacy_engine_health_timeout_cli_argument_is_rejected() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([*_KVCM_ARGS, "--engine-health-timeout-s", "0.1"])


def test_zmq_transport_config_defaults() -> None:
    config = SubscriberConfig()
    assert config.zmq_reconnect_ivl_ms == 100
    assert config.zmq_reconnect_ivl_max_ms == 5000
    assert config.zmq_replay_timeout_s == 1.0
    assert config.zmq_tcp_keepalive is True
    assert config.zmq_tcp_keepalive_idle_s == 30
    assert config.zmq_tcp_keepalive_intvl_s == 5
    assert config.zmq_tcp_keepalive_cnt == 3


def test_zmq_replay_timeout_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--zmq-replay-timeout-s", "0.25"])

    config = SubscriberConfig.from_args(args)

    assert config.zmq_replay_timeout_s == 0.25


def test_zmq_replay_timeout_cli_overrides_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("zmq_replay_timeout_s: 0.75\n")
    parser = build_parser()
    args = parser.parse_args(
        [
            *_KVCM_ARGS,
            "--config",
            str(yaml_file),
            "--zmq-replay-timeout-s",
            "0.25",
        ]
    )

    config = SubscriberConfig.from_args(args)

    assert config.zmq_replay_timeout_s == 0.25


def test_zmq_replay_timeout_must_be_positive() -> None:
    config = SubscriberConfig(
        kvcm_base_url="http://kvcm-test:8080",
        zmq_replay_timeout_s=0,
        host_port=8080,
    )

    with pytest.raises(ValueError, match="zmq_replay_timeout_s must be > 0"):
        config.validate()


def test_zmq_transport_config_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            *_KVCM_ARGS,
            "--zmq-reconnect-ivl-ms",
            "200",
            "--zmq-reconnect-ivl-max-ms",
            "3000",
            "--no-zmq-tcp-keepalive",
            "--zmq-tcp-keepalive-idle-s",
            "10",
            "--zmq-tcp-keepalive-intvl-s",
            "2",
            "--zmq-tcp-keepalive-cnt",
            "4",
        ]
    )
    config = SubscriberConfig.from_args(args)
    assert config.zmq_reconnect_ivl_ms == 200
    assert config.zmq_reconnect_ivl_max_ms == 3000
    assert config.zmq_tcp_keepalive is False
    assert config.zmq_tcp_keepalive_idle_s == 10
    assert config.zmq_tcp_keepalive_intvl_s == 2
    assert config.zmq_tcp_keepalive_cnt == 4


def test_log_level_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--log-level", "debug"])
    config = SubscriberConfig.from_args(args)
    assert config.log_level == "debug"


def test_log_level_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("log_level: 'warning'\n")
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--config", str(yaml_file)])
    config = SubscriberConfig.from_args(args)
    assert config.log_level == "warning"


def test_invalid_log_level_yaml_raises(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("log_level: 'verbose'\n")
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--config", str(yaml_file)])

    with pytest.raises(
        ValueError,
        match="^log_level must be one of: critical, debug, error, info, warning$",
    ):
        SubscriberConfig.from_args(args)


def test_non_string_log_level_yaml_raises(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("log_level: 123\n")
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--config", str(yaml_file)])

    with pytest.raises(
        ValueError,
        match="^log_level must be one of: critical, debug, error, info, warning$",
    ):
        SubscriberConfig.from_args(args)


def test_non_positive_health_failure_threshold_raises() -> None:
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--engine-health-failure-threshold", "0"])

    with pytest.raises(
        ValueError,
        match="^engine_health_failure_threshold must be >= 1$",
    ):
        SubscriberConfig.from_args(args)


# --- Subscriber health integration config ---


def test_subscriber_health_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DSV_CONTROL_PORT", raising=False)
    config = SubscriberConfig()
    assert config.subscriber_health_enabled is True
    assert config.subscriber_state_url == "http://127.0.0.1:8601/subscriber_state"
    assert config.subscriber_heartbeat_interval_s == 3.0
    assert config.subscriber_state_request_timeout_s == 2.0
    assert config.subscriber_shutdown_report_timeout_s == 2.0


def test_removed_dashserving_readiness_and_ttl_fields_are_not_in_config() -> None:
    config = SubscriberConfig()

    assert not hasattr(config, "subscriber_state_ttl_s")
    assert not hasattr(config, "dashserving_readiness_url")
    assert not hasattr(config, "dashserving_readiness_interval_s")


def test_subscriber_health_config_cli_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSV_CONTROL_PORT", "9999")
    parser = build_parser()
    args = parser.parse_args(
        [
            *_KVCM_ARGS,
            "--subscriber-health-enabled",
            "--subscriber-heartbeat-interval-s",
            "5.0",
            "--subscriber-state-request-timeout-s",
            "3.0",
            "--subscriber-shutdown-report-timeout-s",
            "4.0",
        ]
    )
    config = SubscriberConfig.from_args(args)
    assert config.subscriber_health_enabled is True
    assert config.subscriber_state_url == "http://127.0.0.1:9999/subscriber_state"
    assert config.subscriber_heartbeat_interval_s == 5.0
    assert config.subscriber_state_request_timeout_s == 3.0
    assert config.subscriber_shutdown_report_timeout_s == 4.0


def test_subscriber_health_config_yaml_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DSV_CONTROL_PORT", "7777")
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        textwrap.dedent(
            """\
            subscriber_health_enabled: true
            subscriber_heartbeat_interval_s: 4.0
            subscriber_state_request_timeout_s: 1.5
            subscriber_shutdown_report_timeout_s: 3.5
            """
        )
    )
    parser = build_parser()
    args = parser.parse_args([*_KVCM_ARGS, "--config", str(yaml_file)])
    config = SubscriberConfig.from_args(args)
    assert config.subscriber_health_enabled is True
    assert config.subscriber_state_url == "http://127.0.0.1:7777/subscriber_state"
    assert config.subscriber_heartbeat_interval_s == 4.0
    assert config.subscriber_state_request_timeout_s == 1.5
    assert config.subscriber_shutdown_report_timeout_s == 3.5


def test_subscriber_health_cli_overrides_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        textwrap.dedent(
            """\
            subscriber_health_enabled: true
            subscriber_heartbeat_interval_s: 4.0
            """
        )
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            *_KVCM_ARGS,
            "--config",
            str(yaml_file),
            "--subscriber-heartbeat-interval-s",
            "2.0",
        ]
    )
    config = SubscriberConfig.from_args(args)
    assert config.subscriber_health_enabled is True
    assert config.subscriber_heartbeat_interval_s == 2.0


def test_subscriber_health_cli_disable_overrides_yaml_true(tmp_path: Path) -> None:
    """--no-subscriber-health-enabled must override YAML `true`.

    Guards against regressing back to action="store_true", which has no
    --no- form and would make a YAML-enabled flag impossible to disable
    from the CLI.
    """
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("subscriber_health_enabled: true\n")
    parser = build_parser()
    args = parser.parse_args(
        [*_KVCM_ARGS, "--config", str(yaml_file), "--no-subscriber-health-enabled"]
    )
    config = SubscriberConfig.from_args(args)
    assert config.subscriber_health_enabled is False


def test_subscriber_health_disabled_skips_validation() -> None:
    """When subscriber_health_enabled=False, no health validation runs."""
    config = SubscriberConfig(
        kvcm_base_url="http://kvcm-test:8080",
        subscriber_health_enabled=False,
        subscriber_heartbeat_interval_s=-1.0,
        host_port=8080,
    )
    # Should not raise — validation only applies when enabled
    config.validate()


def test_subscriber_health_enabled_positive_intervals() -> None:
    config = SubscriberConfig(
        kvcm_base_url="http://kvcm-test:8080",
        subscriber_health_enabled=True,
        subscriber_heartbeat_interval_s=0,
        host_port=8080,
    )
    with pytest.raises(ValueError, match="subscriber_heartbeat_interval_s must be > 0"):
        config.validate()


def test_subscriber_health_enabled_positive_request_timeout() -> None:
    config = SubscriberConfig(
        kvcm_base_url="http://kvcm-test:8080",
        subscriber_health_enabled=True,
        subscriber_state_request_timeout_s=0,
        host_port=8080,
    )
    with pytest.raises(
        ValueError, match="subscriber_state_request_timeout_s must be > 0"
    ):
        config.validate()


def test_subscriber_health_enabled_positive_shutdown_timeout() -> None:
    config = SubscriberConfig(
        kvcm_base_url="http://kvcm-test:8080",
        subscriber_health_enabled=True,
        subscriber_shutdown_report_timeout_s=-0.5,
        host_port=8080,
    )
    with pytest.raises(
        ValueError, match="subscriber_shutdown_report_timeout_s must be > 0"
    ):
        config.validate()


def test_subscriber_health_valid_config_passes() -> None:
    """A fully valid enabled config passes validation."""
    config = SubscriberConfig(
        kvcm_base_url="http://kvcm-test:8080",
        subscriber_health_enabled=True,
        subscriber_heartbeat_interval_s=3.0,
        subscriber_state_request_timeout_s=2.0,
        subscriber_shutdown_report_timeout_s=2.0,
        host_port=8080,
    )
    config.validate()  # Should not raise


def test_engine_snapshot_full_sync_interval_must_be_positive() -> None:
    config = SubscriberConfig(
        kvcm_base_url="http://kvcm-test:8080",
        engine_snapshot_full_sync_interval_s=0,
        host_port=8080,
    )

    with pytest.raises(
        ValueError, match="engine_snapshot_full_sync_interval_s must be > 0"
    ):
        config.validate()


def test_engine_snapshot_full_sync_interval_negative_raises() -> None:
    config = SubscriberConfig(
        kvcm_base_url="http://kvcm-test:8080",
        engine_snapshot_full_sync_interval_s=-5.0,
        host_port=8080,
    )

    with pytest.raises(
        ValueError, match="engine_snapshot_full_sync_interval_s must be > 0"
    ):
        config.validate()
