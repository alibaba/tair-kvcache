from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from subscriber.config import SubscriberConfig, build_parser


def test_defaults() -> None:
    config = SubscriberConfig()
    assert config.zmq_pub_endpoint == "tcp://localhost:5557"
    assert config.zmq_replay_endpoint == "tcp://localhost:5558"
    assert config.zmq_topic == ""
    assert config.engine_type == "vllm"


def test_engine_type_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args(["--engine-type", "sglang"])
    config = SubscriberConfig.from_args(args)
    assert config.engine_type == "sglang"


def test_rtp_cli_overrides_and_endpoint_normalization() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--engine-type",
            "rtp_llm",
            "--rtp-endpoints",
            "127.0.0.1:8089, 127.0.0.1:8097",
            "--rtp-rpc-timeout-s",
            "2.5",
            "--rtp-poll-interval-s",
            "0.5",
            "--rtp-deletion-confirmations",
            "3",
            "--rtp-full-refresh-interval-s",
            "60",
            "--no-rtp-reset-on-start",
        ]
    )

    config = SubscriberConfig.from_args(args)

    assert config.rtp_endpoint_list == (
        "127.0.0.1:8089",
        "127.0.0.1:8097",
    )
    assert config.rtp_rpc_timeout_s == 2.5
    assert config.rtp_poll_interval_s == 0.5
    assert config.rtp_deletion_confirmations == 3
    assert config.rtp_full_refresh_interval_s == 60
    assert config.rtp_reset_on_start is False


def test_rtp_requires_at_least_one_endpoint() -> None:
    config = SubscriberConfig(engine_type="rtp_llm", rtp_endpoints=" , ")

    with pytest.raises(
        ValueError,
        match="^rtp_endpoints must contain at least one endpoint$",
    ):
        config.validate()


def test_kv_event_queue_maxsize_defaults_to_1024() -> None:
    config = SubscriberConfig()

    assert config.kv_event_queue_maxsize == 1024


def test_kv_event_queue_maxsize_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args(["--kv-event-queue-maxsize", "7"])

    config = SubscriberConfig.from_args(args)

    assert config.kv_event_queue_maxsize == 7


def test_kv_event_queue_maxsize_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("kv_event_queue_maxsize: 9\n")
    parser = build_parser()
    args = parser.parse_args(["--config", str(yaml_file)])

    config = SubscriberConfig.from_args(args)

    assert config.kv_event_queue_maxsize == 9


def test_kv_event_queue_maxsize_must_be_positive() -> None:
    config = SubscriberConfig(kv_event_queue_maxsize=0)

    with pytest.raises(ValueError, match="kv_event_queue_maxsize must be >= 1"):
        config.validate()


def test_kvcm_report_batch_size_must_be_positive() -> None:
    config = SubscriberConfig(kvcm_report_batch_size=0)

    with pytest.raises(ValueError, match="kvcm_report_batch_size must be >= 1"):
        config.validate()


def test_kvcm_runtime_config_defaults() -> None:
    config = SubscriberConfig()
    assert config.kvcm_heartbeat_interval_s == 1.0


def test_kvcm_runtime_config_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
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
    args = parser.parse_args(["--config", str(yaml_file)])
    config = SubscriberConfig.from_args(args)
    assert config.kvcm_heartbeat_interval_s == 3.0


def test_non_positive_kvcm_heartbeat_interval_raises() -> None:
    parser = build_parser()
    args = parser.parse_args(["--kvcm-heartbeat-interval-s", "0"])

    with pytest.raises(ValueError, match="^kvcm_heartbeat_interval_s must be > 0$"):
        SubscriberConfig.from_args(args)


def test_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        textwrap.dedent("""\
            zmq_pub_endpoint: "tcp://remote:5557"
        """)
    )
    parser = build_parser()
    args = parser.parse_args(["--config", str(yaml_file)])
    config = SubscriberConfig.from_args(args)
    assert config.zmq_pub_endpoint == "tcp://remote:5557"
    assert config.zmq_replay_endpoint == "tcp://localhost:5558"


def test_cli_overrides_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("engine_type: vllm\n")
    parser = build_parser()
    args = parser.parse_args(["--config", str(yaml_file), "--engine-type", "sglang"])
    config = SubscriberConfig.from_args(args)
    assert config.engine_type == "sglang"


def test_health_config_defaults() -> None:
    config = SubscriberConfig()
    assert config.engine_health_url == "http://127.0.0.1:8000/health"
    assert config.engine_health_interval_s == 1.0
    assert config.engine_health_timeout_s == 1.0
    assert config.engine_health_failure_threshold == 3


def test_health_config_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--engine-health-url",
            "http://10.0.0.1:9000/health",
            "--engine-health-interval-s",
            "0.25",
            "--engine-health-timeout-s",
            "0.1",
            "--engine-health-failure-threshold",
            "5",
        ]
    )
    config = SubscriberConfig.from_args(args)
    assert config.engine_health_url == "http://10.0.0.1:9000/health"
    assert config.engine_health_interval_s == 0.25
    assert config.engine_health_timeout_s == 0.1
    assert config.engine_health_failure_threshold == 5


def test_health_config_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        textwrap.dedent(
            """\
            engine_health_url: "http://10.0.0.2:9001/health"
            engine_health_interval_s: 2.5
            engine_health_timeout_s: 0.75
            engine_health_failure_threshold: 7
            """
        )
    )
    parser = build_parser()
    args = parser.parse_args(["--config", str(yaml_file)])
    config = SubscriberConfig.from_args(args)
    assert config.engine_health_url == "http://10.0.0.2:9001/health"
    assert config.engine_health_interval_s == 2.5
    assert config.engine_health_timeout_s == 0.75
    assert config.engine_health_failure_threshold == 7


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
    args = parser.parse_args(["--zmq-replay-timeout-s", "0.25"])

    config = SubscriberConfig.from_args(args)

    assert config.zmq_replay_timeout_s == 0.25


def test_zmq_replay_timeout_cli_overrides_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("zmq_replay_timeout_s: 0.75\n")
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(yaml_file),
            "--zmq-replay-timeout-s",
            "0.25",
        ]
    )

    config = SubscriberConfig.from_args(args)

    assert config.zmq_replay_timeout_s == 0.25


def test_zmq_replay_timeout_must_be_positive() -> None:
    config = SubscriberConfig(zmq_replay_timeout_s=0)

    with pytest.raises(ValueError, match="zmq_replay_timeout_s must be > 0"):
        config.validate()


def test_data_parallel_config_defaults() -> None:
    config = SubscriberConfig()
    assert config.data_parallel_size == 1
    assert config.dp_endpoints[0].rank == 0
    assert config.dp_endpoints[0].zmq_pub_endpoint == "tcp://localhost:5557"
    assert config.dp_endpoints[0].zmq_replay_endpoint == "tcp://localhost:5558"
    assert config.dp_endpoints[0].zmq_topic == ""


def test_data_parallel_size_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args(["--data-parallel-size", "2"])
    config = SubscriberConfig.from_args(args)
    assert config.data_parallel_size == 2


def test_data_parallel_size_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("data_parallel_size: 3\n")
    parser = build_parser()
    args = parser.parse_args(["--config", str(yaml_file)])
    config = SubscriberConfig.from_args(args)
    assert config.data_parallel_size == 3


def test_dp_endpoints_expand_tcp_ports_by_rank() -> None:
    config = SubscriberConfig(data_parallel_size=3)
    assert [endpoint.rank for endpoint in config.dp_endpoints] == [0, 1, 2]
    assert [endpoint.zmq_pub_endpoint for endpoint in config.dp_endpoints] == [
        "tcp://localhost:5557",
        "tcp://localhost:5558",
        "tcp://localhost:5559",
    ]
    assert [endpoint.zmq_replay_endpoint for endpoint in config.dp_endpoints] == [
        "tcp://localhost:5558",
        "tcp://localhost:5559",
        "tcp://localhost:5560",
    ]


def test_non_positive_data_parallel_size_raises() -> None:
    parser = build_parser()
    args = parser.parse_args(["--data-parallel-size", "0"])

    with pytest.raises(ValueError, match="^data_parallel_size must be >= 1$"):
        SubscriberConfig.from_args(args)


def test_multi_dp_requires_tcp_endpoint() -> None:
    config = SubscriberConfig(
        data_parallel_size=2,
        zmq_pub_endpoint="ipc://kv-events",
        zmq_replay_endpoint="tcp://localhost:5558",
    )

    with pytest.raises(
        ValueError,
        match="^zmq_pub_endpoint must be a tcp://host:port endpoint for multi-DP$",
    ):
        config.validate()


def test_multi_dp_requires_endpoint_port() -> None:
    config = SubscriberConfig(
        data_parallel_size=2,
        zmq_pub_endpoint="tcp://localhost",
        zmq_replay_endpoint="tcp://localhost:5558",
    )

    with pytest.raises(
        ValueError,
        match="^zmq_pub_endpoint must include a TCP port for multi-DP$",
    ):
        config.validate()


def test_zmq_transport_config_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
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
    args = parser.parse_args(["--log-level", "debug"])
    config = SubscriberConfig.from_args(args)
    assert config.log_level == "debug"


def test_log_level_yaml_loading(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("log_level: 'warning'\n")
    parser = build_parser()
    args = parser.parse_args(["--config", str(yaml_file)])
    config = SubscriberConfig.from_args(args)
    assert config.log_level == "warning"


def test_invalid_log_level_yaml_raises(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("log_level: 'verbose'\n")
    parser = build_parser()
    args = parser.parse_args(["--config", str(yaml_file)])

    with pytest.raises(
        ValueError,
        match="^log_level must be one of: critical, debug, error, info, warning$",
    ):
        SubscriberConfig.from_args(args)


def test_non_string_log_level_yaml_raises(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("log_level: 123\n")
    parser = build_parser()
    args = parser.parse_args(["--config", str(yaml_file)])

    with pytest.raises(
        ValueError,
        match="^log_level must be one of: critical, debug, error, info, warning$",
    ):
        SubscriberConfig.from_args(args)


def test_non_positive_health_failure_threshold_raises() -> None:
    parser = build_parser()
    args = parser.parse_args(["--engine-health-failure-threshold", "0"])

    with pytest.raises(
        ValueError,
        match="^engine_health_failure_threshold must be >= 1$",
    ):
        SubscriberConfig.from_args(args)
