from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, fields
from dataclasses import field as dataclass_field

import yaml

from subscriber.kvcm.enum import KvcmQueryType, KvcmStorageType
from subscriber.kvcm.kinds import validate_extra_attention_types

_LOG_LEVELS = ("debug", "info", "warning", "error", "critical")
_KVCM_PROTOCOLS = ("grpc", "http")


def _parse_extra_attention_types(raw: str) -> dict[str, str]:
    """Parse the CLI JSON object used to extend attention-type mappings."""

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "extra_attention_types must be a JSON object"
        ) from exc
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(category, str)
        for key, category in value.items()
    ):
        raise argparse.ArgumentTypeError(
            "extra_attention_types must be a JSON object with string keys and values"
        )
    return value


@dataclass(frozen=True)
class DpEndpoint:
    rank: int
    zmq_pub_endpoint: str
    zmq_replay_endpoint: str
    zmq_topic: str = ""


@dataclass
class SubscriberConfig:
    """Runtime configuration for the subscriber process."""

    # ZMQ
    zmq_replay_timeout_s: float = 1.0
    zmq_reconnect_ivl_ms: int = 100
    zmq_reconnect_ivl_max_ms: int = 5000
    zmq_tcp_keepalive: bool = True
    zmq_tcp_keepalive_idle_s: int = 30
    zmq_tcp_keepalive_intvl_s: int = 5
    zmq_tcp_keepalive_cnt: int = 3

    kv_event_queue_maxsize: int = 4096
    kv_event_merge_max_report_events: int = 16
    kv_event_merge_max_queue_items: int = 4
    snapshot_queue_maxsize: int = 16

    # Engine
    engine_type: str = "vllm"

    # Engine liveness (gRPC GetWorkerStatus polling).
    engine_health_interval_s: float = 5.0
    engine_health_failure_threshold: int = 3

    # Hybrid adapter: full snapshot reconciliation interval
    engine_snapshot_full_sync_interval_s: float = 30.0
    engine_kvcache_snapshot_timeout_ms: float = 5000.0
    engine_kvcache_worker_status_timeout_ms: float = 1000.0
    engine_kv_event_bootstrap_timeout_ms: float = 1000.0
    engine_kv_event_bootstrap_max_retries: int = 5

    # Remote worker-status TCP endpoint and same-Pod KV-event control UDS.
    engine_grpc_endpoint: str = "127.0.0.1:18002"
    engine_kv_event_control_uds_path: str = "/tmp/dashllm-kv-event-control.sock"

    # Pipeline enable/disable.
    incremental_kv_event_pipeline_enabled: bool = True
    snapshot_kv_event_pipeline_enabled: bool = False

    # Worker identity port advertised to KVCM as host_ip_port. Must match the
    # engine endpoint port that FlexLB discovers via Spectrum. No default;
    # validate() rejects a missing value at startup.
    host_port: int | None = None

    # kvcm SDK
    kvcm_heartbeat_interval_s: float = 5.0
    kvcm_request_timeout_s: float = 5.0
    kvcm_query_type: str = KvcmQueryType.QT_PREFIX_MATCH_WITH_MAMBA
    kvcm_storage_type: str = KvcmStorageType.ST_EVENT_REPORT_L1P5
    kvcm_base_url: str = ""
    kvcm_protocol: str = "grpc"
    kvcm_instance_group: str = ""
    extra_attention_types: dict[str, str] = dataclass_field(default_factory=dict)

    # Dynamic polling interval (reserved for future adaptive polling).
    poll_min_interval_s: float = 0.05
    poll_max_interval_s: float = 0.2
    poll_initial_interval_s: float = 0.1
    poll_target_snapshot_size: int = 100

    # Logging
    log_level: str = "info"

    # Subscriber health integration (DashServing same-host design).
    subscriber_health_enabled: bool = True
    subscriber_heartbeat_interval_s: float = 3.0
    subscriber_state_request_timeout_s: float = 2.0
    subscriber_shutdown_report_timeout_s: float = 2.0

    # Config-file keys that matched no known field, recorded by ``from_args``
    # so the CLI entry point can warn about them after logging is initialized.
    unknown_config_keys: list[str] = dataclass_field(default_factory=list)

    @property
    def subscriber_state_url(self) -> str:
        """Derive the DashServing state URL from DSV_CONTROL_PORT."""
        port = os.environ.get("DSV_CONTROL_PORT", "8601")
        return f"http://127.0.0.1:{port}/subscriber_state"

    @classmethod
    def add_cli_args(
        cls,
        parser: argparse.ArgumentParser,
        default: object | str | None = None,
    ) -> None:
        """Register CLI overrides without masking YAML or dataclass defaults."""

        parser.add_argument("--zmq-replay-timeout-s", type=float, default=default)
        parser.add_argument("--kvcm-heartbeat-interval-s", type=float, default=default)
        parser.add_argument("--kvcm-request-timeout-s", type=float, default=default)
        parser.add_argument("--kv-event-queue-maxsize", type=int, default=default)
        parser.add_argument("--snapshot-queue-maxsize", type=int, default=default)
        parser.add_argument(
            "--kv-event-merge-max-report-events", type=int, default=default
        )
        parser.add_argument(
            "--kv-event-merge-max-queue-items", type=int, default=default
        )
        parser.add_argument("--engine-type", default=default)
        parser.add_argument(
            "--log-level",
            choices=_LOG_LEVELS,
            default=default,
        )
        parser.add_argument("--engine-health-interval-s", type=float, default=default)
        parser.add_argument(
            "--engine-health-failure-threshold", type=int, default=default
        )
        parser.add_argument(
            "--engine-kvcache-worker-status-timeout-ms", type=float, default=default
        )
        parser.add_argument("--engine-grpc-endpoint", default=default)
        parser.add_argument("--engine-kv-event-control-uds-path", default=default)
        parser.add_argument(
            "--engine-kv-event-bootstrap-max-retries", type=int, default=default
        )
        parser.add_argument(
            "--engine-kv-event-bootstrap-timeout-ms", type=float, default=default
        )
        parser.add_argument(
            "--host-port",
            type=int,
            default=default,
            help="Worker identity port advertised to KVCM as host_ip_port "
            "(must match the engine endpoint port FlexLB discovers via Spectrum)",
        )
        parser.add_argument("--kvcm-query-type", default=default)
        parser.add_argument("--kvcm-storage-type", default=default)
        parser.add_argument("--kvcm-base-url", default=default)
        parser.add_argument(
            "--kvcm-protocol",
            choices=_KVCM_PROTOCOLS,
            default=default,
        )
        parser.add_argument("--kvcm-instance-group", default=default)
        parser.add_argument(
            "--extra-attention-types",
            type=_parse_extra_attention_types,
            default=default,
        )
        parser.add_argument("--poll-min-interval-s", type=float, default=default)
        parser.add_argument("--poll-max-interval-s", type=float, default=default)
        parser.add_argument("--poll-initial-interval-s", type=float, default=default)
        parser.add_argument("--poll-target-snapshot-size", type=int, default=default)
        parser.add_argument(
            "--engine-snapshot-full-sync-interval-s", type=float, default=default
        )
        parser.add_argument(
            "--engine-kvcache-snapshot-timeout-ms", type=float, default=default
        )
        parser.add_argument(
            "--incremental-kv-event-pipeline-enabled",
            action=argparse.BooleanOptionalAction,
            default=default,
        )
        parser.add_argument(
            "--snapshot-kv-event-pipeline-enabled",
            action=argparse.BooleanOptionalAction,
            default=default,
        )
        parser.add_argument("--zmq-reconnect-ivl-ms", type=int, default=default)
        parser.add_argument("--zmq-reconnect-ivl-max-ms", type=int, default=default)
        parser.add_argument(
            "--zmq-tcp-keepalive",
            action=argparse.BooleanOptionalAction,
            default=default,
        )
        parser.add_argument("--zmq-tcp-keepalive-idle-s", type=int, default=default)
        parser.add_argument("--zmq-tcp-keepalive-intvl-s", type=int, default=default)
        parser.add_argument("--zmq-tcp-keepalive-cnt", type=int, default=default)
        parser.add_argument(
            "--subscriber-health-enabled",
            action=argparse.BooleanOptionalAction,
            default=default,
        )
        parser.add_argument(
            "--subscriber-heartbeat-interval-s", type=float, default=default
        )
        parser.add_argument(
            "--subscriber-state-request-timeout-s", type=float, default=default
        )
        parser.add_argument(
            "--subscriber-shutdown-report-timeout-s", type=float, default=default
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SubscriberConfig:
        """Build config with priority: CLI > yaml file > dataclass defaults."""

        config = cls()
        field_names = {f.name for f in fields(cls)}

        if getattr(args, "config", None):
            with open(args.config, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for key, value in data.items():
                normalized = key.replace("-", "_")
                if normalized in field_names:
                    setattr(config, normalized, value)
                else:
                    config.unknown_config_keys.append(key)

        for field in fields(cls):
            dest = field.name
            cli_val = getattr(args, dest, None)
            if cli_val is not None:
                setattr(config, field.name, cli_val)

        config.validate()
        return config

    def validate(self) -> None:
        validate_extra_attention_types(self.extra_attention_types)
        if not isinstance(self.log_level, str):
            raise ValueError(
                f"log_level must be one of: {', '.join(sorted(_LOG_LEVELS))}"
            )
        self.log_level = self.log_level.lower()
        if self.log_level not in _LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of: {', '.join(sorted(_LOG_LEVELS))}"
            )
        if self.engine_health_failure_threshold < 1:
            raise ValueError("engine_health_failure_threshold must be >= 1")
        if self.engine_kv_event_bootstrap_max_retries < 1:
            raise ValueError("engine_kv_event_bootstrap_max_retries must be >= 1")
        if not self.engine_kv_event_control_uds_path.startswith("/"):
            raise ValueError("engine_kv_event_control_uds_path must be absolute")
        if self.kv_event_queue_maxsize < 1:
            raise ValueError("kv_event_queue_maxsize must be >= 1")
        if self.snapshot_queue_maxsize < 1:
            raise ValueError("snapshot_queue_maxsize must be >= 1")
        if self.kv_event_merge_max_report_events < 1:
            raise ValueError("kv_event_merge_max_report_events must be >= 1")
        if self.kv_event_merge_max_queue_items < 1:
            raise ValueError("kv_event_merge_max_queue_items must be >= 1")
        if self.host_port is None:
            raise ValueError("host_port is required; pass --host-port at startup")
        if not 1 <= self.host_port <= 65535:
            raise ValueError("host_port must be in [1, 65535]")
        if not self.kvcm_base_url.strip():
            self.kvcm_base_url = os.environ.get("KVS_KVCM_ENDPOINT", "")
            if not self.kvcm_base_url.strip():
                raise ValueError("kvcm_base_url is required")
        if not isinstance(self.kvcm_protocol, str):
            raise ValueError(
                f"kvcm_protocol must be one of: {', '.join(_KVCM_PROTOCOLS)}"
            )
        self.kvcm_protocol = self.kvcm_protocol.lower()
        if self.kvcm_protocol not in _KVCM_PROTOCOLS:
            raise ValueError(
                f"kvcm_protocol must be one of: {', '.join(_KVCM_PROTOCOLS)}"
            )
        try:
            self.kvcm_storage_type = KvcmStorageType(self.kvcm_storage_type)
        except ValueError as exc:
            raise ValueError(
                "kvcm_storage_type must be one of: "
                f"{', '.join(storage_type.value for storage_type in KvcmStorageType)}"
            ) from exc
        if self.kvcm_storage_type == KvcmStorageType.ST_UNSPECIFIED:
            raise ValueError("kvcm_storage_type must not be ST_UNSPECIFIED")
        if self.kvcm_heartbeat_interval_s <= 0:
            raise ValueError("kvcm_heartbeat_interval_s must be > 0")
        if self.kvcm_request_timeout_s <= 0:
            raise ValueError("kvcm_request_timeout_s must be > 0")
        if self.zmq_replay_timeout_s <= 0:
            raise ValueError("zmq_replay_timeout_s must be > 0")
        if self.poll_min_interval_s <= 0:
            raise ValueError("poll_min_interval_s must be > 0")
        if self.poll_max_interval_s <= self.poll_min_interval_s:
            raise ValueError("poll_max_interval_s must be > poll_min_interval_s")
        if self.poll_target_snapshot_size < 1:
            raise ValueError("poll_target_snapshot_size must be >= 1")
        if self.engine_snapshot_full_sync_interval_s <= 0:
            raise ValueError("engine_snapshot_full_sync_interval_s must be > 0")
        self.poll_initial_interval_s = max(
            self.poll_min_interval_s,
            min(self.poll_max_interval_s, self.poll_initial_interval_s),
        )

        if self.subscriber_health_enabled:
            self._validate_subscriber_health()

    def _validate_subscriber_health(self) -> None:
        """Validate subscriber health integration fields when enabled."""
        for field_name in (
            "subscriber_heartbeat_interval_s",
            "subscriber_state_request_timeout_s",
            "subscriber_shutdown_report_timeout_s",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be > 0")


def build_parser() -> argparse.ArgumentParser:
    """Build the subscriber command-line parser."""

    parser = argparse.ArgumentParser(
        description="tair-kvcache subscriber — vLLM KV event forwarder"
    )
    parser.add_argument(
        "--config", default=None, metavar="FILE", help="Path to YAML config file"
    )
    SubscriberConfig.add_cli_args(parser)
    return parser
