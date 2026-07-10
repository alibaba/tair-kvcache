from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from urllib.parse import SplitResult, urlsplit, urlunsplit

import yaml

_LOG_LEVELS = ("debug", "info", "warning", "error", "critical")


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
    zmq_pub_endpoint: str = "tcp://localhost:5557"
    zmq_replay_endpoint: str = "tcp://localhost:5558"
    zmq_topic: str = ""

    # Data parallelism
    data_parallel_size: int = 1

    # kvcm SDK
    kvcm_addr: str = "localhost:50051"
    kvcm_host_ip_port: str = ""
    kvcm_heartbeat_interval_s: float = 1.0
    kv_event_queue_maxsize: int = 1024

    # Engine
    engine_type: str = "vllm"

    # Engine liveness (HTTP /health polling)
    engine_health_url: str = "http://127.0.0.1:8000/health"
    engine_health_interval_s: float = 1.0
    engine_health_timeout_s: float = 0.5
    engine_health_failure_threshold: int = 3

    # ZMQ SUB socket transport
    zmq_reconnect_ivl_ms: int = 100
    zmq_reconnect_ivl_max_ms: int = 5000
    zmq_tcp_keepalive: bool = True
    zmq_tcp_keepalive_idle_s: int = 30
    zmq_tcp_keepalive_intvl_s: int = 5
    zmq_tcp_keepalive_cnt: int = 3

    # Logging
    log_level: str = "info"

    @property
    def dp_endpoints(self) -> list[DpEndpoint]:
        return [
            DpEndpoint(
                rank=rank,
                zmq_pub_endpoint=_offset_tcp_endpoint(
                    self.zmq_pub_endpoint,
                    rank,
                    field_name="zmq_pub_endpoint",
                ),
                zmq_replay_endpoint=_offset_tcp_endpoint(
                    self.zmq_replay_endpoint,
                    rank,
                    field_name="zmq_replay_endpoint",
                ),
                zmq_topic=self.zmq_topic,
            )
            for rank in range(self.data_parallel_size)
        ]

    @classmethod
    def add_cli_args(
        cls,
        parser: argparse.ArgumentParser,
        default: object | str | None = None,
    ) -> None:
        """Register CLI overrides without masking YAML or dataclass defaults."""

        parser.add_argument("--zmq-pub-endpoint", default=default)
        parser.add_argument("--zmq-replay-endpoint", default=default)
        parser.add_argument("--zmq-topic", default=default)
        parser.add_argument("--data-parallel-size", type=int, default=default)
        parser.add_argument("--kvcm-addr", default=default)
        parser.add_argument("--kvcm-host-ip-port", default=default)
        parser.add_argument("--kvcm-heartbeat-interval-s", type=float, default=default)
        parser.add_argument("--kv-event-queue-maxsize", type=int, default=default)
        parser.add_argument("--engine-type", default=default)
        parser.add_argument(
            "--log-level",
            choices=_LOG_LEVELS,
            default=default,
        )
        parser.add_argument("--engine-health-url", default=default)
        parser.add_argument("--engine-health-interval-s", type=float, default=default)
        parser.add_argument("--engine-health-timeout-s", type=float, default=default)
        parser.add_argument(
            "--engine-health-failure-threshold", type=int, default=default
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

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SubscriberConfig:
        """Build config with priority: CLI > yaml file > dataclass defaults."""

        config = cls()

        if getattr(args, "config", None):
            with open(args.config, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for key, value in data.items():
                normalized = key.replace("-", "_")
                if hasattr(config, normalized):
                    setattr(config, normalized, value)

        for field in fields(cls):
            dest = field.name
            cli_val = getattr(args, dest, None)
            if cli_val is not None:
                setattr(config, field.name, cli_val)

        config.validate()
        return config

    def validate(self) -> None:
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
        if self.data_parallel_size < 1:
            raise ValueError("data_parallel_size must be >= 1")
        if self.data_parallel_size > 1:
            max_port_offset = self.data_parallel_size - 1
            for endpoint, field_name in (
                (self.zmq_pub_endpoint, "zmq_pub_endpoint"),
                (self.zmq_replay_endpoint, "zmq_replay_endpoint"),
            ):
                _offset_tcp_endpoint(
                    endpoint,
                    max_port_offset,
                    field_name=field_name,
                )
        if self.kv_event_queue_maxsize < 1:
            raise ValueError("kv_event_queue_maxsize must be >= 1")
        if self.kvcm_heartbeat_interval_s <= 0:
            raise ValueError("kvcm_heartbeat_interval_s must be > 0")


def _parse_tcp_endpoint_for_multi_dp(
    endpoint: str, *, field_name: str
) -> tuple[SplitResult, int]:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "tcp":
        raise ValueError(
            f"{field_name} must be a tcp://host:port endpoint for multi-DP"
        )
    try:
        port = parsed.port
    except ValueError:
        raise ValueError(f"{field_name} must include a valid TCP port") from None
    if port is None:
        raise ValueError(f"{field_name} must include a TCP port for multi-DP")
    return parsed, port


def _offset_tcp_endpoint(endpoint: str, offset: int, *, field_name: str) -> str:
    if offset == 0:
        return endpoint
    parsed, port = _parse_tcp_endpoint_for_multi_dp(endpoint, field_name=field_name)
    offset_port = port + offset
    if offset_port > 65535:
        raise ValueError(f"{field_name} port offset exceeds 65535")
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        host = f"{auth}@{host}"
    netloc = f"{host}:{offset_port}"
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


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
