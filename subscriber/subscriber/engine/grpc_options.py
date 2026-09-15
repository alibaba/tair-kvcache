"""Shared gRPC channel options for DashLLM engine clients."""

from __future__ import annotations

CHANNEL_OPTIONS: list[tuple[str, int]] = [
    ("grpc.max_receive_message_length", 8 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 2_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.enable_retries", 0),
    ("grpc.initial_reconnect_backoff_ms", 100),
    ("grpc.max_reconnect_backoff_ms", 1_000),
    ("grpc.tcp_receive_buffer_size", 512 * 1024),
]
