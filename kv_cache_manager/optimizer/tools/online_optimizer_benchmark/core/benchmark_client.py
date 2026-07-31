"""Benchmark client adapter for the public Online Optimizer SDK.

This module converts ``BenchmarkConfig`` (environment-derived benchmark
configuration) into the transport-agnostic ``OptimizerClientConfig`` used by
the public ``kv_cache_manager.optimizer.client`` SDK. It does not implement
HTTP or gRPC itself.
"""

from kv_cache_manager.optimizer.client import OptimizerClientConfig
from kv_cache_manager.optimizer.client import create_optimizer_client as _create_optimizer_client

from .config import BenchmarkConfig


def create_benchmark_client(config: BenchmarkConfig):
    if config.protocol not in ("http", "grpc"):
        raise ValueError(f"Unsupported BENCH_PROTOCOL={config.protocol!r}; expected http or grpc")
    client_config = OptimizerClientConfig(
        address=config.optimizer_address,
        timeout=config.request_timeout,
        connection_timeout=config.connection_timeout,
    )
    return _create_optimizer_client(config.protocol, client_config)
