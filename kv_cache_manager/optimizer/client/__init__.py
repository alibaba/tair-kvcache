"""Python client SDK for Online Optimizer.

Provides two interchangeable transports for ``OptimizerService``:
``OptimizerGrpcClient`` (gRPC) and ``OptimizerHttpClient`` (HTTP REST). Both
implement the shared ``OptimizerClientBase`` interface and return the same
protobuf response message types, so callers can switch transports without
changing call sites. Use ``create_optimizer_client(protocol, config)`` for a
single cross-protocol construction entry point.
"""

from typing import Optional

from .base import (
    OptimizerClientBase,
    OptimizerClientConfig,
    OptimizerClientError,
    OptimizerClientInitParams,
)
from .grpc_client import OptimizerGrpcClient
from .http_client import OptimizerHttpClient

__all__ = [
    "OptimizerClientBase",
    "OptimizerClientConfig",
    "OptimizerClientError",
    "OptimizerClientInitParams",
    "OptimizerGrpcClient",
    "OptimizerHttpClient",
    "create_optimizer_client",
]

_TRANSPORTS = {
    "grpc": OptimizerGrpcClient,
    "http": OptimizerHttpClient,
}


def create_optimizer_client(
    protocol: str = "grpc",
    config=None,
    init_params: Optional[OptimizerClientInitParams] = None,
) -> OptimizerClientBase:
    """Construct an Online Optimizer client for the given transport protocol.

    ``protocol`` defaults to ``"grpc"`` and can also be ``"http"``. This is
    the single public entry point for cross-protocol construction; the returned
    client always implements ``OptimizerClientBase``.
    """
    client_class = _TRANSPORTS.get(protocol.lower())
    if client_class is None:
        raise ValueError(f"Unsupported protocol={protocol!r}; expected one of {sorted(_TRANSPORTS)}")
    return client_class.Create(config, init_params)
