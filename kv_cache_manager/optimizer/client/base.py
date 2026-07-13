"""Shared abstractions for Online Optimizer client SDKs.

This module defines the transport-agnostic pieces shared by
``OptimizerGrpcClient`` and ``OptimizerHttpClient``:

- ``OptimizerClientConfig`` / ``OptimizerClientInitParams``: construction
  inputs, following the KVCacheManager client style
  (``Create(config, init_params)``).
- ``OptimizerClientError``: unified business-level error raised when
  ``header.status.code`` is not one of the accepted codes, regardless of
  transport.
- ``OptimizerClientBase``: an abstract base class describing the 11
  ``OptimizerService`` RPCs. Both transports return the *same* protobuf
  message types (HTTP responses are parsed back into protobuf messages via
  ``google.protobuf.json_format.ParseDict``), so response handling
  (``parse_trace_query``, ``_check_response``) is implemented once here and
  reused by both subclasses.
"""

import abc
import json
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from kv_cache_manager.protocol.protobuf import optimizer_service_pb2 as pb2


@dataclass
class OptimizerClientConfig:
    """Connection configuration shared by all Online Optimizer client transports.

    ``address`` has no implicit default and must be provided explicitly by
    the caller, matching the KVCacheManager C++ client convention where the
    connection address is a required config field rather than a baked-in
    default (which could otherwise silently connect to localhost).

    ``options`` / ``wait_for_ready`` are gRPC-specific and ignored by
    ``OptimizerHttpClient``.
    """

    address: str
    timeout: float = 10.0
    connection_timeout: float = 5.0
    max_retries: int = 0
    retry_backoff_seconds: float = 0.1
    options: Sequence[Tuple[str, object]] = field(default_factory=tuple)
    wait_for_ready: Optional[bool] = None


@dataclass
class OptimizerClientInitParams:
    """Reserved init params for parity with KVCacheManager client construction."""

    pass


class OptimizerClientError(RuntimeError):
    """Business-level Online Optimizer error, raised by either transport."""

    def __init__(self, operation: str, code, message: str = ""):
        self.operation = operation
        self.code = code
        self.message = message or "unknown error"
        super().__init__(f"{operation} failed: code={code} message={self.message}")


class OptimizerClientBase(abc.ABC):
    """Common interface implemented by ``OptimizerGrpcClient`` and ``OptimizerHttpClient``.

    Both concrete clients return the *same* protobuf message types for every
    RPC (HTTP responses are parsed into protobuf via ``ParseDict``), so
    response parsing helpers below are shared instead of duplicated.
    """

    DEFAULT_TIMEOUT = 10.0
    DEFAULT_CONNECTION_TIMEOUT = 5.0

    # ------------------------------------------------------------------
    # Construction helpers shared by both transports.
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_config(config) -> OptimizerClientConfig:
        if config is None:
            raise ValueError(
                "Optimizer client requires an explicit address; "
                "pass OptimizerClientConfig(address=...), an address string, "
                "or a JSON/dict config containing 'address'")
        if isinstance(config, OptimizerClientConfig):
            if not config.address:
                raise ValueError("OptimizerClientConfig.address must not be empty")
            return config
        if isinstance(config, Mapping):
            return OptimizerClientBase._parse_config(OptimizerClientConfig(**dict(config)))
        if isinstance(config, str):
            stripped = config.strip()
            if not stripped:
                raise ValueError("Optimizer client requires a non-empty address")
            if stripped.startswith("{"):
                return OptimizerClientBase._parse_config(OptimizerClientConfig(**json.loads(stripped)))
            return OptimizerClientConfig(address=stripped)
        raise TypeError(
            "config must be an address string, JSON string, dict, or OptimizerClientConfig")

    @staticmethod
    def new_trace_id() -> str:
        import uuid
        return str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Shared response handling. Both transports feed the same protobuf
    # message types into these helpers.
    # ------------------------------------------------------------------
    @staticmethod
    def _check_response(operation: str, response, ok_codes: Sequence[int] = (pb2.OK,)):
        header = getattr(response, "header", None)
        if header is None:
            return
        code = header.status.code
        if code not in ok_codes:
            raise OptimizerClientError(operation, code, header.status.message)

    @staticmethod
    def parse_trace_query(response) -> Tuple[int, List[Tuple[float, int]], Optional[int]]:
        total_blocks = int(response.total_blocks)
        per_capacity = [
            (float(item.capacity_gb), int(item.cache_hit_count))
            for item in response.capacity_results
        ]
        theoretical_hits = None
        if response.HasField("theoretical_result"):
            max_hit = int(response.theoretical_result.max_hit_count)
            if max_hit >= 0:
                theoretical_hits = max_hit
        return total_blocks, per_capacity, theoretical_hits

    def trace_query_for_stats(self, instance_id: str, block_keys: Iterable[int]):
        import time
        start = time.monotonic()
        response = self.trace_query(instance_id, block_keys)
        latency_ms = (time.monotonic() - start) * 1000
        total_blocks, per_capacity, theoretical_hits = self.parse_trace_query(response)
        return latency_ms, total_blocks, per_capacity, theoretical_hits

    @staticmethod
    def _eviction_policy(value):
        if isinstance(value, int):
            return value
        return getattr(pb2, value, pb2.OPTIMIZER_EVICTION_POLICY_LRU)

    # ------------------------------------------------------------------
    # Context manager.
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Abstract RPC surface (11 OptimizerService methods).
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def create_instance_group(
        self,
        name: str,
        capacity_gb,
        eviction_policy=pb2.OPTIMIZER_EVICTION_POLICY_LRU,
        ttl_seconds: int = 0,
        shared_group_quota: bool = False,
        enable_theoretical_max_cache: bool = False,
        allow_duplicate: bool = True,
        trace_id: Optional[str] = None,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def update_instance_group(
        self,
        name: str,
        capacity_gb,
        eviction_policy=pb2.OPTIMIZER_EVICTION_POLICY_LRU,
        ttl_seconds: int = 0,
        shared_group_quota: bool = False,
        enable_theoretical_max_cache: bool = False,
        trace_id: Optional[str] = None,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def remove_instance_group(self, name: str, trace_id: Optional[str] = None):
        raise NotImplementedError

    @abc.abstractmethod
    def get_instance_group(self, name: str, trace_id: Optional[str] = None):
        raise NotImplementedError

    @abc.abstractmethod
    def list_instance_groups(self, trace_id: Optional[str] = None):
        raise NotImplementedError

    @abc.abstractmethod
    def register_instance(
        self,
        instance_group: str,
        instance_id: str,
        block_size: int,
        block_bytes: int = 0,
        location_spec_infos=None,
        location_spec_groups=None,
        optimizer_state_info=None,
        linear_step: int = 0,
        allow_duplicate: bool = True,
        trace_id: Optional[str] = None,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def remove_instance(self, instance_id: str, trace_id: Optional[str] = None):
        raise NotImplementedError

    @abc.abstractmethod
    def get_instance(self, instance_id: str, trace_id: Optional[str] = None):
        raise NotImplementedError

    @abc.abstractmethod
    def list_instances(self, instance_group: str = "", trace_id: Optional[str] = None):
        raise NotImplementedError

    @abc.abstractmethod
    def trace_query(
        self,
        instance_id: str,
        block_keys: Iterable[int],
        token_ids: Optional[Iterable[int]] = None,
        trace_id: Optional[str] = None,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def reset_stats(self, instance_id: str, trace_id: Optional[str] = None):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError
