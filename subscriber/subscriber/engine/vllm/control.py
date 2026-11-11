from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator

import grpc.aio

from subscriber import logger
from subscriber.config import SubscriberConfig
from subscriber.engine.kv_event_control_client import DashllmKvEventControlClient
from subscriber.engine.metadata import (
    KvEventBootstrap,
    MetadataProtocolError,
    MetadataTemporarilyUnavailable,
    parse_kv_event_bootstrap,
)
from subscriber.engine.worker_status_client import DashllmWorkerStatusClient
from subscriber.health.events import LivenessEvent
from subscriber.metrics import report_engine_probe
from subscriber.proto import engine_service_rpc_pb2

_METADATA_RETRY_BASE_S = 0.5
_METADATA_RETRY_MAX_S = 30.0

_HEALTH_LOG_SUPPRESS_INTERVAL = 12


class VllmControl:
    """Own vLLM health polling and DashLLM metadata control RPCs."""

    def __init__(
        self,
        config: SubscriberConfig,
        status_client: DashllmWorkerStatusClient,
        kv_event_control_client: DashllmKvEventControlClient,
    ) -> None:
        self._config = config
        self._status_client = status_client
        self._kv_event_control_client = kv_event_control_client
        self._closed = False

    async def watch_liveness(self) -> AsyncGenerator[LivenessEvent, None]:
        """Poll engine liveness via gRPC GetWorkerStatus."""

        was_healthy: bool | None = None
        consecutive_failures = 0

        while True:
            probe_started_at = time.monotonic()
            probe_result = "alive"
            try:
                status = await self._status_client.get_worker_status(
                    self._config.engine_kvcache_worker_status_timeout_ms / 1000
                )
                event = (
                    LivenessEvent.HEALTHY if status.alive else LivenessEvent.UNHEALTHY
                )
                probe_result = "alive" if status.alive else "dead"
                if not status.alive and was_healthy is not False:
                    logger.warning(
                        "engine health probe reported not alive",
                        step="engine_health",
                        tags={
                            "target": self._config.engine_grpc_endpoint,
                        },
                    )
            except asyncio.CancelledError:
                raise
            except grpc.aio.AioRpcError as exc:
                event = LivenessEvent.UNHEALTHY
                code = exc.code()
                probe_result = (
                    "timeout"
                    if code is not None and code.name == "DEADLINE_EXCEEDED"
                    else "rpc_error"
                )
                if was_healthy is not False:
                    logger.warning(
                        "engine health probe failed",
                        step="engine_health",
                        tags={
                            "error": type(exc).__name__,
                            "code": code.name if code is not None else "UNKNOWN",
                            "details": exc.details(),
                            "target": self._config.engine_grpc_endpoint,
                        },
                    )
            except Exception as exc:
                event = LivenessEvent.UNHEALTHY
                probe_result = "rpc_error"
                if was_healthy is not False:
                    logger.warning(
                        "engine health probe failed unexpectedly",
                        step="engine_health",
                        tags={
                            "error": exc.__class__.__name__,
                            "message": str(exc),
                            "target": self._config.engine_grpc_endpoint,
                        },
                    )

            report_engine_probe(
                result=probe_result,
                latency_ms=(time.monotonic() - probe_started_at) * 1000,
            )

            is_healthy = event is LivenessEvent.HEALTHY

            if not is_healthy:
                consecutive_failures += 1
                if was_healthy is False and (
                    consecutive_failures % _HEALTH_LOG_SUPPRESS_INTERVAL == 0
                ):
                    logger.warning(
                        "engine still unhealthy",
                        step="engine_health",
                        tags={
                            "consecutive_failures": consecutive_failures,
                            "target": self._config.engine_grpc_endpoint,
                        },
                    )
            else:
                if was_healthy is False:
                    logger.info(
                        "engine health recovered",
                        step="engine_health",
                        tags={
                            "consecutive_failures_before_recovery": (
                                consecutive_failures
                            ),
                            "target": self._config.engine_grpc_endpoint,
                        },
                    )
                consecutive_failures = 0

            was_healthy = is_healthy
            yield event
            await asyncio.sleep(self._config.engine_health_interval_s)

    @property
    def _engine_kind(self) -> str:
        return "vllm"

    async def fetch_kv_event_bootstrap(self) -> KvEventBootstrap:
        """Fetch and validate bootstrap with bounded exponential backoff.

        Retryable failures (transport errors and explicit UNAVAILABLE responses)
        exhaust the bounded per-attempt retry policy, then raise
        :class:`MetadataTemporarilyUnavailable`. Non-retryable failures (any
        other explicit error code, or a malformed response) raise
        :class:`MetadataProtocolError` immediately. A successful response returns
        :class:`KvEventBootstrap`; an empty component tuple is valid.
        """

        max_retries = self._config.engine_kv_event_bootstrap_max_retries
        delay = _METADATA_RETRY_BASE_S
        for attempt in range(1, max_retries + 1):
            try:
                payload = (
                    await self._kv_event_control_client.get_kv_event_bootstrap_info(
                        self._config.engine_kv_event_bootstrap_timeout_ms / 1000
                    )
                )
                if (
                    payload.err_code
                    == engine_service_rpc_pb2.KV_EVENT_BOOTSTRAP_UNAVAILABLE
                ):
                    raise MetadataTemporarilyUnavailable(
                        f"DashLLM bootstrap unavailable: {payload.err_msg}"
                    )
                metadata = parse_kv_event_bootstrap(
                    payload,
                    expected_engine_kind=self._engine_kind,
                    require_incremental_transport=(
                        self._config.incremental_kv_event_pipeline_enabled
                    ),
                )
            except asyncio.CancelledError:
                raise
            except MetadataProtocolError:
                logger.warning(
                    "received non-retryable kv event bootstrap response",
                    step="kv_metadata",
                    tags={
                        "target": self._config.engine_kv_event_control_uds_path,
                        "attempt": attempt,
                    },
                    exc_info=True,
                )
                raise
            except Exception as exc:
                if attempt >= max_retries:
                    logger.warning(
                        "failed to fetch kv event bootstrap; max retries exhausted",
                        step="kv_metadata",
                        tags={
                            "error": type(exc).__name__,
                            "message": str(exc),
                            "target": self._config.engine_kv_event_control_uds_path,
                            "attempts": attempt,
                            "max_retries": max_retries,
                        },
                    )
                    raise MetadataTemporarilyUnavailable(
                        f"kv event bootstrap unavailable after "
                        f"{max_retries} attempts: {exc}"
                    ) from exc
                logger.warning(
                    "failed to fetch kv event bootstrap; retrying",
                    step="kv_metadata",
                    tags={
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "target": self._config.engine_kv_event_control_uds_path,
                        "retry_in_s": delay,
                        "attempt": attempt,
                        "max_retries": max_retries,
                    },
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, _METADATA_RETRY_MAX_S)
                continue
            if logger.is_debug_enabled():
                logger.debug(
                    "fetched kv event bootstrap",
                    step="kv_metadata",
                    tags={
                        "component_count": len(metadata.components),
                        "components": [
                            {
                                "component_id": component.component_id,
                                "kind": component.component_kind,
                                "block_size": (component.geometry.block_size_tokens),
                            }
                            for component in metadata.components
                        ],
                    },
                )
            return metadata
        raise AssertionError("unreachable")

    async def close(self) -> None:
        """Mark control as closed. The gRPC client is owned by the adapter."""

        if self._closed:
            return
        self._closed = True
