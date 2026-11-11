"""Engine-neutral polling lifecycle for full KV cache snapshots."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from typing import cast

import grpc
import msgspec

from subscriber import logger
from subscriber.config import SubscriberConfig
from subscriber.engine.base import EngineEventBatch
from subscriber.engine.kv_event_control_client import DashllmKvEventControlClient
from subscriber.metrics import BatchTelemetry
from subscriber.proto import engine_service_rpc_pb2
from subscriber.trace import generate_trace_id
from subscriber.types import BlockSnapshot, BlockSnapshotItem, KVEventBatch

SnapshotDecoder = Callable[[bytes], list[BlockSnapshotItem]]


class SnapshotSchemaError(ValueError):
    """An engine snapshot payload violates its accepted engine schema."""


class GrpcSnapshotSource:
    """Poll DashLLM's UDS snapshot RPC and apply an engine-specific decoder.

    The gRPC client lifecycle is owned by the composing adapter. Generation
    tracking discards a response when an engine reset spans the in-flight poll.
    """

    def __init__(
        self,
        config: SubscriberConfig,
        grpc_client: DashllmKvEventControlClient,
        decoder: SnapshotDecoder,
    ) -> None:
        self._config = config
        self._grpc_client = grpc_client
        self._decoder = decoder
        self._generation = 0
        self._snapshot_requested = asyncio.Event()
        self._signal_warned = False

    def request_immediate_snapshot(self) -> None:
        """Wake the polling loop; repeated pending signals coalesce."""
        if self._snapshot_requested.is_set():
            if not self._signal_warned:
                logger.warning(
                    "snapshot signal coalesced: previous signal not yet consumed",
                    step="grpc_snapshot",
                )
                self._signal_warned = True
        else:
            self._signal_warned = False
        if logger.is_debug_enabled():
            logger.debug("snapshot signal received", step="grpc_snapshot")
        self._snapshot_requested.set()

    async def _wait_interval(self, interval_s: float) -> None:
        try:
            await asyncio.wait_for(self._snapshot_requested.wait(), timeout=interval_s)
        except TimeoutError:
            pass
        self._snapshot_requested.clear()
        self._signal_warned = False

    @staticmethod
    def _dropped_batch(telemetry: BatchTelemetry, trace_id: str) -> EngineEventBatch:
        return EngineEventBatch(batches=[], telemetry=telemetry, trace_id=trace_id)

    async def subscribe(self) -> AsyncGenerator[EngineEventBatch, None]:
        """Yield decoded snapshots and report every failed/discarded poll."""
        interval_s = self._config.engine_snapshot_full_sync_interval_s
        while True:
            telemetry = BatchTelemetry(pipeline="snapshot")
            trace_id = generate_trace_id()
            generation = self._generation

            try:
                response = cast(
                    engine_service_rpc_pb2.KvCacheBlockListPB,
                    await self._grpc_client.get_all_kv_cache_blocks(
                        timeout_s=(
                            self._config.engine_kvcache_snapshot_timeout_ms / 1000
                        ),
                    ),
                )
            except grpc.aio.AioRpcError as exc:
                telemetry.mark("snapshot_fetch")
                telemetry.mark_dropped("fetch_failed")
                logger.warning(
                    "gRPC snapshot poll failed",
                    step="grpc_snapshot",
                    tags={
                        "code": exc.code().name if exc.code() else "UNKNOWN",
                        "details": exc.details() or "",
                        "trace_id": trace_id,
                    },
                )
                yield self._dropped_batch(telemetry, trace_id)
                await self._wait_interval(interval_s)
                continue
            except Exception as exc:
                telemetry.mark("snapshot_fetch")
                telemetry.mark_dropped("fetch_failed")
                logger.warning(
                    "gRPC snapshot poll unexpected error, will retry",
                    step="grpc_snapshot",
                    tags={
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                        "trace_id": trace_id,
                    },
                )
                yield self._dropped_batch(telemetry, trace_id)
                await self._wait_interval(interval_s)
                continue

            fetch_ms = telemetry.mark("snapshot_fetch") * 1000
            payload_bytes = response.ByteSize()
            telemetry.gauge("full_snapshot_payload_bytes", payload_bytes)

            if generation != self._generation:
                telemetry.mark_dropped("generation_reset")
                yield self._dropped_batch(telemetry, trace_id)
                continue

            if not response.raw_snapshot:
                telemetry.mark_dropped("empty_snapshot")
                yield self._dropped_batch(telemetry, trace_id)
                await self._wait_interval(interval_s)
                continue

            try:
                items = self._decoder(response.raw_snapshot)
            except (msgspec.ValidationError, SnapshotSchemaError) as exc:
                telemetry.mark_dropped("schema_mismatch")
                logger.warning(
                    "snapshot schema mismatch",
                    step="grpc_snapshot",
                    tags={
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                        "trace_id": trace_id,
                    },
                )
                yield self._dropped_batch(telemetry, trace_id)
                await self._wait_interval(interval_s)
                continue
            except msgspec.DecodeError as exc:
                telemetry.mark_dropped("decode_failed")
                logger.warning(
                    "snapshot msgpack decode failed",
                    step="grpc_snapshot",
                    tags={
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                        "trace_id": trace_id,
                    },
                )
                yield self._dropped_batch(telemetry, trace_id)
                await self._wait_interval(interval_s)
                continue

            telemetry.mark("decode")
            batch = KVEventBatch(
                ts=time.time(),
                events=[
                    BlockSnapshot(
                        medium="GPU",
                        block_size=response.block_size,
                        items=items,
                        snapshot_version=response.snapshot_version,
                    )
                ],
            )
            telemetry.mark("snapshot_build")

            if generation != self._generation:
                telemetry.mark_dropped("generation_reset")
                yield self._dropped_batch(telemetry, trace_id)
                continue

            if logger.is_debug_enabled():
                logger.debug(
                    "snapshot poll completed",
                    step="grpc_snapshot",
                    tags={
                        "block_count": len(items),
                        "snapshot_version": response.snapshot_version,
                        "payload_bytes": payload_bytes,
                        "fetch_ms": round(fetch_ms, 2),
                        "trace_id": trace_id,
                    },
                )

            yield EngineEventBatch(
                batches=[batch], telemetry=telemetry, trace_id=trace_id
            )
            await self._wait_interval(interval_s)

    async def reset_generation_state(self) -> None:
        """Advance generation so an in-flight old-engine result is discarded."""
        self._generation += 1

    async def close(self) -> None:
        """No-op; the composing adapter owns the shared gRPC client."""
        return None
