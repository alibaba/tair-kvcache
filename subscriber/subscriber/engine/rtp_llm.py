from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import grpc
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import Message

from subscriber import logger
from subscriber.config import SubscriberConfig
from subscriber.engine.base import AbstractEngineAdapter, EngineEventBatch
from subscriber.health.events import LivenessEvent
from subscriber.metrics import StageTimer
from subscriber.types import AllBlocksCleared, BlockRemoved, BlockStored, KVEventBatch

_GET_CACHE_STATUS_METHOD = "/RpcService/GetCacheStatus"
_MAX_RECEIVE_MESSAGE_LENGTH = 256 * 1024 * 1024
_RTP_MEDIUM = "hbm"


@dataclass(frozen=True)
class CacheSnapshot:
    """One complete cache-key view collected from all configured RTP workers."""

    keys: frozenset[int]
    block_size: int
    version: int


@dataclass(frozen=True)
class CacheDiff:
    """KVCM mutations required to converge from the acknowledged baseline."""

    added: tuple[int, ...]
    removed: tuple[int, ...]

    @property
    def empty(self) -> bool:
        return not self.added and not self.removed


class CacheDiffTracker:
    """Track the KVCM-acknowledged key set and debounce removals."""

    def __init__(self, deletion_confirmations: int = 2) -> None:
        if deletion_confirmations < 1:
            raise ValueError("deletion_confirmations must be >= 1")
        self._deletion_confirmations = deletion_confirmations
        self._acknowledged: set[int] = set()
        self._possibly_reported: set[int] = set()
        self._missing_counts: dict[int, int] = {}

    @property
    def acknowledged_keys(self) -> frozenset[int]:
        return frozenset(self._acknowledged)

    def plan(
        self,
        observed: frozenset[int],
        *,
        force_full_add: bool = False,
    ) -> CacheDiff:
        for key in observed:
            self._missing_counts.pop(key, None)

        deletion_domain = self._acknowledged | self._possibly_reported
        for key in deletion_domain - observed:
            self._missing_counts[key] = self._missing_counts.get(key, 0) + 1

        added = observed if force_full_add else observed - self._acknowledged
        removed = {
            key
            for key, count in self._missing_counts.items()
            if key in deletion_domain and count >= self._deletion_confirmations
        }
        return CacheDiff(tuple(sorted(added)), tuple(sorted(removed)))

    def mark_uncertain(self, diff: CacheDiff) -> None:
        """Remember adds that may have succeeded before a batch-group failure."""

        self._possibly_reported.update(set(diff.added) - self._acknowledged)

    def commit(self, diff: CacheDiff) -> None:
        self._acknowledged.update(diff.added)
        self._acknowledged.difference_update(diff.removed)
        self._possibly_reported.difference_update(diff.added)
        self._possibly_reported.difference_update(diff.removed)
        for key in diff.removed:
            self._missing_counts.pop(key, None)

    def reset(self) -> None:
        self._acknowledged.clear()
        self._possibly_reported.clear()
        self._missing_counts.clear()


def _add_field(
    message: descriptor_pb2.DescriptorProto,
    *,
    name: str,
    number: int,
    field_type: descriptor_pb2.FieldDescriptorProto.Type.ValueType,
    label: descriptor_pb2.FieldDescriptorProto.Label.ValueType = (
        descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    ),
    type_name: str = "",
) -> None:
    field = message.field.add()
    field.name = name
    field.number = number
    field.type = field_type
    field.label = label
    if type_name:
        field.type_name = type_name


def _build_cache_status_message_types() -> tuple[type[Message], type[Message]]:
    """Build the stable RTP CacheStatus protobuf subset without RTP imports."""

    file_descriptor = descriptor_pb2.FileDescriptorProto(
        name="rtp_cache_status.proto",
        syntax="proto3",
    )
    version = file_descriptor.message_type.add(name="CacheVersionPB")
    _add_field(
        version,
        name="latest_cache_version",
        number=1,
        field_type=descriptor_pb2.FieldDescriptorProto.TYPE_INT64,
    )
    _add_field(
        version,
        name="need_cache_keys",
        number=2,
        field_type=descriptor_pb2.FieldDescriptorProto.TYPE_BOOL,
    )

    status = file_descriptor.message_type.add(name="CacheStatusPB")
    for name, number in (
        ("available_kv_cache", 1),
        ("total_kv_cache", 2),
        ("block_size", 3),
        ("version", 4),
    ):
        _add_field(
            status,
            name=name,
            number=number,
            field_type=descriptor_pb2.FieldDescriptorProto.TYPE_INT64,
        )
    cache_keys_entry = status.nested_type.add(name="CacheKeysEntry")
    cache_keys_entry.options.map_entry = True
    _add_field(
        cache_keys_entry,
        name="key",
        number=1,
        field_type=descriptor_pb2.FieldDescriptorProto.TYPE_INT64,
    )
    _add_field(
        cache_keys_entry,
        name="value",
        number=2,
        field_type=descriptor_pb2.FieldDescriptorProto.TYPE_BOOL,
    )
    _add_field(
        status,
        name="cache_keys",
        number=5,
        field_type=descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        type_name=".CacheStatusPB.CacheKeysEntry",
    )

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_descriptor)
    return (
        message_factory.GetMessageClass(pool.FindMessageTypeByName("CacheVersionPB")),
        message_factory.GetMessageClass(pool.FindMessageTypeByName("CacheStatusPB")),
    )


CacheVersionPB, CacheStatusPB = _build_cache_status_message_types()


class CacheStatusSource(Protocol):
    async def fetch_snapshot(self) -> CacheSnapshot:
        """Return a complete cache view or raise without returning partial data."""

    async def close(self) -> None:
        """Release transport resources."""


class RtpGrpcCacheStatusSource:
    """Fetch and merge full GetCacheStatus responses from RTP DP workers."""

    def __init__(self, endpoints: tuple[str, ...], timeout_s: float) -> None:
        if not endpoints:
            raise ValueError("at least one RTP gRPC endpoint is required")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self._endpoints = endpoints
        self._timeout_s = timeout_s
        self._channels: list[Any] = []
        self._calls: list[Any] = []

    def _ensure_connections(self) -> None:
        if self._calls:
            return
        for endpoint in self._endpoints:
            channel = grpc.aio.insecure_channel(
                endpoint,
                options=[
                    ("grpc.max_receive_message_length", _MAX_RECEIVE_MESSAGE_LENGTH)
                ],
            )
            call: Any = channel.unary_unary(
                _GET_CACHE_STATUS_METHOD,
                request_serializer=lambda request: request.SerializeToString(),
                response_deserializer=CacheStatusPB.FromString,
            )
            self._channels.append(channel)
            self._calls.append(call)

    async def _fetch_one(self, call: Any) -> Any:
        request = CacheVersionPB(
            latest_cache_version=-1,
            need_cache_keys=True,
        )
        return await call(request, timeout=self._timeout_s)

    async def fetch_snapshot(self) -> CacheSnapshot:
        self._ensure_connections()
        results = await asyncio.gather(
            *(self._fetch_one(call) for call in self._calls),
            return_exceptions=True,
        )
        failures = [
            (endpoint, result)
            for endpoint, result in zip(self._endpoints, results, strict=True)
            if isinstance(result, BaseException)
        ]
        if failures:
            details = ", ".join(
                f"{endpoint}: {failure.__class__.__name__}: {failure}"
                for endpoint, failure in failures
            )
            raise RuntimeError(
                f"RTP GetCacheStatus failed for configured endpoint(s): {details}"
            ) from failures[0][1]
        responses = [
            result for result in results if not isinstance(result, BaseException)
        ]
        block_sizes = {int(response.block_size) for response in responses}
        if len(block_sizes) != 1:
            raise RuntimeError(
                f"RTP workers returned inconsistent block sizes: {sorted(block_sizes)}"
            )
        block_size = block_sizes.pop()
        if block_size <= 0:
            raise RuntimeError(
                f"RTP workers returned invalid cache block size: {block_size}"
            )

        keys: set[int] = set()
        versions: list[int] = []
        for response in responses:
            keys.update(
                int(key) for key, present in response.cache_keys.items() if present
            )
            versions.append(int(response.version))
        return CacheSnapshot(
            keys=frozenset(keys),
            block_size=block_size,
            version=min(versions) if versions else -1,
        )

    async def close(self) -> None:
        await asyncio.gather(*(channel.close() for channel in self._channels))
        self._channels.clear()
        self._calls.clear()


@AbstractEngineAdapter.register("rtp_llm")
class RtpLlmAdapter(AbstractEngineAdapter):
    """RTP-LLM adapter based on acknowledged full-snapshot diffs."""

    def __init__(
        self,
        config: SubscriberConfig,
        *,
        source: CacheStatusSource | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._source = source or RtpGrpcCacheStatusSource(
            config.rtp_endpoint_list,
            config.rtp_rpc_timeout_s,
        )
        self._clock = clock
        self._tracker = CacheDiffTracker(config.rtp_deletion_confirmations)
        self._event_queue: asyncio.Queue[EngineEventBatch] = asyncio.Queue()
        self._liveness_queue: asyncio.Queue[LivenessEvent] = asyncio.Queue()
        self._poll_task: asyncio.Task[None] | None = None
        self._close_lock = asyncio.Lock()
        self._healthy_streak = 0
        self._force_full_add = True
        self._cold_reset_pending = config.rtp_reset_on_start
        self._next_full_refresh_at = 0.0
        self._block_size: int | None = None
        self._delivery_done: asyncio.Event | None = None

    @property
    def tracker(self) -> CacheDiffTracker:
        return self._tracker

    def _ensure_poller(self) -> None:
        if self._poll_task is None:
            self._poll_task = asyncio.create_task(
                self._poll_loop(),
                name="rtp-cache-status-poller",
            )

    async def _stop_poller(self) -> None:
        async with self._close_lock:
            task = self._poll_task
            if task is None:
                return
            self._poll_task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await self._source.close()

    async def _poll_loop(self) -> None:
        while True:
            timer = StageTimer()
            try:
                snapshot = await self._source.fetch_snapshot()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._healthy_streak = 0
                await self._liveness_queue.put(LivenessEvent.UNHEALTHY)
                logger.warning(
                    "RTP GetCacheStatus full pull failed",
                    step="rtp_snapshot_fetch",
                    tags={
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
                await asyncio.sleep(self._config.rtp_poll_interval_s)
                continue

            timer.mark("rtp_snapshot_fetch")
            try:
                self._validate_snapshot(snapshot)
                self._healthy_streak += 1
                event = (
                    self._event_for_snapshot(snapshot, timer)
                    if self._healthy_streak > 1
                    else None
                )
            except Exception as exc:
                self._healthy_streak = 0
                await self._liveness_queue.put(LivenessEvent.UNHEALTHY)
                logger.warning(
                    "failed to diff RTP cache snapshot",
                    step="rtp_snapshot_diff",
                    tags={
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
                await asyncio.sleep(self._config.rtp_poll_interval_s)
                continue

            await self._liveness_queue.put(LivenessEvent.HEALTHY)
            if self._healthy_streak == 1:
                # Let the health coordinator open or recover the send epoch
                # before an authoritative snapshot reaches the forwarding gate.
                await asyncio.sleep(self._config.rtp_poll_interval_s)
                continue
            if event is not None:
                await self._event_queue.put(event)
                delivery_done = self._delivery_done
                if delivery_done is not None:
                    await delivery_done.wait()
                    if self._delivery_done is delivery_done:
                        self._delivery_done = None
            await asyncio.sleep(self._config.rtp_poll_interval_s)

    def _validate_snapshot(self, snapshot: CacheSnapshot) -> None:
        """Reject inconsistent cache metadata before reporting engine health."""

        if snapshot.block_size != self._config.block_size:
            raise RuntimeError(
                "RTP cache block size does not match KVCM registration: "
                f"configured={self._config.block_size}, "
                f"observed={snapshot.block_size}"
            )
        if self._block_size is None:
            self._block_size = snapshot.block_size
        elif self._block_size != snapshot.block_size:
            raise RuntimeError(
                "RTP cache block size changed while subscriber was running: "
                f"{self._block_size} -> {snapshot.block_size}"
            )

    def _event_for_snapshot(
        self,
        snapshot: CacheSnapshot,
        timer: StageTimer,
    ) -> EngineEventBatch | None:
        self._validate_snapshot(snapshot)

        now = self._clock()
        force_full_add = self._force_full_add or now >= self._next_full_refresh_at
        diff = self._tracker.plan(
            snapshot.keys,
            force_full_add=force_full_add,
        )
        if diff.empty and not self._cold_reset_pending:
            return None

        events: list[BlockStored | BlockRemoved | AllBlocksCleared] = []
        if self._cold_reset_pending:
            events.append(AllBlocksCleared())
        if diff.added:
            events.append(
                BlockStored(
                    block_hashes=list(diff.added),
                    parent_block_hash=None,
                    token_ids=[],
                    block_size=snapshot.block_size,
                    lora_id=None,
                    medium=_RTP_MEDIUM,
                    lora_name=None,
                )
            )
        if diff.removed:
            events.append(
                BlockRemoved(
                    block_hashes=list(diff.removed),
                    medium=_RTP_MEDIUM,
                )
            )
        timer.mark("rtp_snapshot_diff")
        delivery_done = asyncio.Event()
        self._delivery_done = delivery_done

        async def on_delivery(delivered: bool) -> None:
            try:
                if delivered:
                    self._tracker.commit(diff)
                    self._cold_reset_pending = False
                    if force_full_add:
                        self._force_full_add = False
                        self._next_full_refresh_at = (
                            self._clock() + self._config.rtp_full_refresh_interval_s
                        )
                else:
                    self._tracker.mark_uncertain(diff)
            finally:
                delivery_done.set()

        return EngineEventBatch(
            batches=[KVEventBatch(ts=time.time(), events=events)],
            timer=timer,
            on_delivery=on_delivery,
        )

    async def subscribe_kv_events(self) -> AsyncGenerator[EngineEventBatch, None]:
        self._ensure_poller()
        try:
            while True:
                yield await self._event_queue.get()
        finally:
            await self._stop_poller()

    async def watch_liveness(self) -> AsyncGenerator[LivenessEvent, None]:
        self._ensure_poller()
        try:
            while True:
                yield await self._liveness_queue.get()
        finally:
            await self._stop_poller()

    async def reset_generation_state(self) -> None:
        self._tracker.reset()
        self._force_full_add = True
        self._next_full_refresh_at = 0.0
        # The coordinator already reported HOST_DOWN for this recovery path.
        self._cold_reset_pending = False

    def map_medium(self, medium: str | None) -> str:
        return _RTP_MEDIUM if medium == _RTP_MEDIUM else ""

    def supported_mediums(self) -> list[str]:
        return [_RTP_MEDIUM]

    def storage_type(self) -> str:
        return "ST_EVENT_REPORT"

    def location_spec_name(self, block_size: int) -> str:
        return f"rtp_llm_{block_size}"

    def location_uri(self, host_ip_port: str, medium: str) -> str:
        return f"rtp-llm://{host_ip_port}/{medium}"
