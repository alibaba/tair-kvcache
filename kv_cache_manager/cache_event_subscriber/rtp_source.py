from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from .models import BlockRecord, EngineUpdate

_RPC_METHOD = "/RpcService/GetCacheStatus"
_STORED = 0
_REMOVED = 1


class RtpTransport(Protocol):
    async def call(self, endpoint: str, request: Any, timeout_s: float) -> Any: ...

    async def close(self) -> None: ...


@dataclass
class _EndpointState:
    generation: int
    cursor: int
    blocks: dict[int, set[int]]

    def copy(self) -> "_EndpointState":
        return _EndpointState(
            generation=self.generation,
            cursor=self.cursor,
            blocks={key: set(groups) for key, groups in self.blocks.items()},
        )


@dataclass(frozen=True)
class _CommitToken:
    states: tuple[_EndpointState, ...]
    aggregate: dict[int, frozenset[int]]
    full_snapshot: bool


class RtpCacheSource:
    """Transactional RTP cache changefeed consumer.

    ``prepare`` reconstructs candidate state without advancing durable cursors.
    The caller must invoke ``commit`` only after KVCM acknowledges the update.
    """

    def __init__(
        self,
        endpoints: Sequence[str],
        *,
        timeout_s: float = 1.0,
        page_size: int = 4096,
        max_pages: int = 1024,
        full_refresh_interval_s: float = 300.0,
        medium: str = "hbm",
        expected_block_size: int | None = None,
        cache_group_count: int = 1,
        transport: RtpTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not endpoints:
            raise ValueError("at least one RTP endpoint is required")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("RTP endpoints must be unique")
        if timeout_s <= 0 or page_size <= 0 or max_pages <= 0:
            raise ValueError("timeouts and pagination limits must be positive")
        if cache_group_count <= 0:
            raise ValueError("cache_group_count must be positive")
        if expected_block_size is not None and expected_block_size <= 0:
            raise ValueError("expected_block_size must be positive")
        self._endpoints = tuple(endpoints)
        self._timeout_s = timeout_s
        self._page_size = page_size
        self._max_pages = max_pages
        self._full_refresh_interval_s = full_refresh_interval_s
        self._medium = medium
        self._expected_block_size = expected_block_size
        self._cache_group_count = cache_group_count
        self._clock = clock
        self._request_type, response_type = _build_message_types()
        self._transport = transport or _GrpcTransport(response_type)
        self._states = tuple(_EndpointState(0, -1, {}) for _ in self._endpoints)
        self._aggregate: dict[int, frozenset[int]] = {}
        self._initialized = False
        self._next_full_refresh = 0.0
        self._pending: _CommitToken | None = None

    async def prepare(self) -> EngineUpdate:
        if self._pending is not None:
            raise RuntimeError("previous RTP update has not been committed or aborted")
        force_snapshot = not self._initialized or self._clock() >= self._next_full_refresh
        results = await asyncio.gather(
            *(
                self._fetch_endpoint(endpoint, state, force_snapshot)
                for endpoint, state in zip(self._endpoints, self._states, strict=True)
            )
        )
        states = tuple(state for state, _ in results)
        source_reset = any(reset for _, reset in results)
        aggregate = _aggregate_states(states)
        full_snapshot = force_snapshot or source_reset
        token = _CommitToken(states, aggregate, full_snapshot)
        self._pending = token

        if full_snapshot:
            blocks = tuple(
                BlockRecord(key, self._medium, tuple(sorted(groups)))
                for key, groups in sorted(aggregate.items())
            )
            return EngineUpdate(True, blocks=blocks, commit_token=token)

        changed = {
            key
            for key, groups in aggregate.items()
            if self._aggregate.get(key) != groups
        }
        removed = set(self._aggregate) - set(aggregate)
        return EngineUpdate(
            False,
            upserts=tuple(
                BlockRecord(key, self._medium, tuple(sorted(aggregate[key])))
                for key in sorted(changed)
            ),
            removals=tuple(
                BlockRecord(key, self._medium, tuple(sorted(self._aggregate[key])))
                for key in sorted(removed)
            ),
            commit_token=token,
        )

    def commit(self, update: EngineUpdate) -> None:
        if update.commit_token is not self._pending:
            raise RuntimeError("RTP update is stale or belongs to another source")
        token = self._pending
        assert token is not None
        self._states = token.states
        self._aggregate = token.aggregate
        self._initialized = True
        if token.full_snapshot:
            self._next_full_refresh = self._clock() + self._full_refresh_interval_s
        self._pending = None

    def abort(self, update: EngineUpdate) -> None:
        if update.commit_token is not self._pending:
            raise RuntimeError("RTP update is stale or belongs to another source")
        self._pending = None

    async def close(self) -> None:
        await self._transport.close()

    async def _fetch_endpoint(
        self, endpoint: str, committed: _EndpointState, force_snapshot: bool
    ) -> tuple[_EndpointState, bool]:
        candidate = committed.copy()
        reset_seen = False
        for _ in range(self._max_pages):
            request = self._request_type(
                latest_cache_version=candidate.cursor,
                need_cache_keys=True,
                need_cache_events=True,
                max_cache_events=self._page_size,
                force_cache_event_snapshot=force_snapshot and not reset_seen,
                cache_event_generation=candidate.generation,
            )
            response = await self._transport.call(endpoint, request, self._timeout_s)
            if self._expected_block_size is not None:
                observed_block_size = int(getattr(response, "block_size", 0))
                if observed_block_size != self._expected_block_size:
                    raise RuntimeError(
                        "RTP cache block size does not match KVCM registration: "
                        f"configured={self._expected_block_size}, "
                        f"observed={observed_block_size}, endpoint={endpoint}"
                    )
            protocol = int(getattr(response, "cache_event_protocol_version", 0))
            if protocol == 0:
                if self._cache_group_count != 1:
                    raise RuntimeError(
                        "legacy RTP cache status has no cache-group identity; "
                        "cache_group_count must be 1 or RTP cache-event protocol v2 is required"
                    )
                legacy_version = int(getattr(response, "version", -1))
                candidate.blocks = {
                    int(key): {0}
                    for key, present in getattr(response, "cache_keys", {}).items()
                    if present
                }
                candidate.cursor = legacy_version
                candidate.generation = 0
                # A legacy RTP endpoint only exposes a full key map. Treat it
                # as authoritative input on every poll, but force an outgoing
                # full snapshot only for reconciliation or version rollback;
                # otherwise compute and report the normal aggregate diff.
                return candidate, force_snapshot or legacy_version < committed.cursor

            generation = int(getattr(response, "cache_event_generation", 0))
            reset = bool(getattr(response, "cache_event_reset_required", False))
            if candidate.generation and generation and generation != candidate.generation and not reset:
                raise RuntimeError(
                    f"RTP generation changed without snapshot recovery at {endpoint}"
                )
            if reset:
                candidate.blocks = {
                    int(entry.cache_key): _validated_groups(
                        entry.group_ids, self._cache_group_count
                    )
                    for entry in response.cache_event_snapshot
                }
                candidate.cursor = _next_cursor(response, protocol, candidate.cursor)
                candidate.generation = generation
                reset_seen = True
            else:
                expected = candidate.cursor + 1
                for event in response.cache_events:
                    version = int(event.version)
                    if version != expected:
                        raise RuntimeError(
                            f"RTP event gap at {endpoint}: expected={expected}, got={version}"
                        )
                    key = int(event.cache_key)
                    groups = _validated_groups(
                        event.group_ids, self._cache_group_count
                    )
                    if int(event.event_type) == _STORED:
                        candidate.blocks.setdefault(key, set()).update(groups)
                    elif int(event.event_type) == _REMOVED:
                        remaining = candidate.blocks.get(key)
                        if remaining is not None:
                            remaining.difference_update(groups)
                            if not remaining:
                                candidate.blocks.pop(key, None)
                    else:
                        raise RuntimeError(f"unknown RTP cache event type at {endpoint}")
                    expected += 1
                candidate.cursor = _next_cursor(response, protocol, candidate.cursor)
                candidate.generation = generation

            if not bool(getattr(response, "cache_event_has_more", False)):
                head = int(getattr(response, "cache_event_version", candidate.cursor))
                if candidate.cursor != head:
                    raise RuntimeError(
                        f"RTP page ended before head at {endpoint}: cursor={candidate.cursor}, head={head}"
                    )
                return candidate, reset_seen
            if not response.cache_events or candidate.cursor < 0:
                raise RuntimeError(f"RTP pagination made no progress at {endpoint}")
            force_snapshot = False
        raise RuntimeError(f"RTP pagination exceeded {self._max_pages} pages at {endpoint}")


def _next_cursor(response: Any, protocol: int, previous: int) -> int:
    if protocol >= 2:
        return int(response.next_cache_event_version)
    events = getattr(response, "cache_events", ())
    if events:
        return int(events[-1].version)
    if bool(getattr(response, "cache_event_reset_required", False)):
        return int(response.cache_event_version)
    return previous


def _aggregate_states(states: Sequence[_EndpointState]) -> dict[int, frozenset[int]]:
    aggregate: dict[int, set[int]] = {}
    for state in states:
        for key, groups in state.blocks.items():
            aggregate.setdefault(key, set()).update(groups)
    return {key: frozenset(groups) for key, groups in aggregate.items()}


def _validated_groups(values: Sequence[int], cache_group_count: int) -> set[int]:
    groups = {int(group) for group in values}
    if not groups:
        raise RuntimeError("RTP cache event has no cache group ids")
    invalid = sorted(group for group in groups if group < 0 or group >= cache_group_count)
    if invalid:
        raise RuntimeError(f"RTP returned unregistered cache group ids: {invalid}")
    return groups


class _GrpcTransport:
    def __init__(self, response_type: type[Any]) -> None:
        import grpc

        self._grpc = grpc
        self._response_type = response_type
        self._channels: dict[str, Any] = {}
        self._calls: dict[str, Any] = {}

    async def call(self, endpoint: str, request: Any, timeout_s: float) -> Any:
        call = self._calls.get(endpoint)
        if call is None:
            channel = self._grpc.aio.insecure_channel(
                endpoint, options=[("grpc.max_receive_message_length", 256 * 1024 * 1024)]
            )
            self._channels[endpoint] = channel
            call = channel.unary_unary(
                _RPC_METHOD,
                request_serializer=lambda message: message.SerializeToString(),
                response_deserializer=self._response_type.FromString,
            )
            self._calls[endpoint] = call
        return await call(request, timeout=timeout_s)

    async def close(self) -> None:
        await asyncio.gather(*(channel.close() for channel in self._channels.values()))
        self._channels.clear()
        self._calls.clear()


def _add_field(
    message: descriptor_pb2.DescriptorProto,
    name: str,
    number: int,
    field_type: int,
    *,
    repeated: bool = False,
    type_name: str = "",
) -> None:
    field = message.field.add(name=name, number=number, type=field_type)
    field.label = (
        descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
        if repeated
        else descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    )
    if type_name:
        field.type_name = type_name


def _build_message_types() -> tuple[type[Any], type[Any]]:
    """Build the stable wire subset so tair-kvcache need not import RTP code."""

    fd = descriptor_pb2.FileDescriptorProto(name="rtp_cache_event_subset.proto", syntax="proto3")
    request = fd.message_type.add(name="CacheVersionPB")
    for name, number, field_type in (
        ("latest_cache_version", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
        ("need_cache_keys", 2, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
        ("need_cache_events", 3, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
        ("max_cache_events", 4, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32),
        ("force_cache_event_snapshot", 5, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
        ("cache_event_generation", 6, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64),
    ):
        _add_field(request, name, number, field_type)

    event_enum = fd.enum_type.add(name="CacheEventTypePB")
    event_enum.value.add(name="CACHE_EVENT_STORED", number=0)
    event_enum.value.add(name="CACHE_EVENT_REMOVED", number=1)
    reason_enum = fd.enum_type.add(name="CacheEventSnapshotReasonPB")
    for number, name in enumerate(
        (
            "CACHE_EVENT_SNAPSHOT_NONE",
            "CACHE_EVENT_SNAPSHOT_FORCED",
            "CACHE_EVENT_SNAPSHOT_HISTORY_GAP",
            "CACHE_EVENT_SNAPSHOT_FUTURE_CURSOR",
            "CACHE_EVENT_SNAPSHOT_GENERATION_MISMATCH",
            "CACHE_EVENT_SNAPSHOT_INVALID_CURSOR",
        )
    ):
        reason_enum.value.add(name=name, number=number)

    event = fd.message_type.add(name="CacheEventPB")
    _add_field(event, "version", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _add_field(
        event,
        "event_type",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
        type_name=".CacheEventTypePB",
    )
    _add_field(event, "cache_key", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _add_field(event, "group_ids", 4, descriptor_pb2.FieldDescriptorProto.TYPE_INT32, repeated=True)

    snapshot = fd.message_type.add(name="CacheSnapshotEntryPB")
    _add_field(snapshot, "cache_key", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _add_field(snapshot, "group_ids", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT32, repeated=True)

    status = fd.message_type.add(name="CacheStatusPB")
    cache_map = status.nested_type.add(name="CacheKeysEntry")
    cache_map.options.map_entry = True
    _add_field(cache_map, "key", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _add_field(cache_map, "value", 2, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)
    scalar_fields = (
        ("available_kv_cache", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
        ("total_kv_cache", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
        ("block_size", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
        ("version", 4, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
    )
    for name, number, field_type in scalar_fields:
        _add_field(status, name, number, field_type)
    _add_field(
        status,
        "cache_keys",
        5,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        repeated=True,
        type_name=".CacheStatusPB.CacheKeysEntry",
    )
    for name, number, field_type in (
        ("cache_event_version", 6, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
        ("oldest_cache_event_version", 7, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
        ("cache_event_reset_required", 8, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
        ("cache_event_has_more", 9, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
    ):
        _add_field(status, name, number, field_type)
    _add_field(
        status,
        "cache_events",
        10,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        repeated=True,
        type_name=".CacheEventPB",
    )
    _add_field(
        status,
        "cache_event_snapshot",
        11,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        repeated=True,
        type_name=".CacheSnapshotEntryPB",
    )
    _add_field(status, "cache_event_protocol_version", 12, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    _add_field(status, "cache_event_generation", 13, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    _add_field(
        status,
        "cache_event_snapshot_reason",
        14,
        descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
        type_name=".CacheEventSnapshotReasonPB",
    )
    _add_field(status, "next_cache_event_version", 15, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)

    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    def message_class(name: str) -> type[Any]:
        descriptor = pool.FindMessageTypeByName(name)
        if hasattr(message_factory, "GetMessageClass"):
            return message_factory.GetMessageClass(descriptor)
        return message_factory.MessageFactory(pool).GetPrototype(descriptor)

    return (
        message_class("CacheVersionPB"),
        message_class("CacheStatusPB"),
    )
