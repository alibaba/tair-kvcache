"""Authoritative KV-event bootstrap and KVCM registration metadata."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, TypeAlias

from subscriber.proto import engine_service_rpc_pb2
from subscriber.types import KvCacheGroupSpec

SettingScalar: TypeAlias = bool | int | float | str | bytes
SettingValue: TypeAlias = SettingScalar | tuple[SettingScalar, ...]
TypedSettings: TypeAlias = tuple[tuple[str, SettingValue], ...]

_SUPPORTED_PROTOCOL_VERSIONS = frozenset({1})
_SUPPORTED_EVENT_SCHEMA_VERSIONS = frozenset({2})
_SUPPORTED_SERIALIZATIONS = frozenset({"msgpack-v1"})


@dataclass(frozen=True)
class EventTransport:
    """Engine-owned ZMQ live/replay transport coordinates."""

    live_endpoint: str
    topic: str
    replay_supported: bool
    replay_endpoint: str
    serialization: str


@dataclass(frozen=True)
class RuntimeTopology:
    """Parallel topology for the event publisher represented by this bootstrap."""

    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_rank: int
    tensor_parallel_rank: int
    pipeline_parallel_rank: int


@dataclass(frozen=True)
class SnapshotCapability:
    """Whether the engine exposes full snapshots and versions them."""

    supported: bool
    versioned: bool


@dataclass(frozen=True)
class CacheGeometry:
    """Engine-neutral geometry for one independently reported cache component."""

    block_size_tokens: int
    page_size_tokens: int | None
    block_count: int | None
    group_payload_size_bytes: int | None
    cache_dtype: str
    element_size_bytes: int | None
    layer_count: int | None
    cache_layout: str
    sliding_window_tokens: int | None
    checkpoint_alignment_tokens: int | None


@dataclass(frozen=True)
class CacheComponent:
    """One stable numeric component identity and its cache interpretation.

    The normalized ``component_id`` maps to vLLM event ``group_idx`` or SGLang
    event ``component_id``; it does not require a shared engine wire field.
    """

    component_id: int
    component_kind: str
    geometry: CacheGeometry
    compatibility_settings: TypedSettings = ()
    diagnostic_settings: TypedSettings = ()


@dataclass(frozen=True)
class VllmEventSchema:
    """vLLM-specific event and cache-compatibility semantics."""

    event_schema_version: int
    use_eagle_pop: bool
    mamba_cache_mode: str
    hash_algorithm: str
    hash_version: str
    cache_settings: TypedSettings = ()


@dataclass(frozen=True)
class SglangEventSchema:
    """SGLang-specific native cache-key and event semantics."""

    event_schema_version: int
    cache_key_mode: str
    native_hash_algorithm: str
    cache_settings: TypedSettings = ()


@dataclass(frozen=True)
class KvEventBootstrap:
    """Validated immutable contract shared by adapters and KVCM translation."""

    protocol_version: int
    engine_kind: str
    event_transport: EventTransport
    runtime_topology: RuntimeTopology
    snapshot: SnapshotCapability
    components: tuple[CacheComponent, ...]
    compatibility_settings: TypedSettings
    diagnostic_settings: TypedSettings
    vllm: VllmEventSchema | None = None
    sglang: SglangEventSchema | None = None

    def to_log_json(self) -> str:
        """Serialize the complete accepted bootstrap for one startup info log."""

        return json.dumps(asdict(self), default=_json_default, sort_keys=True)

    def to_kv_cache_descriptor(self) -> KvCacheDescriptor:
        """Build the existing KVCM-facing descriptor without engine protobufs."""

        groups = tuple(
            KvCacheGroupSpec(
                group_idx=component.component_id,
                kind=component.component_kind,
                block_size=component.geometry.block_size_tokens,
                group_payload_size_bytes=(component.geometry.group_payload_size_bytes),
                sliding_window=component.geometry.sliding_window_tokens,
            )
            for component in self.components
        )
        return KvCacheDescriptor(
            groups=groups,
            use_eagle_pop=self.vllm.use_eagle_pop if self.vllm else False,
            mamba_cache_mode=self.vllm.mamba_cache_mode if self.vllm else "none",
        )


@dataclass(frozen=True)
class KvCacheDescriptor:
    """Immutable result of an authoritative metadata fetch.

    ``groups`` is ordered as returned by the engine; each spec carries its own
    ``group_idx`` (do not infer the index from tuple position). An empty tuple
    is a valid success (e.g. a single-group / attention-free model) and must
    never be represented as ``None``.

    ``use_eagle_pop`` indicates whether the engine pops the last matched
    FullAttention block during prefix cache lookup (MTP/Eagle correctness).
    ``mamba_cache_mode`` is the engine's mamba state caching strategy
    ("none" | "all" | "light" | "light_flex").
    """

    groups: tuple[KvCacheGroupSpec, ...]
    use_eagle_pop: bool = False
    mamba_cache_mode: str = "none"


def _json_default(value: object) -> object:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    raise TypeError(f"cannot serialize {type(value).__name__}")


class MetadataFetchError(RuntimeError):
    """Base error for an authoritative metadata fetch."""


class MetadataTemporarilyUnavailable(MetadataFetchError):
    """Retryable transport failure or explicit UNAVAILABLE response.

    Raised only after the bounded per-attempt retry policy is exhausted. The
    startup supervisor treats this as a transient condition: the process stays
    in ``starting`` until the deployment startup/liveness policy replaces it.
    """


class MetadataProtocolError(MetadataFetchError):
    """Non-retryable response or malformed metadata.

    Covers invalid status codes, wrong field shape, duplicate group ids, and
    invalid sizes. This is a fatal configuration/protocol error that causes
    startup ``failed``.
    """


def _optional_int(message: Any, field_name: str) -> int | None:
    if not message.HasField(field_name):
        return None
    return int(getattr(message, field_name).value)


def _typed_settings(values: Iterable[Any], *, scope: str) -> TypedSettings:
    parsed: list[tuple[str, SettingValue]] = []
    names: set[str] = set()
    for setting in values:
        name = str(setting.name)
        if not name:
            raise MetadataProtocolError(f"{scope} contains an empty setting name")
        if name in names:
            raise MetadataProtocolError(
                f"{scope} contains duplicate setting name {name!r}"
            )
        names.add(name)
        value_kind = setting.WhichOneof("value")
        if value_kind is None:
            raise MetadataProtocolError(f"{scope} setting {name!r} has no typed value")
        raw_value = getattr(setting, value_kind)
        if value_kind.endswith("_list"):
            value: SettingValue = tuple(raw_value.values)
        else:
            value = raw_value
        parsed.append((name, value))
    return tuple(parsed)


def _parse_transport(
    payload: Any, *, require_incremental_transport: bool
) -> EventTransport:
    transport = EventTransport(
        live_endpoint=str(payload.live_endpoint),
        topic=str(payload.topic),
        replay_supported=bool(payload.replay_supported),
        replay_endpoint=str(payload.replay_endpoint),
        serialization=str(payload.serialization),
    )
    if not require_incremental_transport:
        return transport
    if not transport.live_endpoint:
        raise MetadataProtocolError("event transport is missing live_endpoint")
    if transport.serialization not in _SUPPORTED_SERIALIZATIONS:
        raise MetadataProtocolError(
            f"unsupported event serialization {transport.serialization!r}"
        )
    if transport.replay_supported and not transport.replay_endpoint:
        raise MetadataProtocolError(
            "event transport declares replay support without replay_endpoint"
        )
    if not transport.replay_supported and transport.replay_endpoint:
        raise MetadataProtocolError(
            "event transport provides replay_endpoint while replay is unsupported"
        )
    return transport


def _parse_topology(payload: Any) -> RuntimeTopology:
    topology = RuntimeTopology(
        data_parallel_size=int(payload.data_parallel_size),
        tensor_parallel_size=int(payload.tensor_parallel_size),
        pipeline_parallel_size=int(payload.pipeline_parallel_size),
        data_parallel_rank=int(payload.data_parallel_rank),
        tensor_parallel_rank=int(payload.tensor_parallel_rank),
        pipeline_parallel_rank=int(payload.pipeline_parallel_rank),
    )
    for size_name, rank_name in (
        ("data_parallel_size", "data_parallel_rank"),
        ("tensor_parallel_size", "tensor_parallel_rank"),
        ("pipeline_parallel_size", "pipeline_parallel_rank"),
    ):
        size = getattr(topology, size_name)
        rank = getattr(topology, rank_name)
        if size < 1:
            raise MetadataProtocolError(f"runtime topology {size_name} must be >= 1")
        if rank < 0 or rank >= size:
            raise MetadataProtocolError(
                f"runtime topology {rank_name}={rank} is outside {size_name}={size}"
            )
    return topology


def _parse_geometry(payload: Any, *, component_id: int) -> CacheGeometry:
    geometry = CacheGeometry(
        block_size_tokens=int(payload.block_size_tokens),
        page_size_tokens=_optional_int(payload, "page_size_tokens"),
        block_count=_optional_int(payload, "block_count"),
        group_payload_size_bytes=_optional_int(payload, "group_payload_size_bytes"),
        cache_dtype=str(payload.cache_dtype),
        element_size_bytes=_optional_int(payload, "element_size_bytes"),
        layer_count=_optional_int(payload, "layer_count"),
        cache_layout=str(payload.cache_layout),
        sliding_window_tokens=_optional_int(payload, "sliding_window_tokens"),
        checkpoint_alignment_tokens=_optional_int(
            payload, "checkpoint_alignment_tokens"
        ),
    )
    if geometry.block_size_tokens <= 0:
        raise MetadataProtocolError(
            f"component_id {component_id} has invalid block_size_tokens "
            f"{geometry.block_size_tokens}"
        )
    for name in (
        "page_size_tokens",
        "block_count",
        "group_payload_size_bytes",
        "element_size_bytes",
        "layer_count",
        "checkpoint_alignment_tokens",
    ):
        value = getattr(geometry, name)
        if value is not None and value <= 0:
            raise MetadataProtocolError(
                f"component_id {component_id} has invalid {name} {value}"
            )
    return geometry


def _parse_components(payload: Iterable[Any]) -> tuple[CacheComponent, ...]:
    components: list[CacheComponent] = []
    seen_ids: set[int] = set()
    for entry in payload:
        component_id = int(entry.component_id)
        if component_id in seen_ids:
            raise MetadataProtocolError(
                f"bootstrap contains duplicate component_id {component_id}"
            )
        seen_ids.add(component_id)
        component_kind = str(entry.component_kind)
        if not component_kind:
            raise MetadataProtocolError(
                f"component_id {component_id} has empty component_kind"
            )
        components.append(
            CacheComponent(
                component_id=component_id,
                component_kind=component_kind,
                geometry=_parse_geometry(entry.geometry, component_id=component_id),
                compatibility_settings=_typed_settings(
                    entry.compatibility_settings,
                    scope=f"component_id {component_id} compatibility_settings",
                ),
                diagnostic_settings=_typed_settings(
                    entry.diagnostic_settings,
                    scope=f"component_id {component_id} diagnostic_settings",
                ),
            )
        )
    return tuple(components)


def _validate_event_schema_version(version: int, *, engine_kind: str) -> None:
    if version not in _SUPPORTED_EVENT_SCHEMA_VERSIONS:
        raise MetadataProtocolError(
            f"unsupported {engine_kind} event_schema_version {version}"
        )


def parse_kv_event_bootstrap(
    payload: Any,
    *,
    expected_engine_kind: str,
    require_incremental_transport: bool = True,
) -> KvEventBootstrap:
    """Validate a protobuf bootstrap response and detach it from protobuf.

    The returned immutable model is the only cache/event metadata passed into
    adapters and KVCM translation. Unknown semantic versions, contradictory
    engine schemas and duplicate component IDs fail before any ZMQ socket is
    opened. Incremental transport completeness is required only when that
    pipeline is enabled.
    """

    err_code = int(getattr(payload, "err_code", 0))
    if err_code != engine_service_rpc_pb2.KV_EVENT_BOOTSTRAP_OK:
        raise MetadataProtocolError(
            "DashLLM bootstrap returned error "
            f"{err_code}: {str(getattr(payload, 'err_msg', ''))}"
        )
    protocol_version = int(getattr(payload, "protocol_version", 0))
    if protocol_version not in _SUPPORTED_PROTOCOL_VERSIONS:
        raise MetadataProtocolError(
            f"unsupported KV event bootstrap protocol_version {protocol_version}"
        )
    engine_kind = str(getattr(payload, "engine_kind", ""))
    if engine_kind != expected_engine_kind:
        raise MetadataProtocolError(
            f"bootstrap engine_kind {engine_kind!r} does not match configured "
            f"engine {expected_engine_kind!r}"
        )
    schema_kind = payload.WhichOneof("engine_schema")
    if schema_kind != engine_kind:
        raise MetadataProtocolError(
            f"bootstrap engine schema {schema_kind!r} does not match engine_kind "
            f"{engine_kind!r}"
        )

    vllm: VllmEventSchema | None = None
    sglang: SglangEventSchema | None = None
    if engine_kind == "vllm":
        schema = payload.vllm
        _validate_event_schema_version(
            int(schema.event_schema_version), engine_kind=engine_kind
        )
        vllm = VllmEventSchema(
            event_schema_version=int(schema.event_schema_version),
            use_eagle_pop=bool(schema.use_eagle_pop),
            mamba_cache_mode=str(schema.mamba_cache_mode),
            hash_algorithm=str(schema.hash_algorithm),
            hash_version=str(schema.hash_version),
            cache_settings=_typed_settings(
                schema.cache_settings, scope="vllm.cache_settings"
            ),
        )
    elif engine_kind == "sglang":
        schema = payload.sglang
        _validate_event_schema_version(
            int(schema.event_schema_version), engine_kind=engine_kind
        )
        sglang = SglangEventSchema(
            event_schema_version=int(schema.event_schema_version),
            cache_key_mode=str(schema.cache_key_mode),
            native_hash_algorithm=str(schema.native_hash_algorithm),
            cache_settings=_typed_settings(
                schema.cache_settings, scope="sglang.cache_settings"
            ),
        )
    else:
        raise MetadataProtocolError(f"unsupported engine_kind {engine_kind!r}")

    return KvEventBootstrap(
        protocol_version=protocol_version,
        engine_kind=engine_kind,
        event_transport=_parse_transport(
            payload.event_transport,
            require_incremental_transport=require_incremental_transport,
        ),
        runtime_topology=_parse_topology(payload.runtime_topology),
        snapshot=SnapshotCapability(
            supported=bool(payload.snapshot.supported),
            versioned=bool(payload.snapshot.versioned),
        ),
        components=_parse_components(payload.components),
        compatibility_settings=_typed_settings(
            payload.compatibility_settings, scope="compatibility_settings"
        ),
        diagnostic_settings=_typed_settings(
            payload.diagnostic_settings, scope="diagnostic_settings"
        ),
        vllm=vllm,
        sglang=sglang,
    )
