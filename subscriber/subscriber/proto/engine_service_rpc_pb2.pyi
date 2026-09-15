from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class KvEventBootstrapErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    KV_EVENT_BOOTSTRAP_OK: _ClassVar[KvEventBootstrapErrorCode]
    KV_EVENT_BOOTSTRAP_UNAVAILABLE: _ClassVar[KvEventBootstrapErrorCode]
    KV_EVENT_BOOTSTRAP_INCOMPATIBLE: _ClassVar[KvEventBootstrapErrorCode]
KV_EVENT_BOOTSTRAP_OK: KvEventBootstrapErrorCode
KV_EVENT_BOOTSTRAP_UNAVAILABLE: KvEventBootstrapErrorCode
KV_EVENT_BOOTSTRAP_INCOMPATIBLE: KvEventBootstrapErrorCode

class StatusVersionPB(_message.Message):
    __slots__ = ("latest_cache_version", "latest_finished_version")
    LATEST_CACHE_VERSION_FIELD_NUMBER: _ClassVar[int]
    LATEST_FINISHED_VERSION_FIELD_NUMBER: _ClassVar[int]
    latest_cache_version: int
    latest_finished_version: int
    def __init__(self, latest_cache_version: _Optional[int] = ..., latest_finished_version: _Optional[int] = ...) -> None: ...

class WorkerStatusPB(_message.Message):
    __slots__ = ("status_version", "alive", "latest_finished_version")
    STATUS_VERSION_FIELD_NUMBER: _ClassVar[int]
    ALIVE_FIELD_NUMBER: _ClassVar[int]
    LATEST_FINISHED_VERSION_FIELD_NUMBER: _ClassVar[int]
    status_version: int
    alive: bool
    latest_finished_version: int
    def __init__(self, status_version: _Optional[int] = ..., alive: _Optional[bool] = ..., latest_finished_version: _Optional[int] = ...) -> None: ...

class KvCacheBlockListPB(_message.Message):
    __slots__ = ("raw_snapshot", "snapshot_version", "block_size")
    RAW_SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_VERSION_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    raw_snapshot: bytes
    snapshot_version: int
    block_size: int
    def __init__(self, raw_snapshot: _Optional[bytes] = ..., snapshot_version: _Optional[int] = ..., block_size: _Optional[int] = ...) -> None: ...

class KvCacheBlocksRequestPB(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class OptionalUInt64PB(_message.Message):
    __slots__ = ("value",)
    value: int
    def __init__(self, value: _Optional[int] = ...) -> None: ...

class OptionalInt64PB(_message.Message):
    __slots__ = ("value",)
    value: int
    def __init__(self, value: _Optional[int] = ...) -> None: ...

class StringListPB(_message.Message):
    __slots__ = ("values",)
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, values: _Optional[_Iterable[str]] = ...) -> None: ...

class Int64ListPB(_message.Message):
    __slots__ = ("values",)
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class UInt64ListPB(_message.Message):
    __slots__ = ("values",)
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class DoubleListPB(_message.Message):
    __slots__ = ("values",)
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class BoolListPB(_message.Message):
    __slots__ = ("values",)
    values: _containers.RepeatedScalarFieldContainer[bool]
    def __init__(self, values: _Optional[_Iterable[bool]] = ...) -> None: ...

class TypedSettingPB(_message.Message):
    __slots__ = ("name", "bool_value", "int64_value", "uint64_value", "double_value", "string_value", "bytes_value", "string_list", "int64_list", "uint64_list", "double_list", "bool_list")
    name: str
    bool_value: bool
    int64_value: int
    uint64_value: int
    double_value: float
    string_value: str
    bytes_value: bytes
    string_list: StringListPB
    int64_list: Int64ListPB
    uint64_list: UInt64ListPB
    double_list: DoubleListPB
    bool_list: BoolListPB
    def __init__(self, name: _Optional[str] = ..., bool_value: _Optional[bool] = ..., int64_value: _Optional[int] = ..., uint64_value: _Optional[int] = ..., double_value: _Optional[float] = ..., string_value: _Optional[str] = ..., bytes_value: _Optional[bytes] = ..., string_list: _Optional[_Union[StringListPB, _Mapping]] = ..., int64_list: _Optional[_Union[Int64ListPB, _Mapping]] = ..., uint64_list: _Optional[_Union[UInt64ListPB, _Mapping]] = ..., double_list: _Optional[_Union[DoubleListPB, _Mapping]] = ..., bool_list: _Optional[_Union[BoolListPB, _Mapping]] = ...) -> None: ...

class EventTransportPB(_message.Message):
    __slots__ = ("live_endpoint", "topic", "replay_supported", "replay_endpoint", "serialization")
    live_endpoint: str
    topic: str
    replay_supported: bool
    replay_endpoint: str
    serialization: str
    def __init__(self, live_endpoint: _Optional[str] = ..., topic: _Optional[str] = ..., replay_supported: _Optional[bool] = ..., replay_endpoint: _Optional[str] = ..., serialization: _Optional[str] = ...) -> None: ...

class RuntimeTopologyPB(_message.Message):
    __slots__ = ("data_parallel_size", "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_rank", "tensor_parallel_rank", "pipeline_parallel_rank")
    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_rank: int
    tensor_parallel_rank: int
    pipeline_parallel_rank: int
    def __init__(self, data_parallel_size: _Optional[int] = ..., tensor_parallel_size: _Optional[int] = ..., pipeline_parallel_size: _Optional[int] = ..., data_parallel_rank: _Optional[int] = ..., tensor_parallel_rank: _Optional[int] = ..., pipeline_parallel_rank: _Optional[int] = ...) -> None: ...

class SnapshotCapabilityPB(_message.Message):
    __slots__ = ("supported", "versioned")
    supported: bool
    versioned: bool
    def __init__(self, supported: _Optional[bool] = ..., versioned: _Optional[bool] = ...) -> None: ...

class CacheGeometryPB(_message.Message):
    __slots__ = ("block_size_tokens", "page_size_tokens", "block_count", "group_payload_size_bytes", "cache_dtype", "element_size_bytes", "layer_count", "cache_layout", "sliding_window_tokens", "checkpoint_alignment_tokens")
    block_size_tokens: int
    page_size_tokens: OptionalUInt64PB
    block_count: OptionalUInt64PB
    group_payload_size_bytes: OptionalUInt64PB
    cache_dtype: str
    element_size_bytes: OptionalUInt64PB
    layer_count: OptionalUInt64PB
    cache_layout: str
    sliding_window_tokens: OptionalInt64PB
    checkpoint_alignment_tokens: OptionalUInt64PB
    def __init__(self, block_size_tokens: _Optional[int] = ..., page_size_tokens: _Optional[_Union[OptionalUInt64PB, _Mapping]] = ..., block_count: _Optional[_Union[OptionalUInt64PB, _Mapping]] = ..., group_payload_size_bytes: _Optional[_Union[OptionalUInt64PB, _Mapping]] = ..., cache_dtype: _Optional[str] = ..., element_size_bytes: _Optional[_Union[OptionalUInt64PB, _Mapping]] = ..., layer_count: _Optional[_Union[OptionalUInt64PB, _Mapping]] = ..., cache_layout: _Optional[str] = ..., sliding_window_tokens: _Optional[_Union[OptionalInt64PB, _Mapping]] = ..., checkpoint_alignment_tokens: _Optional[_Union[OptionalUInt64PB, _Mapping]] = ...) -> None: ...

class CacheComponentPB(_message.Message):
    __slots__ = ("component_id", "component_kind", "geometry", "compatibility_settings", "diagnostic_settings")
    component_id: int
    component_kind: str
    geometry: CacheGeometryPB
    compatibility_settings: _containers.RepeatedCompositeFieldContainer[TypedSettingPB]
    diagnostic_settings: _containers.RepeatedCompositeFieldContainer[TypedSettingPB]
    def __init__(self, component_id: _Optional[int] = ..., component_kind: _Optional[str] = ..., geometry: _Optional[_Union[CacheGeometryPB, _Mapping]] = ..., compatibility_settings: _Optional[_Iterable[_Union[TypedSettingPB, _Mapping]]] = ..., diagnostic_settings: _Optional[_Iterable[_Union[TypedSettingPB, _Mapping]]] = ...) -> None: ...

class VllmKvEventSchemaPB(_message.Message):
    __slots__ = ("event_schema_version", "use_eagle_pop", "mamba_cache_mode", "hash_algorithm", "hash_version", "cache_settings")
    event_schema_version: int
    use_eagle_pop: bool
    mamba_cache_mode: str
    hash_algorithm: str
    hash_version: str
    cache_settings: _containers.RepeatedCompositeFieldContainer[TypedSettingPB]
    def __init__(self, event_schema_version: _Optional[int] = ..., use_eagle_pop: _Optional[bool] = ..., mamba_cache_mode: _Optional[str] = ..., hash_algorithm: _Optional[str] = ..., hash_version: _Optional[str] = ..., cache_settings: _Optional[_Iterable[_Union[TypedSettingPB, _Mapping]]] = ...) -> None: ...

class SglangKvEventSchemaPB(_message.Message):
    __slots__ = ("cache_key_mode", "event_schema_version", "native_hash_algorithm", "cache_settings")
    cache_key_mode: str
    event_schema_version: int
    native_hash_algorithm: str
    cache_settings: _containers.RepeatedCompositeFieldContainer[TypedSettingPB]
    def __init__(self, cache_key_mode: _Optional[str] = ..., event_schema_version: _Optional[int] = ..., native_hash_algorithm: _Optional[str] = ..., cache_settings: _Optional[_Iterable[_Union[TypedSettingPB, _Mapping]]] = ...) -> None: ...

class KvEventBootstrapInfoPB(_message.Message):
    __slots__ = ("protocol_version", "engine_kind", "event_transport", "runtime_topology", "snapshot", "components", "err_code", "err_msg", "compatibility_settings", "diagnostic_settings", "vllm", "sglang")
    protocol_version: int
    engine_kind: str
    event_transport: EventTransportPB
    runtime_topology: RuntimeTopologyPB
    snapshot: SnapshotCapabilityPB
    components: _containers.RepeatedCompositeFieldContainer[CacheComponentPB]
    err_code: KvEventBootstrapErrorCode
    err_msg: str
    compatibility_settings: _containers.RepeatedCompositeFieldContainer[TypedSettingPB]
    diagnostic_settings: _containers.RepeatedCompositeFieldContainer[TypedSettingPB]
    vllm: VllmKvEventSchemaPB
    sglang: SglangKvEventSchemaPB
    def __init__(self, protocol_version: _Optional[int] = ..., engine_kind: _Optional[str] = ..., event_transport: _Optional[_Union[EventTransportPB, _Mapping]] = ..., runtime_topology: _Optional[_Union[RuntimeTopologyPB, _Mapping]] = ..., snapshot: _Optional[_Union[SnapshotCapabilityPB, _Mapping]] = ..., components: _Optional[_Iterable[_Union[CacheComponentPB, _Mapping]]] = ..., err_code: _Optional[_Union[KvEventBootstrapErrorCode, str]] = ..., err_msg: _Optional[str] = ..., compatibility_settings: _Optional[_Iterable[_Union[TypedSettingPB, _Mapping]]] = ..., diagnostic_settings: _Optional[_Iterable[_Union[TypedSettingPB, _Mapping]]] = ..., vllm: _Optional[_Union[VllmKvEventSchemaPB, _Mapping]] = ..., sglang: _Optional[_Union[SglangKvEventSchemaPB, _Mapping]] = ...) -> None: ...

class KvEventBootstrapInfoRequestPB(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
