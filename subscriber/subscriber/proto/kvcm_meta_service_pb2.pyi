from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from typing import (
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class ErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper): ...
UNSPECIFIED: ErrorCode
OK: ErrorCode
UNSUPPORTED: ErrorCode
INTERNAL_ERROR: ErrorCode
SERVICE_NOT_READY: ErrorCode
INVALID_ARGUMENT: ErrorCode
DUPLICATE_ENTITY: ErrorCode
REACH_MAX_ENTITY_CAPACITY: ErrorCode
INSTANCE_NOT_EXIST: ErrorCode
SERVER_NOT_LEADER: ErrorCode
NODE_NOT_REGISTERED: ErrorCode
SNAPSHOT_IN_PROGRESS: ErrorCode
SNAPSHOT_RATE_LIMITED: ErrorCode
SNAPSHOT_REQUIRED: ErrorCode
IO_ERROR: ErrorCode
UNKNOWN_ERROR: ErrorCode
ERROR_MAX: ErrorCode

class StorageType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper): ...
ST_UNSPECIFIED: StorageType
ST_3FS: StorageType
ST_MOONCAKE: StorageType
ST_TAIRMEMPOOL: StorageType
ST_NFS: StorageType
ST_VCNS_3FS: StorageType
ST_DUMMY: StorageType
ST_EVENT_REPORT_L1P5: StorageType
ST_EVENT_REPORT_L2: StorageType

class ReportEventType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper): ...
EVENT_UNSPECIFIED: ReportEventType
EVENT_NODE_REGISTER: ReportEventType
EVENT_BLOCK_ADD: ReportEventType
EVENT_BLOCK_DELETE: ReportEventType
EVENT_HOST_DOWN: ReportEventType
EVENT_HEARTBEAT: ReportEventType
EVENT_BLOCK_SNAPSHOT: ReportEventType

class QueryType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper): ...
QT_UNSPECIFIED: QueryType
QT_BATCH_GET: QueryType
QT_PREFIX_MATCH: QueryType
QT_REVERSE_ROLL_SW_MATCH: QueryType
QT_PREFIX_MATCH_WITH_MAMBA: QueryType

class Status(_message.Message):
    code: ErrorCode
    message: str
    def __init__(
        self,
        code: _Optional[_Union[ErrorCode, str]] = ...,
        message: _Optional[str] = ...,
    ) -> None: ...

class CommonResponseHeader(_message.Message):
    status: Status
    request_id: str
    tracer_result: str
    def __init__(
        self,
        status: _Optional[_Union[Status, _Mapping]] = ...,
        request_id: _Optional[str] = ...,
        tracer_result: _Optional[str] = ...,
    ) -> None: ...

class NodeRegisterEventParams(_message.Message):
    mediums: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, mediums: _Optional[_Iterable[str]] = ...) -> None: ...

class LocationSpec(_message.Message):
    name: str
    uri: str
    def __init__(
        self,
        name: _Optional[str] = ...,
        uri: _Optional[str] = ...,
    ) -> None: ...

class BlockAddEventParams(_message.Message):
    block_key: str
    uri: str
    medium: str
    specs: _containers.RepeatedCompositeFieldContainer[LocationSpec]
    def __init__(
        self,
        block_key: _Optional[str] = ...,
        uri: _Optional[str] = ...,
        medium: _Optional[str] = ...,
        specs: _Optional[_Iterable[_Union[LocationSpec, _Mapping]]] = ...,
    ) -> None: ...

class BlockDeleteEventParams(_message.Message):
    block_key: str
    medium: str
    spec_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        block_key: _Optional[str] = ...,
        medium: _Optional[str] = ...,
        spec_names: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class BlockSnapshotItem(_message.Message):
    block_key: str
    medium: str
    specs: _containers.RepeatedCompositeFieldContainer[LocationSpec]
    def __init__(
        self,
        block_key: _Optional[str] = ...,
        medium: _Optional[str] = ...,
        specs: _Optional[_Iterable[_Union[LocationSpec, _Mapping]]] = ...,
    ) -> None: ...

class BlockSnapshotEventParams(_message.Message):
    medium: str
    blocks: _containers.RepeatedCompositeFieldContainer[BlockSnapshotItem]
    def __init__(
        self,
        medium: _Optional[str] = ...,
        blocks: _Optional[_Iterable[_Union[BlockSnapshotItem, _Mapping]]] = ...,
    ) -> None: ...

class HostDownEventParams(_message.Message):
    def __init__(self) -> None: ...

class HeartbeatEventParams(_message.Message):
    system_status: _Mapping[str, str]
    def __init__(self, system_status: _Optional[_Mapping[str, str]] = ...) -> None: ...

class EventItem(_message.Message):
    event_type: ReportEventType
    node_register: NodeRegisterEventParams
    block_add: BlockAddEventParams
    block_delete: BlockDeleteEventParams
    host_down: HostDownEventParams
    heartbeat: HeartbeatEventParams
    block_snapshot: BlockSnapshotEventParams
    def __init__(
        self,
        event_type: _Optional[_Union[ReportEventType, str]] = ...,
        node_register: _Optional[
            _Union[NodeRegisterEventParams, _Mapping]
        ] = ...,
        block_add: _Optional[_Union[BlockAddEventParams, _Mapping]] = ...,
        block_delete: _Optional[_Union[BlockDeleteEventParams, _Mapping]] = ...,
        host_down: _Optional[_Union[HostDownEventParams, _Mapping]] = ...,
        heartbeat: _Optional[_Union[HeartbeatEventParams, _Mapping]] = ...,
        block_snapshot: _Optional[
            _Union[BlockSnapshotEventParams, _Mapping]
        ] = ...,
    ) -> None: ...

class ReportEventRequest(_message.Message):
    trace_id: str
    instance_id: str
    host_ip_port: str
    events: _containers.RepeatedCompositeFieldContainer[EventItem]
    storage_type: StorageType
    def __init__(
        self,
        trace_id: _Optional[str] = ...,
        instance_id: _Optional[str] = ...,
        host_ip_port: _Optional[str] = ...,
        events: _Optional[_Iterable[_Union[EventItem, _Mapping]]] = ...,
        storage_type: _Optional[_Union[StorageType, str]] = ...,
    ) -> None: ...

class ReportEventResponse(_message.Message):
    header: CommonResponseHeader
    item_results: _containers.RepeatedScalarFieldContainer[ErrorCode]
    committed_snapshot_version: str
    retry_after_ms: int
    snapshot_required: bool
    extra_info: str
    def __init__(
        self,
        header: _Optional[_Union[CommonResponseHeader, _Mapping]] = ...,
        item_results: _Optional[_Iterable[_Union[ErrorCode, str]]] = ...,
        committed_snapshot_version: _Optional[str] = ...,
        retry_after_ms: _Optional[int] = ...,
        snapshot_required: _Optional[bool] = ...,
        extra_info: _Optional[str] = ...,
    ) -> None: ...

class LocationSpecInfo(_message.Message):
    name: str
    size: int
    def __init__(
        self,
        name: _Optional[str] = ...,
        size: _Optional[int] = ...,
    ) -> None: ...

class ModelDeployment(_message.Message):
    model_name: str
    dtype: str
    use_mla: bool
    tp_size: int
    dp_size: int
    lora_name: str
    pp_size: int
    extra: str
    user_data: str
    use_eagle_pop: bool
    def __init__(
        self,
        model_name: _Optional[str] = ...,
        dtype: _Optional[str] = ...,
        use_mla: _Optional[bool] = ...,
        tp_size: _Optional[int] = ...,
        dp_size: _Optional[int] = ...,
        lora_name: _Optional[str] = ...,
        pp_size: _Optional[int] = ...,
        extra: _Optional[str] = ...,
        user_data: _Optional[str] = ...,
        use_eagle_pop: _Optional[bool] = ...,
    ) -> None: ...

class LocationSpecGroup(_message.Message):
    name: str
    spec_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        name: _Optional[str] = ...,
        spec_names: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class RegisterInstanceRequest(_message.Message):
    trace_id: str
    instance_group: str
    instance_id: str
    block_size: int
    location_spec_infos: _containers.RepeatedCompositeFieldContainer[LocationSpecInfo]
    model_deployment: ModelDeployment
    location_spec_groups: _containers.RepeatedCompositeFieldContainer[LocationSpecGroup]
    default_query_type: QueryType
    def __init__(
        self,
        trace_id: _Optional[str] = ...,
        instance_group: _Optional[str] = ...,
        instance_id: _Optional[str] = ...,
        block_size: _Optional[int] = ...,
        location_spec_infos: _Optional[
            _Iterable[_Union[LocationSpecInfo, _Mapping]]
        ] = ...,
        model_deployment: _Optional[_Union[ModelDeployment, _Mapping]] = ...,
        location_spec_groups: _Optional[
            _Iterable[_Union[LocationSpecGroup, _Mapping]]
        ] = ...,
        default_query_type: _Optional[_Union[QueryType, str]] = ...,
    ) -> None: ...

class RegisterInstanceResponse(_message.Message):
    header: CommonResponseHeader
    storage_configs: str
    extra_info: str
    def __init__(
        self,
        header: _Optional[_Union[CommonResponseHeader, _Mapping]] = ...,
        storage_configs: _Optional[str] = ...,
        extra_info: _Optional[str] = ...,
    ) -> None: ...

class MetaNodeEndpoint(_message.Message):
    node_id: str
    host: str
    meta_rpc_port: int
    meta_http_port: int
    custom_info: str
    def __init__(
        self,
        node_id: _Optional[str] = ...,
        host: _Optional[str] = ...,
        meta_rpc_port: _Optional[int] = ...,
        meta_http_port: _Optional[int] = ...,
        custom_info: _Optional[str] = ...,
    ) -> None: ...

class GetClusterInfoRequest(_message.Message):
    trace_id: str
    instance_id: str
    def __init__(
        self,
        trace_id: _Optional[str] = ...,
        instance_id: _Optional[str] = ...,
    ) -> None: ...

class GetClusterInfoResponse(_message.Message):
    header: CommonResponseHeader
    self_node_id: str
    leader_node_id: str
    leader_endpoint: MetaNodeEndpoint
    def __init__(
        self,
        header: _Optional[_Union[CommonResponseHeader, _Mapping]] = ...,
        self_node_id: _Optional[str] = ...,
        leader_node_id: _Optional[str] = ...,
        leader_endpoint: _Optional[_Union[MetaNodeEndpoint, _Mapping]] = ...,
    ) -> None: ...
