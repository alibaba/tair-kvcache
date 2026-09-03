# -*- coding: utf-8 -*-
"""Hand-written protobuf definitions for KVCM meta_service.proto.

Compatible with protobuf>=3.20.3 (no runtime_version / _builder dependency).
This file intentionally keeps only the KVCM MetaService subset used by the
subscriber transport.
"""

from __future__ import annotations

from google.protobuf import descriptor_pb2 as _descriptor_pb2
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message_factory as _message_factory
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

_sym_db = _symbol_database.Default()
_PKG = "kv_cache_manager.proto.meta"
_T = _descriptor_pb2.FieldDescriptorProto
_LABEL_OPTIONAL = _T.LABEL_OPTIONAL
_LABEL_REPEATED = _T.LABEL_REPEATED


def _type_name(name: str) -> str:
    return f".{_PKG}.{name}"


def _add_field(
    message: _descriptor_pb2.DescriptorProto,
    name: str,
    number: int,
    label: int,
    field_type: int,
    type_name: str = "",
    oneof_index: int | None = None,
) -> None:
    field = message.field.add()
    field.name = name
    field.number = number
    field.label = label
    field.type = field_type
    if type_name:
        field.type_name = type_name
    if oneof_index is not None:
        field.oneof_index = oneof_index


def _add_enum(
    file_proto: _descriptor_pb2.FileDescriptorProto,
    name: str,
    values: tuple[tuple[str, int], ...],
) -> None:
    enum = file_proto.enum_type.add()
    enum.name = name
    for value_name, number in values:
        enum.value.add(name=value_name, number=number)


def _add_message(
    file_proto: _descriptor_pb2.FileDescriptorProto, name: str
) -> _descriptor_pb2.DescriptorProto:
    message = file_proto.message_type.add()
    message.name = name
    return message


def _build_file_descriptor():
    file_proto = _descriptor_pb2.FileDescriptorProto()
    file_proto.name = "subscriber/proto/kvcm_meta_service.proto"
    file_proto.package = _PKG
    file_proto.syntax = "proto3"

    _add_enum(
        file_proto,
        "ErrorCode",
        (
            ("UNSPECIFIED", 0),
            ("OK", 1),
            ("UNSUPPORTED", 2),
            ("INTERNAL_ERROR", 3),
            ("SERVICE_NOT_READY", 4),
            ("INVALID_ARGUMENT", 5),
            ("DUPLICATE_ENTITY", 6),
            ("REACH_MAX_ENTITY_CAPACITY", 7),
            ("INSTANCE_NOT_EXIST", 8),
            ("SERVER_NOT_LEADER", 9),
            ("NODE_NOT_REGISTERED", 10),
            ("SNAPSHOT_IN_PROGRESS", 11),
            ("SNAPSHOT_RATE_LIMITED", 13),
            ("SNAPSHOT_REQUIRED", 14),
            ("IO_ERROR", 20),
            ("UNKNOWN_ERROR", 100),
            ("ERROR_MAX", 65535),
        ),
    )
    _add_enum(
        file_proto,
        "StorageType",
        (
            ("ST_UNSPECIFIED", 0),
            ("ST_3FS", 1),
            ("ST_MOONCAKE", 2),
            ("ST_TAIRMEMPOOL", 3),
            ("ST_NFS", 4),
            ("ST_VCNS_3FS", 5),
            ("ST_DUMMY", 6),
            ("ST_EVENT_REPORT_L1P5", 7),
            ("ST_EVENT_REPORT_L2", 8),
        ),
    )
    _add_enum(
        file_proto,
        "ReportEventType",
        (
            ("EVENT_UNSPECIFIED", 0),
            ("EVENT_NODE_REGISTER", 1),
            ("EVENT_BLOCK_ADD", 2),
            ("EVENT_BLOCK_DELETE", 3),
            ("EVENT_HOST_DOWN", 4),
            ("EVENT_HEARTBEAT", 5),
            ("EVENT_BLOCK_SNAPSHOT", 6),
        ),
    )
    _add_enum(
        file_proto,
        "QueryType",
        (
            ("QT_UNSPECIFIED", 0),
            ("QT_BATCH_GET", 1),
            ("QT_PREFIX_MATCH", 2),
            ("QT_REVERSE_ROLL_SW_MATCH", 3),
            ("QT_PREFIX_MATCH_WITH_MAMBA", 4),
        ),
    )

    status = _add_message(file_proto, "Status")
    _add_field(
        status,
        "code",
        1,
        _LABEL_OPTIONAL,
        _T.TYPE_ENUM,
        _type_name("ErrorCode"),
    )
    _add_field(status, "message", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)

    header = _add_message(file_proto, "CommonResponseHeader")
    _add_field(
        header,
        "status",
        1,
        _LABEL_OPTIONAL,
        _T.TYPE_MESSAGE,
        _type_name("Status"),
    )
    _add_field(header, "request_id", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(header, "tracer_result", 3, _LABEL_OPTIONAL, _T.TYPE_STRING)

    node_register = _add_message(file_proto, "NodeRegisterEventParams")
    _add_field(node_register, "mediums", 1, _LABEL_REPEATED, _T.TYPE_STRING)

    location_spec = _add_message(file_proto, "LocationSpec")
    _add_field(location_spec, "name", 1, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(location_spec, "uri", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)

    block_add = _add_message(file_proto, "BlockAddEventParams")
    _add_field(block_add, "block_key", 1, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(block_add, "uri", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(block_add, "medium", 3, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(
        block_add,
        "specs",
        4,
        _LABEL_REPEATED,
        _T.TYPE_MESSAGE,
        _type_name("LocationSpec"),
    )

    block_delete = _add_message(file_proto, "BlockDeleteEventParams")
    _add_field(block_delete, "block_key", 1, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(block_delete, "medium", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(block_delete, "spec_names", 3, _LABEL_REPEATED, _T.TYPE_STRING)

    block_snapshot_item = _add_message(file_proto, "BlockSnapshotItem")
    _add_field(block_snapshot_item, "block_key", 1, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(block_snapshot_item, "medium", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(
        block_snapshot_item,
        "specs",
        3,
        _LABEL_REPEATED,
        _T.TYPE_MESSAGE,
        _type_name("LocationSpec"),
    )

    block_snapshot = _add_message(file_proto, "BlockSnapshotEventParams")
    _add_field(block_snapshot, "medium", 1, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(
        block_snapshot,
        "blocks",
        2,
        _LABEL_REPEATED,
        _T.TYPE_MESSAGE,
        _type_name("BlockSnapshotItem"),
    )

    _add_message(file_proto, "HostDownEventParams")

    heartbeat = _add_message(file_proto, "HeartbeatEventParams")
    system_status = heartbeat.nested_type.add()
    system_status.name = "SystemStatusEntry"
    system_status.options.map_entry = True
    _add_field(system_status, "key", 1, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(system_status, "value", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(
        heartbeat,
        "system_status",
        1,
        _LABEL_REPEATED,
        _T.TYPE_MESSAGE,
        f"{_type_name('HeartbeatEventParams')}.SystemStatusEntry",
    )

    event_item = _add_message(file_proto, "EventItem")
    event_item.oneof_decl.add().name = "event_params"
    _add_field(
        event_item,
        "event_type",
        1,
        _LABEL_OPTIONAL,
        _T.TYPE_ENUM,
        _type_name("ReportEventType"),
    )
    _add_field(
        event_item,
        "node_register",
        2,
        _LABEL_OPTIONAL,
        _T.TYPE_MESSAGE,
        _type_name("NodeRegisterEventParams"),
        0,
    )
    _add_field(
        event_item,
        "block_add",
        3,
        _LABEL_OPTIONAL,
        _T.TYPE_MESSAGE,
        _type_name("BlockAddEventParams"),
        0,
    )
    _add_field(
        event_item,
        "block_delete",
        4,
        _LABEL_OPTIONAL,
        _T.TYPE_MESSAGE,
        _type_name("BlockDeleteEventParams"),
        0,
    )
    _add_field(
        event_item,
        "host_down",
        5,
        _LABEL_OPTIONAL,
        _T.TYPE_MESSAGE,
        _type_name("HostDownEventParams"),
        0,
    )
    _add_field(
        event_item,
        "heartbeat",
        6,
        _LABEL_OPTIONAL,
        _T.TYPE_MESSAGE,
        _type_name("HeartbeatEventParams"),
        0,
    )
    _add_field(
        event_item,
        "block_snapshot",
        7,
        _LABEL_OPTIONAL,
        _T.TYPE_MESSAGE,
        _type_name("BlockSnapshotEventParams"),
        0,
    )

    report_request = _add_message(file_proto, "ReportEventRequest")
    _add_field(report_request, "trace_id", 1, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(report_request, "instance_id", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(report_request, "host_ip_port", 3, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(
        report_request,
        "events",
        4,
        _LABEL_REPEATED,
        _T.TYPE_MESSAGE,
        _type_name("EventItem"),
    )
    _add_field(
        report_request,
        "storage_type",
        5,
        _LABEL_OPTIONAL,
        _T.TYPE_ENUM,
        _type_name("StorageType"),
    )

    report_response = _add_message(file_proto, "ReportEventResponse")
    _add_field(
        report_response,
        "header",
        1,
        _LABEL_OPTIONAL,
        _T.TYPE_MESSAGE,
        _type_name("CommonResponseHeader"),
    )
    _add_field(
        report_response,
        "item_results",
        2,
        _LABEL_REPEATED,
        _T.TYPE_ENUM,
        _type_name("ErrorCode"),
    )
    _add_field(
        report_response,
        "committed_snapshot_version",
        3,
        _LABEL_OPTIONAL,
        _T.TYPE_STRING,
    )
    _add_field(report_response, "retry_after_ms", 4, _LABEL_OPTIONAL, _T.TYPE_UINT64)
    _add_field(report_response, "snapshot_required", 5, _LABEL_OPTIONAL, _T.TYPE_BOOL)
    _add_field(report_response, "extra_info", 6, _LABEL_OPTIONAL, _T.TYPE_STRING)

    location_spec_info = _add_message(file_proto, "LocationSpecInfo")
    _add_field(location_spec_info, "name", 1, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(location_spec_info, "size", 2, _LABEL_OPTIONAL, _T.TYPE_INT64)

    model_deployment = _add_message(file_proto, "ModelDeployment")
    _add_field(model_deployment, "model_name", 1, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(model_deployment, "dtype", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(model_deployment, "use_mla", 3, _LABEL_OPTIONAL, _T.TYPE_BOOL)
    _add_field(model_deployment, "tp_size", 4, _LABEL_OPTIONAL, _T.TYPE_INT32)
    _add_field(model_deployment, "dp_size", 5, _LABEL_OPTIONAL, _T.TYPE_INT32)
    _add_field(model_deployment, "lora_name", 6, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(model_deployment, "pp_size", 7, _LABEL_OPTIONAL, _T.TYPE_INT32)
    _add_field(model_deployment, "extra", 8, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(model_deployment, "user_data", 9, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(model_deployment, "use_eagle_pop", 10, _LABEL_OPTIONAL, _T.TYPE_BOOL)

    location_spec_group = _add_message(file_proto, "LocationSpecGroup")
    _add_field(location_spec_group, "name", 1, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(location_spec_group, "spec_names", 2, _LABEL_REPEATED, _T.TYPE_STRING)

    register_request = _add_message(file_proto, "RegisterInstanceRequest")
    _add_field(register_request, "trace_id", 1, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(register_request, "instance_group", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(register_request, "instance_id", 3, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(register_request, "block_size", 4, _LABEL_OPTIONAL, _T.TYPE_INT32)
    _add_field(
        register_request,
        "location_spec_infos",
        5,
        _LABEL_REPEATED,
        _T.TYPE_MESSAGE,
        _type_name("LocationSpecInfo"),
    )
    _add_field(
        register_request,
        "model_deployment",
        6,
        _LABEL_OPTIONAL,
        _T.TYPE_MESSAGE,
        _type_name("ModelDeployment"),
    )
    _add_field(
        register_request,
        "location_spec_groups",
        7,
        _LABEL_REPEATED,
        _T.TYPE_MESSAGE,
        _type_name("LocationSpecGroup"),
    )
    _add_field(
        register_request,
        "default_query_type",
        8,
        _LABEL_OPTIONAL,
        _T.TYPE_ENUM,
        _type_name("QueryType"),
    )

    register_response = _add_message(file_proto, "RegisterInstanceResponse")
    _add_field(
        register_response,
        "header",
        1,
        _LABEL_OPTIONAL,
        _T.TYPE_MESSAGE,
        _type_name("CommonResponseHeader"),
    )
    _add_field(register_response, "storage_configs", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(register_response, "extra_info", 3, _LABEL_OPTIONAL, _T.TYPE_STRING)

    endpoint = _add_message(file_proto, "MetaNodeEndpoint")
    _add_field(endpoint, "node_id", 1, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(endpoint, "host", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(endpoint, "meta_rpc_port", 3, _LABEL_OPTIONAL, _T.TYPE_INT32)
    _add_field(endpoint, "meta_http_port", 4, _LABEL_OPTIONAL, _T.TYPE_INT32)
    _add_field(endpoint, "custom_info", 5, _LABEL_OPTIONAL, _T.TYPE_STRING)

    cluster_request = _add_message(file_proto, "GetClusterInfoRequest")
    _add_field(cluster_request, "trace_id", 1, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(cluster_request, "instance_id", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)

    cluster_response = _add_message(file_proto, "GetClusterInfoResponse")
    _add_field(
        cluster_response,
        "header",
        1,
        _LABEL_OPTIONAL,
        _T.TYPE_MESSAGE,
        _type_name("CommonResponseHeader"),
    )
    _add_field(cluster_response, "self_node_id", 2, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(cluster_response, "leader_node_id", 3, _LABEL_OPTIONAL, _T.TYPE_STRING)
    _add_field(
        cluster_response,
        "leader_endpoint",
        4,
        _LABEL_OPTIONAL,
        _T.TYPE_MESSAGE,
        _type_name("MetaNodeEndpoint"),
    )

    service = file_proto.service.add()
    service.name = "MetaService"
    for method_name, request_name, response_name in (
        ("RegisterInstance", "RegisterInstanceRequest", "RegisterInstanceResponse"),
        ("GetClusterInfo", "GetClusterInfoRequest", "GetClusterInfoResponse"),
        ("ReportEvent", "ReportEventRequest", "ReportEventResponse"),
    ):
        method = service.method.add()
        method.name = method_name
        method.input_type = _type_name(request_name)
        method.output_type = _type_name(response_name)

    pool = _descriptor_pool.Default()
    try:
        return pool.AddSerializedFile(file_proto.SerializeToString())
    except Exception:
        return pool.FindFileByName(file_proto.name)


DESCRIPTOR = _build_file_descriptor()


def _message_class(name: str):
    descriptor = DESCRIPTOR.message_types_by_name[name]
    get_message_class = getattr(_message_factory, "GetMessageClass", None)
    if get_message_class is not None:
        return get_message_class(descriptor)
    factory = _message_factory.MessageFactory()
    return factory.GetPrototype(descriptor)


Status = _message_class("Status")
CommonResponseHeader = _message_class("CommonResponseHeader")
NodeRegisterEventParams = _message_class("NodeRegisterEventParams")
LocationSpec = _message_class("LocationSpec")
BlockAddEventParams = _message_class("BlockAddEventParams")
BlockDeleteEventParams = _message_class("BlockDeleteEventParams")
BlockSnapshotItem = _message_class("BlockSnapshotItem")
BlockSnapshotEventParams = _message_class("BlockSnapshotEventParams")
HostDownEventParams = _message_class("HostDownEventParams")
HeartbeatEventParams = _message_class("HeartbeatEventParams")
EventItem = _message_class("EventItem")
ReportEventRequest = _message_class("ReportEventRequest")
ReportEventResponse = _message_class("ReportEventResponse")
LocationSpecInfo = _message_class("LocationSpecInfo")
ModelDeployment = _message_class("ModelDeployment")
LocationSpecGroup = _message_class("LocationSpecGroup")
RegisterInstanceRequest = _message_class("RegisterInstanceRequest")
RegisterInstanceResponse = _message_class("RegisterInstanceResponse")
MetaNodeEndpoint = _message_class("MetaNodeEndpoint")
GetClusterInfoRequest = _message_class("GetClusterInfoRequest")
GetClusterInfoResponse = _message_class("GetClusterInfoResponse")

ErrorCode = _enum_type_wrapper.EnumTypeWrapper(
    DESCRIPTOR.enum_types_by_name["ErrorCode"]
)
StorageType = _enum_type_wrapper.EnumTypeWrapper(
    DESCRIPTOR.enum_types_by_name["StorageType"]
)
ReportEventType = _enum_type_wrapper.EnumTypeWrapper(
    DESCRIPTOR.enum_types_by_name["ReportEventType"]
)
QueryType = _enum_type_wrapper.EnumTypeWrapper(
    DESCRIPTOR.enum_types_by_name["QueryType"]
)

UNSPECIFIED = ErrorCode.UNSPECIFIED
OK = ErrorCode.OK
UNSUPPORTED = ErrorCode.UNSUPPORTED
INTERNAL_ERROR = ErrorCode.INTERNAL_ERROR
SERVICE_NOT_READY = ErrorCode.SERVICE_NOT_READY
INVALID_ARGUMENT = ErrorCode.INVALID_ARGUMENT
DUPLICATE_ENTITY = ErrorCode.DUPLICATE_ENTITY
REACH_MAX_ENTITY_CAPACITY = ErrorCode.REACH_MAX_ENTITY_CAPACITY
INSTANCE_NOT_EXIST = ErrorCode.INSTANCE_NOT_EXIST
SERVER_NOT_LEADER = ErrorCode.SERVER_NOT_LEADER
NODE_NOT_REGISTERED = ErrorCode.NODE_NOT_REGISTERED
SNAPSHOT_IN_PROGRESS = ErrorCode.SNAPSHOT_IN_PROGRESS
SNAPSHOT_RATE_LIMITED = ErrorCode.SNAPSHOT_RATE_LIMITED
SNAPSHOT_REQUIRED = ErrorCode.SNAPSHOT_REQUIRED
IO_ERROR = ErrorCode.IO_ERROR
UNKNOWN_ERROR = ErrorCode.UNKNOWN_ERROR
ERROR_MAX = ErrorCode.ERROR_MAX

ST_UNSPECIFIED = StorageType.ST_UNSPECIFIED
ST_3FS = StorageType.ST_3FS
ST_MOONCAKE = StorageType.ST_MOONCAKE
ST_TAIRMEMPOOL = StorageType.ST_TAIRMEMPOOL
ST_NFS = StorageType.ST_NFS
ST_VCNS_3FS = StorageType.ST_VCNS_3FS
ST_DUMMY = StorageType.ST_DUMMY
ST_EVENT_REPORT_L1P5 = StorageType.ST_EVENT_REPORT_L1P5
ST_EVENT_REPORT_L2 = StorageType.ST_EVENT_REPORT_L2

EVENT_UNSPECIFIED = ReportEventType.EVENT_UNSPECIFIED
EVENT_NODE_REGISTER = ReportEventType.EVENT_NODE_REGISTER
EVENT_BLOCK_ADD = ReportEventType.EVENT_BLOCK_ADD
EVENT_BLOCK_DELETE = ReportEventType.EVENT_BLOCK_DELETE
EVENT_HOST_DOWN = ReportEventType.EVENT_HOST_DOWN
EVENT_HEARTBEAT = ReportEventType.EVENT_HEARTBEAT
EVENT_BLOCK_SNAPSHOT = ReportEventType.EVENT_BLOCK_SNAPSHOT

QT_UNSPECIFIED = QueryType.QT_UNSPECIFIED
QT_BATCH_GET = QueryType.QT_BATCH_GET
QT_PREFIX_MATCH = QueryType.QT_PREFIX_MATCH
QT_REVERSE_ROLL_SW_MATCH = QueryType.QT_REVERSE_ROLL_SW_MATCH
QT_PREFIX_MATCH_WITH_MAMBA = QueryType.QT_PREFIX_MATCH_WITH_MAMBA

for _message in (
    Status,
    CommonResponseHeader,
    NodeRegisterEventParams,
    LocationSpec,
    BlockAddEventParams,
    BlockDeleteEventParams,
    BlockSnapshotItem,
    BlockSnapshotEventParams,
    HostDownEventParams,
    HeartbeatEventParams,
    EventItem,
    ReportEventRequest,
    ReportEventResponse,
    LocationSpecInfo,
    ModelDeployment,
    LocationSpecGroup,
    RegisterInstanceRequest,
    RegisterInstanceResponse,
    MetaNodeEndpoint,
    GetClusterInfoRequest,
    GetClusterInfoResponse,
):
    _sym_db.RegisterMessage(_message)
