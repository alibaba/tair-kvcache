# -*- coding: utf-8 -*-
"""Hand-written protobuf definitions for engine_service_rpc.proto.

Compatible with protobuf>=3.20.3 (no runtime_version / _builder dependency).
Mirrors the worker-status and KV-event bootstrap/control schema.
"""

from google.protobuf import descriptor_pb2 as _descriptor_pb2
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import message_factory as _message_factory
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()


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


def _build_file_descriptor():
    file_proto = _descriptor_pb2.FileDescriptorProto()
    file_proto.name = "subscriber/proto/engine_service_rpc.proto"
    file_proto.syntax = "proto3"

    # message StatusVersionPB
    status_version = file_proto.message_type.add()
    status_version.name = "StatusVersionPB"
    _add_field(status_version, "latest_cache_version", 1, 1, 3)  # int64
    _add_field(status_version, "latest_finished_version", 2, 1, 3)  # int64

    # message WorkerStatusPB
    worker_status = file_proto.message_type.add()
    worker_status.name = "WorkerStatusPB"
    _add_field(worker_status, "status_version", 12, 1, 3)  # int64
    _add_field(worker_status, "alive", 13, 1, 8)  # bool
    _add_field(worker_status, "latest_finished_version", 15, 1, 3)  # int64

    # message KvCacheBlockListPB
    block_list = file_proto.message_type.add()
    block_list.name = "KvCacheBlockListPB"
    _add_field(block_list, "raw_snapshot", 1, 1, 12)  # bytes
    _add_field(block_list, "snapshot_version", 2, 1, 3)  # int64
    _add_field(block_list, "block_size", 3, 1, 5)  # int32

    # message KvCacheBlocksRequestPB (empty)
    file_proto.message_type.add().name = "KvCacheBlocksRequestPB"

    bootstrap_error = file_proto.enum_type.add()
    bootstrap_error.name = "KvEventBootstrapErrorCode"
    bootstrap_error.value.add(name="KV_EVENT_BOOTSTRAP_OK", number=0)
    bootstrap_error.value.add(name="KV_EVENT_BOOTSTRAP_UNAVAILABLE", number=1)
    bootstrap_error.value.add(name="KV_EVENT_BOOTSTRAP_INCOMPATIBLE", number=2)

    optional_uint64 = file_proto.message_type.add()
    optional_uint64.name = "OptionalUInt64PB"
    _add_field(optional_uint64, "value", 1, 1, 4)  # uint64

    optional_int64 = file_proto.message_type.add()
    optional_int64.name = "OptionalInt64PB"
    _add_field(optional_int64, "value", 1, 1, 3)  # int64

    string_list = file_proto.message_type.add()
    string_list.name = "StringListPB"
    _add_field(string_list, "values", 1, 3, 9)  # repeated string

    int64_list = file_proto.message_type.add()
    int64_list.name = "Int64ListPB"
    _add_field(int64_list, "values", 1, 3, 18)  # repeated sint64

    uint64_list = file_proto.message_type.add()
    uint64_list.name = "UInt64ListPB"
    _add_field(uint64_list, "values", 1, 3, 4)  # repeated uint64

    double_list = file_proto.message_type.add()
    double_list.name = "DoubleListPB"
    _add_field(double_list, "values", 1, 3, 1)  # repeated double

    bool_list = file_proto.message_type.add()
    bool_list.name = "BoolListPB"
    _add_field(bool_list, "values", 1, 3, 8)  # repeated bool

    typed_setting = file_proto.message_type.add()
    typed_setting.name = "TypedSettingPB"
    typed_setting.oneof_decl.add().name = "value"
    _add_field(typed_setting, "name", 1, 1, 9)
    _add_field(typed_setting, "bool_value", 2, 1, 8, oneof_index=0)
    _add_field(typed_setting, "int64_value", 3, 1, 18, oneof_index=0)
    _add_field(typed_setting, "uint64_value", 4, 1, 4, oneof_index=0)
    _add_field(typed_setting, "double_value", 5, 1, 1, oneof_index=0)
    _add_field(typed_setting, "string_value", 6, 1, 9, oneof_index=0)
    _add_field(typed_setting, "bytes_value", 7, 1, 12, oneof_index=0)
    _add_field(typed_setting, "string_list", 8, 1, 11, ".StringListPB", 0)
    _add_field(typed_setting, "int64_list", 9, 1, 11, ".Int64ListPB", 0)
    _add_field(typed_setting, "uint64_list", 10, 1, 11, ".UInt64ListPB", 0)
    _add_field(typed_setting, "double_list", 11, 1, 11, ".DoubleListPB", 0)
    _add_field(typed_setting, "bool_list", 12, 1, 11, ".BoolListPB", 0)

    event_transport = file_proto.message_type.add()
    event_transport.name = "EventTransportPB"
    _add_field(event_transport, "live_endpoint", 1, 1, 9)
    _add_field(event_transport, "topic", 2, 1, 9)
    _add_field(event_transport, "replay_supported", 3, 1, 8)
    _add_field(event_transport, "replay_endpoint", 4, 1, 9)
    _add_field(event_transport, "serialization", 5, 1, 9)

    topology = file_proto.message_type.add()
    topology.name = "RuntimeTopologyPB"
    _add_field(topology, "data_parallel_size", 1, 1, 13)
    _add_field(topology, "tensor_parallel_size", 2, 1, 13)
    _add_field(topology, "pipeline_parallel_size", 3, 1, 13)
    _add_field(topology, "data_parallel_rank", 4, 1, 13)
    _add_field(topology, "tensor_parallel_rank", 5, 1, 13)
    _add_field(topology, "pipeline_parallel_rank", 6, 1, 13)

    snapshot_capability = file_proto.message_type.add()
    snapshot_capability.name = "SnapshotCapabilityPB"
    _add_field(snapshot_capability, "supported", 1, 1, 8)
    _add_field(snapshot_capability, "versioned", 2, 1, 8)

    geometry = file_proto.message_type.add()
    geometry.name = "CacheGeometryPB"
    _add_field(geometry, "block_size_tokens", 1, 1, 4)
    _add_field(geometry, "page_size_tokens", 2, 1, 11, ".OptionalUInt64PB")
    _add_field(geometry, "block_count", 3, 1, 11, ".OptionalUInt64PB")
    _add_field(
        geometry,
        "group_payload_size_bytes",
        4,
        1,
        11,
        ".OptionalUInt64PB",
    )
    _add_field(geometry, "cache_dtype", 5, 1, 9)
    _add_field(geometry, "element_size_bytes", 6, 1, 11, ".OptionalUInt64PB")
    _add_field(geometry, "layer_count", 7, 1, 11, ".OptionalUInt64PB")
    _add_field(geometry, "cache_layout", 8, 1, 9)
    _add_field(geometry, "sliding_window_tokens", 9, 1, 11, ".OptionalInt64PB")
    _add_field(
        geometry,
        "checkpoint_alignment_tokens",
        10,
        1,
        11,
        ".OptionalUInt64PB",
    )

    cache_component = file_proto.message_type.add()
    cache_component.name = "CacheComponentPB"
    _add_field(cache_component, "component_id", 1, 1, 13)
    _add_field(cache_component, "component_kind", 2, 1, 9)
    _add_field(cache_component, "geometry", 3, 1, 11, ".CacheGeometryPB")
    _add_field(
        cache_component,
        "compatibility_settings",
        4,
        3,
        11,
        ".TypedSettingPB",
    )
    _add_field(
        cache_component,
        "diagnostic_settings",
        5,
        3,
        11,
        ".TypedSettingPB",
    )

    vllm_schema = file_proto.message_type.add()
    vllm_schema.name = "VllmKvEventSchemaPB"
    _add_field(vllm_schema, "event_schema_version", 1, 1, 13)
    _add_field(vllm_schema, "use_eagle_pop", 2, 1, 8)
    _add_field(vllm_schema, "mamba_cache_mode", 3, 1, 9)
    _add_field(vllm_schema, "hash_algorithm", 4, 1, 9)
    _add_field(vllm_schema, "hash_version", 5, 1, 9)
    _add_field(vllm_schema, "cache_settings", 6, 3, 11, ".TypedSettingPB")

    sglang_schema = file_proto.message_type.add()
    sglang_schema.name = "SglangKvEventSchemaPB"
    _add_field(sglang_schema, "cache_key_mode", 1, 1, 9)
    _add_field(sglang_schema, "event_schema_version", 2, 1, 13)
    _add_field(sglang_schema, "native_hash_algorithm", 3, 1, 9)
    _add_field(sglang_schema, "cache_settings", 4, 3, 11, ".TypedSettingPB")

    bootstrap = file_proto.message_type.add()
    bootstrap.name = "KvEventBootstrapInfoPB"
    bootstrap.oneof_decl.add().name = "engine_schema"
    _add_field(bootstrap, "protocol_version", 1, 1, 13)
    _add_field(bootstrap, "engine_kind", 2, 1, 9)
    _add_field(bootstrap, "event_transport", 3, 1, 11, ".EventTransportPB")
    _add_field(bootstrap, "runtime_topology", 4, 1, 11, ".RuntimeTopologyPB")
    _add_field(bootstrap, "snapshot", 5, 1, 11, ".SnapshotCapabilityPB")
    _add_field(bootstrap, "components", 6, 3, 11, ".CacheComponentPB")
    _add_field(
        bootstrap,
        "err_code",
        7,
        1,
        _descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
        ".KvEventBootstrapErrorCode",
    )
    _add_field(bootstrap, "err_msg", 8, 1, 9)
    _add_field(
        bootstrap,
        "compatibility_settings",
        9,
        3,
        11,
        ".TypedSettingPB",
    )
    _add_field(
        bootstrap,
        "diagnostic_settings",
        10,
        3,
        11,
        ".TypedSettingPB",
    )
    _add_field(bootstrap, "vllm", 20, 1, 11, ".VllmKvEventSchemaPB", 0)
    _add_field(bootstrap, "sglang", 21, 1, 11, ".SglangKvEventSchemaPB", 0)

    file_proto.message_type.add().name = "KvEventBootstrapInfoRequestPB"

    # service RpcService
    service = file_proto.service.add()
    service.name = "RpcService"
    method = service.method.add()
    method.name = "GetWorkerStatus"
    method.input_type = ".StatusVersionPB"
    method.output_type = ".WorkerStatusPB"

    control_service = file_proto.service.add()
    control_service.name = "KvEventControlService"
    method = control_service.method.add()
    method.name = "GetKvEventBootstrapInfo"
    method.input_type = ".KvEventBootstrapInfoRequestPB"
    method.output_type = ".KvEventBootstrapInfoPB"
    method = control_service.method.add()
    method.name = "GetAllKvCacheBlocks"
    method.input_type = ".KvCacheBlocksRequestPB"
    method.output_type = ".KvCacheBlockListPB"

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


StatusVersionPB = _message_class("StatusVersionPB")
WorkerStatusPB = _message_class("WorkerStatusPB")
KvCacheBlockListPB = _message_class("KvCacheBlockListPB")
KvCacheBlocksRequestPB = _message_class("KvCacheBlocksRequestPB")
OptionalUInt64PB = _message_class("OptionalUInt64PB")
OptionalInt64PB = _message_class("OptionalInt64PB")
StringListPB = _message_class("StringListPB")
Int64ListPB = _message_class("Int64ListPB")
UInt64ListPB = _message_class("UInt64ListPB")
DoubleListPB = _message_class("DoubleListPB")
BoolListPB = _message_class("BoolListPB")
TypedSettingPB = _message_class("TypedSettingPB")
EventTransportPB = _message_class("EventTransportPB")
RuntimeTopologyPB = _message_class("RuntimeTopologyPB")
SnapshotCapabilityPB = _message_class("SnapshotCapabilityPB")
CacheGeometryPB = _message_class("CacheGeometryPB")
CacheComponentPB = _message_class("CacheComponentPB")
VllmKvEventSchemaPB = _message_class("VllmKvEventSchemaPB")
SglangKvEventSchemaPB = _message_class("SglangKvEventSchemaPB")
KvEventBootstrapInfoPB = _message_class("KvEventBootstrapInfoPB")
KvEventBootstrapInfoRequestPB = _message_class("KvEventBootstrapInfoRequestPB")

KvEventBootstrapErrorCode = _enum_type_wrapper.EnumTypeWrapper(
    DESCRIPTOR.enum_types_by_name["KvEventBootstrapErrorCode"]
)
KV_EVENT_BOOTSTRAP_OK = KvEventBootstrapErrorCode.KV_EVENT_BOOTSTRAP_OK
KV_EVENT_BOOTSTRAP_UNAVAILABLE = (
    KvEventBootstrapErrorCode.KV_EVENT_BOOTSTRAP_UNAVAILABLE
)
KV_EVENT_BOOTSTRAP_INCOMPATIBLE = (
    KvEventBootstrapErrorCode.KV_EVENT_BOOTSTRAP_INCOMPATIBLE
)

_sym_db.RegisterMessage(StatusVersionPB)
_sym_db.RegisterMessage(WorkerStatusPB)
_sym_db.RegisterMessage(KvCacheBlockListPB)
_sym_db.RegisterMessage(KvCacheBlocksRequestPB)
_sym_db.RegisterMessage(OptionalUInt64PB)
_sym_db.RegisterMessage(OptionalInt64PB)
_sym_db.RegisterMessage(StringListPB)
_sym_db.RegisterMessage(Int64ListPB)
_sym_db.RegisterMessage(UInt64ListPB)
_sym_db.RegisterMessage(DoubleListPB)
_sym_db.RegisterMessage(BoolListPB)
_sym_db.RegisterMessage(TypedSettingPB)
_sym_db.RegisterMessage(EventTransportPB)
_sym_db.RegisterMessage(RuntimeTopologyPB)
_sym_db.RegisterMessage(SnapshotCapabilityPB)
_sym_db.RegisterMessage(CacheGeometryPB)
_sym_db.RegisterMessage(CacheComponentPB)
_sym_db.RegisterMessage(VllmKvEventSchemaPB)
_sym_db.RegisterMessage(SglangKvEventSchemaPB)
_sym_db.RegisterMessage(KvEventBootstrapInfoPB)
_sym_db.RegisterMessage(KvEventBootstrapInfoRequestPB)
