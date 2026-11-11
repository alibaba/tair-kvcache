from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from subscriber.engine.kv_event_control_client import DashllmKvEventControlClient
from subscriber.engine.worker_status_client import DashllmWorkerStatusClient
from subscriber.proto import (
    engine_service_rpc_pb2,
    engine_service_rpc_pb2_grpc,
)


class _PathCapture:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def unary_unary(self, path: str, **_: Any) -> AsyncMock:
        self.paths.append(path)
        return AsyncMock()


class _InvokingChannel(_PathCapture):
    def __init__(self, response: object) -> None:
        super().__init__()
        self.response = response
        self.requests: list[object] = []
        self.close = AsyncMock()

    def unary_unary(self, path: str, **_: Any) -> Any:
        self.paths.append(path)

        async def invoke(request: object, *, timeout: float) -> object:
            del timeout
            self.requests.append(request)
            return self.response

        return invoke


def test_bootstrap_response_uses_versioned_engine_oneof_and_stable_fields() -> None:
    descriptor = engine_service_rpc_pb2.KvEventBootstrapInfoPB.DESCRIPTOR

    assert descriptor.fields_by_name["protocol_version"].number == 1
    assert descriptor.fields_by_name["engine_kind"].number == 2
    assert descriptor.fields_by_name["event_transport"].number == 3
    assert descriptor.fields_by_name["runtime_topology"].number == 4
    assert descriptor.fields_by_name["snapshot"].number == 5
    assert descriptor.fields_by_name["components"].number == 6
    assert descriptor.fields_by_name["err_code"].number == 7
    assert descriptor.fields_by_name["err_msg"].number == 8
    assert descriptor.fields_by_name["compatibility_settings"].number == 9
    assert descriptor.fields_by_name["diagnostic_settings"].number == 10
    assert descriptor.fields_by_name["vllm"].number == 20
    assert descriptor.fields_by_name["sglang"].number == 21
    assert descriptor.oneofs_by_name["engine_schema"].fields == [
        descriptor.fields_by_name["vllm"],
        descriptor.fields_by_name["sglang"],
    ]
    topic = engine_service_rpc_pb2.EventTransportPB.DESCRIPTOR.fields_by_name["topic"]
    assert topic.number == 2
    assert topic.type == topic.TYPE_STRING


def test_bootstrap_raw_wire_is_shared_without_a_proto_package() -> None:
    # protocol_version=1, engine_kind="sglang", sglang={cache_key_mode:"token"}.
    raw = b"\x08\x01\x12\x06sglang\xaa\x01\x07\x0a\x05token"

    response = engine_service_rpc_pb2.KvEventBootstrapInfoPB.FromString(raw)

    assert response.protocol_version == 1
    assert response.engine_kind == "sglang"
    assert response.WhichOneof("engine_schema") == "sglang"
    assert response.sglang.cache_key_mode == "token"


def test_component_geometry_preserves_unavailable_payload_size() -> None:
    component = engine_service_rpc_pb2.CacheComponentPB(
        component_id=0,
        component_kind="full_attention",
        geometry=engine_service_rpc_pb2.CacheGeometryPB(
            block_size_tokens=128,
            page_size_tokens=engine_service_rpc_pb2.OptionalUInt64PB(value=1),
        ),
    )

    assert component.geometry.HasField("page_size_tokens")
    assert not component.geometry.HasField("group_payload_size_bytes")


def test_typed_setting_retains_value_type() -> None:
    setting = engine_service_rpc_pb2.TypedSettingPB(
        name="vllm.use_eagle_pop",
        bool_value=True,
    )

    assert setting.WhichOneof("value") == "bool_value"
    assert setting.bool_value is True


def test_control_and_remote_status_use_separate_services() -> None:
    status_channel = _PathCapture()
    control_channel = _PathCapture()

    engine_service_rpc_pb2_grpc.RpcServiceStub(status_channel)
    engine_service_rpc_pb2_grpc.KvEventControlServiceStub(control_channel)

    assert status_channel.paths == ["/RpcService/GetWorkerStatus"]
    assert control_channel.paths == [
        "/KvEventControlService/GetKvEventBootstrapInfo",
        "/KvEventControlService/GetAllKvCacheBlocks",
    ]


async def test_client_uses_tcp_for_status_and_uds_for_bootstrap(
    mocker: Any,
) -> None:
    status_response = engine_service_rpc_pb2.WorkerStatusPB(alive=True)
    bootstrap_response = engine_service_rpc_pb2.KvEventBootstrapInfoPB(
        protocol_version=1,
        engine_kind="vllm",
        vllm=engine_service_rpc_pb2.VllmKvEventSchemaPB(),
    )
    status_channel = _InvokingChannel(status_response)
    control_channel = _InvokingChannel(bootstrap_response)
    insecure_channel = mocker.patch(
        "grpc.aio.insecure_channel",
        side_effect=[status_channel, control_channel],
    )
    status_client = DashllmWorkerStatusClient("127.0.0.1:18002")
    control_client = DashllmKvEventControlClient("/tmp/dashllm-kv-event-control.sock")

    assert (await status_client.get_worker_status(1.0)).alive is True
    assert (await control_client.get_kv_event_bootstrap_info(1.0)).engine_kind == "vllm"

    assert [call.args[0] for call in insecure_channel.call_args_list] == [
        "127.0.0.1:18002",
        "unix:///tmp/dashllm-kv-event-control.sock",
    ]
    assert status_channel.paths == ["/RpcService/GetWorkerStatus"]
    assert control_channel.paths == [
        "/KvEventControlService/GetKvEventBootstrapInfo",
        "/KvEventControlService/GetAllKvCacheBlocks",
    ]

    await status_client.close()
    await control_client.close()
    status_channel.close.assert_awaited_once()
    control_channel.close.assert_awaited_once()
