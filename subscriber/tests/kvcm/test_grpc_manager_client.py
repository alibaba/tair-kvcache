from __future__ import annotations

import asyncio
import os
import socket
import time
import uuid

import grpc
import httpx
import pytest
from pytest_mock import MockerFixture

from subscriber.kvcm.enum import KvcmStorageType
from subscriber.kvcm.errors import (
    KvcmResponseRejectedError,
    KvcmUnavailableError,
    report_event_transport_diagnostics,
)
from subscriber.kvcm.grpc_manager_client import (
    GrpcKvCacheManagerClient,
    _get_cluster_info_response_to_dict,
    _register_instance_request_from_dict,
    _report_event_request_from_dict,
    _report_event_request_to_wire_bytes,
    _report_event_response_to_dict,
)
from subscriber.proto import kvcm_meta_service_pb2, kvcm_meta_service_pb2_grpc


class FakeKvcmMetaService(kvcm_meta_service_pb2_grpc.MetaServiceServicer):
    def __init__(
        self,
        *,
        report_status: int = kvcm_meta_service_pb2.OK,
        cluster_status: int = kvcm_meta_service_pb2.OK,
        leader_port: int = 0,
        item_results: list[int] | None = None,
        report_status_by_event: dict[int, int] | None = None,
        heartbeat_started: asyncio.Event | None = None,
        heartbeat_release: asyncio.Event | None = None,
        cluster_abort_code: grpc.StatusCode | None = None,
    ) -> None:
        self.report_status = report_status
        self.cluster_status = cluster_status
        self.leader_port = leader_port
        self.item_results = item_results or []
        self.report_status_by_event = report_status_by_event or {}
        self.heartbeat_started = heartbeat_started
        self.heartbeat_release = heartbeat_release
        self.cluster_abort_code = cluster_abort_code
        self.register_requests: list[kvcm_meta_service_pb2.RegisterInstanceRequest] = []
        self.report_requests: list[kvcm_meta_service_pb2.ReportEventRequest] = []
        self.cluster_requests: list[kvcm_meta_service_pb2.GetClusterInfoRequest] = []

    async def RegisterInstance(
        self,
        request: kvcm_meta_service_pb2.RegisterInstanceRequest,
        context: grpc.aio.ServicerContext,
    ) -> kvcm_meta_service_pb2.RegisterInstanceResponse:
        self.register_requests.append(request)
        return kvcm_meta_service_pb2.RegisterInstanceResponse(
            header={"status": {"code": kvcm_meta_service_pb2.OK}}
        )

    async def ReportEvent(
        self,
        request: kvcm_meta_service_pb2.ReportEventRequest,
        context: grpc.aio.ServicerContext,
    ) -> kvcm_meta_service_pb2.ReportEventResponse:
        self.report_requests.append(request)
        if (
            request.events
            and request.events[0].event_type == kvcm_meta_service_pb2.EVENT_HEARTBEAT
            and self.heartbeat_started is not None
            and self.heartbeat_release is not None
        ):
            self.heartbeat_started.set()
            await self.heartbeat_release.wait()
        event_type = request.events[0].event_type if request.events else 0
        return kvcm_meta_service_pb2.ReportEventResponse(
            header={
                "status": {
                    "code": self.report_status_by_event.get(
                        event_type, self.report_status
                    )
                }
            },
            item_results=self.item_results,
            snapshot_required=True,
        )

    async def GetClusterInfo(
        self,
        request: kvcm_meta_service_pb2.GetClusterInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> kvcm_meta_service_pb2.GetClusterInfoResponse:
        self.cluster_requests.append(request)
        if self.cluster_abort_code is not None:
            await context.abort(self.cluster_abort_code, "cluster unavailable")
        return kvcm_meta_service_pb2.GetClusterInfoResponse(
            header={"status": {"code": self.cluster_status}},
            leader_endpoint={
                "host": "127.0.0.1",
                "meta_rpc_port": self.leader_port,
            },
        )


async def _start_fake_kvcm_server(
    service: FakeKvcmMetaService,
) -> tuple[grpc.aio.Server, int]:
    server = grpc.aio.server()
    kvcm_meta_service_pb2_grpc.add_MetaServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    assert port != 0
    await server.start()
    return server, port


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_kvcm_proto_critical_field_numbers_match_authoritative_schema() -> None:
    register = kvcm_meta_service_pb2.RegisterInstanceRequest.DESCRIPTOR
    report = kvcm_meta_service_pb2.ReportEventRequest.DESCRIPTOR
    cluster = kvcm_meta_service_pb2.GetClusterInfoResponse.DESCRIPTOR

    assert register.fields_by_name["location_spec_infos"].number == 5
    assert register.fields_by_name["model_deployment"].number == 6
    assert register.fields_by_name["default_query_type"].number == 8
    assert report.fields_by_name["events"].number == 4
    assert report.fields_by_name["storage_type"].number == 5
    assert cluster.fields_by_name["leader_endpoint"].number == 4


def test_kvcm_storage_types_match_authoritative_proto() -> None:
    assert {storage_type.value for storage_type in KvcmStorageType} == set(
        kvcm_meta_service_pb2.StorageType.DESCRIPTOR.values_by_name
    )


def test_kvcm_grpc_stub_uses_full_package_rpc_paths() -> None:
    class FakeChannel:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def unary_unary(self, path: str, **kwargs: object) -> object:
            self.paths.append(path)
            return object()

    channel = FakeChannel()
    kvcm_meta_service_pb2_grpc.MetaServiceStub(channel)

    assert channel.paths == [
        "/kv_cache_manager.proto.meta.MetaService/RegisterInstance",
        "/kv_cache_manager.proto.meta.MetaService/GetClusterInfo",
        "/kv_cache_manager.proto.meta.MetaService/ReportEvent",
    ]


def test_register_instance_dict_maps_to_proto() -> None:
    request = _register_instance_request_from_dict(
        {
            "trace_id": "trace-1",
            "instance_group": "group-a",
            "instance_id": "deploy_16",
            "block_size": 16,
            "location_spec_infos": [{"name": "full_0", "size": 4096}],
            "model_deployment": {
                "model_name": "default",
                "dtype": "bytes",
                "use_mla": False,
                "tp_size": 2,
                "dp_size": 1,
                "pp_size": 1,
                "use_eagle_pop": True,
            },
            "location_spec_groups": [
                {"name": "full_0", "spec_names": ["full_0"]},
            ],
            "default_query_type": "QT_PREFIX_MATCH_WITH_MAMBA",
        }
    )

    assert request.trace_id == "trace-1"
    assert request.location_spec_infos[0].name == "full_0"
    assert request.location_spec_infos[0].size == 4096
    assert request.model_deployment.tp_size == 2
    assert request.model_deployment.use_eagle_pop is True
    assert list(request.location_spec_groups[0].spec_names) == ["full_0"]
    assert (
        request.default_query_type == kvcm_meta_service_pb2.QT_PREFIX_MATCH_WITH_MAMBA
    )


def test_report_event_dict_maps_all_subscriber_event_shapes() -> None:
    data = {
        "trace_id": "trace-2",
        "instance_id": "deploy_16",
        "host_ip_port": "10.0.0.8:9000",
        "storage_type": "ST_EVENT_REPORT_L1P5",
        "events": [
            {
                "event_type": "EVENT_NODE_REGISTER",
                "node_register": {"mediums": ["hbm"]},
            },
            {
                "event_type": "EVENT_HEARTBEAT",
                "heartbeat": {"system_status": {"alive": "1"}},
            },
            {
                "event_type": "EVENT_BLOCK_ADD",
                "block_add": {
                    "block_key": "11",
                    "medium": "hbm",
                    "specs": [{"name": "full_0", "uri": "vllm://host/hbm"}],
                },
            },
            {
                "event_type": "EVENT_BLOCK_DELETE",
                "block_delete": {
                    "block_key": "11",
                    "medium": "hbm",
                    "spec_names": ["full_0"],
                },
            },
            {"event_type": "EVENT_HOST_DOWN", "host_down": {}},
            {
                "event_type": "EVENT_BLOCK_SNAPSHOT",
                "block_snapshot": {
                    "blocks": [
                        {
                            "block_key": "11",
                            "medium": "hbm",
                            "specs": [
                                {
                                    "name": "full_0",
                                    "uri": "vllm://host/hbm",
                                }
                            ],
                        }
                    ]
                },
            },
        ],
    }
    request = _report_event_request_from_dict(data)
    fast_request = kvcm_meta_service_pb2.ReportEventRequest.FromString(
        _report_event_request_to_wire_bytes(data)
    )

    assert request.storage_type == kvcm_meta_service_pb2.ST_EVENT_REPORT_L1P5
    assert list(request.events[0].node_register.mediums) == ["hbm"]
    assert request.events[1].heartbeat.system_status["alive"] == "1"
    assert request.events[2].block_add.specs[0].name == "full_0"
    assert list(request.events[3].block_delete.spec_names) == ["full_0"]
    assert request.events[4].HasField("host_down")
    assert request.events[5].block_snapshot.blocks[0].specs[0].uri == "vllm://host/hbm"
    assert fast_request == request


def test_report_event_wire_bytes_match_proto_for_many_add_delete_events() -> None:
    data = {
        "trace_id": "trace-many-events",
        "instance_id": "deploy_16",
        "host_ip_port": "10.0.0.8:9000",
        "storage_type": "ST_EVENT_REPORT_L1P5",
        "events": [
            {
                "event_type": "EVENT_BLOCK_ADD",
                "block_add": {
                    "block_key": str(block_key),
                    "medium": "hbm",
                    "specs": [{"name": "full_0", "uri": "vllm://host/hbm"}],
                },
            }
            for block_key in range(16)
        ]
        + [
            {
                "event_type": "EVENT_BLOCK_DELETE",
                "block_delete": {
                    "block_key": str(block_key),
                    "medium": "hbm",
                    "spec_names": ["full_0"],
                },
            }
            for block_key in range(16)
        ],
    }

    request = _report_event_request_from_dict(data)
    fast_request = kvcm_meta_service_pb2.ReportEventRequest.FromString(
        _report_event_request_to_wire_bytes(data)
    )

    assert fast_request == request


def test_report_event_wire_bytes_match_proto_for_4k_block_snapshot() -> None:
    data = {
        "trace_id": "trace-4k-snapshot",
        "instance_id": "deploy_16",
        "host_ip_port": "10.0.0.8:9000",
        "storage_type": "ST_EVENT_REPORT_L1P5",
        "events": [
            {
                "event_type": "EVENT_BLOCK_SNAPSHOT",
                "block_snapshot": {
                    "medium": "hbm",
                    "blocks": [
                        {
                            "block_key": str(block_key),
                            "medium": "hbm" if block_key % 2 == 0 else "cpu",
                            "specs": [
                                {
                                    "name": f"full_{block_key % 2}",
                                    "uri": "vllm://host/hbm",
                                },
                                {
                                    "name": "cpu_0",
                                    "uri": "vllm://host/cpu",
                                },
                            ],
                        }
                        for block_key in range(4_096)
                    ],
                },
            }
        ],
    }

    request = _report_event_request_from_dict(data)
    fast_request = kvcm_meta_service_pb2.ReportEventRequest.FromString(
        _report_event_request_to_wire_bytes(data)
    )

    assert fast_request == request


def test_report_event_proto_round_trip_preserves_storage_type_and_oneof() -> None:
    request = _report_event_request_from_dict(
        {
            "storage_type": "ST_EVENT_REPORT_L1P5",
            "events": [
                {
                    "event_type": "EVENT_HEARTBEAT",
                    "heartbeat": {"system_status": {"alive": "1"}},
                }
            ],
        }
    )
    parsed = kvcm_meta_service_pb2.ReportEventRequest()

    parsed.ParseFromString(request.SerializeToString())

    assert parsed.storage_type == kvcm_meta_service_pb2.ST_EVENT_REPORT_L1P5
    assert parsed.events[0].HasField("heartbeat")
    assert parsed.events[0].heartbeat.system_status["alive"] == "1"


@pytest.mark.parametrize("storage_type", ["ST_EVENT_REPORT", 99, -1])
def test_report_event_rejects_unknown_storage_type(storage_type: object) -> None:
    data = {
        "storage_type": storage_type,
        "events": [{"event_type": "EVENT_HEARTBEAT"}],
    }

    with pytest.raises(ValueError):
        _report_event_request_from_dict(data)
    with pytest.raises(ValueError):
        _report_event_request_to_wire_bytes(data)


@pytest.mark.parametrize("event_type", [99, -1])
def test_report_event_rejects_unknown_numeric_event_type(event_type: int) -> None:
    data = {
        "storage_type": "ST_EVENT_REPORT_L1P5",
        "events": [{"event_type": event_type}],
    }

    with pytest.raises(ValueError):
        _report_event_request_from_dict(data)
    with pytest.raises(ValueError):
        _report_event_request_to_wire_bytes(data)


@pytest.mark.parametrize(
    "event",
    [{}, {"event_type": ""}, {"event_type": "EVENT_FUTURE"}],
)
def test_report_event_dict_defaults_unrecognized_event_type_to_unspecified(
    event: dict[str, str],
) -> None:
    request = _report_event_request_from_dict(
        {
            "storage_type": "ST_EVENT_REPORT_L1P5",
            "events": [event],
        }
    )

    assert request.events[0].event_type == kvcm_meta_service_pb2.EVENT_UNSPECIFIED
    assert request.events[0].WhichOneof("event_params") is None


def test_report_event_wire_bytes_preserve_unspecified_oneof_behavior() -> None:
    data = {
        "storage_type": "ST_EVENT_REPORT_L1P5",
        "events": [
            {
                "block_add": {
                    "block_key": "11",
                    "medium": "hbm",
                    "specs": [{"name": "full_0", "uri": "vllm://host/hbm"}],
                },
            },
            {
                "event_type": "EVENT_FUTURE",
                "block_delete": {
                    "block_key": "12",
                    "medium": "hbm",
                    "spec_names": ["full_0"],
                },
            },
        ],
    }

    request = _report_event_request_from_dict(data)
    fast_request = kvcm_meta_service_pb2.ReportEventRequest.FromString(
        _report_event_request_to_wire_bytes(data)
    )

    assert fast_request == request
    assert [event.WhichOneof("event_params") for event in fast_request.events] == [
        None,
        None,
    ]


def test_grpc_responses_convert_to_domain_dict_shape() -> None:
    report_response = kvcm_meta_service_pb2.ReportEventResponse(
        header={"status": {"code": kvcm_meta_service_pb2.OK, "message": ""}},
        item_results=[kvcm_meta_service_pb2.INTERNAL_ERROR],
        committed_snapshot_version="7",
        retry_after_ms=123,
        snapshot_required=True,
        extra_info='{"source":"test"}',
    )
    cluster_response = kvcm_meta_service_pb2.GetClusterInfoResponse(
        header={"status": {"code": kvcm_meta_service_pb2.OK}},
        self_node_id="node-a",
        leader_node_id="node-b",
        leader_endpoint={
            "node_id": "node-b",
            "host": "10.0.0.9",
            "meta_rpc_port": 6381,
            "meta_http_port": 6382,
        },
    )

    report = _report_event_response_to_dict(report_response)
    cluster = _get_cluster_info_response_to_dict(cluster_response)

    assert report["header"]["status"]["code"] == "OK"
    assert report["item_results"] == ["INTERNAL_ERROR"]
    assert report["snapshot_required"] is True
    assert cluster["leader_endpoint"]["host"] == "10.0.0.9"
    assert cluster["leader_endpoint"]["meta_rpc_port"] == 6381


async def test_grpc_manager_client_sends_real_aio_rpc_requests() -> None:
    service = FakeKvcmMetaService()
    server, port = await _start_fake_kvcm_server(service)
    client = GrpcKvCacheManagerClient(
        f"127.0.0.1:{port}",
        auto_discover_leader=False,
        request_timeout_seconds=1.0,
    )

    try:
        await client.start()
        register = await client.register_instance(
            {
                "trace_id": "trace-register",
                "instance_id": "deploy_16",
                "block_size": 16,
            }
        )
        report = await client.report_event(
            {
                "trace_id": "trace-report",
                "instance_id": "deploy_16",
                "storage_type": "ST_EVENT_REPORT_L1P5",
                "events": [{"event_type": "EVENT_HEARTBEAT"}],
            }
        )
    finally:
        await client.close()
        await server.stop(None)

    assert register["header"]["status"]["code"] == "OK"
    assert report["snapshot_required"] is True
    assert service.register_requests[0].trace_id == "trace-register"
    assert service.register_requests[0].block_size == 16
    assert service.report_requests[0].events[0].event_type == (
        kvcm_meta_service_pb2.EVENT_HEARTBEAT
    )


async def test_grpc_report_event_returns_wire_diagnostics() -> None:
    service = FakeKvcmMetaService()
    server, port = await _start_fake_kvcm_server(service)
    client = GrpcKvCacheManagerClient(
        f"127.0.0.1:{port}",
        auto_discover_leader=False,
        request_timeout_seconds=1.0,
    )
    request = {
        "trace_id": "trace-report",
        "instance_id": "deploy_16",
        "storage_type": "ST_EVENT_REPORT_L1P5",
        "events": [{"event_type": "EVENT_HEARTBEAT"}],
    }

    try:
        await client.start()
        response = await client.report_event(request)
    finally:
        await client.close()
        await server.stop(None)

    assert response["_subscriber_request_bytes"] == len(
        _report_event_request_to_wire_bytes(request)
    )
    assert response["_subscriber_wire_encode_ms"] >= 0
    assert response["_subscriber_grpc_call_ms"] >= 0


async def test_grpc_report_event_logs_completed_attempt_dimensions(
    mocker: MockerFixture,
) -> None:
    service = FakeKvcmMetaService()
    server, port = await _start_fake_kvcm_server(service)
    client = GrpcKvCacheManagerClient(
        f"127.0.0.1:{port}",
        auto_discover_leader=False,
        request_timeout_seconds=1.0,
    )
    debug = mocker.patch("subscriber.kvcm.grpc_manager_client.logger.debug")
    mocker.patch(
        "subscriber.kvcm.grpc_manager_client.logger.is_debug_enabled",
        return_value=True,
    )

    try:
        await client.start()
        await client.report_event(
            {
                "storage_type": "ST_EVENT_REPORT_L1P5",
                "events": [{"event_type": "EVENT_HEARTBEAT"}],
            }
        )
    finally:
        await client.close()
        await server.stop(None)

    report_log = next(
        entry
        for entry in debug.call_args_list
        if entry.args == ("kvcm grpc request completed",)
        and entry.kwargs["tags"]["method"] == "ReportEvent"
    )
    tags = report_log.kwargs["tags"]
    assert tags["attempt"] == 1
    assert tags["grpc_status"] == "OK"
    assert tags["request_bytes"] > 0
    assert tags["grpc_call_ms"] >= 0


async def test_grpc_manager_client_switches_to_leader_meta_rpc_port() -> None:
    leader_service = FakeKvcmMetaService()
    leader_server, leader_port = await _start_fake_kvcm_server(leader_service)
    follower_service = FakeKvcmMetaService(
        report_status=kvcm_meta_service_pb2.SERVER_NOT_LEADER,
        cluster_status=kvcm_meta_service_pb2.INTERNAL_ERROR,
        leader_port=leader_port,
    )
    follower_server, follower_port = await _start_fake_kvcm_server(follower_service)
    client = GrpcKvCacheManagerClient(
        f"127.0.0.1:{follower_port}",
        leader_retry_base_interval_seconds=0.0,
        request_timeout_seconds=1.0,
    )

    try:
        await client.start()
        follower_service.cluster_status = kvcm_meta_service_pb2.OK
        follower_service.cluster_requests.clear()
        report = await client.report_event(
            {
                "trace_id": "trace-report",
                "instance_id": "deploy_16",
                "storage_type": "ST_EVENT_REPORT_L1P5",
                "events": [{"event_type": "EVENT_HOST_DOWN"}],
            }
        )
    finally:
        await client.close()
        await follower_server.stop(None)
        await leader_server.stop(None)

    assert report["header"]["status"]["code"] == "OK"
    assert report["_subscriber_retry_count"] == 1
    assert len(follower_service.report_requests) == 1
    assert len(follower_service.cluster_requests) == 1
    assert len(leader_service.report_requests) == 1


async def test_grpc_retry_then_rejection_preserves_retry_count() -> None:
    leader_service = FakeKvcmMetaService(
        report_status=kvcm_meta_service_pb2.INVALID_ARGUMENT,
    )
    leader_server, leader_port = await _start_fake_kvcm_server(leader_service)
    follower_service = FakeKvcmMetaService(
        report_status=kvcm_meta_service_pb2.SERVER_NOT_LEADER,
        cluster_status=kvcm_meta_service_pb2.INTERNAL_ERROR,
        leader_port=leader_port,
    )
    follower_server, follower_port = await _start_fake_kvcm_server(follower_service)
    client = GrpcKvCacheManagerClient(
        f"127.0.0.1:{follower_port}",
        leader_retry_base_interval_seconds=0.0,
        request_timeout_seconds=1.0,
    )

    try:
        await client.start()
        follower_service.cluster_status = kvcm_meta_service_pb2.OK
        with pytest.raises(KvcmResponseRejectedError) as error:
            await client.report_event(
                {
                    "storage_type": "ST_EVENT_REPORT_L1P5",
                    "events": [{"event_type": "EVENT_HOST_DOWN"}],
                }
            )
    finally:
        await client.close()
        await follower_server.stop(None)
        await leader_server.stop(None)

    assert error.value.status_code == "INVALID_ARGUMENT"
    assert error.value.retry_count == 1
    assert error.value.request_bytes is not None
    assert error.value.request_bytes > 0
    assert error.value.wire_encode_ms is not None
    assert error.value.wire_encode_ms >= 0
    assert error.value.grpc_call_ms is not None
    assert error.value.grpc_call_ms >= 0


async def test_grpc_retry_then_transport_failure_preserves_retry_count() -> None:
    unavailable_leader_port = _unused_loopback_port()
    follower_service = FakeKvcmMetaService(
        report_status=kvcm_meta_service_pb2.SERVER_NOT_LEADER,
        cluster_status=kvcm_meta_service_pb2.INTERNAL_ERROR,
        leader_port=unavailable_leader_port,
    )
    follower_server, follower_port = await _start_fake_kvcm_server(follower_service)
    client = GrpcKvCacheManagerClient(
        f"127.0.0.1:{follower_port}",
        leader_retry_base_interval_seconds=0.0,
        request_timeout_seconds=1.0,
    )

    try:
        await client.start()
        follower_service.cluster_status = kvcm_meta_service_pb2.OK
        with pytest.raises(KvcmUnavailableError) as error:
            await client.report_event(
                {
                    "storage_type": "ST_EVENT_REPORT_L1P5",
                    "events": [{"event_type": "EVENT_HOST_DOWN"}],
                }
            )
    finally:
        await client.close()
        await follower_server.stop(None)

    assert error.value.status_code == "GRPC_UNAVAILABLE"
    assert error.value.reason == "transport"
    assert error.value.retry_count == 1


async def test_grpc_leader_switch_does_not_cancel_inflight_heartbeat() -> None:
    heartbeat_started = asyncio.Event()
    heartbeat_release = asyncio.Event()
    leader_service = FakeKvcmMetaService()
    leader_server, leader_port = await _start_fake_kvcm_server(leader_service)
    follower_service = FakeKvcmMetaService(
        cluster_status=kvcm_meta_service_pb2.INTERNAL_ERROR,
        leader_port=leader_port,
        report_status_by_event={
            kvcm_meta_service_pb2.EVENT_HOST_DOWN: (
                kvcm_meta_service_pb2.SERVER_NOT_LEADER
            )
        },
        heartbeat_started=heartbeat_started,
        heartbeat_release=heartbeat_release,
    )
    follower_server, follower_port = await _start_fake_kvcm_server(follower_service)
    client = GrpcKvCacheManagerClient(
        f"127.0.0.1:{follower_port}",
        leader_retry_base_interval_seconds=0.0,
        request_timeout_seconds=1.0,
    )

    try:
        await client.start()
        follower_service.cluster_status = kvcm_meta_service_pb2.OK
        follower_service.cluster_requests.clear()
        heartbeat_task = asyncio.create_task(
            client.report_event(
                {
                    "storage_type": "ST_EVENT_REPORT_L1P5",
                    "events": [{"event_type": "EVENT_HEARTBEAT"}],
                }
            )
        )
        await asyncio.wait_for(heartbeat_started.wait(), timeout=0.5)
        switched_report_task = asyncio.create_task(
            client.report_event(
                {
                    "storage_type": "ST_EVENT_REPORT_L1P5",
                    "events": [{"event_type": "EVENT_HOST_DOWN"}],
                }
            )
        )
        await asyncio.sleep(0.05)

        assert follower_service.cluster_requests
        assert not heartbeat_task.done()
        heartbeat_release.set()
        heartbeat_response = await heartbeat_task
        switched_response = await switched_report_task
    finally:
        heartbeat_release.set()
        await client.close()
        await follower_server.stop(None)
        await leader_server.stop(None)

    assert heartbeat_response["header"]["status"]["code"] == "OK"
    assert switched_response["header"]["status"]["code"] == "OK"
    assert len(leader_service.report_requests) == 1


async def test_grpc_server_not_leader_with_failed_discovery_is_unavailable() -> None:
    follower_service = FakeKvcmMetaService(
        report_status=kvcm_meta_service_pb2.SERVER_NOT_LEADER
    )
    follower_server, follower_port = await _start_fake_kvcm_server(follower_service)
    client = GrpcKvCacheManagerClient(
        f"127.0.0.1:{follower_port}",
        leader_retry_count=2,
        leader_retry_base_interval_seconds=0.0,
        request_timeout_seconds=1.0,
    )

    try:
        await client.start()
        with pytest.raises(KvcmUnavailableError):
            await client.report_event(
                {
                    "storage_type": "ST_EVENT_REPORT_L1P5",
                    "events": [{"event_type": "EVENT_HEARTBEAT"}],
                }
            )
    finally:
        await client.close()
        await follower_server.stop(None)

    assert len(follower_service.report_requests) == 3


async def test_grpc_refused_target_is_unavailable_without_deadline_wait() -> None:
    client = GrpcKvCacheManagerClient(
        f"127.0.0.1:{_unused_loopback_port()}",
        auto_discover_leader=False,
        request_timeout_seconds=0.4,
    )

    try:
        await client.start()
        started_at = time.monotonic()
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await client.report_event(
                {
                    "storage_type": "ST_EVENT_REPORT_L1P5",
                    "events": [{"event_type": "EVENT_HEARTBEAT"}],
                }
            )
    finally:
        await client.close()

    assert error.value.code() == grpc.StatusCode.UNAVAILABLE
    assert time.monotonic() - started_at < 0.3
    diagnostics = report_event_transport_diagnostics(error.value)
    assert diagnostics.request_bytes is not None
    assert diagnostics.request_bytes > 0
    assert diagnostics.wire_encode_ms is not None
    assert diagnostics.wire_encode_ms >= 0
    assert diagnostics.grpc_call_ms is not None
    assert diagnostics.grpc_call_ms >= 0


async def test_grpc_report_event_before_rpc_sets_zero_rpc_diagnostic(
    mocker: MockerFixture,
) -> None:
    client = GrpcKvCacheManagerClient("127.0.0.1:1", auto_discover_leader=False)
    mocker.patch.object(
        client,
        "_get_stub",
        side_effect=KvcmUnavailableError("KVCM gRPC channel is unavailable"),
    )

    try:
        with pytest.raises(KvcmUnavailableError) as error:
            await client.report_event(
                {
                    "storage_type": "ST_EVENT_REPORT_L1P5",
                    "events": [{"event_type": "EVENT_HEARTBEAT"}],
                }
            )
    finally:
        await client.close()

    assert error.value.request_bytes is not None
    assert error.value.request_bytes > 0
    assert error.value.wire_encode_ms is not None
    assert error.value.wire_encode_ms >= 0
    assert error.value.grpc_call_ms == 0.0


async def test_failed_grpc_discovery_does_not_trigger_rapid_refresh() -> None:
    service = FakeKvcmMetaService(cluster_abort_code=grpc.StatusCode.UNAVAILABLE)
    server, port = await _start_fake_kvcm_server(service)
    client = GrpcKvCacheManagerClient(
        f"127.0.0.1:{port}",
        min_discover_interval_seconds=0.01,
        discovery_refresh_interval_seconds=1,
        request_timeout_seconds=0.1,
    )

    try:
        await client.start()
        await asyncio.sleep(0.05)
    finally:
        await client.close()
        await server.stop(None)

    assert len(service.cluster_requests) == 1


async def test_grpc_manager_client_rejects_non_ok_response() -> None:
    service = FakeKvcmMetaService(
        report_status=kvcm_meta_service_pb2.INTERNAL_ERROR,
        item_results=[kvcm_meta_service_pb2.INTERNAL_ERROR],
    )
    server, port = await _start_fake_kvcm_server(service)
    client = GrpcKvCacheManagerClient(
        f"127.0.0.1:{port}",
        auto_discover_leader=False,
        request_timeout_seconds=1.0,
    )

    try:
        with pytest.raises(KvcmResponseRejectedError, match="INTERNAL_ERROR"):
            await client.report_event(
                {
                    "trace_id": "trace-report",
                    "instance_id": "deploy_16",
                    "storage_type": "ST_EVENT_REPORT_L1P5",
                    "events": [{"event_type": "EVENT_HEARTBEAT"}],
                }
            )
    finally:
        await client.close()
        await server.stop(None)


async def test_grpc_manager_client_unavailable_target_raises_retryable() -> None:
    client = GrpcKvCacheManagerClient(
        "",
        auto_discover_leader=False,
        request_timeout_seconds=1.0,
    )

    with pytest.raises(KvcmUnavailableError, match="target is unavailable"):
        await client.report_event(
            {"storage_type": "ST_EVENT_REPORT_L1P5", "events": []}
        )


async def _setup_real_event_report_instance_group(
    admin_url: str,
    *,
    instance_group: str,
    storage_name: str,
) -> None:
    async with httpx.AsyncClient(
        base_url=admin_url.rstrip("/"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=2.0,
    ) as http_client:
        storage = {
            "global_unique_name": storage_name,
            "storage_type": "ST_EVENT_REPORT_L1P5",
            "event_report": {
                "heartbeat_timeout_ms": 1000,
                "cleanup_grace_ms": 2000,
                "liveness_check_interval_ms": 200,
            },
            "check_storage_available_when_open": False,
        }
        add_response = await http_client.post(
            "/api/addStorage",
            json={"trace_id": "real-grpc-add-storage", "storage": storage},
        )
        add_response.raise_for_status()
        add_body = add_response.json()
        add_code = add_body.get("header", {}).get("status", {}).get("code")
        assert add_code in {"OK", "DUPLICATE_ENTITY"}, add_body

        group_response = await http_client.post(
            "/api/createInstanceGroup",
            json={
                "trace_id": "real-grpc-create-instance-group",
                "instance_group": {
                    "name": instance_group,
                    "storage_candidates": ["nfs_01"],
                    "global_quota_group_name": "default_quota_group",
                    "max_instance_count": 100,
                    "quota": {
                        "capacity": 10737418240,
                        "quota_config": [{"storage_type": 4, "capacity": 10737418240}],
                    },
                    "cache_config": {
                        "reclaim_strategy": {
                            "storage_unique_name": "nfs_01",
                            "reclaim_policy": 1,
                            "trigger_strategy": {
                                "used_size": 1073741824,
                                "used_percentage": 0.8,
                            },
                            "trigger_period_seconds": 60,
                            "reclaim_step_size": 1073741824,
                            "reclaim_step_percentage": 10,
                        },
                        "data_storage_strategy": 2,
                        "meta_indexer_config": {
                            "max_key_count": 1000000,
                            "mutex_shard_num": 16,
                            "batch_key_size": 16,
                            "meta_storage_backend_config": {
                                "storage_type": "local",
                                "storage_uri": "",
                            },
                            "meta_cache_policy_config": {
                                "type": "LRU",
                                "capacity": 10000,
                            },
                        },
                    },
                    "event_report_storage_candidates": [storage_name],
                    "version": 1,
                },
            },
        )
        group_response.raise_for_status()
        group_body = group_response.json()
        group_code = group_body.get("header", {}).get("status", {}).get("code")
        assert group_code in {"OK", "DUPLICATE_ENTITY"}, group_body


# Opt-in real KVCM integration test.
#
# Setup in the KVCM repo:
#   bazel build //kv_cache_manager:kv_cache_manager_bin
#   bazel-bin/kv_cache_manager/kv_cache_manager_bin \
#     --env kvcm.service.rpc_port=56010 \
#     --env kvcm.service.http_port=56020 \
#     --env kvcm.service.admin_rpc_port=56031 \
#     --env kvcm.service.admin_http_port=56040 \
#     --env kvcm.service.enable_debug_service=false
#
# Run in this subscriber repo:
#   KVCM_REAL_GRPC_TARGET=127.0.0.1:56010 \
#   KVCM_REAL_ADMIN_HTTP_URL=http://127.0.0.1:56040 \
#   uv run pytest \
#     tests/kvcm/test_grpc_manager_client.py \
#     -k test_real_kvcm_grpc_register_and_report_event \
#     -vv
#
# KVCM_REAL_ADMIN_HTTP_URL is used only for setup; tested requests use gRPC.
async def test_real_kvcm_grpc_register_and_report_event() -> None:
    target = os.environ.get("KVCM_REAL_GRPC_TARGET")
    if not target:
        pytest.skip("set KVCM_REAL_GRPC_TARGET to run against a real KVCM server")

    suffix = uuid.uuid4().hex
    instance_id = f"subscriber-real-grpc-{suffix}"
    instance_group = "default"
    admin_url = os.environ.get("KVCM_REAL_ADMIN_HTTP_URL")
    if admin_url:
        instance_group = f"subscriber-real-grpc-group-{suffix}"
        await _setup_real_event_report_instance_group(
            admin_url,
            instance_group=instance_group,
            storage_name=f"subscriber-real-grpc-l1p5-{suffix}",
        )
    client = GrpcKvCacheManagerClient(
        target,
        auto_discover_leader=True,
        request_timeout_seconds=2.0,
    )

    try:
        await client.start()
        register_response = await client.register_instance(
            {
                "trace_id": "real-grpc-register",
                "instance_group": instance_group,
                "instance_id": instance_id,
                "block_size": 16,
                "location_spec_infos": [{"name": "vllm_16", "size": 16}],
                "location_spec_groups": [
                    {"name": "default", "spec_names": ["vllm_16"]}
                ],
                "model_deployment": {
                    "model_name": "subscriber-real-grpc",
                    "dtype": "bytes",
                    "tp_size": 1,
                    "dp_size": 1,
                    "pp_size": 1,
                },
                "default_query_type": "QT_PREFIX_MATCH_WITH_MAMBA",
            }
        )
        report_request = {
            "trace_id": "real-grpc-report",
            "instance_id": instance_id,
            "host_ip_port": "127.0.0.1:8080",
            "storage_type": "ST_EVENT_REPORT_L1P5",
            "events": [
                {
                    "event_type": "EVENT_NODE_REGISTER",
                    "node_register": {"mediums": ["hbm"]},
                },
                {"event_type": "EVENT_HEARTBEAT"},
            ],
        }
        try:
            report_response = await client.report_event(report_request)
        except KvcmResponseRejectedError as exc:
            if "EventReportBackend not found" in str(exc):
                pytest.skip(
                    "real KVCM server has no event_report_l1p5 backend configured"
                )
            raise
    finally:
        await client.close()

    assert register_response["header"]["status"]["code"] == "OK"
    assert report_response["header"]["status"]["code"] == "OK"
