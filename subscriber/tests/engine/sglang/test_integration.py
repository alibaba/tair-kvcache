"""Real UDS integration for SGLang bootstrap and snapshot decoding."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import grpc
import msgspec
import pytest

from subscriber.config import SubscriberConfig
from subscriber.engine.sglang import SGLangAdapter
from subscriber.proto import (
    engine_service_rpc_pb2,
    engine_service_rpc_pb2_grpc,
)
from subscriber.types import BlockSnapshot, BlockSnapshotItem

pytestmark = pytest.mark.integration


class _SglangControlService(engine_service_rpc_pb2_grpc.KvEventControlServiceServicer):
    def __init__(self) -> None:
        self.snapshot_requests = 0

    async def GetKvEventBootstrapInfo(
        self,
        request: engine_service_rpc_pb2.KvEventBootstrapInfoRequestPB,
        context: grpc.aio.ServicerContext,
    ) -> engine_service_rpc_pb2.KvEventBootstrapInfoPB:
        del request, context
        response = engine_service_rpc_pb2.KvEventBootstrapInfoPB(
            protocol_version=1,
            engine_kind="sglang",
            err_code=engine_service_rpc_pb2.KV_EVENT_BOOTSTRAP_OK,
        )
        response.event_transport.live_endpoint = "tcp://127.0.0.1:5557"
        response.event_transport.topic = "kv-events"
        response.event_transport.replay_supported = True
        response.event_transport.replay_endpoint = "tcp://127.0.0.1:5558"
        response.event_transport.serialization = "msgpack-v1"
        response.runtime_topology.data_parallel_size = 1
        response.runtime_topology.tensor_parallel_size = 1
        response.runtime_topology.pipeline_parallel_size = 1
        response.snapshot.supported = True
        response.snapshot.versioned = True
        response.sglang.cache_key_mode = "token"
        response.sglang.event_schema_version = 2
        response.sglang.native_hash_algorithm = "sglang-radix-native-int64"
        response.components.add(
            component_id=0,
            component_kind="full_attention",
        ).geometry.block_size_tokens = 16
        response.components.add(
            component_id=2,
            component_kind="mamba",
        ).geometry.block_size_tokens = 128
        return response

    async def GetAllKvCacheBlocks(
        self,
        request: engine_service_rpc_pb2.KvCacheBlocksRequestPB,
        context: grpc.aio.ServicerContext,
    ) -> engine_service_rpc_pb2.KvCacheBlockListPB:
        del request, context
        self.snapshot_requests += 1
        return engine_service_rpc_pb2.KvCacheBlockListPB(
            raw_snapshot=msgspec.msgpack.encode([(-11, 0), (22, 2)]),
            snapshot_version=9,
        )


def _short_uds_path() -> Path:
    return Path("/tmp") / f"sglang-snapshot-{uuid4().hex[:12]}.sock"


async def test_adapter_round_trips_mixed_component_snapshot_over_uds() -> None:
    service = _SglangControlService()
    server = grpc.aio.server()
    engine_service_rpc_pb2_grpc.add_KvEventControlServiceServicer_to_server(
        service, server
    )
    uds_path = _short_uds_path()
    assert server.add_insecure_port(f"unix://{uds_path}") == 1
    await server.start()

    adapter = SGLangAdapter(
        SubscriberConfig(
            engine_kv_event_control_uds_path=str(uds_path),
            engine_snapshot_full_sync_interval_s=60.0,
            snapshot_kv_event_pipeline_enabled=True,
        )
    )
    events = adapter.subscribe_snapshot_events()
    try:
        await adapter.fetch_kv_event_bootstrap()
        event = await asyncio.wait_for(anext(events), timeout=5.0)
    finally:
        await events.aclose()
        await adapter.close()
        await server.stop(None)
        uds_path.unlink(missing_ok=True)

    assert service.snapshot_requests == 1
    assert event.batches[0].events == [
        BlockSnapshot(
            medium="GPU",
            block_size=0,
            items=[
                BlockSnapshotItem(block_hash=-11, group_idx=0),
                BlockSnapshotItem(block_hash=22, group_idx=2),
            ],
            snapshot_version=9,
        )
    ]
