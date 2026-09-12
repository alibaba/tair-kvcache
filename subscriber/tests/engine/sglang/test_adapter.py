from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest
import zmq
import zmq.asyncio

from subscriber.config import DpEndpoint, SubscriberConfig
from subscriber.engine.base import AbstractEngineAdapter
from subscriber.engine.metadata import MetadataProtocolError
from subscriber.engine.sglang import SGLangAdapter
from subscriber.engine.sglang.incremental import SglangIncrementalSource
from subscriber.health.events import LivenessEvent
from subscriber.kvcm.enum import KvcmStorageType
from subscriber.proto import engine_service_rpc_pb2
from subscriber.types import AllBlocksCleared, BlockRemoved, BlockStored, KVEventBatch


def _publisher_endpoint(
    publisher: zmq.asyncio.Socket,
    replay: zmq.asyncio.Socket,
) -> DpEndpoint:
    return DpEndpoint(
        rank=0,
        zmq_pub_endpoint=publisher.getsockopt_string(zmq.LAST_ENDPOINT),
        zmq_replay_endpoint=replay.getsockopt_string(zmq.LAST_ENDPOINT),
    )


class _SglangWireEvent(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
    tag=True,
):
    pass


class _SglangWireBlockStored(_SglangWireEvent, tag="BlockStored"):
    block_hashes: list[int]
    parent_block_hash: int | None
    token_ids: list[int | tuple[int, int]]
    block_size: int
    lora_id: int | None
    component_type: str
    component_id: int | None = None
    snapshot_version: int = 0
    medium: str | None = None


class _SglangWireBlockRemoved(_SglangWireEvent, tag="BlockRemoved"):
    block_hashes: list[int]
    component_type: str
    component_id: int | None = None
    snapshot_version: int = 0
    medium: str | None = None


class _SglangWireAllBlocksCleared(_SglangWireEvent, tag="AllBlocksCleared"):
    pass


class _SglangWireBatch(msgspec.Struct, array_like=True, gc=False):
    ts: float
    events: list[
        _SglangWireBlockStored | _SglangWireBlockRemoved | _SglangWireAllBlocksCleared
    ]
    attn_dp_rank: int | None = None


async def _publish_until_received(
    publisher: zmq.asyncio.Socket,
    frames: list[bytes],
    received: asyncio.Event,
) -> None:
    while not received.is_set():
        await publisher.send_multipart(frames)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(received.wait(), timeout=0.02)


def test_sglang_adapter_is_registered() -> None:
    assert "sglang" in AbstractEngineAdapter._registry
    assert AbstractEngineAdapter._registry["sglang"] is SGLangAdapter


def test_snapshot_subscription_is_a_required_adapter_contract() -> None:
    assert "subscribe_snapshot_events" in AbstractEngineAdapter.__abstractmethods__


async def test_sglang_snapshot_subscription_requires_bootstrap() -> None:
    adapter = SGLangAdapter(SubscriberConfig())

    with pytest.raises(RuntimeError, match="bootstrap must be fetched"):
        await anext(adapter.subscribe_snapshot_events())


def test_snapshot_signal_is_a_required_adapter_contract() -> None:
    assert "request_immediate_snapshot" in AbstractEngineAdapter.__abstractmethods__


def test_sglang_adapter_can_be_created_via_factory() -> None:
    config = SubscriberConfig()
    adapter = AbstractEngineAdapter.create("sglang", config)
    assert isinstance(adapter, SGLangAdapter)


def test_sglang_adapter_maps_only_supported_sglang_media() -> None:
    config = SubscriberConfig()
    adapter = SGLangAdapter(config)

    assert adapter.map_medium("GPU") == "hbm"
    assert adapter.map_medium("CPU_PINNED") == "mem"
    assert adapter.map_medium("EXTERNAL") == ""
    assert adapter.map_medium(None) == ""
    assert adapter.supported_mediums() == ["hbm", "mem"]
    assert adapter.storage_type() == KvcmStorageType.ST_EVENT_REPORT_L1P5


async def test_sglang_adapter_delegates_snapshot_signal_after_bootstrap(
    mocker: Any,
) -> None:
    _mock_dashllm_clients(mocker, _bootstrap_response())
    adapter = SGLangAdapter(SubscriberConfig(snapshot_kv_event_pipeline_enabled=True))
    await adapter.fetch_kv_event_bootstrap()
    assert adapter._snapshot is not None
    request = mocker.patch.object(adapter._snapshot, "request_immediate_snapshot")

    adapter.request_immediate_snapshot()

    request.assert_called_once_with()


def _bootstrap_response() -> engine_service_rpc_pb2.KvEventBootstrapInfoPB:
    response = engine_service_rpc_pb2.KvEventBootstrapInfoPB(
        protocol_version=1,
        engine_kind="sglang",
        err_code=engine_service_rpc_pb2.KV_EVENT_BOOTSTRAP_OK,
    )
    response.event_transport.live_endpoint = "tcp://127.0.0.1:5557"
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
    full = response.components.add(component_id=0, component_kind="full_attention")
    full.geometry.block_size_tokens = 16
    mamba = response.components.add(component_id=2, component_kind="mamba")
    mamba.geometry.block_size_tokens = 128
    mamba.geometry.checkpoint_alignment_tokens.value = 128
    return response


class _MockDashllmClients:
    def __init__(self, mocker: Any, response: object) -> None:
        self.control = MagicMock()
        self.control.get_kv_event_bootstrap_info = AsyncMock(return_value=response)
        self.control.close = AsyncMock()
        self.status = MagicMock()
        self.status.get_worker_status = AsyncMock(
            return_value=engine_service_rpc_pb2.WorkerStatusPB(alive=True)
        )
        self.status.close = AsyncMock()
        mocker.patch(
            "subscriber.engine.sglang.adapter.DashllmKvEventControlClient",
            return_value=self.control,
        )
        mocker.patch(
            "subscriber.engine.sglang.adapter.DashllmWorkerStatusClient",
            return_value=self.status,
        )


def _mock_dashllm_clients(mocker: Any, response: object) -> _MockDashllmClients:
    return _MockDashllmClients(mocker, response)


async def test_sglang_adapter_fetches_bootstrap_before_opening_zmq(
    mocker: Any,
) -> None:
    response = _bootstrap_response()
    clients = _mock_dashllm_clients(mocker, response)
    adapter = SGLangAdapter(SubscriberConfig(snapshot_kv_event_pipeline_enabled=True))

    assert adapter._incremental is None
    bootstrap = await adapter.fetch_kv_event_bootstrap()

    assert [component.component_id for component in bootstrap.components] == [0, 2]
    assert adapter._incremental is not None
    assert adapter._incremental._component_group_idxs == {0: 0, 2: 2}
    assert adapter._snapshot is not None
    assert adapter._snapshot._component_group_idxs == {0: 0, 2: 2}
    clients.control.get_kv_event_bootstrap_info.assert_awaited_once()


async def test_sglang_adapter_accepts_snapshot_unsupported_when_pipeline_is_disabled(
    mocker: Any,
) -> None:
    response = _bootstrap_response()
    response.snapshot.supported = False
    response.snapshot.versioned = False
    _mock_dashllm_clients(mocker, response)
    adapter = SGLangAdapter(SubscriberConfig())

    await adapter.fetch_kv_event_bootstrap()

    assert adapter._incremental is not None
    assert adapter._snapshot is None


async def test_sglang_adapter_accepts_snapshot_only_bootstrap_without_event_transport(
    mocker: Any,
) -> None:
    response = _bootstrap_response()
    response.event_transport.Clear()
    _mock_dashllm_clients(mocker, response)
    adapter = SGLangAdapter(
        SubscriberConfig(
            incremental_kv_event_pipeline_enabled=False,
            snapshot_kv_event_pipeline_enabled=True,
        )
    )

    await adapter.fetch_kv_event_bootstrap()

    assert adapter._incremental is None
    assert adapter._snapshot is not None


async def test_sglang_adapter_rejects_duplicate_component_id(
    mocker: Any,
) -> None:
    response = _bootstrap_response()
    duplicate = response.components.add(component_id=0, component_kind="full_attention")
    duplicate.geometry.block_size_tokens = 16
    _mock_dashllm_clients(mocker, response)
    adapter = SGLangAdapter(SubscriberConfig())

    with pytest.raises(MetadataProtocolError, match="duplicate component_id"):
        await adapter.fetch_kv_event_bootstrap()


async def test_sglang_adapter_rejects_unknown_component_kind(
    mocker: Any,
) -> None:
    response = _bootstrap_response()
    response.components[0].component_kind = "unknown"
    _mock_dashllm_clients(mocker, response)
    adapter = SGLangAdapter(SubscriberConfig())

    with pytest.raises(MetadataProtocolError, match="unsupported component_kind"):
        await adapter.fetch_kv_event_bootstrap()


@pytest.mark.parametrize(
    "obsolete_kind",
    ["linear_state_checkpoint", "sliding_window_attention"],
)
async def test_sglang_adapter_rejects_obsolete_component_kind(
    mocker: Any,
    obsolete_kind: str,
) -> None:
    response = _bootstrap_response()
    response.components[0].component_kind = obsolete_kind
    _mock_dashllm_clients(mocker, response)
    adapter = SGLangAdapter(SubscriberConfig())

    with pytest.raises(MetadataProtocolError, match="unsupported component_kind"):
        await adapter.fetch_kv_event_bootstrap()


async def test_sglang_adapter_accepts_configured_extra_component_kind(
    mocker: Any,
) -> None:
    response = _bootstrap_response()
    response.components[0].component_kind = "custom_attention"
    _mock_dashllm_clients(mocker, response)
    adapter = SGLangAdapter(
        SubscriberConfig(extra_attention_types={"custom_attention": "custom"})
    )

    bootstrap = await adapter.fetch_kv_event_bootstrap()

    assert bootstrap.components[0].component_kind == "custom_attention"


def _disable_replay(response: engine_service_rpc_pb2.KvEventBootstrapInfoPB) -> None:
    response.event_transport.replay_supported = False
    response.event_transport.replay_endpoint = ""


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda response: setattr(
                response.runtime_topology, "data_parallel_size", 2
            ),
            "data_parallel_size=1",
        ),
        (
            lambda response: setattr(response.snapshot, "supported", False),
            "snapshot transport must be supported",
        ),
        (
            lambda response: setattr(response.snapshot, "versioned", False),
            "snapshot transport must be versioned",
        ),
        (_disable_replay, "event replay must be supported"),
    ],
)
async def test_sglang_adapter_rejects_unsupported_bootstrap_contract(
    mocker: Any, mutate: Any, match: str
) -> None:
    response = _bootstrap_response()
    mutate(response)
    _mock_dashllm_clients(mocker, response)
    adapter = SGLangAdapter(SubscriberConfig(snapshot_kv_event_pipeline_enabled=True))

    with pytest.raises(MetadataProtocolError, match=match):
        await adapter.fetch_kv_event_bootstrap()


async def test_sglang_adapter_reset_generation_state_resets_zmq_source(
    mocker: Any,
) -> None:
    _mock_dashllm_clients(mocker, _bootstrap_response())
    adapter = SGLangAdapter(SubscriberConfig(snapshot_kv_event_pipeline_enabled=True))
    await adapter.fetch_kv_event_bootstrap()
    assert adapter._incremental is not None
    assert adapter._snapshot is not None
    reset = mocker.patch.object(adapter._incremental, "reset_generation_state")
    reset_snapshot = mocker.patch.object(adapter._snapshot, "reset_generation_state")

    await adapter.reset_generation_state()

    reset.assert_awaited_once()
    reset_snapshot.assert_awaited_once()


async def test_sglang_adapter_close_releases_transport_and_grpc_client(
    mocker: Any,
) -> None:
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    adapter = SGLangAdapter(SubscriberConfig(snapshot_kv_event_pipeline_enabled=True))
    await adapter.fetch_kv_event_bootstrap()
    assert adapter._incremental is not None
    assert adapter._snapshot is not None
    close_transport = mocker.patch.object(adapter._incremental, "close")
    close_snapshot = mocker.patch.object(adapter._snapshot, "close")

    await adapter.close()

    close_transport.assert_awaited_once()
    close_snapshot.assert_awaited_once()
    clients.status.close.assert_awaited_once()
    clients.control.close.assert_awaited_once()


async def test_sglang_adapter_delegates_existing_worker_status_liveness(
    mocker: Any,
) -> None:
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    adapter = SGLangAdapter(SubscriberConfig())
    events = adapter.watch_liveness()

    try:
        assert await anext(events) is LivenessEvent.HEALTHY
    finally:
        await events.aclose()

    clients.status.get_worker_status.assert_awaited_once()


async def test_sglang_adapter_reports_unhealthy_worker_status(
    mocker: Any,
) -> None:
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    clients.status.get_worker_status.return_value = (
        engine_service_rpc_pb2.WorkerStatusPB(alive=False)
    )
    adapter = SGLangAdapter(SubscriberConfig(engine_health_interval_s=60.0))
    events = adapter.watch_liveness()

    try:
        assert await anext(events) is LivenessEvent.UNHEALTHY
    finally:
        await events.aclose()

    clients.status.get_worker_status.assert_awaited_once()


@pytest.mark.integration
async def test_sglang_incremental_source_decodes_full_gpu_event() -> None:
    context = zmq.asyncio.Context.instance()
    publisher = context.socket(zmq.PUB)
    replay = context.socket(zmq.ROUTER)
    publisher.bind("tcp://127.0.0.1:0")
    replay.bind("tcp://127.0.0.1:0")
    source = SglangIncrementalSource(
        SubscriberConfig(),
        component_group_idxs={0: 0},
        endpoint=_publisher_endpoint(publisher, replay),
    )
    wire_batch = _SglangWireBatch(
        ts=1.0,
        events=[
            _SglangWireBlockStored(
                block_hashes=[101],
                parent_block_hash=None,
                token_ids=[11, 12],
                block_size=2,
                lora_id=None,
                component_type="full",
                component_id=0,
                medium="GPU",
            ),
            _SglangWireBlockRemoved(
                block_hashes=[102],
                component_type="full",
                component_id=0,
                medium="GPU",
            ),
            _SglangWireAllBlocksCleared(),
        ],
        attn_dp_rank=0,
    )
    delivered = asyncio.Event()
    results: list[KVEventBatch] = []

    async def _consume() -> None:
        async for event_batch in source.subscribe():
            results.extend(event_batch.batches)
            delivered.set()
            break

    consumer = asyncio.create_task(_consume())
    try:
        await asyncio.wait_for(
            _publish_until_received(
                publisher,
                [b"", (0).to_bytes(8, "big"), msgspec.msgpack.encode(wire_batch)],
                delivered,
            ),
            timeout=5.0,
        )
        await asyncio.wait_for(consumer, timeout=5.0)
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
        publisher.close(linger=0)
        replay.close(linger=0)

    assert results == [
        KVEventBatch(
            ts=1.0,
            events=[
                BlockStored(
                    block_hashes=[101],
                    parent_block_hash=None,
                    token_ids=[11, 12],
                    block_size=2,
                    lora_id=None,
                    medium="GPU",
                    lora_name=None,
                    group_idx=0,
                    component_id=0,
                    kv_cache_spec_kind="full",
                ),
                BlockRemoved(
                    block_hashes=[102],
                    medium="GPU",
                    group_idx=0,
                    component_id=0,
                    remaining_copy_counts=None,
                ),
                AllBlocksCleared(),
            ],
            data_parallel_rank=0,
        )
    ]


@pytest.mark.integration
async def test_sglang_incremental_source_logs_rich_debug_batch_summary(
    mocker: pytest.MockFixture,
) -> None:
    context = zmq.asyncio.Context.instance()
    publisher = context.socket(zmq.PUB)
    replay = context.socket(zmq.ROUTER)
    publisher.bind("tcp://127.0.0.1:0")
    replay.bind("tcp://127.0.0.1:0")
    source = SglangIncrementalSource(
        SubscriberConfig(),
        component_group_idxs={0: 0},
        endpoint=_publisher_endpoint(publisher, replay),
    )
    mocker.patch(
        "subscriber.engine.sglang.incremental.logger.is_debug_enabled",
        return_value=True,
    )
    mocker.patch(
        "subscriber.engine.zmq_source.generate_trace_id",
        return_value="sglang-live-trace",
    )
    debug = mocker.patch("subscriber.engine.sglang.incremental.logger.debug")
    wire_batch = _SglangWireBatch(
        ts=1.0,
        events=[
            _SglangWireBlockStored(
                block_hashes=[101, 102],
                parent_block_hash=100,
                token_ids=[11, 12],
                block_size=2,
                lora_id=None,
                component_type="full",
                component_id=0,
                medium="GPU",
            ),
            _SglangWireBlockRemoved(
                block_hashes=[103],
                component_type="full",
                component_id=0,
                medium="GPU",
            ),
            _SglangWireAllBlocksCleared(),
        ],
        attn_dp_rank=0,
    )
    delivered = asyncio.Event()

    async def _consume() -> None:
        async for _ in source.subscribe():
            delivered.set()
            break

    consumer = asyncio.create_task(_consume())
    try:
        await asyncio.wait_for(
            _publish_until_received(
                publisher,
                [b"", (0).to_bytes(8, "big"), msgspec.msgpack.encode(wire_batch)],
                delivered,
            ),
            timeout=5.0,
        )
        await asyncio.wait_for(consumer, timeout=5.0)
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
        publisher.close(linger=0)
        replay.close(linger=0)

    debug.assert_any_call(
        "decoded SGLang KV event batch",
        step="zmq_subscribe",
        tags={
            "seq": 0,
            "trace_id": "sglang-live-trace",
            "event_count": 3,
            "event_types": "AllBlocksCleared:1,BlockRemoved:1,BlockStored:1",
            "data_parallel_rank": 0,
            "stored_block_count": 1,
            "stored_blocks": [
                {
                    "block_hashes": ["101", "102"],
                    "parent_block_hash": "100",
                    "block_size": 2,
                    "lora_id": None,
                    "medium": "GPU",
                    "lora_name": None,
                    "extra_keys": None,
                    "group_idx": 0,
                    "component_id": 0,
                    "kv_cache_spec_kind": "full",
                    "kv_cache_spec_sliding_window": None,
                    "snapshot_version": 0,
                }
            ],
            "stored_blocks_truncated": False,
            "removed_block_count": 1,
            "removed_blocks": [
                {
                    "block_hashes": ["103"],
                    "medium": "GPU",
                    "group_idx": 0,
                    "component_id": 0,
                    "remaining_copy_counts": None,
                    "snapshot_version": 0,
                }
            ],
            "removed_blocks_truncated": False,
        },
    )


@pytest.mark.integration
async def test_sglang_incremental_source_keeps_valid_event_without_drop_metric(
    mocker: pytest.MockFixture,
) -> None:
    counter = mocker.patch("subscriber.metrics.lifecycle._dashlog_counter")
    context = zmq.asyncio.Context.instance()
    publisher = context.socket(zmq.PUB)
    replay = context.socket(zmq.ROUTER)
    publisher.bind("tcp://127.0.0.1:0")
    replay.bind("tcp://127.0.0.1:0")
    source = SglangIncrementalSource(
        SubscriberConfig(),
        component_group_idxs={0: 0},
        endpoint=_publisher_endpoint(publisher, replay),
    )
    wire_batch = _SglangWireBatch(
        ts=2.0,
        events=[
            _SglangWireBlockStored(
                block_hashes=[201],
                parent_block_hash=None,
                token_ids=[21],
                block_size=1,
                lora_id=None,
                component_type="full",
                component_id=0,
                medium="GPU",
            ),
            _SglangWireBlockStored(
                block_hashes=[202],
                parent_block_hash=None,
                token_ids=[22],
                block_size=1,
                lora_id=None,
                component_type="full",
                component_id=0,
                medium="EXTERNAL",
            ),
        ],
    )
    delivered = asyncio.Event()
    results: list[KVEventBatch] = []

    async def _consume() -> None:
        async for event_batch in source.subscribe():
            results.extend(event_batch.batches)
            delivered.set()
            break

    consumer = asyncio.create_task(_consume())
    try:
        await asyncio.wait_for(
            _publish_until_received(
                publisher,
                [b"", (0).to_bytes(8, "big"), msgspec.msgpack.encode(wire_batch)],
                delivered,
            ),
            timeout=5.0,
        )
        await asyncio.wait_for(consumer, timeout=5.0)
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
        publisher.close(linger=0)
        replay.close(linger=0)

    assert results[0].events[0].block_hashes == [201]
    assert len(results[0].events) == 1
    assert all(
        call.args[0] != "sglang_event_drop_count" for call in counter.call_args_list
    )


@pytest.mark.integration
async def test_sglang_incremental_source_replays_missing_batch_before_live() -> None:
    context = zmq.asyncio.Context.instance()
    publisher = context.socket(zmq.PUB)
    replay = context.socket(zmq.ROUTER)
    publisher.bind("tcp://127.0.0.1:0")
    replay.bind("tcp://127.0.0.1:0")
    source = SglangIncrementalSource(
        SubscriberConfig(),
        component_group_idxs={0: 0},
        endpoint=_publisher_endpoint(publisher, replay),
    )

    def _wire_batch(block_hash: int) -> bytes:
        return msgspec.msgpack.encode(
            _SglangWireBatch(
                ts=float(block_hash),
                events=[
                    _SglangWireBlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=None,
                        token_ids=[block_hash],
                        block_size=1,
                        lora_id=None,
                        component_type="full",
                        component_id=0,
                        medium="GPU",
                    )
                ],
            )
        )

    first_live = asyncio.Event()
    delivered = asyncio.Event()
    results: list[KVEventBatch] = []

    async def _consume() -> None:
        async for event_batch in source.subscribe():
            results.extend(event_batch.batches)
            if len(results) == 1:
                first_live.set()
            if len(results) == 3:
                delivered.set()
                break

    async def _serve_replay() -> None:
        client_id, delimiter, start_seq = await replay.recv_multipart()
        assert delimiter == b""
        assert int.from_bytes(start_seq, "big") == 1
        await replay.send_multipart(
            [
                client_id,
                b"",
                b"",
                (1).to_bytes(8, "big"),
                _wire_batch(301),
            ]
        )
        await replay.send_multipart(
            [
                client_id,
                b"",
                b"",
                (-1).to_bytes(8, "big", signed=True),
                b"",
            ]
        )

    consumer = asyncio.create_task(_consume())
    replay_server = asyncio.create_task(_serve_replay())
    try:
        await asyncio.wait_for(
            _publish_until_received(
                publisher,
                [b"", (0).to_bytes(8, "big"), _wire_batch(300)],
                first_live,
            ),
            timeout=5.0,
        )
        await asyncio.wait_for(
            _publish_until_received(
                publisher,
                [b"", (2).to_bytes(8, "big"), _wire_batch(302)],
                delivered,
            ),
            timeout=5.0,
        )
        await asyncio.wait_for(replay_server, timeout=5.0)
        await asyncio.wait_for(consumer, timeout=5.0)
    finally:
        for task in (consumer, replay_server):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        publisher.close(linger=0)
        replay.close(linger=0)

    assert [
        (
            batch.events[0].block_hashes,
            batch.events[0].group_idx,
            batch.events[0].medium,
        )
        for batch in results
    ] == [
        ([300], 0, "GPU"),
        ([301], 0, "GPU"),
        ([302], 0, "GPU"),
    ]
