from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import grpc
import msgspec
import pytest
import zmq

from subscriber.config import DpEndpoint, SubscriberConfig
from subscriber.engine.base import EngineEventBatch
from subscriber.engine.metadata import MetadataProtocolError
from subscriber.engine.vllm import VllmAdapter
from subscriber.engine.vllm import control as vllm_control_module
from subscriber.engine.vllm import incremental as vllm_transport_module
from subscriber.engine.vllm.incremental import VllmIncrementalSource
from subscriber.health.events import LivenessEvent
from subscriber.kvcm.enum import KvcmStorageType
from subscriber.proto import engine_service_rpc_pb2
from subscriber.types import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)


@pytest.fixture
def config() -> SubscriberConfig:
    return SubscriberConfig(snapshot_kv_event_pipeline_enabled=True)


def _dp_endpoint() -> DpEndpoint:
    return DpEndpoint(
        rank=0,
        zmq_pub_endpoint="tcp://localhost:5557",
        zmq_replay_endpoint="tcp://localhost:5558",
    )


def _encode_batch(batch: KVEventBatch) -> bytes:
    return msgspec.msgpack.encode(batch)


def _seq_bytes(seq: int) -> bytes:
    return seq.to_bytes(8, "big")


def _mock_adapter_sockets(mocker: Any) -> tuple[MagicMock, MagicMock]:
    mock_sub = MagicMock()
    mock_dealer = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.socket.side_effect = [mock_sub, mock_dealer]
    mocker.patch(
        "subscriber.engine.zmq_source.zmq.asyncio.Context.instance",
        return_value=mock_ctx,
    )
    return mock_sub, mock_dealer


def _mock_adapter_socket_sequence(mocker: Any, *sockets: MagicMock) -> MagicMock:
    mock_ctx = MagicMock()
    mock_ctx.socket.side_effect = list(sockets)
    mocker.patch(
        "subscriber.engine.zmq_source.zmq.asyncio.Context.instance",
        return_value=mock_ctx,
    )
    return mock_ctx


def _bootstrap_response() -> engine_service_rpc_pb2.KvEventBootstrapInfoPB:
    response = engine_service_rpc_pb2.KvEventBootstrapInfoPB(
        protocol_version=1,
        engine_kind="vllm",
        err_code=engine_service_rpc_pb2.KV_EVENT_BOOTSTRAP_OK,
    )
    response.event_transport.live_endpoint = "tcp://127.0.0.1:6557"
    response.event_transport.topic = "kv-events"
    response.event_transport.replay_supported = True
    response.event_transport.replay_endpoint = "tcp://127.0.0.1:6558"
    response.event_transport.serialization = "msgpack-v1"
    response.runtime_topology.data_parallel_size = 1
    response.runtime_topology.tensor_parallel_size = 1
    response.runtime_topology.pipeline_parallel_size = 1
    response.snapshot.supported = True
    response.snapshot.versioned = True
    response.vllm.event_schema_version = 2
    response.vllm.mamba_cache_mode = "none"
    response.vllm.hash_algorithm = "sha256"
    response.vllm.hash_version = "vllm-block-hash-v1"
    component = response.components.add(component_id=0, component_kind="full_attention")
    component.geometry.block_size_tokens = 16
    component.geometry.group_payload_size_bytes.value = 1024
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
            "subscriber.engine.vllm.adapter.DashllmKvEventControlClient",
            return_value=self.control,
        )
        mocker.patch(
            "subscriber.engine.vllm.adapter.DashllmWorkerStatusClient",
            return_value=self.status,
        )


def _mock_dashllm_clients(mocker: Any, response: object) -> _MockDashllmClients:
    return _MockDashllmClients(mocker, response)


def _make_rpc_error(code: grpc.StatusCode, details: str = "") -> grpc.aio.AioRpcError:
    """Create a real AioRpcError for testing."""
    return grpc.aio.AioRpcError(code, {}, {}, details, "")


def test_adapter_composes_focused_transport_and_control_helpers(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)

    transport_type = getattr(vllm_transport_module, "VllmIncrementalSource", None)
    control_type = getattr(vllm_control_module, "VllmControl", None)

    assert transport_type is not None
    assert control_type is not None

    adapter = VllmAdapter(config)

    assert adapter._incremental is None
    assert adapter._snapshot is None
    assert isinstance(adapter._control, control_type)


def test_adapter_storage_type_reflects_config(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)

    adapter = VllmAdapter(config)
    assert adapter.storage_type() == KvcmStorageType.ST_EVENT_REPORT_L1P5

    config.kvcm_storage_type = "ST_CUSTOM"
    assert adapter.storage_type() == "ST_CUSTOM"


async def test_adapter_fetches_bootstrap_before_opening_engine_owned_zmq(
    config: SubscriberConfig, mocker: Any
) -> None:
    sub, dealer = _mock_adapter_sockets(mocker)
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    adapter = VllmAdapter(config)

    assert adapter._incremental is None
    bootstrap = await adapter.fetch_kv_event_bootstrap()

    assert bootstrap.components[0].component_id == 0
    assert adapter._incremental is not None
    assert adapter._snapshot is not None
    sub.connect.assert_called_once_with("tcp://127.0.0.1:6557")
    sub.setsockopt_string.assert_called_once_with(zmq.SUBSCRIBE, "kv-events")
    dealer.connect.assert_called_once_with("tcp://127.0.0.1:6558")
    clients.control.get_kv_event_bootstrap_info.assert_awaited_once_with(
        config.engine_kv_event_bootstrap_timeout_ms / 1000
    )


async def test_adapter_accepts_snapshot_unsupported_when_pipeline_is_disabled(
    mocker: Any,
) -> None:
    sub, dealer = _mock_adapter_sockets(mocker)
    response = _bootstrap_response()
    response.snapshot.supported = False
    response.snapshot.versioned = False
    _mock_dashllm_clients(mocker, response)
    adapter = VllmAdapter(SubscriberConfig())

    await adapter.fetch_kv_event_bootstrap()

    assert adapter._incremental is not None
    assert adapter._snapshot is None
    sub.connect.assert_called_once()
    dealer.connect.assert_called_once()


async def test_adapter_accepts_snapshot_only_bootstrap_without_event_transport(
    mocker: Any,
) -> None:
    response = _bootstrap_response()
    response.event_transport.Clear()
    _mock_dashllm_clients(mocker, response)
    adapter = VllmAdapter(
        SubscriberConfig(
            incremental_kv_event_pipeline_enabled=False,
            snapshot_kv_event_pipeline_enabled=True,
        )
    )

    await adapter.fetch_kv_event_bootstrap()

    assert adapter._incremental is None
    assert adapter._snapshot is not None


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
    ],
)
async def test_adapter_rejects_unsupported_bootstrap_contract(
    config: SubscriberConfig,
    mocker: Any,
    mutate: Any,
    match: str,
) -> None:
    response = _bootstrap_response()
    mutate(response)
    _mock_dashllm_clients(mocker, response)
    adapter = VllmAdapter(config)

    with pytest.raises(MetadataProtocolError, match=match):
        await adapter.fetch_kv_event_bootstrap()


async def test_adapter_close_before_bootstrap_only_closes_grpc(
    config: SubscriberConfig, mocker: Any
) -> None:
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    adapter = VllmAdapter(config)

    await adapter.close()

    clients.status.close.assert_awaited_once()
    clients.control.close.assert_awaited_once()


# --- gRPC-based watch_liveness tests ---


async def test_watch_liveness_grpc_alive_true_is_healthy(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    clients.status.get_worker_status = AsyncMock(
        return_value=engine_service_rpc_pb2.WorkerStatusPB(alive=True)
    )
    mocker.patch("subscriber.engine.vllm.control.asyncio.sleep", AsyncMock())

    adapter = VllmAdapter(config)
    events: list[LivenessEvent] = []
    async for event in adapter.watch_liveness():
        events.append(event)
        if len(events) == 2:
            break

    assert events == [LivenessEvent.HEALTHY, LivenessEvent.HEALTHY]
    assert clients.status.get_worker_status.await_count == 2
    clients.status.get_worker_status.assert_awaited_with(
        config.engine_kvcache_worker_status_timeout_ms / 1000
    )


async def test_watch_liveness_grpc_alive_false_is_unhealthy(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    clients.status.get_worker_status = AsyncMock(
        return_value=engine_service_rpc_pb2.WorkerStatusPB(alive=False)
    )
    mocker.patch("subscriber.engine.vllm.control.asyncio.sleep", AsyncMock())
    mocker.patch("subscriber.engine.vllm.control.logger.warning")

    adapter = VllmAdapter(config)
    events: list[LivenessEvent] = []
    async for event in adapter.watch_liveness():
        events.append(event)
        if len(events) == 1:
            break

    assert events == [LivenessEvent.UNHEALTHY]


async def test_watch_liveness_grpc_deadline_exceeded_is_unhealthy(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    clients.status.get_worker_status = AsyncMock(
        side_effect=_make_rpc_error(
            grpc.StatusCode.DEADLINE_EXCEEDED, "deadline exceeded"
        )
    )
    mocker.patch("subscriber.engine.vllm.control.asyncio.sleep", AsyncMock())
    mocker.patch("subscriber.engine.vllm.control.logger.warning")

    adapter = VllmAdapter(config)
    events: list[LivenessEvent] = []
    async for event in adapter.watch_liveness():
        events.append(event)
        if len(events) == 1:
            break

    assert events == [LivenessEvent.UNHEALTHY]


async def test_watch_liveness_grpc_unavailable_is_unhealthy(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    clients.status.get_worker_status = AsyncMock(
        side_effect=_make_rpc_error(grpc.StatusCode.UNAVAILABLE, "connection refused")
    )
    mocker.patch("subscriber.engine.vllm.control.asyncio.sleep", AsyncMock())
    mocker.patch("subscriber.engine.vllm.control.logger.warning")

    adapter = VllmAdapter(config)
    events: list[LivenessEvent] = []
    async for event in adapter.watch_liveness():
        events.append(event)
        if len(events) == 1:
            break

    assert events == [LivenessEvent.UNHEALTHY]


async def test_watch_liveness_grpc_other_rpc_error_is_unhealthy(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    clients.status.get_worker_status = AsyncMock(
        side_effect=_make_rpc_error(grpc.StatusCode.INTERNAL, "internal error")
    )
    mocker.patch("subscriber.engine.vllm.control.asyncio.sleep", AsyncMock())
    mocker.patch("subscriber.engine.vllm.control.logger.warning")

    adapter = VllmAdapter(config)
    events: list[LivenessEvent] = []
    async for event in adapter.watch_liveness():
        events.append(event)
        if len(events) == 1:
            break

    assert events == [LivenessEvent.UNHEALTHY]


async def test_watch_liveness_grpc_reraises_cancellation(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    clients.status.get_worker_status = AsyncMock(side_effect=asyncio.CancelledError())
    mocker.patch("subscriber.engine.vllm.control.asyncio.sleep", AsyncMock())

    adapter = VllmAdapter(config)
    with pytest.raises(asyncio.CancelledError):
        async for _ in adapter.watch_liveness():
            pass


async def test_watch_liveness_logs_first_failure_not_every_poll(
    config: SubscriberConfig, mocker: Any
) -> None:
    """Only the first failure (transition) is logged, not every poll."""
    _mock_adapter_sockets(mocker)
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    clients.status.get_worker_status = AsyncMock(
        return_value=engine_service_rpc_pb2.WorkerStatusPB(alive=False)
    )
    mocker.patch("subscriber.engine.vllm.control.asyncio.sleep", AsyncMock())
    warning = mocker.patch("subscriber.engine.vllm.control.logger.warning")

    adapter = VllmAdapter(config)
    events: list[LivenessEvent] = []
    async for event in adapter.watch_liveness():
        events.append(event)
        if len(events) == 4:
            break

    assert events == [LivenessEvent.UNHEALTHY] * 4
    # Only the first failure should produce a warning log
    assert warning.call_count == 1


async def test_watch_liveness_logs_recovery_transition(
    config: SubscriberConfig, mocker: Any
) -> None:
    """Recovery from unhealthy to healthy is logged."""
    _mock_adapter_sockets(mocker)
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    clients.status.get_worker_status = AsyncMock(
        side_effect=[
            engine_service_rpc_pb2.WorkerStatusPB(alive=False),
            engine_service_rpc_pb2.WorkerStatusPB(alive=True),
        ]
    )
    mocker.patch("subscriber.engine.vllm.control.asyncio.sleep", AsyncMock())
    warning = mocker.patch("subscriber.engine.vllm.control.logger.warning")
    info = mocker.patch("subscriber.engine.vllm.control.logger.info")

    adapter = VllmAdapter(config)
    events: list[LivenessEvent] = []
    async for event in adapter.watch_liveness():
        events.append(event)
        if len(events) == 2:
            break

    assert events == [LivenessEvent.UNHEALTHY, LivenessEvent.HEALTHY]
    # First failure logged as warning
    assert warning.call_count == 1
    # Recovery logged as info
    info.assert_any_call(
        "engine health recovered",
        step="engine_health",
        tags={
            "consecutive_failures_before_recovery": 1,
            "target": config.engine_grpc_endpoint,
        },
    )


async def test_watch_liveness_unexpected_error_is_unhealthy(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    clients.status.get_worker_status = AsyncMock(
        side_effect=[
            RuntimeError("boom"),
            engine_service_rpc_pb2.WorkerStatusPB(alive=True),
        ]
    )
    mocker.patch("subscriber.engine.vllm.control.asyncio.sleep", AsyncMock())
    mocker.patch("subscriber.engine.vllm.control.logger.warning")

    adapter = VllmAdapter(config)
    events: list[LivenessEvent] = []
    async for event in adapter.watch_liveness():
        events.append(event)
        if len(events) == 2:
            break

    assert events == [LivenessEvent.UNHEALTHY, LivenessEvent.HEALTHY]


async def test_watch_liveness_polls_at_configured_interval(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    clients.status.get_worker_status = AsyncMock(
        return_value=engine_service_rpc_pb2.WorkerStatusPB(alive=True)
    )
    sleep_mock = mocker.patch(
        "subscriber.engine.vllm.control.asyncio.sleep", AsyncMock()
    )

    adapter = VllmAdapter(config)
    events: list[LivenessEvent] = []
    async for event in adapter.watch_liveness():
        events.append(event)
        if len(events) == 3:
            break

    # Events are yielded before the interval sleep, so the third event is
    # observed after only two sleeps.
    assert sleep_mock.await_count == 2
    sleep_mock.assert_awaited_with(config.engine_health_interval_s)


async def test_watch_liveness_yields_probe_result_before_interval_sleep(
    config: SubscriberConfig, mocker: Any
) -> None:
    """The probe result must reach the health coordinator immediately;
    sleeping first would delay DEAD detection and gate reopening by one
    full engine_health_interval_s."""

    _mock_adapter_sockets(mocker)
    clients = _mock_dashllm_clients(mocker, _bootstrap_response())
    clients.status.get_worker_status = AsyncMock(
        return_value=engine_service_rpc_pb2.WorkerStatusPB(alive=True)
    )
    sleep_mock = mocker.patch(
        "subscriber.engine.vllm.control.asyncio.sleep", AsyncMock()
    )

    adapter = VllmAdapter(config)
    async for event in adapter.watch_liveness():
        assert event is LivenessEvent.HEALTHY
        break

    assert sleep_mock.await_count == 0


# --- ZMQ transport tests ---


def test_adapter_opens_sub_and_dealer_without_monitor(
    config: SubscriberConfig, mocker: Any
) -> None:
    mock_sub, mock_dealer = _mock_adapter_sockets(mocker)

    VllmIncrementalSource(config, endpoint=_dp_endpoint())

    mock_sub.connect.assert_called_once_with(_dp_endpoint().zmq_pub_endpoint)
    mock_sub.setsockopt_string.assert_called_once()
    mock_dealer.connect.assert_called_once_with(_dp_endpoint().zmq_replay_endpoint)
    mock_sub.get_monitor_socket.assert_not_called()


def test_component_validation_uses_group_idx_and_filters_only_invalid_events(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    source = VllmIncrementalSource(
        config,
        endpoint=_dp_endpoint(),
        valid_component_ids={1},
    )
    valid = BlockStored(
        block_hashes=[11],
        parent_block_hash=None,
        token_ids=[1, 2],
        block_size=2,
        lora_id=None,
        medium="GPU",
        lora_name=None,
        group_idx=1,
    )
    unknown = BlockRemoved(block_hashes=[12], medium="GPU", group_idx=7)
    missing = BlockRemoved(block_hashes=[13], medium="GPU")

    decoded = source._decode_payload(
        _encode_batch(KVEventBatch(ts=1.0, events=[valid, unknown, missing])),
        step="decode",
        tags={"source": "test"},
    )

    assert decoded is not None
    assert decoded.events == [valid]


async def test_live_receive_records_zmq_queue_backlog_signal(
    config: SubscriberConfig, mocker: Any
) -> None:
    mock_sub, _ = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(return_value=[b"topic", _seq_bytes(7), b"body"])
    mock_sub.getsockopt.return_value = zmq.POLLIN
    mocker.patch(
        "subscriber.engine.vllm.incremental.generate_trace_id",
        return_value="live-trace",
    )
    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    diagnostics = mocker.Mock()
    adapter._zmq_gap_diagnostics = diagnostics

    assert await adapter._recv_live_message() == (7, b"body", "live-trace")

    diagnostics.record_message.assert_called_once_with(
        message_bytes=len(b"topic") + len(_seq_bytes(7)) + len(b"body"),
        queue_nonempty_after_receive=True,
    )


def test_adapter_reports_current_zmq_queue_state(
    config: SubscriberConfig, mocker: Any
) -> None:
    mock_sub, _ = _mock_adapter_sockets(mocker)
    mock_sub.getsockopt.side_effect = lambda option: {
        zmq.EVENTS: zmq.POLLIN,
        zmq.RCVHWM: 1000,
    }[option]
    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())

    assert adapter._zmq_queue_state() == {
        "zmq_sub_readable": True,
        "zmq_sub_rcvhwm": 1000,
        "zmq_exact_queue_depth_available": False,
    }


def test_adapter_logs_zmq_endpoints_at_info(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    info = mocker.patch("subscriber.engine.vllm.incremental.logger.info")

    VllmIncrementalSource(config, endpoint=_dp_endpoint())

    info.assert_any_call(
        "connecting vLLM ZMQ sockets",
        step="zmq_connect",
        tags={
            "pub_endpoint": _dp_endpoint().zmq_pub_endpoint,
            "replay_endpoint": _dp_endpoint().zmq_replay_endpoint,
            "topic": _dp_endpoint().zmq_topic,
            "reconnect_ivl_ms": config.zmq_reconnect_ivl_ms,
            "reconnect_ivl_max_ms": config.zmq_reconnect_ivl_max_ms,
        },
    )


async def test_reset_generation_state_resets_seq_and_recreates_sockets(
    config: SubscriberConfig, mocker: Any
) -> None:
    old_sub = MagicMock()
    old_dealer = MagicMock()
    new_sub = MagicMock()
    new_dealer = MagicMock()
    _mock_adapter_socket_sequence(mocker, old_sub, old_dealer, new_sub, new_dealer)
    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    adapter._last_seq = 99

    await adapter.reset_generation_state()

    assert adapter._last_seq == -1
    old_sub.close.assert_called_once_with(linger=0)
    old_dealer.close.assert_called_once_with(linger=0)
    new_sub.connect.assert_called_once_with(_dp_endpoint().zmq_pub_endpoint)
    new_sub.setsockopt_string.assert_called_once()
    new_dealer.connect.assert_called_once_with(_dp_endpoint().zmq_replay_endpoint)


async def test_reset_generation_drops_inflight_live_message(
    config: SubscriberConfig, mocker: Any
) -> None:
    batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    payload = _encode_batch(batch)
    old_sub = MagicMock()
    old_dealer = MagicMock()
    new_sub = MagicMock()
    new_dealer = MagicMock()
    _mock_adapter_socket_sequence(mocker, old_sub, old_dealer, new_sub, new_dealer)
    recv_started = asyncio.Event()
    release_recv = asyncio.Event()

    async def recv_multipart() -> list[bytes]:
        recv_started.set()
        await release_recv.wait()
        return [b"", _seq_bytes(0), payload]

    old_sub.recv_multipart = AsyncMock(side_effect=recv_multipart)

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    receive_task = asyncio.create_task(adapter._recv_live_message())
    await recv_started.wait()

    await adapter.reset_generation_state()
    release_recv.set()

    assert await asyncio.wait_for(receive_task, timeout=1.0) is None
    old_sub.close.assert_called_once_with(linger=0)
    old_dealer.close.assert_called_once_with(linger=0)


async def test_reset_generation_drops_inflight_replay_batches(
    config: SubscriberConfig, mocker: Any
) -> None:
    """Race scenario: reset lands between replay send and replay recv.

    Without the generation guard, ``_replay_missing_batches`` would return
    the stale replay batch decoded from the old dealer socket and the
    subscribe loop would re-anchor ``_last_seq`` to a sequence belonging to
    the previous engine instance.
    """
    stale_batch = KVEventBatch(ts=0.5, events=[AllBlocksCleared()])
    stale_payload = _encode_batch(stale_batch)
    old_sub = MagicMock()
    old_dealer = MagicMock()
    new_sub = MagicMock()
    new_dealer = MagicMock()
    _mock_adapter_socket_sequence(mocker, old_sub, old_dealer, new_sub, new_dealer)
    send_started = asyncio.Event()
    release_send = asyncio.Event()
    recv_started = asyncio.Event()
    release_recv = asyncio.Event()

    async def fake_send(_frames: list[bytes]) -> None:
        send_started.set()
        await release_send.wait()

    async def fake_recv() -> list[bytes]:
        recv_started.set()
        await release_recv.wait()
        return [b"", _seq_bytes(1), stale_payload]

    old_dealer.send_multipart = AsyncMock(side_effect=fake_send)
    old_dealer.recv_multipart = AsyncMock(side_effect=fake_recv)

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    adapter._last_seq = 0
    replay_task = asyncio.create_task(
        adapter._replay_missing_batches(5, 0, "replay-trace")
    )
    await send_started.wait()
    release_send.set()
    await recv_started.wait()

    await adapter.reset_generation_state()
    release_recv.set()

    assert await asyncio.wait_for(replay_task, timeout=1.0) is None
    old_sub.close.assert_called_once_with(linger=0)
    old_dealer.close.assert_called_once_with(linger=0)
    assert adapter._last_seq == -1, (
        "reset must keep _last_seq at -1; replay must not have anchored it "
        "to the stale batch's sequence"
    )


async def test_replay_missing_batches_uses_new_dealer_after_reset(
    config: SubscriberConfig, mocker: Any
) -> None:
    """After reset, a fresh replay call must use the new dealer socket and
    succeed — proving generation advancement does not permanently disable
    replay.
    """
    stale_batch = KVEventBatch(ts=0.5, events=[AllBlocksCleared()])
    stale_payload = _encode_batch(stale_batch)
    fresh_batch = KVEventBatch(ts=2.0, events=[AllBlocksCleared()])
    fresh_payload = _encode_batch(fresh_batch)
    old_sub = MagicMock()
    old_dealer = MagicMock()
    new_sub = MagicMock()
    new_dealer = MagicMock()
    _mock_adapter_socket_sequence(mocker, old_sub, old_dealer, new_sub, new_dealer)
    recv_started = asyncio.Event()
    release_recv = asyncio.Event()

    async def stale_recv() -> list[bytes]:
        recv_started.set()
        await release_recv.wait()
        return [b"", _seq_bytes(1), stale_payload]

    old_dealer.send_multipart = AsyncMock()
    old_dealer.recv_multipart = AsyncMock(side_effect=stale_recv)
    new_dealer.send_multipart = AsyncMock()
    new_dealer.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(1), fresh_payload],
            [b"", (-1).to_bytes(8, "big", signed=True), b""],
        ]
    )

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    adapter._last_seq = 0
    stale_task = asyncio.create_task(
        adapter._replay_missing_batches(5, 0, "stale-trace")
    )
    await recv_started.wait()

    await adapter.reset_generation_state()
    release_recv.set()
    assert await asyncio.wait_for(stale_task, timeout=1.0) is None

    result = await adapter._replay_missing_batches(
        5, adapter._generation, "fresh-trace"
    )
    assert result == [fresh_batch]
    new_dealer.send_multipart.assert_awaited_once_with([b"", (0).to_bytes(8, "big")])


async def _collect_n(adapter: VllmAdapter, n: int) -> list[list[KVEventBatch]]:
    events = await _collect_events_n(adapter, n)
    return [event.batches for event in events]


async def _collect_events_n(adapter: VllmAdapter, n: int) -> list[EngineEventBatch]:
    results: list[EngineEventBatch] = []

    async def _run() -> None:
        async for event in adapter.subscribe():
            results.append(event)
            if len(results) >= n:
                break

    await asyncio.wait_for(_run(), timeout=1.0)
    return results


async def test_live_event_carries_decode_span(
    config: SubscriberConfig, mocker: Any
) -> None:
    batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    payload = _encode_batch(batch)
    mock_sub, _ = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(return_value=[b"", _seq_bytes(0), payload])

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    events = await _collect_events_n(adapter, 1)

    spans = events[0].telemetry.spans
    assert [span.name for span in spans] == ["decode"]
    assert spans[0].duration_s >= 0


async def test_replayed_event_carries_replay_fetch_span(
    config: SubscriberConfig, mocker: Any
) -> None:
    batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    payload = _encode_batch(batch)
    replay_batch = KVEventBatch(ts=0.5, events=[AllBlocksCleared()])
    replay_payload = _encode_batch(replay_batch)

    mock_sub, mock_dealer = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(1), payload],
            [b"", _seq_bytes(2), payload],
        ]
    )
    mock_dealer.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), replay_payload],
            [b"", (-1).to_bytes(8, "big", signed=True), b""],
        ]
    )
    mock_dealer.send_multipart = AsyncMock()

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    events = await _collect_events_n(adapter, 1)

    spans = events[0].telemetry.spans
    assert [span.name for span in spans] == ["replay_fetch"]
    assert spans[0].duration_s >= 0


async def test_yields_single_batch_no_gap(
    config: SubscriberConfig, mocker: Any
) -> None:
    batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    payload = _encode_batch(batch)

    mock_sub, mock_dealer = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), payload],
            [b"", _seq_bytes(1), payload],
        ]
    )

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    results = await _collect_n(adapter, 2)

    assert len(results) == 2
    assert len(results[0]) == 1
    assert len(results[1]) == 1
    mock_dealer.send_multipart.assert_not_called()


async def test_recv_live_message_logs_debug_metadata(
    config: SubscriberConfig, mocker: Any
) -> None:
    batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    payload = _encode_batch(batch)
    mock_sub, _ = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(return_value=[b"topic", _seq_bytes(7), payload])
    mocker.patch(
        "subscriber.engine.vllm.incremental.logger.is_debug_enabled", return_value=True
    )
    mocker.patch(
        "subscriber.engine.vllm.incremental.generate_trace_id",
        return_value="live-trace",
    )
    debug = mocker.patch("subscriber.engine.vllm.incremental.logger.debug")

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    assert await adapter._recv_live_message() == (7, payload, "live-trace")

    debug.assert_any_call(
        "received vLLM ZMQ live message",
        step="zmq_subscribe",
        tags={
            "topic": "topic",
            "seq": 7,
            "payload_bytes": len(payload),
            "trace_id": "live-trace",
        },
    )


async def test_subscribe_logs_decoded_event_types_and_block_hashes(
    config: SubscriberConfig, mocker: Any
) -> None:
    batch = KVEventBatch(
        ts=1.0,
        events=[
            BlockStored(
                block_hashes=[11, (1 << 63) + 1],
                parent_block_hash=None,
                token_ids=[1, 2, 3],
                block_size=16,
                lora_id=None,
                medium="gpu",
                lora_name=None,
            ),
            BlockRemoved(block_hashes=[33], medium="gpu"),
            AllBlocksCleared(),
        ],
        data_parallel_rank=2,
    )
    payload = _encode_batch(batch)
    mock_sub, _ = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(return_value=[b"", _seq_bytes(0), payload])
    mocker.patch(
        "subscriber.engine.vllm.incremental.logger.is_debug_enabled", return_value=True
    )
    mocker.patch(
        "subscriber.engine.vllm.incremental.generate_trace_id",
        return_value="live-trace",
    )
    debug = mocker.patch("subscriber.engine.vllm.incremental.logger.debug")

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    results = await _collect_n(adapter, 1)

    assert results == [[batch]]
    debug.assert_any_call(
        "decoded vLLM KV event batch",
        step="zmq_subscribe",
        tags={
            "seq": 0,
            "trace_id": "live-trace",
            "event_count": 3,
            "event_types": "AllBlocksCleared:1,BlockRemoved:1,BlockStored:1",
            "data_parallel_rank": 2,
            "stored_block_count": 1,
            "stored_blocks": [
                {
                    "block_hashes": ["11", "9223372036854775809"],
                    "parent_block_hash": None,
                    "block_size": 16,
                    "lora_id": None,
                    "medium": "gpu",
                    "lora_name": None,
                    "extra_keys": None,
                    "group_idx": None,
                    "component_id": None,
                    "kv_cache_spec_kind": None,
                    "kv_cache_spec_sliding_window": None,
                    "snapshot_version": 0,
                }
            ],
            "stored_blocks_truncated": False,
            "removed_block_count": 1,
            "removed_blocks": [
                {
                    "block_hashes": ["33"],
                    "medium": "gpu",
                    "group_idx": None,
                    "component_id": None,
                    "remaining_copy_counts": None,
                    "snapshot_version": 0,
                }
            ],
            "removed_blocks_truncated": False,
        },
    )


async def test_recv_live_message_skips_debug_when_disabled(
    config: SubscriberConfig, mocker: Any
) -> None:
    batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    payload = _encode_batch(batch)
    mock_sub, _ = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(return_value=[b"topic", _seq_bytes(7), payload])
    mocker.patch(
        "subscriber.engine.vllm.incremental.logger.is_debug_enabled", return_value=False
    )
    mocker.patch(
        "subscriber.engine.vllm.incremental.generate_trace_id",
        return_value="live-trace",
    )
    debug = mocker.patch("subscriber.engine.vllm.incremental.logger.debug")

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    assert await adapter._recv_live_message() == (7, payload, "live-trace")

    debug.assert_not_called()


async def test_triggers_replay_on_gap(config: SubscriberConfig, mocker: Any) -> None:
    batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    payload = _encode_batch(batch)
    missed_batch = KVEventBatch(ts=0.5, events=[AllBlocksCleared()])
    missed_payload = _encode_batch(missed_batch)

    mock_sub, mock_dealer = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), payload],
            [b"", _seq_bytes(2), payload],
        ]
    )
    END_SEQ = (-1).to_bytes(8, "big", signed=True)
    mock_dealer.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(1), missed_payload],
            [b"", END_SEQ, b""],
        ]
    )
    mock_dealer.send_multipart = AsyncMock()
    mock_sub.getsockopt.side_effect = lambda option: {
        zmq.EVENTS: 0,
        zmq.RCVHWM: 1000,
    }[option]
    mocker.patch(
        "subscriber.engine.vllm.incremental.generate_trace_id",
        side_effect=["first-live-trace", "second-live-trace", "replay-trace"],
    )
    warning = mocker.patch("subscriber.engine.vllm.incremental.logger.warning")

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    events = await _collect_events_n(adapter, 3)

    assert len(events) == 3
    assert len(events[1].batches) == 1
    mock_dealer.send_multipart.assert_called_once_with([b"", (1).to_bytes(8, "big")])
    # After the refactor, the gap warning only carries static socket state
    # plus the per-window byte / queue-nonempty observations. Cumulative
    # gap/miss counters are attached to the replay batch's BatchTelemetry
    # so downstream dashboards can attribute the loss to the batch that
    # exposed it.
    warning.assert_any_call(
        "kv event sequence gap detected, triggering replay",
        step="zmq_replay",
        tags={
            "last_seq": 0,
            "current_seq": 2,
            "missed": 1,
            "trace_id": "replay-trace",
            "zmq_sub_readable": False,
            "zmq_sub_rcvhwm": 1000,
            "zmq_exact_queue_depth_available": False,
            "zmq_received_message_bytes": 2 * (8 + len(payload)),
            "zmq_queue_nonempty_observation_count": 0,
        },
    )
    replay_event = events[1]
    assert replay_event.telemetry.counters == {
        "zmq_sequence_gap_count": 1,
        "zmq_missed_message_count": 1,
    }


async def test_replay_logs_decoded_event_types_and_block_hashes(
    config: SubscriberConfig, mocker: Any
) -> None:
    live_batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    live_payload = _encode_batch(live_batch)
    replay_batch = KVEventBatch(
        ts=0.5,
        events=[
            BlockStored(
                block_hashes=[44, 55],
                parent_block_hash=None,
                token_ids=[4, 5],
                block_size=16,
                lora_id=None,
                medium="gpu",
                lora_name=None,
            )
        ],
        data_parallel_rank=1,
    )
    replay_payload = _encode_batch(replay_batch)

    mock_sub, mock_dealer = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), live_payload],
            [b"", _seq_bytes(2), live_payload],
        ]
    )
    mock_dealer.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(1), replay_payload],
            [b"", (-1).to_bytes(8, "big", signed=True), b""],
        ]
    )
    mock_dealer.send_multipart = AsyncMock()
    mocker.patch(
        "subscriber.engine.vllm.incremental.logger.is_debug_enabled", return_value=True
    )
    mocker.patch(
        "subscriber.engine.vllm.incremental.generate_trace_id",
        side_effect=["first-live-trace", "second-live-trace", "replay-trace"],
    )
    debug = mocker.patch("subscriber.engine.vllm.incremental.logger.debug")

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    await _collect_n(adapter, 3)

    debug.assert_any_call(
        "decoded vLLM KV event batch",
        step="zmq_replay",
        tags={
            "gap_start_seq": 1,
            "current_seq": 2,
            "replay_seq": 1,
            "trace_id": "replay-trace",
            "event_count": 1,
            "event_types": "BlockStored:1",
            "data_parallel_rank": 1,
            "stored_block_count": 1,
            "stored_blocks": [
                {
                    "block_hashes": ["44", "55"],
                    "parent_block_hash": None,
                    "block_size": 16,
                    "lora_id": None,
                    "medium": "gpu",
                    "lora_name": None,
                    "extra_keys": None,
                    "group_idx": None,
                    "component_id": None,
                    "kv_cache_spec_kind": None,
                    "kv_cache_spec_sliding_window": None,
                    "snapshot_version": 0,
                }
            ],
            "stored_blocks_truncated": False,
            "removed_block_count": 0,
            "removed_blocks": [],
            "removed_blocks_truncated": False,
        },
    )


async def test_skips_bad_live_payload_without_advancing_sequence(
    config: SubscriberConfig, mocker: Any
) -> None:
    batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    payload = _encode_batch(batch)
    replay_batch = KVEventBatch(ts=0.5, events=[AllBlocksCleared()])
    replay_payload = _encode_batch(replay_batch)

    mock_sub, mock_dealer = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), b"not msgpack"],
            [b"", _seq_bytes(1), payload],
        ]
    )
    mock_dealer.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), replay_payload],
            [b"", (-1).to_bytes(8, "big", signed=True), b""],
        ]
    )
    mock_dealer.send_multipart = AsyncMock()

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    results = await _collect_n(adapter, 2)

    assert results == [[replay_batch], [batch]]
    mock_dealer.send_multipart.assert_called_once_with([b"", (0).to_bytes(8, "big")])


async def test_replay_pr_45177_four_frames_filters_current_and_newer_seqs(
    config: SubscriberConfig, mocker: Any
) -> None:
    """vLLM PR #45177 replay replies include topic, so DEALER receives four
    frames. The replay endpoint returns every buffered batch with seq >=
    gap_start_seq, including the live batch that exposed the gap and any newer
    batches. Only the missing range must be forwarded via replay; the rest
    arrive live."""

    batch0 = KVEventBatch(ts=0.0, events=[AllBlocksCleared()])
    batch1 = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    batch2 = KVEventBatch(ts=2.0, events=[AllBlocksCleared()])
    batch3 = KVEventBatch(ts=3.0, events=[AllBlocksCleared()])

    mock_sub, mock_dealer = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), _encode_batch(batch0)],
            [b"", _seq_bytes(2), _encode_batch(batch2)],
            [b"", _seq_bytes(3), _encode_batch(batch3)],
        ]
    )
    mock_dealer.recv_multipart = AsyncMock(
        side_effect=[
            [b"", b"", _seq_bytes(1), _encode_batch(batch1)],
            [b"", b"", _seq_bytes(2), _encode_batch(batch2)],
            [b"", b"", _seq_bytes(3), _encode_batch(batch3)],
            [b"", b"", (-1).to_bytes(8, "big", signed=True), b""],
        ]
    )
    mock_dealer.send_multipart = AsyncMock()

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    results = await _collect_n(adapter, 4)

    assert results == [[batch0], [batch1], [batch2], [batch3]]
    mock_dealer.send_multipart.assert_awaited_once_with([b"", (1).to_bytes(8, "big")])


async def test_successful_replay_advances_sequence_despite_live_decode_failure(
    config: SubscriberConfig, mocker: Any
) -> None:
    """A completed replay resolves the gap up to current_seq - 1. If the live
    payload that exposed the gap then fails to decode, the next live message
    must not replay the already-forwarded range again."""

    batch0 = KVEventBatch(ts=0.0, events=[AllBlocksCleared()])
    batch1 = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    batch2 = KVEventBatch(ts=2.0, events=[AllBlocksCleared()])
    batch3 = KVEventBatch(ts=3.0, events=[AllBlocksCleared()])

    mock_sub, mock_dealer = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), _encode_batch(batch0)],
            [b"", _seq_bytes(2), b"not msgpack"],
            [b"", _seq_bytes(3), _encode_batch(batch3)],
        ]
    )
    mock_dealer.recv_multipart = AsyncMock(
        side_effect=[
            # First replay for gap [1, 1].
            [b"", b"", _seq_bytes(1), _encode_batch(batch1)],
            [b"", b"", (-1).to_bytes(8, "big", signed=True), b""],
            # Second replay must request the undecoded live seq 2, not seq 1.
            [b"", b"", _seq_bytes(2), _encode_batch(batch2)],
            [b"", b"", (-1).to_bytes(8, "big", signed=True), b""],
        ]
    )
    mock_dealer.send_multipart = AsyncMock()

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    results = await _collect_n(adapter, 4)

    assert results == [[batch0], [batch1], [batch2], [batch3]]
    assert mock_dealer.send_multipart.await_args_list == [
        mocker.call([b"", (1).to_bytes(8, "big")]),
        mocker.call([b"", (2).to_bytes(8, "big")]),
    ]


@pytest.mark.parametrize("abort_reason", ["malformed_frames", "decode_failure"])
async def test_replay_abort_replaces_dealer_socket(
    config: SubscriberConfig, mocker: Any, abort_reason: str
) -> None:
    """Malformed frames or a decode failure abort the replay mid-stream; the
    DEALER socket must be replaced so leftover frames (later batches and the
    END marker) cannot poison the next replay request."""

    batch0 = KVEventBatch(ts=0.0, events=[AllBlocksCleared()])
    batch2 = KVEventBatch(ts=2.0, events=[AllBlocksCleared()])
    batch3 = KVEventBatch(ts=3.0, events=[AllBlocksCleared()])
    batch4 = KVEventBatch(ts=4.0, events=[AllBlocksCleared()])
    old_sub = MagicMock()
    old_dealer = MagicMock()
    replacement_dealer = MagicMock()
    _mock_adapter_socket_sequence(mocker, old_sub, old_dealer, replacement_dealer)
    old_sub.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), _encode_batch(batch0)],
            [b"", _seq_bytes(2), _encode_batch(batch2)],
            [b"", _seq_bytes(4), _encode_batch(batch4)],
        ]
    )
    if abort_reason == "malformed_frames":
        aborting_frames = [b"", _seq_bytes(1)]
    else:
        aborting_frames = [b"", _seq_bytes(1), b"not msgpack"]
    old_dealer.send_multipart = AsyncMock()
    old_dealer.recv_multipart = AsyncMock(side_effect=[aborting_frames])
    replacement_dealer.send_multipart = AsyncMock()
    replacement_dealer.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(3), _encode_batch(batch3)],
            [b"", (-1).to_bytes(8, "big", signed=True), b""],
        ]
    )

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    results = await _collect_n(adapter, 4)

    assert results == [[batch0], [batch2], [batch3], [batch4]]
    old_dealer.close.assert_called_once_with(linger=0)
    replacement_dealer.send_multipart.assert_awaited_once_with(
        [b"", (3).to_bytes(8, "big")]
    )


@pytest.mark.parametrize("failure_point", ["send", "recv"])
async def test_replay_transport_failure_forwards_current_and_later_live_batches(
    config: SubscriberConfig, mocker: Any, failure_point: str
) -> None:
    first_batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    gapped_batch = KVEventBatch(ts=2.0, events=[AllBlocksCleared()])
    later_batch = KVEventBatch(ts=3.0, events=[AllBlocksCleared()])
    old_sub = MagicMock()
    old_dealer = MagicMock()
    replacement_dealer = MagicMock()
    _mock_adapter_socket_sequence(mocker, old_sub, old_dealer, replacement_dealer)
    old_sub.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), _encode_batch(first_batch)],
            [b"", _seq_bytes(2), _encode_batch(gapped_batch)],
            [b"", _seq_bytes(3), _encode_batch(later_batch)],
        ]
    )
    if failure_point == "send":
        old_dealer.send_multipart = AsyncMock(side_effect=zmq.ZMQError("down"))
    else:
        old_dealer.send_multipart = AsyncMock()
        old_dealer.recv_multipart = AsyncMock(side_effect=zmq.ZMQError("down"))
    mocker.patch(
        "subscriber.engine.vllm.incremental.generate_trace_id",
        side_effect=[
            "first-live-trace",
            "second-live-trace",
            "replay-trace",
            "later-live-trace",
        ],
    )
    warning = mocker.patch("subscriber.engine.vllm.incremental.logger.warning")

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    results = await _collect_n(adapter, 3)

    assert results == [[first_batch], [gapped_batch], [later_batch]]
    assert old_dealer.send_multipart.await_count == 1
    old_dealer.close.assert_called_once_with(linger=0)
    warning.assert_any_call(
        "kv event replay unavailable; sequence gap remains; forwarding live batch",
        step="zmq_replay",
        tags={
            "gap_start_seq": 1,
            "current_seq": 2,
            "error": "ZMQError",
            "message": "down",
            "trace_id": "replay-trace",
        },
    )


async def test_replay_timeout_forwards_current_live_batch(
    config: SubscriberConfig, mocker: Any
) -> None:
    config.zmq_replay_timeout_s = 0.01
    first_batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    gapped_batch = KVEventBatch(ts=2.0, events=[AllBlocksCleared()])
    old_sub = MagicMock()
    old_dealer = MagicMock()
    replacement_dealer = MagicMock()
    _mock_adapter_socket_sequence(mocker, old_sub, old_dealer, replacement_dealer)
    old_sub.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), _encode_batch(first_batch)],
            [b"", _seq_bytes(2), _encode_batch(gapped_batch)],
        ]
    )
    old_dealer.send_multipart = AsyncMock()

    async def wait_forever() -> list[bytes]:
        await asyncio.Event().wait()
        return []

    old_dealer.recv_multipart = AsyncMock(side_effect=wait_forever)
    mocker.patch(
        "subscriber.engine.vllm.incremental.generate_trace_id",
        side_effect=["first-live-trace", "second-live-trace", "replay-trace"],
    )
    warning = mocker.patch("subscriber.engine.vllm.incremental.logger.warning")

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    results = await _collect_n(adapter, 2)

    assert results == [[first_batch], [gapped_batch]]
    old_dealer.close.assert_called_once_with(linger=0)
    warning.assert_any_call(
        "kv event replay unavailable; sequence gap remains; forwarding live batch",
        step="zmq_replay",
        tags={
            "gap_start_seq": 1,
            "current_seq": 2,
            "error": "TimeoutError",
            "message": "replay timed out",
            "trace_id": "replay-trace",
        },
    )


async def test_replay_send_timeout_forwards_current_live_batch(
    config: SubscriberConfig, mocker: Any
) -> None:
    config.zmq_replay_timeout_s = 0.01
    first_batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    gapped_batch = KVEventBatch(ts=2.0, events=[AllBlocksCleared()])
    old_sub = MagicMock()
    old_dealer = MagicMock()
    replacement_dealer = MagicMock()
    _mock_adapter_socket_sequence(mocker, old_sub, old_dealer, replacement_dealer)
    old_sub.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), _encode_batch(first_batch)],
            [b"", _seq_bytes(2), _encode_batch(gapped_batch)],
        ]
    )

    async def wait_forever(_frames: list[bytes]) -> None:
        await asyncio.Event().wait()

    old_dealer.send_multipart = AsyncMock(side_effect=wait_forever)
    mocker.patch(
        "subscriber.engine.vllm.incremental.generate_trace_id",
        side_effect=["first-live-trace", "second-live-trace", "replay-trace"],
    )
    warning = mocker.patch("subscriber.engine.vllm.incremental.logger.warning")

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    results = await _collect_n(adapter, 2)

    assert results == [[first_batch], [gapped_batch]]
    old_dealer.close.assert_called_once_with(linger=0)
    warning.assert_any_call(
        "kv event replay unavailable; sequence gap remains; forwarding live batch",
        step="zmq_replay",
        tags={
            "gap_start_seq": 1,
            "current_seq": 2,
            "error": "TimeoutError",
            "message": "replay timed out",
            "trace_id": "replay-trace",
        },
    )


async def test_replacement_replay_socket_recovers_a_later_gap(
    config: SubscriberConfig, mocker: Any
) -> None:
    first_batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    first_gapped_batch = KVEventBatch(ts=2.0, events=[AllBlocksCleared()])
    later_batch = KVEventBatch(ts=3.0, events=[AllBlocksCleared()])
    replayed_batch = KVEventBatch(ts=4.0, events=[AllBlocksCleared()])
    second_gapped_batch = KVEventBatch(ts=5.0, events=[AllBlocksCleared()])
    old_sub = MagicMock()
    old_dealer = MagicMock()
    replacement_dealer = MagicMock()
    _mock_adapter_socket_sequence(mocker, old_sub, old_dealer, replacement_dealer)
    old_sub.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), _encode_batch(first_batch)],
            [b"", _seq_bytes(2), _encode_batch(first_gapped_batch)],
            [b"", _seq_bytes(3), _encode_batch(later_batch)],
            [b"", _seq_bytes(5), _encode_batch(second_gapped_batch)],
        ]
    )
    old_dealer.send_multipart = AsyncMock(side_effect=zmq.ZMQError("down"))
    replacement_dealer.send_multipart = AsyncMock()
    replacement_dealer.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(4), _encode_batch(replayed_batch)],
            [b"", (-1).to_bytes(8, "big", signed=True), b""],
        ]
    )

    adapter = VllmIncrementalSource(config, endpoint=_dp_endpoint())
    results = await _collect_n(adapter, 5)

    assert results == [
        [first_batch],
        [first_gapped_batch],
        [later_batch],
        [replayed_batch],
        [second_gapped_batch],
    ]
    replacement_dealer.send_multipart.assert_awaited_once_with(
        [b"", (4).to_bytes(8, "big")]
    )


def test_adapter_configures_sub_socket_transport(
    config: SubscriberConfig, mocker: Any
) -> None:
    mock_sub, mock_dealer = _mock_adapter_sockets(mocker)
    config.zmq_reconnect_ivl_ms = 250
    config.zmq_reconnect_ivl_max_ms = 4000
    config.zmq_tcp_keepalive = True
    config.zmq_tcp_keepalive_idle_s = 9
    config.zmq_tcp_keepalive_intvl_s = 3
    config.zmq_tcp_keepalive_cnt = 5

    VllmIncrementalSource(config, endpoint=_dp_endpoint())

    mock_sub.setsockopt.assert_any_call(zmq.RECONNECT_IVL, 250)
    mock_sub.setsockopt.assert_any_call(zmq.RECONNECT_IVL_MAX, 4000)
    mock_sub.setsockopt.assert_any_call(zmq.TCP_KEEPALIVE, 1)
    mock_sub.setsockopt.assert_any_call(zmq.TCP_KEEPALIVE_IDLE, 9)
    mock_sub.setsockopt.assert_any_call(zmq.TCP_KEEPALIVE_INTVL, 3)
    mock_sub.setsockopt.assert_any_call(zmq.TCP_KEEPALIVE_CNT, 5)


def test_adapter_disables_keepalive_when_configured(
    config: SubscriberConfig, mocker: Any
) -> None:
    mock_sub, mock_dealer = _mock_adapter_sockets(mocker)
    config.zmq_tcp_keepalive = False

    VllmIncrementalSource(config, endpoint=_dp_endpoint())

    mock_sub.setsockopt.assert_any_call(zmq.TCP_KEEPALIVE, 0)
