from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import msgspec
import pytest
import zmq

from subscriber.config import SubscriberConfig
from subscriber.engine import vllm as vllm_module
from subscriber.engine.base import EngineEventBatch
from subscriber.engine.vllm import VllmAdapter, _probe_health
from subscriber.health.events import LivenessEvent
from subscriber.proto import kv_cache_group_metadata_pb2
from subscriber.types import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KvCacheGroupSpec,
    KVEventBatch,
)


@pytest.fixture
def config() -> SubscriberConfig:
    return SubscriberConfig()


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
        "subscriber.engine.vllm.zmq.asyncio.Context.instance", return_value=mock_ctx
    )
    return mock_sub, mock_dealer


def _mock_adapter_socket_sequence(mocker: Any, *sockets: MagicMock) -> MagicMock:
    mock_ctx = MagicMock()
    mock_ctx.socket.side_effect = list(sockets)
    mocker.patch(
        "subscriber.engine.vllm.zmq.asyncio.Context.instance", return_value=mock_ctx
    )
    return mock_ctx


class _FakeAsyncClient:
    """Async context manager that returns queued responses / exceptions."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.get_calls: list[str] = []

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def get(self, url: str) -> httpx.Response:
        self.get_calls.append(url)
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _kv_cache_group_metadata_response(
    items: list[dict[str, int | str]],
) -> object:
    response = kv_cache_group_metadata_pb2.KvCacheGroupListPB()
    for item in items:
        response.items.add(**item)
    return response


def _mock_dashllm_grpc_client(mocker: Any, response: object) -> MagicMock:
    client = MagicMock()
    client.get_kv_cache_group_metadata = AsyncMock(return_value=response)
    client.close = AsyncMock()
    mocker.patch(
        "subscriber.engine.vllm.DashllmGrpcClient",
        return_value=client,
        create=True,
    )
    return client


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("GET", "http://x/health"),
    )


async def test_probe_health_2xx_is_healthy() -> None:
    client = _FakeAsyncClient([_response(200)])
    async with client as active:
        assert await _probe_health(active, "http://x/health") is LivenessEvent.HEALTHY


async def test_probe_health_non_2xx_is_unhealthy() -> None:
    client = _FakeAsyncClient([_response(503)])
    async with client as active:
        assert await _probe_health(active, "http://x/health") is LivenessEvent.UNHEALTHY


async def test_probe_health_http_error_is_unhealthy() -> None:
    client = _FakeAsyncClient([httpx.ConnectError("boom")])
    async with client as active:
        assert await _probe_health(active, "http://x/health") is LivenessEvent.UNHEALTHY


async def test_probe_health_timeout_is_unhealthy() -> None:
    client = _FakeAsyncClient([httpx.ReadTimeout("slow")])
    async with client as active:
        assert await _probe_health(active, "http://x/health") is LivenessEvent.UNHEALTHY


async def test_probe_health_2xx_does_not_warn(mocker: Any) -> None:
    warning = mocker.patch("subscriber.engine.vllm.logger.warning")
    client = _FakeAsyncClient([_response(200)])
    async with client as active:
        await _probe_health(active, "http://x/health")
    warning.assert_not_called()


async def test_probe_health_timeout_logs_timeout(mocker: Any) -> None:
    warning = mocker.patch("subscriber.engine.vllm.logger.warning")
    client = _FakeAsyncClient([httpx.ReadTimeout("slow")])
    async with client as active:
        await _probe_health(active, "http://x/health")
    warning.assert_called_once()
    tags = warning.call_args.kwargs["tags"]
    assert tags["error"] == "ReadTimeout"


async def test_probe_health_connect_timeout_logs_connect_timeout(mocker: Any) -> None:
    warning = mocker.patch("subscriber.engine.vllm.logger.warning")
    client = _FakeAsyncClient([httpx.ConnectTimeout("conn slow")])
    async with client as active:
        await _probe_health(active, "http://x/health")
    warning.assert_called_once()
    tags = warning.call_args.kwargs["tags"]
    assert tags["error"] == "ConnectTimeout"


def test_parse_kv_cache_group_metadata_builds_specs_by_group_idx() -> None:
    response = _kv_cache_group_metadata_response(
        [
            {
                "group_idx": 0,
                "kind": "full_attention",
                "block_size": 16,
                "sliding_window": -1,
            },
            {
                "group_idx": 1,
                "kind": "sliding_window",
                "block_size": 16,
                "sliding_window": 4096,
            },
        ]
    )
    parse = getattr(vllm_module, "_parse_kv_cache_group_metadata", None)
    assert parse is not None
    assert parse(response) == [
        KvCacheGroupSpec(
            group_idx=0, kind="full_attention", block_size=16, sliding_window=None
        ),
        KvCacheGroupSpec(
            group_idx=1, kind="sliding_window", block_size=16, sliding_window=4096
        ),
    ]


def test_parse_kv_cache_group_metadata_null_or_missing_returns_none() -> None:
    parse = getattr(vllm_module, "_parse_kv_cache_group_metadata", None)
    assert parse is not None
    assert parse(_kv_cache_group_metadata_response([])) is None
    assert parse("not a protobuf response") is None


async def test_adapter_owns_one_dashllm_grpc_client(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    client = _mock_dashllm_grpc_client(
        mocker,
        _kv_cache_group_metadata_response([]),
    )

    adapter = VllmAdapter(config)

    assert adapter._dashllm_grpc_client is client


async def test_adapter_close_releases_grpc_client_and_zmq_sockets(
    config: SubscriberConfig, mocker: Any
) -> None:
    sub, dealer = _mock_adapter_sockets(mocker)
    client = _mock_dashllm_grpc_client(
        mocker,
        _kv_cache_group_metadata_response([]),
    )
    adapter = VllmAdapter(config)

    await adapter.close()

    client.close.assert_awaited_once()
    sub.close.assert_called_once_with(linger=0)
    dealer.close.assert_called_once_with(linger=0)


async def test_fetch_kv_cache_group_metadata_parses_response(
    config: SubscriberConfig, mocker: Any
) -> None:
    response = _kv_cache_group_metadata_response(
        [
            {
                "group_idx": 0,
                "kind": "mamba",
                "block_size": 16,
                "sliding_window": -1,
            }
        ]
    )
    _mock_adapter_sockets(mocker)
    client = _mock_dashllm_grpc_client(mocker, response)
    mocker.patch(
        "subscriber.engine.vllm.httpx.AsyncClient",
        side_effect=lambda **_: pytest.fail("metadata must not use HTTP"),
    )

    adapter = VllmAdapter(config)
    assert await adapter.fetch_kv_cache_group_metadata() == [
        KvCacheGroupSpec(group_idx=0, kind="mamba", block_size=16, sliding_window=None)
    ]
    client.get_kv_cache_group_metadata.assert_awaited_once_with(
        config.engine_health_timeout_s
    )


async def test_fetch_kv_cache_group_metadata_retries_until_success(
    config: SubscriberConfig, mocker: Any
) -> None:
    response = _kv_cache_group_metadata_response([])
    _mock_adapter_sockets(mocker)
    client = _mock_dashllm_grpc_client(mocker, response)
    client.get_kv_cache_group_metadata = AsyncMock(
        side_effect=[RuntimeError("down"), response]
    )
    mocker.patch(
        "subscriber.engine.vllm.httpx.AsyncClient",
        side_effect=lambda **_: pytest.fail("metadata must not use HTTP"),
    )
    sleep_mock = mocker.patch("subscriber.engine.vllm.asyncio.sleep", AsyncMock())

    adapter = VllmAdapter(config)
    assert await adapter.fetch_kv_cache_group_metadata() is None
    assert sleep_mock.await_count == 1
    assert client.get_kv_cache_group_metadata.await_count == 2


async def test_fetch_kv_cache_group_metadata_retries_on_retryable_response_code(
    config: SubscriberConfig, mocker: Any
) -> None:
    retryable_response = kv_cache_group_metadata_pb2.KvCacheGroupListPB()
    retryable_response.err_code = (
        kv_cache_group_metadata_pb2.KV_CACHE_GROUP_METADATA_UNAVAILABLE
    )
    retryable_response.err_msg = "engine metadata is not ready"
    success_response = _kv_cache_group_metadata_response(
        [
            {
                "group_idx": 0,
                "kind": "full_attention",
                "block_size": 16,
                "sliding_window": -1,
            }
        ]
    )
    _mock_adapter_sockets(mocker)
    client = _mock_dashllm_grpc_client(mocker, success_response)
    client.get_kv_cache_group_metadata = AsyncMock(
        side_effect=[retryable_response, success_response]
    )
    sleep_mock = mocker.patch("subscriber.engine.vllm.asyncio.sleep", AsyncMock())

    adapter = VllmAdapter(config)

    assert await adapter.fetch_kv_cache_group_metadata() == [
        KvCacheGroupSpec(
            group_idx=0,
            kind="full_attention",
            block_size=16,
            sliding_window=None,
        )
    ]
    assert sleep_mock.await_count == 1
    assert client.get_kv_cache_group_metadata.await_count == 2


async def test_probe_health_connect_error_logs_connect_error(mocker: Any) -> None:
    warning = mocker.patch("subscriber.engine.vllm.logger.warning")
    client = _FakeAsyncClient([httpx.ConnectError("refused")])
    async with client as active:
        await _probe_health(active, "http://x/health")
    warning.assert_called_once()
    tags = warning.call_args.kwargs["tags"]
    assert tags["error"] == "ConnectError"


async def test_probe_health_non_2xx_logs_status_code(mocker: Any) -> None:
    warning = mocker.patch("subscriber.engine.vllm.logger.warning")
    client = _FakeAsyncClient([_response(503)])
    async with client as active:
        await _probe_health(active, "http://x/health")
    warning.assert_called_once()
    tags = warning.call_args.kwargs["tags"]
    assert tags["status_code"] == 503


async def test_watch_liveness_polls_and_maps_health(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    fake_client = _FakeAsyncClient(
        [
            _response(200),
            _response(500),
            httpx.ConnectError("boom"),
            _response(200),
        ]
    )
    mocker.patch("subscriber.engine.vllm.httpx.AsyncClient", return_value=fake_client)
    sleep_mock = mocker.patch("subscriber.engine.vllm.asyncio.sleep", AsyncMock())

    adapter = VllmAdapter(config)
    events: list[LivenessEvent] = []
    async for event in adapter.watch_liveness():
        events.append(event)
        if len(events) == 4:
            break

    assert events == [
        LivenessEvent.HEALTHY,
        LivenessEvent.UNHEALTHY,
        LivenessEvent.UNHEALTHY,
        LivenessEvent.HEALTHY,
    ]
    assert fake_client.get_calls == ["http://127.0.0.1:8601/readiness"] * 4
    assert sleep_mock.await_count == 3
    sleep_mock.assert_awaited_with(config.engine_health_interval_s)


def test_adapter_opens_sub_and_dealer_without_monitor(
    config: SubscriberConfig, mocker: Any
) -> None:
    mock_sub, mock_dealer = _mock_adapter_sockets(mocker)

    VllmAdapter(config)

    mock_sub.connect.assert_called_once_with(config.zmq_pub_endpoint)
    mock_sub.setsockopt_string.assert_called_once()
    mock_dealer.connect.assert_called_once_with(config.zmq_replay_endpoint)
    mock_sub.get_monitor_socket.assert_not_called()


async def test_live_receive_records_zmq_queue_backlog_signal(
    config: SubscriberConfig, mocker: Any
) -> None:
    mock_sub, _ = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(return_value=[b"topic", _seq_bytes(7), b"body"])
    mock_sub.getsockopt.return_value = zmq.POLLIN
    adapter = VllmAdapter(config)
    queue_metrics = mocker.Mock()
    adapter._zmq_queue_metrics = queue_metrics

    assert await adapter._recv_live_message() == (7, b"body")

    queue_metrics.record_message.assert_called_once_with(
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
    adapter = VllmAdapter(config)

    assert adapter._zmq_queue_state() == {
        "zmq_sub_readable": True,
        "zmq_sub_rcvhwm": 1000,
        "zmq_exact_queue_depth_available": False,
    }


def test_adapter_logs_zmq_debug_endpoints(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    mocker.patch("subscriber.engine.vllm.logger.is_debug_enabled", return_value=True)
    debug = mocker.patch("subscriber.engine.vllm.logger.debug")

    VllmAdapter(config)

    debug.assert_any_call(
        "connecting vLLM ZMQ sockets",
        step="zmq_connect",
        tags={
            "pub_endpoint": config.zmq_pub_endpoint,
            "replay_endpoint": config.zmq_replay_endpoint,
            "topic": config.zmq_topic,
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
    adapter = VllmAdapter(config)
    adapter._last_seq = 99

    await adapter.reset_generation_state()

    assert adapter._last_seq == -1
    old_sub.close.assert_called_once_with(linger=0)
    old_dealer.close.assert_called_once_with(linger=0)
    new_sub.connect.assert_called_once_with(config.zmq_pub_endpoint)
    new_sub.setsockopt_string.assert_called_once()
    new_dealer.connect.assert_called_once_with(config.zmq_replay_endpoint)


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

    adapter = VllmAdapter(config)
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

    adapter = VllmAdapter(config)
    adapter._last_seq = 0
    replay_task = asyncio.create_task(adapter._replay_missing_batches(5, 0))
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

    adapter = VllmAdapter(config)
    adapter._last_seq = 0
    stale_task = asyncio.create_task(adapter._replay_missing_batches(5, 0))
    await recv_started.wait()

    await adapter.reset_generation_state()
    release_recv.set()
    assert await asyncio.wait_for(stale_task, timeout=1.0) is None

    result = await adapter._replay_missing_batches(5, adapter._generation)
    assert result == [fresh_batch]
    new_dealer.send_multipart.assert_awaited_once_with([b"", (0).to_bytes(8, "big")])


async def _collect_n(adapter: VllmAdapter, n: int) -> list[list[KVEventBatch]]:
    events = await _collect_events_n(adapter, n)
    return [event.batches for event in events]


async def _collect_events_n(adapter: VllmAdapter, n: int) -> list[EngineEventBatch]:
    results: list[EngineEventBatch] = []

    async def _run() -> None:
        async for event in adapter.subscribe_kv_events():
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

    adapter = VllmAdapter(config)
    events = await _collect_events_n(adapter, 1)

    spans = events[0].timer.spans()
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
            [b"", _seq_bytes(1), replay_payload],
            [b"", (-1).to_bytes(8, "big", signed=True), b""],
        ]
    )
    mock_dealer.send_multipart = AsyncMock()

    adapter = VllmAdapter(config)
    events = await _collect_events_n(adapter, 1)

    spans = events[0].timer.spans()
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

    adapter = VllmAdapter(config)
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
    mocker.patch("subscriber.engine.vllm.logger.is_debug_enabled", return_value=True)
    debug = mocker.patch("subscriber.engine.vllm.logger.debug")

    adapter = VllmAdapter(config)
    assert await adapter._recv_live_message() == (7, payload)

    debug.assert_any_call(
        "received vLLM ZMQ live message",
        step="zmq_subscribe",
        tags={"topic": "topic", "seq": 7, "payload_bytes": len(payload)},
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
    mocker.patch("subscriber.engine.vllm.logger.is_debug_enabled", return_value=True)
    debug = mocker.patch("subscriber.engine.vllm.logger.debug")

    adapter = VllmAdapter(config)
    results = await _collect_n(adapter, 1)

    assert results == [[batch]]
    debug.assert_any_call(
        "decoded vLLM KV event batch",
        step="zmq_subscribe",
        tags={
            "seq": 0,
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
                    "kv_cache_spec_kind": None,
                    "kv_cache_spec_sliding_window": None,
                }
            ],
            "stored_blocks_truncated": False,
            "removed_block_count": 1,
            "removed_blocks": [
                {
                    "block_hashes": ["33"],
                    "medium": "gpu",
                    "group_idx": None,
                    "remaining_copy_counts": None,
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
    mocker.patch("subscriber.engine.vllm.logger.is_debug_enabled", return_value=False)
    debug = mocker.patch("subscriber.engine.vllm.logger.debug")

    adapter = VllmAdapter(config)
    assert await adapter._recv_live_message() == (7, payload)

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

    adapter = VllmAdapter(config)
    results = await _collect_n(adapter, 3)

    assert len(results) == 3
    assert len(results[1]) == 1
    mock_dealer.send_multipart.assert_called_once_with([b"", (1).to_bytes(8, "big")])


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
    mocker.patch("subscriber.engine.vllm.logger.is_debug_enabled", return_value=True)
    debug = mocker.patch("subscriber.engine.vllm.logger.debug")

    adapter = VllmAdapter(config)
    await _collect_n(adapter, 3)

    debug.assert_any_call(
        "decoded vLLM KV event batch",
        step="zmq_replay",
        tags={
            "gap_start_seq": 1,
            "current_seq": 2,
            "replay_seq": 1,
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
                    "kv_cache_spec_kind": None,
                    "kv_cache_spec_sliding_window": None,
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

    adapter = VllmAdapter(config)
    results = await _collect_n(adapter, 2)

    assert results == [[replay_batch], [batch]]
    mock_dealer.send_multipart.assert_called_once_with([b"", (0).to_bytes(8, "big")])


async def test_skips_bad_replay_payload_and_continues_live_stream(
    config: SubscriberConfig, mocker: Any
) -> None:
    batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    payload = _encode_batch(batch)

    mock_sub, mock_dealer = _mock_adapter_sockets(mocker)
    mock_sub.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(0), payload],
            [b"", _seq_bytes(2), payload],
            [b"", _seq_bytes(3), payload],
        ]
    )
    mock_dealer.recv_multipart = AsyncMock(
        side_effect=[
            [b"", _seq_bytes(1), b"not msgpack"],
            [b"", _seq_bytes(1), payload],
            [b"", _seq_bytes(2), payload],
            [b"", (-1).to_bytes(8, "big", signed=True), b""],
        ]
    )
    mock_dealer.send_multipart = AsyncMock()

    adapter = VllmAdapter(config)
    results = await _collect_n(adapter, 3)

    assert results == [[batch], [batch], [batch]]
    mock_dealer.send_multipart.assert_awaited_once_with([b"", (1).to_bytes(8, "big")])


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
    warning = mocker.patch("subscriber.engine.vllm.logger.warning")

    adapter = VllmAdapter(config)
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
    warning = mocker.patch("subscriber.engine.vllm.logger.warning")

    adapter = VllmAdapter(config)
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
    warning = mocker.patch("subscriber.engine.vllm.logger.warning")

    adapter = VllmAdapter(config)
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

    adapter = VllmAdapter(config)
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


async def test_watch_liveness_converts_unexpected_probe_error_to_unhealthy(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
    fake_client = _FakeAsyncClient(
        [
            RuntimeError("boom"),
            _response(200),
        ]
    )
    mocker.patch("subscriber.engine.vllm.httpx.AsyncClient", return_value=fake_client)
    mocker.patch("subscriber.engine.vllm.asyncio.sleep", AsyncMock())

    adapter = VllmAdapter(config)
    events: list[LivenessEvent] = []
    async for event in adapter.watch_liveness():
        events.append(event)
        if len(events) == 2:
            break

    assert events == [LivenessEvent.UNHEALTHY, LivenessEvent.HEALTHY]


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

    VllmAdapter(config)

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

    VllmAdapter(config)

    mock_sub.setsockopt.assert_any_call(zmq.TCP_KEEPALIVE, 0)
