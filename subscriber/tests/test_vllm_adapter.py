from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import msgspec
import pytest
import zmq

from subscriber.config import SubscriberConfig
from subscriber.engine.vllm import VllmAdapter, _probe_health
from subscriber.health.events import LivenessEvent
from subscriber.types import AllBlocksCleared, BlockRemoved, BlockStored, KVEventBatch


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
    assert fake_client.get_calls == [config.engine_health_url] * 4
    assert sleep_mock.await_count == 4
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


def test_adapter_logs_zmq_debug_endpoints(
    config: SubscriberConfig, mocker: Any
) -> None:
    _mock_adapter_sockets(mocker)
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


def test_reset_generation_state_resets_seq_and_recreates_dealer(
    config: SubscriberConfig, mocker: Any
) -> None:
    mock_sub = MagicMock()
    old_dealer = MagicMock()
    new_dealer = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.socket.side_effect = [mock_sub, old_dealer, new_dealer]
    mocker.patch(
        "subscriber.engine.vllm.zmq.asyncio.Context.instance", return_value=mock_ctx
    )
    adapter = VllmAdapter(config)
    adapter._last_seq = 99

    adapter.reset_generation_state()

    assert adapter._last_seq == -1
    old_dealer.close.assert_called_once_with(linger=0)
    new_dealer.connect.assert_called_once_with(config.zmq_replay_endpoint)


async def _collect_n(adapter: VllmAdapter, n: int) -> list[list[KVEventBatch]]:
    results: list[list[KVEventBatch]] = []

    async def _run() -> None:
        async for batches in adapter.subscribe_kv_events():
            results.append(batches)
            if len(results) >= n:
                break

    await asyncio.wait_for(_run(), timeout=1.0)
    return results


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
                block_hashes=[11, 22],
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
            "stored_block_hash_count": 2,
            "stored_block_hashes": [11, 22],
            "stored_block_hashes_truncated": False,
            "removed_block_hash_count": 1,
            "removed_block_hashes": [33],
            "removed_block_hashes_truncated": False,
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

    debug.assert_called_once_with(
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
            "stored_block_hash_count": 2,
            "stored_block_hashes": [44, 55],
            "stored_block_hashes_truncated": False,
            "removed_block_hash_count": 0,
            "removed_block_hashes": [],
            "removed_block_hashes_truncated": False,
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

    assert results == [[batch], [batch, batch], [batch]]


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
