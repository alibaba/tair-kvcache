from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from subscriber.config import SubscriberConfig
from subscriber.main import (
    QueuedKVEventBatch,
    consume_kv_events,
    kv_event_loop,
    send_kv_events,
)
from subscriber.types import AllBlocksCleared, KVEventBatch


def _batch() -> list[KVEventBatch]:
    return [KVEventBatch(ts=1.0, events=[AllBlocksCleared()])]


@pytest.fixture
def adapter() -> MagicMock:
    mock = MagicMock()

    async def _subscribe():
        return
        yield  # pragma: no cover

    mock.subscribe_kv_events = _subscribe
    return mock


@pytest.fixture
def kvcm() -> MagicMock:
    mock = MagicMock()
    mock.send_batch = AsyncMock()
    return mock


@pytest.fixture
def coordinator() -> MagicMock:
    mock = MagicMock()
    mock.wait_ready_epoch = AsyncMock(return_value=1)
    mock.capture_epoch = MagicMock(return_value=1)
    mock.is_epoch_current = MagicMock(return_value=True)
    return mock


async def test_consume_kv_events_queues_batch_with_epoch_snapshot(
    adapter: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch()
    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue(maxsize=1)

    async def _subscribe():
        yield batch

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_epoch.return_value = 3

    await consume_kv_events(adapter, coordinator, queue)

    queued = queue.get_nowait()
    assert queued.batches == batch
    assert queued.epoch_snapshot == 3


async def test_consume_kv_events_drops_batch_when_not_ready(
    adapter: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch()
    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue(maxsize=1)

    async def _subscribe():
        yield batch

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_epoch.return_value = None

    await consume_kv_events(adapter, coordinator, queue)

    assert queue.empty()


async def test_send_kv_events_sends_batch_when_epoch_unchanged(
    kvcm: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch()
    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue()
    await queue.put(QueuedKVEventBatch(batch, 1))

    sender = asyncio.create_task(send_kv_events(kvcm, coordinator, queue))
    await asyncio.sleep(0)
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    kvcm.send_batch.assert_awaited_once_with(batch, 1)
    assert queue.empty()


async def test_send_kv_events_drops_batch_when_epoch_changed(
    kvcm: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch()
    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue()
    await queue.put(QueuedKVEventBatch(batch, 1))
    coordinator.wait_ready_epoch.return_value = 2
    coordinator.is_epoch_current.return_value = False

    sender = asyncio.create_task(send_kv_events(kvcm, coordinator, queue))
    await asyncio.sleep(0)
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    kvcm.send_batch.assert_not_awaited()
    assert queue.empty()


async def test_kv_event_loop_sends_batch_when_epoch_unchanged(
    adapter: MagicMock, kvcm: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch()

    async def _subscribe():
        yield batch

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_epoch.return_value = 1
    coordinator.wait_ready_epoch.return_value = 1
    coordinator.is_epoch_current.return_value = True

    with pytest.raises(RuntimeError, match="kv event subscription ended unexpectedly"):
        await kv_event_loop(adapter, kvcm, coordinator)

    kvcm.send_batch.assert_awaited_once_with(batch, 1)


async def test_kv_event_loop_drops_batch_when_epoch_changed(
    adapter: MagicMock, kvcm: MagicMock, coordinator: MagicMock
) -> None:
    """Engine crashes after batch captured; epoch bumps before gate opens."""
    batch = _batch()

    async def _subscribe():
        yield batch

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_epoch.return_value = 1
    coordinator.wait_ready_epoch.return_value = 2
    coordinator.is_epoch_current.return_value = False

    with pytest.raises(RuntimeError, match="kv event subscription ended unexpectedly"):
        await kv_event_loop(adapter, kvcm, coordinator)

    kvcm.send_batch.assert_not_awaited()


async def test_kv_event_loop_drops_batch_when_not_ready(
    adapter: MagicMock, kvcm: MagicMock, coordinator: MagicMock
) -> None:
    """Batch captured while gate is closed (cold start before first healthy)."""
    batch = _batch()

    async def _subscribe():
        yield batch

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_epoch.return_value = None
    coordinator.wait_ready_epoch.return_value = 1

    with pytest.raises(RuntimeError, match="kv event subscription ended unexpectedly"):
        await kv_event_loop(adapter, kvcm, coordinator)

    kvcm.send_batch.assert_not_awaited()


async def test_kv_event_loop_resumes_after_dropping_stale_batch(
    adapter: MagicMock, kvcm: MagicMock, coordinator: MagicMock
) -> None:
    """After dropping a stale batch, the loop continues and sends the next one."""
    batch_a = _batch()
    batch_b = _batch()

    async def _subscribe():
        yield batch_a
        yield batch_b

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_epoch.side_effect = [1, 2]
    coordinator.wait_ready_epoch.side_effect = [2, 2]
    coordinator.is_epoch_current.side_effect = [False, True]

    with pytest.raises(RuntimeError, match="kv event subscription ended unexpectedly"):
        await kv_event_loop(adapter, kvcm, coordinator)

    kvcm.send_batch.assert_awaited_once_with(batch_b, 2)


async def test_kv_event_loop_consumes_next_batch_while_kvcm_send_is_slow(
    adapter: MagicMock, kvcm: MagicMock, coordinator: MagicMock
) -> None:
    batch_a = _batch()
    batch_b = _batch()
    second_batch_consumed = asyncio.Event()
    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def _subscribe():
        yield batch_a
        yield batch_b
        second_batch_consumed.set()
        await asyncio.Event().wait()

    async def _send_batch(_batches: list[KVEventBatch], _epoch: int) -> None:
        send_started.set()
        await release_send.wait()

    adapter.subscribe_kv_events = _subscribe
    kvcm.send_batch = AsyncMock(side_effect=_send_batch)
    coordinator.capture_epoch.return_value = 1
    coordinator.wait_ready_epoch.return_value = 1
    coordinator.is_epoch_current.return_value = True

    loop_task = asyncio.create_task(
        kv_event_loop(adapter, kvcm, coordinator, queue_maxsize=2)
    )
    await send_started.wait()
    await asyncio.wait_for(second_batch_consumed.wait(), timeout=1)

    release_send.set()
    loop_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await loop_task

    assert kvcm.send_batch.await_count >= 1


async def test_run_uses_configured_kv_event_queue_maxsize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SubscriberConfig(kv_event_queue_maxsize=7)
    adapter = MagicMock()
    kvcm = MagicMock()
    kvcm.start = AsyncMock()
    kvcm.close = AsyncMock()
    coordinator = MagicMock()
    coordinator.watch_loop = AsyncMock()
    event_loop_called = asyncio.Event()
    watch_loop_started = asyncio.Event()

    async def _kv_event_loop(
        _adapter: MagicMock,
        _kvcm: MagicMock,
        _coordinator: MagicMock,
        queue_maxsize: int = 1024,
    ) -> None:
        assert _adapter is adapter
        assert _kvcm is kvcm
        assert _coordinator is coordinator
        assert queue_maxsize == 7
        kvcm.start.assert_awaited_once()
        event_loop_called.set()

    async def _watch_loop() -> None:
        watch_loop_started.set()
        await asyncio.Event().wait()

    coordinator.watch_loop = _watch_loop
    monkeypatch.setattr(
        "subscriber.main.AbstractEngineAdapter.create", MagicMock(return_value=adapter)
    )
    monkeypatch.setattr("subscriber.main.KvcmClient", MagicMock(return_value=kvcm))
    monkeypatch.setattr(
        "subscriber.main.EngineHealthCoordinator", MagicMock(return_value=coordinator)
    )
    monkeypatch.setattr("subscriber.main.kv_event_loop", _kv_event_loop)

    from subscriber.main import run

    run_task = asyncio.create_task(run(config))
    await asyncio.wait_for(event_loop_called.wait(), timeout=1)
    await asyncio.wait_for(watch_loop_started.wait(), timeout=1)
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run_task

    kvcm.close.assert_awaited_once()


async def test_run_awaits_kvcm_close_when_start_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SubscriberConfig()
    adapter = MagicMock()
    kvcm = MagicMock()
    kvcm.start = AsyncMock(side_effect=RuntimeError("start failed"))
    kvcm.close = AsyncMock()
    coordinator = MagicMock()

    monkeypatch.setattr(
        "subscriber.main.AbstractEngineAdapter.create", MagicMock(return_value=adapter)
    )
    monkeypatch.setattr("subscriber.main.KvcmClient", MagicMock(return_value=kvcm))
    monkeypatch.setattr(
        "subscriber.main.EngineHealthCoordinator", MagicMock(return_value=coordinator)
    )

    from subscriber.main import run

    with pytest.raises(RuntimeError, match="start failed"):
        await run(config)

    kvcm.close.assert_awaited_once()


async def test_kv_event_loop_propagates_sender_exception_and_cancels_producer(
    adapter: MagicMock, kvcm: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch()
    producer_closed = asyncio.Event()

    async def _subscribe():
        try:
            while True:
                yield batch
                await asyncio.sleep(0)
        finally:
            producer_closed.set()

    adapter.subscribe_kv_events = _subscribe
    kvcm.send_batch.side_effect = RuntimeError("kvcm send failed")

    with pytest.raises(RuntimeError, match="kvcm send failed"):
        await asyncio.wait_for(
            kv_event_loop(adapter, kvcm, coordinator, queue_maxsize=1),
            timeout=0.5,
        )

    assert producer_closed.is_set()


async def test_kv_event_loop_rejects_invalid_queue_maxsize(
    adapter: MagicMock, kvcm: MagicMock, coordinator: MagicMock
) -> None:
    with pytest.raises(ValueError, match="queue_maxsize must be >= 1"):
        await kv_event_loop(adapter, kvcm, coordinator, queue_maxsize=0)
