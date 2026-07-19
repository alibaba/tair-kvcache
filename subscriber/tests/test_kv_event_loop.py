from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from subscriber.config import SubscriberConfig
from subscriber.engine.base import EngineEventBatch
from subscriber.main import (
    QueuedKVEventBatch,
    consume_kv_events,
    kv_event_loop,
    send_kv_events,
)
from subscriber.metrics import StageTimer
from subscriber.types import AllBlocksCleared, BlockRemoved, BlockStored, KVEventBatch


def _batch() -> list[KVEventBatch]:
    return [KVEventBatch(ts=1.0, events=[AllBlocksCleared()])]


def _batch_with_block_hashes() -> list[KVEventBatch]:
    return [
        KVEventBatch(
            ts=1.0,
            events=[
                BlockStored(
                    block_hashes=[1, 2],
                    parent_block_hash=None,
                    token_ids=[10, 11],
                    block_size=16,
                    lora_id=None,
                    medium="GPU",
                    lora_name=None,
                ),
                BlockRemoved(block_hashes=[3], medium="GPU"),
            ],
        ),
        KVEventBatch(
            ts=2.0,
            events=[
                BlockStored(
                    block_hashes=[4],
                    parent_block_hash=None,
                    token_ids=[12],
                    block_size=16,
                    lora_id=None,
                    medium="GPU",
                    lora_name=None,
                ),
                BlockRemoved(block_hashes=[5, 6], medium="GPU"),
            ],
        ),
    ]


def _event(batches: list[KVEventBatch]) -> EngineEventBatch:
    return EngineEventBatch(batches, StageTimer())


def test_queued_kv_event_batch_requires_timer() -> None:
    with pytest.raises(TypeError):
        QueuedKVEventBatch(_batch(), 1)


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
    mock.capture_event_epoch = MagicMock(return_value=1)
    mock.is_epoch_current = MagicMock(return_value=True)
    return mock


async def test_consume_kv_events_queues_batch_with_epoch_snapshot(
    adapter: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch()
    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue(maxsize=1)

    async def _subscribe():
        yield _event(batch)

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_event_epoch.return_value = 3

    await consume_kv_events(adapter, coordinator, queue)

    queued = queue.get_nowait()
    assert queued.batches == batch
    assert queued.epoch_snapshot == 3


async def test_consume_buffers_snapshot_batch_when_gate_is_closed(
    adapter: MagicMock, coordinator: MagicMock
) -> None:
    delivery = AsyncMock()
    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue(maxsize=1)

    async def _subscribe():
        yield EngineEventBatch(_batch(), StageTimer(), delivery)

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_event_epoch.return_value = 1

    await consume_kv_events(adapter, coordinator, queue)

    queued = queue.get_nowait()
    assert queued.epoch_snapshot == 1
    assert queued.on_delivery is delivery
    delivery.assert_not_awaited()


async def test_send_kv_events_sends_batch_when_epoch_unchanged(
    kvcm: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch()
    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue()
    await queue.put(QueuedKVEventBatch(batch, 1, StageTimer()))

    sender = asyncio.create_task(send_kv_events(kvcm, coordinator, queue, MagicMock()))
    await asyncio.sleep(0)
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    kvcm.send_batch.assert_awaited_once_with(batch, 1)
    assert queue.empty()


async def test_send_notifies_snapshot_adapter_after_kvcm_ack(
    kvcm: MagicMock, coordinator: MagicMock
) -> None:
    delivery = AsyncMock()
    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue()
    await queue.put(QueuedKVEventBatch(_batch(), 1, StageTimer(), delivery))

    sender = asyncio.create_task(send_kv_events(kvcm, coordinator, queue, MagicMock()))
    await asyncio.sleep(0)
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    delivery.assert_awaited_once_with(True)


async def test_send_kv_events_reports_successful_batch_latency(
    kvcm: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch_with_block_hashes()
    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue()
    reporter = MagicMock()
    await queue.put(QueuedKVEventBatch(batch, 1, StageTimer()))

    sender = asyncio.create_task(
        send_kv_events(kvcm, coordinator, queue, latency_reporter=reporter)
    )
    await asyncio.sleep(0)
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    reporter.report.assert_called_once()
    sample = reporter.report.call_args.args[0]
    spans = sample.spans
    assert [span.name for span in spans] == ["queue_wait", "gate_wait", "kvcm_send"]
    assert all(span.duration_s >= 0 for span in spans)
    assert sample.counters == {
        "stored_block_hash_count": 3,
        "removed_block_hash_count": 3,
    }


async def test_send_kv_events_drops_batch_when_epoch_changed(
    kvcm: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch_with_block_hashes()
    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue()
    reporter = MagicMock()
    await queue.put(QueuedKVEventBatch(batch, 1, StageTimer()))
    coordinator.wait_ready_epoch.return_value = 2
    coordinator.is_epoch_current.return_value = False

    sender = asyncio.create_task(send_kv_events(kvcm, coordinator, queue, reporter))
    await asyncio.sleep(0)
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    kvcm.send_batch.assert_not_awaited()
    reporter.report.assert_not_called()
    assert queue.empty()


async def test_send_retries_same_batch_and_notifies_after_kvcm_recovers(
    kvcm: MagicMock, coordinator: MagicMock
) -> None:
    delivery = AsyncMock()
    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue()
    await queue.put(QueuedKVEventBatch(_batch(), 1, StageTimer(), delivery))
    kvcm.send_batch.side_effect = [RuntimeError("send failed"), None]

    sender = asyncio.create_task(
        send_kv_events(
            kvcm,
            coordinator,
            queue,
            MagicMock(),
            retry_interval_s=0.001,
        )
    )
    await asyncio.wait_for(queue.join(), timeout=1)
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    assert kvcm.send_batch.await_count == 2
    delivery.assert_awaited_once_with(True)


async def test_send_stops_retrying_when_engine_epoch_changes(
    kvcm: MagicMock, coordinator: MagicMock
) -> None:
    delivery = AsyncMock()
    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue()
    await queue.put(QueuedKVEventBatch(_batch(), 1, StageTimer(), delivery))
    kvcm.send_batch.side_effect = RuntimeError("send failed")
    coordinator.wait_ready_epoch.side_effect = [1, 2]
    coordinator.is_epoch_current.side_effect = [True, False]

    sender = asyncio.create_task(
        send_kv_events(
            kvcm,
            coordinator,
            queue,
            MagicMock(),
            retry_interval_s=0.001,
        )
    )
    await asyncio.wait_for(queue.join(), timeout=1)
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    kvcm.send_batch.assert_awaited_once()
    delivery.assert_awaited_once_with(False)


async def test_send_kv_events_retries_in_order_before_next_batch(
    kvcm: MagicMock, coordinator: MagicMock, mocker
) -> None:
    first_batch = _batch_with_block_hashes()
    second_batch = _batch()
    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue()
    reporter = MagicMock()
    await queue.put(QueuedKVEventBatch(first_batch, 1, StageTimer()))
    await queue.put(QueuedKVEventBatch(second_batch, 1, StageTimer()))
    error_message = (
        "KVCM /api/reportEvent failed: INTERNAL_ERROR "
        "ReportEvent partially failed; item_results=['OK', 'INTERNAL_ERROR']"
    )
    kvcm.send_batch.side_effect = [RuntimeError(error_message), None, None]
    warning = mocker.patch("subscriber.main.logger.warning")

    sender = asyncio.create_task(
        send_kv_events(
            kvcm,
            coordinator,
            queue,
            reporter,
            retry_interval_s=0.001,
        )
    )
    await asyncio.wait_for(queue.join(), timeout=1)
    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender

    assert [call.args[0] for call in kvcm.send_batch.await_args_list] == [
        first_batch,
        first_batch,
        second_batch,
    ]
    assert reporter.report.call_count == 2
    warning.assert_called_once_with(
        "failed to send kv event batch to kvcm; retrying in order",
        step="kvcm_send",
        tags={
            "epoch": 1,
            "batch_count": 2,
            "event_count": 4,
            "retry_attempt": 1,
            "retry_interval_s": 0.001,
            "error": "RuntimeError",
            "message": error_message,
        },
        exc_info=True,
    )


async def test_kv_event_loop_sends_batch_when_epoch_unchanged(
    adapter: MagicMock, kvcm: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch()

    async def _subscribe():
        yield _event(batch)

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_event_epoch.return_value = 1
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
        yield _event(batch)

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_event_epoch.return_value = 1
    coordinator.wait_ready_epoch.return_value = 2
    coordinator.is_epoch_current.return_value = False

    with pytest.raises(RuntimeError, match="kv event subscription ended unexpectedly"):
        await kv_event_loop(adapter, kvcm, coordinator)

    kvcm.send_batch.assert_not_awaited()


async def test_kv_event_loop_buffers_batch_until_first_ready_epoch(
    adapter: MagicMock, kvcm: MagicMock, coordinator: MagicMock
) -> None:
    """Batch captured while gate is closed (cold start before first healthy)."""
    batch = _batch()

    async def _subscribe():
        yield _event(batch)

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_event_epoch.return_value = 1
    coordinator.wait_ready_epoch.return_value = 1
    coordinator.is_epoch_current.return_value = True

    with pytest.raises(RuntimeError, match="kv event subscription ended unexpectedly"):
        await kv_event_loop(adapter, kvcm, coordinator)

    kvcm.send_batch.assert_awaited_once_with(batch, 1)


async def test_kv_event_loop_resumes_after_dropping_stale_batch(
    adapter: MagicMock, kvcm: MagicMock, coordinator: MagicMock
) -> None:
    """After dropping a stale batch, the loop continues and sends the next one."""
    batch_a = _batch()
    batch_b = _batch()

    async def _subscribe():
        yield _event(batch_a)
        yield _event(batch_b)

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_event_epoch.side_effect = [1, 2]
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
        yield _event(batch_a)
        yield _event(batch_b)
        second_batch_consumed.set()
        await asyncio.Event().wait()

    async def _send_batch(_batches: list[KVEventBatch], _epoch: int) -> None:
        send_started.set()
        await release_send.wait()

    adapter.subscribe_kv_events = _subscribe
    kvcm.send_batch = AsyncMock(side_effect=_send_batch)
    coordinator.capture_event_epoch.return_value = 1
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
        retry_interval_s: float = 1.0,
    ) -> None:
        assert _adapter is adapter
        assert _kvcm is kvcm
        assert _coordinator is coordinator
        assert queue_maxsize == 7
        assert retry_interval_s == config.kvcm_send_retry_interval_s
        kvcm.start.assert_awaited_once()
        event_loop_called.set()
        await asyncio.Event().wait()

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


async def test_run_cancels_event_loop_when_health_loop_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SubscriberConfig()
    adapter = MagicMock()
    kvcm = MagicMock(start=AsyncMock(), close=AsyncMock())
    coordinator = MagicMock()
    event_started = asyncio.Event()
    event_cancelled = asyncio.Event()

    async def _kv_event_loop(*_args, **_kwargs) -> None:
        event_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            event_cancelled.set()

    async def _watch_loop() -> None:
        await event_started.wait()

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

    with pytest.raises(RuntimeError, match="engine-health-loop ended unexpectedly"):
        await run(config)

    assert event_cancelled.is_set()
    kvcm.close.assert_awaited_once()


async def test_run_cancels_health_loop_when_event_loop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SubscriberConfig()
    adapter = MagicMock()
    kvcm = MagicMock(start=AsyncMock(), close=AsyncMock())
    coordinator = MagicMock()
    health_started = asyncio.Event()
    health_cancelled = asyncio.Event()

    async def _kv_event_loop(*_args, **_kwargs) -> None:
        await health_started.wait()
        raise RuntimeError("event loop failed")

    async def _watch_loop() -> None:
        health_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            health_cancelled.set()

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

    with pytest.raises(RuntimeError, match="event loop failed"):
        await run(config)

    assert health_cancelled.is_set()
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


async def test_kv_event_loop_survives_sender_exception_until_cancelled(
    adapter: MagicMock, kvcm: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch()
    producer_closed = asyncio.Event()
    second_send_finished = asyncio.Event()

    async def _subscribe():
        try:
            while True:
                yield _event(batch)
                await asyncio.sleep(0)
        finally:
            producer_closed.set()

    async def _send_batch(_batches: list[KVEventBatch], _epoch: int) -> None:
        if kvcm.send_batch.await_count == 1:
            raise RuntimeError("kvcm send failed")
        second_send_finished.set()

    adapter.subscribe_kv_events = _subscribe
    kvcm.send_batch.side_effect = _send_batch

    loop_task = asyncio.create_task(
        kv_event_loop(
            adapter,
            kvcm,
            coordinator,
            queue_maxsize=1,
            retry_interval_s=0.001,
        )
    )
    await asyncio.wait_for(second_send_finished.wait(), timeout=0.5)
    assert not loop_task.done()
    loop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop_task

    assert producer_closed.is_set()


async def test_kv_event_loop_rejects_invalid_queue_maxsize(
    adapter: MagicMock, kvcm: MagicMock, coordinator: MagicMock
) -> None:
    with pytest.raises(ValueError, match="queue_maxsize must be >= 1"):
        await kv_event_loop(adapter, kvcm, coordinator, queue_maxsize=0)


async def test_kv_event_loop_rejects_invalid_retry_interval(
    adapter: MagicMock, kvcm: MagicMock, coordinator: MagicMock
) -> None:
    with pytest.raises(ValueError, match="retry_interval_s must be > 0"):
        await kv_event_loop(adapter, kvcm, coordinator, retry_interval_s=0)
