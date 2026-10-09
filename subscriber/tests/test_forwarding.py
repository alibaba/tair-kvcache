from __future__ import annotations

import asyncio
from unittest.mock import ANY, AsyncMock, MagicMock, call

import pytest

from subscriber.config import SubscriberConfig
from subscriber.engine.base import EngineEventBatch
from subscriber.engine.metadata import (
    EventTransport,
    KvEventBootstrap,
    RuntimeTopology,
    SnapshotCapability,
    VllmEventSchema,
)
from subscriber.kvcm.errors import (
    KvcmReportRejectedError,
    KvcmUnavailableError,
)
from subscriber.main import (
    consume_incremental_events,
    consume_snapshot_events,
    send_incremental_events,
    send_snapshot_events,
)
from subscriber.metrics import BatchTelemetry
from subscriber.pipeline.context import PipelineContext
from subscriber.trace import generate_trace_id
from subscriber.types import (
    AllBlocksCleared,
    BlockRemoved,
    BlockSnapshot,
    BlockSnapshotItem,
    BlockStored,
    KVEventBatch,
)


def _batch() -> list[KVEventBatch]:
    return [KVEventBatch(ts=1.0, events=[AllBlocksCleared()])]


def _empty_bootstrap() -> KvEventBootstrap:
    return KvEventBootstrap(
        protocol_version=1,
        engine_kind="vllm",
        event_transport=EventTransport(
            live_endpoint="tcp://127.0.0.1:5557",
            topic=b"",
            replay_supported=True,
            replay_endpoint="tcp://127.0.0.1:5558",
            serialization="msgpack-v1",
        ),
        runtime_topology=RuntimeTopology(1, 1, 1, 0, 0, 0),
        snapshot=SnapshotCapability(True, True),
        components=(),
        compatibility_settings=(),
        diagnostic_settings=(),
        vllm=VllmEventSchema(2, False, "none", "sha256", "v1"),
    )


def _batch_with_report_event_count(
    ts: float, report_event_count: int
) -> list[KVEventBatch]:
    return [
        KVEventBatch(
            ts=ts,
            events=[AllBlocksCleared() for _ in range(report_event_count)],
        )
    ]


def _batch_with_zero_report_events(ts: float) -> list[KVEventBatch]:
    return [
        KVEventBatch(
            ts=ts,
            events=[
                BlockStored(
                    block_hashes=[],
                    parent_block_hash=None,
                    token_ids=[],
                    block_size=16,
                    lora_id=None,
                    medium="GPU",
                    lora_name=None,
                )
            ],
        )
    ]


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
    return EngineEventBatch(
        batches, BatchTelemetry(pipeline="incremental"), trace_id=generate_trace_id()
    )


def _queued(
    batches: list,
    epoch: int,
    telemetry: BatchTelemetry,
    reporter: object | None = None,
    trace_id: str | None = None,
) -> PipelineContext:
    return PipelineContext(
        event=EngineEventBatch(
            batches=batches,
            telemetry=telemetry,
            trace_id=trace_id if trace_id is not None else generate_trace_id(),
        ),
        epoch_snapshot=epoch,
        reporter=reporter if reporter is not None else MagicMock(),
    )


def test_main_reexports_forwarding_public_api() -> None:
    from subscriber import forwarding
    from subscriber.pipeline.context import PipelineContext as ContextPipelineContext

    assert PipelineContext is ContextPipelineContext
    assert consume_incremental_events is forwarding.consume_incremental_events
    assert send_incremental_events is forwarding.send_incremental_events
    assert send_snapshot_events is forwarding.send_snapshot_events


def test_pipeline_context_requires_event() -> None:
    with pytest.raises(TypeError):
        PipelineContext()


@pytest.fixture
def adapter() -> MagicMock:
    mock = MagicMock()

    async def _subscribe():
        return
        yield  # pragma: no cover

    mock.subscribe_kv_events = _subscribe
    mock.subscribe_snapshot_events = _subscribe
    return mock


@pytest.fixture
def kvcm() -> MagicMock:
    mock = MagicMock()
    mock.report_kv_events = AsyncMock()
    mock.report_snapshot = AsyncMock()
    return mock


@pytest.fixture
def coordinator() -> MagicMock:
    mock = MagicMock()
    mock.wait_ready_epoch = AsyncMock(return_value=1)
    mock.capture_epoch = MagicMock(return_value=1)
    mock.is_epoch_current = MagicMock(return_value=True)
    return mock


async def test_consume_incremental_events_queues_batch_with_epoch_snapshot(
    adapter: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch()
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue(maxsize=1)

    async def _subscribe():
        yield _event(batch)

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_epoch.return_value = 3

    await consume_incremental_events(adapter, coordinator, queue, MagicMock())

    queued = queue.get_nowait()
    assert queued.batches == batch
    assert queued.epoch_snapshot == 3


async def test_incremental_and_snapshot_consumers_share_not_ready_log_step(
    adapter: MagicMock,
    coordinator: MagicMock,
    mocker,
) -> None:
    async def _subscribe_incremental():
        yield EngineEventBatch(
            _batch(), BatchTelemetry(pipeline="incremental"), "incremental-trace"
        )

    async def _subscribe_snapshot():
        yield EngineEventBatch(
            _batch(), BatchTelemetry(pipeline="snapshot"), "snapshot-trace"
        )

    adapter.subscribe_kv_events = _subscribe_incremental
    adapter.subscribe_snapshot_events = _subscribe_snapshot
    coordinator.capture_epoch.return_value = None
    warning = mocker.patch("subscriber.forwarding.logger.warning")

    await consume_incremental_events(adapter, coordinator, asyncio.Queue(), MagicMock())
    await consume_snapshot_events(adapter, coordinator, asyncio.Queue(), MagicMock())

    assert warning.call_args_list == [
        call(
            "dropping pipeline batch captured while engine is not ready",
            step="engine_event_consume",
            tags={"pipeline": "incremental", "trace_id": "incremental-trace"},
        ),
        call(
            "dropping pipeline batch captured while engine is not ready",
            step="engine_event_consume",
            tags={"pipeline": "snapshot", "trace_id": "snapshot-trace"},
        ),
    ]


async def test_consume_incremental_events_marks_time_after_queue_admission(
    adapter: MagicMock, coordinator: MagicMock
) -> None:
    timer = BatchTelemetry(pipeline="incremental")
    batch = _batch()
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue(maxsize=1)
    await queue.put(_queued(_batch(), 1, BatchTelemetry(pipeline="incremental")))

    async def _subscribe():
        yield EngineEventBatch(batch, timer, trace_id=generate_trace_id())

    adapter.subscribe_kv_events = _subscribe
    consumer = asyncio.create_task(
        consume_incremental_events(adapter, coordinator, queue, MagicMock())
    )
    await asyncio.sleep(0)

    assert timer.elapsed_since_checkpoint("queue_enqueued") is None
    queue.get_nowait()
    queue.task_done()
    await consumer

    assert timer.elapsed_since_checkpoint("queue_enqueued") is not None


async def test_consume_incremental_events_drops_batch_when_not_ready(
    adapter: MagicMock, coordinator: MagicMock
) -> None:
    batch = _batch()
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue(maxsize=1)

    async def _subscribe():
        yield _event(batch)

    adapter.subscribe_kv_events = _subscribe
    coordinator.capture_epoch.return_value = None

    await consume_incremental_events(adapter, coordinator, queue, MagicMock())

    assert queue.empty()


async def test_send_kv_events_sends_immediately_when_queue_is_empty(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    batch = _batch()
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    await queue.put(_queued(batch, 1, BatchTelemetry(pipeline="incremental")))

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=1,
            max_merged_report_events=1,
        )
    )
    await asyncio.sleep(0)
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    kvcm.report_kv_events.assert_awaited_once_with(
        batch, 1, telemetries=ANY, trace_id=ANY
    )
    assert queue.empty()


async def test_snapshot_signal_callback_failure_keeps_incremental_sender_running(
    kvcm: MagicMock,
    coordinator: MagicMock,
    mocker,
) -> None:
    """A local signal failure must not turn a successful KVCM send into a drop."""

    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    reporter = MagicMock()
    telemetry = BatchTelemetry(pipeline="incremental")
    await queue.put(_queued(_batch(), 1, telemetry, reporter=reporter))
    kvcm.report_kv_events.return_value = True
    signal_error = RuntimeError("snapshot signal failed")
    on_snapshot_required = MagicMock(side_effect=signal_error)
    warning = mocker.patch("subscriber.forwarding.logger.warning")

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            on_snapshot_required=on_snapshot_required,
            max_merged_queue_items=1,
            max_merged_report_events=1,
        )
    )
    try:
        await queue.join()

        on_snapshot_required.assert_called_once_with()
        reporter.submit.assert_called_once_with(telemetry)
        assert not sender.done()
        warning.assert_called_once_with(
            "failed to request immediate snapshot; continuing incremental forwarding",
            step="snapshot_signal",
            tags={
                "pipeline": "incremental",
                "trace_id": ANY,
                "merged_trace_ids": ANY,
                "error": "RuntimeError",
                "message": "snapshot signal failed",
            },
            exc_info=True,
        )
    finally:
        sender.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sender


async def test_send_kv_events_merges_contiguous_batches_from_same_epoch(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    first_batch = _batch()
    second_batch = [KVEventBatch(ts=2.0, events=[AllBlocksCleared()])]
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    await queue.put(_queued(first_batch, 1, BatchTelemetry(pipeline="incremental")))
    await queue.put(_queued(second_batch, 1, BatchTelemetry(pipeline="incremental")))

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=100,
            max_merged_report_events=100,
        )
    )
    await queue.join()
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    kvcm.report_kv_events.assert_awaited_once_with(
        first_batch + second_batch, 1, telemetries=ANY, trace_id=ANY
    )


async def test_send_kv_events_keeps_different_epochs_in_separate_sends(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    first_batch = _batch()
    second_batch = [KVEventBatch(ts=2.0, events=[AllBlocksCleared()])]
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    await queue.put(_queued(first_batch, 1, BatchTelemetry(pipeline="incremental")))
    await queue.put(_queued(second_batch, 2, BatchTelemetry(pipeline="incremental")))

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=1,
            max_merged_report_events=1,
        )
    )
    await queue.join()
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    assert kvcm.report_kv_events.await_args_list == [
        ((first_batch, 1), {"telemetries": ANY, "trace_id": ANY}),
        ((second_batch, 1), {"telemetries": ANY, "trace_id": ANY}),
    ]


async def test_send_kv_events_stops_merge_at_report_event_limit(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    batches = [_batch_with_report_event_count(float(index), 8) for index in range(3)]
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    for batch in batches:
        await queue.put(_queued(batch, 1, BatchTelemetry(pipeline="incremental")))

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=4,
            max_merged_report_events=16,
        )
    )
    await queue.join()
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    assert kvcm.report_kv_events.await_args_list == [
        ((batches[0] + batches[1], 1), {"telemetries": ANY, "trace_id": ANY}),
        ((batches[2], 1), {"telemetries": ANY, "trace_id": ANY}),
    ]


async def test_send_kv_events_uses_configured_report_event_limit(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    batches = [_batch_with_report_event_count(float(index), 5) for index in range(3)]
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    for batch in batches:
        await queue.put(_queued(batch, 1, BatchTelemetry(pipeline="incremental")))

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_report_events=10,
            max_merged_queue_items=100,
        )
    )
    await queue.join()
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    assert kvcm.report_kv_events.await_args_list == [
        ((batches[0] + batches[1], 1), {"telemetries": ANY, "trace_id": ANY}),
        ((batches[2], 1), {"telemetries": ANY, "trace_id": ANY}),
    ]


async def test_send_kv_events_stops_merge_at_default_queue_item_limit(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    batches = [_batch_with_report_event_count(float(index), 1) for index in range(5)]
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    for batch in batches:
        await queue.put(_queued(batch, 1, BatchTelemetry(pipeline="incremental")))

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=4,
            max_merged_report_events=16,
        )
    )
    await queue.join()
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    assert kvcm.report_kv_events.await_args_list == [
        (
            (batches[0] + batches[1] + batches[2] + batches[3], 1),
            {"telemetries": ANY, "trace_id": ANY},
        ),
        ((batches[4], 1), {"telemetries": ANY, "trace_id": ANY}),
    ]


async def test_send_kv_events_uses_configured_queue_item_limit(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    batches = [_batch_with_report_event_count(float(index), 1) for index in range(5)]
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    for batch in batches:
        await queue.put(_queued(batch, 1, BatchTelemetry(pipeline="incremental")))

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=2,
            max_merged_report_events=100,
        )
    )
    await queue.join()
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    assert kvcm.report_kv_events.await_args_list == [
        ((batches[0] + batches[1], 1), {"telemetries": ANY, "trace_id": ANY}),
        ((batches[2] + batches[3], 1), {"telemetries": ANY, "trace_id": ANY}),
        ((batches[4], 1), {"telemetries": ANY, "trace_id": ANY}),
    ]


async def test_send_kv_events_continues_when_diagnostics_logging_fails(
    kvcm: MagicMock,
    coordinator: MagicMock,
    mocker,
) -> None:
    batch = _batch()
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    await queue.put(
        _queued(
            batch,
            1,
            BatchTelemetry(pipeline="incremental"),
            trace_id="diagnostics-trace",
        )
    )
    mocker.patch(
        "subscriber.forwarding.log_merge_diagnostics",
        side_effect=RuntimeError("diagnostics failed"),
    )
    warning = mocker.patch("subscriber.forwarding.logger.warning")

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=1,
            max_merged_report_events=1,
        )
    )
    await queue.join()
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    kvcm.report_kv_events.assert_awaited_once_with(
        batch, 1, telemetries=ANY, trace_id=ANY
    )
    warning.assert_called_once_with(
        "failed to log kv event batch diagnostics; continuing with send",
        step="kvcm_send",
        tags={
            "pipeline": "incremental",
            "trace_id": "diagnostics-trace",
            "merged_trace_ids": ["diagnostics-trace"],
            "error": "RuntimeError",
            "message": "diagnostics failed",
        },
        exc_info=True,
    )


async def test_send_kv_events_does_not_limit_source_batch_count(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    batches = [
        batch
        for index in range(65)
        for batch in _batch_with_zero_report_events(float(index))
    ]
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    await queue.put(_queued(batches, 1, BatchTelemetry(pipeline="incremental")))

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=1,
            max_merged_report_events=1,
        )
    )
    await queue.join()
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    assert kvcm.report_kv_events.await_args_list == [
        ((batches, 1), {"telemetries": ANY, "trace_id": ANY}),
    ]


async def test_send_kv_events_logs_enqueue_to_send_and_merge_diagnostics(
    kvcm: MagicMock,
    coordinator: MagicMock,
    mocker,
) -> None:
    first_batch = _batch()
    second_batch = [KVEventBatch(ts=2.0, events=[AllBlocksCleared()])]
    first_timer = BatchTelemetry(pipeline="incremental", clock=lambda: 10.0)
    second_timer = BatchTelemetry(pipeline="incremental", clock=lambda: 10.25)
    first_timer.checkpoint("queue_enqueued")
    second_timer.checkpoint("queue_enqueued")
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    first_context = _queued(first_batch, 1, first_timer, trace_id="first-trace")
    second_context = _queued(second_batch, 1, second_timer, trace_id="second-trace")
    await queue.put(first_context)
    await queue.put(second_context)
    debug = mocker.patch("subscriber.main.logger.debug")
    mocker.patch("subscriber.main.logger.is_debug_enabled", return_value=True)
    mocker.patch("subscriber.forwarding.time.monotonic", return_value=10.5)

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=4,
            max_merged_report_events=16,
        )
    )
    await queue.join()
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    debug.assert_called_once_with(
        "sending merged kv event batches to kvcm",
        step="kvcm_send",
        tags={
            "pipeline": "incremental",
            "trace_id": "first-trace",
            "merged_trace_ids": ["first-trace", "second-trace"],
            "queue_qsize_before_merge": 1,
            "queue_qsize_after_merge": 0,
            "merged_queue_item_count": 2,
            "source_batch_count": 2,
            "source_event_count": 2,
            "merged_report_event_count": 2,
            "oldest_enqueue_to_kvcm_send_ms": 500.0,
            "newest_enqueue_to_kvcm_send_ms": 250.0,
        },
    )
    assert second_context.batch_trace_id == "first-trace"


async def test_send_kv_events_records_merged_request_gauges_once(
    kvcm: MagicMock,
    coordinator: MagicMock,
    mocker,
) -> None:
    first_batch = _batch()
    second_batch = [KVEventBatch(ts=2.0, events=[AllBlocksCleared()])]
    first_telemetry = BatchTelemetry(pipeline="incremental", clock=lambda: 10.0)
    second_telemetry = BatchTelemetry(pipeline="incremental", clock=lambda: 10.25)
    first_telemetry.checkpoint("queue_enqueued")
    second_telemetry.checkpoint("queue_enqueued")
    first_reporter = MagicMock()
    second_reporter = MagicMock()
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    await queue.put(_queued(first_batch, 1, first_telemetry, reporter=first_reporter))
    await queue.put(
        _queued(second_batch, 1, second_telemetry, reporter=second_reporter)
    )
    mocker.patch("subscriber.forwarding.time.monotonic", return_value=10.5)

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=4,
            max_merged_report_events=16,
        )
    )
    await queue.join()
    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender

    assert first_telemetry.gauges == {
        "kvcm_merged_queue_item_count": 2,
        "kvcm_source_batch_count": 2,
        "kvcm_source_event_count": 2,
        "kvcm_merged_report_event_count": 2,
        "kvcm_queue_size_before_merge": 1,
        "kvcm_queue_size_after_merge": 0,
        "kvcm_oldest_enqueue_to_send_ms": 500,
        "kvcm_newest_enqueue_to_send_ms": 250,
    }
    assert second_telemetry.gauges == {}
    first_reporter.submit.assert_called_once_with(first_telemetry)
    second_reporter.submit.assert_called_once_with(second_telemetry)


async def test_send_kv_events_logs_pending_batch_dropped_on_cancellation(
    kvcm: MagicMock,
    coordinator: MagicMock,
    mocker,
) -> None:
    first_batch = _batch()
    second_batch = [KVEventBatch(ts=2.0, events=[AllBlocksCleared()])]
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    await queue.put(
        _queued(
            first_batch,
            1,
            BatchTelemetry(pipeline="incremental"),
            trace_id="first-trace",
        )
    )
    await queue.put(
        _queued(
            second_batch,
            2,
            BatchTelemetry(pipeline="incremental"),
            trace_id="pending-trace",
        )
    )
    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def _send_kv_events(
        _batches: list[KVEventBatch], _epoch: int, **kwargs
    ) -> None:
        send_started.set()
        await release_send.wait()

    kvcm.report_kv_events.side_effect = _send_kv_events
    info = mocker.patch("subscriber.main.logger.info")
    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=4,
            max_merged_report_events=16,
        )
    )
    await send_started.wait()

    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender
    await asyncio.wait_for(queue.join(), timeout=0.1)

    info.assert_called_once_with(
        "dropping pending kv event batch because sender stopped",
        step="kv_event_loop",
        tags={
            "pipeline": "incremental",
            "trace_id": "pending-trace",
            "captured_epoch": 2,
            "batch_count": 1,
            "event_count": 1,
        },
    )


async def test_send_kv_events_drops_merged_batch_after_kvcm_failure(
    kvcm: MagicMock,
    coordinator: MagicMock,
    mocker,
) -> None:
    first_batch = _batch()
    second_batch = [KVEventBatch(ts=2.0, events=[AllBlocksCleared()])]
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    await queue.put(
        _queued(
            first_batch,
            1,
            BatchTelemetry(pipeline="incremental"),
            trace_id="first-trace",
        )
    )
    await queue.put(
        _queued(
            second_batch,
            1,
            BatchTelemetry(pipeline="incremental"),
            trace_id="second-trace",
        )
    )
    kvcm.report_kv_events.side_effect = KvcmUnavailableError("request failed")
    warning = mocker.patch("subscriber.main.logger.warning")

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=4,
            max_merged_report_events=16,
        )
    )
    await queue.join()
    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender

    kvcm.report_kv_events.assert_awaited_once_with(
        first_batch + second_batch, 1, telemetries=ANY, trace_id="first-trace"
    )
    warning.assert_called_once_with(
        "failed to send kv event batch to kvcm; dropping batch",
        step="kvcm_send",
        tags={
            "pipeline": "incremental",
            "trace_id": "first-trace",
            "merged_trace_ids": ["first-trace", "second-trace"],
            "epoch": 1,
            "batch_count": 2,
            "event_count": 2,
            "error": "KvcmUnavailableError",
            "message": "request failed",
            "reason": "unknown",
            "dropped_batch_total": 2,
            "dropped_event_total": 2,
        },
    )


async def test_send_kv_events_reports_successful_batch_latency(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    batch = _batch_with_block_hashes()
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    reporter = MagicMock()
    await queue.put(
        _queued(batch, 1, BatchTelemetry(pipeline="incremental"), reporter=reporter)
    )

    async def _mark_stages(_batches, _epoch, telemetries=None, **kw):
        if telemetries:
            for t in telemetries:
                t.mark("expand")
                t.mark("kvcm_send")

    kvcm.report_kv_events.side_effect = _mark_stages

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=1,
            max_merged_report_events=1,
        )
    )
    await asyncio.sleep(0)
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    reporter.submit.assert_called_once()
    telemetry = reporter.submit.call_args.args[0]
    spans = telemetry.spans
    assert [span.name for span in spans] == [
        "queue_wait",
        "engine_gate_wait",
        "block_filter",
        "expand",
        "kvcm_send",
    ]
    assert all(span.duration_s >= 0 for span in spans)
    assert telemetry.counters == {
        "stored_block_hash_count": 3,
        "removed_block_hash_count": 3,
    }
    assert telemetry.drop_reason is None


async def test_send_kv_events_suppresses_removal_with_remaining_copies(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    store = BlockStored(
        block_hashes=[1],
        parent_block_hash=None,
        token_ids=[10],
        block_size=16,
        lora_id=None,
        medium="GPU",
        lora_name=None,
    )
    suppressed_removal = BlockRemoved(
        block_hashes=[1], medium="GPU", remaining_copy_counts=[1]
    )
    final_removal = BlockRemoved(
        block_hashes=[1], medium="GPU", remaining_copy_counts=[0]
    )
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    await queue.put(
        _queued(
            [KVEventBatch(ts=1.0, events=[store, suppressed_removal])],
            1,
            BatchTelemetry(pipeline="incremental"),
        )
    )
    await queue.put(
        _queued(
            [KVEventBatch(ts=2.0, events=[final_removal])],
            2,
            BatchTelemetry(pipeline="incremental"),
        )
    )

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=1,
            max_merged_report_events=1,
        )
    )
    await queue.join()
    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender

    first_sent = [
        event
        for batch in kvcm.report_kv_events.await_args_list[0].args[0]
        for event in batch.events
    ]
    second_sent = [
        event
        for batch in kvcm.report_kv_events.await_args_list[1].args[0]
        for event in batch.events
    ]
    assert first_sent == [store]
    assert second_sent == [final_removal]


async def test_send_kv_events_submits_telemetry_when_filter_empties_batch(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    suppressed_removal = BlockRemoved(
        block_hashes=[1], medium="GPU", remaining_copy_counts=[1]
    )
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    reporter = MagicMock()
    await queue.put(
        _queued(
            [KVEventBatch(ts=1.0, events=[suppressed_removal])],
            1,
            BatchTelemetry(pipeline="incremental"),
            reporter=reporter,
        )
    )

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=1,
            max_merged_report_events=1,
        )
    )
    await queue.join()
    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender

    kvcm.report_kv_events.assert_not_awaited()
    reporter.submit.assert_called_once()
    telemetry = reporter.submit.call_args.args[0]
    assert telemetry.drop_reason == "filtered_empty"
    assert [span.name for span in telemetry.spans] == [
        "queue_wait",
        "engine_gate_wait",
        "block_filter",
    ]


async def test_send_kv_events_drops_batch_when_epoch_changed(
    kvcm: MagicMock,
    coordinator: MagicMock,
    mocker,
) -> None:
    batch = _batch_with_block_hashes()
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    reporter = MagicMock()
    await queue.put(
        _queued(batch, 1, BatchTelemetry(pipeline="incremental"), reporter=reporter)
    )
    coordinator.wait_ready_epoch.return_value = 2
    coordinator.is_epoch_current.return_value = False

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=1,
            max_merged_report_events=1,
            pipeline="snapshot",
        )
    )
    await asyncio.sleep(0)
    sender.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sender

    kvcm.report_kv_events.assert_not_awaited()
    # Dropped batches are submitted to the reporter with drop_reason set;
    # the reporter fans out to the drop-count metric internally.
    reporter.submit.assert_called_once()
    dropped_telemetry = reporter.submit.call_args.args[0]
    assert dropped_telemetry.drop_reason == "epoch_changed"
    assert queue.empty()


def _snapshot_batch_list() -> list[KVEventBatch]:
    return [
        KVEventBatch(
            ts=1.0,
            events=[
                BlockSnapshot(
                    medium="GPU",
                    block_size=16,
                    items=[
                        BlockSnapshotItem(block_hash=101, group_idx=0),
                        BlockSnapshotItem(block_hash=102, group_idx=0),
                    ],
                    snapshot_version=1,
                )
            ],
        )
    ]


async def test_send_snapshot_events_forwards_batch_and_reports_success_telemetry(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    batches = _snapshot_batch_list()
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    reporter = MagicMock()
    telemetry = BatchTelemetry(pipeline="snapshot")
    await queue.put(_queued(batches, 1, telemetry, reporter=reporter))

    async def _fake_report_snapshot(
        _batches: list,
        _epoch: int,
        *,
        telemetry: BatchTelemetry | None = None,
        trace_id: str | None = None,
    ) -> None:
        if telemetry is not None:
            telemetry.mark("expand")
            telemetry.mark("kvcm_send")

    kvcm.report_snapshot = AsyncMock(side_effect=_fake_report_snapshot)

    sender = asyncio.create_task(send_snapshot_events(kvcm, coordinator, queue))
    await asyncio.sleep(0)
    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender

    kvcm.report_snapshot.assert_awaited_once_with(
        batches, 1, telemetry=telemetry, trace_id=ANY
    )
    reporter.submit.assert_called_once()
    submitted = reporter.submit.call_args.args[0]
    assert [span.name for span in submitted.spans] == [
        "queue_wait",
        "engine_gate_wait",
        "expand",
        "kvcm_send",
    ]
    assert submitted.counters == {
        "snapshot_block_count": 2,
    }
    assert telemetry.drop_reason is None


async def test_send_snapshot_events_drops_batch_when_epoch_changed(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    batches = _snapshot_batch_list()
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    reporter = MagicMock()
    await queue.put(
        _queued(batches, 1, BatchTelemetry(pipeline="snapshot"), reporter=reporter)
    )
    coordinator.wait_ready_epoch.return_value = 2
    coordinator.is_epoch_current.return_value = False

    sender = asyncio.create_task(send_snapshot_events(kvcm, coordinator, queue))
    await asyncio.sleep(0)
    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender

    kvcm.report_snapshot.assert_not_awaited()
    reporter.submit.assert_called_once()
    dropped_telemetry = reporter.submit.call_args.args[0]
    assert dropped_telemetry.drop_reason == "epoch_changed"
    assert queue.empty()


async def test_send_snapshot_events_drops_batch_on_kvcm_failure_and_continues(
    kvcm: MagicMock,
    coordinator: MagicMock,
    mocker,
) -> None:
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    await queue.put(
        _queued(
            _snapshot_batch_list(),
            1,
            BatchTelemetry(pipeline="snapshot"),
            trace_id="snapshot-first-trace",
        )
    )
    await queue.put(
        _queued(_snapshot_batch_list(), 1, BatchTelemetry(pipeline="snapshot"))
    )
    kvcm.report_snapshot.side_effect = KvcmUnavailableError("request failed")
    warning = mocker.patch("subscriber.main.logger.warning")

    sender = asyncio.create_task(send_snapshot_events(kvcm, coordinator, queue))
    await queue.join()
    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender

    assert kvcm.report_snapshot.await_count == 2
    warning.assert_called_once_with(
        "failed to send kv event batch to kvcm; dropping batch",
        step="kvcm_send",
        tags={
            "pipeline": "snapshot",
            "trace_id": "snapshot-first-trace",
            "merged_trace_ids": ["snapshot-first-trace"],
            "epoch": 1,
            "batch_count": 1,
            "event_count": 1,
            "error": "KvcmUnavailableError",
            "message": "request failed",
            "reason": "unknown",
            "dropped_batch_total": 1,
            "dropped_event_total": 1,
        },
    )


async def test_send_kv_events_logs_failure_and_continues_with_next_batch(
    kvcm: MagicMock,
    coordinator: MagicMock,
    mocker,
) -> None:
    first_batch = _batch_with_block_hashes()
    second_batch = _batch()
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    reporter = MagicMock()
    await queue.put(
        _queued(
            first_batch,
            1,
            BatchTelemetry(pipeline="incremental"),
            reporter=reporter,
            trace_id="first-trace",
        )
    )
    await queue.put(
        _queued(
            second_batch,
            2,
            BatchTelemetry(pipeline="incremental"),
            reporter=reporter,
            trace_id="second-trace",
        )
    )
    error_message = (
        "KVCM /api/reportEvent failed: INTERNAL_ERROR "
        "ReportEvent partially failed; item_results=['OK', 'INTERNAL_ERROR']"
    )
    kvcm.report_kv_events.side_effect = [KvcmReportRejectedError(error_message), None]
    warning = mocker.patch("subscriber.main.logger.warning")

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=1,
            max_merged_report_events=1,
        )
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender

    assert kvcm.report_kv_events.await_count == 2
    # The failed first batch is submitted as dropped; the second batch is
    # submitted as a successful send with block-hash counters attached.
    assert reporter.submit.call_count == 2
    dropped_telemetry = reporter.submit.call_args_list[0].args[0]
    success_telemetry = reporter.submit.call_args_list[1].args[0]
    assert dropped_telemetry.drop_reason == "send_failed"
    assert success_telemetry.drop_reason is None
    assert success_telemetry.counters == {
        "stored_block_hash_count": 0,
        "removed_block_hash_count": 0,
    }
    warning.assert_called_once_with(
        "failed to send kv event batch to kvcm; dropping batch",
        step="kvcm_send",
        tags={
            "pipeline": "incremental",
            "trace_id": "first-trace",
            "merged_trace_ids": ["first-trace"],
            "epoch": 1,
            "batch_count": 2,
            "event_count": 4,
            "error": "KvcmReportRejectedError",
            "message": error_message,
            "reason": "unknown",
            "dropped_batch_total": 2,
            "dropped_event_total": 4,
        },
    )


async def test_send_kv_events_marks_metadata_protocol_drop_and_notifies_lifecycle(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    reporter = MagicMock()
    await queue.put(
        _queued(
            _batch(),
            1,
            BatchTelemetry(pipeline="incremental"),
            reporter=reporter,
        )
    )
    kvcm.report_kv_events.side_effect = KvcmReportRejectedError(
        "component identity drift",
        status_code="METADATA_PROTOCOL",
        reason="metadata_protocol",
    )
    report_inactive = AsyncMock()

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=1,
            max_merged_report_events=1,
            on_metadata_protocol_error=report_inactive,
        )
    )
    await queue.join()
    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender

    dropped_telemetry = reporter.submit.call_args.args[0]
    assert dropped_telemetry.drop_reason == "metadata_protocol"
    report_inactive.assert_awaited_once_with()


async def test_send_kv_events_filters_correctly_after_kvcm_failure(
    kvcm: MagicMock,
    coordinator: MagicMock,
) -> None:
    store = BlockStored(
        block_hashes=[1],
        parent_block_hash=None,
        token_ids=[10],
        block_size=16,
        lora_id=None,
        medium="GPU",
        lora_name=None,
    )
    suppressed = BlockRemoved(block_hashes=[1], medium="GPU", remaining_copy_counts=[1])
    final_removal = BlockRemoved(
        block_hashes=[1], medium="GPU", remaining_copy_counts=[0]
    )
    queue: asyncio.Queue[PipelineContext] = asyncio.Queue()
    await queue.put(
        _queued([KVEventBatch(1.0, [store])], 1, BatchTelemetry(pipeline="incremental"))
    )
    await queue.put(
        _queued(
            [KVEventBatch(2.0, [suppressed])], 2, BatchTelemetry(pipeline="incremental")
        )
    )
    await queue.put(
        _queued(
            [KVEventBatch(3.0, [store, final_removal])],
            3,
            BatchTelemetry(pipeline="incremental"),
        )
    )
    # Second send fails — filter state should not matter (stateless).
    kvcm.report_kv_events.side_effect = [
        None,
        KvcmUnavailableError("delete failed"),
        None,
    ]

    sender = asyncio.create_task(
        send_incremental_events(
            kvcm,
            coordinator,
            queue,
            max_merged_queue_items=1,
            max_merged_report_events=1,
        )
    )
    await queue.join()
    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender

    # Suppressed removal batch is never sent to kvcm (empty after filtering).
    # send_kv_events is called for batch 1 (store), then skips batch 2 (suppressed,
    # no events left), then sends batch 3 (store + final_removal).
    assert kvcm.report_kv_events.await_count == 2
    third_sent_events = [
        event
        for batch in kvcm.report_kv_events.await_args_list[1].args[0]
        for event in batch.events
    ]
    assert third_sent_events == [store, final_removal]


async def test_run_uses_configured_queue_and_merge_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SubscriberConfig(
        kv_event_queue_maxsize=7,
        kv_event_merge_max_report_events=11,
        kv_event_merge_max_queue_items=3,
    )
    adapter = MagicMock()
    adapter.fetch_kv_event_bootstrap = AsyncMock(return_value=_empty_bootstrap())
    adapter.close = AsyncMock()
    kvcm = MagicMock()
    kvcm.start = AsyncMock()
    kvcm.close = AsyncMock()
    kvcm.is_registered = True
    coordinator = MagicMock()

    async def _wait_ready_epoch() -> int:
        await asyncio.sleep(0)
        return 1

    coordinator.wait_ready_epoch = _wait_ready_epoch
    coordinator.attach_kvcm_client = MagicMock()
    coordinator.capture_epoch = MagicMock(return_value=1)
    coordinator.report_host_down = AsyncMock()
    coordinator_factory = MagicMock(return_value=coordinator)
    sender_started = asyncio.Event()
    watch_loop_started = asyncio.Event()
    captured_kwargs: dict = {}

    async def _send_incremental_events(
        _kvcm,
        _coordinator,
        _queue,
        max_merged_report_events=16,
        drop_tracker=None,
        max_merged_queue_items=4,
        pipeline="incremental",
        on_snapshot_required=None,
        on_metadata_protocol_error=None,
    ):
        if _queue.maxsize != config.kv_event_queue_maxsize:
            await asyncio.Event().wait()
            return
        captured_kwargs["max_merged_report_events"] = max_merged_report_events
        captured_kwargs["max_merged_queue_items"] = max_merged_queue_items
        captured_kwargs["queue_maxsize"] = _queue.maxsize
        sender_started.set()
        await asyncio.Event().wait()

    async def _consume_incremental_events(*args, **kwargs):
        await asyncio.Event().wait()

    async def _consume_snapshot_events(*args, **kwargs):
        await asyncio.Event().wait()

    async def _watch_loop() -> None:
        watch_loop_started.set()
        await asyncio.Event().wait()

    coordinator.watch_loop = _watch_loop
    monkeypatch.setattr(
        "subscriber.main.AbstractEngineAdapter.create", MagicMock(return_value=adapter)
    )
    monkeypatch.setattr("subscriber.main.KvcmClient", MagicMock(return_value=kvcm))
    monkeypatch.setattr("subscriber.main.EngineHealthCoordinator", coordinator_factory)
    monkeypatch.setattr(
        "subscriber.main.send_incremental_events", _send_incremental_events
    )
    monkeypatch.setattr(
        "subscriber.main.consume_incremental_events", _consume_incremental_events
    )
    monkeypatch.setattr(
        "subscriber.main.consume_snapshot_events", _consume_snapshot_events
    )

    from subscriber.main import run

    run_task = asyncio.create_task(run(config))
    await asyncio.wait_for(sender_started.wait(), timeout=1)
    await asyncio.wait_for(watch_loop_started.wait(), timeout=1)

    assert captured_kwargs["max_merged_report_events"] == 11
    assert captured_kwargs["max_merged_queue_items"] == 3
    assert captured_kwargs["queue_maxsize"] == 7

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    coordinator_factory.assert_called_once_with(adapter, None, config)
    coordinator.attach_kvcm_client.assert_called_once_with(kvcm)
    kvcm.close.assert_awaited_once()


async def test_run_awaits_kvcm_close_when_start_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SubscriberConfig()
    adapter = MagicMock()
    adapter.fetch_kv_event_bootstrap = AsyncMock(return_value=_empty_bootstrap())
    adapter.close = AsyncMock()
    kvcm = MagicMock()
    kvcm.start = AsyncMock(side_effect=RuntimeError("start failed"))
    kvcm.close = AsyncMock()
    coordinator = MagicMock()

    async def _wait_ready_epoch() -> int:
        await asyncio.sleep(0)
        return 1

    coordinator.wait_ready_epoch = _wait_ready_epoch

    async def _watch_loop() -> None:
        await asyncio.Event().wait()

    coordinator.watch_loop = _watch_loop
    coordinator.report_host_down = AsyncMock()

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
    adapter.close.assert_awaited_once()


async def test_run_starts_watch_loop_before_metadata_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """watch_loop must start before fetch_kv_event_bootstrap is called."""
    config = SubscriberConfig()
    call_order: list[str] = []

    adapter = MagicMock()

    async def _fetch_metadata():
        call_order.append("fetch_metadata")
        return _empty_bootstrap()

    adapter.fetch_kv_event_bootstrap = _fetch_metadata
    adapter.close = AsyncMock()

    kvcm = MagicMock()
    kvcm.start = AsyncMock()
    kvcm.close = AsyncMock()
    kvcm.is_registered = True

    coordinator = MagicMock()

    async def _watch_loop():
        call_order.append("watch_loop_started")
        await asyncio.Event().wait()

    coordinator.watch_loop = _watch_loop

    async def _wait_ready_epoch():
        await asyncio.sleep(0)
        call_order.append("wait_ready_epoch")
        return 1

    coordinator.wait_ready_epoch = _wait_ready_epoch

    attach_called = False

    def _attach_kvcm_client(kvcm_ref):
        nonlocal attach_called
        attach_called = True

    coordinator.attach_kvcm_client = _attach_kvcm_client
    coordinator.report_host_down = AsyncMock()

    pipeline_started = asyncio.Event()

    async def _send_kv_events(*args, **kwargs):
        pipeline_started.set()
        await asyncio.Event().wait()

    async def _consume_incremental_events(*args, **kwargs):
        await asyncio.Event().wait()

    async def _consume_snapshot_events(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "subscriber.main.AbstractEngineAdapter.create",
        MagicMock(return_value=adapter),
    )
    kvcm_factory = MagicMock(return_value=kvcm)
    monkeypatch.setattr("subscriber.main.KvcmClient", kvcm_factory)
    monkeypatch.setattr(
        "subscriber.main.EngineHealthCoordinator", MagicMock(return_value=coordinator)
    )
    monkeypatch.setattr("subscriber.main.send_incremental_events", _send_kv_events)
    monkeypatch.setattr(
        "subscriber.main.consume_incremental_events", _consume_incremental_events
    )
    monkeypatch.setattr(
        "subscriber.main.consume_snapshot_events", _consume_snapshot_events
    )

    from subscriber.main import run

    run_task = asyncio.create_task(run(config))
    await asyncio.wait_for(pipeline_started.wait(), timeout=1.0)

    assert call_order[0] == "watch_loop_started"
    assert "fetch_metadata" in call_order
    assert call_order.index("watch_loop_started") < call_order.index("fetch_metadata")
    assert attach_called
    assert kvcm_factory.call_args.kwargs["descriptor"].groups == ()

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    kvcm.close.assert_awaited_once()
    adapter.close.assert_awaited_once()


async def test_run_fatal_metadata_error_reports_failed_and_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fatal metadata error (MetadataProtocolError) reports failed and exits."""
    from subscriber.engine.metadata import MetadataProtocolError

    config = SubscriberConfig()

    adapter = MagicMock()
    adapter.fetch_kv_event_bootstrap = AsyncMock(
        side_effect=MetadataProtocolError("bad schema")
    )
    adapter.close = AsyncMock()

    coordinator = MagicMock()

    async def _watch_loop() -> None:
        await asyncio.Event().wait()

    coordinator.watch_loop = _watch_loop

    async def _wait_ready_epoch() -> int:
        await asyncio.sleep(0)
        return 1

    coordinator.wait_ready_epoch = _wait_ready_epoch
    coordinator.report_host_down = AsyncMock()

    kvcm_factory = MagicMock()

    monkeypatch.setattr(
        "subscriber.main.AbstractEngineAdapter.create",
        MagicMock(return_value=adapter),
    )
    monkeypatch.setattr(
        "subscriber.main.EngineHealthCoordinator", MagicMock(return_value=coordinator)
    )
    monkeypatch.setattr("subscriber.main.KvcmClient", kvcm_factory)

    from subscriber.main import run

    # Fatal metadata error exits cleanly (reports failed internally, no raise).
    await asyncio.wait_for(run(config), timeout=1.0)

    kvcm_factory.assert_not_called()
    adapter.close.assert_awaited_once()


async def test_run_initializes_dashlog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run() entrypoint initializes dashlog metrics before serving."""
    config = SubscriberConfig()
    adapter = MagicMock()
    adapter.fetch_kv_event_bootstrap = AsyncMock(return_value=_empty_bootstrap())
    adapter.close = AsyncMock()
    kvcm = MagicMock()
    kvcm.start = AsyncMock()
    kvcm.close = AsyncMock()
    kvcm.is_registered = True
    coordinator = MagicMock()

    async def _wait_ready_epoch() -> int:
        await asyncio.sleep(0)
        return 1

    coordinator.wait_ready_epoch = _wait_ready_epoch
    coordinator.attach_kvcm_client = MagicMock()
    coordinator.capture_epoch = MagicMock(return_value=1)
    coordinator.report_host_down = AsyncMock()

    async def _watch_loop() -> None:
        await asyncio.Event().wait()

    coordinator.watch_loop = _watch_loop

    init_dashlog_mock = MagicMock()
    sender_started = asyncio.Event()

    async def _send_kv_events(*args, **kwargs):
        sender_started.set()
        await asyncio.Event().wait()

    async def _consume_incremental_events(*args, **kwargs):
        await asyncio.Event().wait()

    async def _consume_snapshot_events(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "subscriber.main.AbstractEngineAdapter.create", MagicMock(return_value=adapter)
    )
    monkeypatch.setattr("subscriber.main.KvcmClient", MagicMock(return_value=kvcm))
    monkeypatch.setattr(
        "subscriber.main.EngineHealthCoordinator", MagicMock(return_value=coordinator)
    )
    monkeypatch.setattr("subscriber.main.init_dashlog", init_dashlog_mock)
    monkeypatch.setattr("subscriber.main.send_incremental_events", _send_kv_events)
    monkeypatch.setattr(
        "subscriber.main.consume_incremental_events", _consume_incremental_events
    )
    monkeypatch.setattr(
        "subscriber.main.consume_snapshot_events", _consume_snapshot_events
    )

    from subscriber.main import run

    run_task = asyncio.create_task(run(config))
    await asyncio.wait_for(sender_started.wait(), timeout=1)
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run_task

    init_dashlog_mock.assert_called_once_with("kvcache-subscriber")
