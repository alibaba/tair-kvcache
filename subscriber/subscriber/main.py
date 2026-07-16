from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass

from subscriber import logger
from subscriber.config import SubscriberConfig
from subscriber.engine.base import AbstractEngineAdapter
from subscriber.health.coordinator import EngineHealthCoordinator
from subscriber.kvcm.client import KvcmClient
from subscriber.metrics import MetricSample, SpanMetricsReporter, StageTimer
from subscriber.types import BlockRemoved, BlockStored, KVEventBatch


@dataclass(frozen=True)
class QueuedKVEventBatch:
    """KV batches captured with the engine epoch that made them sendable."""

    batches: list[KVEventBatch]
    epoch_snapshot: int
    timer: StageTimer


async def consume_kv_events(
    adapter: AbstractEngineAdapter,
    coordinator: EngineHealthCoordinator,
    queue: asyncio.Queue[QueuedKVEventBatch],
) -> None:
    """Enqueue only batches captured while the coordinator has a ready epoch."""

    events = adapter.subscribe_kv_events()
    try:
        async for event in events:
            epoch_snapshot = coordinator.capture_epoch()
            if epoch_snapshot is None:
                logger.warning(
                    "dropping kv event batch captured while engine is not ready",
                    step="kv_event_loop",
                )
                continue
            await queue.put(
                QueuedKVEventBatch(event.batches, epoch_snapshot, event.timer)
            )
    finally:
        await events.aclose()


async def send_kv_events(
    kvcm: KvcmClient,
    coordinator: EngineHealthCoordinator,
    queue: asyncio.Queue[QueuedKVEventBatch],
    latency_reporter: SpanMetricsReporter,
) -> None:
    """Send queued batches only if their captured epoch is still current."""

    while True:
        queued = await queue.get()
        try:
            queued.timer.mark("queue_wait")
            epoch = await coordinator.wait_ready_epoch()
            queued.timer.mark("gate_wait")
            if not coordinator.is_epoch_current(queued.epoch_snapshot):
                logger.warning(
                    "dropping kv event batch because engine epoch changed before send",
                    step="kv_event_loop",
                    tags={
                        "captured_epoch": queued.epoch_snapshot,
                        "current_epoch": epoch,
                    },
                )
                continue
            try:
                await kvcm.send_batch(queued.batches, epoch)
            except Exception as exc:
                # TODO: Buffer or replay batches dropped while KVCM is unavailable.
                logger.warning(
                    "failed to send kv event batch to kvcm; dropping batch",
                    step="kvcm_send",
                    tags={
                        "epoch": epoch,
                        "batch_count": len(queued.batches),
                        "event_count": sum(
                            len(batch.events) for batch in queued.batches
                        ),
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                    },
                    exc_info=True,
                )
                continue
            # Metrics stay outside the send try/except: reporting is best-effort
            # (report never raises) and must never be misread as a send failure.
            stored_block_hash_count = sum(
                len(event.block_hashes)
                for batch in queued.batches
                for event in batch.events
                if isinstance(event, BlockStored)
            )
            removed_block_hash_count = sum(
                len(event.block_hashes)
                for batch in queued.batches
                for event in batch.events
                if isinstance(event, BlockRemoved)
            )
            queued.timer.mark("kvcm_send")
            latency_reporter.report(
                MetricSample(
                    spans=queued.timer.spans(),
                    counters={
                        "stored_block_hash_count": stored_block_hash_count,
                        "removed_block_hash_count": removed_block_hash_count,
                    },
                )
            )
        finally:
            queue.task_done()


async def kv_event_loop(
    adapter: AbstractEngineAdapter,
    kvcm: KvcmClient,
    coordinator: EngineHealthCoordinator,
    queue_maxsize: int = 1024,
) -> None:
    """Forward engine KV batches to kvcm through a bounded producer/sender queue."""

    if queue_maxsize < 1:
        raise ValueError("queue_maxsize must be >= 1")

    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue(maxsize=queue_maxsize)
    latency_reporter = SpanMetricsReporter()
    await latency_reporter.start()
    producer = asyncio.create_task(consume_kv_events(adapter, coordinator, queue))
    sender = asyncio.create_task(
        send_kv_events(kvcm, coordinator, queue, latency_reporter)
    )
    try:
        done, _ = await asyncio.wait(
            {producer, sender}, return_when=asyncio.FIRST_COMPLETED
        )
        producer_exited_first = sender not in done
        if sender in done:
            producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer
            await sender

        await producer
        queue_drained = asyncio.create_task(queue.join())
        done, _ = await asyncio.wait(
            {queue_drained, sender}, return_when=asyncio.FIRST_COMPLETED
        )
        if sender in done:
            queue_drained.cancel()
            with suppress(asyncio.CancelledError):
                await queue_drained
            await sender

        await queue_drained
        raise RuntimeError(
            "kv event subscription ended unexpectedly"
            + (
                "; producer exited before sender"
                if producer_exited_first
                else "; sender outlived producer"
            )
        )
    finally:
        sender.cancel()
        with suppress(asyncio.CancelledError):
            await sender
        producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer
        await latency_reporter.stop()


async def run(config: SubscriberConfig) -> None:
    """Run the subscriber event and liveness loops until cancellation or error."""

    adapter = AbstractEngineAdapter.create(config.engine_type, config)
    kvcm = KvcmClient(
        config,
        medium_mapper=adapter.map_medium,
        storage_type=adapter.storage_type(),
        supported_mediums=adapter.supported_mediums(),
    )
    try:
        await kvcm.start()
        coordinator = EngineHealthCoordinator(adapter, kvcm, config)

        logger.info(
            "subscriber started",
            step="startup",
            tags={
                "engine_type": config.engine_type,
                "zmq_pub_endpoint": config.zmq_pub_endpoint,
                "zmq_replay_endpoint": config.zmq_replay_endpoint,
                "kvcm_heartbeat_interval_s": config.kvcm_heartbeat_interval_s,
                "engine_health_url": config.engine_health_url,
                "engine_health_interval_s": config.engine_health_interval_s,
                "engine_health_timeout_s": config.engine_health_timeout_s,
                "engine_health_failure_threshold": (
                    config.engine_health_failure_threshold
                ),
            },
        )

        await asyncio.gather(
            kv_event_loop(adapter, kvcm, coordinator, config.kv_event_queue_maxsize),
            coordinator.watch_loop(),
        )
    finally:
        await kvcm.close()
