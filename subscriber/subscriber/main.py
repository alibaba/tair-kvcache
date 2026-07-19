from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
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
    on_delivery: Callable[[bool], Awaitable[None]] | None = None


async def _notify_delivery(
    callback: Callable[[bool], Awaitable[None]] | None,
    delivered: bool,
) -> None:
    """Best-effort delivery feedback for snapshot adapters with acked baselines."""

    if callback is None:
        return
    try:
        await callback(delivered)
    except Exception as exc:
        logger.warning(
            "engine adapter delivery callback failed",
            step="kv_event_loop",
            tags={
                "delivered": delivered,
                "error": exc.__class__.__name__,
                "message": str(exc),
            },
            exc_info=True,
        )


async def consume_kv_events(
    adapter: AbstractEngineAdapter,
    coordinator: EngineHealthCoordinator,
    queue: asyncio.Queue[QueuedKVEventBatch],
) -> None:
    """Enqueue batches with their engine generation for ordered delivery."""

    events = adapter.subscribe_kv_events()
    try:
        async for event in events:
            epoch_snapshot = coordinator.capture_event_epoch()
            await queue.put(
                QueuedKVEventBatch(
                    event.batches,
                    epoch_snapshot,
                    event.timer,
                    event.on_delivery,
                )
            )
    finally:
        await events.aclose()


async def send_kv_events(
    kvcm: KvcmClient,
    coordinator: EngineHealthCoordinator,
    queue: asyncio.Queue[QueuedKVEventBatch],
    latency_reporter: SpanMetricsReporter,
    retry_interval_s: float = 1.0,
) -> None:
    """Send queued batches in order, retrying while their epoch stays current."""

    if retry_interval_s <= 0:
        raise ValueError("retry_interval_s must be > 0")

    while True:
        queued = await queue.get()
        try:
            queued.timer.mark("queue_wait")
            retry_attempt = 0
            delivered = False
            while True:
                epoch = await coordinator.wait_ready_epoch()
                if retry_attempt == 0:
                    queued.timer.mark("gate_wait")
                if not coordinator.is_epoch_current(queued.epoch_snapshot):
                    logger.warning(
                        "dropping kv event batch because engine epoch changed "
                        "before send",
                        step="kv_event_loop",
                        tags={
                            "captured_epoch": queued.epoch_snapshot,
                            "current_epoch": epoch,
                        },
                    )
                    await _notify_delivery(queued.on_delivery, False)
                    break
                try:
                    await kvcm.send_batch(queued.batches, epoch)
                except Exception as exc:
                    retry_attempt += 1
                    logger.warning(
                        "failed to send kv event batch to kvcm; retrying in order",
                        step="kvcm_send",
                        tags={
                            "epoch": epoch,
                            "batch_count": len(queued.batches),
                            "event_count": sum(
                                len(batch.events) for batch in queued.batches
                            ),
                            "retry_attempt": retry_attempt,
                            "retry_interval_s": retry_interval_s,
                            "error": exc.__class__.__name__,
                            "message": str(exc),
                        },
                        exc_info=True,
                    )
                    await asyncio.sleep(retry_interval_s)
                    continue
                delivered = True
                break
            if not delivered:
                continue
            await _notify_delivery(queued.on_delivery, True)
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
    retry_interval_s: float = 1.0,
) -> None:
    """Forward engine KV batches to kvcm through a bounded producer/sender queue."""

    if queue_maxsize < 1:
        raise ValueError("queue_maxsize must be >= 1")
    if retry_interval_s <= 0:
        raise ValueError("retry_interval_s must be > 0")

    queue: asyncio.Queue[QueuedKVEventBatch] = asyncio.Queue(maxsize=queue_maxsize)
    latency_reporter = SpanMetricsReporter()
    await latency_reporter.start()
    producer = asyncio.create_task(consume_kv_events(adapter, coordinator, queue))
    sender = asyncio.create_task(
        send_kv_events(
            kvcm,
            coordinator,
            queue,
            latency_reporter,
            retry_interval_s,
        )
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
        location_spec_namer=adapter.location_spec_name,
        location_uri_builder=adapter.location_uri,
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
                "kvcm_send_retry_interval_s": config.kvcm_send_retry_interval_s,
                "engine_health_url": config.engine_health_url,
                "engine_health_interval_s": config.engine_health_interval_s,
                "engine_health_timeout_s": config.engine_health_timeout_s,
                "engine_health_failure_threshold": (
                    config.engine_health_failure_threshold
                ),
                "rtp_endpoints": (
                    config.rtp_endpoints if config.engine_type == "rtp_llm" else ""
                ),
            },
        )

        event_task = asyncio.create_task(
            kv_event_loop(
                adapter,
                kvcm,
                coordinator,
                config.kv_event_queue_maxsize,
                config.kvcm_send_retry_interval_s,
            ),
            name="kv-event-loop",
        )
        health_task = asyncio.create_task(
            coordinator.watch_loop(),
            name="engine-health-loop",
        )
        tasks = {event_task, health_task}
        try:
            done, _ = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            failed = next(
                (
                    task
                    for task in done
                    if not task.cancelled() and task.exception() is not None
                ),
                None,
            )
            if failed is not None:
                await failed
            completed = next(iter(done))
            raise RuntimeError(f"{completed.get_name()} ended unexpectedly")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await kvcm.close()
