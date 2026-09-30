"""Queue, gate, merge, and forward engine KV events to KVCM."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable

from subscriber import logger
from subscriber.engine.base import AbstractEngineAdapter, EngineEventBatch
from subscriber.health.coordinator import EngineHealthCoordinator
from subscriber.kvcm.client import KvcmClient
from subscriber.kvcm.errors import KvcmReportError
from subscriber.metrics import MetricsReporter
from subscriber.pipeline.block_filter import filter_block_removals
from subscriber.pipeline.context import PipelineContext
from subscriber.pipeline.merge import (
    MergedBatch,
    dequeue_merged_batches,
    log_merge_diagnostics,
)
from subscriber.types import BlockRemoved, BlockSnapshot, BlockStored, KVEventBatch

_QUEUE_FULL_WARN_INTERVAL_S = 30.0


class KvcmDropTracker:
    """Rate-limits warnings for KV event batches dropped on KVCM failures.

    The first failure is logged immediately. Subsequent failures accumulate
    dropped batch/event counts and are summarized every ``summary_every``
    failures or every ``summary_interval_s`` seconds, whichever comes first.
    This keeps a sustained KVCM outage from spamming the log while still
    surfacing the first failure and periodic totals.
    """

    def __init__(
        self,
        *,
        summary_every: int = 100,
        summary_interval_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._summary_every = max(1, summary_every)
        self._summary_interval_s = summary_interval_s
        self._clock = clock
        self._dropped_batch_count = 0
        self._dropped_event_count = 0
        self._dropped_since_last_summary = 0
        self._last_summary_at_s = clock()

    @property
    def dropped_batch_count(self) -> int:
        return self._dropped_batch_count

    @property
    def dropped_event_count(self) -> int:
        return self._dropped_event_count

    def record_drop(
        self,
        *,
        pipeline: str = "incremental",
        epoch: int,
        batch_count: int,
        event_count: int,
        error: KvcmReportError,
        trace_id: str | None = None,
        merged_trace_ids: list[str] | None = None,
    ) -> None:
        """Record one dropped batch and emit a rate-limited warning."""

        self._dropped_batch_count += batch_count
        self._dropped_event_count += event_count
        if self._dropped_batch_count == batch_count:
            # First failure: log immediately with full detail.
            tags: dict[str, object] = {
                "pipeline": pipeline,
                "epoch": epoch,
                "batch_count": batch_count,
                "event_count": event_count,
                "error": error.__class__.__name__,
                "message": str(error),
                "reason": error.reason,
                "dropped_batch_total": self._dropped_batch_count,
                "dropped_event_total": self._dropped_event_count,
            }
            if trace_id is not None:
                tags["trace_id"] = trace_id
            if merged_trace_ids is not None:
                tags["merged_trace_ids"] = merged_trace_ids
            logger.warning(
                "failed to send kv event batch to kvcm; dropping batch",
                step="kvcm_send",
                tags=tags,
            )
            self._last_summary_at_s = self._clock()
            return

        self._dropped_since_last_summary += batch_count
        now_s = self._clock()
        due_by_count = self._dropped_since_last_summary >= self._summary_every
        due_by_time = (now_s - self._last_summary_at_s) >= self._summary_interval_s
        if not (due_by_count or due_by_time):
            return
        logger.warning(
            "kvcm send failures continue; dropping batches",
            step="kvcm_send",
            tags={
                "pipeline": pipeline,
                "epoch": epoch,
                "error": error.__class__.__name__,
                "message": str(error),
                "reason": error.reason,
                "dropped_batch_since_summary": self._dropped_since_last_summary,
                "dropped_batch_total": self._dropped_batch_count,
                "dropped_event_total": self._dropped_event_count,
            },
        )
        self._dropped_since_last_summary = 0
        self._last_summary_at_s = now_s


def _mark_stage(queued_items: list[PipelineContext], stage: str) -> None:
    for item in queued_items:
        item.mark(stage)


def _trace_tags(queued_items: list[PipelineContext]) -> dict[str, object]:
    """Build structured-log tags that identify one merged KVCM report."""

    return {
        "trace_id": queued_items[0].trace_id,
        "merged_trace_ids": [item.trace_id for item in queued_items],
    }


def _record_merged_request_gauges(
    merged: MergedBatch,
    *,
    send_started_at_s: float,
) -> None:
    """Record dimensions once on the telemetry representing one merged request.

    The first queued item owns request-level gauges because all merged items
    produce one physical KVCM ReportEvent. Per-item latency spans and terminal
    submission remain owned by their respective ``PipelineContext`` objects.
    """

    try:
        telemetry = merged.queued_items[0].telemetry
        telemetry.gauge("kvcm_merged_queue_item_count", len(merged.queued_items))
        telemetry.gauge("kvcm_source_batch_count", merged.source_batch_count)
        telemetry.gauge("kvcm_source_event_count", merged.source_event_count)
        telemetry.gauge("kvcm_merged_report_event_count", merged.report_event_count)
        telemetry.gauge("kvcm_queue_size_before_merge", merged.queue_qsize_before_merge)
        telemetry.gauge("kvcm_queue_size_after_merge", merged.queue_qsize_after_merge)
        oldest_enqueue_to_send_s = merged.queued_items[
            0
        ].telemetry.elapsed_since_checkpoint(
            "queue_enqueued",
            at_s=send_started_at_s,
        )
        if oldest_enqueue_to_send_s is not None:
            telemetry.gauge(
                "kvcm_oldest_enqueue_to_send_ms", oldest_enqueue_to_send_s * 1000
            )
        newest_enqueue_to_send_s = merged.queued_items[
            -1
        ].telemetry.elapsed_since_checkpoint(
            "queue_enqueued",
            at_s=send_started_at_s,
        )
        if newest_enqueue_to_send_s is not None:
            telemetry.gauge(
                "kvcm_newest_enqueue_to_send_ms", newest_enqueue_to_send_s * 1000
            )
    except Exception:
        pass


def _submit_dropped(
    queued_items: list[PipelineContext],
    reason: str,
) -> None:
    """Mark and submit every queued item as dropped with the given reason."""

    for item in queued_items:
        item.submit_dropped(reason)


async def _notify_metadata_protocol_error(
    callback: Callable[[], Awaitable[None]] | None,
) -> None:
    """Best-effort notify lifecycle of a local descriptor/event mismatch."""

    if callback is None:
        return
    try:
        await callback()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "failed to report metadata protocol failure to DashServing",
            step="kvcm_send",
            tags={"error": exc.__class__.__name__, "message": str(exc)},
            exc_info=True,
        )


def _submit_incremental_success(
    batches: list[KVEventBatch],
    merged: MergedBatch,
) -> None:
    """Submit telemetries after a successful send.

    Aggregate counters (stored/removed block hash counts) are attributed to the
    first queued item so each physical block hash is counted once per merged
    send, matching the pre-refactor behavior. The ``event_expand`` and
    ``kvcm_send`` stage marks are applied by :meth:`KvcmClient.report_kv_events`
    directly on each telemetry.
    """

    stored_block_hash_count = sum(
        len(event.block_hashes)
        for batch in batches
        for event in batch.events
        if isinstance(event, BlockStored)
    )
    removed_block_hash_count = sum(
        len(event.block_hashes)
        for batch in batches
        for event in batch.events
        if isinstance(event, BlockRemoved)
    )
    for index, item in enumerate(merged.queued_items):
        if index == 0:
            item.telemetry.count("stored_block_hash_count", stored_block_hash_count)
            item.telemetry.count("removed_block_hash_count", removed_block_hash_count)
        item.submit()


def _submit_snapshot_success(
    queued: PipelineContext,
    batches: list[KVEventBatch],
) -> None:
    """Submit snapshot counters after a successful send.

    Snapshot counterpart of :func:`_submit_success`: a snapshot batch is never
    merged, so the counters are attributed to the single queued item directly.
    The ``snapshot_build`` and ``kvcm_send`` stage marks are applied by
    :meth:`KvcmClient.report_snapshot` directly on the telemetry.
    """

    block_count = 0
    for batch in batches:
        for event in batch.events:
            if isinstance(event, BlockSnapshot):
                block_count += len(event.items)
    queued.telemetry.count("snapshot_block_count", block_count)
    queued.submit()


async def _consume_engine_events(
    events: AsyncGenerator[EngineEventBatch, None],
    coordinator: EngineHealthCoordinator,
    queue: asyncio.Queue[PipelineContext],
    reporter: MetricsReporter,
    *,
    pipeline: str,
) -> None:
    """Capture, enqueue, and time one engine event stream.

    Adapters may yield an event whose telemetry is already marked dropped
    (for example, a snapshot poll that failed the gRPC round-trip). Those
    events are reported through the pipeline's metrics reporter but never
    enqueued for downstream forwarding — the adapter has already logged the
    reason.

    Batches captured while the engine is not ready (no sendable epoch) are
    marked as dropped on their own telemetry and submitted to the reporter
    here, which fans out the drop-count metric and any partial-stage spans.
    """

    last_queue_full_warn_s: float | None = None
    try:
        async for event in events:
            if event.telemetry.drop_reason is not None:
                # Adapter-originated drop: report it and skip enqueue.
                reporter.submit(event.telemetry)
                continue
            epoch_snapshot = coordinator.capture_epoch()
            if epoch_snapshot is None:
                event.telemetry.mark_dropped("engine_not_ready")
                reporter.submit(event.telemetry)
                logger.warning(
                    "dropping pipeline batch captured while engine is not ready",
                    step="engine_event_consume",
                    tags={"pipeline": pipeline, "trace_id": event.trace_id},
                )
                continue
            queued = PipelineContext(
                event=event, epoch_snapshot=epoch_snapshot, reporter=reporter
            )
            if queue.full():
                now_s = time.monotonic()
                if (
                    last_queue_full_warn_s is None
                    or now_s - last_queue_full_warn_s >= _QUEUE_FULL_WARN_INTERVAL_S
                ):
                    last_queue_full_warn_s = now_s
                    logger.warning(
                        "pipeline queue is full; producer blocked until sender drains",
                        step="engine_event_consume",
                        tags={
                            "pipeline": pipeline,
                            "queue_maxsize": queue.maxsize,
                            "trace_id": event.trace_id,
                        },
                    )
            await queue.put(queued)
            queued.telemetry.checkpoint("queue_enqueued")
    finally:
        await events.aclose()


async def consume_incremental_events(
    adapter: AbstractEngineAdapter,
    coordinator: EngineHealthCoordinator,
    queue: asyncio.Queue[PipelineContext],
    reporter: MetricsReporter,
) -> None:
    """Enqueue incremental batches captured in a ready engine epoch."""

    await _consume_engine_events(
        adapter.subscribe_kv_events(),
        coordinator,
        queue,
        reporter,
        pipeline="incremental",
    )


async def consume_snapshot_events(
    adapter: AbstractEngineAdapter,
    coordinator: EngineHealthCoordinator,
    queue: asyncio.Queue[PipelineContext],
    reporter: MetricsReporter,
) -> None:
    """Enqueue snapshot batches independently from the incremental pipeline."""

    await _consume_engine_events(
        adapter.subscribe_snapshot_events(),
        coordinator,
        queue,
        reporter,
        pipeline="snapshot",
    )


async def send_snapshot_events(
    kvcm: KvcmClient,
    coordinator: EngineHealthCoordinator,
    queue: asyncio.Queue[PipelineContext],
    drop_tracker: KvcmDropTracker | None = None,
    pipeline: str = "snapshot",
    on_metadata_protocol_error: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Gate and forward queued full snapshots to kvcm in queue order.

    Each queued snapshot batch is sent individually rather than merged: a
    snapshot is full state, so contiguous batches are independent reports, not
    deltas to accumulate. Mirrors :func:`send_incremental_events` minus the merging and
    block-removal filtering that only apply to incremental deltas. Failures are
    intentionally lossy — a KVCM control-plane error drops the batch through the
    rate-limited tracker and continues consuming so the bounded producer queue
    never stalls the engine snapshot poller. KVCM availability is not a
    forwarding gate input; the KVCM heartbeat loop owns reconnection.
    ``asyncio.CancelledError`` and any non-:class:`KvcmReportError` exception
    propagate.
    """

    tracker = drop_tracker if drop_tracker is not None else KvcmDropTracker()
    while True:
        queued_context = await queue.get()
        try:
            queued_context.telemetry.mark("queue_wait")
            epoch = await coordinator.wait_ready_epoch()
            queued_context.telemetry.mark("engine_gate_wait")
            if not coordinator.is_epoch_current(queued_context.epoch_snapshot):
                _submit_dropped([queued_context], "epoch_changed")
                logger.warning(
                    "dropping snapshot batch because engine epoch changed before send",
                    step="kv_event_loop",
                    tags={
                        "pipeline": pipeline,
                        "trace_id": queued_context.trace_id,
                        "captured_epoch": queued_context.epoch_snapshot,
                        "current_epoch": epoch,
                    },
                )
                continue
            batches = queued_context.batches
            try:
                await kvcm.report_snapshot(
                    batches,
                    epoch,
                    telemetry=queued_context.telemetry,
                    trace_id=queued_context.trace_id,
                )
            except KvcmReportError as exc:
                drop_reason = (
                    "metadata_protocol"
                    if exc.reason == "metadata_protocol"
                    else "send_failed"
                )
                _submit_dropped([queued_context], drop_reason)
                tracker.record_drop(
                    pipeline=pipeline,
                    epoch=epoch,
                    batch_count=len(batches),
                    event_count=sum(len(batch.events) for batch in batches),
                    error=exc,
                    trace_id=queued_context.trace_id,
                    merged_trace_ids=[queued_context.trace_id],
                )
                if drop_reason == "metadata_protocol":
                    await _notify_metadata_protocol_error(on_metadata_protocol_error)
                continue
            _submit_snapshot_success(queued_context, batches)
        finally:
            queue.task_done()


async def _forward_merged_incremental_batches(
    kvcm: KvcmClient,
    coordinator: EngineHealthCoordinator,
    merged: MergedBatch,
    drop_tracker: KvcmDropTracker,
    pipeline: str,
    on_snapshot_required: Callable[[], None] | None = None,
    on_metadata_protocol_error: Callable[[], Awaitable[None]] | None = None,
) -> None:
    _mark_stage(merged.queued_items, "queue_wait")
    if len(merged.queued_items) > 1:
        first_trace = merged.queued_items[0].trace_id
        for item in merged.queued_items[1:]:
            item.batch_trace_id = first_trace
    epoch = await coordinator.wait_ready_epoch()
    _mark_stage(merged.queued_items, "engine_gate_wait")
    if not coordinator.is_epoch_current(merged.epoch_snapshot):
        _submit_dropped(merged.queued_items, "epoch_changed")
        logger.warning(
            "dropping kv event batch because engine epoch changed before send",
            step="kv_event_loop",
            tags={
                "pipeline": pipeline,
                **_trace_tags(merged.queued_items),
                "captured_epoch": merged.epoch_snapshot,
                "current_epoch": epoch,
            },
        )
        return

    batches = filter_block_removals(merged.batches)
    _mark_stage(merged.queued_items, "block_filter")
    if not batches:
        _submit_dropped(merged.queued_items, "filtered_empty")
        return
    send_started_at_s = time.monotonic()
    _record_merged_request_gauges(merged, send_started_at_s=send_started_at_s)
    try:
        log_merge_diagnostics(
            merged,
            pipeline=pipeline,
            send_started_at_s=send_started_at_s,
        )
    except Exception as exc:
        logger.warning(
            "failed to log kv event batch diagnostics; continuing with send",
            step="kvcm_send",
            tags={
                "pipeline": pipeline,
                **_trace_tags(merged.queued_items),
                "error": exc.__class__.__name__,
                "message": str(exc),
            },
            exc_info=True,
        )
    try:
        # TODO(kvcm-protocol): Align an explicit generation/session or ordering
        # contract with KVCM. The local engine epoch is not carried in the
        # ReportEvent payload, so an in-flight old-epoch BLOCK_ADD can race with
        # HOST_DOWN. This release accepts that limitation and does not add a
        # local send/HostDown lock until the wire-level contract is agreed.
        snapshot_required = await kvcm.report_kv_events(
            batches,
            epoch,
            telemetries=[item.telemetry for item in merged.queued_items],
            trace_id=merged.queued_items[0].trace_id,
        )
    except KvcmReportError as exc:
        # Intentionally lossy: a KVCM control-plane failure must not
        # backpressure or stop engine serving. Drop this batch, record it
        # through the rate-limited tracker, and continue consuming. KVCM
        # availability is not a forwarding gate input; the KVCM heartbeat
        # loop owns reconnection. asyncio.CancelledError and any
        # non-KvcmReportError (programming) exception propagate.
        drop_reason = (
            "metadata_protocol" if exc.reason == "metadata_protocol" else "send_failed"
        )
        _submit_dropped(merged.queued_items, drop_reason)
        drop_tracker.record_drop(
            pipeline=pipeline,
            epoch=epoch,
            batch_count=len(batches),
            event_count=sum(len(batch.events) for batch in batches),
            error=exc,
            trace_id=merged.queued_items[0].trace_id,
            merged_trace_ids=[item.trace_id for item in merged.queued_items],
        )
        if drop_reason == "metadata_protocol":
            await _notify_metadata_protocol_error(on_metadata_protocol_error)
        return
    if snapshot_required and on_snapshot_required is not None:
        try:
            on_snapshot_required()
        except Exception as exc:
            logger.warning(
                "failed to request immediate snapshot; "
                "continuing incremental forwarding",
                step="snapshot_signal",
                tags={
                    "pipeline": pipeline,
                    **_trace_tags(merged.queued_items),
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                },
                exc_info=True,
            )
    _submit_incremental_success(batches, merged)


async def send_incremental_events(
    kvcm: KvcmClient,
    coordinator: EngineHealthCoordinator,
    queue: asyncio.Queue[PipelineContext],
    max_merged_report_events: int,
    max_merged_queue_items: int,
    drop_tracker: KvcmDropTracker | None = None,
    pipeline: str = "incremental",
    on_snapshot_required: Callable[[], None] | None = None,
    on_metadata_protocol_error: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Merge, gate, and forward queued KV batches in queue order."""

    if max_merged_report_events < 1:
        raise ValueError("max_merged_report_events must be >= 1")
    if max_merged_queue_items < 1:
        raise ValueError("max_merged_queue_items must be >= 1")

    tracker = drop_tracker if drop_tracker is not None else KvcmDropTracker()
    pending: PipelineContext | None = None
    try:
        while True:
            merged, pending = await dequeue_merged_batches(
                queue,
                pending,
                max_merged_report_events,
                max_merged_queue_items,
            )
            try:
                await _forward_merged_incremental_batches(
                    kvcm,
                    coordinator,
                    merged,
                    tracker,
                    pipeline,
                    on_snapshot_required,
                    on_metadata_protocol_error,
                )
            finally:
                for _item in merged.queued_items:
                    queue.task_done()
    except asyncio.CancelledError:
        if pending is not None:
            logger.info(
                "dropping pending kv event batch because sender stopped",
                step="kv_event_loop",
                tags={
                    "pipeline": pipeline,
                    "trace_id": pending.trace_id,
                    "captured_epoch": pending.epoch_snapshot,
                    "batch_count": len(pending.batches),
                    "event_count": sum(len(batch.events) for batch in pending.batches),
                },
            )
        raise
    finally:
        if pending is not None:
            queue.task_done()
