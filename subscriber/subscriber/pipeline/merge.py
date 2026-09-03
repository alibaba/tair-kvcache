"""Greedy same-epoch merge of queued pipeline contexts.

Selects contiguous items from an asyncio queue that share the same engine
epoch, bounded by both a maximum number of queue items and a maximum number
of report events.  Returns a :class:`MergedBatch` describing the selection
and any item that was dequeued but not merged (handed back as ``pending``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from subscriber import logger
from subscriber.kvcm.event_payload import report_event_count
from subscriber.pipeline.context import PipelineContext
from subscriber.types import KVEventBatch


@dataclass(frozen=True)
class MergedBatch:
    """Contiguous same-epoch queue items selected for one KVCM request."""

    queued_items: list[PipelineContext]
    epoch_snapshot: int
    queue_qsize_before_merge: int
    queue_qsize_after_merge: int
    source_batch_count: int
    source_event_count: int
    report_event_count: int

    @property
    def batches(self) -> list[KVEventBatch]:
        return [batch for item in self.queued_items for batch in item.batches]


def exceeds_merge_limit(
    count: int,
    max_merged_report_events: int,
) -> bool:
    return count > max_merged_report_events


async def dequeue_merged_batches(
    queue: asyncio.Queue[PipelineContext],
    pending: PipelineContext | None,
    max_merged_report_events: int,
    max_merged_queue_items: int,
) -> tuple[MergedBatch, PipelineContext | None]:
    """Select immediately available same-epoch items within both merge caps."""

    first = pending if pending is not None else await queue.get()
    queue_qsize_before_merge = queue.qsize()
    queued_items = [first]
    source_batch_count = len(first.batches)
    source_event_count = sum(len(batch.events) for batch in first.batches)
    merged_report_event_count = report_event_count(first.batches)
    next_pending: PipelineContext | None = None

    while len(queued_items) < max_merged_queue_items:
        try:
            candidate = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if candidate.epoch_snapshot != first.epoch_snapshot:
            next_pending = candidate
            break

        candidate_report_event_count = report_event_count(candidate.batches)
        if exceeds_merge_limit(
            merged_report_event_count + candidate_report_event_count,
            max_merged_report_events,
        ):
            next_pending = candidate
            break

        queued_items.append(candidate)
        source_batch_count += len(candidate.batches)
        source_event_count += sum(len(batch.events) for batch in candidate.batches)
        merged_report_event_count += candidate_report_event_count

    return (
        MergedBatch(
            queued_items=queued_items,
            epoch_snapshot=first.epoch_snapshot,
            queue_qsize_before_merge=queue_qsize_before_merge,
            queue_qsize_after_merge=queue.qsize(),
            source_batch_count=source_batch_count,
            source_event_count=source_event_count,
            report_event_count=merged_report_event_count,
        ),
        next_pending,
    )


def log_merge_diagnostics(
    merged: MergedBatch,
    *,
    pipeline: str,
    send_started_at_s: float,
) -> None:
    if logger.is_debug_enabled():
        oldest_enqueue_to_kvcm_send_s = merged.queued_items[
            0
        ].telemetry.elapsed_since_checkpoint(
            "queue_enqueued",
            at_s=send_started_at_s,
        )
        newest_enqueue_to_kvcm_send_s = merged.queued_items[
            -1
        ].telemetry.elapsed_since_checkpoint(
            "queue_enqueued",
            at_s=send_started_at_s,
        )
        tags: dict[str, object] = {
            "pipeline": pipeline,
            "trace_id": merged.queued_items[0].trace_id,
            "merged_trace_ids": [item.trace_id for item in merged.queued_items],
            "queue_qsize_before_merge": merged.queue_qsize_before_merge,
            "queue_qsize_after_merge": merged.queue_qsize_after_merge,
            "merged_queue_item_count": len(merged.queued_items),
            "source_batch_count": merged.source_batch_count,
            "source_event_count": merged.source_event_count,
            "merged_report_event_count": merged.report_event_count,
        }
        if oldest_enqueue_to_kvcm_send_s is not None:
            tags["oldest_enqueue_to_kvcm_send_ms"] = round(
                oldest_enqueue_to_kvcm_send_s * 1000,
                3,
            )
        if newest_enqueue_to_kvcm_send_s is not None:
            tags["newest_enqueue_to_kvcm_send_ms"] = round(
                newest_enqueue_to_kvcm_send_s * 1000,
                3,
            )
        logger.debug(
            "sending merged kv event batches to kvcm",
            step="kvcm_send",
            tags=tags,
        )
