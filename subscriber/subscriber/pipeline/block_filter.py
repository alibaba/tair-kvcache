from __future__ import annotations

from subscriber import logger
from subscriber.types import (
    AllBlocksCleared,
    BlockRemoved,
    BlockSnapshot,
    BlockStored,
    KVEventBatch,
)


def filter_block_removals(batches: list[KVEventBatch]) -> list[KVEventBatch]:
    """Suppress BlockRemoved hashes whose remaining physical copies are non-zero.

    When ``remaining_copy_counts`` is provided by the engine, only block hashes
    with a remaining count of 0 are forwarded to KVCM.  When the field is
    ``None`` (engine does not support it), the event passes through unmodified.
    BlockSnapshot events pass through unchanged.
    """

    result: list[KVEventBatch] = []
    for batch in batches:
        events = _filter_events(batch.events)
        if not events:
            continue
        if len(events) == len(batch.events) and all(
            e is orig for e, orig in zip(events, batch.events, strict=True)
        ):
            result.append(batch)
        else:
            result.append(
                KVEventBatch(
                    ts=batch.ts,
                    events=events,
                    data_parallel_rank=batch.data_parallel_rank,
                )
            )
    return result


def _filter_events(
    events: list[BlockStored | BlockRemoved | AllBlocksCleared | BlockSnapshot],
) -> list[BlockStored | BlockRemoved | AllBlocksCleared | BlockSnapshot]:
    filtered: list[BlockStored | BlockRemoved | AllBlocksCleared | BlockSnapshot] = []
    for event in events:
        if not isinstance(event, BlockRemoved):
            filtered.append(event)
            continue
        if event.remaining_copy_counts is None:
            filtered.append(event)
            continue
        if len(event.block_hashes) != len(event.remaining_copy_counts):
            logger.warning(
                "dropping block removal with mismatched remaining copy counts",
                step="block_filter",
                tags={
                    "block_hash_count": len(event.block_hashes),
                    "remaining_copy_count": len(event.remaining_copy_counts),
                    "medium": event.medium,
                    "group_idx": event.group_idx,
                },
            )
            continue
        kept_hashes: list[bytes | int] = []
        kept_counts: list[int] = []
        for block_hash, count in zip(
            event.block_hashes, event.remaining_copy_counts, strict=True
        ):
            if count == 0:
                kept_hashes.append(block_hash)
                kept_counts.append(0)
        if not kept_hashes:
            continue
        if len(kept_hashes) == len(event.block_hashes):
            filtered.append(event)
        else:
            filtered.append(
                BlockRemoved(
                    block_hashes=kept_hashes,
                    medium=event.medium,
                    group_idx=event.group_idx,
                    component_id=event.component_id,
                    remaining_copy_counts=kept_counts,
                    snapshot_version=event.snapshot_version,
                )
            )
    return filtered
