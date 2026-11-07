"""Shared debug summaries for normalized KV event batches."""

from __future__ import annotations

from collections import Counter

import msgspec

from subscriber.types import BlockRemoved, BlockStored, KVEventBatch, format_block_hash

_MAX_DEBUG_BLOCK_HASHES = 32


def _event_for_debug(event: BlockStored | BlockRemoved) -> dict[str, object]:
    """Copy an event for logging without token IDs or forwarding mutations."""

    event_data = msgspec.structs.asdict(event)
    event_data.pop("token_ids", None)
    event_data["block_hashes"] = [
        format_block_hash(block_hash) for block_hash in event.block_hashes
    ]
    if isinstance(event, BlockStored) and event.parent_block_hash is not None:
        event_data["parent_block_hash"] = format_block_hash(event.parent_block_hash)
    return event_data


def summarize_kv_event_batch_for_debug(batch: KVEventBatch) -> dict[str, object]:
    """Return bounded, token-free details for a decoded ZMQ batch debug log."""

    event_type_counts = Counter(type(event).__name__ for event in batch.events)
    stored_blocks: list[dict[str, object]] = []
    removed_blocks: list[dict[str, object]] = []
    for event in batch.events:
        if isinstance(event, BlockStored):
            if len(stored_blocks) < _MAX_DEBUG_BLOCK_HASHES:
                stored_blocks.append(_event_for_debug(event))
        elif isinstance(event, BlockRemoved):
            if len(removed_blocks) < _MAX_DEBUG_BLOCK_HASHES:
                removed_blocks.append(_event_for_debug(event))
    return {
        "event_count": len(batch.events),
        "event_types": ",".join(
            f"{name}:{count}" for name, count in sorted(event_type_counts.items())
        ),
        "data_parallel_rank": batch.data_parallel_rank,
        "stored_block_count": len(stored_blocks),
        "stored_blocks": stored_blocks,
        "stored_blocks_truncated": sum(
            1 for event in batch.events if isinstance(event, BlockStored)
        )
        > len(stored_blocks),
        "removed_block_count": len(removed_blocks),
        "removed_blocks": removed_blocks,
        "removed_blocks_truncated": sum(
            1 for event in batch.events if isinstance(event, BlockRemoved)
        )
        > len(removed_blocks),
    }
