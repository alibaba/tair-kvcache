from __future__ import annotations

from subscriber.pipeline.block_filter import filter_block_removals
from subscriber.types import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    ExternalBlockHash,
    KVEventBatch,
)


def _stored(
    block_hashes: list[ExternalBlockHash], *, medium: str = "GPU"
) -> BlockStored:
    return BlockStored(
        block_hashes=block_hashes,
        parent_block_hash=None,
        token_ids=[],
        block_size=16,
        lora_id=None,
        medium=medium,
        lora_name=None,
    )


def _removed(
    block_hashes: list[ExternalBlockHash],
    *,
    medium: str = "GPU",
    group_idx: int | None = None,
    remaining_copy_counts: list[int] | None = None,
) -> BlockRemoved:
    return BlockRemoved(
        block_hashes=block_hashes,
        medium=medium,
        group_idx=group_idx,
        remaining_copy_counts=remaining_copy_counts,
    )


def _batch(
    *events: BlockStored | BlockRemoved | AllBlocksCleared,
) -> list[KVEventBatch]:
    return [KVEventBatch(ts=1.0, events=list(events))]


def _events(batches: list[KVEventBatch]) -> list[object]:
    return [event for batch in batches for event in batch.events]


# --- remaining_copy_counts provided: filter by count ---


def test_all_zero_remaining_counts_forwards_all_hashes() -> None:
    """All remaining counts are 0 → forward all block hashes."""
    batches = _batch(
        _removed([1, 2, 3], remaining_copy_counts=[0, 0, 0]),
    )
    result = filter_block_removals(batches)
    assert _events(result) == [_removed([1, 2, 3], remaining_copy_counts=[0, 0, 0])]


def test_all_nonzero_remaining_counts_suppresses_entire_event() -> None:
    """All remaining counts > 0 → suppress entire BlockRemoved event."""
    batches = _batch(
        _removed([1, 2], remaining_copy_counts=[2, 1]),
    )
    result = filter_block_removals(batches)
    assert _events(result) == []


def test_mixed_remaining_counts_keeps_only_zero_count_hashes() -> None:
    """Mixed counts → only forward hashes with remaining_copy_counts == 0."""
    batches = _batch(
        _removed([1, 2, 3], group_idx=5, remaining_copy_counts=[1, 0, 0]),
    )
    result = filter_block_removals(batches)
    assert _events(result) == [
        _removed([2, 3], group_idx=5, remaining_copy_counts=[0, 0]),
    ]


def test_mismatched_remaining_counts_drops_removal_and_preserves_later_batch(
    mocker,
) -> None:
    warning = mocker.patch("subscriber.logger.warning")
    stored = _stored([3])

    result = filter_block_removals(
        [
            KVEventBatch(
                ts=1.0,
                events=[_removed([1, 2], group_idx=5, remaining_copy_counts=[0])],
            ),
            KVEventBatch(ts=2.0, events=[stored]),
        ]
    )

    assert _events(result) == [stored]
    warning.assert_called_once_with(
        "dropping block removal with mismatched remaining copy counts",
        step="block_filter",
        tags={
            "block_hash_count": 2,
            "remaining_copy_count": 1,
            "medium": "GPU",
            "group_idx": 5,
        },
    )


# --- remaining_copy_counts is None: passthrough ---


def test_none_remaining_counts_forwards_all_hashes() -> None:
    """remaining_copy_counts is None → forward all (engine didn't provide counts)."""
    batches = _batch(
        _removed([1, 2]),
    )
    result = filter_block_removals(batches)
    assert _events(result) == [_removed([1, 2])]


# --- non-BlockRemoved events pass through ---


def test_block_stored_and_all_cleared_pass_through() -> None:
    """BlockStored and AllBlocksCleared are never filtered."""
    batches = _batch(
        _stored([1]),
        AllBlocksCleared(),
    )
    result = filter_block_removals(batches)
    assert _events(result) == [_stored([1]), AllBlocksCleared()]


# --- empty batch is dropped ---


def test_empty_batch_after_full_suppression_is_dropped() -> None:
    """If all events in a batch are suppressed, the batch itself is omitted."""
    batches = _batch(
        _removed([1], remaining_copy_counts=[3]),
    )
    result = filter_block_removals(batches)
    assert result == []


# --- multiple batches ---


def test_filters_across_multiple_batches() -> None:
    """Filter applies independently to each batch."""
    batch1 = KVEventBatch(
        ts=1.0,
        events=[_removed([1, 2], remaining_copy_counts=[0, 1])],
    )
    batch2 = KVEventBatch(
        ts=2.0,
        events=[
            _stored([3]),
            _removed([4], remaining_copy_counts=[0]),
        ],
    )
    result = filter_block_removals([batch1, batch2])
    assert len(result) == 2
    assert _events([result[0]]) == [
        _removed([1], remaining_copy_counts=[0]),
    ]
    assert _events([result[1]]) == [
        _stored([3]),
        _removed([4], remaining_copy_counts=[0]),
    ]
