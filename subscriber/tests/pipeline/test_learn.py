from __future__ import annotations

from subscriber.pipeline.learn import GroupMetadataLearner
from subscriber.types import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KvCacheGroupSpec,
    KVEventBatch,
)


def _stored(
    *,
    group_idx: int | None = None,
    kind: str | None = None,
    sliding_window: int | None = None,
    block_size: int = 16,
    block_hash: int = 1,
) -> BlockStored:
    return BlockStored(
        block_hashes=[block_hash],
        parent_block_hash=None,
        token_ids=[1],
        block_size=block_size,
        lora_id=None,
        medium="GPU",
        lora_name=None,
        group_idx=group_idx,
        kv_cache_spec_kind=kind,
        kv_cache_spec_sliding_window=sliding_window,
    )


def _batch(events: list) -> list[KVEventBatch]:
    return [KVEventBatch(ts=1.0, events=events)]


def test_initial_state_has_no_metadata() -> None:
    learner = GroupMetadataLearner()
    assert learner.snapshot() == {}
    assert not learner.has_new_groups()


def test_learn_single_group_from_block_stored() -> None:
    learner = GroupMetadataLearner()
    learned = learner.observe_batch(
        _batch([_stored(group_idx=0, kind="full_attention", block_size=16)])
    )

    assert learned is True
    assert learner.has_new_groups()
    snap = learner.snapshot()
    assert snap == {
        0: KvCacheGroupSpec(
            group_idx=0, kind="full_attention", block_size=16, sliding_window=None
        )
    }


def test_learn_multiple_groups() -> None:
    learner = GroupMetadataLearner()
    learner.observe_batch(
        _batch(
            [
                _stored(
                    group_idx=0, kind="full_attention", block_size=16, block_hash=1
                ),
                _stored(
                    group_idx=1,
                    kind="sliding_window",
                    block_size=16,
                    sliding_window=32768,
                    block_hash=2,
                ),
                _stored(group_idx=2, kind="mamba", block_size=1, block_hash=3),
            ]
        )
    )

    snap = learner.snapshot()
    assert len(snap) == 3
    assert snap[0].kind == "full_attention"
    assert snap[1].sliding_window == 32768
    assert snap[2].block_size == 1


def test_does_not_overwrite_already_learned_group() -> None:
    learner = GroupMetadataLearner()
    learner.observe_batch(
        _batch([_stored(group_idx=0, kind="full_attention", block_size=16)])
    )
    learner.consume_new_groups()

    learned = learner.observe_batch(
        _batch([_stored(group_idx=0, kind="sliding_window", block_size=32)])
    )

    assert learned is False
    assert learner.snapshot()[0].kind == "full_attention"
    assert learner.snapshot()[0].block_size == 16


def test_ignore_events_without_group_idx() -> None:
    learner = GroupMetadataLearner()
    learned = learner.observe_batch(
        _batch([_stored(group_idx=None, kind="full_attention")])
    )
    assert learned is False
    assert learner.snapshot() == {}


def test_ignore_events_without_kind() -> None:
    learner = GroupMetadataLearner()
    learned = learner.observe_batch(_batch([_stored(group_idx=0, kind=None)]))
    assert learned is False
    assert learner.snapshot() == {}


def test_ignore_block_removed_and_all_blocks_cleared() -> None:
    learner = GroupMetadataLearner()
    learned = learner.observe_batch(
        _batch(
            [
                BlockRemoved(block_hashes=[1], medium="GPU", group_idx=0),
                AllBlocksCleared(),
            ]
        )
    )
    assert learned is False
    assert learner.snapshot() == {}


def test_sliding_window_minus_one_decoded_as_none() -> None:
    learner = GroupMetadataLearner()
    learner.observe_batch(
        _batch(
            [
                _stored(
                    group_idx=0,
                    kind="full_attention",
                    block_size=16,
                    sliding_window=-1,
                )
            ]
        )
    )
    assert learner.snapshot()[0].sliding_window is None


def test_consume_new_groups_clears_flag() -> None:
    learner = GroupMetadataLearner()
    learner.observe_batch(
        _batch([_stored(group_idx=0, kind="full_attention", block_size=16)])
    )
    assert learner.has_new_groups()

    learner.consume_new_groups()
    assert not learner.has_new_groups()

    learned = learner.observe_batch(
        _batch([_stored(group_idx=0, kind="full_attention", block_size=16)])
    )
    assert learned is False
    assert not learner.has_new_groups()


def test_mixed_batch_only_learns_valid_entries() -> None:
    learner = GroupMetadataLearner()
    learned = learner.observe_batch(
        _batch(
            [
                _stored(
                    group_idx=0, kind="full_attention", block_size=16, block_hash=1
                ),
                _stored(group_idx=None, kind="mamba", block_hash=2),
                BlockRemoved(block_hashes=[3], medium="GPU"),
                _stored(group_idx=1, kind="mamba", block_size=1, block_hash=4),
            ]
        )
    )

    assert learned is True
    snap = learner.snapshot()
    assert len(snap) == 2
    assert snap[0].kind == "full_attention"
    assert snap[1].kind == "mamba"
