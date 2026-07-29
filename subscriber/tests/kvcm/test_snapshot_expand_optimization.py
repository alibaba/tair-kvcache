"""Tests for the unique fused snapshot block builder."""

from __future__ import annotations

from subscriber.kvcm.event_payload import build_merged_snapshot_blocks
from subscriber.types import (
    BlockSnapshot,
    BlockSnapshotItem,
    BlockStored,
    KVEventBatch,
)


def _medium_mapper(medium: str | None) -> str:
    return {"GPU": "hbm", "CPU": "mem"}.get(medium, "")


def _block_specs(medium: str, group_idx: int | None) -> list[dict[str, str]]:
    return [
        {
            "name": f"spec-{group_idx}",
            "uri": f"vllm://127.0.0.1:9000/{medium}",
        }
    ]


class TestBuildMergedSnapshotBlocks:
    """Core behavior of the fused collect+merge function."""

    def test_empty_batches_returns_empty(self) -> None:
        assert (
            build_merged_snapshot_blocks(
                [], medium_mapper=_medium_mapper, block_specs=_block_specs
            )
            == []
        )

    def test_no_snapshot_events_returns_empty(self) -> None:
        batches = [
            KVEventBatch(
                ts=1.0,
                events=[
                    BlockStored(
                        block_hashes=[1],
                        parent_block_hash=None,
                        token_ids=[1],
                        block_size=2,
                        lora_id=None,
                        medium="GPU",
                        lora_name=None,
                        group_idx=0,
                    )
                ],
            )
        ]
        assert (
            build_merged_snapshot_blocks(
                batches, medium_mapper=_medium_mapper, block_specs=_block_specs
            )
            == []
        )

    def test_single_block_produces_one_entry(self) -> None:
        batches = [
            KVEventBatch(
                ts=1.0,
                events=[
                    BlockSnapshot(
                        medium="GPU",
                        block_size=16,
                        items=[BlockSnapshotItem(block_hash=101, group_idx=0)],
                    )
                ],
            )
        ]
        result = build_merged_snapshot_blocks(
            batches, medium_mapper=_medium_mapper, block_specs=_block_specs
        )
        assert result == [
            {
                "block_key": "101",
                "medium": "hbm",
                "specs": [{"name": "spec-0", "uri": "vllm://127.0.0.1:9000/hbm"}],
            }
        ]

    def test_merges_same_block_hash_different_group_idx(self) -> None:
        """Hybrid model: same block_hash appears with group 0 and group 1."""
        batches = [
            KVEventBatch(
                ts=1.0,
                events=[
                    BlockSnapshot(
                        medium="GPU",
                        block_size=16,
                        items=[
                            BlockSnapshotItem(block_hash=100, group_idx=0),
                            BlockSnapshotItem(block_hash=100, group_idx=1),
                            BlockSnapshotItem(block_hash=100, group_idx=1),
                        ],
                    )
                ],
            )
        ]
        result = build_merged_snapshot_blocks(
            batches, medium_mapper=_medium_mapper, block_specs=_block_specs
        )
        assert result == [
            {
                "block_key": "100",
                "medium": "hbm",
                "specs": [
                    {"name": "spec-0", "uri": "vllm://127.0.0.1:9000/hbm"},
                    {"name": "spec-1", "uri": "vllm://127.0.0.1:9000/hbm"},
                ],
            }
        ]

    def test_deduplicates_same_block_hash_same_group_idx(self) -> None:
        """High-concurrency duplicate: same (block_hash, group_idx) twice."""
        batches = [
            KVEventBatch(
                ts=1.0,
                events=[
                    BlockSnapshot(
                        medium="GPU",
                        block_size=16,
                        items=[
                            BlockSnapshotItem(block_hash=100, group_idx=0),
                            BlockSnapshotItem(block_hash=100, group_idx=0),
                        ],
                    )
                ],
            )
        ]
        result = build_merged_snapshot_blocks(
            batches, medium_mapper=_medium_mapper, block_specs=_block_specs
        )
        assert result == [
            {
                "block_key": "100",
                "medium": "hbm",
                "specs": [{"name": "spec-0", "uri": "vllm://127.0.0.1:9000/hbm"}],
            }
        ]

    def test_preserves_first_occurrence_order(self) -> None:
        batches = [
            KVEventBatch(
                ts=1.0,
                events=[
                    BlockSnapshot(
                        medium="GPU",
                        block_size=16,
                        items=[
                            BlockSnapshotItem(block_hash=10, group_idx=0),
                            BlockSnapshotItem(block_hash=20, group_idx=0),
                            BlockSnapshotItem(block_hash=10, group_idx=1),
                        ],
                    )
                ],
            )
        ]
        result = build_merged_snapshot_blocks(
            batches, medium_mapper=_medium_mapper, block_specs=_block_specs
        )
        assert len(result) == 2
        assert result[0]["block_key"] == "10"
        assert result[0]["specs"] == [
            {"name": "spec-0", "uri": "vllm://127.0.0.1:9000/hbm"},
            {"name": "spec-1", "uri": "vllm://127.0.0.1:9000/hbm"},
        ]
        assert result[1]["block_key"] == "20"

    def test_different_mediums_not_merged(self) -> None:
        batches = [
            KVEventBatch(
                ts=1.0,
                events=[
                    BlockSnapshot(
                        medium="GPU",
                        block_size=16,
                        items=[BlockSnapshotItem(block_hash=1, group_idx=0)],
                    ),
                    BlockSnapshot(
                        medium="CPU",
                        block_size=16,
                        items=[BlockSnapshotItem(block_hash=1, group_idx=0)],
                    ),
                ],
            )
        ]
        result = build_merged_snapshot_blocks(
            batches, medium_mapper=_medium_mapper, block_specs=_block_specs
        )
        assert len(result) == 2
        assert result[0]["medium"] == "hbm"
        assert result[1]["medium"] == "mem"

    def test_multiple_batches_merged_together(self) -> None:
        """Blocks across separate batches merge by (block_key, medium)."""
        batches = [
            KVEventBatch(
                ts=1.0,
                events=[
                    BlockSnapshot(
                        medium="GPU",
                        block_size=16,
                        items=[BlockSnapshotItem(block_hash=5, group_idx=0)],
                    )
                ],
            ),
            KVEventBatch(
                ts=2.0,
                events=[
                    BlockSnapshot(
                        medium="GPU",
                        block_size=16,
                        items=[BlockSnapshotItem(block_hash=5, group_idx=1)],
                    )
                ],
            ),
        ]
        result = build_merged_snapshot_blocks(
            batches, medium_mapper=_medium_mapper, block_specs=_block_specs
        )
        assert result == [
            {
                "block_key": "5",
                "medium": "hbm",
                "specs": [
                    {"name": "spec-0", "uri": "vllm://127.0.0.1:9000/hbm"},
                    {"name": "spec-1", "uri": "vllm://127.0.0.1:9000/hbm"},
                ],
            }
        ]


class TestSpecCaching:
    """Verify that block_specs callable can return cached (shared) references."""

    def test_cached_spec_dicts_shared_across_blocks(self) -> None:
        """Inner spec dicts are shared references from the cache."""
        cache: dict[tuple[str, int | None], list[dict[str, str]]] = {}

        def cached_block_specs(
            medium: str, group_idx: int | None
        ) -> list[dict[str, str]]:
            key = (medium, group_idx)
            if key not in cache:
                cache[key] = [
                    {"name": f"spec-{group_idx}", "uri": f"vllm://host/{medium}"}
                ]
            return cache[key]

        batches = [
            KVEventBatch(
                ts=1.0,
                events=[
                    BlockSnapshot(
                        medium="GPU",
                        block_size=16,
                        items=[
                            BlockSnapshotItem(block_hash=1, group_idx=0),
                            BlockSnapshotItem(block_hash=2, group_idx=0),
                            BlockSnapshotItem(block_hash=3, group_idx=0),
                        ],
                    )
                ],
            )
        ]
        result = build_merged_snapshot_blocks(
            batches, medium_mapper=_medium_mapper, block_specs=cached_block_specs
        )
        # Inner spec dicts share the same cached object
        specs_0 = result[0]["specs"]
        specs_1 = result[1]["specs"]
        assert isinstance(specs_0, list) and isinstance(specs_1, list)
        assert specs_0[0] is specs_1[0]
        # Only one cache entry was created
        assert len(cache) == 1

    def test_cached_specs_not_corrupted_by_merge(self) -> None:
        """Merging group 1 into block 100 must not pollute cached spec for group 0."""
        cache: dict[tuple[str, int | None], list[dict[str, str]]] = {}

        def cached_block_specs(
            medium: str, group_idx: int | None
        ) -> list[dict[str, str]]:
            key = (medium, group_idx)
            if key not in cache:
                cache[key] = [
                    {"name": f"spec-{group_idx}", "uri": f"vllm://host/{medium}"}
                ]
            return cache[key]

        batches = [
            KVEventBatch(
                ts=1.0,
                events=[
                    BlockSnapshot(
                        medium="GPU",
                        block_size=16,
                        items=[
                            BlockSnapshotItem(block_hash=100, group_idx=0),
                            BlockSnapshotItem(block_hash=100, group_idx=1),
                            BlockSnapshotItem(block_hash=200, group_idx=0),
                        ],
                    )
                ],
            )
        ]
        result = build_merged_snapshot_blocks(
            batches, medium_mapper=_medium_mapper, block_specs=cached_block_specs
        )
        # Block 100 merged: has both specs
        merged_specs = result[0]["specs"]
        assert isinstance(merged_specs, list)
        assert len(merged_specs) == 2
        # Block 200 must have only spec-0
        assert result[1]["specs"] == [{"name": "spec-0", "uri": "vllm://host/hbm"}]
        # Cache must not be corrupted
        assert cache[("hbm", 0)] == [{"name": "spec-0", "uri": "vllm://host/hbm"}]
        assert cache[("hbm", 1)] == [{"name": "spec-1", "uri": "vllm://host/hbm"}]
