from __future__ import annotations

from subscriber import logger
from subscriber.types import BlockStored, KvCacheGroupSpec, KVEventBatch


class GroupMetadataLearner:
    """Learn KV cache group topology from live BlockStored events.

    Activated only when the gRPC metadata fetch fails. Each ``BlockStored``
    event carries ``group_idx``, ``kv_cache_spec_kind``, ``block_size``, and
    optionally ``kv_cache_spec_sliding_window`` — enough to reconstruct the
    same ``KvCacheGroupSpec`` that the gRPC endpoint would have returned.

    The learner is append-only: once a ``group_idx`` is learned, subsequent
    observations for the same index are ignored (the engine's group topology
    is stable within a single process lifetime).
    """

    def __init__(self) -> None:
        self._group_by_idx: dict[int, KvCacheGroupSpec] = {}
        self._has_new = False

    def observe_batch(self, batches: list[KVEventBatch]) -> bool:
        """Scan batches for BlockStored events with group metadata.

        Returns True if at least one new group was learned.
        """
        learned_any = False
        for batch in batches:
            for event in batch.events:
                if not isinstance(event, BlockStored):
                    continue
                group_idx = event.group_idx
                kind = event.kv_cache_spec_kind
                if group_idx is None or kind is None:
                    logger.debug(
                        "skipping block_stored without group metadata",
                        step="kv_metadata_learn",
                        tags={
                            "group_idx": group_idx,
                            "kind": kind,
                        },
                    )
                    continue
                if group_idx in self._group_by_idx:
                    continue
                sliding_window = event.kv_cache_spec_sliding_window
                if sliding_window is not None and sliding_window == -1:
                    sliding_window = None
                spec = KvCacheGroupSpec(
                    group_idx=group_idx,
                    kind=kind,
                    block_size=event.block_size,
                    sliding_window=sliding_window,
                )
                self._group_by_idx[group_idx] = spec
                learned_any = True
                logger.debug(
                    "learned new kv cache group from live event",
                    step="kv_metadata_learn",
                    tags={
                        "group_idx": group_idx,
                        "kind": kind,
                        "block_size": event.block_size,
                        "sliding_window": sliding_window,
                        "total_learned": len(self._group_by_idx),
                    },
                )
        if learned_any:
            self._has_new = True
        return learned_any

    def has_new_groups(self) -> bool:
        """True if at least one group was learned since last consume."""
        return self._has_new

    def consume_new_groups(self) -> None:
        """Acknowledge that the caller has processed the newly learned groups."""
        self._has_new = False

    def snapshot(self) -> dict[int, KvCacheGroupSpec]:
        """Return a copy of the currently learned group metadata."""
        return dict(self._group_by_idx)
