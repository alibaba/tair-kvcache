from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import msgspec

# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of `bytes` and `int` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
ExternalBlockHash: TypeAlias = bytes | int


@dataclass(frozen=True)
class KvCacheGroupSpec:
    """Per-group KV cache metadata fetched from the engine.

    ``group_idx`` and ``block_size`` are provided explicitly by the engine's
    metadata endpoint (do not infer ``group_idx`` from list position).
    ``sliding_window`` is ``None`` for non-windowed kinds.
    """

    group_idx: int
    kind: str
    block_size: int
    sliding_window: int | None = None


class EventBatch(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    ts: float
    events: list[Any]
    data_parallel_rank: int | None = None


class KVCacheEvent(
    msgspec.Struct,
    omit_defaults=True,
    gc=False,
    tag=True,
):
    pass


class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None
    lora_name: str | None
    extra_keys: list[tuple[Any, ...] | None] | None = None
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None


class BlockRemoved(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    medium: str | None
    group_idx: int | None = None
    remaining_copy_counts: list[int] | None = None


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    events: list[BlockStored | BlockRemoved | AllBlocksCleared]
