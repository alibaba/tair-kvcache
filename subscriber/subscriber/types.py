from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import msgspec

# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of `bytes` and `int` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
ExternalBlockHash: TypeAlias = bytes | int


def format_block_hash(block_hash: ExternalBlockHash) -> str:
    """Human-readable string encoding of a block hash for debug logging only.

    Bytes hashes render as lowercase hex; int hashes as decimal ``str``.
    This is used exclusively in debug log output (e.g. ``_event_for_debug``)
    to make block hashes readable. Wire/forwarding paths use ``str()``
    directly since production hashes are always ints.
    """

    if isinstance(block_hash, bytes):
        return block_hash.hex()
    return str(block_hash)


@dataclass(frozen=True)
class KvCacheGroupSpec:
    """Per-group KV cache metadata fetched from the engine.

    ``block_size`` is the group's token span. ``group_payload_size_bytes`` is
    the complete logical group-block payload used as the KVCM location size.
    Both are provided explicitly by the engine metadata endpoint; do not infer
    ``group_idx`` from list position. ``sliding_window`` is ``None`` for
    non-windowed kinds.
    """

    group_idx: int
    kind: str
    block_size: int
    group_payload_size_bytes: int | None
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
    """Common pipeline event with engine-native component identity.

    vLLM publishes ``group_idx`` while SGLang publishes ``component_id`` for
    the same semantic concept. SGLang's adapter also maps its component ID to
    the common pipeline ``group_idx`` used by KVCM; publishers are never
    required to put both fields on their wire event.
    """


class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None
    lora_name: str | None
    extra_keys: list[tuple[Any, ...] | None] | None = None
    # vLLM-native identity; for SGLang this is the adapter's normalized mapping.
    group_idx: int | None = None
    # SGLang-native identity retained after normalization; absent on vLLM wire.
    component_id: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    snapshot_version: int = 0


class BlockRemoved(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    medium: str | None
    # vLLM-native identity; for SGLang this is the adapter's normalized mapping.
    group_idx: int | None = None
    # SGLang-native identity retained after normalization; absent on vLLM wire.
    component_id: int | None = None
    remaining_copy_counts: list[int] | None = None
    snapshot_version: int = 0


class AllBlocksCleared(KVCacheEvent):
    pass


class BlockSnapshotItem(msgspec.Struct, omit_defaults=True, gc=False):
    """One block in a full snapshot report."""

    block_hash: ExternalBlockHash
    group_idx: int


class BlockSnapshot(KVCacheEvent):
    """Full snapshot of all currently cached blocks.

    Sent periodically by the hybrid adapter for kvcm reconciliation.
    KVCM handles add/delete bookkeeping from successive snapshots.
    """

    medium: str
    block_size: int
    items: list[BlockSnapshotItem]
    snapshot_version: int = 0


class KVEventBatch(EventBatch):
    events: list[BlockStored | BlockRemoved | AllBlocksCleared | BlockSnapshot]
