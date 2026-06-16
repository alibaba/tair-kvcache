"""
Convert block_ids between different block sizes while preserving prefix relationships.

When dataset block_ids are captured at gateway level (e.g., block_size=256) but the
actual engine uses a different page_size (e.g., 2048), consecutive source blocks
need to be merged into target blocks.

Example:
    source block_size=256, target page_size=2048, ratio=8
    [b0,b1,b2,b3,b4,b5,b6,b7, b8,b9,...] → [hash(b0..b7), hash(b8..b15), ...]
"""
import struct
import hashlib
from typing import Optional


def convert_block_ids(
    block_ids: list[int],
    source_block_size: int,
    target_block_size: int,
) -> list[int]:
    """
    Merge consecutive source blocks into target blocks.

    Args:
        block_ids: Original block hash IDs (each representing source_block_size tokens).
        source_block_size: Token count per block in the input data (e.g., 256).
        target_block_size: Token count per block for the engine (e.g., 2048).

    Returns:
        New block_ids where each ID represents target_block_size tokens.
        Incomplete trailing groups are dropped.

    Raises:
        ValueError: If target is not a multiple of source, or sizes are invalid.
    """
    if source_block_size <= 0 or target_block_size <= 0:
        raise ValueError(
            f"Block sizes must be positive: source={source_block_size}, "
            f"target={target_block_size}"
        )

    if source_block_size == target_block_size:
        return list(block_ids)

    if target_block_size < source_block_size:
        raise ValueError(
            f"target_block_size ({target_block_size}) must be >= "
            f"source_block_size ({source_block_size}). "
            f"Splitting blocks is not supported."
        )

    if target_block_size % source_block_size != 0:
        raise ValueError(
            f"target_block_size ({target_block_size}) must be a multiple of "
            f"source_block_size ({source_block_size})"
        )

    ratio = target_block_size // source_block_size
    result = []
    for i in range(0, len(block_ids) - ratio + 1, ratio):
        group = block_ids[i : i + ratio]
        result.append(_hash_block_group(group))
    return result


def _hash_block_group(block_ids: list[int]) -> int:
    """
    Compute a deterministic int64 hash for a group of block IDs.

    Uses SHA-256 on the packed int64 values, then takes the first 8 bytes
    as a signed int64. This ensures:
    - Determinism: same input always produces same output
    - Low collision: SHA-256 has excellent distribution
    - Stability: independent of Python hash seed
    """
    data = struct.pack(f">{len(block_ids)}q", *block_ids)
    digest = hashlib.sha256(data).digest()[:8]
    return struct.unpack(">q", digest)[0]
