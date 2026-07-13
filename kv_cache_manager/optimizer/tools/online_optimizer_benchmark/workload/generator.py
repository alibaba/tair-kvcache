"""Synthetic workload generator for TraceQuery benchmark."""

import random
from typing import List, Optional

from ..core.config import BenchmarkConfig


class WorkloadGenerator:
    """Generates block_keys lists simulating cache access patterns."""

    def __init__(self, config: BenchmarkConfig, seed: Optional[int] = None):
        self._block_keys_count = config.block_keys_count
        self._max_block_key = config.max_block_key
        self._prefix_reuse_ratio = config.prefix_reuse_ratio
        self._prefix_reuse_length = config.prefix_reuse_length
        self._rng = random.Random(seed)
        self._last_keys: List[int] = []

    def _random_key(self) -> int:
        if self._max_block_key < 0:
            return self._rng.getrandbits(63)
        return self._rng.randint(0, self._max_block_key)

    def generate(self) -> List[int]:
        """Generate a block_keys list for one TraceQuery request."""
        count = self._block_keys_count

        if self._last_keys and self._rng.random() < self._prefix_reuse_ratio:
            max_reuse = min(len(self._last_keys), count)
            if self._prefix_reuse_length > 0:
                reuse_length = min(self._prefix_reuse_length, max_reuse)
            else:
                reuse_length = self._rng.randint(1, max_reuse)
            prefix = self._last_keys[:reuse_length]
            remaining = count - reuse_length
            suffix = [self._random_key() for _ in range(remaining)]
            keys = prefix + suffix
        else:
            keys = [self._random_key() for _ in range(count)]

        self._last_keys = keys
        return keys
