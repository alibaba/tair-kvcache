import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, List, NamedTuple, Optional, Union, Iterator
from schedule_simulator.schedule_emulator.types import FakeRequest

# import torch
Req = FakeRequest


class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        # device_indices  :   Indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
        host_hit_length :   Length of the KV cache hit on the host, if applicable.
                            0 if HiCache is not enabled.
    """

    device_indices: List[int]
    last_device_node: Any
    last_host_node: Any
    host_hit_length: int = 0

    def __len__(self) -> int:
        return len(self.device_indices)


class BasePrefixCache(ABC):
    """Cache can be indexed by either rid or key."""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, key: Any, **kwargs) -> MatchResult:
        pass

    @abstractmethod
    def cache_finished_req(self, req: Req, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        pass

    @abstractmethod
    def evict(self, num_tokens: int):
        pass

    @abstractmethod
    def inc_lock_ref(self, node: Any):
        pass

    @abstractmethod
    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        pass

    def evictable_size(self):
        return 0

    def full_evictable_size(self):
        return 0

    def swa_evictable_size(self):
        return 0

    def protected_size(self):
        return 0

    def full_protected_size(self):
        return 0

    def swa_protected_size(self):
        return 0

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()

    def init_load_back(
        self,
        last_host_node: Any,
        host_hit_length: int,
    ):
        # ) -> Tuple[torch.Tensor, Any]:
        """
        Preparing KV cache loading from host to device.
        """
        raise NotImplementedError()

    def ready_to_load_host_cache(self) -> Any:
        """
        Notify the cache controller to start the KV cache loading
        """
        raise NotImplementedError()

    def check_hicache_events(self) -> Any:
        """
        Check HiCache related activities to update radix tree and synchronize across TP workers if needed
        """
        raise NotImplementedError()

    def take_events(self):
        return []


class RadixKey:
    def __init__(self, token_ids: List[int], extra_key: Optional[str] = None):
        # token ids sequence
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''})"


class TreeNode:
    counter = 0

    # 尝试与torch解耦
    def __init__(self, id: Optional[int] = None):
        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        # self.value: Optional[torch.Tensor] = None
        self.value: Optional[List[int]] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        # self.host_value: Optional[torch.Tensor] = None
        self.host_value: Optional[List[int]] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time
