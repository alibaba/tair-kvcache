from __future__ import annotations
import heapq
import time
from functools import partial
from .evict_policy import EvictionStrategy, LFUStrategy, LRUStrategy
from schedule_simulator.schedule_emulator.types import FakeRequest
from .kvcache_utils import (
    ReqToTokenPoolHost,
    KVCachePool,
)
from .kvcache_base_classes import BasePrefixCache, MatchResult, RadixKey, TreeNode

Req = FakeRequest


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    if key0.extra_key != key1.extra_key:
        raise ValueError(
            f"_key_match should be run on the same extra key, but got key0.extra_key={key0.extra_key} != key1.extra_key={key1.extra_key}"
        )


def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size

    return i


def get_child_key(key: RadixKey, page_size: int = 1):
    if page_size == 1:
        plain_key = key.token_ids[0]
    else:
        plain_key = tuple(key.token_ids[:page_size])
    if key.extra_key is None:
        return plain_key
    else:
        return (key.extra_key, plain_key)


class RadixCache(BasePrefixCache):
    def __init__(
        self,
        page_size: int,
        req_to_token_pool: ReqToTokenPoolHost = None,
        kv_pool: KVCachePool = None,
        disable: bool = False,
        enable_kv_cache_events: bool = False,
        eviction_policy: str = "lru",
        is_eagle: bool = False,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.kv_pool = kv_pool
        self.page_size = page_size
        self.disable = disable
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue = []
        self.is_eagle = is_eagle

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=page_size)

        self.key_convert_fn = lambda key: key

        if eviction_policy.lower() == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif eviction_policy.lower() == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        else:
            raise ValueError(
                f"Unknown eviction policy: {eviction_policy}. Supported policies: 'lru', 'lfu'."
            )
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        key.token_ids = self.key_convert_fn(key.token_ids)

        def empty_match_result():
            return MatchResult(
                [],
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        if self.disable or len(key) == 0:
            return empty_match_result()

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        if len(key) == 0:
            return empty_match_result()

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if not value:
            value = []
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )


    def query_prefix_length(self, key: RadixKey) -> int:
        """Read-only prefix length query. Does NOT split nodes or update access times."""
        key.token_ids = self.key_convert_fn(key.token_ids)
        if self.disable or len(key) == 0:
            return 0
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]
        if len(key) == 0:
            return 0
        return self._query_prefix_length_helper(self.root_node, key)

    def _query_prefix_length_helper(self, node: TreeNode, key: RadixKey) -> int:
        """Traverse tree read-only, counting matched prefix length without splitting."""
        child_key = self.get_child_key_fn(key)
        matched = 0
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            matched += prefix_len
            if prefix_len < len(child.key):
                break
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)
        return matched

    def insert(self, key: RadixKey, value=None, chunked=False):
        if self.disable:
            return 0

        key.token_ids = self.key_convert_fn(key.token_ids)

        if value is None:
            value = key.token_ids

        return self._insert_helper(self.root_node, key, value)

    # 通过简单的KVCachePool类来模拟KV缓存池
    def cache_finished_req(self, req: Req):
        """Cache request when it finishes."""
        if self.disable:
            occ_kv_length = self.req_to_token_pool.req_length[req.req_pool_idx]
            self.kv_pool.free(occ_kv_length)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = req.fill_ids[:-1]
        all_token_len = len(token_ids)
        actual_kv_len = all_token_len - 1 if self.is_eagle else all_token_len
        occ_kv_length = self.req_to_token_pool.req_length[
            req.req_pool_idx
        ]  # 长度是否可以直接从req中获取？

        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
            # 此处释放的内存为 当前推理结束req占用的 且后续不用的（可能因为尚未使用或没对齐page_size）
            self.kv_pool.free(occ_kv_length - page_aligned_len)
        else:
            page_aligned_len = actual_kv_len
            if self.is_eagle:
                self.kv_pool.free(occ_kv_length - page_aligned_len)

        page_aligned_token_len = (
            page_aligned_len + 1 if self.is_eagle else page_aligned_len
        )
        old_prefix_len = req.prefix_indices_len

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(
            RadixKey(token_ids[:page_aligned_token_len], req.extra_key),
            # page_aligned_kv_indices,
            None,
        )
        # 如果有新增匹配，说明这部分前缀在缓存中也命中了，不需要重复存储，因此这部分kvcache可以释放
        self.kv_pool.free(new_prefix_len - old_prefix_len)

        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        all_token_len = len(token_ids)
        actual_kv_len = all_token_len - 1 if self.is_eagle else all_token_len
        occ_kv_length = self.req_to_token_pool.req_length[
            req.req_pool_idx
        ]  # 长度是否可以直接从req中获取？

        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
        else:
            page_aligned_len = actual_kv_len

        # For EAGLE, the page_aligned_len is for the bigram key, the normal key len should +1
        page_aligned_token_len = (
            page_aligned_len + 1 if self.is_eagle else page_aligned_len
        )
        page_aligned_token_ids = token_ids[:page_aligned_token_len]

        # 旧的匹配前缀是否需要在请求内维护
        old_prefix_len = req.prefix_indices_len

        new_prefix_len = self.insert(
            RadixKey(page_aligned_token_ids, req.extra_key),
            None,
            chunked=chunked,
        )
        # 如果有新增匹配，说明这部分前缀在缓存中也命中了，不需要重复存储，因此这部分kvcache可以释放
        self.kv_pool.free(new_prefix_len - old_prefix_len)

        new_indices, new_last_node, _, _ = self.match_prefix(
            RadixKey(token_ids=page_aligned_token_ids, extra_key=req.extra_key)
        )
        # 根据前缀树中的匹配结果，更新req_pool中的对应下标（行）中的指定位置（列）的kvcache地址
        self.req_to_token_pool.write(req.req_pool_idx, len(new_indices))
        req.last_matched_prefix_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        if self.page_size != 1:
            req.prefix_indices_len = occ_kv_length
        else:
            req.prefix_indices_len = len(new_indices)
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")
        if self.kv_pool:
            print(
                f"# memory used: available_size:{self.kv_pool.available_size}, evictable_size:{self.kv_pool.evictable_size}"
            )

    def total_size(self):
        return self._total_size_helper()

    def evict(self, num_tokens: int):
        if self.disable:
            return

        leaves = self._collect_leaves()
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            if self.kv_pool:
                self.kv_pool.free(len(x.value))
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        result = []
        for value in values:
            result.extend(value)
        return result

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.extend(new_node.value)
                node = new_node
                break
            else:
                value.extend(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        return new_node

    def _insert_helper(self, node: TreeNode, key: RadixKey, value):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        global_dev_len = 0
        global_host_len = 0
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                f"key={current_node.key.token_ids[:5]}",
                f"lock_ref={current_node.lock_ref}",
                f"host_ref_counter={current_node.host_ref_counter}",
                f" dev_len={len(current_node.value) if current_node.value else 0}",
                f" host_len={len(current_node.host_value) if current_node.host_value else 0}",
            )
            global_dev_len += len(current_node.value) if current_node.value else 0
            global_host_len += (
                len(current_node.host_value) if current_node.host_value else 0
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(child.key), (
                    f"{key=}, {self.get_child_key_fn(child.key)=}"
                )
        print(
            f"Summarize all node info: global dev len is {global_dev_len}, global host len is {global_host_len}"
        )
        if hasattr(self, "cache_controller") and self.cache_controller:
            self.cache_controller.print_mem_stats()

    def get_mem_usage(self):
        node = self.root_node
        stack = [(node)]
        global_dev_len = 0
        global_host_len = 0
        while stack:
            current_node = stack.pop()
            global_dev_len += len(current_node.value) if current_node.value else 0
            global_host_len += (
                len(current_node.host_value) if current_node.host_value else 0
            )
            for child in current_node.children.values():
                stack.append((child))
        return global_dev_len, global_host_len

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list
