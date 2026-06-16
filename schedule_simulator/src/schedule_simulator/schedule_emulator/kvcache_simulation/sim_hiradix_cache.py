import heapq
import time
from typing import List, Optional, Any
from schedule_simulator.schedule_emulator.types import (
    FakeRequest,
    RequestStage,
    PrefixCacheFetchResult,
    PlatformConfig,
    RequestCacheFetchStats,
    IterationCacheFetchStats,
)
from schedule_simulator.schedule_emulator.base import GlobalValues

from .kvcache_base_classes import (
    RadixKey,
    TreeNode,
    MatchResult,
)

from .pure_radix_tree import RadixCache
from .kvcache_utils import (
    ReqToTokenPoolHost,
    KVCachePool,
    SimHiCacheController,
)
from kunlun_commons.utils import get_logger

logger = get_logger("sim_hiradix_cache")


class SimHiRadixCache(RadixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPoolHost,
        token_to_kv_pool_allocator: KVCachePool,
        page_size: int,
        hicache_size: int,
        hicache_write_policy: str = "write_through",
        eviction_policy: str = "lru",
        prefetch_queue: List[FakeRequest] = None,
        hicache_storage_backend: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = "best_effort",
        storage_backend_extra_config: Any = None,
        load_back_threshold: int = 0,
        global_values: GlobalValues = None,
        kv_cache_space_per_token: int = None,
        platform_config: PlatformConfig = None,
        kvcm_block_size: Optional[int] = None,
        is_eagle: bool = False,
        enable_stats: bool = False,
    ):
        # 仿真流程中暂不考虑tp组相关问题

        self.enable_storage = hicache_storage_backend is not None

        # prefetch_threshold、prefetch_timeout_base、prefetch_timeout_per_ki_token等参数
        # 从传入的配置参数中读取
        (
            self.prefetch_threshold,
            self.prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
        ) = storage_backend_extra_config
        self.prefetch_timeout_per_page = (
            page_size / 1024 * prefetch_timeout_per_ki_token
        )

        self.prefetch_stop_policy = hicache_storage_prefetch_policy
        self.kv_cache_space_per_token = kv_cache_space_per_token
        self.kvcm_block_size = kvcm_block_size
        self.platform_config = platform_config
        self.global_values = global_values
        self.enable_stats = enable_stats

        self.kv_pool_host = KVCachePool(
            size=hicache_size,
            page_size=page_size,
        )
        # 仿真器中独立定义一套cache controller
        self.cache_controller = SimHiCacheController(
            mem_pool_device=token_to_kv_pool_allocator,
            mem_pool_host=self.kv_pool_host,
            page_size=page_size,
            hicache_write_policy=hicache_write_policy,
        )

        # 记录正在进行预取、备份到disk的节点
        self.prefetch_queue = prefetch_queue

        self.write_through_threshold = (
            1 if hicache_write_policy == "write_through" else 2
        )
        # 控制L2 -> L1的阈值，默认为3
        self.load_back_threshold = load_back_threshold
        # 统计请求信息
        self.fetch_request_stats: dict[FakeRequest, RequestCacheFetchStats] = {}
        self.fetch_iteration_stats: dict[int, IterationCacheFetchStats] = {}

        super().__init__(
            page_size,
            req_to_token_pool,
            token_to_kv_pool_allocator,
            disable=False,
            eviction_policy=eviction_policy,
            is_eagle=is_eagle,
        )

    def get_request_fetch_stats(self):
        return self.fetch_request_stats

    def get_iteration_fetch_stats(self):
        # TODO: 目前fetch_iteration_stats的更新还未实现
        return self.fetch_iteration_stats

    def reset(self):
        TreeNode.counter = 0
        super().reset()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def clear_storage_backend(self):
        # storage提供清空接口,预留
        return None

    def write_backup(self, node: TreeNode, write_back=False):
        # 写入操作，hbm -> dram
        if node.value is None:
            return 0
        host_indices_len = self.cache_controller.write(indices_len=len(node.value))

        # 写入失败则尝试驱逐，再写入
        if host_indices_len is None:
            self.evict_host(len(node.value))
            host_indices_len = self.cache_controller.write(indices_len=len(node.value))
        if host_indices_len is not None:
            node.host_value = node.value
            assert len(node.host_value) > 0

            # 此处由于write_backup由异步队列控制，因此原框架中需要加锁防止节点释放，
            # 但在仿真器中，由于假定操作为原子操作,且跳过了解锁操作所在op，因此不需要加锁
        else:
            raise MemoryError("ERROR! write backup failed!")
        return host_indices_len

    def write_backup_storage(self, req: FakeRequest, timestamp=0):
        # 用于实现L2 -> L3的备份，即dram -> disk, 由kvcm提供insert操作
        # 与原框架流程不同，此处由scheduler直接调用写入L3的操作
        self.cache_controller.write_storage(req.fill_ids[:-1], timestamp)

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        # 为当前输入节点增加一次命中计数，且当达到阈值(默认为1)时，执行write_backup；write_back策略跳过此步骤
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1

        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                self.write_backup(node)

    # writing check与loading check暂不模拟
    # writing check中通过write_back分为两个分支，true在evict时调用，强制处理完所有write_through任务，不写入L3
    # false分支在get_new_batch_prefill中调用；会调用write_backup_storage往L3写入数据，调用时间为L1到L2的writeback操作确认完成（出ack队列）时
    # loading check中遍历确认加载队列，检查每个事件是否已经完成
    # 需要注意的是，writing check与loading check中对于节点解锁的操作,当从确认队列中移除节点时，节点的锁计数器会减1

    def evictable_size(self):
        return self.evictable_size_

    def _collect_leaves_device(self):
        def is_leaf(node):
            if node.evicted:
                return False
            if node == self.root_node:
                return False
            if len(node.children) == 0:
                return True
            for child in node.children.values():
                if not child.evicted:
                    return False
            return True

        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if is_leaf(cur_node):
                ret_list.append(cur_node)
            else:
                for cur_child in cur_node.children.values():
                    if not cur_child.evicted:
                        stack.append(cur_child)
        return ret_list

    def evict(self, num_tokens: int):
        leaves = self._collect_leaves_device()
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                # 预留，先支持write_through
                if self.cache_controller.write_policy == "write_back":
                    num_evicted += self.write_backup(x, write_back=True)
                    write_back_nodes.append(x)
                else:
                    num_evicted += self._evict_regular(x)
            else:
                num_evicted += self._evict_backuped(x)

            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

        # 预留，先支持write_through
        if self.cache_controller.write_policy == "write_back":
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

        return num_evicted

    def _evict_backuped(self, node: TreeNode):
        # 因为不管理实际cache位置（value），替换成长度输入
        num_evicted = self.cache_controller.evict_device(len(node.value))
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # 因为不管理实际cache位置（value），替换成长度输入
        self.cache_controller.mem_pool_device.free(len(node.value))
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
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

            # 节点在L1处有缓存时不能从L2缓存中驱逐
            if not x.evicted:
                continue

            # 节点还有未完成的预取或L3写入操作时不能从L2缓存中驱逐
            if x.host_ref_counter > 0:
                continue

            num_evicted += self.cache_controller.evict_host(len(x.host_value))

            for k, v in x.parent.children.items():
                if v == x:
                    break
            del x.parent.children[k]

            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self,
        node: TreeNode,
    ) -> Optional[int]:
        nodes_to_load = []
        while node.evicted:
            assert node.backuped, (
                "No backup available on evicted nodes, should not happen"
            )
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancestor_node = node

        self.inc_lock_ref(ancestor_node)

        host_indices_len = sum(len(n.host_value) for n in nodes_to_load)
        if host_indices_len < self.load_back_threshold:
            # 小于加载最下小阈值则跳过
            self.dec_lock_ref(ancestor_node)
            return None

        # dram -> hbm
        device_indices_len = self.cache_controller.load(
            host_indices_len=host_indices_len,
        )
        if device_indices_len is None:
            self.evict(host_indices_len)
            device_indices_len = self.cache_controller.load(
                host_indices_len=host_indices_len,
            )
        self.dec_lock_ref(ancestor_node)
        if device_indices_len is None:
            # device 内存不足
            logger.warning("device out of memory, load_back failed")
            return None

        for node in nodes_to_load:
            # 因为仿真器中不记录具体内存的位置，value用token替代
            node.value = node.host_value
        self.evictable_size_ += device_indices_len

        return device_indices_len

    def init_load_back(
        self,
        last_node: TreeNode,
    ):
        if last_node.evicted:
            loading_len = self.load_back(last_node)
            if loading_len is not None:
                return loading_len, last_node

            while last_node.evicted:
                last_node = last_node.parent

        # 返回值改为长度信息
        return (
            0,
            last_node,
        )

    def can_terminate_prefetch(self, req: FakeRequest):
        can_terminate = True
        # TODO: 支持不同的预取策略
        if self.prefetch_stop_policy == "best_effort":
            return can_terminate
        elif self.prefetch_stop_policy == "wait_complete":
            completed = req.remain_prefetch_len == 0
            return completed

    def check_prefetch_progress(self, req: FakeRequest) -> bool:
        # 根据请求id检查预取进度，在原框架中的get_new_batch_prefill中调用
        if not self.can_terminate_prefetch(req):
            return False
        else:
            if req.is_prefetching():
                req.stage = RequestStage.READY
                self.prefetch_queue.remove(req)
                # 入预取队列时加锁防止host侧节点释放，预取完成时解锁
                req.last_host_node.release_host()
                if self.enable_stats:
                    req_stats = self.fetch_request_stats.get(req)
                    if req_stats is not None:
                        req_stats.prefetch_queue_end = self.global_values.clock
            return True

    def match_prefix(self, key: RadixKey, **kwargs):
        # 前缀匹配方法
        empty_value = []
        key.token_ids = self.key_convert_fn(key.token_ids)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if not value:
            value = empty_value

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )


    def query_prefix_length(self, key: RadixKey) -> int:
        """Read-only query returning device + host hit length."""
        key.token_ids = self.key_convert_fn(key.token_ids)
        if self.disable or len(key) == 0:
            return 0
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]
        if len(key) == 0:
            return 0
        device_len = self._query_prefix_length_helper(self.root_node, key)
        host_len = self._query_host_hit_length(self.root_node, key, device_len)
        return device_len + host_len

    def _query_host_hit_length(self, root: "TreeNode", key: RadixKey, device_matched: int) -> int:
        """Read-only walk past device match to count evicted host tokens."""
        if device_matched == 0:
            return 0
        node = root
        pos = 0
        child_key = self.get_child_key_fn(key)
        while pos < device_matched and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key[pos:])
            pos += prefix_len
            node = child
            if pos < device_matched and len(key) > pos:
                child_key = self.get_child_key_fn(key[pos:])
            else:
                break
        host_len = 0
        remaining = key[device_matched:]
        if len(remaining) > 0:
            next_key = self.get_child_key_fn(remaining)
            while next_key in node.children:
                child = node.children[next_key]
                if not child.evicted:
                    break
                prefix_len = self.key_match_fn(child.key, remaining)
                host_len += prefix_len
                if prefix_len < len(child.key):
                    break
                node = child
                remaining = remaining[prefix_len:]
                if len(remaining):
                    next_key = self.get_child_key_fn(remaining)
                else:
                    break
        return host_len

    def add_to_prefetch_queue(
        self,
        req: FakeRequest,
        timestamp=0,
    ):
        # 先进行一次前缀匹配，更新req内与prefix相关的成员变量
        match_result = self.match_prefix(
            key=RadixKey(token_ids=req.origin_input_ids, extra_key=None)
        )
        req.prefix_indices_len = len(match_result.device_indices)
        req.last_node = match_result.last_device_node
        req.last_host_node = match_result.last_host_node
        req.host_hit_len = match_result.host_hit_length
        req.last_matched_prefix_len = len(match_result.device_indices)

        # 根据匹配结果，调用kvcm接口确定预取长度
        cur_matched_len = req.prefix_indices_len + req.host_hit_len
        disk_total_hit_len = self.cache_controller.get_prefetch_length(
            req.origin_input_ids,
            timestamp=timestamp,
            # cur_matched_len
        )

        logger.debug(
            f"req[{req.id}]: in add_to_prefetch_queue, disk_total_hit_len is {disk_total_hit_len}, "
            f"prefetch_length is {disk_total_hit_len - cur_matched_len}, {req.prefix_indices_len=}, {req.host_hit_len=}"
        )

        # 记录预取信息，可能为空
        if self.enable_stats:
            self.fetch_request_stats[req] = RequestCacheFetchStats(
                req_id=req.id,
                prefetch_queue_start=max(self.global_values.clock, req.last_event_time),
            )

        if disk_total_hit_len - cur_matched_len >= self.prefetch_threshold:
            # L3 层级有更多命中的前缀，且满足阈值限制，执行预取，通过更新req的成员变量来记录预取长度
            prefetch_length = disk_total_hit_len - cur_matched_len
            prefetch_length = prefetch_length - (prefetch_length % self.page_size)
            # 1.申请空间
            req.last_host_node.protect_host()
            if self.cache_controller.mem_pool_host.available_size < prefetch_length:
                self.evict_host(prefetch_length)
                if self.cache_controller.mem_pool_host.available_size < prefetch_length:
                    # 空间不足且无法释放足够空间，则放弃预取
                    req.last_host_node.release_host()
                    logger.warning(
                        f"not enough host memory for prefetch! try to evict {prefetch_length} host memory, but failed."
                    )
                    self.pretty_print()
                    return
            # 预取前为请求申请预取长度的内存
            self.cache_controller.mem_pool_host.alloc(prefetch_length)

            # 2.更新req对应成员变量，包括加入预取队列的时间
            req.remain_prefetch_len = prefetch_length
            req.disk_hit_len = prefetch_length
            req.prefetch_start_record = self.global_values.clock

            # 3.加入预取队列
            req.stage = RequestStage.PREFETCHING
            self.prefetch_queue.append(req)

    def estimate_prefetch_from_storage(self, req: FakeRequest, max_time: float):
        if not req.is_idle() and not req.is_prefetching():
            return PrefixCacheFetchResult()

        if req.remain_prefetch_len <= 0 or max_time <= 0:
            return PrefixCacheFetchResult()

        if self.kvcm_block_size:
            max_time = max_time / self.kvcm_block_size

        latency_disk_to_host = (
            req.remain_prefetch_len * self.kv_cache_space_per_token
        ) / self.platform_config.disk_read_bandwidth
        if latency_disk_to_host <= max_time:
            retrieved_tokens = req.remain_prefetch_len
        else:
            retrieved_tokens = (
                max_time * self.platform_config.disk_read_bandwidth
            ) // self.kv_cache_space_per_token
            latency_disk_to_host = max_time

        return PrefixCacheFetchResult(
            latency_disk_to_host=latency_disk_to_host,
            fetched_tokens=int(retrieved_tokens),
        )

    def prefetch_from_storage(self, req: FakeRequest, max_time: float):
        if req.remain_prefetch_len <= 0:
            # self.prefetch_queue.remove(req)
            return PrefixCacheFetchResult()

        fetch_result = self.estimate_prefetch_from_storage(req, max_time)
        if fetch_result.fetched_tokens > 0:
            # 获取到预取的长度后，将预取到的token插入前缀树中,插入节点为req的最后一个host节点
            # 需要注意预取是有可能分多次进行的，每次预取一部分，并将预取后的长度更新在请求的remain_prefetch_len中
            prefetched_len = req.disk_hit_len - req.remain_prefetch_len
            prefetched_token = (req.origin_input_ids + req.output_ids)[
                req.prefix_indices_len + req.host_hit_len : req.prefix_indices_len
                + req.host_hit_len
                + prefetched_len
                + fetch_result.fetched_tokens
            ]
            self._insert_helper_host(
                req.last_host_node, RadixKey(token_ids=prefetched_token, extra_key=None)
            )
            req.remain_prefetch_len -= fetch_result.fetched_tokens

            # 更新预取记录
            if self.enable_stats:
                self.fetch_request_stats[
                    req
                ].num_token_from_disk += fetch_result.fetched_tokens
                self.fetch_request_stats[
                    req
                ].latency_disk_to_host += fetch_result.latency_disk_to_host
                self.fetch_request_stats[req].prefetch_queue_end += (
                    self.global_values.clock + fetch_result.latency_disk_to_host
                )
        return fetch_result

    def _insert_helper_host(self, node: TreeNode, key: RadixKey, host_value=None):
        # 插入host侧节点，即新节点只有host value，前面匹配的dev侧或host侧都可以
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0
        if host_value is None:
            host_value = key.token_ids

        child_key = self.get_child_key_fn(key)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.host_value = host_value
            node.children[child_key] = new_node
        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        # 在前缀树中匹配给定键的最长前缀
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.extend(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.extend(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]

        if child.hash_value:
            new_node.hash_value = child.hash_value[: split_len // self.page_size]
            child.hash_value = child.hash_value[split_len // self.page_size :]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def insert(self, key: RadixKey, value=None, chunked=False):
        key.token_ids = self.key_convert_fn(key.token_ids)

        if len(key) == 0:
            return 0

        # 因为不涉及实际存储，用token值来代替value
        if value is None:
            value = key.token_ids

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted:
                    # 如果节点被逐出，则更改value值，发生在kv重新计算的时候
                    node.value = value[:prefix_len]
                    self.evictable_size_ += len(node.value)
                else:
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # 部分匹配则分裂节点
                new_node = self._split_node(node.key, node, prefix_len)
                if new_node.evicted:
                    new_node.value = value[:prefix_len]
                    self.evictable_size_ += len(new_node.value)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value) if value else 0

            if self.enable_storage:
                last_hash = node.get_last_hash_value()
                assert (node == self.root_node) or (last_hash is not None), (
                    "Parent node must have a hash value with storage enabled"
                )
                new_node.hash_value = []
                for idx in range(0, len(key), self.page_size):
                    new_node.hash_value.append(
                        self.cache_controller.get_hash_str(
                            key.token_ids[idx : idx + self.page_size],
                            prior_hash=last_hash,
                        )
                    )
                    last_hash = new_node.hash_value[-1]

            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked)
        return total_prefix_length
