from dataclasses import dataclass, field
from typing import Optional
from schedule_simulator.schedule_emulator.types import (
    FakeRequest,
    PlatformConfig,
    PrefixCacheMatchResult,
    PrefixCacheFetchResult,
    RequestCacheFetchStats,
    IterationCacheFetchStats,
    RequestStage,
)
from schedule_simulator.schedule_emulator.prefix_cache import PrefixCache
from schedule_simulator.schedule_emulator.base import GlobalValues


@dataclass
class HierarchicalHitRecord:
    req_id: int
    timestamp: float = 0
    engine_hit: int = 0
    peer_hit: int = 0
    pool_hit: int = 0
    total_hit: int = 0
    input_length: int = 0
    num_blocks: int = 0


@dataclass
class HierarchicalWriteRecord:
    req_id: int
    timestamp: float = 0
    write_blocks: int = 0


class HierarchicalCacheAdapter(PrefixCache):
    """
    Adapter bridging PrefixCache interface to HierarchicalReplayManager.
    Each adapter instance maps to one engine_instance_id.
    Multiple adapters share the same HierarchicalReplayManager.

    Hit mapping:
      engine_hit_length  -> device_cache_hit_length (no transfer)
      peer_hit_length    -> host_cache_hit_length   (peer_read_bandwidth)
      pool_hit_length    -> disk_cache_hit_length   (disk_read_bandwidth)
    """

    def __init__(
        self,
        manager,
        engine_instance_id: str,
        platform_config: PlatformConfig,
        kv_cache_space_per_token: int,
        page_size: int,
        global_values: GlobalValues,
        prefetch_stop_policy: str = "best_effort",
        read_query_type: str = "prefix_match",
        enable_stats: bool = False,
    ):
        super().__init__(
            platform_config,
            kv_cache_space_per_token,
            page_size,
            global_values,
            enable_stats,
        )
        self.manager = manager
        self.engine_id = engine_instance_id
        self.prefetch_stop_policy = prefetch_stop_policy
        self.read_query_type = read_query_type
        self.prefetch_timeout = 3
        self.prefetch_threshold = 256

        self.total_engine_hit_blocks: int = 0
        self.total_peer_hit_blocks: int = 0
        self.total_pool_hit_blocks: int = 0

        self.read_records: list[HierarchicalHitRecord] = []
        self.write_records: list[HierarchicalWriteRecord] = []

    def _req_to_block_ids(self, req: FakeRequest) -> list:
        if req.origin_input_ids is not None:
            return req.origin_input_ids
        return [req.id * 1000000 + i for i in range(req.input_token_length)]

    def _req_to_write_block_ids(self, req: FakeRequest) -> list:
        if req.origin_input_ids is not None:
            return list(req.origin_input_ids)
        return [req.id * 1000000 + i for i in range(req.input_token_length)]

    def add_to_prefetch_queue(self, req: FakeRequest):
        block_ids = self._req_to_block_ids(req)
        timestamp_ns = int(self.global_values.clock * 1e9)
        
        # Optimizer 验证: len(block_ids) <= input_tokens / block_size
        # 需要丢弃不完整的 tail blocks
        block_size = self.page_size  # 从配置中获取
        max_full_blocks = req.input_token_length // block_size
        # 如果 input_token_length < block_size，至少查询 1 个 block（向后兼容）
        if max_full_blocks == 0 and block_ids:
            max_full_blocks = 1
        full_block_ids = block_ids[:max_full_blocks]
        
        res = self.manager.GetCacheLocation(
            self.engine_id,
            str(req.id),
            timestamp_ns,
            full_block_ids,
            req.input_token_length,
            self.read_query_type,
        )
        req.device_cache_hit_length = res.engine_hit_length
        req.host_cache_hit_length = res.peer_hit_length
        req.disk_cache_hit_length = res.storage_pool_hit_length

        self.total_engine_hit_blocks += res.engine_hit_length
        self.total_peer_hit_blocks += res.peer_hit_length
        self.total_pool_hit_blocks += res.storage_pool_hit_length

        self.read_records.append(HierarchicalHitRecord(
            req_id=req.id,
            timestamp=self.global_values.clock,
            engine_hit=res.engine_hit_length,
            peer_hit=res.peer_hit_length,
            pool_hit=res.storage_pool_hit_length,
            total_hit=res.total_hit_length,
            input_length=req.input_token_length,
            num_blocks=len(full_block_ids),
        ))

        super().add_to_prefetch_queue(req)

    def match_prefix(self, req: FakeRequest) -> PrefixCacheMatchResult:
        match_result = self.cache_controller.get(req)
        if match_result is not None:
            return match_result

        # GetCacheLocation returns block counts; convert to token counts
        # so that downstream (time predictor, uncached calculation) uses correct units.
        page_size = self.page_size if self.page_size else 1
        match_result = PrefixCacheMatchResult(
            disk_hit_length=req.disk_cache_hit_length * page_size,
            host_hit_length=req.host_cache_hit_length * page_size,
            device_hit_length=req.device_cache_hit_length * page_size,
        )
        self.cache_controller[req] = match_result
        return match_result

    def can_terminate_prefetch(self, req: FakeRequest) -> bool:
        if self.prefetch_stop_policy == "best_effort":
            return True

        matched = self.match_prefix(req)
        completed = matched.disk_hit_length == 0

        if self.prefetch_stop_policy == "wait_complete":
            return completed
        elif self.prefetch_stop_policy == "timeout":
            elapsed = self.global_values.last_batch_start - self.prefetch_start_record.get(req, 0)
            return completed or elapsed > self.prefetch_timeout

        return True

    def check_prefetch_progress(self, req: FakeRequest) -> bool:
        if not self.can_terminate_prefetch(req):
            return False
        self.terminate_prefetch(req)
        return True

    def estimate_prefetch_from_storage(
        self, req: FakeRequest, max_time: float
    ) -> PrefixCacheFetchResult:
        if not req.is_idle() and not req.is_prefetching():
            return PrefixCacheFetchResult()

        matched = self.cache_controller.get(req)
        if matched is None:
            matched = self.match_prefix(req)

        if matched.disk_hit_length == 0 or max_time <= 0:
            return PrefixCacheFetchResult()

        bw = self.platform_config.disk_read_bandwidth
        if bw is None or bw == 0:
            return PrefixCacheFetchResult()

        latency = (matched.disk_hit_length * self.kv_cache_space_per_token) / bw
        if latency <= max_time:
            fetched = matched.disk_hit_length
        else:
            fetched = int(max_time * bw / self.kv_cache_space_per_token)
            latency = max_time

        return PrefixCacheFetchResult(
            latency_disk_to_host=latency,
            fetched_tokens=fetched,
        )

    def prefetch_from_storage(
        self, req: FakeRequest, max_time: float
    ) -> PrefixCacheFetchResult:
        fetch_result = self.estimate_prefetch_from_storage(req, max_time)

        matched = self.cache_controller.get(req)
        if matched is None:
            matched = self.match_prefix(req)

        new_input_tokens = req.input_token_length - (
            matched.device_hit_length + matched.host_hit_length
        )
        prefetch_length = new_input_tokens - (new_input_tokens % self.page_size)
        if prefetch_length < self.prefetch_threshold:
            req.stage = RequestStage.READY
            matched.disk_hit_length = 0
            if req in self.prefetch_queue:
                self.prefetch_queue.remove(req)
            self.prefetch_start_record.pop(req, None)
            return PrefixCacheFetchResult()

        if fetch_result.fetched_tokens > 0:
            matched.disk_hit_length -= fetch_result.fetched_tokens
            matched.host_hit_length += fetch_result.fetched_tokens

        if matched.disk_hit_length <= 0:
            matched.disk_hit_length = 0
            req.stage = RequestStage.READY
            if req in self.prefetch_queue:
                self.prefetch_queue.remove(req)
            self.prefetch_start_record.pop(req, None)

        if self.enable_stats:
            req_stats = self.fetch_request_stats.get(req)
            if req_stats is not None:
                req_stats.latency_disk_to_host += fetch_result.latency_disk_to_host
                req_stats.num_token_from_disk += fetch_result.fetched_tokens

        return fetch_result

    def estimate_on_board_from_host(
        self, req: FakeRequest
    ) -> PrefixCacheFetchResult:
        matched = self.cache_controller.get(req)
        if matched is None:
            matched = self.match_prefix(req)

        if matched.host_hit_length == 0:
            return PrefixCacheFetchResult()

        bw = self.platform_config.peer_read_bandwidth
        if bw is None:
            bw = self.platform_config.memory_read_bandwidth
        if bw is None or bw == 0:
            return PrefixCacheFetchResult(fetched_tokens=matched.host_hit_length)

        latency = (matched.host_hit_length * self.kv_cache_space_per_token) / bw
        return PrefixCacheFetchResult(
            latency_host_to_device=latency,
            fetched_tokens=matched.host_hit_length,
        )

    def on_board_from_host(self, req: FakeRequest) -> PrefixCacheFetchResult:
        result = self.estimate_on_board_from_host(req)

        matched = self.cache_controller.get(req)
        if matched is None:
            matched = self.match_prefix(req)

        if result.fetched_tokens > 0:
            matched.device_hit_length += result.fetched_tokens
            matched.host_hit_length -= result.fetched_tokens

        if self.enable_stats:
            self.total_input_length += req.input_token_length
            self.total_hit_kv_length += matched.device_hit_length
            req_stats = self.fetch_request_stats.get(req)
            if req_stats is not None:
                req_stats.latency_host_to_device = result.latency_host_to_device
                req_stats.num_token_from_host = result.fetched_tokens

        return result

    def on_request_complete(self, req: FakeRequest, timestamp: float):
        block_ids = self._req_to_write_block_ids(req)
        timestamp_ns = int(timestamp * 1e9)
        
        # 与 add_to_prefetch_queue 保持一致: 只写完整的 blocks
        block_size = self.page_size
        max_full_blocks = req.input_token_length // block_size
        # 如果 input_token_length < block_size，至少写 1 个 block（向后兼容）
        if max_full_blocks == 0 and block_ids:
            max_full_blocks = 1
        full_block_ids = block_ids[:max_full_blocks]
        
        self.manager.WriteCache(
            self.engine_id,
            str(req.id),
            timestamp_ns,
            full_block_ids,
        )
        self.write_records.append(HierarchicalWriteRecord(
            req_id=req.id,
            timestamp=timestamp,
            write_blocks=len(full_block_ids),
        ))


    def query_prefix_length(self, key, input_token_length: int) -> int:
        """Query prefix match length from Optimizer for this engine (lightweight PrefixMatchCount)."""
        block_ids = list(key.token_ids) if hasattr(key, "token_ids") else list(key)
        if not block_ids:
            return 0
        # 只查询完整的 blocks
        block_size = self.page_size
        max_full_blocks = input_token_length // block_size
        # 如果 input_token_length < block_size，至少查询 1 个 block（向后兼容）
        if max_full_blocks == 0 and block_ids:
            max_full_blocks = 1
        full_block_ids = block_ids[:max_full_blocks]
        if not full_block_ids:
            return 0
        timestamp_ns = int(self.global_values.clock * 1e9)
        res = self.manager.ChooseBestEngine(full_block_ids, timestamp_ns)
        if res.engine_instance_id == self.engine_id:
            return res.hit_count
        return 0

    def choose_best_engine(self, block_ids: list, timestamp_ns: int, input_token_length: int):
        """Find the engine with the best prefix match in one C++ call."""
        # 查询所有 blocks（与 add_to_prefetch_queue 保持一致）
        if not block_ids:
            return None
        return self.manager.ChooseBestEngine(block_ids, timestamp_ns)

    def get_hierarchical_metrics(self) -> dict:
        total_input = sum(r.input_length for r in self.read_records)
        total_blocks = sum(r.num_blocks for r in self.read_records)
        total_hit = self.total_engine_hit_blocks + self.total_peer_hit_blocks + self.total_pool_hit_blocks
        return {
            "total_engine_hit_blocks": self.total_engine_hit_blocks,
            "total_peer_hit_blocks": self.total_peer_hit_blocks,
            "total_pool_hit_blocks": self.total_pool_hit_blocks,
            "engine_hit_block_ratio": self.total_engine_hit_blocks / max(total_input, 1),
            "peer_hit_block_ratio": self.total_peer_hit_blocks / max(total_input, 1),
            "pool_hit_block_ratio": self.total_pool_hit_blocks / max(total_input, 1),
            "block_hit_ratio": total_hit / max(total_input, 1),
            "total_blocks_queried": total_blocks,
            "total_blocks_hit": total_hit,
            "block_hit_ratio": total_hit / max(total_blocks, 1),
            "num_reads": len(self.read_records),
            "num_writes": len(self.write_records),
        }
