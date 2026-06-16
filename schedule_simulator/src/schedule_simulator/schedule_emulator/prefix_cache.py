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
from schedule_simulator.schedule_emulator.base import GlobalValues


class PrefixCache:
    def __init__(
        self,
        platform_config: PlatformConfig,
        kv_cache_space_per_token: int,
        page_size: int,
        global_values: GlobalValues,
        enable_stats: bool = False,
    ):
        self.platform_config = platform_config
        self.kv_cache_space_per_token = kv_cache_space_per_token
        self.page_size = page_size
        self.global_values: GlobalValues = global_values
        self.enable_stats: bool = enable_stats

        self.prefetch_queue: list[FakeRequest] = []
        self.cache_controller: dict[FakeRequest, PrefixCacheMatchResult] = {}
        self.host_buffer_tokens: float = 0
        self.host_buffer_req_tokens: dict[int, int] = {}
        self.fetch_request_stats: dict[FakeRequest, RequestCacheFetchStats] = {}
        self.fetch_iteration_stats: dict[int, IterationCacheFetchStats] = {}
        # kv cache stats
        self.total_input_length: int = 0
        self.total_hit_kv_length: int = (
            0  # The final length of the KV cache that was hit on the device.
        )
        # Record the prefetching start time of every request.
        self.prefetch_start_record: dict[FakeRequest, float] = {}

    def add_to_prefetch_queue(self, req: FakeRequest):
        req.stage = RequestStage.PREFETCHING
        self.prefetch_queue.append(req)
        self.prefetch_start_record[req] = self.global_values.clock

        if self.enable_stats:
            self.fetch_request_stats[req] = RequestCacheFetchStats(
                req_id=req.id,
                prefetch_queue_start=max(self.global_values.clock, req.last_event_time),
            )

    def get_request_fetch_stats(self) -> dict[FakeRequest, RequestCacheFetchStats]:
        return self.fetch_request_stats

    def get_iteration_fetch_stats(self) -> dict[int, IterationCacheFetchStats]:
        return self.fetch_iteration_stats

    def drop_match_result(self, req: FakeRequest):
        if req in self.cache_controller:
            self.cache_controller.pop(req)

    def reset(self):
        self.cache_controller.clear()
        self.prefetch_queue.clear()

        self.fetch_request_stats.clear()
        self.fetch_iteration_stats.clear()

    def match_prefix(self, req: FakeRequest) -> PrefixCacheMatchResult:
        # Do nothing.
        return PrefixCacheMatchResult()

    def estimate_prefetch_from_storage(
        self, req: FakeRequest, max_time: float
    ) -> PrefixCacheFetchResult:
        return PrefixCacheFetchResult()

    def prefetch_from_storage(
        self, req: FakeRequest, max_time: float
    ) -> PrefixCacheFetchResult:
        # Do nothing.
        return PrefixCacheFetchResult()

    def check_prefetch_progress(self, req: FakeRequest) -> bool:
        req.stage = RequestStage.READY
        return True

    def terminate_prefetch(self, req: FakeRequest):
        if req.is_prefetching():
            req.stage = RequestStage.READY
            self.prefetch_queue.remove(req)
            if self.enable_stats:
                req_stats = self.fetch_request_stats.get(req)
                if req_stats is not None:
                    req_stats.prefetch_queue_end = self.global_values.clock

    def estimate_on_board_from_host(self, req: FakeRequest):
        return PrefixCacheFetchResult()

    def on_board_from_host(self, req: FakeRequest):
        return PrefixCacheFetchResult()



    def on_request_complete(self, req: FakeRequest, timestamp: float):
        pass

class HiRadixCache(PrefixCache):
    def __init__(
        self,
        platform_config: PlatformConfig,
        kv_cache_space_per_token: int,
        page_size: int,
        global_values: GlobalValues,
        hicache_storage_prefetch_policy: Optional[str] = "best_effort",
        enable_stats: bool = False,
    ):
        super().__init__(
            platform_config,
            kv_cache_space_per_token,
            page_size,
            global_values,
            enable_stats,
        )
        if (
            self.platform_config.disk_read_bandwidth is None
            or self.platform_config.memory_read_bandwidth is None
        ):
            raise ValueError("Fail to initialize the hierarchical storage.")
        # https://github.com/sgl-project/sglang/blob/v0.5.2rc2/python/sglang/srt/mem_cache/hiradix_cache.py#L77
        self.prefetch_threshold = 256
        self.prefetch_timeout = 3  # seconds
        self.prefetch_stop_policy = hicache_storage_prefetch_policy

    def match_prefix(self, req: FakeRequest) -> PrefixCacheMatchResult:
        match_result = self.cache_controller.get(req)
        if match_result is not None:
            return match_result

        # TODO: Match the prefix cache from hbm and host
        match_result = PrefixCacheMatchResult(
            disk_hit_length=req.disk_cache_hit_length,
            host_hit_length=req.host_cache_hit_length,
            device_hit_length=req.device_cache_hit_length,
        )

        self.cache_controller[req] = match_result

        return match_result

    def check_prefetch_progress(self, req: FakeRequest) -> bool:
        if not self.can_terminate_prefetch(req):
            return False
        else:
            self.terminate_prefetch(req)
            return True

    def can_terminate_prefetch(self, req: FakeRequest):
        # REF: https://github.com/sgl-project/sglang/blob/v0.5.2rc2/python/sglang/srt/mem_cache/hiradix_cache.py#L430
        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        matched = self.match_prefix(req)
        completed = matched.disk_hit_length == 0

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or (
                self.global_values.last_batch_start - self.prefetch_start_record[req]
                > self.prefetch_timeout
            )
        else:
            # unknown prefetch stop policy, just return True
            return True

        return can_terminate

    def estimate_prefetch_from_storage(self, req, max_time):
        if not req.is_idle() and not req.is_prefetching():
            return PrefixCacheFetchResult()

        matched = self.cache_controller.get(req)
        if matched is None:
            matched = self.match_prefix(req)

        if matched.disk_hit_length == 0:
            return PrefixCacheFetchResult()

        if max_time <= 0:
            return PrefixCacheFetchResult()

        latency_disk_to_host = (
            matched.disk_hit_length * self.kv_cache_space_per_token
        ) / self.platform_config.disk_read_bandwidth
        if latency_disk_to_host <= max_time:
            retrieved_tokens = matched.disk_hit_length
        else:
            retrieved_tokens = (
                max_time * self.platform_config.disk_read_bandwidth
            ) // self.kv_cache_space_per_token
            latency_disk_to_host = max_time

        return PrefixCacheFetchResult(
            latency_disk_to_host=latency_disk_to_host,
            fetched_tokens=int(retrieved_tokens),
        )

    def prefetch_from_storage(
        self, req: FakeRequest, max_time: float
    ) -> PrefixCacheFetchResult:
        fetch_result = self.estimate_prefetch_from_storage(req, max_time)

        matched = self.cache_controller.get(req)
        if matched is None:
            matched = self.match_prefix(req)

        # REF: https://github.com/sgl-project/sglang/blob/v0.5.2rc2/python/sglang/srt/mem_cache/hiradix_cache.py#L563
        # In the real implementation of the framework,
        # it doesn't know the actual disk hit length until it connects to the remote controller.
        # Therefore, the prefetch process is determined based on the new input length.
        new_input_tokens = req.input_token_length - (
            matched.device_hit_length + matched.host_hit_length
        )
        prefetch_length = new_input_tokens - (new_input_tokens % self.page_size)
        if prefetch_length < self.prefetch_threshold:
            # TODO: add the prefetch rate limitation
            req.stage = RequestStage.READY
            matched.disk_hit_length = 0
            self.prefetch_queue.remove(req)
            self.prefetch_start_record.pop(req)
            return PrefixCacheFetchResult()

        if self.enable_stats:
            if req.id not in self.host_buffer_req_tokens:
                self.host_buffer_req_tokens[req.id] = matched.disk_hit_length
                self.host_buffer_tokens += matched.disk_hit_length
            req_stats = self.fetch_request_stats.get(req)
            if req_stats is not None:
                req_stats.latency_disk_to_host += fetch_result.latency_disk_to_host
                req_stats.num_token_from_disk += fetch_result.fetched_tokens
            # The prefetch process happened in the second-to-last execution.
            iter_stats = self.fetch_iteration_stats.get(
                self.global_values.iteration - 2
            )
            if iter_stats is None:
                iter_stats = IterationCacheFetchStats(
                    iter=self.global_values.iteration - 2,
                    timestamp=self.global_values.clock
                    - max_time,  # FIXME: Get the timestamp of `iteration - 2`,
                    last_batch_run_time=self.global_values.last_batch_run_time,
                )
                self.fetch_iteration_stats[self.global_values.iteration - 2] = (
                    iter_stats
                )
            iter_stats.latency_disk_to_host += fetch_result.latency_disk_to_host
            iter_stats.num_token_from_disk += fetch_result.fetched_tokens
            iter_stats.host_buffer_size_gb = (
                self.kv_cache_space_per_token * self.host_buffer_tokens
            ) / (1 << 30)
            iter_stats.mean_kv_reuse_ratio = (
                self.total_hit_kv_length / self.total_input_length
            )

        # move the kv cache from disk to the host.
        matched.disk_hit_length -= fetch_result.fetched_tokens
        matched.host_hit_length += fetch_result.fetched_tokens

        if matched.disk_hit_length == 0:
            req.stage = RequestStage.READY
            self.prefetch_queue.remove(req)
            self.prefetch_start_record.pop(req)

            if self.enable_stats:
                if req_stats is not None:
                    req_stats.prefetch_queue_end = (
                        self.global_values.clock + fetch_result.latency_disk_to_host
                    )

        return fetch_result

    def estimate_on_board_from_host(self, req):
        matched = self.cache_controller.get(req)
        if matched is None:
            matched = self.match_prefix(req)

        latency_host_to_device = (
            matched.host_hit_length
            * self.kv_cache_space_per_token
            / self.platform_config.memory_read_bandwidth
        )

        stats = PrefixCacheFetchResult(
            latency_host_to_device=latency_host_to_device,
            fetched_tokens=matched.host_hit_length,
        )

        return stats

    def on_board_from_host(self, req: FakeRequest):
        fetch_result = self.estimate_on_board_from_host(req)

        matched = self.cache_controller.get(req)
        if matched is None:
            matched = self.match_prefix(req)

        # move the kv cache from host to the device.
        matched.device_hit_length += fetch_result.fetched_tokens
        matched.host_hit_length -= fetch_result.fetched_tokens

        if self.enable_stats:
            if req.id in self.host_buffer_req_tokens:
                allocated_tokens = self.host_buffer_req_tokens.pop(req.id)
                self.host_buffer_tokens -= allocated_tokens

            stats = self.fetch_request_stats.get(req)
            if stats is not None:
                stats.latency_host_to_device += fetch_result.latency_host_to_device
                stats.num_token_from_host += fetch_result.fetched_tokens

            # The fetch process happened in the last execution.
            iter_stats = self.fetch_iteration_stats.get(
                self.global_values.iteration - 1
            )
            if iter_stats is None:
                iter_stats = IterationCacheFetchStats(
                    iter=self.global_values.iteration - 1,
                    timestamp=self.global_values.clock,
                    last_batch_run_time=self.global_values.last_batch_run_time,
                )
                self.fetch_iteration_stats[self.global_values.iteration - 1] = (
                    iter_stats
                )
            iter_stats.latency_host_to_device += fetch_result.latency_host_to_device
            iter_stats.num_token_from_host += fetch_result.fetched_tokens
            iter_stats.host_buffer_size_gb = (
                self.kv_cache_space_per_token * self.host_buffer_tokens
            ) / (1 << 30)
            self.total_hit_kv_length += matched.device_hit_length
            self.total_input_length += req.input_token_length
            iter_stats.mean_kv_reuse_ratio = (
                self.total_hit_kv_length / self.total_input_length
            )

        return fetch_result
