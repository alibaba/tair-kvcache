from dataclasses import dataclass, field
from typing import Optional, Union, Any
from schedule_simulator._compat import ModelInfo, AcceleratorInfo, DataType
from enum import Enum, IntEnum, unique
from schedule_simulator.dataset import BaseDataset


@dataclass
class BenchmarkConfig:
    # The request's length config could be None if the dataset is used.
    num_prompts: Optional[int] = None
    min_input_length: Optional[int] = None
    max_input_length: Optional[int] = None
    min_output_length: Optional[int] = None
    max_output_length: Optional[int] = None
    max_concurrency: Optional[int] = None
    request_rate: float = float("inf")
    dataset: Optional[BaseDataset] = None
    dataset_path: Optional[str] = None
    # The dataset, which stored with jsonl format, should be organized as following:
    # {"timestamp": 0, "input_length": 6755, "output_length": 500}
    # {"timestamp": 0, "input_length": 7319, "output_length": 490}
    # Ref: https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/mooncake_trace.jsonl
    # The unit of timestamp is milsecond.
    min_prefix_disk_hit_rate: Optional[float] = None
    max_prefix_disk_hit_rate: Optional[float] = None
    min_prefix_host_hit_rate: Optional[float] = None
    max_prefix_host_hit_rate: Optional[float] = None
    disable_tqdm: bool = False
    num_instances: int = 1  # The number of server instance.
    data_block_size: Optional[int] = None  # Block size (tokens/block) used in dataset block_ids

    # Control the timestamp parameter when sending requests to the LLM server from the benchmark client.
    with_queue_start: bool = False
    ignore_request_timestamp: bool = False

    def __post_init__(self):
        if self.dataset_path is not None:
            # If dataset is provided, the request rate should be set to infinity and
            # the max concurrency should be set to None.
            self.request_rate = float("inf")
            self.max_concurrency = None

        if self.min_prefix_disk_hit_rate is not None:
            assert 0 <= self.min_prefix_disk_hit_rate <= 1
            assert self.min_prefix_disk_hit_rate <= self.max_prefix_disk_hit_rate <= 1

        if self.min_prefix_host_hit_rate is not None:
            assert 0 <= self.min_prefix_host_hit_rate <= 1
            assert self.min_prefix_host_hit_rate <= self.max_prefix_host_hit_rate <= 1

        if (
            self.max_prefix_disk_hit_rate is not None
            and self.max_prefix_host_hit_rate is not None
        ):
            assert self.max_prefix_disk_hit_rate + self.max_prefix_host_hit_rate <= 1

        if self.dataset is not None and self.num_prompts is None:
            self.num_prompts = len(self.dataset)


class RoutingPolicy(Enum):
    RANDOM = "random"
    CACHE_AWARE = "cache_aware"
    CACHE_AWARE_OLD = "cache_aware_old"  # Deprecated, use CACHE_AWARE instead
    POWER_OF_TWO = "power_of_two"
    ROUND_ROBIN = "round_robin"
    DIRECT_CACHE_AWARE = "direct_cache_aware"


@dataclass
class RouterConfig:
    """
    路由器配置中心
    """

    p_policy: RoutingPolicy = RoutingPolicy.ROUND_ROBIN
    d_policy: RoutingPolicy = RoutingPolicy.ROUND_ROBIN
    # CacheAwareConfig
    cache_threshold: float = 0.3
    # balance_abs_threshold: int = 64
    balance_abs_threshold: int = 8
    balance_rel_threshold: float = 1.5
    eviction_interval_secs: int = 120
    max_tree_size: int = 2**26
    # Power_of_two
    worker_startup_check_interval: float = 30  # s

    @classmethod
    def from_string_policy(
        cls,
        p_policy_str: str = "round_robin",
        d_policy_str: str = "round_robin",
        **kwargs,
    ):
        try:
            p_enum = RoutingPolicy(p_policy_str)
            d_enum = RoutingPolicy(d_policy_str)
        except ValueError as e:
            raise ValueError(f"Can't find policy: {e}")

        return cls(p_policy=p_enum, d_policy=d_enum, **kwargs)


@dataclass
class SchedulerConfig:
    model: Union[ModelInfo, str]
    # For the default value, please refer to "https://docs.sglang.ai/backend/server_arguments.html".
    max_prefill_tokens: int = 16384
    chunked_prefill_size: Optional[int] = None
    data_type: Optional[DataType] = (
        None  # Data type for model weights and activations. If none is set, it will be automatically detected.
    )
    kv_cache_data_type: Optional[DataType] = None
    mem_fraction_static: Optional[float] = None
    hicache_storage_backend: Optional[str] = None
    hicache_write_policy: str = "write_through"  # choices: write_back, write_through, write_through_selective
    hicache_read_query_type: str = "prefix_match"  # choices: prefix_match, batch_get
    hicache_storage_prefetch_policy: str = (
        "best_effort"  # choices: best_effort, wait_complete, timeout
    )
    schedule_policy: str = "fcfs"
    scenario: str = "normal"  # choices: normal / disagg_prefill / disagg_decode
    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    enable_stats: bool = False
    # If real-time mode is enabled, the clock will be modified only when a future request arrives.
    # It needs to set the number of requests normally.
    enable_real_time_request: bool = False
    max_running_requests: int = (1 << 31) - 1
    # Log the request metrics at every interval of completed requests.
    # If 0 < interval < 1: the interval is calculated as int(num_requests * interval)
    log_metrics_interval: Optional[float] = None
    page_size: Optional[int] = None

    # framework backend
    # Hardware parameter overrides (skip auto-derivation from ModelInfo/AcceleratorInfo)
    kv_cache_space_per_token: Optional[int] = None  # bytes per token in KV cache
    max_num_tokens: Optional[int] = None             # L1 (device) KV cache capacity in tokens
    l2_cache_num_tokens: Optional[int] = None        # L2 (host) cache capacity in tokens (default: 2x L1)

    request_level_scheduling: bool = False
    backend_name: str = "sglang"
    backend_version: Optional[str] = None

    def __post_init__(self):
        assert self.scenario in ["normal", "disagg_prefill", "disagg_decode"]
        assert self.log_metrics_interval is None or self.log_metrics_interval > 0

        if self.schedule_policy in ["mcr", "plg"] or self.scenario == "disagg_decode":
            self.hicache_storage_prefetch_policy = "wait_complete"


@dataclass
class PlatformConfig:
    device: Union[AcceleratorInfo, str]
    # Storage configuration for hierarchical cache management.
    hbm_capacity_gb: Optional[float] = None
    disk_capacity_gb: Optional[float] = None
    disk_read_bandwidth_gb: Optional[float] = None
    disk_write_bandwidth_gb: Optional[float] = None
    memory_capacity_gb: Optional[float] = None
    memory_read_bandwidth_gb: Optional[float] = None
    memory_write_bandwidth_gb: Optional[float] = None
    num_device_per_node: int = 8
    peer_read_bandwidth_gb: Optional[float] = None

    @property
    def disk_read_bandwidth(self):
        return (
            self.disk_read_bandwidth_gb * 1e9 if self.disk_read_bandwidth_gb else None
        )

    @property
    def disk_write_bandwidth(self):
        return (
            self.disk_write_bandwidth_gb * 1e9 if self.disk_write_bandwidth_gb else None
        )

    @property
    def memory_read_bandwidth(self):
        return (
            self.memory_read_bandwidth_gb * 1e9
            if self.memory_read_bandwidth_gb
            else None
        )


    @property
    def peer_read_bandwidth(self):
        return (
            self.peer_read_bandwidth_gb * 1e9
            if self.peer_read_bandwidth_gb
            else None
        )

    @property
    def memory_write_bandwidth(self):
        return (
            self.memory_write_bandwidth_gb * 1e9
            if self.memory_write_bandwidth_gb
            else None
        )


@unique
class RequestStage(IntEnum):
    IDLE = 0
    PREFETCHING = 1
    READY = 2
    PREFILLING = 3
    DECODING = 4
    COMPLETE = 5
    FAILED = 6


@dataclass
class FakeRequest:
    id: int
    input_token_length: int
    output_token_length: int
    disk_cache_hit_length: int = 0
    # The session_id is used to identify the requests from the same session, which is useful for simulating multi-turn conversations.
    session_id: Optional[int] = None
    host_cache_hit_length: int = 0
    device_cache_hit_length: int = 0
    # The start and end of context prefill are used to track the running tokens in the chunked prefill process.
    context_prefill_start: int = 0
    context_prefill_end: int = 0
    last_event_time: float = 0
    gen_token_latencies: list[float] = field(default_factory=list)
    stage: RequestStage = RequestStage.IDLE
    queue_time_start: float = -1
    queue_time_end: float = -1
    final_reused_tokens: int = 0
    # radix cache
    origin_input_ids: Optional[list[int]] = None
    output_ids: Optional[list[int]] = None
    fill_ids: Optional[list[int]] = None  # fill_ids = origin_input_ids + output_ids.
    last_node: Any = None
    last_host_node: Any = None
    req_pool_idx: Optional[int] = None  # Index in req_to_token pool
    last_matched_prefix_len: int = 0
    prefix_indices_len: Optional[int] = 0
    host_hit_len: Optional[int] = 0
    disk_hit_len: Optional[int] = 0
    remain_prefetch_len: Optional[int] = 0
    prefetch_start_record: Optional[float] = None
    # if block_size is not None, then the ids represents the hash value of each blocks
    block_size: Optional[int] = None
    extra_key: Optional[str] = None

    def __eq__(self, req):
        if not isinstance(req, FakeRequest):
            return NotImplemented
        return self.id == req.id

    def __hash__(self):
        # The request's id is unique
        return self.id

    def __lt__(self, req: "FakeRequest"):
        # This is used solely for ordering requests in the idle state,
        # as the last event time will be modified during generation.
        # Currently used in the future queue.
        if self.last_event_time == req.last_event_time:
            return self.id < req.id
        return self.last_event_time < req.last_event_time

    def __post_init__(self):
        # self.context_prefill_start = 0
        # self.context_prefill_end = 0
        pass

    def create_time(self) -> float:
        return self.last_event_time - sum(self.gen_token_latencies)

    @property
    def remaining_prefill_tokens(self) -> int:
        return int(self.input_token_length - self.context_prefill_start)

    @property
    def input_length(self) -> int:
        return max(1, self.context_prefill_length)

    @property
    def past_kv_length(self) -> int:
        return int(self.context_prefill_start + len(self.gen_token_latencies))

    @property
    def context_prefill_length(self) -> int:
        return int(self.context_prefill_end - self.context_prefill_start)

    def is_idle(self) -> bool:
        return self.stage == RequestStage.IDLE

    def is_prefetching(self) -> bool:
        return self.stage == RequestStage.PREFETCHING

    def is_ready(self) -> bool:
        return self.stage == RequestStage.READY

    def is_prefilling(self) -> bool:
        return self.stage == RequestStage.PREFILLING

    def is_decoding(self) -> bool:
        return self.stage == RequestStage.DECODING

    def is_complete(self) -> bool:
        return self.stage == RequestStage.COMPLETE

    def is_finished(self) -> bool:
        return len(self.gen_token_latencies) == self.output_token_length

    def has_next_chunk(self) -> bool:
        return self.input_token_length > self.context_prefill_end


@dataclass
class RequestStats:
    id: int
    state: str
    context_prefill_start: int
    context_prefill_end: int
    num_generated_tokens: int


@dataclass
class IterationStats:
    timestamp: float
    iter: int
    iter_latency_ms: float
    num_context_requests: int
    num_ctx_tokens: int
    num_gen_requests: int
    request_stats: list[RequestStats]
    num_waiting_requests: int


@dataclass
class PrefixCacheMatchResult:
    device_hit_length: int = 0
    host_hit_length: int = 0
    disk_hit_length: int = 0

    def hit_length(self):
        return self.device_hit_length + self.host_hit_length + self.disk_hit_length


@dataclass
class PrefixCacheFetchResult:
    latency_disk_to_host: float = 0
    latency_host_to_device: float = 0
    fetched_tokens: int = 0


@dataclass
class RequestCacheFetchStats:
    req_id: int
    prefetch_queue_start: float = 0
    prefetch_queue_end: float = 0
    num_token_from_disk: int = 0
    num_token_from_host: int = 0
    latency_disk_to_host: float = 0
    latency_host_to_device: float = 0
    actual_prefix_len = 0


@dataclass
class IterationCacheFetchStats:
    iter: int
    timestamp: float = 0
    last_batch_run_time: float = 0
    num_token_from_disk: int = 0
    num_token_from_host: int = 0
    latency_disk_to_host: float = 0
    latency_host_to_device: float = 0
    host_buffer_size_gb: float = 0
    mean_kv_reuse_ratio: float = 0
