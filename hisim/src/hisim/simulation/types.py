from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union
from hisim.spec import ModelInfo, DataType, AcceleratorInfo


@dataclass
class BenchmarkConfig:
    request_rate: float = float("inf")
    max_concurrency: Optional[int] = None
    with_queue_start: bool = (
        False  # For hisim hooks: include queue start time in replay schedule.
    )
    ignore_request_timestamp: bool = False


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
    hicache_storage_prefetch_policy: str = (
        "best_effort"  # choices: best_effort, wait_complete, timeout
    )
    schedule_policy: str = "fcfs"
    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    max_running_requests: int = (1 << 31) - 1
    page_size: Optional[int] = None

    # framework backend
    backend_name: str = "sglang"
    backend_version: Optional[str] = None


class MockSimulationMode(Enum):
    BLOCKING = "BLOCKING"
    OFFLINE = "OFFLINE"


@dataclass
class RequestStats:
    rid: str = ""
    last_event_time: float = 1.0
    input_length: int = 1
    output_length: int = 1
    final_reused_tokens: int = 0
    prefetch_complete_tokens: int = 0
    queue_start: float = -1
    queue_end: float = -1
    created_time: float = -1
    gen_token_latencies: list[float] = field(default_factory=list)

    def is_complete(self) -> bool:
        return True


@dataclass
class PlatformConfig:
    device: Union[AcceleratorInfo, str]
    # Storage configuration for hierarchical cache management.
    disk_capacity_gb: Optional[float] = None
    disk_read_bandwidth_gb: Optional[float] = None
    disk_write_bandwidth_gb: Optional[float] = None
    memory_capacity_gb: Optional[float] = None
    memory_read_bandwidth_gb: Optional[float] = None
    memory_write_bandwidth_gb: Optional[float] = None
    num_device_per_node: int = 8

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
    def memory_write_bandwidth(self):
        return (
            self.memory_write_bandwidth_gb * 1e9
            if self.memory_write_bandwidth_gb
            else None
        )
