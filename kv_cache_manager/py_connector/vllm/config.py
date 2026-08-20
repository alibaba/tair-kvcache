"""kv_connector_extra_config, as a validated Pydantic model.

Field defaults double as the documentation of every knob; unknown keys are
rejected (typos in extra_config fail at startup instead of silently
defaulting)."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class TairKvCacheConnectorExtraConfig(BaseModel):
    # --- Identity / registration ---
    manager_uri: str
    coordinator_base_port: int
    instance_group: str
    instance_id: str
    # Manager block size override; 0 keeps the vLLM scheduler block size.
    # Ignored for hybrid models (mamba state is per scheduler block).
    preferred_block_size: int = 0
    storage_configs: Dict[str, Any] = Field(default_factory=dict)

    # --- Write sessions ---
    write_timeout_seconds: int = 30

    # --- Transfer SDK ---
    sdk_thread_num: int = 32
    sdk_queue_size: int = 1000
    sdk_get_timeout_ms: int = 15000
    sdk_put_timeout_ms: int = 15000
    read_iov_block_size: int = 0
    write_iov_block_size: int = 0
    hf3fs_concurrent_io_block_count: int = 32

    # --- Transfer task granularity ---
    block_per_save_task: int = 128
    block_per_load_task: int = 128

    # --- Staging buffers ---
    # Pre-allocated contiguous staging slots per transfer group (shared by
    # save and load). GPU side is a *fixed* HBM reservation of
    # staging_pool_blocks * per_block_bytes per group; per-task dynamic
    # allocation instead competes with the engine's own memory and OOMs it
    # under load on low-headroom GPUs. An exhausted pool blocks the task
    # (backpressure). Must be >= max(block_per_save_task, block_per_load_task)
    # because one task stages its whole batch contiguously; shrink both
    # together on small cards.
    staging_pool_blocks: int = 128

    # --- Manager queries ---
    async_get_cache_location: bool = True

    # --- Leader discovery / HTTP ---
    auto_discover_leader: bool = False
    leader_retry_count: int = 1
    leader_retry_base_interval_seconds: float = 0.005
    discovery_refresh_interval_seconds: int = 30
    min_discover_interval_seconds: int = 1
    request_timeout_seconds: float = 1.0

    log_level: str = ""

    # Escape hatch for the hybrid capability probe: when
    # Scheduler._mamba_block_aligned_split exists but its source cannot be
    # inspected (frozen/bytecode-only vLLM) the connector fails closed;
    # set this to true to force-enable hybrid models there.
    force_hybrid_support: bool = False

    model_config = {"extra": "forbid"}
