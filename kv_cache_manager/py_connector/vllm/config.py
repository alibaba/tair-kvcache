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
    # save and load). The pool is *pinned host memory only* -- the gather/
    # scatter kernel and the state copies reach it directly over PCIe, so
    # the connector's device-memory footprint is zero and this knob sizes
    # host RAM, not VRAM. An exhausted pool blocks the task (backpressure).
    #
    # The pool is the *concurrency* of staging: tasks hold their slots for
    # gather + the synchronous SDK transfer, so the pool size caps how many
    # tasks feed the SDK at once. At 128 (one full task) the SDK transfer
    # serializes and burst loads queue behind saves: measured -3.5% ab
    # throughput and +54% vs +65% TP2 hit throughput against a 512-block
    # pool (perf 2026-08-20, doc protocol). 1024 restores origin/main's
    # concurrency (8 full tasks) for ~896 MiB pinned host RAM per attention
    # group -- host RAM is cheap, and the old 1024-slot behaviour is what
    # the merge-base AB baseline ran. Shrink only together with
    # block_per_save_task/block_per_load_task (must be >= the larger one,
    # one task stages its whole batch contiguously).
    staging_pool_blocks: int = 1024

    # Per-group ceiling on the staging pool's pinned host RAM. The block
    # count above was derived for full-attention blocks (~0.875 MiB each:
    # 1024 blocks ~= 896 MiB); hybrid blocks are ~17.3 MiB, and the same
    # count would pin ~17 GiB per group -- four groups then die in the
    # pinned allocator at engine start on an ordinary host. Effective blocks
    # per group = min(staging_pool_blocks, this_cap // block_bytes), never
    # below one full task batch (the contiguity invariant). Raise this cap
    # (not just the block count) to give hybrid deployments more in-flight
    # transfer concurrency.
    staging_pool_max_bytes_per_group: int = 2**30

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
