from typing import Any


class TairKvCacheConnectorExtraConfig:
    def __init__(self, extra_config: dict[str, Any]):
        self.manager_uri: str = extra_config["manager_uri"]
        self.coordinator_base_port: int = extra_config["coordinator_base_port"]
        self.instance_group: str = extra_config["instance_group"]
        self.instance_id: str = extra_config["instance_id"]
        self.preferred_block_size: int = extra_config.get("preferred_block_size", 0)
        self.storage_configs: dict[str, dict] = extra_config.get("storage_configs", {})

        self.write_timeout_seconds: int = extra_config.get("write_timeout_seconds", 30)
        self.sdk_thread_num = extra_config.get("sdk_thread_num", 32)
        self.sdk_queue_size = extra_config.get("sdk_queue_size", 1000)
        self.sdk_get_timeout_ms = extra_config.get("sdk_get_timeout_ms", 15000)
        self.sdk_put_timeout_ms = extra_config.get("sdk_put_timeout_ms", 15000)

        self.read_iov_block_size = extra_config.get("read_iov_block_size", 0)
        self.write_iov_block_size = extra_config.get("write_iov_block_size", 0)
        self.hf3fs_concurrent_io_block_count = extra_config.get("hf3fs_concurrent_io_block_count", 32)

        self.block_per_save_task = extra_config.get("block_per_save_task", 128)
        self.block_per_load_task = extra_config.get("block_per_load_task", 128)

        # Cap (bytes) on concurrently allocated pinned staging buffers for
        # save/load transfers; <= 0 selects a default derived from the task
        # sizes (see DataTransferManager).
        self.staging_buffer_max_bytes: int = extra_config.get("staging_buffer_max_bytes", 0)

        self.async_get_cache_location = extra_config.get("async_get_cache_location", True)
        # TODO: add async and try wait
        # self.async_get_cache_location_wait_time = extra_config.get("async_get_cache_location_wait_time", 0)

        # Leader discovery
        self.auto_discover_leader: bool = extra_config.get("auto_discover_leader", False)
        self.leader_retry_count: int = extra_config.get("leader_retry_count", 1)
        self.leader_retry_base_interval_seconds: float = extra_config.get("leader_retry_base_interval_seconds", 0.005)
        self.discovery_refresh_interval_seconds: int = extra_config.get("discovery_refresh_interval_seconds", 30)
        self.min_discover_interval_seconds: int = extra_config.get("min_discover_interval_seconds", 1)
        self.request_timeout_seconds: float = extra_config.get("request_timeout_seconds", 1.0)

        self.log_level: str = extra_config.get("log_level", "")

        # Escape hatch for the hybrid capability probe: when
        # Scheduler._mamba_block_aligned_split exists but its source cannot be
        # inspected (frozen/bytecode-only vLLM) the connector fails closed;
        # set this to true to force-enable hybrid models there.
        self.force_hybrid_support: bool = extra_config.get("force_hybrid_support", False)
