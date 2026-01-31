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
        self.sdk_get_timeout_ms = extra_config.get("sdk_get_timeout_ms", 5000)
        self.sdk_put_timeout_ms = extra_config.get("sdk_put_timeout_ms", 10000)

        self.read_iov_block_size = extra_config.get("read_iov_block_size", 0)
        self.write_iov_block_size = extra_config.get("write_iov_block_size", 0)
        self.hf3fs_concurrent_io_block_count = extra_config.get("hf3fs_concurrent_io_block_count", 32)

        self.block_per_save_task = extra_config.get("block_per_save_task", 128)
        self.block_per_load_task = extra_config.get("block_per_load_task", 128)

        self.async_get_cache_location = extra_config.get("async_get_cache_location", True)
        # TODO: add async and try wait
        # self.async_get_cache_location_wait_time = extra_config.get("async_get_cache_location_wait_time", 0)
