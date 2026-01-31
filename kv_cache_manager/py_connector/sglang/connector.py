import hashlib
import logging
import math
import uuid
from typing import Any, List, Optional
import time
import json

import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.metrics.collector import StorageMetrics
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import get_attention_tp_group, is_dp_attention_enabled

from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient
from kv_cache_manager.client.pybind import kvcm_py_client

logger = logging.getLogger(__name__)


class HiCacheKVCM(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, kwargs):
        self.storage_config = storage_config
        # --hicache-storage-backend-extra-config '{"k":"v"}'
        self.extra_config = self.storage_config.extra_config

        # deployment
        self.instance_group = self.extra_config["instance_group"]
        self.instance_id = self.extra_config["instance_id"]

        self._manager_client = KvCacheManagerClient(
            self.extra_config["manager_uri"]
        )

        self.prefetch_pgs = []
        self.backup_pgs = []
        self.prefetch_bandwidth = []
        self.backup_bandwidth = []

    def _init_kvcm_client(self):
        # parallelism
        self.tp_rank = self.storage_config.tp_rank
        self.tp_size = self.storage_config.tp_size
        # TODO: pp
        self.dp_size = 1
        self.pp_size = 1

        tp_group = (get_attention_tp_group().cpu_group if is_dp_attention_enabled() else get_tp_group().cpu_group)
        self.tp_world_size = torch.distributed.get_world_size(group=tp_group)
        if self.tp_world_size > 1:
            group_ranks = torch.distributed.get_process_group_ranks(tp_group)
            self.storage_tp_group = torch.distributed.new_group(
                group_ranks, backend="gloo"
            )

        # model
        self.model_name = self.storage_config.model_name
        self.is_mla_model = self.storage_config.is_mla_model
        self.kv_factor = 1 if self.is_mla_model else 2
        self.kv_dtype = self.mem_pool_host.dtype
        self.num_layer = self.mem_pool_host.layer_num

        self.deployment = {
            "model_name": self.model_name,
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "pp_size": self.pp_size,
            "use_mla": self.is_mla_model,
            "dtype": str(self.kv_dtype)[6:],  # remove "torch."
        }

        # manager
        self.block_size = self.mem_pool_host.page_size

        self.location_spec_size = self.mem_pool_host.get_size_per_token() * self.block_size
        self.location_spec_infos = [{
            "name": self._tp_rank_to_spec_name(rank),
            "size": self.location_spec_size,
        } for rank in range(self.tp_size)]

        register_request = {
            "trace_id": self._get_trace_id(),
            "instance_group": self.instance_group,
            "instance_id": self.instance_id,
            "model_deployment": self.deployment,
            "block_size": self.block_size,
            "location_spec_infos": self.location_spec_infos,
        }
        # TODO: check conflict and update
        register_response = self._manager_client.register_instance(register_request)
        logger.debug(f"register_instance {register_response=}")

        self.storage_configs = register_response["storage_configs"]

        # data transfer setup
        self.location_spec_name = self._tp_rank_to_spec_name(self.tp_rank)

        self.write_timeout_seconds = self.extra_config.get("write_timeout_seconds", 30)

        # sdk
        self.sdk_thread_num = self.extra_config.get("sdk_thread_num", 4)
        self.sdk_queue_size = self.extra_config.get("sdk_queue_size", 1000)
        self.sdk_get_timeout_ms = self.extra_config.get("sdk_get_timeout_ms", 5000)
        self.sdk_put_timeout_ms = self.extra_config.get("sdk_put_timeout_ms", 10000)

        self.read_iov_block_size = self.extra_config.get("read_iov_block_size", 0)
        self.write_iov_block_size = self.extra_config.get("write_iov_block_size", 0)
        self.iov_size = self.location_spec_size * 1024

        sdk_backend_configs = []

        hf3fs_configs = self.parse_hf3fs_configs(self.storage_configs)
        sdk_backend_configs.extend(hf3fs_configs)
        logger.debug(sdk_backend_configs)
        transfer_client_json = {
            "instance_group": self.instance_group,
            "instance_id": self.instance_id,
            "block_size": self.block_size,
            "sdk_config": {
                "thread_num": self.sdk_thread_num,
                "queue_size": self.sdk_queue_size,
                "sdk_backend_configs": sdk_backend_configs,
                "timeout_config": {
                    "get_timeout_ms": self.sdk_get_timeout_ms,
                    "put_timeout_ms": self.sdk_put_timeout_ms,
                },
            },
            "location_spec_infos": {
                self.location_spec_name: self.location_spec_size,
            },
        }
        self.transfer_client_config = json.dumps(transfer_client_json)

        self.init_params = kvcm_py_client.InitParams()
        self.init_params.role_type = kvcm_py_client.RoleType.WORKER
        self.init_params.self_location_spec_name = self.location_spec_name
        self.init_params.storage_configs = f"{self.storage_configs}"

        self.transfer_client = kvcm_py_client.TransferClient.Create(
            self.transfer_client_config, self.init_params
        )
        assert self.transfer_client is not None, "kvcm_py_client.TransferClient.Create failed"

    def parse_hf3fs_configs(self, storage_configs):
        hf3fs_configs = []
        storage_configs_json = json.loads(storage_configs)
        for storage_config in storage_configs_json:
            if storage_config["type"] == "hf3fs" and storage_config["is_available"]:
                hf3fs_config = {
                    "type": "hf3fs",
                    "mountpoint": storage_config["storage_spec"]["mountpoint"],
                    "root_dir": storage_config["storage_spec"]["root_dir"],
                    "read_iov_block_size": self.read_iov_block_size,
                    "read_iov_size": self.iov_size,
                    "write_iov_block_size": self.write_iov_block_size,
                    "write_iov_size": self.iov_size,
                }
                hf3fs_configs.append(hf3fs_config)
        return hf3fs_configs

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host
        self._init_kvcm_client()

    def _batch_get(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        trace_id: str,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Prepare keys
        block_keys, len_prefix, len_new = self._prepare_block_keys(keys, extra_info)

        get_request = {
            "trace_id": trace_id,
            "block_keys": block_keys,
            "instance_id": self.instance_id,
            "query_type": "QT_PREFIX_MATCH",
            "block_mask": {"offset": len_prefix},
        }
        result = self._manager_client.get_cache_location(get_request)
        logger.debug(f"get_cache_location {result=}")
        locations = result["locations"]

        matched = len(locations)
        if matched == 0:
            return [False] * len_new

        # Data transfer preparation
        buffer_ptrs, buffer_sizes = self.mem_pool_host.get_page_buffer_meta(host_indices)
        buffer_matched = matched * self.kv_factor
        buffer_ptrs = buffer_ptrs[:buffer_matched]
        buffer_sizes = buffer_sizes[:buffer_matched]

        # Extract URIs and prepare buffers
        uris = self._extract_uris(locations)
        buffers = self._prepare_buffers(buffer_ptrs, buffer_sizes)
        assert len(uris) == len(buffers)
        # Perform data transfer
        start_time = time.perf_counter()
        result = self.transfer_client.LoadKvCaches(uris, buffers)
        end_time = time.perf_counter()
        self.prefetch_pgs.append(matched)
        self.prefetch_bandwidth.append(matched * self.location_spec_size / (1 << 30) / (end_time - start_time))
        logger.debug(f"LoadKvCaches {result=}")

        flag = (result == kvcm_py_client.ClientErrorCode.ER_OK)
        if not flag:
            logger.error(f"{result}")
        return [flag] * len_new

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        trace_id = self._get_trace_id()
        try:
            result = self._batch_get(keys=keys, host_indices=host_indices, trace_id=trace_id, extra_info=extra_info)
            return result
        except Exception as e:
            logger.error(f"batch_get_v1 failed: {trace_id=} {e=}")
            return [False] * len(keys)

    def _batch_set(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        trace_id: str,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Prepare keys
        block_keys, len_prefix, len_new = self._prepare_block_keys(keys, extra_info)

        # Start write cache
        if self.tp_rank == 0:
            start_trace_id = f"start-{trace_id}"
            request = {
                "trace_id": start_trace_id,
                "instance_id": self.instance_id,
                "block_keys": block_keys,
                "write_timeout_seconds": self.write_timeout_seconds,
            }
            logger.debug(f"start_write_cache {request=}")
            result = self._manager_client.start_write_cache(request)

            if self.tp_world_size > 1:
                torch.distributed.broadcast_object_list(
                    [result], src=0, group=self.storage_tp_group
                )
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(
                recv, src=0, group=self.storage_tp_group
            )
            result = recv[0]

        logger.debug(f"start_write_cache {result=}")
        locations = result["locations"]
        write_session_id = result["write_session_id"]
        block_mask = result["block_mask"]
        save_indices = self._parse_block_mask(block_mask, len_prefix, len_new)
        len_save = max(save_indices, default=-1) + 1

        unmatched = len(save_indices)

        finish_trace_id = f"finish-{trace_id}"
        # Early return if no locations
        if unmatched == 0:
            if self.tp_rank == 0:
                self._manager_client.finish_write_cache(
                    {
                        "trace_id": finish_trace_id,
                        "instance_id": self.instance_id,
                        "write_session_id": write_session_id,
                        "success_blocks": {"bool_masks": {"values": [False] * len(locations)}},
                    }
                )
            return [False] * len(keys)

        assert unmatched == len(locations)

        # Data transfer preparation
        buffer_ptrs, buffer_sizes = self.mem_pool_host.get_page_buffer_meta(host_indices)
        buffer_ptrs = [buffer_ptr for i, buffer_ptr in enumerate(buffer_ptrs) if (i // self.kv_factor) in save_indices]
        buffer_sizes = [buffer_size for i, buffer_size in enumerate(
            buffer_sizes) if (i // self.kv_factor) in save_indices]

        # Extract URIs and prepare buffers
        uris = self._extract_uris(locations)
        buffers = self._prepare_buffers(buffer_ptrs, buffer_sizes)
        assert len(uris) == len(buffers)
        # Perform data transfer
        start_time = time.perf_counter()
        result = self.transfer_client.SaveKvCaches(uris, buffers)
        end_time = time.perf_counter()
        self.backup_pgs.append(unmatched)
        self.backup_bandwidth.append(unmatched * self.location_spec_size / (1 << 30) / (end_time - start_time))
        logger.debug(f"SaveKvCaches {result=}")

        # Finish write cache
        flag = (result[0] == kvcm_py_client.ClientErrorCode.ER_OK)
        if self.tp_world_size > 1:
            flag_tensor = torch.tensor(flag, dtype=torch.int)
            torch.distributed.all_reduce(
                flag_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.storage_tp_group,
            )
            flag = bool(flag_tensor.item())

        finish_mask = [flag] * unmatched
        success_mask = finish_mask + [False] * (len_new - len_save)
        if self.tp_rank == 0:
            self._manager_client.finish_write_cache(
                {
                    "trace_id": finish_trace_id,
                    "instance_id": self.instance_id,
                    "write_session_id": write_session_id,
                    "success_blocks": {"bool_masks": {"values": finish_mask}},
                }
            )
        return success_mask

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        trace_id = self._get_trace_id()
        try:
            result = self._batch_set(keys=keys, host_indices=host_indices, trace_id=trace_id, extra_info=extra_info)
            return result
        except Exception as e:
            logger.error(f"batch_set_v1 failed: {trace_id=} {e=}")
            return [False] * len(keys)

    def _batch_exists(
        self,
        keys: List[str],
        trace_id: str,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        block_keys, len_prefix, len_new = self._prepare_block_keys(keys, extra_info)
        get_request = {
            "trace_id": trace_id,
            "block_keys": block_keys,
            "instance_id": self.instance_id,
            "query_type": "QT_PREFIX_MATCH",
            "block_mask": {"offset": len_prefix},
        }
        result = self._manager_client.get_cache_location(get_request)
        logger.debug(f"get_cache_location {result=}")
        return len(result["locations"])

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        trace_id = self._get_trace_id()
        try:
            result = self._batch_exists(keys=keys, trace_id=trace_id, extra_info=extra_info)
            return result
        except Exception as e:
            logger.error(f"batch_exists failed: {trace_id=} {e=}")
            return 0

    def get_stats(self):
        storage_metrics = StorageMetrics()
        storage_metrics.prefetch_pgs.extend(self.prefetch_pgs)
        storage_metrics.backup_pgs.extend(self.backup_pgs)
        storage_metrics.prefetch_bandwidth.extend(self.prefetch_bandwidth)
        storage_metrics.backup_bandwidth.extend(self.backup_bandwidth)
        self.prefetch_pgs.clear()
        self.backup_pgs.clear()
        self.prefetch_bandwidth.clear()
        self.backup_bandwidth.clear()
        return storage_metrics

    ##################################################

    def _tp_rank_to_spec_name(self, tp_rank: int) -> str:
        return f"tp_{tp_rank}"

    def _get_trace_id(self) -> str:
        return str(uuid.uuid1())

    def _sha256_to_int64(self, data: str) -> int:
        data = data.encode("utf-8")
        hash_digest = hashlib.sha256(data).digest()
        hash_int64 = int.from_bytes(hash_digest[:8], "big", signed=True)
        return hash_int64

    def _prepare_block_keys(
            self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None) -> tuple[List[int], int]:
        """Prepare block keys and return them along with the prefix offset."""
        prefix_keys = (
            extra_info.prefix_keys
            if (extra_info is not None) and (extra_info.prefix_keys is not None)
            else []
        )
        block_keys = prefix_keys + keys
        block_keys = [
            self._sha256_to_int64(block_key) for block_key in block_keys
        ]
        return block_keys, len(prefix_keys), len(keys)

    def _extract_uris(self, locations: List[dict]) -> List[str]:
        """Extract URIs from locations for the current TP rank."""
        uris = []
        for location in locations:
            for location_spec in location["location_specs"]:
                if location_spec["name"] == self.location_spec_name:
                    uris.append(location_spec["uri"])
        return uris

    def _prepare_buffers(self, buffer_ptrs: List[int], buffer_sizes: List[int]) -> List[kvcm_py_client.BlockBuffer]:
        """Prepare buffers for data transfer."""
        buffers = []
        for i in range(0, len(buffer_ptrs), self.kv_factor):
            buffer = kvcm_py_client.BlockBuffer()
            iovs = []
            for j in range(self.kv_factor):
                iov = kvcm_py_client.Iov()
                iov.type = kvcm_py_client.MemoryType.CPU
                iov.base = buffer_ptrs[i + j]
                iov.size = buffer_sizes[i + j]
                iov.ignore = False
                iovs.append(iov)
            buffer.iovs = iovs
            buffers.append(buffer)
        return buffers

    def _parse_block_mask(self, block_mask: dict, len_prefix: int, len_new: int) -> int:
        save_indices = []
        if "offset" in block_mask:
            offset = block_mask["offset"]
            if offset >= len_prefix:
                save_indices.extend(range(offset, len_prefix + len_new))
        else:
            # False: need to store
            bool_masks = block_mask.get("bool_masks", {}).get("values", [])
            if all(bool_masks[:len_prefix]):
                max_index = max([i for i, x in enumerate(bool_masks) if not x], default=-1)
                save_indices.extend([i for i in range(len_prefix, max_index + 1) if not bool_masks[i]])
        save_indices = [(i - len_prefix) for i in save_indices if i >= len_prefix]
        return save_indices

    ##################################################

    def clear(self) -> None:
        raise NotImplementedError()

    def exists(self, key: str) -> bool:
        raise NotImplementedError()

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError()

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        raise NotImplementedError()

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError()

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError()
