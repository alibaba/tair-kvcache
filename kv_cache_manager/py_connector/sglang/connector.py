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
    PoolName,
    PoolTransfer,
    PoolTransferResult,
    PoolHitPolicy,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
StorageMetrics = None
try:
    from sglang.srt.observability.metrics_collector import StorageMetrics
except ImportError:
    pass
if StorageMetrics is None:
    try:
        from sglang.srt.metrics.collector import StorageMetrics
    except ImportError:
        raise ImportError(
            "Cannot import StorageMetrics from sglang. "
            "Tried sglang.srt.observability.metrics_collector and "
            "sglang.srt.metrics.collector. "
            "Please check your sglang version is compatible."
        )
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import get_attention_tp_group, is_dp_attention_enabled

from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient
from kv_cache_manager.client.pybind import kvcm_py_client
from kv_cache_manager.py_connector.common._version_info import FULL_VERSION, GIT_COMMIT, BUILD_TIME

logger = logging.getLogger(__name__)


class HiCacheKVCM(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, kwargs):
        logger.warning("KVCM sglang connector version: %s (commit: %s, build: %s)", FULL_VERSION, GIT_COMMIT, BUILD_TIME)
        self.storage_config = storage_config
        # --hicache-storage-backend-extra-config '{"k":"v"}'
        self.extra_config = self.storage_config.extra_config

        # deployment
        self.instance_group = self.extra_config["instance_group"]
        self.instance_id = self.extra_config["instance_id"]

        self._manager_client = KvCacheManagerClient(
            self.extra_config["manager_uri"],
            instance_id=self.instance_id,
            auto_discover_leader=self.extra_config.get("auto_discover_leader", False),
            leader_retry_count=self.extra_config.get("leader_retry_count", 1),
            leader_retry_base_interval_seconds=self.extra_config.get("leader_retry_base_interval_seconds", 0.005),
            discovery_refresh_interval_seconds=self.extra_config.get("discovery_refresh_interval_seconds", 30),
            min_discover_interval_seconds=self.extra_config.get("min_discover_interval_seconds", 1),
        )

        self.registered_pools = {}

        self.prefetch_pgs = []
        self.backup_pgs = []
        self.prefetch_bandwidth = []
        self.backup_bandwidth = []

        # Declare v1 interface support so that sglang's cache_controller uses
        # batch_set_v1/batch_get_v1 (zero-copy path) instead of the legacy
        # batch_set/batch_get. This avoids requiring users to manually add
        # "interface_v1": 1 in --hicache-storage-backend-extra-config.
        self.extra_config.setdefault("interface_v1", 1)

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
        kv_pool = self.registered_pools[PoolName.KV]
        self.kv_dtype = kv_pool.dtype

        # manager
        self.block_size = self.mem_pool_host.page_size

        # Detect extra pools early — _tp_rank_to_spec_name depends on these.
        self.has_mamba = PoolName.MAMBA in self.registered_pools
        self.has_indexer = getattr(PoolName, "INDEXER", None) is not None and PoolName.INDEXER in self.registered_pools
        self.has_extra_pool = self.has_mamba or self.has_indexer

        # location specs & groups (KV / Mamba / Indexer)
        self.location_spec_infos = []
        self.location_spec_groups = []

        # KV pool (always registered)
        self.location_spec_size = kv_pool.get_size_per_token() * self.block_size
        self.location_spec_name = self._register_pool_specs(
            self._get_kv_spec_group(), self._tp_rank_to_spec_name, self.location_spec_size)

        # Mamba pool (optional)
        if self.has_mamba:
            mamba_pool = self.registered_pools[PoolName.MAMBA]
            self.mamba_spec_size = mamba_pool.get_size_per_token()
            self.mamba_location_spec_name = self._register_pool_specs(
                self._get_extra_pool_spec_group(PoolName.MAMBA),
                self._tp_rank_to_linear_spec_name, self.mamba_spec_size)

        # Indexer pool (optional)
        if self.has_indexer:
            indexer_pool = self.registered_pools[PoolName.INDEXER]
            self.indexer_spec_size = indexer_pool.get_size_per_token() * self.block_size
            self.indexer_location_spec_name = self._register_pool_specs(
                self._get_extra_pool_spec_group(PoolName.INDEXER),
                self._tp_rank_to_indexer_spec_name, self.indexer_spec_size)

        # Backward compat with older managers: only send location_spec_groups
        # when extra pools exist; pure-KV deployments keep groups empty.
        if not self.has_extra_pool:
            self.location_spec_groups = []

        self._register_instance()
        self._init_transfer_client()

    def _register_pool_specs(self, group_name: str, name_fn, spec_size: int) -> str:
        """Register location specs for all TP ranks and append a spec group.

        Returns the spec name used for read/write on the current rank. For MLA
        models only rank 0 owns the data, so every rank uses rank 0's spec.
        """
        spec_names = []
        for rank in range(self.tp_size):
            name = name_fn(rank)
            self.location_spec_infos.append({"name": name, "size": spec_size})
            spec_names.append(name)
        self.location_spec_groups.append({
            "name": group_name,
            "spec_names": spec_names,
        })
        effective_rank = 0 if self.is_mla_model else self.tp_rank
        return name_fn(effective_rank)

    def _register_instance(self):
        """Register this instance with the manager and retrieve storage_configs."""
        self.deployment = {
            "model_name": self.model_name,
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "pp_size": self.pp_size,
            "use_mla": self.is_mla_model,
            "dtype": str(self.kv_dtype)[6:],  # remove "torch."
        }

        register_request = {
            "trace_id": self._get_trace_id(),
            "instance_group": self.instance_group,
            "instance_id": self.instance_id,
            "model_deployment": self.deployment,
            "block_size": self.block_size,
            "location_spec_infos": self.location_spec_infos,
        }
        if self.location_spec_groups:
            register_request["location_spec_groups"] = self.location_spec_groups
        # TODO: check conflict and update
        register_response = self._manager_client.register_instance(register_request)
        logger.debug(f"register_instance {register_response=}")

        self.storage_configs = register_response["storage_configs"]
        self.write_timeout_seconds = self.extra_config.get("write_timeout_seconds", 30)

    def _init_transfer_client(self):
        """Assemble SDK config and create TransferClient."""
        # sdk
        self.sdk_thread_num = self.extra_config.get("sdk_thread_num", 4)
        self.sdk_queue_size = self.extra_config.get("sdk_queue_size", 1000)
        self.sdk_get_timeout_ms = self.extra_config.get("sdk_get_timeout_ms", 5000)
        self.sdk_put_timeout_ms = self.extra_config.get("sdk_put_timeout_ms", 10000)

        self.read_iov_block_size = self.extra_config.get("read_iov_block_size", 0)
        self.write_iov_block_size = self.extra_config.get("write_iov_block_size", 0)

        # TODO: the HF3FS backend is currently not well suited for hybrid
        # (KV + Mamba) transfers.  A single IOV mempool is shared across
        # all spec types, so when location_spec_size and mamba_spec_size
        # differ significantly the mempool is either over-allocated for the
        # smaller spec or too small for the larger one.  Per-spec IOV
        # sizing would require changes to the HF3FS SDK itself.
        self.iov_size = max(
            self.location_spec_size * 1024,
            self.mamba_spec_size * 1024 if self.has_mamba else 0,
            self.indexer_spec_size * 1024 if self.has_indexer else 0,
        )

        sdk_backend_configs = list(self.extra_config.get("sdk_backend_configs", []))

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
                **(
                    {self.mamba_location_spec_name: self.mamba_spec_size}
                    if self.has_mamba else {}
                ),
                **(
                    {self.indexer_location_spec_name: self.indexer_spec_size}
                    if self.has_indexer else {}
                ),
            },
        }
        self.transfer_client_config = json.dumps(transfer_client_json)

        # InitParams carries metadata consumed by the C++ SdkWrapper::Init:
        #   - self_location_spec_name: validated against location_spec_infos;
        #     also used to construct a unique Mooncake hostname when the
        #     Mooncake backend is present (format: {host}_{spec_name}_{rand}).
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
        # Extract all pools from HostPoolGroup.entries if available
        if hasattr(mem_pool_host, 'entries'):
            for entry in mem_pool_host.entries:
                self.registered_pools[entry.name] = entry.host_pool
                logger.info(
                    "register_mem_pool_host: found pool entry name=%s, "
                    "host_pool type=%s, is_anchor=%s",
                    entry.name,
                    type(entry.host_pool).__name__,
                    getattr(entry, 'is_primary_index_anchor', None),
                )
        else:
            self.registered_pools[PoolName.KV] = mem_pool_host
            logger.info(
                "register_mem_pool_host: single pool, type=%s",
                type(mem_pool_host).__name__,
            )
        logger.info(
            "register_mem_pool_host: registered_pools=%s",
            {k: type(v).__name__ for k, v in self.registered_pools.items()},
        )
        self._init_kvcm_client()

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        # All pools already extracted from HostPoolGroup in register_mem_pool_host,
        # so this is a no-op for KVCM connector.
        pass

    def _batch_get(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        trace_id: str,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Prepare keys
        block_keys, len_prefix, len_new = self._prepare_block_keys(keys, extra_info)

        locations = self._get_locations(trace_id, block_keys, "QT_PREFIX_MATCH", len_prefix,
                                         tag="v1")

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
        buffers = self._prepare_buffers(buffer_ptrs, buffer_sizes, self.kv_factor)
        assert len(uris) == len(buffers)
        # Perform data transfer
        start_time = time.perf_counter()
        result = self.transfer_client.LoadKvCaches(uris, buffers)
        end_time = time.perf_counter()
        self._record_bandwidth(matched, self.location_spec_size, end_time - start_time, is_read=True)
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

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        results = {}
        trace_id = self._get_trace_id()
        try:
            for transfer in transfers:
                spec_name = self._get_extra_pool_spec_name(transfer.name)
                pool = self.registered_pools.get(transfer.name)
                keys = transfer.keys or []
                if spec_name is None or pool is None or not keys:
                    results[transfer.name] = [False] * len(keys)
                    continue

                block_keys, _, _ = self._prepare_block_keys(keys)
                locations = self._get_locations(trace_id, block_keys, "QT_BATCH_GET", 0,
                                                 tag=f"v2 {transfer.name}")

                uris = []
                valid_indices = []
                for i, loc in enumerate(locations):
                    uri = self._extract_single_spec_uri(loc, spec_name)
                    if uri:
                        uris.append(uri)
                        valid_indices.append(i)

                if not uris:
                    results[transfer.name] = [False] * len(keys)
                    continue

                ptr_list, size_list = pool.get_page_buffer_meta(transfer.host_indices)
                components = self._get_extra_pool_components_per_page(transfer.name)
                valid_set = set(valid_indices)
                ptr_list = [p for i, p in enumerate(ptr_list) if (i // components) in valid_set]
                size_list = [s for i, s in enumerate(size_list) if (i // components) in valid_set]
                buffers = self._prepare_buffers(ptr_list, size_list, components)
                assert len(uris) == len(buffers)

                start_time = time.perf_counter()
                load_result = self.transfer_client.LoadKvCaches(uris, buffers)
                end_time = time.perf_counter()
                flag = (load_result == kvcm_py_client.ClientErrorCode.ER_OK)
                if flag:
                    self._record_bandwidth(
                        len(valid_indices),
                        self._get_extra_pool_spec_size(transfer.name),
                        end_time - start_time, is_read=True,
                    )
                per_key = [False] * len(keys)
                for idx in valid_indices:
                    per_key[idx] = flag
                results[transfer.name] = per_key

            return results
        except Exception as e:
            logger.error(f"batch_get_v2 failed: {trace_id=} {e=}")
            return {t.name: [False] * len(t.keys or []) for t in transfers}

    def _batch_set(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        trace_id: str,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Prepare keys
        block_keys, len_prefix, len_new = self._prepare_block_keys(keys, extra_info)
        local_len_new = len_new  # Preserve local key count for return value

        # When extra pools exist, use KV spec group to write KV specs only.
        location_spec_group_names = (
            [self._get_kv_spec_group()] * len(block_keys) if self.has_extra_pool else []
        )

        # Start write: rank 0 calls start_write_cache, broadcasts result + hash
        # + (len_prefix, len_new) so all ranks share the same block_mask parse.
        start_trace_id = f"start-{trace_id}"
        finish_trace_id = f"finish-{trace_id}"
        result, extras, skip_transfer = self._start_write(
            start_trace_id, block_keys, location_spec_group_names,
            extra_fields=[("len_prefix", len_prefix), ("len_new", len_new)],
            tag="v1",
        )
        # MLA non-rank-0 short circuits without participating in collectives.
        if result is None and skip_transfer:
            return [False] * local_len_new
        if result is None:
            return [False] * local_len_new
        len_prefix = extras["len_prefix"]
        len_new = extras["len_new"]

        locations = result["locations"]
        write_session_id = result["write_session_id"]
        block_mask = result["block_mask"]
        parsed = self._parse_block_mask(block_mask, len_prefix, len_new)

        # None means truly broken manager data -- treat as write failure.
        if parsed is None:
            logger.warning(f"_batch_set: inconsistent block_mask from manager, "
                           f"aborting write session {write_session_id}")
            self._finish_write(finish_trace_id, write_session_id,
                               [False] * len(locations), tag="v1")
            return [False] * local_len_new

        save_indices, prefix_write_count = parsed
        unmatched = len(save_indices)

        # Early return if all new blocks are already cached.
        if unmatched == 0:
            self._finish_write(finish_trace_id, write_session_id,
                               [False] * len(locations), tag="v1")
            return [False] * local_len_new if skip_transfer else [True] * local_len_new

        assert unmatched + prefix_write_count == len(locations)

        # Data transfer preparation and execution.
        # Skip prefix locations -- sglang cannot write prefix blocks.
        # Best-effort: each rank writes only the blocks it has local data for.
        # A per-block flag vector is all_reduced (MIN) so only blocks written
        # by ALL ranks are considered successful.
        # Wrapped in try-except so that every rank always reaches the
        # all_reduce below, preventing cross-rank NCCL/gloo hangs.
        new_locations = locations[prefix_write_count:]
        per_block_flags = torch.zeros(unmatched, dtype=torch.int)
        if skip_transfer:
            logger.warning("_batch_set: skipping data transfer on this rank due to input divergence")
        else:
            try:
                buffer_ptrs, buffer_sizes = self.mem_pool_host.get_page_buffer_meta(host_indices)
                local_block_count = len(buffer_ptrs) // self.kv_factor

                # Determine which save_indices have local data available
                valid_save_mask = [(idx < local_block_count) for idx in save_indices]
                valid_save_set = set(idx for idx, valid in zip(save_indices, valid_save_mask) if valid)
                num_valid = sum(valid_save_mask)

                if num_valid > 0:
                    buffer_ptrs = [ptr for i, ptr in enumerate(buffer_ptrs)
                                   if (i // self.kv_factor) in valid_save_set]
                    buffer_sizes = [sz for i, sz in enumerate(buffer_sizes)
                                    if (i // self.kv_factor) in valid_save_set]

                    # Extract URIs only for blocks with local data
                    valid_locations = [loc for loc, valid in zip(new_locations, valid_save_mask) if valid]
                    uris = self._extract_uris(valid_locations)
                    buffers = self._prepare_buffers(buffer_ptrs, buffer_sizes, self.kv_factor)
                    assert len(uris) == len(buffers)

                    # Perform data transfer
                    start_time = time.perf_counter()
                    save_result = self.transfer_client.SaveKvCaches(uris, buffers)
                    end_time = time.perf_counter()
                    self._record_bandwidth(num_valid, self.location_spec_size,
                                           end_time - start_time, is_read=False)
                    logger.debug(f"SaveKvCaches v1 {save_result=}")

                    transfer_ok = (save_result[0] == kvcm_py_client.ClientErrorCode.ER_OK)
                    if not transfer_ok:
                        logger.error(f"SaveKvCaches error: {save_result}")
                else:
                    transfer_ok = True  # nothing to write on this rank

                # Mark blocks this rank successfully wrote
                for j, valid in enumerate(valid_save_mask):
                    if valid and transfer_ok:
                        per_block_flags[j] = 1

            except Exception as e:
                logger.error(f"Data transfer (SaveKvCaches) failed: {e}")
                # per_block_flags remains all zeros

        # Per-block all_reduce: only blocks ALL ranks wrote are marked success
        per_block_flags = self._sync_per_block_flags(per_block_flags)

        new_block_success = [bool(per_block_flags[j]) for j in range(unmatched)]
        finish_mask = [False] * prefix_write_count + new_block_success
        commit_ok = self._finish_write(finish_trace_id, write_session_id,
                                       finish_mask, tag="v1")
        if not commit_ok:
            # Mark all as failed on rank 0 for the return value (see _finish_write docstring).
            new_block_success = [False] * unmatched

        # Build result list: 1:1 positional mapping with input keys.
        # - keys not in save_indices -> True (assumed cached / no-op)
        # - keys in save_indices     -> per-block success from all_reduce
        # When input diverged (skip_transfer), nothing was written and the
        # local keys don't match rank 0's -- return all False.
        if skip_transfer:
            return [False] * local_len_new
        block_flag_map = {save_indices[j]: new_block_success[j] for j in range(unmatched)}
        result_list = [
            block_flag_map.get(i, True)
            for i in range(local_len_new)
        ]
        return result_list

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

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        """Write extra pool data (e.g. Mamba/Indexer) in independent write sessions.

        Each PoolTransfer gets its own StartWriteCache -> Save -> FinishWriteCache
        using the pool's spec group, allowing writes to blocks whose KV cache
        was already committed in a separate write session.
        """
        results = {}
        trace_id = self._get_trace_id()
        try:
            for transfer in transfers:
                spec_name = self._get_extra_pool_spec_name(transfer.name)
                pool = self.registered_pools.get(transfer.name)
                keys = transfer.keys or []
                if spec_name is None or pool is None or not keys:
                    results[transfer.name] = [False] * len(keys)
                    continue

                spec_group = self._get_extra_pool_spec_group(transfer.name)
                block_keys, _, _ = self._prepare_block_keys(keys)
                spec_groups = [spec_group] * len(block_keys)

                v2_tag = f"v2 {transfer.name}"
                start_trace_id = f"start-v2-{trace_id}"
                finish_trace_id = f"finish-v2-{trace_id}"

                # Start write: same cross-rank protocol as v1 (hash-based
                # divergence detection, MLA non-rank-0 short-circuit).
                write_result, _, skip_transfer = self._start_write(
                    start_trace_id, block_keys, spec_groups, tag=v2_tag)

                if write_result is None:
                    # Includes both start_write_cache failure and MLA non-rank-0.
                    results[transfer.name] = [False] * len(keys)
                    continue

                locations = write_result["locations"]
                write_session_id = write_result["write_session_id"]
                block_mask = write_result["block_mask"]

                parsed = self._parse_block_mask(block_mask, 0, len(keys))
                if parsed is None:
                    logger.warning(f"batch_set_v2 {v2_tag}: inconsistent block_mask from manager, "
                                   f"aborting write session {write_session_id}")
                    self._finish_write(finish_trace_id, write_session_id,
                                       [False] * len(locations), tag=v2_tag)
                    results[transfer.name] = [False] * len(keys)
                    continue

                save_indices, prefix_write_count = parsed
                unmatched = len(save_indices)

                if unmatched == 0:
                    self._finish_write(finish_trace_id, write_session_id,
                                       [False] * len(locations), tag=v2_tag)
                    results[transfer.name] = (
                        [False] * len(keys) if skip_transfer else [True] * len(keys)
                    )
                    continue

                assert unmatched + prefix_write_count == len(locations)

                # Data transfer preparation and execution.
                # Best-effort: each rank writes only the blocks it has local
                # data for.  A per-block flag vector is all_reduced (MIN) so
                # only blocks written by ALL ranks are marked success.
                # Wrapped in try-except so every rank reaches the all_reduce.
                new_locations = locations[prefix_write_count:]
                per_block_flags = torch.zeros(unmatched, dtype=torch.int)
                if skip_transfer:
                    logger.warning(f"batch_set_v2 {v2_tag}: skipping data transfer due to input divergence")
                else:
                    try:
                        ptr_list, size_list = pool.get_page_buffer_meta(transfer.host_indices)
                        components = self._get_extra_pool_components_per_page(transfer.name)
                        local_block_count = len(ptr_list) // components

                        # Determine which save_indices have local data available
                        valid_save_mask = [(idx < local_block_count) for idx in save_indices]
                        valid_save_set = set(idx for idx, valid in zip(save_indices, valid_save_mask) if valid)
                        num_valid = sum(valid_save_mask)

                        if num_valid > 0:
                            ptr_list = [p for i, p in enumerate(ptr_list)
                                        if (i // components) in valid_save_set]
                            size_list = [s for i, s in enumerate(size_list)
                                         if (i // components) in valid_save_set]

                            valid_locations = [loc for loc, valid in zip(new_locations, valid_save_mask) if valid]
                            uris = []
                            for loc in valid_locations:
                                uri = self._extract_single_spec_uri(loc, spec_name)
                                if uri:
                                    uris.append(uri)
                            buffers = self._prepare_buffers(ptr_list, size_list, components)
                            assert len(uris) == len(buffers)

                            start_time = time.perf_counter()
                            save_result = self.transfer_client.SaveKvCaches(uris, buffers)
                            end_time = time.perf_counter()
                            logger.debug(f"SaveKvCaches {v2_tag} {save_result=}")

                            transfer_ok = (save_result[0] == kvcm_py_client.ClientErrorCode.ER_OK)
                            if transfer_ok:
                                self._record_bandwidth(
                                    num_valid, self._get_extra_pool_spec_size(transfer.name),
                                    end_time - start_time, is_read=False,
                                )
                            else:
                                logger.error(f"SaveKvCaches v2 error: {transfer.name} {save_result}")
                        else:
                            transfer_ok = True  # nothing to write on this rank

                        for j, valid in enumerate(valid_save_mask):
                            if valid and transfer_ok:
                                per_block_flags[j] = 1

                    except Exception as e:
                        logger.error(f"Data transfer v2 (SaveKvCaches) failed: {transfer.name} {e}")
                        # per_block_flags remains all zeros

                # Per-block MIN all_reduce (same as v1).
                per_block_flags = self._sync_per_block_flags(per_block_flags)

                new_block_success = [bool(per_block_flags[j]) for j in range(unmatched)]
                finish_mask = [False] * prefix_write_count + new_block_success
                commit_ok = self._finish_write(finish_trace_id, write_session_id,
                                               finish_mask, tag=v2_tag)
                if not commit_ok:
                    new_block_success = [False] * unmatched

                if skip_transfer:
                    results[transfer.name] = [False] * len(keys)
                else:
                    block_flag_map = {save_indices[j]: new_block_success[j] for j in range(unmatched)}
                    results[transfer.name] = [
                        block_flag_map.get(i, True)
                        for i in range(len(keys))
                    ]

            return results
        except Exception as e:
            logger.error(f"batch_set_v2 failed: {trace_id=} {e=}")
            return {t.name: [False] * len(t.keys or []) for t in transfers}

    def _batch_exists(
        self,
        keys: List[str],
        trace_id: str,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        block_keys, len_prefix, len_new = self._prepare_block_keys(keys, extra_info)
        return len(self._get_locations(trace_id, block_keys, "QT_PREFIX_MATCH", len_prefix,
                                         tag="exists"))

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

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        trace_id = self._get_trace_id()
        try:
            # Reuse the same get_cache_location call as batch_exists,
            # but inspect per-location specs for extra pool existence.
            block_keys, len_prefix, len_new = self._prepare_block_keys(keys, extra_info)
            locations = self._get_locations(trace_id, block_keys, "QT_PREFIX_MATCH", len_prefix,
                                             tag="exists_v2")

            # Count KV hit pages: prefix-match only locations that carry
            # the KV ("Full") spec.  get_cache_location returns any block
            # registered via start_write_cache regardless of spec group,
            # so we must filter by checking each location's specs.
            kv_hit_pages = 0
            for loc in locations:
                if any(
                    spec["name"] == self.location_spec_name
                    for spec in loc.get("location_specs", [])
                ):
                    kv_hit_pages += 1
                else:
                    break  # prefix match: stop at first gap
            pool_hit_pages = {PoolName.KV: kv_hit_pages} if kv_hit_pages else {}
            final_pages = kv_hit_pages

            # Check extra pool spec existence
            for transfer in (pool_transfers or []):
                if final_pages == 0:
                    break
                boundary = self._check_pool_spec_existence(
                    locations, kv_hit_pages, transfer
                )
                pool_hit_pages[transfer.name] = boundary
                final_pages = min(final_pages, boundary)

            return PoolTransferResult(final_pages, pool_hit_pages)
        except Exception as e:
            logger.error(f"batch_exists_v2 failed: {trace_id=} {e=}")
            return PoolTransferResult.empty()

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
    # ---- shared write protocol primitives (v1 + v2) ----
    # v1 and v2 share the same cross-rank protocol:
    #   rank 0 start_write_cache -> broadcast(result, hash, extras)
    #   per-block MIN all_reduce  -> rank 0 finish_write_cache
    # Only the data-transfer step (buffer layout, URI extraction) differs.

    def _start_write(self, trace_id: str, block_keys: List[int],
                     spec_group_names: List[str],
                     extra_fields: Optional[List[tuple]] = None,
                     tag: str = "") -> tuple[Optional[dict], dict, bool]:
        """Rank 0 initiates write session; broadcast result + hash (+ extras) to other ranks.

        Args:
            extra_fields: optional list of ``(name, value)`` pairs to sync
                alongside the write result (e.g. len_prefix/len_new for v1).
                Values come from rank 0 and overwrite other ranks' locals so
                every rank uses the same boundaries when parsing block_mask.

        Returns:
            (result, extras, skip_transfer) where
              - result: start_write_cache response on all ranks, or None on failure
              - extras: dict of synced extra_fields (rank 0's values on every rank)
              - skip_transfer: True if this rank must not touch storage
                (non-rank-0 hash divergence, or MLA non-rank-0)
        """
        extra_fields = extra_fields or []

        # MLA: only rank 0 owns data. Non-rank-0 ranks short-circuit as no-op.
        if self.is_mla_model and self.tp_rank != 0:
            logger.warning(f"_start_write {tag}: non-rank-0 (tp_rank={self.tp_rank}) "
                           f"on MLA model; skipping write.")
            return None, {name: val for name, val in extra_fields}, True

        local_hash = hash(tuple(block_keys) + tuple(v for _, v in extra_fields))

        if self.tp_rank == 0:
            request = {
                "trace_id": trace_id,
                "instance_id": self.instance_id,
                "block_keys": block_keys,
                "location_spec_group_names": spec_group_names,
                "write_timeout_seconds": self.write_timeout_seconds,
            }
            logger.debug(f"start_write_cache {tag} {request=}")
            try:
                result = self._manager_client.start_write_cache(request)
            except Exception as e:
                logger.error(f"start_write_cache {tag} failed: {e}")
                result = None
            logger.debug(f"start_write_cache {tag} {result=}")

            if self.tp_world_size > 1 and not self.is_mla_model:
                payload = [result, local_hash] + [v for _, v in extra_fields]
                torch.distributed.broadcast_object_list(
                    payload, src=0, group=self.storage_tp_group
                )
            return result, {name: val for name, val in extra_fields}, False

        # non-rank-0: receive and validate
        recv = [None] * (2 + len(extra_fields))
        torch.distributed.broadcast_object_list(
            recv, src=0, group=self.storage_tp_group
        )
        result = recv[0]
        rank0_hash = recv[1]
        extras = {name: recv[2 + i] for i, (name, _) in enumerate(extra_fields)}

        skip_transfer = (local_hash != rank0_hash)
        if skip_transfer:
            logger.warning(f"_start_write {tag}: local hash ({local_hash}) != "
                           f"rank 0 hash ({rank0_hash}), inputs diverged across TP ranks")
        return result, extras, skip_transfer

    def _finish_write(self, trace_id: str, write_session_id: str,
                      mask: List[bool], tag: str = "") -> bool:
        """Rank 0 commits write result to manager.

        Returns True if commit succeeded (or rank != 0, which is a no-op).
        On rank 0 failure we swallow the exception: adding a second all_reduce
        just for this rare error path would penalise the hot path, so the
        inconsistency is accepted and the caller should tolerate subsequent
        batch_get misses gracefully.
        """
        if self.tp_rank != 0:
            return True
        try:
            self._manager_client.finish_write_cache({
                "trace_id": trace_id,
                "instance_id": self.instance_id,
                "write_session_id": write_session_id,
                "success_blocks": {"bool_masks": {"values": mask}},
            })
            logger.debug(f"finish_write_cache {tag} session={write_session_id}")
            return True
        except Exception as e:
            logger.error(f"finish_write_cache {tag} failed: {e}")
            return False

    def _sync_per_block_flags(self, flags: torch.Tensor) -> torch.Tensor:
        """Per-block MIN all_reduce: only blocks ALL ranks wrote are marked success.

        MLA models only write on rank 0, so all_reduce would mix real flags from
        rank 0 with meaningless zeros from other ranks.  Skip the collective
        entirely in that case.
        """
        if self.tp_world_size <= 1 or self.is_mla_model:
            return flags
        torch.distributed.all_reduce(
            flags, op=torch.distributed.ReduceOp.MIN,
            group=self.storage_tp_group,
        )
        return flags

    def _get_locations(self, trace_id: str, block_keys: List[int],
                       query_type: str, offset: int,
                       tag: str = "") -> List[dict]:
        """Query manager for cache location list."""
        result = self._manager_client.get_cache_location({
            "trace_id": trace_id,
            "block_keys": block_keys,
            "instance_id": self.instance_id,
            "query_type": query_type,
            "block_mask": {"offset": offset},
        })
        logger.debug(f"get_cache_location {tag} {result=}")
        return result["locations"]

    def _record_bandwidth(self, pages: int, spec_size: int,
                          elapsed: float, is_read: bool) -> None:
        target_pgs = self.prefetch_pgs if is_read else self.backup_pgs
        target_bw = self.prefetch_bandwidth if is_read else self.backup_bandwidth
        target_pgs.append(pages)
        target_bw.append(pages * spec_size / (1 << 30) / elapsed)

    def _tp_rank_to_spec_name(self, tp_rank: int) -> str:
        # For pure FullAttention models (no Mamba/Indexer), use old format "tp_{rank}"
        # for backward compatibility with existing cached data.
        # For hybrid models, use "tp_{rank}_full" to distinguish KV specs from
        # Mamba/SSM specs ("tp_{rank}_linear") and Indexer specs.
        if self.has_mamba or self.has_indexer:
            return f"tp_{tp_rank}_full"
        return f"tp_{tp_rank}"

    def _tp_rank_to_linear_spec_name(self, tp_rank: int) -> str:
        return f"tp_{tp_rank}_linear"

    def _tp_rank_to_indexer_spec_name(self, tp_rank: int) -> str:
        return f"tp_{tp_rank}_indexer"

    def _get_trace_id(self) -> str:
        return str(uuid.uuid1())

    def _sha256_to_int64(self, data: str) -> int:
        data = data.encode("utf-8")
        hash_digest = hashlib.sha256(data).digest()
        hash_int64 = int.from_bytes(hash_digest[:8], "big", signed=True)
        return hash_int64

    def _prepare_block_keys(
            self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None) -> tuple[List[int], int, int]:
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

    def _make_iov(self, base_ptr: int, size: int) -> kvcm_py_client.Iov:
        iov = kvcm_py_client.Iov()
        iov.type = kvcm_py_client.MemoryType.CPU
        iov.base = base_ptr
        iov.size = size
        iov.ignore = False
        return iov

    def _prepare_buffers(self, ptr_list: List[int], size_list: List[int],
                         components: int) -> List[kvcm_py_client.BlockBuffer]:
        """Prepare buffers for data transfer.

        Groups ptr/size pairs into BlockBuffers, each containing `components` IOVs.
        Works for KV (components=kv_factor), Mamba (temporal + conv), and Indexer (1).
        """
        buffers = []
        for i in range(0, len(ptr_list), components):
            buffer = kvcm_py_client.BlockBuffer()
            buffer.iovs = [
                self._make_iov(ptr_list[i + j], size_list[i + j])
                for j in range(components)
            ]
            buffers.append(buffer)
        return buffers

    def _parse_block_mask(self, block_mask: dict, len_prefix: int, len_new: int) -> Optional[tuple[List[int], int]]:
        """Parse block_mask from manager to determine which new-block indices need writing.

        Returns:
            tuple[List[int], int]:
                - save_indices: indices (relative to new blocks) that need writing.
                  Empty list means all new blocks are already cached.
                - prefix_write_count: number of prefix blocks the manager wants
                  written that we cannot fulfil (best-effort skip).
            None: manager returned truly broken data (e.g. incomplete bool_masks);
                  caller should treat as a total write failure.
        """
        save_indices = []
        prefix_write_count = 0
        if "offset" in block_mask:
            offset = block_mask["offset"]
            if offset < len_prefix:
                # Best-effort: prefix blocks [offset, len_prefix) can't be written
                # by sglang (no data available), but new blocks can still proceed.
                logger.warning(f"_parse_block_mask: offset {offset} < len_prefix {len_prefix}, "
                               "prefix blocks will be skipped (best-effort)")
                prefix_write_count = len_prefix - offset
                save_indices.extend(range(len_prefix, len_prefix + len_new))
            else:
                save_indices.extend(range(offset, len_prefix + len_new))
        else:
            # False: need to store
            bool_masks = block_mask.get("bool_masks", {}).get("values", [])
            if len(bool_masks) < len_prefix + len_new:
                # Incomplete mask data from manager.
                logger.warning(f"_parse_block_mask: bool_masks length {len(bool_masks)} < "
                               f"expected {len_prefix + len_new}, treating as inconsistent state")
                return None
            prefix_write_count = sum(1 for v in bool_masks[:len_prefix] if not v)
            if prefix_write_count > 0:
                logger.warning(f"_parse_block_mask: {prefix_write_count} prefix blocks "
                               "not cached in bool_masks, will be skipped (best-effort)")
            max_index = max([i for i, x in enumerate(bool_masks) if not x], default=-1)
            save_indices.extend([i for i in range(len_prefix, max_index + 1) if not bool_masks[i]])
        save_indices = [(i - len_prefix) for i in save_indices if i >= len_prefix]
        return save_indices, prefix_write_count

    def _extract_single_spec_uri(self, location, spec_name: str):
        """Extract the URI for a named spec from a single location dict."""
        for spec in location.get("location_specs", []):
            if spec["name"] == spec_name and spec.get("uri"):
                return spec["uri"]
        return None

    def _get_extra_pool_components_per_page(self, pool_name: str) -> int:
        """Number of IOV components per logical page for an extra pool."""
        if pool_name == PoolName.MAMBA:
            mamba_pool = self.registered_pools.get(PoolName.MAMBA)
            conv_num = len(getattr(mamba_pool, "conv_buffer", []) or [])
            return 1 + conv_num  # temporal + N conv
        if pool_name == PoolName.INDEXER:
            return 1  # single indexer buffer per page
        return 1

    def _get_kv_spec_group(self) -> str:
        """Spec group name used in start_write_cache for KV pool."""
        return "Full"

    def _get_extra_pool_spec_group(self, pool_name: str) -> str:
        """Spec group name used in start_write_cache for an extra pool."""
        if pool_name == PoolName.MAMBA:
            return "Linear"
        if pool_name == PoolName.INDEXER:
            return "Indexer"
        raise ValueError(f"Unknown extra pool: {pool_name}")

    def _get_extra_pool_spec_size(self, pool_name: str) -> int:
        """Per-block spec size in bytes for bandwidth tracking."""
        if pool_name == PoolName.MAMBA:
            return self.mamba_spec_size
        if pool_name == PoolName.INDEXER:
            return self.indexer_spec_size
        return 0


    def _get_extra_pool_spec_name(self, pool_name: str) -> Optional[str]:
        """Map a PoolName to its KVCM location spec name for the current rank."""
        if pool_name == PoolName.MAMBA and self.has_mamba:
            return self.mamba_location_spec_name
        if pool_name == PoolName.INDEXER and self.has_indexer:
            return self.indexer_location_spec_name
        return None

    def _check_pool_spec_existence(self, locations, kv_hit_pages, transfer):
        """Check how many pages have the extra pool's spec.

        Returns the number of contiguous prefix pages that satisfy
        transfer.hit_policy.  Falls back to 0 for unknown policies so
        a stale/future enum value never causes an UnboundLocalError crash.
        """
        spec_name = self._get_extra_pool_spec_name(transfer.name)
        if spec_name is None:
            return kv_hit_pages

        def has_spec(loc):
            return any(
                spec["name"] == spec_name for spec in loc.get("location_specs", [])
            )

        if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
            # First gap in the prefix is the boundary.
            return next(
                (i for i in range(kv_hit_pages) if not has_spec(locations[i])),
                kv_hit_pages,
            )

        if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
            trailing = max(1, len(transfer.keys) if transfer.keys else 1)
            for prefix_len in range(kv_hit_pages, 0, -1):
                if all(
                    has_spec(locations[i])
                    for i in range(max(0, prefix_len - trailing), prefix_len)
                ):
                    return prefix_len
            return 0

        logger.warning(
            "_check_pool_spec_existence: unknown PoolHitPolicy %r, defaulting to 0",
            transfer.hit_policy,
        )
        return 0

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
