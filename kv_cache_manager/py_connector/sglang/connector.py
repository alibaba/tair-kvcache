import hashlib
import logging
import threading
import uuid
from typing import AbstractSet, Any, Dict, List, Optional
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

try:
    from sglang.srt.observability.metrics_collector import StorageMetrics
except ImportError:
    try:
        # Older sglang versions kept StorageMetrics here.
        from sglang.srt.metrics.collector import StorageMetrics  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise ImportError(
            "Cannot import StorageMetrics from sglang. "
            "Tried sglang.srt.observability.metrics_collector and "
            "sglang.srt.metrics.collector. "
            "Please check your sglang version is compatible."
        ) from e

# get_attn_tp_group lives in sglang.srt.distributed since v0.5.9; the old
# alias sglang.srt.layers.dp_attention.get_attention_tp_group was removed in
# v0.5.16.
from sglang.srt.distributed import get_attn_tp_group, get_tp_group
from sglang.srt.layers.dp_attention import is_dp_attention_enabled

from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient

# kvcm_py_client is the compiled pybind11 client; it ships no type stubs.
from kv_cache_manager.client.pybind import kvcm_py_client  # ty: ignore[unresolved-import]

# Stamped into the wheel at build time; absent in a source checkout.
from kv_cache_manager.py_connector.common._version_info import (  # ty: ignore[unresolved-import]
    FULL_VERSION,
    GIT_COMMIT,
    BUILD_TIME,
)

logger = logging.getLogger(__name__)


class _UnsupportedBackendError(RuntimeError):
    """The connector cannot serve this backend, and retrying cannot change it.

    Raised by ``_ensure_client`` for permanent shape mismatches -- e.g. a KV
    anchor that is not a host pool (sglang's DeepSeek-V4 compressed pool).
    The problem is reported once when it is detected, so the data-path
    entries answer conservatively on every call without logging once per call.
    """


class HiCacheKVCM(HiCacheStorage):
    """KVCM HiCache storage backend for sglang (v0.5.10 ~ v0.5.19).

    Pool registration contract -- what sglang hands the backend, in order:

    * ``register_mem_pool_host`` (v1) runs first on every attach. It carries
      the KV host pool for flat models on all supported versions and, since
      v0.5.19, for every model (sglang passes ``HostPoolGroup.anchor_entry.host_pool``).
      Up to v0.5.18 hybrid stacks pass the ``HostPoolGroup`` itself instead;
      a group is not a ``HostKVCache``, so its KV entry reaches the connector
      through ``register_mem_host_pool_v2`` right after.
    * ``register_mem_host_pool_v2`` carries one host pool per call and is the
      only source of sidecar pools (Mamba/Indexer/...). sglang calls it for
      every entry of the group, KV included.

    No storage call can happen before the attach returns, so the pool set is
    final at the first data-path call -- and that is exactly when the Manager
    registration and the SDK client are created (``_ensure_client``):
    ``registerInstance`` only accepts a repeat when every field matches and
    there is no spec update API, so the location specs must be frozen once.
    A pool registered after that point is reported once, not recorded, and
    marked unusable: even if it reuses the name of an already-registered
    pool, its transfers are answered with misses instead of being paired with
    that pool's spec.  It is never silently assumed to work.

    If the KV anchor itself cannot be served -- sglang's DeepSeek-V4
    compressed KV pool anchors on a tensorless ``LogicalHostPool`` that is not
    a host pool -- initialization reports it once and every storage call for
    that instance degrades to a miss.
    """

    # Pools already reported as unusable -- either unknown to this connector
    # or registered after the specs were frozen -- and therefore answered with
    # misses. Process-wide so each pool is reported once instead of once per
    # block batch, and so the report survives a backend re-creation.
    _reported_pools: set = set()

    # Immutable so hand-built instances (tests bypass __init__) can read it
    # without sharing one mutable object: late names are added by rebinding,
    # never by mutating a shared set.
    _late_pools: AbstractSet[Any] = frozenset()

    def __init__(self, storage_config: HiCacheStorageConfig, kwargs: Any) -> None:
        logger.warning(
            "KVCM sglang connector version: %s (commit: %s, build: %s)",
            FULL_VERSION,
            GIT_COMMIT,
            BUILD_TIME,
        )
        self.storage_config = storage_config
        # --hicache-storage-backend-extra-config '{"k":"v"}'
        # HiCacheStorageConfig types extra_config as Optional; sglang always
        # provides it for a hicache storage backend.
        self.extra_config: Dict[str, Any] = self.storage_config.extra_config or {}

        # deployment
        self.instance_group = self.extra_config["instance_group"]
        self.instance_id = self.extra_config["instance_id"]

        self._manager_client = KvCacheManagerClient.from_connector_config(
            self.extra_config
        )

        # PoolName -> the pool object; sglang's pool classes expose more than
        # the HostKVCache base (e.g. get_page_buffer_meta), hence Any.
        self.registered_pools: Dict[Any, Any] = {}

        # Pool names that arrived after the location specs were frozen (or
        # after close()): the Manager has no spec update API, and such a pool
        # may even reuse the name of an already-registered pool -- whose spec
        # the new host indices do not belong to.  The data path answers miss
        # for these names instead of pairing them with a registered spec.
        self._late_pools = set()

        # _init_lock guards the one-time client initialization; the two flags
        # are instance state (class-level defaults keep hand-built instances
        # -- tests -- from raising on the first check).
        self._init_lock = threading.Lock()
        self._client_ready = False
        self._closed = False

        self.prefetch_pgs = []
        self.backup_pgs = []
        self.prefetch_bandwidth = []
        self.backup_bandwidth = []

        # Declare v1 interface support so that sglang's cache_controller uses
        # batch_set_v1/batch_get_v1 (zero-copy path) instead of the legacy
        # batch_set/batch_get. This avoids requiring users to manually add
        # "interface_v1": 1 in --hicache-storage-backend-extra-config.
        self.extra_config.setdefault("interface_v1", 1)

    def _init_parallel_context(self) -> None:
        """Resolve the TP group and create the connector's own gloo group.

        COLLECTIVE: ``torch.distributed.new_group`` requires every rank of the
        default group to call it in the same order, so this stays in the v1
        hook -- the attach path, where all ranks meet in lockstep. Everything
        else is deferred to the first storage call (``_ensure_client``).
        """
        # parallelism
        self.tp_rank = self.storage_config.tp_rank
        self.tp_size = self.storage_config.tp_size
        # TODO: pp
        self.dp_size = 1
        self.pp_size = 1

        tp_group = (
            get_attn_tp_group().cpu_group
            if is_dp_attention_enabled()
            else get_tp_group().cpu_group
        )
        self.tp_world_size = torch.distributed.get_world_size(group=tp_group)
        if self.tp_world_size > 1:
            group_ranks = torch.distributed.get_process_group_ranks(tp_group)
            self.storage_tp_group = torch.distributed.new_group(
                group_ranks, backend="gloo"
            )

    def _ensure_client(self) -> None:
        """Register with the Manager and create the SDK client, once.

        Deliberately deferred to the first storage call: sglang's registration
        burst (v1 anchor, then one v2 call per pool) has to be over before the
        location specs are frozen into ``registerInstance`` -- a repeat with a
        different configuration is rejected and there is no spec update API.
        Pools registered later are degraded, see ``register_mem_host_pool_v2``.

        No collective happens here, so ranks may initialize at different times.
        Raises on failure: the ``batch_*`` callers turn that into a
        conservative result and retry on their next call.
        """
        if self._client_ready:
            return
        with self._init_lock:
            if self._client_ready:
                return
            if self._closed:
                raise RuntimeError(
                    "connector was closed; storage stays disabled until the "
                    "backend is re-attached"
                )
            kv_pool = self.registered_pools.get(PoolName.KV)
            if kv_pool is None:
                raise RuntimeError(
                    "no KV host pool registered yet; storage calls require "
                    "sglang's register_mem_pool_host/register_mem_host_pool_v2"
                )
            if not isinstance(kv_pool, HostKVCache):
                # Permanent, so it is reported once here instead of failing
                # with an ERROR per storage call: sglang's DeepSeek-V4
                # compressed pool anchors its KV entry on a LogicalHostPool,
                # which has no get_size_per_token() to build specs from.
                self._report_unsupported_kv_pool(kv_pool)
                raise _UnsupportedBackendError(
                    f"the KV anchor of this instance is a "
                    f"{type(kv_pool).__name__}, not a host pool; this pool "
                    f"shape is not supported by this connector"
                )
            self._init_kvcm_client()
            self._client_ready = True

    def _init_kvcm_client(self) -> None:
        # model
        self.model_name = self.storage_config.model_name
        self.is_mla_model = self.storage_config.is_mla_model
        self.kv_factor = 1 if self.is_mla_model else 2
        # The KV pool comes from the v1 hook (plain host pool) or from the v2
        # hook when v1 only carried a HostPoolGroup; _ensure_client rejects an
        # empty pool set before calling this.
        kv_pool = self.registered_pools[PoolName.KV]
        self.kv_dtype = kv_pool.dtype

        # manager
        self.block_size = self.mem_pool_host.page_size

        # Detect extra pools early — _tp_rank_to_spec_name depends on these.
        self.has_mamba = PoolName.MAMBA in self.registered_pools
        self.has_indexer = (
            getattr(PoolName, "INDEXER", None) is not None
            and PoolName.INDEXER in self.registered_pools
        )

        self.location_spec_size = kv_pool.get_size_per_token() * self.block_size
        self.location_spec_infos = [
            {
                "name": self._tp_rank_to_spec_name(rank),
                "size": self.location_spec_size,
            }
            for rank in range(self.tp_size)
        ]

        # LocationSpecGroup: KV specs always prepared, but only sent when
        # extra pools exist (backward compat with older managers).
        kv_spec_names = [
            self._tp_rank_to_spec_name(rank) for rank in range(self.tp_size)
        ]
        self.location_spec_groups = []

        # Mamba/Linear specs
        if self.has_mamba:
            mamba_pool = self.registered_pools[PoolName.MAMBA]
            # No "* block_size" here, unlike KV and Indexer above: a mamba host
            # pool is page-granular (MambaPoolHost.page_size == 1), so
            # get_size_per_token() already returns the bytes of one slot, i.e.
            # of one block, which is what a location spec size means.
            self.mamba_spec_size = mamba_pool.get_size_per_token()
            linear_spec_names = []
            for rank in range(self.tp_size):
                name = self._tp_rank_to_linear_spec_name(rank)
                self.location_spec_infos.append(
                    {"name": name, "size": self.mamba_spec_size}
                )
                linear_spec_names.append(name)
            mamba_spec_rank = 0 if self.is_mla_model else self.tp_rank
            self.mamba_location_spec_name = self._tp_rank_to_linear_spec_name(
                mamba_spec_rank
            )
            self.location_spec_groups.append(
                {
                    "name": self._get_extra_pool_spec_group(PoolName.MAMBA),
                    "spec_names": linear_spec_names,
                }
            )

        # Indexer specs (NSA/DSA)
        if self.has_indexer:
            indexer_pool = self.registered_pools[PoolName.INDEXER]
            self.indexer_spec_size = indexer_pool.get_size_per_token() * self.block_size
            indexer_spec_names = []
            for rank in range(self.tp_size):
                name = self._tp_rank_to_indexer_spec_name(rank)
                self.location_spec_infos.append(
                    {"name": name, "size": self.indexer_spec_size}
                )
                indexer_spec_names.append(name)
            indexer_spec_rank = 0 if self.is_mla_model else self.tp_rank
            self.indexer_location_spec_name = self._tp_rank_to_indexer_spec_name(
                indexer_spec_rank
            )
            self.location_spec_groups.append(
                {
                    "name": self._get_extra_pool_spec_group(PoolName.INDEXER),
                    "spec_names": indexer_spec_names,
                }
            )

        if self.location_spec_groups:
            self.location_spec_groups.insert(
                0,
                {
                    "name": self._get_kv_spec_group(),
                    "spec_names": kv_spec_names,
                },
            )

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

        # data transfer setup
        # MLA: only rank 0 writes, so all ranks use rank 0's spec for read/write
        kv_spec_rank = 0 if self.is_mla_model else self.tp_rank
        self.location_spec_name = self._tp_rank_to_spec_name(kv_spec_rank)

        self.write_timeout_seconds = self.extra_config.get("write_timeout_seconds", 30)

        # sdk
        self.sdk_thread_num = self.extra_config.get("sdk_thread_num", 4)
        self.sdk_queue_size = self.extra_config.get("sdk_queue_size", 1000)
        self.sdk_get_timeout_ms = self.extra_config.get("sdk_get_timeout_ms", 15000)
        self.sdk_put_timeout_ms = self.extra_config.get("sdk_put_timeout_ms", 15000)

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
                    if self.has_mamba
                    else {}
                ),
                **(
                    {self.indexer_location_spec_name: self.indexer_spec_size}
                    if self.has_indexer
                    else {}
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
        assert self.transfer_client is not None, (
            "kvcm_py_client.TransferClient.Create failed"
        )

    def parse_hf3fs_configs(self, storage_configs: str) -> List[Dict[str, Any]]:
        hf3fs_configs = []
        storage_configs_json = json.loads(storage_configs)
        for storage_config in storage_configs_json:
            storage_type = storage_config["type"]
            if storage_type not in ("hf3fs", "vcns_hf3fs") or not storage_config.get(
                "is_available", True
            ):
                continue
            hf3fs_config = {
                "type": storage_type,
                "mountpoint": storage_config["storage_spec"]["mountpoint"],
                "root_dir": storage_config["storage_spec"]["root_dir"],
                "read_iov_block_size": self.read_iov_block_size,
                "read_iov_size": self.iov_size,
                "write_iov_block_size": self.write_iov_block_size,
                "write_iov_size": self.iov_size,
            }
            hf3fs_configs.append(hf3fs_config)
        return hf3fs_configs

    def register_mem_pool_host(self, mem_pool_host: HostKVCache) -> None:
        """v1 hook: the KV (anchor) host pool, or a <= 0.5.18 HostPoolGroup.

        The object is kept for the v1 data path (``get_page_buffer_meta`` and
        ``page_size``); pools themselves are registered from the v2 hook only,
        and the Manager is told about them on the first storage call.
        """
        # The pool objects expose more than the HostKVCache base
        # (get_page_buffer_meta & co).
        self.mem_pool_host: Any = mem_pool_host
        self._init_parallel_context()

        # A host pool handed over here is the KV pool: flat models on every
        # version, and every model on sglang >= 0.5.19 (sglang passes the
        # HostPoolGroup's anchor entry). Up to v0.5.18 a hybrid stack passes
        # the group itself, which is not a HostKVCache -- the KV entry then
        # arrives through register_mem_host_pool_v2 right after.
        if isinstance(mem_pool_host, HostKVCache):
            self.registered_pools[PoolName.KV] = mem_pool_host
        logger.info(
            "register_mem_pool_host: type=%s, kv_pool_registered=%s",
            type(mem_pool_host).__name__,
            PoolName.KV in self.registered_pools,
        )

    def register_mem_host_pool_v2(
        self, host_pool: HostKVCache, host_pool_name: str
    ) -> None:
        """v2 hook: one host pool per call, sidecars included.

        sglang calls this for every entry of its host pool group, the KV
        anchor included, right after the v1 hook. There is no "registration
        finished" callback, so the pool set is only known to be final at the
        first data-path call -- pools showing up after that cannot be added to
        the Manager registration and are degraded (see the class docstring).

        The initialization lock is taken: a pool arriving while the specs are
        being frozen would otherwise be recorded but missing from the
        registration payload, i.e. silently unusable.
        """
        with self._init_lock:
            if self._client_ready or self._closed:
                # Too late for this instance.  Do not record it: if it reuses
                # a managed pool's name, the data path must not pick it up
                # (its memory is not the memory the registered spec refers
                # to).  ``_late_pools`` keeps the name unusable explicitly;
                # rebind instead of add() so the class-level default stays
                # immutable.
                self._late_pools = self._late_pools | {host_pool_name}
                self._report_late_pool(host_pool_name)
                return
            self.registered_pools[host_pool_name] = host_pool
            logger.info(
                "register_mem_host_pool_v2: pool=%s, type=%s",
                host_pool_name,
                type(host_pool).__name__,
            )

    def _report_late_pool(self, pool_name: Any) -> None:
        """Report once that a pool arrived after the specs were frozen."""
        if not self._mark_pool_reported(pool_name):
            return
        logger.error(
            "register_mem_host_pool_v2: pool %s was registered after the "
            "connector was initialized with the Manager; there is no "
            "specification update API, so this pool stays unsupported and its "
            "transfers are reported as misses.",
            pool_name,
        )

    def _mark_pool_reported(self, pool_name: Any) -> bool:
        """Record that a pool has been reported, returning False if it was."""
        if pool_name in self._reported_pools:
            return False
        self._reported_pools.add(pool_name)
        return True

    def _report_unsupported_kv_pool(self, pool: Any) -> None:
        """Report once that the KV anchor is not a host pool.

        Seen with sglang's DeepSeek-V4 compressed KV pool, whose anchor is a
        ``LogicalHostPool``: it holds no tensors and no ``get_size_per_token``,
        so no location spec can be built for it.  The report is deduplicated
        through ``_reported_pools`` like every other unusable pool, so a
        re-created backend in the same process stays quiet.
        """
        if not self._mark_pool_reported(PoolName.KV):
            return
        logger.error(
            "the KV anchor of this instance is a %s, not a host pool: this "
            "pool shape (e.g. sglang's DeepSeek-V4 compressed KV pool) is not "
            "supported by this connector, so every storage call degrades to a "
            "miss for this instance.",
            type(pool).__name__,
        )

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
        buffer_ptrs, buffer_sizes = self.mem_pool_host.get_page_buffer_meta(
            host_indices
        )
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
        self.prefetch_bandwidth.append(
            matched * self.location_spec_size / (1 << 30) / (end_time - start_time)
        )
        logger.debug(f"LoadKvCaches {result=}")

        flag = result == kvcm_py_client.ClientErrorCode.ER_OK
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
            self._ensure_client()
            result = self._batch_get(
                keys=keys,
                host_indices=host_indices,
                trace_id=trace_id,
                extra_info=extra_info,
            )
            return result
        except _UnsupportedBackendError:
            # Already reported once by _ensure_client; keep answering
            # conservatively without logging on every call.
            return [False] * len(keys)
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
            self._ensure_client()
            for transfer in transfers:
                spec_name = self._get_extra_pool_spec_name(transfer.name)
                pool = self.registered_pools.get(transfer.name)
                keys = transfer.keys or []
                if not keys:
                    results[transfer.name] = []
                    continue
                if (
                    transfer.name in self._late_pools
                    or spec_name is None
                    or pool is None
                ):
                    self._report_unknown_pool(transfer.name)
                    results[transfer.name] = [False] * len(keys)
                    continue

                block_keys, _, _ = self._prepare_block_keys(keys)
                get_request = {
                    "trace_id": trace_id,
                    "block_keys": block_keys,
                    "instance_id": self.instance_id,
                    "query_type": "QT_BATCH_GET",
                    "block_mask": {"offset": 0},
                }
                result = self._manager_client.get_cache_location(get_request)
                logger.debug(f"get_cache_location v2 {transfer.name} {result=}")
                locations = result["locations"]

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

                ptr_list, size_list, components = self._page_buffer_meta(pool, transfer)
                ptr_list = [
                    p
                    for i, p in enumerate(ptr_list)
                    if (i // components) in valid_indices
                ]
                size_list = [
                    s
                    for i, s in enumerate(size_list)
                    if (i // components) in valid_indices
                ]
                buffers = self._prepare_extra_pool_buffers(
                    ptr_list, size_list, components
                )
                assert len(uris) == len(buffers)

                start_time = time.perf_counter()
                load_result = self.transfer_client.LoadKvCaches(uris, buffers)
                end_time = time.perf_counter()
                flag = load_result == kvcm_py_client.ClientErrorCode.ER_OK
                if flag:
                    spec_size = self._get_extra_pool_spec_size(transfer.name)
                    self.prefetch_pgs.append(len(valid_indices))
                    self.prefetch_bandwidth.append(
                        len(valid_indices)
                        * spec_size
                        / (1 << 30)
                        / (end_time - start_time)
                    )
                per_key = [False] * len(keys)
                for idx in valid_indices:
                    per_key[idx] = flag
                results[transfer.name] = per_key

            return results
        except _UnsupportedBackendError:
            # Already reported once by _ensure_client; keep answering
            # conservatively without logging on every call.
            return {t.name: [False] * len(t.keys or []) for t in transfers}
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
        # NOTE on cross-rank consistency:
        # start_write_cache and SaveKvCaches failures are synchronised
        # across ranks (via broadcast / all_reduce) so every rank
        # converges on the same result.
        #
        # finish_write_cache is called only on rank 0 *after* the
        # all_reduce has already completed. If it throws, rank 0
        # overrides flag to False while other ranks keep flag = True.
        # Adding a second all_reduce just for this error path would
        # penalise the hot path for a rare event. The practical
        # consequence is that rank 1+ callers may believe the write
        # succeeded, but the manager was never told to commit, so a
        # subsequent batch_get on those blocks will miss. This is an
        # accepted inconsistency -- the caller should tolerate cache
        # misses gracefully.

        # Prepare keys
        block_keys, len_prefix, len_new = self._prepare_block_keys(keys, extra_info)
        local_len_new = len_new  # Preserve local key count for return value
        local_hash = hash(
            (len_prefix, len_new, *block_keys)
        )  # Hash covers prefix/new boundary + all keys

        # Start write cache
        if self.tp_rank == 0:
            start_trace_id = f"start-{trace_id}"
            # When extra pools exist, use KV spec group to write KV specs only
            has_extra_pools = self.has_mamba or self.has_indexer
            location_spec_group_names = (
                [self._get_kv_spec_group()] * len(block_keys) if has_extra_pools else []
            )
            request = {
                "trace_id": start_trace_id,
                "instance_id": self.instance_id,
                "block_keys": block_keys,
                "location_spec_group_names": location_spec_group_names,
                "write_timeout_seconds": self.write_timeout_seconds,
            }
            logger.debug(f"start_write_cache {request=}")
            try:
                result = self._manager_client.start_write_cache(request)
            except Exception as e:
                logger.error(f"start_write_cache failed: {e}")
                result = None

            if self.tp_world_size > 1 and not self.is_mla_model:
                torch.distributed.broadcast_object_list(
                    [result, len_prefix, len_new, local_hash],
                    src=0,
                    group=self.storage_tp_group,
                )
        elif self.is_mla_model:
            logger.warning(
                f"_batch_set called on non-rank-0 (tp_rank={self.tp_rank}) "
                f"for MLA model; only rank 0 should write. Returning all False."
            )
            return [False] * len_new
        else:
            recv: List[Any] = [None, None, None, None]
            torch.distributed.broadcast_object_list(
                recv, src=0, group=self.storage_tp_group
            )
            result, len_prefix, len_new, rank0_hash = recv

        logger.debug(f"start_write_cache {result=}")

        # All ranks now share rank 0's len_prefix/len_new so that
        # _parse_block_mask produces the same save_indices everywhere,
        # preventing control-flow divergence (and NCCL hangs).
        # We also compare a hash of the full block_keys list: if a non-rank-0
        # rank's local block_keys differ from rank 0's, writing would corrupt
        # storage. The rank still participates in all_reduce with all-zero
        # flags so NCCL doesn't hang.
        skip_transfer = False
        if self.tp_rank != 0 and local_hash != rank0_hash:
            logger.warning(
                f"_batch_set: local block_keys hash ({local_hash}) != "
                f"rank 0 hash ({rank0_hash}), inputs diverged across TP ranks. "
                f"local_block_keys={block_keys}"
            )
            skip_transfer = True

        if result is None:
            return [False] * local_len_new

        locations = result["locations"]
        write_session_id = result["write_session_id"]
        block_mask = result["block_mask"]
        parsed = self._parse_block_mask(block_mask, len_prefix, len_new)

        finish_trace_id = f"finish-{trace_id}"

        # None means truly broken manager data — treat as write failure.
        if parsed is None:
            logger.warning(
                f"_batch_set: inconsistent block_mask from manager, "
                f"aborting write session {write_session_id}"
            )
            if self.tp_rank == 0:
                # Mark all locations as failed so manager cleans them up.
                try:
                    self._manager_client.finish_write_cache(
                        {
                            "trace_id": finish_trace_id,
                            "instance_id": self.instance_id,
                            "write_session_id": write_session_id,
                            "success_blocks": {
                                "bool_masks": {"values": [False] * len(locations)}
                            },
                        }
                    )
                except Exception as e:
                    logger.error(f"finish_write_cache failed: {e}")
            return [False] * local_len_new

        save_indices, prefix_write_count = parsed
        unmatched = len(save_indices)

        # Early return if all new blocks are already cached.
        if unmatched == 0:
            if self.tp_rank == 0:
                try:
                    self._manager_client.finish_write_cache(
                        {
                            "trace_id": finish_trace_id,
                            "instance_id": self.instance_id,
                            "write_session_id": write_session_id,
                            "success_blocks": {
                                "bool_masks": {"values": [False] * len(locations)}
                            },
                        }
                    )
                except Exception as e:
                    logger.error(f"finish_write_cache failed: {e}")
            return [False] * local_len_new if skip_transfer else [True] * local_len_new

        assert unmatched + prefix_write_count == len(locations)

        # Data transfer preparation and execution.
        # Skip prefix locations — sglang cannot write prefix blocks.
        # Best-effort: each rank writes only the blocks it has local data for.
        # A per-block flag vector is all_reduced (MIN) so only blocks written
        # by ALL ranks are considered successful.
        # Wrapped in try-except so that every rank always reaches the
        # all_reduce below, preventing cross-rank NCCL/gloo hangs.
        new_locations = locations[prefix_write_count:]
        per_block_flags = torch.zeros(unmatched, dtype=torch.int)
        if skip_transfer:
            # This rank's block_keys diverged from rank 0 — writing would
            # corrupt storage. Keep per_block_flags as all-zero and still
            # participate in all_reduce below.
            logger.warning(
                "_batch_set: skipping data transfer on this rank due to input divergence"
            )
        else:
            try:
                buffer_ptrs, buffer_sizes = self.mem_pool_host.get_page_buffer_meta(
                    host_indices
                )
                local_block_count = len(buffer_ptrs) // self.kv_factor

                # Determine which save_indices have local data available
                valid_save_mask = [(idx < local_block_count) for idx in save_indices]
                valid_save_set = set(
                    idx for idx, valid in zip(save_indices, valid_save_mask) if valid
                )
                num_valid = sum(valid_save_mask)

                if num_valid > 0:
                    buffer_ptrs = [
                        ptr
                        for i, ptr in enumerate(buffer_ptrs)
                        if (i // self.kv_factor) in valid_save_set
                    ]
                    buffer_sizes = [
                        sz
                        for i, sz in enumerate(buffer_sizes)
                        if (i // self.kv_factor) in valid_save_set
                    ]

                    # Extract URIs only for blocks with local data
                    valid_locations = [
                        loc
                        for loc, valid in zip(new_locations, valid_save_mask)
                        if valid
                    ]
                    uris = self._extract_uris(valid_locations)
                    buffers = self._prepare_buffers(buffer_ptrs, buffer_sizes)
                    assert len(uris) == len(buffers)

                    # Perform data transfer
                    start_time = time.perf_counter()
                    result = self.transfer_client.SaveKvCaches(uris, buffers)
                    end_time = time.perf_counter()
                    self.backup_pgs.append(num_valid)
                    self.backup_bandwidth.append(
                        num_valid
                        * self.location_spec_size
                        / (1 << 30)
                        / (end_time - start_time)
                    )
                    logger.debug(f"SaveKvCaches {result=}")

                    transfer_ok = result[0] == kvcm_py_client.ClientErrorCode.ER_OK
                    if not transfer_ok:
                        logger.error(f"SaveKvCaches error: {result}")
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
        if self.tp_world_size > 1 and not self.is_mla_model:
            torch.distributed.all_reduce(
                per_block_flags,
                op=torch.distributed.ReduceOp.MIN,
                group=self.storage_tp_group,
            )

        new_block_success = [bool(per_block_flags[j]) for j in range(unmatched)]
        finish_mask = [False] * prefix_write_count + new_block_success
        if self.tp_rank == 0:
            try:
                self._manager_client.finish_write_cache(
                    {
                        "trace_id": finish_trace_id,
                        "instance_id": self.instance_id,
                        "write_session_id": write_session_id,
                        "success_blocks": {"bool_masks": {"values": finish_mask}},
                    }
                )
            except Exception as e:
                logger.error(f"finish_write_cache failed: {e}")
                # Mark all as failed on rank 0 for the return value.
                new_block_success = [False] * unmatched

        # Build result list: 1:1 positional mapping with input keys.
        # Use local_len_new so the return length matches the caller's input.
        # - keys not in save_indices → True (assumed cached / no-op)
        # - keys in save_indices → per-block success from all_reduce
        # When input diverged (skip_transfer), nothing was written and the
        # local keys don't match rank 0's — return all False.
        if skip_transfer:
            return [False] * local_len_new
        block_flag_map = {
            save_indices[j]: new_block_success[j] for j in range(unmatched)
        }
        result_list = [block_flag_map.get(i, True) for i in range(local_len_new)]
        return result_list

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        trace_id = self._get_trace_id()
        try:
            self._ensure_client()
            result = self._batch_set(
                keys=keys,
                host_indices=host_indices,
                trace_id=trace_id,
                extra_info=extra_info,
            )
            return result
        except _UnsupportedBackendError:
            # Already reported once by _ensure_client; keep answering
            # conservatively without logging on every call.
            return [False] * len(keys)
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
            self._ensure_client()
            for transfer in transfers:
                spec_name = self._get_extra_pool_spec_name(transfer.name)
                pool = self.registered_pools.get(transfer.name)
                keys = transfer.keys or []
                if not keys:
                    results[transfer.name] = []
                    continue
                if (
                    transfer.name in self._late_pools
                    or spec_name is None
                    or pool is None
                ):
                    self._report_unknown_pool(transfer.name)
                    results[transfer.name] = [False] * len(keys)
                    continue

                spec_group = self._get_extra_pool_spec_group(transfer.name)
                block_keys, _, _ = self._prepare_block_keys(keys)

                if self.tp_rank == 0:
                    start_trace_id = f"start-v2-{trace_id}"
                    request = {
                        "trace_id": start_trace_id,
                        "instance_id": self.instance_id,
                        "block_keys": block_keys,
                        "location_spec_group_names": [spec_group] * len(block_keys),
                        "write_timeout_seconds": self.write_timeout_seconds,
                    }
                    try:
                        write_result = self._manager_client.start_write_cache(request)
                    except Exception as e:
                        logger.error(
                            f"start_write_cache failed on rank 0: {trace_id=} {e=}"
                        )
                        write_result = None
                    if self.tp_world_size > 1 and not self.is_mla_model:
                        torch.distributed.broadcast_object_list(
                            [write_result], src=0, group=self.storage_tp_group
                        )
                elif self.is_mla_model:
                    logger.warning(
                        f"batch_set_v2 called on non-rank-0 (tp_rank={self.tp_rank}) "
                        f"for MLA model; only rank 0 should write. Returning all False."
                    )
                    results[transfer.name] = [False] * len(keys)
                    continue
                else:
                    recv: List[Any] = [None]
                    torch.distributed.broadcast_object_list(
                        recv, src=0, group=self.storage_tp_group
                    )
                    write_result = recv[0]
                if write_result is None:
                    results[transfer.name] = [False] * len(keys)
                    continue

                locations = write_result["locations"]
                write_session_id = write_result["write_session_id"]
                block_mask = write_result["block_mask"]
                finish_trace_id = f"finish-v2-{trace_id}"

                parsed = self._parse_block_mask(block_mask, 0, len(keys))

                if parsed is None:
                    logger.warning(
                        f"batch_set_v2: inconsistent block_mask from manager, "
                        f"aborting write session {write_session_id}"
                    )
                    if self.tp_rank == 0:
                        try:
                            self._manager_client.finish_write_cache(
                                {
                                    "trace_id": finish_trace_id,
                                    "instance_id": self.instance_id,
                                    "write_session_id": write_session_id,
                                    "success_blocks": {
                                        "bool_masks": {
                                            "values": [False] * len(locations)
                                        }
                                    },
                                }
                            )
                        except Exception as e:
                            logger.error(f"finish_write_cache failed: {e}")
                    results[transfer.name] = [False] * len(keys)
                    continue

                save_indices, _prefix_write_count = parsed
                unmatched = len(save_indices)

                if unmatched == 0:
                    if self.tp_rank == 0:
                        try:
                            self._manager_client.finish_write_cache(
                                {
                                    "trace_id": finish_trace_id,
                                    "instance_id": self.instance_id,
                                    "write_session_id": write_session_id,
                                    "success_blocks": {
                                        "bool_masks": {
                                            "values": [False] * len(locations)
                                        }
                                    },
                                }
                            )
                        except Exception as e:
                            logger.error(f"finish_write_cache failed: {e}")
                    results[transfer.name] = [True] * len(keys)
                    continue

                assert len(save_indices) + _prefix_write_count == len(locations)

                # Data transfer preparation and execution.
                # Wrapped in try-except so that every rank always reaches the
                # all_reduce below, preventing cross-rank NCCL/gloo hangs.
                try:
                    ptr_list, size_list, components = self._page_buffer_meta(
                        pool, transfer
                    )
                    save_set = set(save_indices)
                    ptr_list = [
                        p
                        for i, p in enumerate(ptr_list)
                        if (i // components) in save_set
                    ]
                    size_list = [
                        s
                        for i, s in enumerate(size_list)
                        if (i // components) in save_set
                    ]

                    uris = []
                    for loc in locations:
                        uri = self._extract_single_spec_uri(loc, spec_name)
                        if uri:
                            uris.append(uri)
                    buffers = self._prepare_extra_pool_buffers(
                        ptr_list, size_list, components
                    )
                    assert len(uris) == len(buffers)
                    start_time = time.perf_counter()
                    save_result = self.transfer_client.SaveKvCaches(uris, buffers)
                    end_time = time.perf_counter()
                    flag = save_result[0] == kvcm_py_client.ClientErrorCode.ER_OK
                    if flag:
                        spec_size = self._get_extra_pool_spec_size(transfer.name)
                        self.backup_pgs.append(unmatched)
                        self.backup_bandwidth.append(
                            unmatched * spec_size / (1 << 30) / (end_time - start_time)
                        )
                    if not flag:
                        logger.error(f"SaveKvCaches v2 error: {transfer.name}")
                except Exception as e:
                    logger.error(
                        f"Data transfer v2 (SaveKvCaches) failed: {transfer.name} {e}"
                    )
                    flag = False

                if self.tp_world_size > 1 and not self.is_mla_model:
                    flag_tensor = torch.tensor(flag, dtype=torch.int)
                    torch.distributed.all_reduce(
                        flag_tensor,
                        op=torch.distributed.ReduceOp.MIN,
                        group=self.storage_tp_group,
                    )
                    flag = bool(flag_tensor.item())

                finish_mask = [flag] * len(locations)
                if self.tp_rank == 0:
                    try:
                        self._manager_client.finish_write_cache(
                            {
                                "trace_id": finish_trace_id,
                                "instance_id": self.instance_id,
                                "write_session_id": write_session_id,
                                "success_blocks": {
                                    "bool_masks": {"values": finish_mask}
                                },
                            }
                        )
                    except Exception as e:
                        logger.error(f"finish_write_cache failed: {e}")

                per_key = [True] * len(keys)
                for idx in save_indices:
                    per_key[idx] = flag
                results[transfer.name] = per_key

            return results
        except _UnsupportedBackendError:
            # Already reported once by _ensure_client; keep answering
            # conservatively without logging on every call.
            return {t.name: [False] * len(t.keys or []) for t in transfers}
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
            self._ensure_client()
            result = self._batch_exists(
                keys=keys, trace_id=trace_id, extra_info=extra_info
            )
            return result
        except _UnsupportedBackendError:
            # Already reported once by _ensure_client; keep answering
            # conservatively without logging on every call.
            return 0
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
            self._ensure_client()
            # Reuse the same get_cache_location call as batch_exists,
            # but inspect per-location specs for extra pool existence.
            block_keys, len_prefix, len_new = self._prepare_block_keys(keys, extra_info)
            get_request = {
                "trace_id": trace_id,
                "block_keys": block_keys,
                "instance_id": self.instance_id,
                "query_type": "QT_PREFIX_MATCH",
                "block_mask": {"offset": len_prefix},
            }
            result = self._manager_client.get_cache_location(get_request)
            locations = result["locations"]

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
            for transfer in pool_transfers or []:
                if final_pages == 0:
                    break
                boundary = self._check_pool_spec_existence(
                    locations, kv_hit_pages, transfer
                )
                pool_hit_pages[transfer.name] = boundary
                final_pages = min(final_pages, boundary)

            return PoolTransferResult(final_pages, pool_hit_pages)
        except _UnsupportedBackendError:
            # Already reported once by _ensure_client; keep answering
            # conservatively without logging on every call.
            return PoolTransferResult.empty()
        except Exception as e:
            logger.error(f"batch_exists_v2 failed: {trace_id=} {e=}")
            return PoolTransferResult.empty()

    def get_stats(self) -> Any:
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
        hash_digest = hashlib.sha256(data.encode("utf-8")).digest()
        return int.from_bytes(hash_digest[:8], "big", signed=True)

    def _prepare_block_keys(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> tuple[List[int], int, int]:
        """Prepare block keys and return them along with the prefix offset."""
        prefix_keys = (
            extra_info.prefix_keys
            if (extra_info is not None) and (extra_info.prefix_keys is not None)
            else []
        )
        block_keys = prefix_keys + keys
        block_keys = [self._sha256_to_int64(block_key) for block_key in block_keys]
        return block_keys, len(prefix_keys), len(keys)

    def _extract_uris(self, locations: List[dict]) -> List[str]:
        """Extract URIs from locations for the current TP rank."""
        uris = []
        for location in locations:
            for location_spec in location["location_specs"]:
                if location_spec["name"] == self.location_spec_name:
                    uris.append(location_spec["uri"])
        return uris

    def _prepare_buffers(
        self, buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> List[kvcm_py_client.BlockBuffer]:
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

    def _parse_block_mask(
        self, block_mask: dict, len_prefix: int, len_new: int
    ) -> Optional[tuple[List[int], int]]:
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
                logger.warning(
                    f"_parse_block_mask: offset {offset} < len_prefix {len_prefix}, "
                    "prefix blocks will be skipped (best-effort)"
                )
                prefix_write_count = len_prefix - offset
                save_indices.extend(range(len_prefix, len_prefix + len_new))
            else:
                save_indices.extend(range(offset, len_prefix + len_new))
        else:
            # False: need to store
            bool_masks = block_mask.get("bool_masks", {}).get("values", [])
            if len(bool_masks) < len_prefix + len_new:
                # Incomplete mask data from manager.
                logger.warning(
                    f"_parse_block_mask: bool_masks length {len(bool_masks)} < "
                    f"expected {len_prefix + len_new}, treating as inconsistent state"
                )
                return None
            prefix_write_count = sum(1 for v in bool_masks[:len_prefix] if not v)
            if prefix_write_count > 0:
                logger.warning(
                    f"_parse_block_mask: {prefix_write_count} prefix blocks "
                    "not cached in bool_masks, will be skipped (best-effort)"
                )
            max_index = max([i for i, x in enumerate(bool_masks) if not x], default=-1)
            save_indices.extend(
                [i for i in range(len_prefix, max_index + 1) if not bool_masks[i]]
            )
        save_indices = [(i - len_prefix) for i in save_indices if i >= len_prefix]
        return save_indices, prefix_write_count

    def _extract_single_spec_uri(self, location: dict, spec_name: str) -> Optional[str]:
        """Extract the URI for a named spec from a single location dict."""
        for spec in location.get("location_specs", []):
            if spec["name"] == spec_name and spec.get("uri"):
                return spec["uri"]
        return None

    @staticmethod
    def _page_buffer_meta(
        pool: Any, transfer: PoolTransfer
    ) -> tuple[List[int], List[int], int]:
        """Per-page zero-copy meta of an extra pool plus its IOVs per page.

        The component count is read back from the pool's own output instead of
        being derived from pool internals: mamba pools emit a temporal IOV only
        while the model has an SSM state, so conv-only models hand out one IOV
        per conv buffer.  Each page must also match exactly one key: callers
        slice the flat IOV list by page position and map it back to key
        position, so both a wrong component count and a page/key mismatch would
        silently shift every page's data.  Both are asserted here instead.
        """
        host_indices = transfer.host_indices
        assert host_indices is not None, (
            f"pool transfer {transfer.name} carries no host indices"
        )
        ptr_list, size_list = pool.get_page_buffer_meta(host_indices)
        num_pages = len(host_indices) // pool.page_size
        assert num_pages == len(transfer.keys or []), (
            f"pool transfer {transfer.name} carries {num_pages} host pages "
            f"(page size {pool.page_size}) for {len(transfer.keys or [])} keys"
        )
        assert num_pages > 0 and len(ptr_list) % num_pages == 0, (
            f"get_page_buffer_meta returned {len(ptr_list)} IOVs for "
            f"{len(host_indices)} host indices (page size {pool.page_size})"
        )
        return ptr_list, size_list, len(ptr_list) // num_pages

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

    def _prepare_extra_pool_buffers(
        self, ptr_list: List[int], size_list: List[int], components_per_page: int
    ) -> List[kvcm_py_client.BlockBuffer]:
        """Convert get_page_buffer_meta output to BlockBuffer list.

        Each logical page maps to `components_per_page` IOVs in a single BlockBuffer.
        Works for both Mamba (temporal + conv components) and Indexer (single component).
        """
        buffers = []
        for i in range(0, len(ptr_list), components_per_page):
            buffer = kvcm_py_client.BlockBuffer()
            iovs = []
            for j in range(components_per_page):
                iov = kvcm_py_client.Iov()
                iov.type = kvcm_py_client.MemoryType.CPU
                iov.base = ptr_list[i + j]
                iov.size = size_list[i + j]
                iov.ignore = False
                iovs.append(iov)
            buffer.iovs = iovs
            buffers.append(buffer)
        return buffers

    def _get_extra_pool_spec_name(self, pool_name: str) -> Optional[str]:
        """Map a PoolName to its KVCM location spec name for the current rank."""
        if pool_name == PoolName.MAMBA and self.has_mamba:
            return self.mamba_location_spec_name
        if pool_name == PoolName.INDEXER and self.has_indexer:
            return self.indexer_location_spec_name
        return None

    def _check_pool_spec_existence(
        self,
        locations: List[dict],
        kv_hit_pages: int,
        transfer: PoolTransfer,
    ) -> int:
        """Check how many pages have the extra pool's spec.

        Returns the number of contiguous prefix pages that satisfy
        transfer.hit_policy.  Falls back to 0 for unknown policies so
        a stale/future enum value never causes an UnboundLocalError crash.
        """
        spec_name = self._get_extra_pool_spec_name(transfer.name)
        if spec_name is None or transfer.name in self._late_pools:
            # Unmanaged pool (e.g. a newer sglang side pool such as SWA) or a
            # pool that showed up after the specs were frozen: its data is
            # never written, so claiming hits here would only make the caller
            # move KV pages that batch_get_v2 then reports as misses.
            self._report_unknown_pool(transfer.name)
            return 0

        def has_spec(loc: dict) -> bool:
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

    def _report_unknown_pool(self, pool_name: Any) -> None:
        """Report once that sglang moved data for a pool we do not manage."""
        if not self._mark_pool_reported(pool_name):
            return
        logger.warning(
            "pool %s is not backed by this connector: its transfers are "
            "reported as misses and its data is neither read from nor written "
            "to KVCM (unknown pool, or registered after initialization).",
            pool_name,
        )

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
