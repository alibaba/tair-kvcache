"""KVCM vLLM connector (v1), built around per-group transfer.

vLLM models expose one or more ``kv_cache_groups`` (``KVCacheConfig``):

* Pure-attention models: a single ``FullAttentionSpec`` group.
* Hybrid models (e.g. Qwen3.5): several ``MambaSpec`` groups plus one (or more)
  ``FullAttentionSpec`` group. With ``mamba_cache_mode="align"`` every group has
  its own block table (``block_ids`` is a tuple indexed by group) but all groups
  share the scheduler block size.

The connector treats every group as an independent transfer unit with its own
KVCM location spec (``tp{rank}_g{group}``), its own block table and its own data
access strategy (token-granular gather/scatter for attention, per-block byte
copy for mamba state). There is no separate "hybrid path": a full-attention
model is simply the one-group case.

A manager block covers the same token range in every group, so one KVCM cache
key (hashed from token ids) owns the location specs of all groups of all ranks.
"""

import copy
import json
import math
import time
import typing
import threading

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor
from kv_cache_manager.client.pybind import kvcm_py_client

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

try:
    # vllm >= v0.11.1
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    from vllm.utils.network_utils import get_ip
except ImportError:
    # vllm <= v0.11.0
    from vllm.utils import get_kv_cache_torch_dtype, get_ip

from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient
from kv_cache_manager.py_connector.common.tp_coordinator import CoordinateMsgSerializer, TpCoordinatorServer, \
    TpCoordinatorClient, SendBlockStartEvent, CoordinateMessage, SaveContext
from kv_cache_manager.py_connector.common.logger import logger, configure_log_level
from kv_cache_manager.py_connector.common._version_info import FULL_VERSION, GIT_COMMIT, BUILD_TIME

from kv_cache_manager.py_connector.common.types import KVCacheInfo, TransferGroup
from kv_cache_manager.py_connector.vllm.metadata import SaveRequest, LoadRequest, FinishRequest, ReqStateToWorker, \
    TairKvCacheConnectorMetadata
from kv_cache_manager.py_connector.vllm.config import TairKvCacheConnectorExtraConfig
from kv_cache_manager.py_connector.vllm.location_query_manager import LocationQueryManager
from kv_cache_manager.py_connector.vllm.data_transfer import MultiResult, DataTransferManager, _get_device_module

if typing.TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.attention import AttentionMetadata
    from vllm.v1.request import Request
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig


def attn_kv_views(ref: torch.Tensor) -> List[torch.Tensor]:
    """Normalize one attention layer's paged KV cache into per-pointer views.

    vLLM's flash_attn backend changed ``get_kv_cache_shape`` twice; the three
    layouts are detected from the tensor shape itself (never from version
    strings):

    * 4-D ``(num_blocks, H, block, 2*D)``  -- K/V packed into the content dim
      (vLLM >= 0.26.0). One transfer pointer per layer.
    * 5-D ``(num_blocks, 2, block, H, D)`` -- N-first split K/V
      (vLLM 0.23.0 - 0.25.x). Two pointers per layer: ``t[:, 0]`` / ``t[:, 1]``.
    * 5-D ``(2, num_blocks, block, H, D)`` -- KV-first split K/V
      (vLLM <= 0.22.1). Two pointers per layer: ``t[0]`` / ``t[1]``.

    Every returned view has the logical shape
    ``(num_blocks, kernel_block_size, heads, content_dim)`` matching the NHD
    memory order, so all downstream math (per-token dim, token-major check,
    block stride, data_ptr) is layout-independent. Unrecognized layouts raise.
    """
    if ref.dim() == 4:
        # Packed content dim; permute to token-major logical order. The permuted
        # view shares storage, data_ptr() is the storage base.
        return [ref.permute(0, 2, 1, 3)]
    if ref.dim() == 5:
        kv_first = ref.shape[0] == 2
        n_first = ref.shape[1] == 2
        if kv_first and n_first:
            raise NotImplementedError(
                f"ambiguous kv layout {tuple(ref.shape)}: cannot tell the K/V "
                f"dim from a num_blocks dim of size 2")
        if kv_first:
            return [ref[0], ref[1]]
        if n_first:
            return [ref[:, 0], ref[:, 1]]
    raise NotImplementedError(
        f"unrecognized kv cache layout {tuple(ref.shape)}; expected the packed "
        f"4-D (vllm >= 0.26.0) or one of the split K/V 5-D layouts "
        f"(vllm <= 0.25.x)")


def _hybrid_external_load_supported() -> Optional[bool]:
    """vLLM <= 0.22.x cannot combine mamba align mode with a KV connector:
    ``Scheduler._mamba_block_aligned_split`` asserts
    ``num_external_computed_tokens == 0`` ("External KV connector is not
    verified yet"), so the first external match would crash the scheduler.
    Probe the installed vLLM for that blocking assert (a capability check,
    not a version-string comparison).

    Returns:
        True   -- supported (assert absent, or the method was removed by a
                  newer vLLM: the assert went away with it);
        False  -- unsupported (the blocking assert is present);
        None   -- the method exists but its source is unavailable (frozen /
                  bytecode-only install), so the assert cannot be ruled out.
    """
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
        method = Scheduler._mamba_block_aligned_split
    except (ImportError, AttributeError):
        # No such method: the blocking assert was removed/refactored away.
        return True
    try:
        import inspect
        src = inspect.getsource(method)
    except Exception:
        return None  # method exists but cannot be inspected
    return "External KV connector is not verified yet" not in src


def ensure_hybrid_supported(force: bool = False):
    """Fail fast with a clear message when a hybrid (mamba) model is served on
    a vLLM whose scheduler rejects external KV loads (vllm <= 0.22.x).

    When the probe is inconclusive (method present but source unavailable) the
    gate fails closed: a wrong guess would crash the scheduler on the first
    external match. ``force`` (extra_config ``force_hybrid_support``) bypasses
    the inconclusive case for source-restricted environments."""
    supported = _hybrid_external_load_supported()
    if supported:
        return
    if supported is None:
        if force:
            logger.warning(
                "force_hybrid_support=true: skipping the hybrid external-load "
                "capability probe; if this vLLM's scheduler still asserts "
                "'External KV connector is not verified yet' the first "
                "external match will crash it")
            return
        raise NotImplementedError(
            "TairKvCacheConnector: cannot verify that this vLLM supports "
            "hybrid (mamba) models with an external KV connector -- "
            "Scheduler._mamba_block_aligned_split exists but its source is "
            "unavailable, so the vllm <= 0.22.x blocking assert cannot be "
            "ruled out. If you know this vLLM is >= 0.23.0, set "
            "kv_connector_extra_config {\"force_hybrid_support\": true} to "
            "bypass this check.")
    raise NotImplementedError(
        "TairKvCacheConnector: this vLLM version cannot combine hybrid "
        "(mamba) models with an external KV connector -- its scheduler "
        "asserts num_external_computed_tokens == 0 in "
        "_mamba_block_aligned_split ('External KV connector is not "
        "verified yet'). Upgrade to vLLM >= 0.23.0 for hybrid model "
        "support; full-attention models are unaffected.")


@dataclass
class GroupMeta:
    """Static description of one kv_cache_group, derived from KVCacheConfig.

    Available in both scheduler and worker roles (before tensors exist)."""

    group_idx: int
    is_attention: bool
    layer_names: List[str]
    # The group's block table granularity in tokens (spec.block_size).
    block_size: int
    # Bytes stored per manager block for the whole group.
    per_block_bytes: int
    # Mamba only: bytes per block per layer (page_size_bytes of the spec).
    page_size_bytes: int = 0


@dataclass
class ReqState:
    """Tracks one request. Lives in the scheduler and (mirrored) in workers."""

    req_id: str
    token_ids: list
    # Per kv_cache_group block table (same length across groups).
    block_ids_per_group: List[List[int]]
    has_saved_block_num: int
    local_matched_token_num: int
    remote_matched_token_num: int

    # vllm_request only available in scheduler
    vllm_request: Optional["Request"]

    # Saving progress counters; only meaningful in scheduler and tp0 worker.
    scheduled_saving_count: int = 0
    sent_saving_count: int = 0
    need_report_after_saving_finished: bool = False

    @staticmethod
    def create_from_delta(delta: "ReqStateToWorker") -> "ReqState":
        return ReqState(
            req_id=delta.req_id,
            token_ids=list(delta.new_tokens_ids),
            block_ids_per_group=[list(b) for b in delta.new_block_ids_per_group],
            has_saved_block_num=delta.has_saved_block_num,
            local_matched_token_num=0,
            remote_matched_token_num=0,
            vllm_request=None,
        )

    def update_from_delta(self, delta: "ReqStateToWorker"):
        self.token_ids.extend(delta.new_tokens_ids)
        if not delta.new_block_ids_per_group:
            return
        if delta.resumed_from_preemption:
            self.block_ids_per_group = [list(b) for b in delta.new_block_ids_per_group]
        else:
            if not self.block_ids_per_group:
                self.block_ids_per_group = [[] for _ in delta.new_block_ids_per_group]
            for group_ids, new_ids in zip(self.block_ids_per_group, delta.new_block_ids_per_group):
                group_ids.extend(new_ids)


class TairKvCacheConnector(KVConnectorBase_V1, SupportsHMA):

    # ------------------------------------------------------------------ #
    # Init / registration
    # ------------------------------------------------------------------ #
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 kv_cache_config: Optional["KVCacheConfig"] = None):
        super().__init__(vllm_config, role, kv_cache_config)
        assert kv_cache_config is not None, \
            "TairKvCacheConnector requires vLLM to pass kv_cache_config (vllm >= 0.11.1)"

        logger.warning("KVCM vllm connector version: %s (commit: %s, build: %s)",
                       FULL_VERSION, GIT_COMMIT, BUILD_TIME)

        self._extra_config = TairKvCacheConnectorExtraConfig(
            vllm_config.kv_transfer_config.kv_connector_extra_config)
        configure_log_level(self._extra_config.log_level)

        model_config = vllm_config.model_config
        assert vllm_config.parallel_config.pipeline_parallel_size == 1
        if getattr(model_config, "use_mla", False):
            raise NotImplementedError("MLA models are not supported by TairKvCacheConnector")

        self._vllm_block_size = vllm_config.cache_config.block_size
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size
        self._kv_dtype = get_kv_cache_torch_dtype(
            vllm_config.cache_config.cache_dtype, model_config.dtype)

        # Manager block size: attention KV is token-granular and can be re-blocked,
        # but mamba state exists once per scheduler block, so hybrid models must
        # keep manager block == scheduler block.
        manager_block_size = self._vllm_block_size
        self._has_state_groups = any(
            isinstance(g.kv_cache_spec, MambaSpec) for g in kv_cache_config.kv_cache_groups)
        if self._has_state_groups:
            ensure_hybrid_supported(force=self._extra_config.force_hybrid_support)
        if self._extra_config.preferred_block_size != 0:
            if self._has_state_groups:
                if self._extra_config.preferred_block_size != self._vllm_block_size:
                    logger.warning(
                        "preferred_block_size=%d ignored for hybrid model: mamba state is "
                        "per scheduler block (%d)", self._extra_config.preferred_block_size,
                        self._vllm_block_size)
            else:
                manager_block_size = self._extra_config.preferred_block_size
        self._manager_block_size = manager_block_size

        self._group_metas = self._parse_groups(kv_cache_config)
        self._num_groups = len(self._group_metas)

        deployment = {
            "model_name": model_config.served_model_name,
            "dtype": str(self._kv_dtype)[6:],  # strip "torch."
            "use_mla": False,
            "tp_size": self._tp_size,
            "dp_size": vllm_config.parallel_config.data_parallel_size,
            "pp_size": vllm_config.parallel_config.pipeline_parallel_size,
        }
        logger.info("deployment: %s, groups: %s", deployment, self._group_metas)

        self._manager_client = KvCacheManagerClient.from_connector_config(vars(self._extra_config))

        self._alive_requests: dict[str, ReqState] = {}
        self._waiting_to_load_requests: List[LoadRequest] = []
        self._waiting_to_save_requests_lock = threading.Lock()
        self._waiting_to_save_requests: List[SaveRequest] = []
        self._waiting_to_finish_requests: List[FinishRequest] = []
        self._canceled_save_request_ids_lock = threading.Lock()
        self._canceled_save_request_ids: List[str] = []

        self._host_ip = get_ip()
        port = self._extra_config.coordinator_base_port

        register_response = self._manager_client.register_instance({
            "trace_id": "register_%s" % self._extra_config.instance_id,
            "instance_group": self._extra_config.instance_group,
            "instance_id": self._extra_config.instance_id,
            "model_deployment": deployment,
            "block_size": manager_block_size,
            "location_spec_infos": [
                {"name": self._spec_name(rank, meta.group_idx), "size": meta.per_block_bytes}
                for rank in range(self._tp_size) for meta in self._group_metas
            ],
        })

        max_group_bytes = max(m.per_block_bytes for m in self._group_metas)
        self._iov_size = max_group_bytes * self._extra_config.hf3fs_concurrent_io_block_count

        if role == KVConnectorRole.SCHEDULER:
            self._epoch = 0
            self._coordinator_client = TpCoordinatorClient(self._host_ip, port)
            self._http_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kvcm_http_")
            self._location_query_manager = LocationQueryManager(
                self._manager_client, self._http_executor, self._extra_config.instance_id,
                self._extra_config.async_get_cache_location)
            logger.warning(
                "TairKvCacheConnector scheduler inited, extra_config: %r, manager block size: %d, "
                "vllm block size: %d, groups: %d",
                self._extra_config.__dict__, self._manager_block_size,
                self._vllm_block_size, self._num_groups)

        elif role == KVConnectorRole.WORKER:
            self._tp_rank = get_tensor_model_parallel_rank()
            self._device_mod = None
            if self._tp_rank == 0:
                self._coordinator_server = TpCoordinatorServer(
                    self._host_ip, port, self._tp_size, self.on_save_finished)
            self._coordinator_client = TpCoordinatorClient(self._host_ip, port)

            self._storage_configs = register_response["storage_configs"]
            sdk_backend_configs = self.parse_hf3fs_configs(self._storage_configs)
            self._self_spec_names = {
                meta.group_idx: self._spec_name(self._tp_rank, meta.group_idx)
                for meta in self._group_metas
            }
            transfer_client_json = {
                "instance_group": self._extra_config.instance_group,
                "instance_id": self._extra_config.instance_id,
                "block_size": self._manager_block_size,
                "sdk_config": {
                    "thread_num": self._extra_config.sdk_thread_num,
                    "queue_size": self._extra_config.sdk_queue_size,
                    "sdk_backend_configs": sdk_backend_configs,
                    "timeout_config": {
                        "get_timeout_ms": self._extra_config.sdk_get_timeout_ms,
                        "put_timeout_ms": self._extra_config.sdk_put_timeout_ms,
                    },
                },
                "location_spec_infos": {
                    self._self_spec_names[meta.group_idx]: meta.per_block_bytes
                    for meta in self._group_metas
                },
            }
            init_params = kvcm_py_client.InitParams()
            init_params.role_type = kvcm_py_client.RoleType.WORKER
            init_params.self_location_spec_name = self._self_spec_names[self._group_metas[0].group_idx]
            init_params.storage_configs = f"{self._storage_configs}"
            transfer_client_config = json.dumps(transfer_client_json)
            logger.info("transfer_client_config: %s", transfer_client_config)
            self._transfer_client = kvcm_py_client.TransferClient.Create(
                transfer_client_config, init_params)
            assert self._transfer_client is not None, "kvcm_py_client.TransferClient.Create failed"
            logger.warning(
                "TairKvCacheConnector worker inited, tp rank: %d/%d, host: %s:%d, groups: %d",
                self._tp_rank, self._tp_size, self._host_ip, port, self._num_groups)

    def _spec_name(self, tp_rank: int, group_idx: int) -> str:
        return f"tp{tp_rank}_g{group_idx}"

    def _num_allocated_blocks(self, req: ReqState) -> int:
        """Min allocated block-table length across the *transferred* groups.

        ``block_ids_per_group`` is indexed by the vLLM group index and includes
        groups skipped by ``_parse_groups`` (EAGLE/MTP drafters). A drafter's
        block table can lag behind the target model's, so including it in the
        min would permanently understate how many blocks are saveable."""
        if not req.block_ids_per_group:
            return 0
        return min(len(req.block_ids_per_group[meta.group_idx])
                   for meta in self._group_metas)

    def _parse_groups(self, kv_cache_config: "KVCacheConfig") -> List[GroupMeta]:
        metas = []
        for idx, group in enumerate(kv_cache_config.kv_cache_groups):
            if getattr(group, "is_eagle_group", False):
                logger.warning("skip eagle group %d (%d layers)", idx, len(group.layer_names))
                continue
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                metas.append(GroupMeta(
                    group_idx=idx,
                    is_attention=False,
                    layer_names=list(group.layer_names),
                    block_size=spec.block_size,
                    per_block_bytes=spec.page_size_bytes * len(group.layer_names),
                    page_size_bytes=spec.page_size_bytes,
                ))
            elif isinstance(spec, FullAttentionSpec):
                # FullAttentionSpec doubles as the merged spec of hybrid
                # SWA/chunked-attention models (vLLM merges window layers into
                # it, keeping sliding_window/attention_chunk_size set). Those
                # blocks hold windowed KV, not the full prefix -- publishing
                # them as prefix caches would corrupt reuse. Refuse explicitly.
                for window_field in ("sliding_window", "attention_chunk_size"):
                    if getattr(spec, window_field, None) is not None:
                        raise NotImplementedError(
                            f"group {idx}: FullAttentionSpec has {window_field}="
                            f"{getattr(spec, window_field)}; sliding-window / "
                            f"chunked attention KV is not full-prefix and is "
                            f"not yet supported by TairKvCacheConnector")
                # Attention KV is token-granular; scale from the spec's page size
                # to the manager block size. Use the *compact* page size:
                # spec.page_size_bytes returns page_size_padded when set, which
                # includes an allocation-alignment gap the gather kernel never
                # copies -- sizing locations/staging buffers with it would break
                # the staging view() and waste storage. real_page_size_bytes is
                # exactly the raw KV bytes (2 * block * heads * head_dim * dtype).
                compact_page_bytes = getattr(spec, "real_page_size_bytes", None)
                if compact_page_bytes is None:
                    if getattr(spec, "page_size_padded", None) is not None:
                        raise NotImplementedError(
                            f"group {idx}: page_size_padded="
                            f"{spec.page_size_padded} but this vLLM exposes no "
                            f"real_page_size_bytes to recover the compact page "
                            f"size; padded attention layouts are unsupported here")
                    compact_page_bytes = spec.page_size_bytes
                per_token_bytes = compact_page_bytes // spec.block_size
                metas.append(GroupMeta(
                    group_idx=idx,
                    is_attention=True,
                    layer_names=list(group.layer_names),
                    block_size=spec.block_size,
                    per_block_bytes=per_token_bytes * self._manager_block_size * len(group.layer_names),
                ))
            else:
                raise NotImplementedError(
                    f"Unsupported kv cache spec {type(spec).__name__} in group {idx}")
        assert metas, "no usable kv cache groups"
        return metas

    def shutdown(self):
        self._manager_client.close()
        if hasattr(self, "_location_query_manager"):
            self._location_query_manager.shutdown()
        return None

    def parse_hf3fs_configs(self, storage_configs):
        hf3fs_configs = []
        storage_configs_json = json.loads(storage_configs)
        for storage_config in storage_configs_json:
            if storage_config["type"] == "vcns_hf3fs":
                storage_config["type"] = "hf3fs"
            if storage_config["type"] == "hf3fs" and storage_config["is_available"]:
                hf3fs_configs.append({
                    "type": storage_config["type"],
                    "mountpoint": storage_config["storage_spec"]["mountpoint"],
                    "root_dir": storage_config["storage_spec"]["root_dir"],
                    "read_iov_block_size": self._extra_config.read_iov_block_size,
                    "read_iov_size": self._iov_size,
                    "write_iov_block_size": self._extra_config.write_iov_block_size,
                    "write_iov_size": self._iov_size,
                })
        self._storage_configs = json.dumps(storage_configs_json)
        return hf3fs_configs

    # ------------------------------------------------------------------ #
    # Worker side: KV cache registration
    # ------------------------------------------------------------------ #
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv_caches = kv_caches
        first_attn = next(kv_caches[name]
                          for meta in self._group_metas if meta.is_attention
                          for name in meta.layer_names)
        self._dtype = first_attn.dtype
        self._device = first_attn.device
        self._device_mod = _get_device_module(self._device)

        groups = [self._build_transfer_group(meta, kv_caches) for meta in self._group_metas]

        self._kvcache_info = KVCacheInfo(
            tp_rank=self._tp_rank,
            world_size=self._tp_size,
            groups=groups,
            device=self._device,
            dtype=self._dtype,
        )
        self._data_transfer = DataTransferManager(
            self._kvcache_info, self._manager_block_size,
            self._transfer_client, self._coordinator_client, self._extra_config)

        logger.warning("register_kv_caches done: %s", [
            (g.spec_name, "attn" if g.is_attention else "state",
             g.layer_num, g.per_block_bytes) for g in groups])

    def _build_transfer_group(self, meta: GroupMeta, kv_caches) -> TransferGroup:
        spec_name = self._self_spec_names[meta.group_idx]
        if meta.is_attention:
            tensors = [kv_caches[name] for name in meta.layer_names]
            ref = tensors[0]
            for t in tensors:
                assert t.shape == ref.shape and t.stride() == ref.stride(), \
                    "attention layers in one group must share shape/stride"
            # Normalize the layout into per-pointer token-major views of shape
            # (num_blocks, kernel_block_size, heads, content_dim); everything
            # below is layout-independent. Split-K/V layouts (vllm <= 0.25.x)
            # yield two views (= two transfer pointers) per layer, the packed
            # layout (vllm >= 0.26.0) yields one.
            ref_views = attn_kv_views(ref)
            view = ref_views[0]
            kernel_block_size = view.shape[1]
            assert meta.block_size % kernel_block_size == 0, \
                f"group block size {meta.block_size} not a multiple of kernel " \
                f"block size {kernel_block_size}"
            per_token_dim = view.shape[2] * view.shape[3]  # heads * content_dim
            # The gather/scatter kernel needs token-major memory inside a page:
            # logical dims (blk, tok, head, dim) contiguous within a block. This
            # is vLLM's NHD order; HND would interleave heads across tokens.
            for v in ref_views:
                assert v.stride()[1:] == (per_token_dim, v.shape[3], 1), \
                    f"kv cache page not token-major: shape={tuple(v.shape)} " \
                    f"stride={v.stride()}; set VLLM_KV_CACHE_LAYOUT=NHD"
            # Non-flat block layouts (page_size_padded gaps, or split K/V
            # interleaved per block as in the 5-D N-first layout) go through the
            # kernel's strided path. Stride 0 = fast flat indexing.
            flat = view.stride(0) == kernel_block_size * per_token_dim
            block_stride = 0 if flat else view.stride(0)
            # Pointer array ordered [K0, V0, K1, V1, ...] for split layouts and
            # [L0, L1, ...] for the packed layout; each view's data_ptr() is its
            # own storage base, so the kernel never adds a K->V offset.
            ptrs = [v.data_ptr() for t in tensors for v in attn_kv_views(t)]
            ptr_tensor = torch.tensor(ptrs, dtype=torch.int64, device="cpu").to(self._device)
            return TransferGroup(
                group_idx=meta.group_idx,
                spec_name=spec_name,
                is_attention=True,
                layer_names=meta.layer_names,
                block_size=meta.block_size,
                per_block_bytes=meta.per_block_bytes,
                kvcache_ptr_tensor_gpu=ptr_tensor,
                layer_num=len(meta.layer_names),
                num_kv_ptrs=len(ptrs),
                per_token_dim=per_token_dim,
                kernel_block_size=kernel_block_size,
                kv_stride=0,
                block_stride=block_stride,
            )

        # Mamba/state group: each layer is a list[Tensor] sharing one storage;
        # rebuild a (num_blocks, page_size_bytes) byte view for opaque copy.
        block_views = []
        for name in meta.layer_names:
            states = kv_caches[name]
            assert isinstance(states, (list, tuple)) and len(states) > 0, \
                f"state layer {name} should be a list of tensors"
            storage = states[0].untyped_storage()
            for st in states[1:]:
                assert st.untyped_storage().data_ptr() == storage.data_ptr(), \
                    f"state layer {name}: tensors do not share storage"
            num_blocks = states[0].shape[0]
            need = num_blocks * meta.page_size_bytes
            assert storage.nbytes() >= need, \
                f"state layer {name}: storage {storage.nbytes()} < {need}"
            byte_view = torch.tensor([], dtype=torch.uint8, device=self._device).set_(storage)
            block_views.append(byte_view[:need].view(num_blocks, meta.page_size_bytes))
        return TransferGroup(
            group_idx=meta.group_idx,
            spec_name=spec_name,
            is_attention=False,
            layer_names=meta.layer_names,
            block_size=meta.block_size,
            per_block_bytes=meta.per_block_bytes,
            layer_num=len(meta.layer_names),
            block_view_tensors=block_views,
            page_size_bytes=meta.page_size_bytes,
        )

    # ------------------------------------------------------------------ #
    # Block index translation
    # ------------------------------------------------------------------ #
    def _attn_token_indices(self, group: TransferGroup, manager_block_idxes,
                            block_table) -> List[List[int]]:
        """Map manager blocks to flat token slots of one attention group.

        Three-tier hierarchy:
          manager block (KVCM unit) -> global token idx
          -> group block (block_table unit, group.block_size tokens)
          -> kernel physical block (tensor unit; ratio physical per group block).
        """
        mbs = self._manager_block_size
        gbs = group.block_size
        kbs = group.kernel_block_size
        ratio = gbs // kbs
        out = []
        for mb in manager_block_idxes:
            idxs = []
            base = mb * mbs
            for i in range(mbs):
                tok = base + i
                logical = tok // gbs
                assert logical < len(block_table), (
                    f"group block {logical} out of range (len={len(block_table)})")
                off = tok % gbs
                phys = block_table[logical] * ratio + off // kbs
                idxs.append(phys * kbs + off % kbs)
            out.append(idxs)
        return out

    def _state_block_ids(self, group: TransferGroup, manager_block_idxes,
                         block_table) -> List[int]:
        """Map manager blocks to block ids of a state (mamba) group.

        State is stored once per group block and covers the whole prefix up to
        that block, so the manager block's last token selects the block.

        KNOWN LIMITATION (mamba_cache_mode="none"): without prefix caching
        vLLM keeps only one resident state block per request, so every manager
        block resolves to that same block id here. Saves then publish the
        *current* state bytes under every manager block's key (mislabeled as
        earlier prefixes) and loads overwrite one block repeatedly (last write
        wins). Hybrid serving is only supported with prefix caching enabled
        (mamba_cache_mode="align"), where the table is position-indexed."""
        mbs = self._manager_block_size
        gbs = group.block_size
        out = []
        for mb in manager_block_idxes:
            logical = ((mb + 1) * mbs - 1) // gbs
            assert logical < len(block_table), (
                f"group block {logical} out of range (len={len(block_table)})")
            out.append(block_table[logical])
        return out

    def _self_uris(self, locations, spec_name: str) -> List[str]:
        uris = []
        for location in locations:
            for spec in location.get("location_specs", []):
                if spec["name"] == spec_name:
                    uris.append(spec["uri"])
        return uris

    # ------------------------------------------------------------------ #
    # Worker side: load / save
    # ------------------------------------------------------------------ #
    def _submit_group_tasks(self, task_fn, multi_result, task_idx, group,
                            uris, token_indices, block_ids, per_task_size, *extra):
        for i in range(0, len(uris), per_task_size):
            end = min(len(uris), i + per_task_size)
            self._data_transfer.submit_task(
                task_fn, multi_result, task_idx, group, uris[i:end],
                token_indices[i:end] if token_indices is not None else None,
                block_ids[i:end] if block_ids is not None else None, *extra)
            task_idx += 1
        return task_idx

    def _plan_group_transfers(self, locations, manager_block_idxes, block_ids_per_group):
        """Build (group, uris, token_indices, block_ids) for every group.

        Returns None if any group's URI list does not cover all blocks."""
        num_blocks = len(manager_block_idxes)
        plans = []
        for group in self._kvcache_info.groups:
            uris = self._self_uris(locations, group.spec_name)
            if len(uris) != num_blocks:
                logger.warning("group %s: %d uris for %d blocks, skip transfer",
                               group.spec_name, len(uris), num_blocks)
                return None
            # block_ids_per_group is indexed by the vLLM group index.
            block_table = block_ids_per_group[group.group_idx]
            if group.is_attention:
                plans.append((group, uris,
                              self._attn_token_indices(group, manager_block_idxes, block_table),
                              None))
            else:
                plans.append((group, uris, None,
                              self._state_block_ids(group, manager_block_idxes, block_table)))
        return plans

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        meta = typing.cast(TairKvCacheConnectorMetadata, self._get_connector_metadata())
        for load_req in meta.to_load_requests:
            if not load_req.need_load_locations:
                continue
            num_blocks = len(load_req.manager_block_idxes)
            plans = self._plan_group_transfers(
                load_req.need_load_locations, load_req.manager_block_idxes,
                load_req.all_block_ids)

            # Report failures against the block table vLLM can act on: map each
            # manager block to the logical block holding its first token. vLLM
            # truncates computed tokens at the first invalid block, so this is
            # sufficient for recovery.
            #
            # UPSTREAM LIMITATION: vLLM's invalid-block recovery
            # (Scheduler._update_requests_with_invalid_blocks, vllm/v1/core/
            # sched/scheduler.py) unpacks a single-group block table --
            # "(req_block_ids,) = ...get_block_ids(req_id)" under
            # "TODO (davidb): add support for hybrid memory allocator" -- so
            # for hybrid (multi-group) models a failed load CANNOT be reported
            # and is only logged; vLLM then decodes from whatever bytes the
            # partial load left in the paged cache, which can produce corrupt
            # output. Remove report_failures gating once upstream supports
            # multi-group invalid-block recovery.
            report_ids = []
            if self._num_groups == 1:
                # Index by the transferred group's own vLLM group index: with
                # skipped groups (EAGLE/MTP drafters) the single transferred
                # group is not necessarily group 0.
                only = self._group_metas[0]
                table = load_req.all_block_ids[only.group_idx]
                gbs = only.block_size
                report_ids = [table[(mb * self._manager_block_size) // gbs]
                              for mb in load_req.manager_block_idxes]
            done_cb = self._data_transfer.create_load_done_callback(
                load_req.req_id, self._tp_rank, meta.epoch,
                copy.copy(report_ids), num_blocks,
                report_failures=self._num_groups == 1)

            if plans is None:
                # Nothing submitted; report the whole load as failed.
                mr = MultiResult(1, done_cb)
                mr.submit_result(0, [False] * num_blocks * self._num_groups)
                continue

            per_task = self._extra_config.block_per_load_task
            task_num = sum(math.ceil(num_blocks / per_task) for _ in plans)
            multi_result = MultiResult(task_num, done_cb)
            task_idx = 0
            for group, uris, token_indices, block_ids in plans:
                task_idx = self._submit_group_tasks(
                    self._data_transfer.load_task, multi_result, task_idx,
                    group, uris, token_indices, block_ids, per_task)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        pass

    def wait_for_save(self):
        meta = typing.cast(TairKvCacheConnectorMetadata, self._get_connector_metadata())
        if not meta.to_save_requests:
            return
        ready_event = self._device_mod.Event()
        ready_event.record(self._device_mod.current_stream())

        for save_req in meta.to_save_requests:
            req = self._alive_requests[save_req.req_id]
            num_blocks = len(save_req.manager_block_idxes)
            plans = self._plan_group_transfers(
                save_req.target_locations, save_req.manager_block_idxes,
                req.block_ids_per_group)

            done_cb = self._data_transfer.create_save_done_callback(
                req.req_id, self._tp_rank, save_req.write_session_id, num_blocks)

            if plans is None:
                mr = MultiResult(1, done_cb)
                mr.submit_result(0, [False] * num_blocks * self._num_groups)
                continue

            per_task = self._extra_config.block_per_save_task
            task_num = sum(math.ceil(num_blocks / per_task) for _ in plans)
            multi_result = MultiResult(task_num, done_cb)
            task_idx = 0
            for group, uris, token_indices, block_ids in plans:
                task_idx = self._submit_group_tasks(
                    self._data_transfer.save_task, multi_result, task_idx,
                    group, uris, token_indices, block_ids, per_task, ready_event)
            if self._tp_rank == 0:
                req.scheduled_saving_count += 1

    def on_save_finished(self, write_session_id: str, save_context: SaveContext):
        for block_idx in range(len(save_context.locations)):
            fully_saved = all(save_context.result_per_rank[rank][block_idx]
                              for rank in range(self._tp_size))
            save_context.success_mask.append(fully_saved)
        logger.debug("finish_write_cache mask:%s session:%s",
                     save_context.success_mask, write_session_id)
        try:
            self._manager_client.finish_write_cache({
                "trace_id": "finish_%s" % write_session_id[:8],
                "instance_id": self._extra_config.instance_id,
                "write_session_id": write_session_id,
                "success_blocks": {"bool_masks": {"values": save_context.success_mask}},
            })
        except Exception as e:
            logger.warning("finish_write_cache failed, session: %s, error: %s",
                           write_session_id, e)

    def get_finished(self, finished_req_ids: set) -> Tuple[Optional[set], Optional[set]]:
        meta = typing.cast(TairKvCacheConnectorMetadata, self._get_connector_metadata())

        if self._tp_rank != 0:
            for finish_req in meta.to_finish_requests:
                self._alive_requests.pop(finish_req.req_id, None)
            return None, None

        finished_saving = []
        finished_saving_tasks, finished_loading_tasks = self._coordinator_server.get_finished_tasks()
        for req_id in finished_saving_tasks:
            req = self._alive_requests[req_id]
            req.sent_saving_count += 1
            assert req.sent_saving_count <= req.scheduled_saving_count
            if (req.need_report_after_saving_finished and
                    req.sent_saving_count == req.scheduled_saving_count):
                finished_saving.append(req_id)
                self._alive_requests.pop(req_id)

        for finish_req in meta.to_finish_requests:
            req = self._alive_requests.get(finish_req.req_id)
            if req is None:
                continue
            if req.sent_saving_count == req.scheduled_saving_count:
                finished_saving.append(req.req_id)
                self._alive_requests.pop(req.req_id)
            else:
                req.need_report_after_saving_finished = True
        return set(finished_saving), set(finished_loading_tasks)

    def get_block_ids_with_load_errors(self) -> set:
        if self._tp_rank != 0:
            return set()
        failed = self._coordinator_server.get_failed_loading_block_idxs()
        if failed:
            logger.warning("block_ids_with_load_errors: %s", failed)
        return failed

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        self._connector_metadata = connector_metadata
        meta = typing.cast(TairKvCacheConnectorMetadata, self._get_connector_metadata())
        for delta in meta.requests:
            if not delta.is_delta:
                self._alive_requests[delta.req_id] = ReqState.create_from_delta(delta)
            else:
                assert delta.req_id in self._alive_requests
                self._alive_requests[delta.req_id].update_from_delta(delta)

    # ------------------------------------------------------------------ #
    # Scheduler side
    # ------------------------------------------------------------------ #
    def get_num_new_matched_tokens(self, request: "Request",
                                   num_computed_tokens: int) -> Tuple[Optional[int], bool]:
        prev = self._alive_requests.get(request.request_id)
        if (prev is not None and prev.remote_matched_token_num
                and prev.block_ids_per_group):
            # The request already went through an external load (blocks were
            # allocated) and returned to WAITING -- a KV load failure or a
            # preemption. The manager may still advertise blocks whose storage
            # is gone, so re-matching risks an endless fail-reschedule loop;
            # recompute locally instead. (A pending re-query after a failed
            # allocation has empty block_ids_per_group and is not affected.)
            logger.warning("req:%s re-queried after an external load attempt, "
                           "skip external match", request.request_id)
            prev.local_matched_token_num = num_computed_tokens
            prev.remote_matched_token_num = 0
            prev.has_saved_block_num = num_computed_tokens // self._manager_block_size
            return 0, False

        computed_blocks = num_computed_tokens // self._manager_block_size

        is_query_done, need_load_locations = (
            self._location_query_manager.get_locations_for_query(request, computed_blocks))
        if not is_query_done:
            # async query in flight; vLLM will ask again
            return None, False

        new_matched_count = len(need_load_locations) * self._manager_block_size
        # This connector loads synchronously (load_kv_async=False), so vLLM will
        # schedule num_tokens - num_computed_tokens new tokens and asserts that
        # count is > 0 (vllm/v1/core/sched/scheduler.py). If the whole prompt is
        # externally cached, drop trailing blocks so at least one token is
        # recomputed locally.
        while new_matched_count and num_computed_tokens + new_matched_count >= request.num_tokens:
            need_load_locations = need_load_locations[:-1]
            new_matched_count -= self._manager_block_size
        total_remote_blocks = computed_blocks + len(need_load_locations)
        logger.info("req:%s matched %d external tokens", request.request_id, new_matched_count)

        if new_matched_count:
            self._waiting_to_load_requests.append(LoadRequest(
                req_id=request.request_id,
                manager_block_idxes=list(range(computed_blocks, total_remote_blocks)),
                need_load_locations=need_load_locations,
            ))

        self._alive_requests[request.request_id] = ReqState(
            req_id=request.request_id,
            token_ids=copy.copy(request.prompt_token_ids),
            block_ids_per_group=[],
            has_saved_block_num=total_remote_blocks,
            local_matched_token_num=num_computed_tokens,
            remote_matched_token_num=new_matched_count,
            vllm_request=request,
        )
        return new_matched_count, new_matched_count > 0

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        req_state = self._alive_requests.get(request.request_id)
        if req_state is None:
            return
        req_state.block_ids_per_group = [list(b) for b in blocks.get_block_ids()]

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = TairKvCacheConnectorMetadata(self._epoch)
        self._epoch += 1

        for load_req in self._waiting_to_load_requests:
            request = self._alive_requests[load_req.req_id]
            if not request.block_ids_per_group:
                # update_state_after_alloc was never called; vLLM will re-query.
                continue
            load_req.all_block_ids = [list(b) for b in request.block_ids_per_group]
            meta.add_load_request(load_req)
        self._waiting_to_load_requests = []

        for vllm_req in scheduler_output.scheduled_new_reqs:
            request = self._alive_requests[vllm_req.req_id]
            request.block_ids_per_group = [list(b) for b in vllm_req.block_ids]
            meta.add_req_state_to_worker(ReqStateToWorker(
                req_id=request.req_id,
                has_saved_block_num=request.has_saved_block_num,
                new_tokens_ids=request.token_ids,
                new_block_ids_per_group=request.block_ids_per_group,
                is_delta=False,
            ))

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            request = self._alive_requests[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_current_tokens = len(request.token_ids)
            new_token_ids = request.vllm_request.all_token_ids[
                num_current_tokens:num_current_tokens + num_new_tokens]
            request.token_ids.extend(new_token_ids)

            delta = ReqStateToWorker(
                req_id=request.req_id,
                has_saved_block_num=request.has_saved_block_num,
                new_tokens_ids=new_token_ids,
            )

            if hasattr(cached_reqs, "resumed_req_ids"):
                resumed = req_id in cached_reqs.resumed_req_ids
            else:
                resumed = cached_reqs.resumed_from_preemption[idx]

            new_block_ids = cached_reqs.new_block_ids[idx]
            if resumed:
                request.block_ids_per_group = [list(b) for b in new_block_ids]
                delta.resumed_from_preemption = True
                delta.new_block_ids_per_group = request.block_ids_per_group
            elif new_block_ids is not None:
                # https://github.com/vllm-project/vllm/pull/23262: may be None
                delta.new_block_ids_per_group = [list(b) for b in new_block_ids]
                for group_ids, new_ids in zip(request.block_ids_per_group,
                                              delta.new_block_ids_per_group):
                    group_ids.extend(new_ids)
            meta.add_req_state_to_worker(delta)

        for req in self._alive_requests.values():
            target_save_num = min(
                len(req.token_ids),
                self._num_allocated_blocks(req) * self._vllm_block_size) // self._manager_block_size
            if target_save_num > req.has_saved_block_num:
                req.scheduled_saving_count += 1
                self._http_executor.submit(
                    self.start_save_kvcache_async, req.req_id,
                    req.token_ids[:target_save_num * self._manager_block_size],
                    target_save_num)
            req.has_saved_block_num = target_save_num

        with self._waiting_to_save_requests_lock:
            new_save_reqs = self._waiting_to_save_requests
            self._waiting_to_save_requests = []
        for save_req in new_save_reqs:
            req = self._alive_requests.get(save_req.req_id)
            if req is None:
                logger.warning("request %s is not alive, skip saving", save_req.req_id)
                continue
            meta.add_save_request(save_req)
            req.sent_saving_count += 1
            if (req.need_report_after_saving_finished and
                    req.scheduled_saving_count == req.sent_saving_count):
                self._waiting_to_finish_requests.append(FinishRequest(req.req_id))
                self._alive_requests.pop(req.req_id)

        self.handle_canceled_save_req()

        for finish_req in self._waiting_to_finish_requests:
            meta.add_finish_request(finish_req)
        self._waiting_to_finish_requests = []
        return meta

    def start_save_kvcache_async(self, req_id, token_ids, target_save_num):
        request = {
            "trace_id": "%s_%d" % (req_id, self._epoch),
            "instance_id": self._extra_config.instance_id,
            "block_keys": [],
            "token_ids": token_ids,
            "write_timeout_seconds": self._extra_config.write_timeout_seconds,
        }
        try:
            response = self._manager_client.start_write_cache(request)
        except Exception as e:
            logger.warning("start_write_cache error, skip saving: %s", e)
            with self._canceled_save_request_ids_lock:
                self._canceled_save_request_ids.append(req_id)
            return

        locations = response["locations"]
        write_session_id = response["write_session_id"]

        if not locations:
            try:
                self._manager_client.finish_write_cache({
                    "trace_id": "finish_%s" % write_session_id[:8],
                    "instance_id": self._extra_config.instance_id,
                    "write_session_id": write_session_id,
                    "success_blocks": {"bool_masks": {"offset": 0}},
                })
            except Exception as e:
                logger.warning("finish_write_cache failed, session: %s, error: %s",
                               write_session_id, e)
            with self._canceled_save_request_ids_lock:
                self._canceled_save_request_ids.append(req_id)
            return

        need_block_idx = self.parse_block_mask_to_save_indices(response, target_save_num)
        message = CoordinateMessage(time.time(), SendBlockStartEvent(
            request_id=req_id, write_session_id=write_session_id, locations=locations))
        self._coordinator_client.send(CoordinateMsgSerializer.dumps(message))

        with self._waiting_to_save_requests_lock:
            self._waiting_to_save_requests.append(SaveRequest(
                req_id, locations, need_block_idx, write_session_id))

    def handle_canceled_save_req(self):
        with self._canceled_save_request_ids_lock:
            canceled = self._canceled_save_request_ids
            self._canceled_save_request_ids = []
        for req_id in canceled:
            # Cancellations come from http_executor threads; the request may
            # already have been finished and removed by the scheduler loop.
            req = self._alive_requests.get(req_id)
            if req is None:
                logger.warning("canceled save for unknown request %s, skip", req_id)
                continue
            req.sent_saving_count += 1
            if (req.need_report_after_saving_finished and
                    req.scheduled_saving_count == req.sent_saving_count):
                self._waiting_to_finish_requests.append(FinishRequest(req.req_id))
                self._alive_requests.pop(req.req_id)

    def get_finished_count(self):
        # Only rank0 reports finished requests.
        return 1

    def update_connector_output(self, connector_output: KVConnectorOutput):
        return

    def parse_block_mask_to_save_indices(self, response: dict, target_save_num: int) -> List[int]:
        block_mask = response.get("block_mask", {})
        if "offset" in block_mask:
            return list(range(block_mask["offset"], target_save_num))
        values = block_mask.get("bool_masks", {}).get("values", [])
        return [idx for idx, saved in enumerate(values) if not saved]

    # ------------------------------------------------------------------ #
    # Request finish
    # ------------------------------------------------------------------ #
    def request_finished_all_groups(
            self, request: "Request",
            block_ids: Tuple[List[int], ...]) -> Tuple[bool, Optional[dict]]:
        return self._finish_request(request)

    def request_finished(self, request: "Request",
                         block_ids: List[int]) -> Tuple[bool, Optional[dict]]:
        return self._finish_request(request)

    def _finish_request(self, request: "Request") -> Tuple[bool, Optional[dict]]:
        req = self._alive_requests.get(request.request_id)
        if req is None:
            logger.info("request_finished for unknown request: %s", request.request_id)
            return False, {}

        extra_info = {"local_matched_token_num": req.local_matched_token_num,
                      "remote_matched_token_num": req.remote_matched_token_num}

        if req.scheduled_saving_count == req.sent_saving_count:
            self._waiting_to_finish_requests.append(FinishRequest(req.req_id))
            self._alive_requests.pop(req.req_id)
            return True, extra_info

        # Saves still in flight; delay freeing the blocks until they land.
        req.need_report_after_saving_finished = True
        return True, extra_info
