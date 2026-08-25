"""Worker side of the connector: translation and data-plane transfer.

Owns the transfer client and the paged-cache views (TransferGroups). Every
instruction arriving through the metadata is self-contained (LoadRequest /
SaveRequest carry their own block tables), so the worker keeps no mirrored
request state -- tp0's per-request save-session ledger is the only
request-level bookkeeping. Per-step metadata is passed in explicitly by the
shell.
"""

import copy
import json
import typing
from typing import List, Optional, Tuple

from kv_cache_manager.client.pybind import kvcm_py_client

import torch
from vllm.distributed import get_tensor_model_parallel_rank

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.common.tp_coordinator import (
    SaveContext, TpCoordinatorClient, TpCoordinatorServer)
from kv_cache_manager.py_connector.vllm.data_transfer import (
    MultiResult, DataTransferManager, _get_device_module)
from kv_cache_manager.py_connector.vllm.metadata import (
    FinishRequest, TairKvCacheConnectorMetadata)
from kv_cache_manager.py_connector.vllm.transfer_types import (
    AttentionTransferGroup, KVCacheInfo, StateTransferGroup, TransferGroup,
    TransferPlan)
from kv_cache_manager.py_connector.vllm.vllm_common import (
    AttentionGroupMeta, GroupMeta, StateGroupMeta, attn_kv_views, spec_name)

if typing.TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.attention import AttentionMetadata


class ConnectorWorker:
    """State and hooks for the worker-role connector instance (one per TP rank)."""

    def __init__(self, extra_config, group_metas: List[GroupMeta],
                 manager_block_size: int, tp_size: int, host_ip: str,
                 manager_client, coordinator_client: TpCoordinatorClient,
                 register_response: dict):
        self._extra_config = extra_config
        self._group_metas = group_metas
        self._num_groups = len(group_metas)
        self._manager_block_size = manager_block_size
        self._tp_size = tp_size
        self._manager_client = manager_client
        self._coordinator_client = coordinator_client

        # Per-request in-flight save sessions (write_session granularity).
        # tp0 only; this is the worker's whole request-level state -- there is
        # no mirrored request table (every instruction is self-contained).
        self._pending_saves: dict = {}
        self._finish_pending: set = set()

        self._tp_rank = get_tensor_model_parallel_rank()
        self._device_mod = None
        port = extra_config.coordinator_base_port
        if self._tp_rank == 0:
            self._coordinator_server = TpCoordinatorServer(
                host_ip, port, tp_size, self.on_save_finished)

        self._self_spec_names = {
            meta.group_idx: spec_name(self._tp_rank, meta.group_idx)
            for meta in self._group_metas
        }
        max_group_bytes = max(m.per_block_bytes for m in self._group_metas)
        self._iov_size = max_group_bytes * extra_config.hf3fs_concurrent_io_block_count

        self._storage_configs = register_response["storage_configs"]
        sdk_backend_configs = self.parse_hf3fs_configs(self._storage_configs)
        transfer_client_json = {
            "instance_group": extra_config.instance_group,
            "instance_id": extra_config.instance_id,
            "block_size": self._manager_block_size,
            "sdk_config": {
                "thread_num": extra_config.sdk_thread_num,
                "queue_size": extra_config.sdk_queue_size,
                "sdk_backend_configs": sdk_backend_configs,
                "timeout_config": {
                    "get_timeout_ms": extra_config.sdk_get_timeout_ms,
                    "put_timeout_ms": extra_config.sdk_put_timeout_ms,
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
            self._tp_rank, self._tp_size, host_ip, port, self._num_groups)

    # ------------------------------------------------------------------ #
    # Storage config plumbing
    # ------------------------------------------------------------------ #
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
    # KV cache registration
    # ------------------------------------------------------------------ #
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv_caches = kv_caches
        first_attn = next(kv_caches[name]
                          for meta in self._group_metas
                          if isinstance(meta, AttentionGroupMeta)
                          for name in meta.layer_names)
        self._dtype = first_attn.dtype
        self._device = first_attn.device
        self._device_mod = _get_device_module(self._device)

        groups: List[TransferGroup] = []
        for meta in self._group_metas:
            if isinstance(meta, AttentionGroupMeta):
                groups.append(self._build_attention_group(meta, kv_caches))
            else:
                groups.append(self._build_state_group(meta, kv_caches))

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
            (g.spec_name, type(g).__name__, g.layer_num, g.per_block_bytes)
            for g in groups])

    def _build_attention_group(self, meta: AttentionGroupMeta,
                               kv_caches) -> AttentionTransferGroup:
        spec = self._self_spec_names[meta.group_idx]
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
        ref_views, kv_layout = attn_kv_views(ref)
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
        ptrs = [v.data_ptr() for t in tensors for v in attn_kv_views(t)[0]]
        ptr_tensor = torch.tensor(ptrs, dtype=torch.int64, device="cpu").to(self._device)
        return AttentionTransferGroup(
            group_idx=meta.group_idx,
            spec_name=spec,
            layer_names=meta.layer_names,
            block_size=meta.block_size,
            per_block_bytes=meta.per_block_bytes,
            layer_num=len(meta.layer_names),
            kv_layout=kv_layout,
            kvcache_ptr_tensor_gpu=ptr_tensor,
            num_kv_ptrs=len(ptrs),
            per_token_dim=per_token_dim,
            kernel_block_size=kernel_block_size,
            block_stride=block_stride,
        )

    def _build_state_group(self, meta: StateGroupMeta,
                           kv_caches) -> StateTransferGroup:
        # Mamba/state group: each layer is a list[Tensor] sharing one storage;
        # rebuild a (num_blocks, page_size_bytes) byte view for opaque copy.
        spec = self._self_spec_names[meta.group_idx]
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
        return StateTransferGroup(
            group_idx=meta.group_idx,
            spec_name=spec,
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
    def _attn_token_indices(self, group: AttentionTransferGroup, manager_block_idxes,
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

    def _state_block_ids(self, group: StateTransferGroup, manager_block_idxes,
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

    # ------------------------------------------------------------------ #
    # Load / save
    # ------------------------------------------------------------------ #
    def _iter_task_chunks(self, plans: List[TransferPlan], per_task: int):
        """Slice each plan's blocks into per-task chunks.

        Pure generator: the task index is the consumer's concern (enumerate
        there); this only reads the plans."""
        for plan in plans:
            n = len(plan.uris)
            for i in range(0, n, per_task):
                end = min(n, i + per_task)
                yield (plan.group,
                       plan.uris[i:end],
                       plan.token_indices[i:end] if plan.token_indices is not None else None,
                       plan.block_ids[i:end] if plan.block_ids is not None else None)

    def _plan_group_transfers(self, locations, manager_block_idxes,
                              block_ids_per_group) -> Optional[List[TransferPlan]]:
        """Build the per-group TransferPlans for a set of manager blocks.

        ``uris`` is positionally aligned with ``manager_block_idxes`` and may
        contain ``None`` where a block carries no data for that group's spec
        (hybrid per-block spec coverage, see ``build_spec_groups``). Deciding
        what a hole means is the transfer task's job: for attention groups a
        hole is a failure, for state groups it must agree with the block
        table (a null state block). Returns None only if the manager returned
        the wrong number of locations altogether.
        """
        num_blocks = len(manager_block_idxes)
        if len(locations) != num_blocks:
            logger.warning("%d locations for %d blocks, skip transfer",
                           len(locations), num_blocks)
            return None
        # One pass over the locations: per block, map spec name -> uri. The
        # per-group extraction below then reads by name instead of rescanning
        # every location's spec list for every group.
        per_block_uri_maps = [
            {s["name"]: s["uri"] for s in location.get("location_specs", [])}
            for location in locations]
        self._check_block_table_covers(block_ids_per_group)
        plans = []
        for group in self._kvcache_info.groups:
            uris = [m.get(group.spec_name) for m in per_block_uri_maps]
            # block_ids_per_group is indexed by the vLLM group index.
            block_table = block_ids_per_group[group.group_idx]
            if isinstance(group, AttentionTransferGroup):
                plans.append(TransferPlan(
                    group=group, uris=uris,
                    token_indices=self._attn_token_indices(
                        group, manager_block_idxes, block_table)))
            else:
                plans.append(TransferPlan(
                    group=group, uris=uris,
                    block_ids=self._state_block_ids(
                        group, manager_block_idxes, block_table)))
        return plans

    def _check_block_table_covers(self, block_ids_per_group) -> None:
        """``block_ids_per_group`` is indexed by the raw vLLM group index --
        a guarantee vLLM provides today
        (https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/sched/output.py,
        CachedRequestData.new_block_ids: one entry per kv cache group) but
        that we must not silently rely on. Fail loudly when the table cannot
        address every transferred group."""
        if not block_ids_per_group:
            return
        need = max(m.group_idx for m in self._group_metas)
        assert len(block_ids_per_group) > need, (
            f"block table has {len(block_ids_per_group)} groups but the "
            f"connector transfers vLLM group {need}; the group-index "
            f"alignment between vLLM and this connector is broken")

    def start_load_kv(self, forward_context: "ForwardContext",
                      meta: TairKvCacheConnectorMetadata, **kwargs) -> None:
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
            # for multi-group models (hybrid attn+mamba, or any model whose
            # vLLM block table has more than one entry -- skipped EAGLE/MTP
            # drafter groups still count) a failed load CANNOT be reported
            # and is only logged; vLLM then decodes from whatever bytes the
            # partial load left in the paged cache, which can produce corrupt
            # output. Gate on the block-table shape (all_block_ids is the
            # snapshot of kv_cache_manager.get_block_ids -- the very table
            # the recovery path unpacks), not on is_hybrid: a multi-group
            # attention-only model is not hybrid yet still breaks the unpack.
            # Remove the gating once upstream supports multi-group
            # invalid-block recovery.
            report_ids = []
            report_failures = len(load_req.all_block_ids) == 1
            if report_failures:
                # With exactly one vLLM block table the transferred group is
                # group 0, and it must be an attention group: a lone state
                # group would feed state block ids into upstream's
                # token-granular recovery math (it divides by block_size).
                # Both hold for every supported single-group model today;
                # assert so a future unsupported shape fails loudly instead
                # of reporting nonsense.
                only = self._group_metas[0]
                assert isinstance(only, AttentionGroupMeta), (
                    f"single vLLM block table but the transferred group is "
                    f"{type(only).__name__}, not attention; cannot report "
                    f"invalid blocks")
                assert only.group_idx == 0, (
                    f"single vLLM block table but the transferred group is "
                    f"group {only.group_idx}; group-index alignment broken")
                table = load_req.all_block_ids[0]
                gbs = only.block_size
                report_ids = [table[(mb * self._manager_block_size) // gbs]
                              for mb in load_req.manager_block_idxes]
            done_cb = self._data_transfer.create_load_done_callback(
                load_req.req_id, self._tp_rank, meta.epoch,
                copy.copy(report_ids), num_blocks,
                report_failures=report_failures)

            if plans is None:
                # Nothing submitted; report the whole load as failed.
                mr = MultiResult(1, done_cb)
                mr.submit_result(0, [False] * num_blocks * self._num_groups)
                continue

            chunks = list(self._iter_task_chunks(
                plans, self._extra_config.block_per_load_task))
            multi_result = MultiResult(len(chunks), done_cb)
            for task_idx, chunk in enumerate(chunks):
                self._data_transfer.submit_task(
                    self._data_transfer.load_task, multi_result, task_idx, *chunk)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        pass

    def wait_for_save(self, meta: TairKvCacheConnectorMetadata):
        if not meta.to_save_requests:
            return
        ready_event = self._device_mod.Event()
        ready_event.record(self._device_mod.current_stream())

        for save_req in meta.to_save_requests:
            num_blocks = len(save_req.manager_block_idxes)
            plans = self._plan_group_transfers(
                save_req.target_locations, save_req.manager_block_idxes,
                save_req.all_block_ids)

            done_cb = self._data_transfer.create_save_done_callback(
                save_req.req_id, self._tp_rank, save_req.write_session_id, num_blocks)

            if plans is None:
                mr = MultiResult(1, done_cb)
                mr.submit_result(0, [False] * num_blocks * self._num_groups)
                continue

            chunks = list(self._iter_task_chunks(
                plans, self._extra_config.block_per_save_task))
            multi_result = MultiResult(len(chunks), done_cb)
            for task_idx, chunk in enumerate(chunks):
                self._data_transfer.submit_task(
                    self._data_transfer.save_task, multi_result, task_idx,
                    *chunk, ready_event)
            if self._tp_rank == 0:
                # One session in flight; the coordinator's completion event
                # for this req decrements it (see get_finished).
                self._pending_saves[save_req.req_id] = \
                    self._pending_saves.get(save_req.req_id, 0) + 1

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

    def get_finished(self, finished_req_ids: set,
                     meta: TairKvCacheConnectorMetadata) -> Tuple[Optional[set], Optional[set]]:
        """Report request completion to vLLM.

        A request is reported once its last save session has settled on the
        coordinator (tp0 aggregates the per-rank verdicts into
        finish_write_cache). The scheduler's FinishRequest -- sent when its
        own ledger says the request is settled -- either reports immediately
        or defers until the last session lands."""
        if self._tp_rank != 0:
            for finish_req in meta.to_finish_requests:
                self._pending_saves.pop(finish_req.req_id, None)
                self._finish_pending.discard(finish_req.req_id)
            return None, None

        finished_saving = []
        finished_saving_tasks, finished_loading_tasks = self._coordinator_server.get_finished_tasks()
        for req_id in finished_saving_tasks:
            remaining = self._pending_saves.get(req_id, 0) - 1
            if remaining > 0:
                self._pending_saves[req_id] = remaining
                continue
            self._pending_saves.pop(req_id, None)
            if req_id in self._finish_pending:
                self._finish_pending.discard(req_id)
                finished_saving.append(req_id)

        for finish_req in meta.to_finish_requests:
            if finish_req.req_id not in self._pending_saves:
                finished_saving.append(finish_req.req_id)
            else:
                self._finish_pending.add(finish_req.req_id)
        return set(finished_saving), set(finished_loading_tasks)

    def get_block_ids_with_load_errors(self) -> set:
        if self._tp_rank != 0:
            return set()
        failed = self._coordinator_server.get_failed_loading_block_idxs()
        if failed:
            logger.warning("block_ids_with_load_errors: %s", failed)
        return failed
