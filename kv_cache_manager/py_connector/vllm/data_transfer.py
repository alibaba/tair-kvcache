"""Per-group KV cache transfer between vLLM's paged cache and KVCM storage.

Each ``TransferGroup`` is an independent transfer unit:

* Attention groups store token-granular KV; a manager block is gathered/scattered
  through the strided Triton kernel (``batch_gather_scatter_helper``) which handles
  both the contiguous full-attention layout and the block-strided hybrid layout.
* Mamba/linear/gdn groups store per-block opaque state; a manager block maps to a
  single logical block whose raw bytes are copied verbatim.

The transport itself is layout-agnostic: for every manager block we hand the SDK a
``BlockBuffer`` (a pinned CPU region) and the block's remote URI. Save gathers HBM
-> CPU then ``SaveKvCaches``; load ``LoadKvCaches`` -> CPU then scatters CPU -> HBM.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from kv_cache_manager.client.pybind import kvcm_py_client

from kv_cache_manager.py_connector.common.tp_coordinator import (
    CoordinateMsgSerializer, TpCoordinatorClient, CoordinateMessage,
    SendBlockFinishedEvent, LoadBlockFinishedEvent,
)
from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.common.types import KVCacheInfo, TransferGroup
from kv_cache_manager.py_connector.kernel import batch_gather_scatter_helper


def _get_device_module(device=None):
    """Return the torch device module matching the runtime device."""
    if device is not None and hasattr(torch, "get_device_module"):
        return torch.get_device_module(device)
    try:
        if hasattr(torch, "musa") and torch.musa.is_available():
            import torch_musa  # noqa: F401
            return torch.musa
    except Exception:
        pass
    return torch.cuda


class MultiResult:
    """Collect the per-block success flags of several async tasks and fire a
    callback once every task has reported. Each result is a list[bool] aligned
    with the manager blocks the task handled (in submission order)."""

    def __init__(self, size: int, callback):
        self._size = size
        self._results = [None] * size
        self._lock = threading.Lock()
        self._finished_num = 0
        self._callback = callback

    def submit_result(self, idx: int, result):
        with self._lock:
            assert self._results[idx] is None
            self._results[idx] = result
            self._finished_num += 1
            if self._finished_num == self._size:
                # Flatten in submission order.
                flat = [ok for part in self._results for ok in part]
                self._callback(flat)


class _PinnedBudget:
    """Byte budget for pinned staging buffers.

    Save/load tasks each stage ``len(valid) * per_block_bytes`` bytes of
    pinned host memory; with a 32-thread executor and no cap the worst case
    is GiB-scale pinned allocation. ``acquire`` blocks until the requested
    bytes fit under the capacity; a request larger than the whole capacity
    is allowed to run alone (never deadlocks)."""

    def __init__(self, capacity: int):
        assert capacity > 0
        self._capacity = capacity
        self._used = 0
        self._cond = threading.Condition()

    def acquire(self, nbytes: int):
        with self._cond:
            while self._used > 0 and self._used + nbytes > self._capacity:
                self._cond.wait()
            self._used += nbytes

    def release(self, nbytes: int):
        with self._cond:
            self._used -= nbytes
            assert self._used >= 0
            self._cond.notify_all()


class DataTransferManager:
    def __init__(self, kvcache_info: KVCacheInfo, manager_block_size: int,
                 transfer_client, coordinator_client: TpCoordinatorClient, extra_config):
        self._info = kvcache_info
        self._manager_block_size = manager_block_size
        self._transfer_client = transfer_client
        self._coordinator_client = coordinator_client
        self._extra_config = extra_config
        self._device = kvcache_info.device
        self._device_mod = _get_device_module(self._device)
        self._save_stream = self._device_mod.Stream()
        self._load_stream = self._device_mod.Stream()

        def _init_worker():
            self._device_mod.set_device(self._device)

        self._io_executor = ThreadPoolExecutor(
            max_workers=32, thread_name_prefix="kvcm_io_", initializer=_init_worker)

        capacity = self._staging_capacity_bytes(kvcache_info, extra_config)
        self._pinned_budget = _PinnedBudget(capacity)
        logger.info("pinned staging budget: %d bytes", capacity)

    @staticmethod
    def _staging_capacity_bytes(kvcache_info: KVCacheInfo, extra_config) -> int:
        """Pinned staging budget: explicit staging_buffer_max_bytes wins;
        otherwise two max-size tasks in flight per group (double buffering:
        one staging on CPU while another is in SDK I/O), floored at 1 GiB so
        small-block configs keep enough parallelism."""
        capacity = extra_config.staging_buffer_max_bytes
        if capacity > 0:
            return capacity
        per_task_blocks = max(extra_config.block_per_save_task,
                              extra_config.block_per_load_task)
        per_task_peak = max(g.per_block_bytes for g in kvcache_info.groups) \
            * per_task_blocks
        return max(2 * per_task_peak * len(kvcache_info.groups), 1 << 30)

    def submit_task(self, func, *args, **kwargs):
        return self._io_executor.submit(func, *args, **kwargs)

    # ------------------------------------------------------------------ #
    # BlockBuffer helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_block_buffers(base_ptr: int, per_block_bytes: int, count: int):
        buffers = []
        for i in range(count):
            buf = kvcm_py_client.BlockBuffer()
            iov = kvcm_py_client.Iov()
            iov.type = kvcm_py_client.MemoryType.CPU
            iov.base = base_ptr + i * per_block_bytes
            iov.size = per_block_bytes
            iov.ignore = False
            buf.iovs = [iov]
            buffers.append(buf)
        return buffers

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    def save_task(self, multi_result: MultiResult, task_idx, group: TransferGroup,
                  remote_uris, block_token_indices, block_ids, ready_event):
        """Gather one group's manager blocks from HBM and save them.

        block_token_indices: attention -> list[list[int]] flat token slots per block.
        block_ids:           state     -> list[int] block id per manager block.
        remote_uris:         positionally aligned with the manager blocks; None
                             where the manager allocated no location for this
                             group's spec (see ``_spec_groups``).

        A block is reported successful only when its data was actually written.
        Where a state group has no state (vLLM's null block in mamba "align"
        mode) the manager was already told so at start_write_cache time -- the
        block simply carries no spec for this group -- so nothing is written and
        nothing is claimed: the block is *excluded* from this group's verdict
        rather than reported as a success.
        """
        n = len(remote_uris)
        # Single exit: whatever happens below, the task reports, otherwise the
        # MultiResult callback never fires (submit_task drops the future, so an
        # escaping exception is silently swallowed) and the save session hangs.
        ok_mask = [False] * n
        try:
            # Three dispositions per block: abstain (this group holds no data
            # for it by design), transfer, or fail outright.
            skipped, failed = self._save_dispositions(group, remote_uris, block_ids, n)
            valid = [i for i in range(n) if i not in skipped and i not in failed]
            ok_mask = [None if i in skipped else False for i in range(n)]
            if valid:
                stage_bytes = len(valid) * group.per_block_bytes
                self._pinned_budget.acquire(stage_bytes)
                try:
                    self._save_valid_blocks(group, remote_uris,
                                            block_token_indices, block_ids,
                                            ready_event, valid, ok_mask)
                finally:
                    self._pinned_budget.release(stage_bytes)
        except Exception:
            # Fail the whole task: partially transferred blocks are unknown, so
            # report conservatively -- never publish a block we cannot vouch
            # for (abstentions included; that only costs hit rate, not truth).
            logger.exception("save task crashed group=%s blocks=%d, failing them all",
                             group.spec_name, n)
            ok_mask = [False] * n
        multi_result.submit_result(task_idx, ok_mask)

    def _save_dispositions(self, group: TransferGroup, remote_uris, block_ids, n):
        """Split the task's blocks into (abstained, failed); the rest transfer.

        A state group abstains for a manager block exactly when vLLM's block
        table points at the null block *and* the manager allocated no URI for
        the group's spec. The two must agree, because the scheduler derived the
        announced spec coverage from that very block table. Either disagreement
        fails the block:

        * location but no state -- writing the null block's bytes would publish
          a state the model never produced;
        * state but no location -- the state cannot be published at all.

        Attention KV is never sparse, so a missing location simply fails.
        """
        if group.is_attention:
            failed = {i for i in range(n) if remote_uris[i] is None}
            if failed:
                logger.warning("save group %s: %d/%d attention blocks have no "
                               "location, failing them", group.spec_name,
                               len(failed), n)
            return set(), failed
        skipped, failed = set(), set()
        for i in range(n):
            is_null = block_ids[i] == 0
            has_uri = remote_uris[i] is not None
            if is_null and not has_uri:
                skipped.add(i)
            elif is_null:
                failed.add(i)
                logger.warning("save group %s: block %d has a location but no "
                               "materialized state; failing it instead of "
                               "publishing bytes the model never produced",
                               group.spec_name, i)
            elif not has_uri:
                failed.add(i)
                logger.warning("save group %s: block %d has a materialized "
                               "state but no location to write it to; failing it",
                               group.spec_name, i)
        if skipped:
            logger.debug("save group %s: %d/%d blocks carry no state "
                         "(not published)", group.spec_name, len(skipped), n)
        return skipped, failed

    @staticmethod
    def _load_skipped_blocks(group: TransferGroup, remote_uris, block_ids, n) -> set:
        """Blocks this group has nothing to load into.

        Asymmetric with the save side on purpose: on load, a null target means
        vLLM does not *need* this group's data for that block (in mamba "align"
        mode only the block ending the reused prefix needs its state), so the
        block is skipped whatever the manager published. The reverse -- a real
        target with nothing published -- is a genuine failure: the request would
        run on an unwritten state.
        """
        if group.is_attention:
            missing = sum(uri is None for uri in remote_uris)
            if missing:
                logger.warning("load group %s: %d/%d attention blocks have no "
                               "location, failing them", group.spec_name, missing, n)
            return set()
        skipped = {i for i in range(n) if block_ids[i] == 0}
        if skipped:
            logger.debug("load group %s: %d/%d blocks need no state",
                         group.spec_name, len(skipped), n)
        return skipped

    def _save_valid_blocks(self, group, remote_uris,
                           block_token_indices, block_ids, ready_event,
                           valid, ok_mask):
        uris = [remote_uris[i] for i in valid]
        assert all(uri is not None for uri in uris), \
            f"group {group.spec_name}: save batch contains a block without a " \
            f"location; _save_dispositions must have failed it"
        cpu_buffer = torch.empty(len(valid) * group.per_block_bytes, dtype=torch.uint8,
                                 device="cpu", pin_memory=True)
        with self._device_mod.stream(self._save_stream):
            ready_event.wait()
            gpu_buffer = torch.empty(len(valid) * group.per_block_bytes,
                                     dtype=torch.uint8, device=self._device)
            if group.is_attention:
                view = gpu_buffer.view(self._info.dtype).view(
                    len(valid), group.num_kv_ptrs,
                    self._manager_block_size, group.per_token_dim)
                batch_gather_scatter_helper.batch_gather_kv_caches(
                    group.kvcache_ptr_tensor_gpu, view,
                    [block_token_indices[i] for i in valid],
                    list(range(len(valid))), self._manager_block_size,
                    group.per_token_dim,
                    kv_stride=group.kv_stride, block_stride=group.block_stride,
                    local_block_size=group.kernel_block_size)
            else:
                for out_i, i in enumerate(valid):
                    for layer_idx in range(group.layer_num):
                        dst = (out_i * group.layer_num + layer_idx) * group.page_size_bytes
                        gpu_buffer[dst:dst + group.page_size_bytes].copy_(
                            group.block_view_tensors[layer_idx][block_ids[i]])
            cpu_buffer.copy_(gpu_buffer, non_blocking=True)
            done = self._device_mod.Event()
            done.record(self._save_stream)
        done.synchronize()

        buffers = self._make_block_buffers(
            cpu_buffer.data_ptr(), group.per_block_bytes, len(valid))
        result = self._transfer_client.SaveKvCaches(uris, buffers)
        ok = (result[0] == kvcm_py_client.ClientErrorCode.ER_OK)
        if not ok:
            logger.warning("save task failed group=%s uris=%d result=%s",
                           group.spec_name, len(uris), result)
        for i in valid:
            ok_mask[i] = ok

    def create_save_done_callback(self, req_id, tp_rank, write_session_id, num_blocks):
        """block success = AND across all groups that had data for the block.

        Task results are ordered group0[blocks], group1[blocks], ... so a
        block's verdict is the stride-AND ``flat[i % num_blocks]``. ``None``
        entries mean "this group holds no data for this block by design" (see
        save_task) and are skipped: they neither pass nor fail the block. A
        block whose every group is None was never written at all and must not
        be published.
        """
        def cb(flat):
            is_success = [None] * num_blocks
            for i, ok in enumerate(flat):
                if ok is None:
                    continue
                b = i % num_blocks
                is_success[b] = ok if is_success[b] is None else (is_success[b] and ok)
            # No group wrote anything for the block -> nothing to publish.
            is_success = [bool(ok) for ok in is_success]
            msg = CoordinateMessage(time.time(), SendBlockFinishedEvent(
                request_id=req_id, tp_rank=tp_rank,
                write_session_id=write_session_id, is_success_list=is_success))
            self._coordinator_client.send(CoordinateMsgSerializer.dumps(msg))
        return cb

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #
    def load_task(self, multi_result: MultiResult, task_idx, group: TransferGroup,
                  remote_uris, block_token_indices, block_ids):
        """Load one group's manager blocks from storage into HBM.

        Mirror of ``save_task``: ``remote_uris`` is positionally aligned with
        the manager blocks and holds None where the published block carries no
        data for this group's spec. A state group's block is skipped only when
        vLLM's target is the null block *and* nothing was published -- i.e. the
        state is neither needed nor available. Everything else must transfer.
        """
        n = len(remote_uris)
        # Single exit, as in save_task: an escaping exception would silently
        # swallow the MultiResult callback, which under the connector's
        # synchronous-load contract leaves vLLM believing KV it never received.
        ok_mask = [False] * n
        try:
            skipped = self._load_skipped_blocks(group, remote_uris, block_ids, n)
            # A block we must restore but nothing was published for cannot be
            # loaded; fail it without letting it shift the staging batch.
            failed = {i for i in range(n)
                      if i not in skipped and remote_uris[i] is None}
            valid = [i for i in range(n) if i not in skipped and i not in failed]
            ok_mask = [None if i in skipped else False for i in range(n)]
            if valid:
                stage_bytes = len(valid) * group.per_block_bytes
                self._pinned_budget.acquire(stage_bytes)
                try:
                    ok = self._load_valid_blocks(group, remote_uris,
                                                 block_token_indices, block_ids,
                                                 valid)
                finally:
                    self._pinned_budget.release(stage_bytes)
                for i in valid:
                    ok_mask[i] = ok
        except Exception:
            logger.exception("load task crashed group=%s blocks=%d, failing them all",
                             group.spec_name, n)
            ok_mask = [False] * n
        multi_result.submit_result(task_idx, ok_mask)

    def _load_valid_blocks(self, group, remote_uris, block_token_indices,
                           block_ids, valid) -> bool:
        cpu_buffer = torch.empty(len(valid) * group.per_block_bytes, dtype=torch.uint8,
                                 device="cpu", pin_memory=True)
        buffers = self._make_block_buffers(cpu_buffer.data_ptr(),
                                           group.per_block_bytes, len(valid))
        uris = [remote_uris[i] for i in valid]
        assert all(uri is not None for uri in uris), \
            f"group {group.spec_name}: load batch contains a block without a " \
            f"location; load_task must have failed it"
        result = self._transfer_client.LoadKvCaches(uris, buffers)
        ok = (result == kvcm_py_client.ClientErrorCode.ER_OK)
        if ok:
            with self._device_mod.stream(self._load_stream):
                gpu_buffer = cpu_buffer.to(self._device, non_blocking=True)
                if group.is_attention:
                    view = gpu_buffer.view(self._info.dtype).view(
                        len(valid), group.num_kv_ptrs,
                        self._manager_block_size, group.per_token_dim)
                    batch_gather_scatter_helper.batch_scatter_kv_caches(
                        group.kvcache_ptr_tensor_gpu, view,
                        [block_token_indices[i] for i in valid],
                        list(range(len(valid))), self._manager_block_size,
                        group.per_token_dim,
                        kv_stride=group.kv_stride, block_stride=group.block_stride,
                        local_block_size=group.kernel_block_size)
                else:
                    for out_i, i in enumerate(valid):
                        for layer_idx in range(group.layer_num):
                            src = (out_i * group.layer_num + layer_idx) * group.page_size_bytes
                            group.block_view_tensors[layer_idx][block_ids[i]].copy_(
                                gpu_buffer[src:src + group.page_size_bytes])
                done = self._device_mod.Event()
                done.record(self._load_stream)
            done.synchronize()
        else:
            logger.warning("load task failed group=%s uris=%d result=%s",
                           group.spec_name, len(uris), result)
        return ok

    def create_load_done_callback(self, req_id, tp_rank, epoch, block_ids, num_blocks,
                                  report_failures=True):
        """A manager block is loaded only if every group that had data for it
        succeeded.

        ``None`` entries mean "this group has no data for this block by design"
        (mamba "align" interior blocks, see load_task) and are skipped, exactly
        as in the save callback.

        block_ids is the block table used to report vLLM-visible invalid block
        ids. vLLM's invalid-block recovery only understands single-group block
        tables, so multi-group (hybrid) connectors pass report_failures=False
        and rely on request rescheduling instead."""
        def cb(flat):
            merged = [None] * num_blocks
            for i, ok in enumerate(flat):
                if ok is None:
                    continue
                b = i % num_blocks
                merged[b] = ok if merged[b] is None else (merged[b] and ok)
            # A block no group loaded anything for was not restored.
            merged = [bool(ok) for ok in merged]
            failed = []
            if report_failures:
                failed = [block_ids[i] for i in range(min(num_blocks, len(block_ids)))
                          if not merged[i]]
            elif not all(merged):
                logger.warning("load failed for %d/%d blocks of req %s (hybrid: "
                               "not reporting invalid block ids)",
                               merged.count(False), num_blocks, req_id)
            msg = CoordinateMessage(time.time(), LoadBlockFinishedEvent(
                request_id=req_id, tp_rank=tp_rank, epoch=epoch, failed_block_idxs=failed))
            self._coordinator_client.send(CoordinateMsgSerializer.dumps(msg))
        return cb
