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
        block_ids:           state     -> list[int] block id per manager block;
                             id 0 is vLLM's null block: the boundary state was
                             never materialized, so that block cannot be saved.
        """
        n = len(remote_uris)
        if group.is_attention:
            valid = list(range(n))
        else:
            valid = [i for i in range(n) if block_ids[i] != 0]
            if len(valid) < n:
                logger.warning("save group %s: %d/%d blocks have no materialized "
                               "state, failing them", group.spec_name, n - len(valid), n)
        ok_mask = [False] * n
        if valid:
            cpu_buffer = torch.empty(len(valid) * group.per_block_bytes, dtype=torch.uint8,
                                     device="cpu", pin_memory=True)
            with self._device_mod.stream(self._save_stream):
                ready_event.wait()
                gpu_buffer = torch.empty(len(valid) * group.per_block_bytes,
                                         dtype=torch.uint8, device=self._device)
                if group.is_attention:
                    view = gpu_buffer.view(self._info.dtype).view(
                        len(valid), group.layer_num,
                        self._manager_block_size, group.per_token_dim)
                    batch_gather_scatter_helper.batch_gather_kv_caches(
                        group.kvcache_ptr_tensor_gpu, view, block_token_indices,
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
            uris = [remote_uris[i] for i in valid]
            result = self._transfer_client.SaveKvCaches(uris, buffers)
            ok = (result[0] == kvcm_py_client.ClientErrorCode.ER_OK)
            if not ok:
                logger.warning("save task failed group=%s uris=%d result=%s",
                               group.spec_name, len(uris), result)
            for i in valid:
                ok_mask[i] = ok
        multi_result.submit_result(task_idx, ok_mask)

    def create_save_done_callback(self, req_id, tp_rank, write_session_id, num_blocks):
        """block success = AND across all groups. task results are ordered
        group0[blocks], group1[blocks], ... so we AND stride-wise."""
        def cb(flat):
            is_success = [True] * num_blocks
            for i, ok in enumerate(flat):
                is_success[i % num_blocks] = is_success[i % num_blocks] and ok
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
        n = len(remote_uris)
        if not group.is_attention and any(b == 0 for b in block_ids):
            # Null block: nowhere to scatter the state. Should not happen for
            # loads (vLLM allocates real blocks for external tokens).
            logger.warning("load group %s: null block in targets, failing task",
                           group.spec_name)
            multi_result.submit_result(task_idx, [False] * n)
            return
        cpu_buffer = torch.empty(n * group.per_block_bytes, dtype=torch.uint8,
                                 device="cpu", pin_memory=True)
        buffers = self._make_block_buffers(cpu_buffer.data_ptr(), group.per_block_bytes, n)
        result = self._transfer_client.LoadKvCaches(remote_uris, buffers)
        ok = (result == kvcm_py_client.ClientErrorCode.ER_OK)
        if ok:
            with self._device_mod.stream(self._load_stream):
                gpu_buffer = cpu_buffer.to(self._device, non_blocking=True)
                if group.is_attention:
                    view = gpu_buffer.view(self._info.dtype).view(
                        n, group.layer_num, self._manager_block_size, group.per_token_dim)
                    batch_gather_scatter_helper.batch_scatter_kv_caches(
                        group.kvcache_ptr_tensor_gpu, view, block_token_indices,
                        list(range(n)), self._manager_block_size, group.per_token_dim,
                        kv_stride=group.kv_stride, block_stride=group.block_stride,
                        local_block_size=group.kernel_block_size)
                else:
                    for i, block_id in enumerate(block_ids):
                        for layer_idx in range(group.layer_num):
                            src = (i * group.layer_num + layer_idx) * group.page_size_bytes
                            group.block_view_tensors[layer_idx][block_id].copy_(
                                gpu_buffer[src:src + group.page_size_bytes])
                done = self._device_mod.Event()
                done.record(self._load_stream)
            done.synchronize()
        else:
            logger.warning("load task failed group=%s uris=%d result=%s",
                           group.spec_name, n, result)
        multi_result.submit_result(task_idx, [ok] * n)

    def create_load_done_callback(self, req_id, tp_rank, epoch, block_ids, num_blocks,
                                  report_failures=True):
        """A manager block is loaded only if every group succeeded for it.

        block_ids is the block table used to report vLLM-visible invalid block
        ids. vLLM's invalid-block recovery only understands single-group block
        tables, so multi-group (hybrid) connectors pass report_failures=False
        and rely on request rescheduling instead."""
        def cb(flat):
            merged = [True] * num_blocks
            for i, ok in enumerate(flat):
                merged[i % num_blocks] = merged[i % num_blocks] and ok
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
