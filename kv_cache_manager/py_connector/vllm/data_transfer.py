"""Per-group KV cache transfer between vLLM's paged cache and KVCM storage.

Each ``TransferGroup`` is an independent transfer unit:

* Attention groups store token-granular KV; a manager block is gathered/scattered
  through the strided Triton kernel (``batch_gather_scatter_helper``) which handles
  both the contiguous full-attention layout and the block-strided hybrid layout.
* Mamba/linear/gdn groups store per-block opaque state; a manager block maps to a
  single logical block whose raw bytes are copied verbatim.

The transport itself is layout-agnostic and **zero-VRAM**: for every manager
block we hand the SDK a ``BlockBuffer`` (a pinned CPU region) and the block's
remote URI. Save gathers HBM -> pinned host directly and ``SaveKvCaches``;
load ``LoadKvCaches`` -> pinned host, then scatters pinned -> HBM directly.
The gather/scatter kernel addresses host pinned memory over PCIe (UVA
zero-copy), and state-group copies use plain ``copy_`` between the GPU tensors
and the pinned slices -- no device-side staging buffer exists anywhere on the
data path, so the connector competes with the engine for exactly zero HBM.
"""

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, List, Optional

import torch

# kvcm_py_client is the compiled pybind11 client; it ships no type stubs.
from kv_cache_manager.client.pybind import kvcm_py_client  # ty: ignore[unresolved-import]

from kv_cache_manager.py_connector.common.tp_coordinator import TpCoordinatorClient
from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.vllm.config import TairKvCacheConnectorExtraConfig
from kv_cache_manager.py_connector.vllm.transfer_backend import (
    BlockVerdicts,
    MultiResult,
    get_device_module,
)
from kv_cache_manager.py_connector.vllm.transfer_types import (
    AttentionTransferGroup,
    KVCacheInfo,
    TransferGroup,
)


class _StagingPool:
    """Bounded, pre-allocated *pinned host* slots for one TransferGroup.

    Save and load share one pool. The slots are both the SDK's I/O buffers
    and the kernel's gather/scatter target: the strided kernel and plain
    ``copy_`` read/write host pinned memory directly over PCIe (UVA
    zero-copy), so the pool needs no device-side mirror and the connector
    reserves zero HBM. An exhausted pool blocks the acquiring task --
    backpressure -- instead of failing. Slots are handed out as *contiguous*
    runs because the kernel view needs one piece of memory.

    Slots are reused once their task reports. This is the historical
    origin/main behaviour: after an SDK timeout/error a background DMA may
    still touch the buffer for a while (the deadline contract is not
    upstream yet), and a slot reused in that window can be scribbled on --
    an accepted trade-off for removing the GPU staging copy.
    """

    def __init__(
        self, device: torch.device, per_block_bytes: int, max_blocks: int
    ) -> None:
        if max_blocks <= 0:
            raise ValueError("staging pool must have at least one block slot")
        self.block_bytes = per_block_bytes
        self.max_blocks = max_blocks
        self._cond = threading.Condition()
        total = max_blocks * per_block_bytes
        # Pinned host memory needs a CUDA context; on other devices (tests,
        # CPU-only runs) fall back to pageable memory.
        self._cpu = torch.empty(
            total, dtype=torch.uint8, device="cpu", pin_memory=(device.type == "cuda")
        )
        # Free runs as [start, start+len) block ranges, kept sorted by start.
        self._runs = [[0, max_blocks]]

    def acquire(self, n: int) -> int:
        """Block until a contiguous run of ``n`` block slots is free; return
        its starting block index."""
        if n <= 0:
            raise ValueError("must acquire at least one block slot")
        if n > self.max_blocks:
            raise ValueError(
                f"staging task of {n} blocks exceeds the pool capacity "
                f"{self.max_blocks}; raise staging_pool_blocks or shrink "
                f"block_per_save_task/block_per_load_task"
            )
        with self._cond:
            while True:
                for i, (start, length) in enumerate(self._runs):
                    if length >= n:
                        rest = length - n
                        if rest:
                            self._runs[i] = [start + n, rest]
                        else:
                            self._runs.pop(i)
                        return start
                self._cond.wait()

    def release(self, start: int, n: int) -> None:
        with self._cond:
            pos, run = 0, [start, n]
            for pos, (s, _len) in enumerate(self._runs):
                if s > start:
                    break
            else:
                pos = len(self._runs)
            self._runs.insert(pos, run)
            # Merge with the neighbours the release just glued together.
            merged = []
            for s, l in self._runs:
                if merged and merged[-1][0] + merged[-1][1] == s:
                    merged[-1][1] += l
                else:
                    merged.append([s, l])
            self._runs = merged
            self._cond.notify_all()

    def cpu_view(self, start: int, n: int) -> torch.Tensor:
        b = self.block_bytes
        return self._cpu[start * b : (start + n) * b]


def _effective_pool_blocks(
    configured: int, need: int, block_bytes: int, max_bytes: int
) -> int:
    """Blocks per staging-pool group under the per-group pinned-RAM ceiling.

    min(configured, max_bytes // block_bytes), floored at one full task
    batch: a task stages its whole batch contiguously, so the pool can never
    be smaller than the batch -- even when the byte cap alone would say so
    (the caller warns in that case). Pure so the sizing contract is
    unit-pinnable.
    """
    by_bytes = max_bytes // block_bytes if block_bytes > 0 else configured
    return max(need, min(configured, by_bytes))


class DataTransferManager(BlockVerdicts):
    def __init__(
        self,
        kvcache_info: KVCacheInfo,
        manager_block_size: int,
        transfer_client: Any,
        coordinator_client: TpCoordinatorClient,
        extra_config: TairKvCacheConnectorExtraConfig,
    ) -> None:
        super().__init__(coordinator_client)
        self._info = kvcache_info
        self._manager_block_size = manager_block_size
        self._transfer_client = transfer_client
        self._extra_config = extra_config
        self._device = kvcache_info.device
        self._device_mod = get_device_module(self._device)
        # The Triton kernels are this package's only Triton consumer. Importing
        # them here rather than at module scope keeps the module (and therefore
        # the connector) importable on a host without Triton, while a GPU host
        # that lost the kernel still fails at engine start instead of at the
        # first transfer.
        from kv_cache_manager.py_connector.kernel import batch_gather_scatter_helper

        self._kernel = batch_gather_scatter_helper
        self._save_stream = self._device_mod.Stream()
        self._load_stream = self._device_mod.Stream()

        pool_blocks = extra_config.staging_pool_blocks
        need = max(extra_config.block_per_save_task, extra_config.block_per_load_task)
        if pool_blocks < need:
            raise ValueError(
                f"staging_pool_blocks={pool_blocks} is smaller than the "
                f"largest task batch ({need}); one task stages its whole "
                f"batch contiguously, so the pool must cover it -- raise "
                f"staging_pool_blocks or shrink block_per_save_task/"
                f"block_per_load_task"
            )
        # One pool per group: block shapes differ between attention and state
        # groups. The pool is pinned host memory only -- the kernel reaches it
        # over PCIe -- so the connector's device-memory footprint is zero.
        # The byte cap keeps the configured block count honest for groups
        # with large blocks (hybrid state/attention blocks are ~17.3 MiB vs
        # ~0.875 MiB for full attention): the same count would otherwise pin
        # ~17 GiB of host RAM per group and engine start dies in the pinned
        # allocator.
        self._pools = {}
        for g in kvcache_info.groups:
            blocks = _effective_pool_blocks(
                pool_blocks,
                need,
                g.per_block_bytes,
                extra_config.staging_pool_max_bytes_per_group,
            )
            if blocks < pool_blocks:
                logger.warning(
                    "staging pool %s: %d blocks x %d bytes would pin "
                    "%.1f GiB; capped to %d blocks (%.1f MiB) by "
                    "staging_pool_max_bytes_per_group=%d",
                    g.spec_name,
                    pool_blocks,
                    g.per_block_bytes,
                    pool_blocks * g.per_block_bytes / 2**30,
                    blocks,
                    blocks * g.per_block_bytes / 2**20,
                    extra_config.staging_pool_max_bytes_per_group,
                )
            self._pools[g.spec_name] = _StagingPool(
                self._device, g.per_block_bytes, blocks
            )
        for name, pool in self._pools.items():
            logger.info(
                "staging pool %s: %d blocks x %d bytes (pinned %.1f MiB, GPU 0)",
                name,
                pool.max_blocks,
                pool.block_bytes,
                pool.max_blocks * pool.block_bytes / 2**20,
            )

        def _init_worker() -> None:
            self._device_mod.set_device(self._device)

        self._io_executor = ThreadPoolExecutor(
            max_workers=32, thread_name_prefix="kvcm_io_", initializer=_init_worker
        )

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        return self._io_executor.submit(func, *args, **kwargs)

    def shutdown(self) -> None:
        # wait=False: queued transfers finish on their own, so teardown never
        # blocks engine exit.
        self._io_executor.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # BlockBuffer helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_block_buffers(
        base_ptr: int, per_block_bytes: int, count: int
    ) -> List[kvcm_py_client.BlockBuffer]:
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
    def save_task(
        self,
        multi_result: MultiResult,
        task_idx: int,
        group: TransferGroup,
        remote_uris: List[Optional[str]],
        block_token_indices: Optional[List[List[int]]],
        block_ids: Optional[List[int]],
        ready_event: Any,
    ) -> None:
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
        # MultiResult callback never fires (submit drops the future, so an
        # escaping exception is silently swallowed) and the save session hangs.
        ok_mask = [False] * n
        try:
            # Three dispositions per block: abstain (this group holds no data
            # for it by design), transfer, or fail outright.
            skipped, failed = self._save_dispositions(group, remote_uris, block_ids, n)
            valid = [i for i in range(n) if i not in skipped and i not in failed]
            ok_mask = [None if i in skipped else False for i in range(n)]
            if valid:
                self._save_valid_blocks(
                    group,
                    remote_uris,
                    block_token_indices,
                    block_ids,
                    ready_event,
                    valid,
                    ok_mask,
                )
        except Exception:
            # Fail the whole task: partially transferred blocks are unknown, so
            # report conservatively -- never publish a block we cannot vouch
            # for (abstentions included; that only costs hit rate, not truth).
            logger.exception(
                "save task crashed group=%s blocks=%d, failing them all",
                group.spec_name,
                n,
            )
            ok_mask = [False] * n
        multi_result.submit_result(task_idx, ok_mask)

    def _save_valid_blocks(
        self,
        group: TransferGroup,
        remote_uris: List[Optional[str]],
        block_token_indices: Optional[List[List[int]]],
        block_ids: Optional[List[int]],
        ready_event: Any,
        valid: List[int],
        ok_mask: List[Optional[bool]],
    ) -> None:
        uris = [remote_uris[i] for i in valid]
        assert all(uri is not None for uri in uris), (
            f"group {group.spec_name}: save batch contains a block without a "
            f"location; _save_dispositions must have failed it"
        )
        # Wait for the engine's forward pass *before* taking a pool slot:
        # the slot is needed only for the gather + SDK transfer, so holding
        # it while the model still runs only extends the pool occupancy and
        # queues concurrent loads behind a save that is not even staging.
        ready_event.wait()
        pool = self._pools[group.spec_name]
        start = pool.acquire(len(valid))
        try:
            cpu_buffer = pool.cpu_view(start, len(valid))
            with self._device_mod.stream(self._save_stream):
                # Gather straight into the pinned host slot: the kernel
                # (attention) and copy_ (state) write host pinned memory
                # directly over PCIe; no device-side staging copy exists.
                if isinstance(group, AttentionTransferGroup):
                    view = cpu_buffer.view(self._info.dtype).view(
                        len(valid),
                        group.num_kv_ptrs,
                        self._manager_block_size,
                        group.per_token_dim,
                    )
                    self._kernel.batch_gather_kv_caches(
                        group.kvcache_ptr_tensor_gpu,
                        view,
                        # Attention tasks always carry token indices.
                        [block_token_indices[i] for i in valid],  # ty: ignore[not-subscriptable]
                        list(range(len(valid))),
                        self._manager_block_size,
                        group.per_token_dim,
                        block_stride=group.block_stride,
                        local_block_size=group.kernel_block_size,
                    )
                else:
                    # State tasks always carry block ids and state views
                    # (see _iter_task_chunks).
                    for out_i, i in enumerate(valid):
                        for layer_idx in range(group.layer_num):
                            dst = (
                                out_i * group.layer_num + layer_idx
                            ) * group.page_size_bytes  # ty: ignore[unresolved-attribute]
                            cpu_buffer[dst : dst + group.page_size_bytes].copy_(  # ty: ignore[unresolved-attribute]
                                group.block_view_tensors[layer_idx][  # ty: ignore[unresolved-attribute]
                                    block_ids[i]  # ty: ignore[not-subscriptable]
                                ],
                                non_blocking=True,
                            )
                done = self._device_mod.Event()
                done.record(self._save_stream)
            done.synchronize()

            buffers = self._make_block_buffers(
                cpu_buffer.data_ptr(), group.per_block_bytes, len(valid)
            )
            result = self._transfer_client.SaveKvCaches(uris, buffers)
            ok = result[0] == kvcm_py_client.ClientErrorCode.ER_OK
            if not ok:
                logger.warning(
                    "save task failed group=%s uris=%d result=%s",
                    group.spec_name,
                    len(uris),
                    result,
                )
            for i in valid:
                ok_mask[i] = ok
        except BaseException:
            # Drain the stream before the slots go back: a failed task may
            # have left kernel/copy work enqueued against the staging views.
            self._save_stream.synchronize()
            raise
        finally:
            pool.release(start, len(valid))

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #
    def load_task(
        self,
        multi_result: MultiResult,
        task_idx: int,
        group: TransferGroup,
        remote_uris: List[Optional[str]],
        block_token_indices: Optional[List[List[int]]],
        block_ids: Optional[List[int]],
    ) -> None:
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
            failed = {
                i for i in range(n) if i not in skipped and remote_uris[i] is None
            }
            valid = [i for i in range(n) if i not in skipped and i not in failed]
            ok_mask = [None if i in skipped else False for i in range(n)]
            if valid:
                ok = self._load_valid_blocks(
                    group, remote_uris, block_token_indices, block_ids, valid
                )
                for i in valid:
                    ok_mask[i] = ok
        except Exception:
            logger.exception(
                "load task crashed group=%s blocks=%d, failing them all",
                group.spec_name,
                n,
            )
            ok_mask = [False] * n
        multi_result.submit_result(task_idx, ok_mask)

    def _load_valid_blocks(
        self,
        group: TransferGroup,
        remote_uris: List[Optional[str]],
        block_token_indices: Optional[List[List[int]]],
        block_ids: Optional[List[int]],
        valid: List[int],
    ) -> bool:
        pool = self._pools[group.spec_name]
        start = pool.acquire(len(valid))
        try:
            cpu_buffer = pool.cpu_view(start, len(valid))
            buffers = self._make_block_buffers(
                cpu_buffer.data_ptr(), group.per_block_bytes, len(valid)
            )
            uris = [remote_uris[i] for i in valid]
            assert all(uri is not None for uri in uris), (
                f"group {group.spec_name}: load batch contains a block without a "
                f"location; load_task must have failed it"
            )
            result = self._transfer_client.LoadKvCaches(uris, buffers)
            ok = result == kvcm_py_client.ClientErrorCode.ER_OK
            if ok:
                with self._device_mod.stream(self._load_stream):
                    # Scatter straight out of the pinned host slot: the
                    # kernel (attention) and copy_ (state) read host pinned
                    # memory directly over PCIe; no device-side staging copy.
                    if isinstance(group, AttentionTransferGroup):
                        view = cpu_buffer.view(self._info.dtype).view(
                            len(valid),
                            group.num_kv_ptrs,
                            self._manager_block_size,
                            group.per_token_dim,
                        )
                        self._kernel.batch_scatter_kv_caches(
                            group.kvcache_ptr_tensor_gpu,
                            view,
                            # Attention tasks always carry token indices
                            # (see _iter_task_chunks).
                            [block_token_indices[i] for i in valid],  # ty: ignore[not-subscriptable]
                            list(range(len(valid))),
                            self._manager_block_size,
                            group.per_token_dim,
                            block_stride=group.block_stride,
                            local_block_size=group.kernel_block_size,
                        )
                    else:
                        # State tasks always carry block ids and state views
                        # (see _iter_task_chunks).
                        for out_i, i in enumerate(valid):
                            for layer_idx in range(group.layer_num):
                                src = (
                                    out_i * group.layer_num + layer_idx
                                ) * group.page_size_bytes  # ty: ignore[unresolved-attribute]
                                group.block_view_tensors[layer_idx][  # ty: ignore[unresolved-attribute]
                                    block_ids[i]  # ty: ignore[not-subscriptable]
                                ].copy_(
                                    cpu_buffer[src : src + group.page_size_bytes],  # ty: ignore[unresolved-attribute]
                                    non_blocking=True,
                                )
                    done = self._device_mod.Event()
                    done.record(self._load_stream)
                done.synchronize()
            else:
                logger.warning(
                    "load task failed group=%s uris=%d result=%s",
                    group.spec_name,
                    len(uris),
                    result,
                )
            return ok
        except BaseException:
            # Drain the stream before the slots go back (as in save).
            self._load_stream.synchronize()
            raise
        finally:
            pool.release(start, len(valid))
