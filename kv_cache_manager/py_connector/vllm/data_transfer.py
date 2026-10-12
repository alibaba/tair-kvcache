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

Pinned host RAM is the connector's only meaningful footprint, so the staging
arenas are sized *per staging shape* rather than per transfer bucket: buckets
whose per-token geometry and element type agree share one arena
(``plan_staging_pools``), which keeps the pinned total proportional to the
transfer window instead of to the number of vLLM groups.
"""

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

# kvcm_py_client is the compiled pybind11 client; it ships no type stubs.
from kv_cache_manager.client.pybind import kvcm_py_client  # ty: ignore[unresolved-import]

from kv_cache_manager.py_connector.common.tp_coordinator import (
    CoordinateMsgSerializer,
    TpCoordinatorClient,
    CoordinateMessage,
    SendBlockFinishedEvent,
    LoadBlockFinishedEvent,
)
from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.vllm.config import TairKvCacheConnectorExtraConfig
from kv_cache_manager.py_connector.vllm.transfer_types import (
    AttentionTransferGroup,
    KVCacheInfo,
    NullBlockRow,
    TransferGroup,
)
from kv_cache_manager.py_connector.vllm.vllm_common import (
    AttentionGroupMeta,
    GroupMeta,
)
from kv_cache_manager.py_connector.kernel import batch_gather_scatter_helper


def _get_device_module(device: Optional[torch.device] = None) -> Any:
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


class _StagingPool:
    """Bounded, pre-allocated *pinned host* slots shared by same-shape buckets.

    One pool serves every transfer bucket of one staging shape (see
    ``plan_staging_pools``): the slots are both the SDK's I/O buffers and the
    kernel's gather/scatter target -- the strided kernel and plain ``copy_``
    read/write host pinned memory directly over PCIe (UVA zero-copy), so the
    pool needs no device-side mirror and the connector reserves zero HBM. An
    exhausted pool blocks the acquiring task -- backpressure -- instead of
    failing. Slots are handed out as *contiguous* runs because the kernel view
    needs one piece of memory; the slot is sized for the *largest* member block,
    so a member with smaller blocks stages a dense prefix of the slots it
    borrows (``cpu_view``) and never touches its neighbours' bytes.

    Slots are reused once their task reports. This is the historical
    origin/main behaviour: after an SDK timeout/error a background DMA may
    still touch the buffer for a while (the deadline contract is not
    upstream yet), and a slot reused in that window can be scribbled on --
    an accepted trade-off for removing the GPU staging copy.
    """

    def __init__(
        self,
        device: torch.device,
        per_block_bytes: int,
        max_blocks: int,
        reserved_blocks: int = 0,
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
        # ``reserved_blocks`` top slots are the transfer tasks' guaranteed
        # staging room: the eager capture may only borrow from the rest (see
        # acquire), so a task always finds free slots to make progress even
        # while captures hold every shared slot -- without the split, a
        # captured slot would be released by a task that cannot start.
        self.reserved_blocks = min(max(0, reserved_blocks), max_blocks)
        self._runs = [[0, max_blocks - self.reserved_blocks]]
        self._reserved_runs = (
            [[max_blocks - self.reserved_blocks, self.reserved_blocks]]
            if self.reserved_blocks
            else []
        )

    def acquire(
        self, n: int, timeout: Optional[float] = None, *, keep_reserved: bool = False
    ) -> Optional[int]:
        """Block until a contiguous run of ``n`` block slots is free; return
        its starting block index, or None when ``timeout`` (seconds) expires
        first.

        The optional deadline exists for the eager capture, whose slots are
        held until the write session lands -- i.e. until the engine makes
        another step. Waiting for those slots forever would stall the engine
        against its own sessions, so the capture waits a bounded time and
        degrades instead (see DataTransferManager.capture_blocks): the
        transfer tasks keep the historical blocking behaviour.

        ``keep_reserved`` (the capture) searches only the shared part of the
        arena and leaves the reserve to the transfer tasks: a capture can never
        hold the last run a task needs, and a captured slot is only released by
        a task, which now always has room to start.
        """
        if n <= 0:
            raise ValueError("must acquire at least one block slot")
        if n > self.max_blocks:
            raise ValueError(
                f"staging task of {n} blocks exceeds the pool capacity "
                f"{self.max_blocks}; a task stages its whole batch as one run "
                f"(see DataTransferManager.save_task_blocks)"
            )
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                start = self._take_run(n, keep_reserved=keep_reserved)
                if start is not None:
                    return start
                if deadline is None:
                    self._cond.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    @property
    def free_blocks(self) -> int:
        """Free slots, reserve included: the arena's accounting, for tests."""
        with self._cond:
            return sum(length for _s, length in self._runs) + sum(
                length for _s, length in self._reserved_runs
            )

    def _take_run(self, n: int, keep_reserved: bool = False) -> Optional[int]:
        """Carve the first run of at least ``n`` slots; caller holds the lock."""
        lists = [self._runs] if keep_reserved else [self._runs, self._reserved_runs]
        for runs in lists:
            for i, (start, length) in enumerate(runs):
                if length >= n:
                    rest = length - n
                    if rest:
                        runs[i] = [start + n, rest]
                    else:
                        runs.pop(i)
                    return start
        return None

    def release(self, start: int, n: int) -> None:
        if n <= 0 or start < 0 or start + n > self.max_blocks:
            raise ValueError(
                f"release of [{start}, {start + n}) is outside the pool "
                f"(0..{self.max_blocks})"
            )
        with self._cond:
            target = (
                self._reserved_runs
                if start >= self.max_blocks - self.reserved_blocks
                else self._runs
            )
            for s, length in target:
                if s < start + n and start < s + length:
                    # Releasing an already-free range means two owners gave the
                    # same slots back: the arena would hand them to two
                    # transfers at once and the pinned budget would drift. Fail
                    # loudly instead of corrupting silently (review_r1 S1).
                    raise ValueError(
                        f"slots [{start}, {start + n}) of the pool "
                        f"(block_bytes={self.block_bytes}, "
                        f"max_blocks={self.max_blocks}) are already free; "
                        f"released twice?"
                    )
            pos, run = 0, [start, n]
            for pos, (s, _len) in enumerate(target):
                if s > start:
                    break
            else:
                pos = len(target)
            target.insert(pos, run)
            # Merge with the neighbours the release just glued together.
            merged = []
            for s, l in target:
                if merged and merged[-1][0] + merged[-1][1] == s:
                    merged[-1][1] += l
                else:
                    merged.append([s, l])
            target[:] = merged
            self._cond.notify_all()

    def cpu_view(
        self, start: int, n: int, block_bytes: Optional[int] = None
    ) -> torch.Tensor:
        """Dense view of the ``n`` blocks of one borrowed run.

        ``block_bytes`` is the borrowing bucket's block size and defaults to the
        slot size (single-shape pools). A smaller block views only its dense
        prefix of the run: the SDK buffers, the kernel view and the state copies
        keep assuming one contiguous piece of memory per block.
        """
        b = self.block_bytes if block_bytes is None else block_bytes
        if not 0 < b <= self.block_bytes:
            raise ValueError(
                f"block size {b} does not fit the pool's {self.block_bytes}-byte slots"
            )
        base = start * self.block_bytes
        return self._cpu[base : base + n * b]

    def slot_base_ptr(self, slot: int) -> int:
        """Byte address of one *held* slot (the eager capture's block buffers).

        A captured block always sits at the start of a slot: the capture
        addresses individual slots, so the save task can point the SDK straight
        at them. (A *transfer* task borrows a run and stages its blocks densely,
        which is why it goes through ``cpu_view`` instead.)
        """
        if not 0 <= slot < self.max_blocks:
            raise ValueError(f"slot {slot} is outside the pool (0..{self.max_blocks})")
        return self._cpu.data_ptr() + slot * self.block_bytes


class MultiResult:
    """Collect the per-block success flags of several async tasks and fire a
    callback once every task has reported. Each result is a list[bool] aligned
    with the manager blocks the task handled (in submission order)."""

    def __init__(self, size: int, callback: Callable[[List[Any]], None]) -> None:
        self._size = size
        self._results: List[Optional[Sequence[Any]]] = [None] * size
        self._lock = threading.Lock()
        self._finished_num = 0
        self._callback = callback

    def submit_result(self, idx: int, result: Sequence[Any]) -> None:
        with self._lock:
            assert self._results[idx] is None
            self._results[idx] = result
            self._finished_num += 1
            if self._finished_num == self._size:
                # Flatten in submission order. Every slot is filled by the
                # count check (submission asserts the slot was None), so the
                # per-part guard only documents the invariant for readers.
                flat = []
                for part in self._results:
                    assert part is not None
                    flat.extend(part)
                self._callback(flat)


@dataclass(frozen=True)
class StagingShape:
    """One transfer bucket's staging geometry: what the arenas key and size by.

    ``per_token_bytes`` is one layer's bytes per token. A state bucket has no
    tokens of its own, so its page bytes are normalized by the block's token
    count -- that is what makes state and attention buckets comparable.
    ``dtype`` is the element type the bucket's rows are read as. Two buckets
    share an arena only when both components match, so an arena always serves
    one row shape.
    """

    per_token_bytes: int
    dtype: Any
    per_block_bytes: int

    @property
    def pool_key(self) -> Tuple[int, Any]:
        return (self.per_token_bytes, self.dtype)


@dataclass(frozen=True)
class StagingPoolPlan:
    """One pinned arena, shared by every bucket of the same staging shape.

    ``slot_bytes`` is the arena's slot -- the largest member block. A member
    stages its ``n`` blocks densely inside the ``n`` slots it borrows, so a
    member with smaller blocks never touches its neighbours' bytes.
    """

    key: Tuple[int, Any]
    slot_bytes: int
    blocks: int
    #: Indices, into the shapes the plan was built from, of the pooled buckets.
    members: Tuple[int, ...]

    @property
    def pinned_bytes(self) -> int:
        return self.slot_bytes * self.blocks


def staging_shape(
    group: TransferGroup, manager_block_size: int, kv_dtype: Any
) -> StagingShape:
    """One bucket's staging shape, from the worker's transfer groups.

    Attention buckets derive the per-token size from their own block size
    (``per_block_bytes`` is built as per-token * manager_block_size * layers,
    see vllm_common.parse_groups; for a compressed bucket that is bytes per row
    divided by the tokens a row spans, see RowGranularMeta) and carry their
    tensors' element type. State
    buckets stage opaque bytes and take ``kv_dtype``: their key then matches an
    attention bucket of the same per-token geometry, which is byte-compatible
    (each member only ever writes its own dense block run) and keeps a hybrid
    instance on a single arena.
    """
    if isinstance(group, AttentionTransferGroup):
        return StagingShape(
            per_token_bytes=group.per_block_bytes
            // (manager_block_size * group.layer_num),
            dtype=group.dtype,
            per_block_bytes=group.per_block_bytes,
        )
    # parse_groups produces only attention/state buckets; ty cannot infer that
    # the complement of AttentionTransferGroup is a StateTransferGroup.
    page_size_bytes = group.page_size_bytes  # ty: ignore[unresolved-attribute]
    return StagingShape(
        per_token_bytes=page_size_bytes // group.block_size,
        dtype=kv_dtype,
        per_block_bytes=group.per_block_bytes,
    )


def group_meta_shape(meta: GroupMeta, kv_dtype: Any) -> StagingShape:
    """The same shape from a GroupMeta (scheduler role, tests, e2e harness).

    Attention and state pages both carry their per-token byte size directly, so
    this needs no manager block size -- and it agrees exactly with
    ``staging_shape``, which divides the worker's per-block size back out.
    An attention bucket carries its own element type (the spec's dtype, which
    the worker's staging view shares), so mixed-dtype instances keep the same
    pool split as the runtime; the instance-wide ``kv_dtype`` is only the
    state buckets' element type (they stage opaque bytes) and the fallback
    for a meta built without one.
    """
    per_token_bytes = (
        meta.page_bytes // meta.block_size
        if isinstance(meta, AttentionGroupMeta)
        else meta.page_size_bytes // meta.block_size  # ty: ignore[unresolved-attribute]
    )
    return StagingShape(
        per_token_bytes=per_token_bytes,
        dtype=getattr(meta, "dtype", None) or kv_dtype
        if isinstance(meta, AttentionGroupMeta)
        else kv_dtype,
        per_block_bytes=meta.per_block_bytes,
    )


def _effective_pool_blocks(
    configured_blocks: int, slot_bytes: int, max_bytes: int
) -> int:
    """Slots in one staging arena under the pinned-RAM ceiling.

    min(configured_blocks, max_bytes // slot_bytes). The byte cap is a hard
    budget: a cap that cannot pay for the configured slot count shrinks the pool
    -- and the task batches riding on it (see
    ``DataTransferManager.save_task_blocks``) -- instead of being breached, and
    a cap smaller than a single slot fails engine start. Pure so the sizing
    contract is unit-pinnable.
    """
    if slot_bytes <= 0:
        raise ValueError("staging slot size must be positive")
    if configured_blocks < 1:
        raise ValueError("staging_pool_blocks must be at least 1")
    by_cap = max_bytes // slot_bytes
    if by_cap < 1:
        raise ValueError(
            f"staging_pool_max_bytes_per_group={max_bytes} is smaller than one "
            f"{slot_bytes}-byte staging slot; raise the cap or shrink the block"
        )
    return min(configured_blocks, by_cap)


def plan_staging_pools(
    shapes: Sequence[StagingShape],
    *,
    configured_blocks: int,
    max_bytes_per_pool: int,
) -> List[StagingPoolPlan]:
    """Group buckets by staging shape and size one pinned arena per shape.

    Arenas follow the shape, not the bucket: every bucket with the same
    per-token geometry and element type shares one. A sliding-window model's
    n+1 vLLM groups (0.31 splits them stride-wise, so their layer counts differ)
    collapse to one arena, and a hybrid instance's state and attention buckets
    join whenever their per-token geometry agrees. Sharing only adds contention
    -- an exhausted pool blocks, exactly as a per-bucket pool did -- while the
    pinned total stops scaling with the group count.
    """
    order: List[Tuple[int, Any]] = []
    slot_bytes: Dict[Tuple[int, Any], int] = {}
    members: Dict[Tuple[int, Any], List[int]] = {}
    for i, shape in enumerate(shapes):
        key = shape.pool_key
        if key not in members:
            order.append(key)
            members[key] = []
            slot_bytes[key] = shape.per_block_bytes
        else:
            slot_bytes[key] = max(slot_bytes[key], shape.per_block_bytes)
        members[key].append(i)
    return [
        StagingPoolPlan(
            key=key,
            slot_bytes=slot_bytes[key],
            blocks=_effective_pool_blocks(
                configured_blocks, slot_bytes[key], max_bytes_per_pool
            ),
            members=tuple(members[key]),
        )
        for key in order
    ]


def total_pinned_bytes(plans: Sequence[StagingPoolPlan]) -> int:
    """Pinned host bytes of one instance's staging arenas.

    The connector's whole pinned footprint, from the same plans the arenas are
    built from -- the e2e/kvperf pinned gates recompute it from the model's
    group metas (``group_meta_shape``) and compare it with the logged total.
    """
    return sum(plan.pinned_bytes for plan in plans)


def group_meta_pinned_bytes(
    group_metas: Sequence[GroupMeta],
    kv_dtype: Any,
    *,
    configured_blocks: int,
    max_bytes_per_pool: int,
) -> int:
    """The pinned total of a model whose buckets are known as GroupMetas.

    The number the worker's ``DataTransferManager`` logs as ``total pinned``
    for the same configuration, for models whose buckets share one element
    type (every model the connector accepts today); the single ``kv_dtype``
    argument cannot express a per-bucket dtype split. It derives the same
    shapes from the group structure (see ``staging_shape``), so an e2e or
    kvperf gate can assert a budget -- on whichever role view it has --
    instead of re-deriving the sizing.
    """
    return total_pinned_bytes(
        plan_staging_pools(
            [group_meta_shape(meta, kv_dtype) for meta in group_metas],
            configured_blocks=configured_blocks,
            max_bytes_per_pool=max_bytes_per_pool,
        )
    )


class DataTransferManager:
    def __init__(
        self,
        kvcache_info: KVCacheInfo,
        manager_block_size: int,
        transfer_client: Any,
        coordinator_client: TpCoordinatorClient,
        extra_config: TairKvCacheConnectorExtraConfig,
    ) -> None:
        self._manager_block_size = manager_block_size
        self._transfer_client = transfer_client
        self._coordinator_client = coordinator_client
        self._extra_config = extra_config
        self._device = kvcache_info.device
        self._device_mod = _get_device_module(self._device)
        self._save_stream = self._device_mod.Stream()
        self._load_stream = self._device_mod.Stream()

        # One pinned arena per *staging shape*, shared by every bucket that has
        # it (see plan_staging_pools). The arena is pinned host memory only --
        # the kernel reaches it over PCIe -- so the connector's device-memory
        # footprint is zero, while the pinned total is bounded by the byte cap
        # per shape instead of growing with the number of vLLM groups.
        shapes = [
            staging_shape(g, manager_block_size, kvcache_info.dtype)
            for g in kvcache_info.groups
        ]
        plans = plan_staging_pools(
            shapes,
            configured_blocks=extra_config.staging_pool_blocks,
            max_bytes_per_pool=extra_config.staging_pool_max_bytes_per_group,
        )
        self._pools: Dict[str, _StagingPool] = {}
        save_batch = extra_config.block_per_save_task
        load_batch = extra_config.block_per_load_task
        # The reserve is the largest batch one transfer task can ask for (see
        # save_task_blocks): with it carved out, a task can always start -- and
        # thereby release the captures its own session holds -- so the
        # capture's bounded wait is a deadlock-free degradation rather than a
        # race for the last run.
        #
        # It never takes the whole arena, though: a plan whose block count is
        # <= the configured batch (a 128-block arena with the default 128-block
        # batches, or a byte-capped shape) would otherwise leave the capture
        # side with zero slots and silently disable windowed caching for an
        # otherwise valid configuration. The arena is split in half instead
        # (capped by the task batch): the eager capture asks for the whole
        # span of a step's completed blocks in *one* contiguous run and cannot
        # split it, so one leftover slot would still time out every step that
        # completes two or more blocks -- a 62-slot arena leaves 31 for
        # captures, which covers a 16-block chunked-prefill step. The task
        # batch is capped to the reserve, so a task can always start. Only in a
        # single-slot arena, where the two cannot coexist, does the slot stay
        # with the tasks (captures off, logged).
        for plan in plans:
            task_batch = min(max(save_batch, load_batch), plan.blocks)
            reserved = task_batch if plan.blocks == 1 else min(task_batch, plan.blocks // 2)
            pool = _StagingPool(
                self._device,
                plan.slot_bytes,
                plan.blocks,
                reserved_blocks=reserved,
            )
            names = [kvcache_info.groups[i].spec_name for i in plan.members]
            for name in names:
                self._pools[name] = pool
            logger.info(
                "staging pool per_token=%dB dtype=%s: %d blocks x %d bytes "
                "(pinned %.1f MiB) shared by %d group(s): %s",
                plan.key[0],
                plan.key[1],
                plan.blocks,
                plan.slot_bytes,
                plan.pinned_bytes / 2**20,
                len(names),
                ",".join(names),
            )
            if pool.reserved_blocks:
                logger.info(
                    "staging pool per_token=%dB dtype=%s: %d slots for captures "
                    "+ %d reserved for transfer tasks%s",
                    plan.key[0],
                    plan.key[1],
                    plan.blocks - pool.reserved_blocks,
                    pool.reserved_blocks,
                    " -- captures are disabled (the reserve takes the whole pool)"
                    if pool.reserved_blocks == plan.blocks
                    else "",
                )
            fitting = (
                min(save_batch, max(1, pool.reserved_blocks)),
                min(load_batch, max(1, pool.reserved_blocks)),
            )
            if fitting != (save_batch, load_batch):
                logger.info(
                    "staging pool per_token=%dB dtype=%s: task batches shrink to "
                    "save %d / load %d blocks (configured %d / %d) to fit the "
                    "pool's transfer reserve",
                    plan.key[0],
                    plan.key[1],
                    fitting[0],
                    fitting[1],
                    save_batch,
                    load_batch,
                )
        pinned_total = total_pinned_bytes(plans)
        logger.info(
            "total pinned = %.3f GiB across %d %s / %d %s (%d bytes)",
            pinned_total / 2**30,
            len(plans),
            "shape" if len(plans) == 1 else "shapes",
            len(kvcache_info.groups),
            "group" if len(kvcache_info.groups) == 1 else "groups",
            pinned_total,
        )

        def _init_worker() -> None:
            self._device_mod.set_device(self._device)

        self._io_executor = ThreadPoolExecutor(
            max_workers=32, thread_name_prefix="kvcm_io_", initializer=_init_worker
        )

    def submit_task(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Future:
        return self._io_executor.submit(func, *args, **kwargs)

    def save_task_blocks(self, group: TransferGroup) -> int:
        """Blocks per save task for this bucket: the configured batch, shrunk to
        the pool's transfer reserve. A task stages its whole batch as one
        contiguous run, so it can never outgrow the pool -- and a batch that fit
        only the *shared* part could not start while captures hold it, which is
        exactly the wait the reserve exists to break (see the pool plan above:
        the granularity yields, the pinned budget does not)."""
        return min(
            self._extra_config.block_per_save_task,
            max(1, self._pools[group.spec_name].reserved_blocks),
        )

    def load_task_blocks(self, group: TransferGroup) -> int:
        """Load-side twin of ``save_task_blocks``."""
        return min(
            self._extra_config.block_per_load_task,
            max(1, self._pools[group.spec_name].reserved_blocks),
        )

    def pool_of(self, group: TransferGroup) -> _StagingPool:
        """The arena a bucket stages in.

        Public because the eager capture has to address *individual* slots:
        it holds them for as long as the write session is in flight, which is
        not the borrow-and-release pattern the task APIs express.
        """
        return self._pools[group.spec_name]

    def capture_blocks(
        self,
        group: AttentionTransferGroup,
        block_slots: List[List[int]],
        timeout: float,
    ) -> Optional[int]:
        """Copy blocks straight out of HBM into the arena, synchronously.

        Called from the worker's wait_for_save hook, i.e. after the step's
        forward and before vLLM's next schedule step frees the blocks: the
        gather is ordered behind the forward on the engine's current stream and
        synchronized before returning, so the arena holds the bytes -- not a
        promise of them -- by the time the caller goes on. That is the whole
        point of the eager capture: a windowed block is only readable in the
        pass that produced it.

        Returns the arena run the caller now owns (it keeps the slots until the
        write lands), or None when the arena has no free run within ``timeout``
        -- the caller must then withhold the blocks rather than publish a
        location whose bytes were never copied.
        """
        pool = self._pools[group.spec_name]
        start = pool.acquire(len(block_slots), timeout=timeout, keep_reserved=True)
        if start is None:
            return None
        try:
            if group.per_block_bytes == pool.block_bytes:
                # The member's block fills the arena's slot, so the whole run is
                # one dense gather and each block ends up slot-aligned.
                cpu_buffer = pool.cpu_view(start, len(block_slots))
                view = self._attn_staging_view(cpu_buffer, group, len(block_slots))
                batch_gather_scatter_helper.batch_gather_kv_caches(
                    group.kvcache_ptr_tensor_gpu,
                    view,
                    block_slots,
                    list(range(len(block_slots))),
                    self._attn_rows_per_block(group),
                    group.per_token_dim,
                    block_stride=group.block_stride,
                    local_block_size=group.kernel_block_size,
                )
            else:
                # A smaller member (shared arena, larger slot): stage each block
                # at the start of *its* slot, one gather per block, so the save
                # task can address captured slots individually.
                for offset, slots in enumerate(block_slots):
                    cpu_buffer = pool.cpu_view(start + offset, 1, group.per_block_bytes)
                    view = self._attn_staging_view(cpu_buffer, group, 1)
                    batch_gather_scatter_helper.batch_gather_kv_caches(
                        group.kvcache_ptr_tensor_gpu,
                        view,
                        [slots],
                        [0],
                        self._attn_rows_per_block(group),
                        group.per_token_dim,
                        block_stride=group.block_stride,
                        local_block_size=group.kernel_block_size,
                    )
            self._device_mod.current_stream().synchronize()
        except BaseException:
            pool.release(start, len(block_slots))
            raise
        return start

    def _attn_rows_per_block(self, group: AttentionTransferGroup) -> int:
        """Staging rows one manager block holds: one per token for a
        token-granular bucket, one per stored state for a compressed one (the
        kernel indexes rows, see AttentionTransferGroup.tokens_per_row)."""
        return self._manager_block_size // group.tokens_per_row

    def _attn_staging_view(
        self, cpu_buffer: torch.Tensor, group: AttentionTransferGroup, batch: int
    ) -> torch.Tensor:
        """Pinned host staging view for one attention bucket: the *bucket's*
        element type (instances may mix element sizes, e.g. a V3.2 bf16 main
        spec next to a uint8 indexer) and one staging row per kernel row.
        """
        return cpu_buffer.view(group.dtype).view(
            batch,
            group.num_kv_ptrs,
            self._attn_rows_per_block(group),
            group.per_token_dim,
        )

    # ------------------------------------------------------------------ #
    # BlockBuffer helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _block_buffer(base_ptr: int, per_block_bytes: int) -> Any:
        buf = kvcm_py_client.BlockBuffer()
        iov = kvcm_py_client.Iov()
        iov.type = kvcm_py_client.MemoryType.CPU
        iov.base = base_ptr
        iov.size = per_block_bytes
        iov.ignore = False
        buf.iovs = [iov]
        return buf

    @classmethod
    def _make_block_buffers(
        cls, base_ptr: int, per_block_bytes: int, count: int
    ) -> List[kvcm_py_client.BlockBuffer]:
        return [
            cls._block_buffer(base_ptr + i * per_block_bytes, per_block_bytes)
            for i in range(count)
        ]

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
        staged_slots: Optional[List[Optional[int]]] = None,
        null_rows: Optional[List[Optional[NullBlockRow]]] = None,
        ready_event: Any = None,
    ) -> None:
        """Move one group's manager blocks out of the engine and save them.

        block_token_indices: attention -> list[list[int]] flat token slots per block.
        block_ids:           state     -> list[int] block id per manager block.
        remote_uris:         positionally aligned with the manager blocks; None
                             where the manager allocated no location for this
                             group's spec (see ``vllm_common.build_spec_groups``)
                             or where the block's windowed capture was withheld.
        staged_slots:        windowed attention groups -> the staging-arena slot
                             the eager capture already copied this block's bytes
                             into, None where the block had no windowed data to
                             publish. Such a group is never gathered from HBM
                             here: the bytes left HBM in the pass that produced
                             them (vLLM frees windowed blocks while the request
                             is alive, so a late gather would read the null page
                             or another request's data).

        ``null_rows`` belongs to the load side and is accepted only to keep
        the two task signatures one shape.

        A block is reported successful only when its data was actually written.
        Where a state group has no state (vLLM's null block in mamba "align"
        mode) the manager was already told so at start_write_cache time -- the
        block simply carries no spec for this group -- so nothing is written and
        nothing is claimed: the block is *excluded* from this group's verdict
        rather than reported as a success. A windowed block whose capture was
        withheld is the mirror image: this group has nothing to write although
        the location was announced, so the block is *failed* -- publishing it
        would advertise a spec whose bytes were never stored.
        """
        n = len(remote_uris)
        # Single exit: whatever happens below, the task reports, otherwise the
        # MultiResult callback never fires (submit_task drops the future, so an
        # escaping exception is silently swallowed) and the save session hangs.
        ok_mask = [False] * n
        try:
            # Three dispositions per block: abstain (this group holds no data
            # for it by design), transfer, or fail outright.
            if staged_slots is not None:
                skipped, failed = self._staged_dispositions(
                    group, remote_uris, staged_slots, n
                )
            else:
                skipped, failed = self._save_dispositions(
                    group, remote_uris, block_ids, n
                )
            valid = [i for i in range(n) if i not in skipped and i not in failed]
            ok_mask = [None if i in skipped else False for i in range(n)]
            if valid:
                if staged_slots is not None:
                    self._save_staged_blocks(
                        group, remote_uris, staged_slots, valid, ok_mask
                    )
                else:
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

    def _staged_dispositions(
        self,
        group: TransferGroup,
        remote_uris: List[Optional[str]],
        staged_slots: List[Optional[int]],
        n: int,
    ) -> Tuple[Set[int], Set[int]]:
        """Split a windowed group's task into (abstained, failed).

        The location of each block says whether the announcement claimed this
        group's spec for it, and the capture says whether its bytes exist; the
        four combinations are three outcomes:

        * claimed + captured -> transfer (the normal case);
        * not claimed + no bytes -> **abstain**: the block's window was already
          gone when it was announced, so the manager was told the group holds
          nothing for it (no spec in the location) and there is nothing to
          write -- not writing is the whole point, since the bytes at those
          slots are the engine's shared zero page;
        * claimed + no bytes -> **fail**: the bytes never made it into the arena
          (the capture was withheld, or no capture ever ran for this block).
          Publishing it would advertise a spec with no data behind it, so the
          whole block is withdrawn instead;
        * not claimed + captured -> **fail** too: there is no location to write
          those bytes to, and the announcement and the capture disagreeing is a
          contract violation, not something to paper over.
        """
        skipped: Set[int] = set()
        failed: Set[int] = set()
        for i in range(n):
            claimed = remote_uris[i] is not None
            captured = staged_slots[i] is not None
            if captured and claimed:
                continue
            if not captured and not claimed:
                skipped.add(i)
                continue
            failed.add(i)
            logger.warning(
                "save group %s: block %d %s; failing it instead of publishing a hole",
                group.spec_name,
                i,
                "was announced without captured bytes (withheld capture?)"
                if claimed
                else "has captured bytes but no announced location",
            )
        if skipped:
            logger.debug(
                "save group %s: %d/%d windowed blocks had no window data "
                "(nothing published for them)",
                group.spec_name,
                len(skipped),
                n,
            )
        return skipped, failed

    def _save_staged_blocks(
        self,
        group: TransferGroup,
        remote_uris: List[Optional[str]],
        staged_slots: List[Optional[int]],
        valid: List[int],
        ok_mask: List[Optional[bool]],
    ) -> None:
        """Write the bytes the eager capture already gathered into the arena.

        The captured slots are already owned by this request (held since the
        capture step), so nothing is acquired here and nothing is copied: each
        block's SDK buffer points straight at its captured slot. The capture's
        gather was synchronized before it returned, so the bytes are visible to
        the SDK thread -- no stream dance is needed either (the save side's
        ``ready_event`` exists for the late gather, which this path has none
        of).

        A captured slot goes back to the arena when the write *lands*, and only
        then -- this task never returns a slot it did not write, not even when
        the SDK raises. It cannot: save_task's single exit always reports
        through the MultiResult, so the session's result callback always fires,
        and the worker owns the failed path. It puts the failed block's entry
        back into its staged map (see _retain_failed_windowed) with the slot
        still held, so the next session publishes it; if that never happens,
        retirement releases it. Keeping the bytes is the point: the block can
        be announced again (the save watermark moves backwards when preemption
        shrinks an allocation, and a retry session announces the whole prefix),
        and by then the HBM window that held its KV has moved on -- these
        captured bytes are the only copy the connector will ever have.
        """
        pool = self._pools[group.spec_name]
        uris = [remote_uris[i] for i in valid]
        assert all(uri is not None for uri in uris), (
            f"group {group.spec_name}: staged save batch contains a block "
            f"without a location; _staged_dispositions must have failed it"
        )
        buffers = [
            self._block_buffer(
                pool.slot_base_ptr(staged_slots[i]),  # ty: ignore[invalid-argument-type]
                group.per_block_bytes,
            )
            for i in valid
        ]
        ok = False
        try:
            result = self._transfer_client.SaveKvCaches(uris, buffers)
            ok = result[0] == kvcm_py_client.ClientErrorCode.ER_OK
        finally:
            # Only a landed write returns the slot. A failed one keeps it,
            # because the worker restores the block's entry to its staged map
            # with this very slot: returning it here would hand the arena a
            # slot the worker still points at, and the retry would publish
            # whatever the next capture wrote there. An SDK exception is not a
            # special case -- save_task reports through its single exit either
            # way, so the callback fires and the worker owns the entry too.
            if ok:
                for i in valid:
                    pool.release(staged_slots[i], 1)  # ty: ignore[invalid-argument-type]
        if not ok:
            logger.warning(
                "save task failed group=%s uris=%d result=%s; the block is not "
                "published and its captured bytes are kept for a retry "
                "(the slot stays held until the request retires)",
                group.spec_name,
                len(uris),
                result,
            )
        for i in valid:
            ok_mask[i] = ok

    def _save_dispositions(
        self,
        group: TransferGroup,
        remote_uris: List[Optional[str]],
        block_ids: Optional[List[int]],
        n: int,
    ) -> Tuple[Set[int], Set[int]]:
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
        if isinstance(group, AttentionTransferGroup):
            failed = {i for i in range(n) if remote_uris[i] is None}
            if failed:
                logger.warning(
                    "save group %s: %d/%d attention blocks have no "
                    "location, failing them",
                    group.spec_name,
                    len(failed),
                    n,
                )
            return set(), failed
        skipped, failed = set(), set()
        for i in range(n):
            is_null = block_ids[i] == 0  # ty: ignore[not-subscriptable]
            has_uri = remote_uris[i] is not None
            if is_null and not has_uri:
                skipped.add(i)
            elif is_null:
                failed.add(i)
                logger.warning(
                    "save group %s: block %d has a location but no "
                    "materialized state; failing it instead of "
                    "publishing bytes the model never produced",
                    group.spec_name,
                    i,
                )
            elif not has_uri:
                failed.add(i)
                logger.warning(
                    "save group %s: block %d has a materialized "
                    "state but no location to write it to; failing it",
                    group.spec_name,
                    i,
                )
        if skipped:
            logger.debug(
                "save group %s: %d/%d blocks carry no state (not published)",
                group.spec_name,
                len(skipped),
                n,
            )
        return skipped, failed

    @staticmethod
    def _load_skipped_blocks(
        group: TransferGroup,
        remote_uris: List[Optional[str]],
        block_ids: Optional[List[int]],
        n: int,
    ) -> Set[int]:
        """Blocks this group has nothing to load into.

        Asymmetric with the save side on purpose: on load, a null target means
        vLLM does not *need* this group's data for that block (in mamba "align"
        mode only the block ending the reused prefix needs its state, and a
        windowed group's targets outside the window are the null block), so the
        block is skipped whatever the manager published. The reverse -- a real
        target with nothing published -- is a genuine failure: the request would
        run on an unwritten state.
        """
        if isinstance(group, AttentionTransferGroup):
            missing = sum(uri is None for uri in remote_uris)
            if missing:
                if group.retention is None:
                    logger.warning(
                        "load group %s: %d/%d attention blocks have no "
                        "location, failing them",
                        group.spec_name,
                        missing,
                        n,
                    )
                else:
                    # Windowed group: a block without a published location is
                    # one whose window was already gone when it was announced,
                    # so its targets must be null slots (skipped). A real target
                    # without a location is a clamp violation and is failed
                    # right below.
                    logger.debug(
                        "load group %s: %d/%d windowed blocks have no published "
                        "location (window already gone at save time)",
                        group.spec_name,
                        missing,
                        n,
                    )
            return set()
        # State tasks always carry block ids (see _iter_task_chunks).
        skipped = {i for i in range(n) if block_ids[i] == 0}  # ty: ignore[not-subscriptable]
        if skipped:
            logger.debug(
                "load group %s: %d/%d blocks need no state",
                group.spec_name,
                len(skipped),
                n,
            )
        return skipped

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
            # The arena's slots serve the largest member block, so this
            # bucket's batch is a dense prefix of the run it borrowed.
            cpu_buffer = pool.cpu_view(start, len(valid), group.per_block_bytes)
            with self._device_mod.stream(self._save_stream):
                # Gather straight into the pinned host slot: the kernel
                # (attention) and copy_ (state) write host pinned memory
                # directly over PCIe; no device-side staging copy exists.
                if isinstance(group, AttentionTransferGroup):
                    view = self._attn_staging_view(cpu_buffer, group, len(valid))
                    batch_gather_scatter_helper.batch_gather_kv_caches(
                        group.kvcache_ptr_tensor_gpu,
                        view,
                        # Attention tasks always carry token indices.
                        [block_token_indices[i] for i in valid],  # ty: ignore[not-subscriptable]
                        list(range(len(valid))),
                        self._attn_rows_per_block(group),
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

    def create_save_done_callback(
        self,
        req_id: str,
        tp_rank: int,
        write_session_id: str,
        num_blocks: int,
    ) -> Callable[[List[Optional[bool]]], None]:
        """block success = AND across all groups that had data for the block.

        Task results are ordered group0[blocks], group1[blocks], ... so a
        block's verdict is the stride-AND ``flat[i % num_blocks]``. ``None``
        entries mean "this group holds no data for this block by design" (see
        save_task) and are skipped: they neither pass nor fail the block. A
        block whose every group is None was never written at all and must not
        be published.
        """

        def cb(flat: List[Optional[bool]]) -> None:
            is_success = [None] * num_blocks
            for i, ok in enumerate(flat):
                if ok is None:
                    continue
                b = i % num_blocks
                is_success[b] = ok if is_success[b] is None else (is_success[b] and ok)
            # No group wrote anything for the block -> nothing to publish.
            is_success = [bool(ok) for ok in is_success]
            msg = CoordinateMessage(
                time.time(),
                SendBlockFinishedEvent(
                    request_id=req_id,
                    tp_rank=tp_rank,
                    write_session_id=write_session_id,
                    is_success_list=is_success,
                ),
            )
            self._coordinator_client.send(CoordinateMsgSerializer.dumps(msg))

        return cb

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
        staged_slots: Optional[List[Optional[int]]] = None,
        null_rows: Optional[List[Optional[NullBlockRow]]] = None,
    ) -> None:
        """Load one group's manager blocks from storage into HBM.

        Mirror of ``save_task``: ``remote_uris`` is positionally aligned with
        the manager blocks and holds None where the published block carries no
        data for this group's spec. A state group's block is skipped only when
        vLLM's target is the null block *and* nothing was published -- i.e. the
        state is neither needed nor available. Everything else must transfer.

        ``null_rows`` (windowed groups, see NullBlockRow) adds the load-side
        zero-page rule: rows whose target is vLLM's null block are never
        written, and a manager block whose every row is null is skipped --
        vLLM does not allocate windowed slots outside the window, so there is
        nothing to restore there. ``staged_slots`` belongs to the save side and
        is accepted only to keep the task signatures uniform.
        """
        del staged_slots
        n = len(remote_uris)
        # Single exit, as in save_task: an escaping exception would silently
        # swallow the MultiResult callback, which under the connector's
        # synchronous-load contract leaves vLLM believing KV it never received.
        ok_mask = [False] * n
        try:
            skipped = self._load_skipped_blocks(group, remote_uris, block_ids, n)
            if null_rows is not None:
                all_null = {
                    i
                    for i in range(n)
                    if null_rows[i] is not None and null_rows[i].source_row is None  # ty: ignore[unresolved-attribute]
                }
                if all_null:
                    logger.debug(
                        "load group %s: %d/%d blocks target only null slots "
                        "(nothing to restore)",
                        group.spec_name,
                        len(all_null),
                        n,
                    )
                skipped |= all_null
            # A block we must restore but nothing was published for cannot be
            # loaded; fail it without letting it shift the staging batch.
            failed = {
                i for i in range(n) if i not in skipped and remote_uris[i] is None
            }
            valid = [i for i in range(n) if i not in skipped and i not in failed]
            ok_mask = [None if i in skipped else False for i in range(n)]
            if valid:
                ok = self._load_valid_blocks(
                    group, remote_uris, block_token_indices, block_ids, valid, null_rows
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
        null_rows: Optional[List[Optional[NullBlockRow]]] = None,
    ) -> bool:
        pool = self._pools[group.spec_name]
        start = pool.acquire(len(valid))
        try:
            # Dense prefix of the borrowed run, as in the save path.
            cpu_buffer = pool.cpu_view(start, len(valid), group.per_block_bytes)
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
                        view = self._attn_staging_view(cpu_buffer, group, len(valid))
                        if null_rows is not None:
                            # Rows whose target is vLLM's null block carry no
                            # data of their own (those tokens are outside the
                            # window): their token slots already point at a row
                            # that does get loaded, so this copies that row's
                            # bytes onto them -- one row per null row, for every
                            # pointer (a pointer is a layer in the packed
                            # layouts, so the copy must be broadcast across the
                            # row axis, not across pointers). The scatter then
                            # writes the same correct value twice and the shared
                            # null block stays untouched.
                            for out_i, i in enumerate(valid):
                                alias = null_rows[i]
                                if alias is not None:
                                    assert alias.source_row is not None, (
                                        f"group {group.spec_name}: block {i} has "
                                        f"no real row to alias its null rows onto; "
                                        f"_load_valid_blocks must not see it"
                                    )
                                    source = view[
                                        out_i, :, alias.source_row, :
                                    ].unsqueeze(1)
                                    view[out_i, :, alias.rows, :] = source.expand(
                                        -1, len(alias.rows), -1
                                    )
                        batch_gather_scatter_helper.batch_scatter_kv_caches(
                            group.kvcache_ptr_tensor_gpu,
                            view,
                            # Attention tasks always carry token indices
                            # (see _iter_task_chunks).
                            [block_token_indices[i] for i in valid],  # ty: ignore[not-subscriptable]
                            list(range(len(valid))),
                            self._attn_rows_per_block(group),
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

    def create_load_done_callback(
        self,
        req_id: str,
        tp_rank: int,
        epoch: int,
        block_ids: List[int],
        num_blocks: int,
        report_failures: bool = True,
    ) -> Callable[[List[Optional[bool]]], None]:
        """A manager block is loaded only if every group that had data for it
        succeeded -- and it is *invalid* only when one of them failed.

        ``None`` entries mean "this group has no data for this block by design"
        (mamba "align" interior blocks, and a windowed group's tokens outside
        the window, see load_task) and are skipped; a block whose groups all
        abstained asked for nothing and must not be reported as a lost block --
        vLLM reacts to an invalid block id by truncating num_computed_tokens to
        where that block first appears (block 0 would throw the whole external
        hit away, and kv_load_failure_policy="fail" would fail the request).

        block_ids is the block table used to report vLLM-visible invalid block
        ids. vLLM's invalid-block recovery unpacks a single-group block table
        (https://github.com/vllm-project/vllm/blob/releases/v0.26.0/vllm/v1/core/sched/scheduler.py#L2693,
        "TODO (davidb): add support for hybrid memory allocator"); for a
        multi-group (hybrid) model the unpack raises ValueError and crashes
        the scheduler, so hybrid connectors pass report_failures=False: the
        failure is logged, vLLM keeps the partially loaded KV, and the request
        may produce corrupt output -- the contained alternative to an
        engine-wide crash. See start_load_kv for the full trade-off."""

        def cb(flat: List[Optional[bool]]) -> None:
            merged = [None] * num_blocks
            for i, ok in enumerate(flat):
                if ok is None:
                    continue
                b = i % num_blocks
                merged[b] = ok if merged[b] is None else (merged[b] and ok)
            # Only an explicit failure makes a block invalid: all-None means
            # no group needed it (a null target in every group), which is a
            # skip, not a hole in the request's KV.
            broken = [ok for ok in merged if ok is False]
            failed = []
            if report_failures:
                failed = [
                    block_ids[i]
                    for i in range(min(num_blocks, len(block_ids)))
                    if merged[i] is False
                ]
            elif broken:
                logger.warning(
                    "load failed for %d/%d blocks of req %s (hybrid: "
                    "not reporting invalid block ids)",
                    len(broken),
                    num_blocks,
                    req_id,
                )
            msg = CoordinateMessage(
                time.time(),
                LoadBlockFinishedEvent(
                    request_id=req_id,
                    tp_rank=tp_rank,
                    epoch=epoch,
                    failed_block_idxs=failed,
                ),
            )
            self._coordinator_client.send(CoordinateMsgSerializer.dumps(msg))

        return cb
