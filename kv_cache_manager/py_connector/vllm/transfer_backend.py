"""The data-plane seam: what the connector needs from a KV transfer backend.

``ConnectorWorker`` owns the protocol half of the connector (block
translation, save sessions, the coordinator's per-rank verdicts); every
byte movement and every KVCM SDK call lives behind :class:`TransferBackend`.
The default backend is the GPU one -- ``data_transfer.DataTransferManager``,
which gathers/scatters through Triton kernels into pinned host memory. An
engine whose device has neither Triton nor PCIe-pinned host memory reuses
the whole connector by naming its own backend module in
``extra_config.transfer_backend_module_path``.

The six methods below are the frozen contract: they are exactly what
``ConnectorWorker`` calls, and ``DataTransferManager`` implements them
verbatim -- the four byte-moving ones itself, the two done callbacks through
:class:`BlockVerdicts`.

Because the worker needs them whichever backend is in use, this module also
hosts the pieces of the data plane that are backend-agnostic: :class:`MultiResult`
(per-task result aggregation), :func:`get_device_module`, and
:class:`BlockVerdicts` (what a per-block verdict means, which a backend must not
re-invent per device). Nothing here imports Triton -- that stays with the
built-in GPU backend, which is why ``triton`` in ``requirements.txt`` is a
dependency of the default data plane rather than of the connector.
"""

import importlib
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Sequence, Set, Tuple

import torch

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.common.tp_coordinator import (
    CoordinateMessage,
    CoordinateMsgSerializer,
    LoadBlockFinishedEvent,
    SendBlockFinishedEvent,
    TpCoordinatorClient,
)
from kv_cache_manager.py_connector.vllm.transfer_types import (
    AttentionTransferGroup,
    TransferGroup,
)

if TYPE_CHECKING:
    from kv_cache_manager.py_connector.vllm.config import (
        TairKvCacheConnectorExtraConfig,
    )
    from kv_cache_manager.py_connector.vllm.transfer_types import KVCacheInfo


def get_device_module(device: Optional[torch.device] = None) -> Any:
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


class TransferBackend(Protocol):
    """The swap in/out hop: host buffer <-> device, and host buffer <-> KVCM.

    ``save_task``/``load_task`` run on the backend's own IO threads (they are
    handed to :meth:`submit`); each must report exactly once through
    ``multi_result`` -- an escaping exception would swallow the callback and
    hang the save session or, worse, tell the engine KV arrived that did not.
    A ``None`` verdict means "this group holds no data for that block by
    design" and neither passes nor fails the block.
    """

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Run a transfer task asynchronously and return its future."""
        ...

    def save_task(
        self,
        multi_result: MultiResult,
        task_idx: int,
        group: "TransferGroup",
        remote_uris: List[Optional[str]],
        block_token_indices: Optional[List[List[int]]],
        block_ids: Optional[List[int]],
        ready_event: Any,
    ) -> None:
        """Write one group's chunk of manager blocks to storage."""
        ...

    def load_task(
        self,
        multi_result: MultiResult,
        task_idx: int,
        group: "TransferGroup",
        remote_uris: List[Optional[str]],
        block_token_indices: Optional[List[List[int]]],
        block_ids: Optional[List[int]],
    ) -> None:
        """Read one group's chunk of manager blocks back into the engine."""
        ...

    def create_save_done_callback(
        self,
        req_id: str,
        tp_rank: int,
        write_session_id: str,
        num_blocks: int,
    ) -> Callable[[List[Optional[bool]]], None]:
        """Build the callback receiving all of a save request's task verdicts;
        it reports the block mask to the coordinator.

        Inherit :class:`BlockVerdicts` for this and for the load side: the
        verdict rules are the manager's protocol, not the backend's, and a
        second copy drifts."""
        ...

    def create_load_done_callback(
        self,
        req_id: str,
        tp_rank: int,
        epoch: int,
        block_ids: List[int],
        num_blocks: int,
        report_failures: bool = True,
    ) -> Callable[[List[Optional[bool]]], None]:
        """Build the callback receiving all of a load request's task verdicts;
        it reports finished ranks and, where vLLM can consume them, the
        invalid block ids. See the save side: inherit :class:`BlockVerdicts`."""
        ...

    def shutdown(self) -> None:
        """Release threads, pools and SDK handles."""
        ...


class BlockVerdicts:
    """A mixin, not a data holder: the block-verdict half of a backend.

    It answers which blocks a group abstains on, and how a task's flattened
    verdicts become the coordinator's per-block mask.

    None of this depends on where the bytes live, so it lives here rather than
    in each backend: a backend that moves bytes differently (Triton gathers on
    the GPU, page DMA on another device) must still agree with the manager
    about what "this block was written" means, and two copies of that agreement
    drift.

    A ``None`` verdict means "this group holds no data for that block by
    design" (a recurrent state vLLM never materialized); it neither passes nor
    fails the block, and a block every group abstained on is never published.
    """

    def __init__(self, coordinator_client: "TpCoordinatorClient") -> None:
        self._coordinator_client = coordinator_client

    # ------------------------------------------------------------------ #
    # Done callbacks
    # ------------------------------------------------------------------ #
    def create_save_done_callback(
        self,
        req_id: str,
        tp_rank: int,
        write_session_id: str,
        num_blocks: int,
    ) -> Callable[[List[Optional[bool]]], None]:
        """Block success = AND across every group that had data for the block.

        Task results are ordered group0[blocks], group1[blocks], ..., so a
        block's verdict is the stride-AND ``flat[i % num_blocks]``.
        """
        self._prepare_done_callback(req_id, num_blocks)

        def cb(flat: List[Optional[bool]]) -> None:
            mask = _stride_and(flat, num_blocks)
            # No group wrote anything for the block -> nothing to publish.
            self._send(
                SendBlockFinishedEvent(
                    request_id=req_id,
                    tp_rank=tp_rank,
                    write_session_id=write_session_id,
                    is_success_list=[bool(ok) for ok in mask],
                )
            )

        return cb

    def create_load_done_callback(
        self,
        req_id: str,
        tp_rank: int,
        epoch: int,
        block_ids: List[int],
        num_blocks: int,
        report_failures: bool = True,
    ) -> Callable[[List[Optional[bool]]], None]:
        """A block was restored only if every group that had data for it
        succeeded; ``block_ids`` turns the failed blocks into the ids vLLM can
        act on.

        vLLM's invalid-block recovery unpacks a single-group block table
        (https://github.com/vllm-project/vllm/blob/releases/v0.26.0/vllm/v1/core/sched/scheduler.py#L2693,
        "TODO (davidb): add support for hybrid memory allocator"), so a
        multi-group (hybrid) connector passes ``report_failures=False``: the
        failure is logged and the request keeps its partially loaded KV -- the
        contained alternative to an engine-wide crash. See
        ``ConnectorWorker.start_load_kv`` for the full trade-off.
        """
        self._prepare_done_callback(req_id, num_blocks)

        def cb(flat: List[Optional[bool]]) -> None:
            mask = _stride_and(flat, num_blocks)
            # A block no group loaded anything for was not restored, so it
            # counts as failed: a load either restores the block or it does not.
            restored = [bool(ok) for ok in mask]
            failed: List[int] = []
            if not all(restored):
                if report_failures:
                    failed = [
                        block_ids[i]
                        for i in range(min(num_blocks, len(block_ids)))
                        if not restored[i]
                    ]
                else:
                    logger.warning(
                        "load failed for %d/%d blocks of req %s (multi-group: "
                        "not reporting invalid block ids)",
                        restored.count(False),
                        num_blocks,
                        req_id,
                    )
            self._send(
                LoadBlockFinishedEvent(
                    request_id=req_id,
                    tp_rank=tp_rank,
                    epoch=epoch,
                    failed_block_idxs=failed,
                )
            )

        return cb

    def _prepare_done_callback(self, req_id: str, num_blocks: int) -> None:
        """Hook for backends that mirror a request's task budget; no-op here.

        The verdict mask itself needs nothing but the flattened results, so a
        backend with per-request bookkeeping (diagnostics, telemetry) can tag
        the tasks the caller is about to submit without touching the mask. Both
        directions call it, so an implementation cannot tell them apart -- no
        caller has needed to.
        """

    # ------------------------------------------------------------------ #
    # Dispositions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _save_dispositions(
        group: "TransferGroup",
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
        group: "TransferGroup",
        remote_uris: List[Optional[str]],
        block_ids: Optional[List[int]],
        n: int,
    ) -> Set[int]:
        """Blocks this group has nothing to load into.

        Asymmetric with the save side on purpose: on load, a null target means
        vLLM does not *need* this group's data for that block (in mamba "align"
        mode only the block ending the reused prefix needs its state), so the
        block is skipped whatever the manager published. The reverse -- a real
        target with nothing published -- is a genuine failure: the request would
        run on an unwritten state.
        """
        if isinstance(group, AttentionTransferGroup):
            missing = sum(uri is None for uri in remote_uris)
            if missing:
                logger.warning(
                    "load group %s: %d/%d attention blocks have no "
                    "location, failing them",
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

    # ------------------------------------------------------------------ #
    # Plumbing
    # ------------------------------------------------------------------ #
    def _send(self, event: Any) -> None:
        message = CoordinateMessage(time.time(), event)
        self._coordinator_client.send(CoordinateMsgSerializer.dumps(message))


def _stride_and(
    flat: Sequence[Optional[bool]], num_blocks: int
) -> List[Optional[bool]]:
    """Folds per-task verdict lists into one per-block mask.

    Results arrive group by group, each aligned with the request's manager
    blocks, so block ``i % num_blocks`` owns index ``i``. Abstentions are
    skipped: they neither pass nor fail the block.
    """
    mask: List[Optional[bool]] = [None] * num_blocks
    for i, ok in enumerate(flat):
        if ok is None:
            continue
        block = i % num_blocks
        previous = mask[block]
        mask[block] = ok if previous is None else (previous and ok)
    return mask


def _missing_methods(backend: Any) -> List[str]:
    """Protocol members ``backend`` does not implement.

    Derived from the protocol itself (a hand-kept list would drift from it):
    every public member of :class:`TransferBackend` is a method the worker
    calls, so an attribute that is missing or not callable is a contract
    breach the factory must reject."""
    return [
        name
        for name in dir(TransferBackend)
        if not name.startswith("_") and not callable(getattr(backend, name, None))
    ]


def make_transfer_backend(
    *,
    extra_config: "TairKvCacheConnectorExtraConfig",
    kvcache_info: "KVCacheInfo",
    manager_block_size: int,
    transfer_client: Any,
    coordinator_client: "TpCoordinatorClient",
    device: torch.device,
) -> TransferBackend:
    """Build the worker's data-plane backend.

    ``extra_config.transfer_backend_module_path`` empty selects the built-in
    GPU backend (``DataTransferManager``). Otherwise the named module must
    expose ``make_backend(**kwargs)``, called with this function's own
    keyword arguments.
    """
    module_path = extra_config.transfer_backend_module_path
    if not module_path:
        # Deferred: the GPU data plane is the only Triton consumer in the
        # package, and it must stay out of the import graph of hosts that
        # bring their own backend.
        from kv_cache_manager.py_connector.vllm.data_transfer import (
            DataTransferManager,
        )

        return DataTransferManager(
            kvcache_info,
            manager_block_size,
            transfer_client,
            coordinator_client,
            extra_config,
        )

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"transfer_backend_module_path={module_path!r}: cannot import the "
            f"transfer backend module ({exc}); it must be importable in every "
            f"worker process"
        ) from exc
    make_backend = getattr(module, "make_backend", None)
    if not callable(make_backend):
        raise TypeError(
            f"transfer_backend_module_path={module_path!r}: the module must "
            f"expose a callable make_backend(**kwargs) returning a "
            f"TransferBackend, got {make_backend!r}"
        )
    backend = make_backend(
        extra_config=extra_config,
        kvcache_info=kvcache_info,
        manager_block_size=manager_block_size,
        transfer_client=transfer_client,
        coordinator_client=coordinator_client,
        device=device,
    )
    # Fail here, at engine start, rather than on an IO thread later.
    missing = _missing_methods(backend)
    if missing:
        raise TypeError(
            f"transfer_backend_module_path={module_path!r}: "
            f"{type(backend).__name__} does not implement {missing}"
        )
    logger.info(
        "transfer backend: %s.%s", type(backend).__module__, type(backend).__qualname__
    )
    return backend
