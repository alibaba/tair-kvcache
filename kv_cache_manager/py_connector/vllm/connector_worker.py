"""Worker side of the connector: translation and data-plane transfer.

Owns the transfer client and the paged-cache views (TransferGroups). Every
instruction arriving through the metadata is self-contained (LoadRequest /
SaveRequest carry their own block tables), so the worker keeps no mirrored
request state -- tp0's per-request save-session ledger is the only
request-level bookkeeping. Per-step metadata is passed in explicitly by the
shell.
"""

import copy
import threading
import time
from dataclasses import dataclass
import json
import typing
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

# kvcm_py_client is the compiled pybind11 client; it ships no type stubs.
from kv_cache_manager.client.pybind import kvcm_py_client  # ty: ignore[unresolved-import]

import torch
from vllm.distributed import get_tensor_model_parallel_rank

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient
from kv_cache_manager.py_connector.common.tp_coordinator import (
    SaveContext,
    TpCoordinatorClient,
    TpCoordinatorServer,
)
from kv_cache_manager.py_connector.vllm.config import TairKvCacheConnectorExtraConfig
from kv_cache_manager.py_connector.vllm.data_transfer import (
    MultiResult,
    DataTransferManager,
    _get_device_module,
)
from kv_cache_manager.py_connector.vllm.metadata import (
    CaptureRequest,
    SaveRequest,
    TairKvCacheConnectorMetadata,
)
from kv_cache_manager.py_connector.vllm.transfer_types import (
    AttentionTransferGroup,
    KVCacheInfo,
    NullBlockRow,
    StateTransferGroup,
    TransferGroup,
    TransferPlan,
)
from kv_cache_manager.py_connector.vllm.vllm_common import (
    AttentionGroupMeta,
    GroupMeta,
    StateGroupMeta,
    attn_kv_views,
    spec_name,
    state_kv_view,
    windowed_block_has_data,
)

#: How long the eager capture waits for arena slots before withholding the
#: blocks of this request-step. Unlike a transfer task, a capture holds its
#: slots until the write session lands -- i.e. until the engine makes another
#: step -- so waiting without a deadline could deadlock the engine against its
#: own sessions. Timing out costs the blocks' hit rate (they are failed, never
#: published with a hole); the deadline itself is not a correctness parameter.
CAPTURE_POOL_WAIT_SECONDS = 0.05

if typing.TYPE_CHECKING:
    from vllm.forward_context import ForwardContext

    # vllm.attention no longer exists in vLLM >= 0.26; kept for the older
    # eras this connector supports.
    from vllm.attention import AttentionMetadata  # ty: ignore[unresolved-import]


@dataclass(frozen=True)
class _Capture:
    """One captured block's staging slot: the entry carries its own slot."""

    pool: Any
    slot: int


class _CaptureLedger:
    """The single accounting point for captured staging slots.

    Every transition of a capture's slot goes through this class -- staging a
    fresh capture, a session taking one (``take``), settling a session's
    outcomes (``settle``: return on success, keep for a retry on failure),
    the deliberate release a mask-derived instruction asks for, and the
    request's retirement. Nothing else may release one of these slots or put
    an entry back: that is the invariant the test family named after it
    checks -- at any moment a slot belongs to exactly one live entry in
    exactly one of the two maps, it is released exactly once, and no path
    reads a released slot.

    Two shapes make the bookkeeping non-trivial, and both are handled here
    instead of at the call sites:

    * in-flight ownership is keyed by ``(session, spec_name, block)``. A
      regressed save watermark can capture a block again while an earlier
      session's write is still running: both owners must keep their own
      record (and their own slot), or a failure callback would move or
      release a slot another task is still reading;
    * a request can retire while its saves are still running. Its staged
      entries are released then, but the in-flight records stay until every
      session settles; a settled *failure* for a retired request releases its
      slot (nothing will ever publish it) instead of staging it for a retry
      that cannot come.
    """

    def __init__(self) -> None:
        # One lock for every ownership transition: the settle callbacks run on
        # the transfer manager's io threads while staging, taking, releasing
        # and retirement run on the engine thread. A decision that reads the
        # maps has to write them before another thread reads, or two settles
        # of the same block can both find no staged entry and each overwrite
        # the other (leaking the slot one of them restaged), and a retirement
        # can leave an entry a racing settle just created.
        self._lock = threading.Lock()
        # (spec_name, manager block) -> _Capture: captured, not yet taken.
        self.staged: Dict[str, Dict[Tuple[str, int], _Capture]] = {}
        # (session id, spec_name, manager block) -> _Capture.
        self.in_flight: Dict[str, Dict[Tuple[int, str, int], _Capture]] = {}
        self._retired: Set[str] = set()

    @staticmethod
    def _release(capture: _Capture) -> None:
        capture.pool.release(capture.slot, 1)

    # ------------------------------------------------------------------ #
    def stage(
        self, req_id: str, key: Tuple[str, int], capture: _Capture
    ) -> Optional[int]:
        """Record a fresh capture; returns the superseded slot, if any.

        A block can be captured twice (the save watermark moves backwards when
        preemption shrinks an allocation). The newer bytes win: the old entry
        is released here -- it has no session reference, a session *moves* the
        entry into the in-flight map when it takes it.
        """
        with self._lock:
            previous = self.staged.setdefault(req_id, {}).get(key)
            self.staged[req_id][key] = capture
            if previous is None:
                return None
            self._release(previous)
            return previous.slot

    def take(
        self, req_id: str, session_id: int, key: Tuple[str, int]
    ) -> Optional[_Capture]:
        """Hand a staged capture to a session, atomically.

        ``None`` means "this session has nothing for that block": either no
        capture is staged, or this very session already holds the block (a
        repeated plan must not overwrite the first task's ownership).
        """
        with self._lock:
            flight_key = (session_id, key[0], key[1])
            sessions = self.in_flight.setdefault(req_id, {})
            if flight_key in sessions:
                return None
            capture = self.staged.get(req_id, {}).pop(key, None)
            if capture is None:
                return None
            sessions[flight_key] = capture
            return capture

    def taken_by(self, req_id: str, session_id: int, key: Tuple[str, int]) -> bool:
        """Is this block already in flight for this session?"""
        with self._lock:
            return (session_id, key[0], key[1]) in self.in_flight.get(req_id, {})

    def settle(
        self,
        req_id: str,
        session_id: int,
        outcomes: "Dict[Tuple[str, int], bool]",
    ) -> None:
        """Close out a session's blocks: ``True`` returns the slot, ``False``
        keeps it (the block is announced again, and the HBM window that
        produced the bytes has moved on -- they are the only copy left).

        A failure whose request already retired cannot be retried, so its slot
        is released. A block that was captured again while this task ran keeps
        the newer entry; the settled one goes back to the arena.
        """
        with self._lock:
            sessions = self.in_flight.get(req_id)
            if not sessions:
                return
            for key, ok in outcomes.items():
                capture = sessions.pop((session_id, key[0], key[1]), None)
                if capture is None:
                    continue  # settled already: nothing to release
                if ok or req_id in self._retired:
                    self._release(capture)
                    continue
                staged = self.staged.setdefault(req_id, {})
                if key in staged:
                    # A fresher capture for the same block wins.
                    self._release(capture)
                else:
                    staged[key] = capture
            if not sessions:
                self.in_flight.pop(req_id, None)
                self._retired.discard(req_id)

    def release_staged(self, req_id: str, blocks: "Sequence[int]") -> int:
        """Release the captures of these manager blocks (returns how many).

        No session will write them (the manager holds them already, or the
        session that announced them allocated nothing), so their bytes are
        dead weight in the arena.
        """
        with self._lock:
            staged = self.staged.get(req_id)
            if not staged:
                return 0
            wanted = set(blocks)
            released = 0
            for key in [key for key in staged if key[1] in wanted]:
                self._release(staged.pop(key))
                released += 1
            return released

    def retire(self, req_id: str) -> int:
        """Release everything still waiting for a session; keep in-flight
        records until their tasks settle (see settle)."""
        with self._lock:
            staged = self.staged.pop(req_id, {})
            for capture in staged.values():
                self._release(capture)
            if req_id in self.in_flight:
                self._retired.add(req_id)
            return len(staged)


class ConnectorWorker:
    """State and hooks for the worker-role connector instance (one per TP rank)."""

    def __init__(
        self,
        extra_config: TairKvCacheConnectorExtraConfig,
        group_metas: List[GroupMeta],
        manager_block_size: int,
        tp_size: int,
        host_ip: str,
        manager_client: KvCacheManagerClient,
        coordinator_client: TpCoordinatorClient,
        register_response: Dict[str, Any],
        max_num_batched_tokens: int = 0,
        max_num_seqs: int = 0,
    ) -> None:
        self._extra_config = extra_config
        # The engine's per-step token budget (chunked prefill's chunk) and its
        # cap on requests per step: the capture partition is sized from both
        # (see DataTransferManager).
        self._max_num_batched_tokens = max_num_batched_tokens
        self._max_num_seqs = max_num_seqs
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

        # Eagerly captured windowed blocks, per request: (spec name, manager
        # block) -> (staging pool, slot) holding that block's bytes, until a
        # write session takes them. The key is the *bucket's* spec name, not
        # the vLLM group index: one group may be split into several transfer
        # buckets (``_uniform_group_metas`` / ``_BucketKey``: a packed group can
        # mix two R-SWA windows), the spec name is unique by construction
        # (``_check_unique_names``) and is what the save plan looks its
        # locations up by -- keying on the group index would let two buckets of
        # one group overwrite each other's slots and hand each other's bytes to
        # the wrong location.
        #
        # Handing a block over *moves* its entry to ``_in_flight``, where the
        # task owns the release: it returns the slot when its transfer lands,
        # success or failure. ``_staged`` is therefore exactly "captured but
        # not yet given to a session", which is what a later capture (or the
        # retirement path) may release.
        #
        # The save verdict itself needs no side state: a block is written when
        # its location claimed the spec and the capture holds its bytes, fails
        # when the location claimed it and no bytes were captured (a withheld
        # or lost capture), and abstains when nothing claimed it
        # (``_staged_dispositions``).
        #
        # Every slot transition goes through this one ledger (see
        # _CaptureLedger): captured-not-taken entries in ``staged``,
        # session-owned ones in ``in_flight`` keyed by (session, spec, block).
        # The two attributes below are the ledger's maps, exposed read-only for
        # the tests and for the plan's diagnostics.
        self._captures = _CaptureLedger()
        self._staged = self._captures.staged
        self._in_flight = self._captures.in_flight

        self._tp_rank = get_tensor_model_parallel_rank()
        # torch device module (torch.cuda / torch.musa / ...), set in
        # register_kv_caches; Any because the module surface is dynamic.
        self._device_mod: Any = None
        port = extra_config.coordinator_base_port
        if self._tp_rank == 0:
            self._coordinator_server = TpCoordinatorServer(
                host_ip, port, tp_size, self.on_save_finished
            )

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
                spec_name(self._tp_rank, meta): meta.per_block_bytes
                for meta in self._group_metas
            },
        }
        init_params = kvcm_py_client.InitParams()
        init_params.role_type = kvcm_py_client.RoleType.WORKER
        init_params.self_location_spec_name = spec_name(
            self._tp_rank, self._group_metas[0]
        )
        init_params.storage_configs = f"{self._storage_configs}"
        transfer_client_config = json.dumps(transfer_client_json)
        logger.info("transfer_client_config: %s", transfer_client_config)
        self._transfer_client = kvcm_py_client.TransferClient.Create(
            transfer_client_config, init_params
        )
        assert self._transfer_client is not None, (
            "kvcm_py_client.TransferClient.Create failed"
        )
        logger.warning(
            "TairKvCacheConnector worker inited, tp rank: %d/%d, host: %s:%d, "
            "transfer buckets: %d",
            self._tp_rank,
            self._tp_size,
            host_ip,
            port,
            self._num_groups,
        )

    # ------------------------------------------------------------------ #
    # Storage config plumbing
    # ------------------------------------------------------------------ #
    def parse_hf3fs_configs(self, storage_configs: str) -> List[Dict[str, Any]]:
        hf3fs_configs: List[Dict[str, Any]] = []
        storage_configs_json = json.loads(storage_configs)
        for storage_config in storage_configs_json:
            if storage_config["type"] == "vcns_hf3fs":
                storage_config["type"] = "hf3fs"
            if storage_config["type"] == "hf3fs" and storage_config["is_available"]:
                hf3fs_configs.append(
                    {
                        "type": storage_config["type"],
                        "mountpoint": storage_config["storage_spec"]["mountpoint"],
                        "root_dir": storage_config["storage_spec"]["root_dir"],
                        "read_iov_block_size": self._extra_config.read_iov_block_size,
                        "read_iov_size": self._iov_size,
                        "write_iov_block_size": self._extra_config.write_iov_block_size,
                        "write_iov_size": self._iov_size,
                    }
                )
        self._storage_configs = json.dumps(storage_configs_json)
        return hf3fs_configs

    # ------------------------------------------------------------------ #
    # KV cache registration
    # ------------------------------------------------------------------ #
    def register_kv_caches(self, kv_caches: Dict[str, Any]) -> None:
        self._kv_caches = kv_caches
        first_attn = next(
            kv_caches[name]
            for meta in self._group_metas
            if isinstance(meta, AttentionGroupMeta)
            for name in meta.layer_names
        )
        self._dtype = first_attn.dtype
        self._device = first_attn.device
        self._device_mod = _get_device_module(self._device)

        groups: List[TransferGroup] = []
        for meta in self._group_metas:
            if isinstance(meta, AttentionGroupMeta):
                groups.append(self._build_attention_group(meta, kv_caches))
            else:
                # parse_groups only produces attention/state GroupMetas;
                # ty cannot infer that the complement is StateGroupMeta.
                groups.append(
                    self._build_state_group(meta, kv_caches)  # ty: ignore[invalid-argument-type]
                )

        self._kvcache_info = KVCacheInfo(
            tp_rank=self._tp_rank,
            world_size=self._tp_size,
            groups=groups,
            device=self._device,
            dtype=self._dtype,
        )
        self._data_transfer = DataTransferManager(
            self._kvcache_info,
            self._manager_block_size,
            self._transfer_client,
            self._coordinator_client,
            self._extra_config,
            max_num_batched_tokens=self._max_num_batched_tokens,
            max_num_seqs=self._max_num_seqs,
        )

        logger.warning(
            "register_kv_caches done: %s",
            [
                (g.spec_name, type(g).__name__, g.layer_num, g.per_block_bytes)
                for g in groups
            ],
        )

    def _build_attention_group(
        self, meta: AttentionGroupMeta, kv_caches: Dict[str, Any]
    ) -> AttentionTransferGroup:
        spec = spec_name(self._tp_rank, meta)
        tensors = [kv_caches[name] for name in meta.layer_names]
        ref = tensors[0]
        for t in tensors:
            assert t.shape == ref.shape and t.stride() == ref.stride(), (
                "attention layers in one group must share shape/stride"
            )
        # Normalize the layout into per-pointer token-major views of shape
        # (num_blocks, kernel_block_size, heads, content_dim); everything
        # below is layout-independent. Split-K/V layouts (vllm <= 0.25.x)
        # yield two views (= two transfer pointers) per layer, the packed
        # layout (vllm >= 0.26.0) and the MLA latent cache yield one.
        ref_views, kv_layout = attn_kv_views(ref)
        view = ref_views[0]
        kernel_block_size = view.shape[1]
        assert meta.block_size % kernel_block_size == 0, (
            f"group block size {meta.block_size} not a multiple of kernel "
            f"block size {kernel_block_size}"
        )
        per_token_dim = view.shape[2] * view.shape[3]  # heads * content_dim
        # The gather/scatter kernel needs token-major memory inside a page:
        # logical dims (blk, tok, head, dim) contiguous within a block. This
        # is vLLM's NHD order; HND would interleave heads across tokens.
        for v in ref_views:
            assert v.stride()[1:] == (per_token_dim, v.shape[3], 1), (
                f"kv cache page not token-major: shape={tuple(v.shape)} "
                f"stride={v.stride()}; the gather/scatter needs a token-major "
                f"page (vLLM's NHD order, spelled BLNHC/LBNHC among its "
                f"VLLM_KV_CACHE_LAYOUT candidates; this tensor is head-major)"
            )
        # Cross-check the spec against the tensor per *token*: the spec page
        # covers meta.block_size tokens while the tensor page covers
        # kernel_block_size rows (a legal vLLM split: the framework block only
        # needs to be a multiple of the kernel's), so byte-per-token is the
        # comparable unit. A backend that packs rows differently (e.g. leaving
        # fp8 scales out) is caught here instead of mis-sizing every location.
        itemsize = view.element_size()
        spec_per_token = meta.page_bytes // meta.block_size
        tensor_per_token = len(ref_views) * per_token_dim * itemsize
        if spec_per_token != tensor_per_token:
            raise NotImplementedError(
                f"group {meta.group_idx}{meta.spec_suffix}: spec page_bytes="
                f"{meta.page_bytes} over block_size={meta.block_size} is "
                f"{spec_per_token} B/token, but the tensor stores "
                f"{tensor_per_token} B/token ({len(ref_views)} view(s) x "
                f"{kernel_block_size} rows x {per_token_dim} elements x "
                f"{itemsize} B per kernel page); the KV cache layout does not "
                f"match the spec (unsupported backend packing or mixed layouts)"
            )
        # Non-flat block layouts (page_size_padded gaps, or split K/V
        # interleaved per block as in the 5-D N-first layout) go through the
        # kernel's strided path. Stride 0 = fast flat indexing.
        flat = view.stride(0) == kernel_block_size * per_token_dim
        block_stride = 0 if flat else view.stride(0)
        # Pointer array ordered [K0, V0, K1, V1, ...] for split layouts and
        # [L0, L1, ...] for the packed layout; each view's data_ptr() is its
        # own storage base, so the kernel never adds a K->V offset.
        ptrs = [v.data_ptr() for t in tensors for v in attn_kv_views(t)[0]]
        ptr_tensor = torch.tensor(ptrs, dtype=torch.int64, device="cpu").to(
            self._device
        )
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
            dtype=ref.dtype,
            retention=meta.retention,
        )

    def _build_state_group(
        self, meta: StateGroupMeta, kv_caches: Dict[str, Any]
    ) -> StateTransferGroup:
        # A state exists once per state block and covers the prefix ending at
        # that block's last token, so _state_block_ids' "last token selects the
        # block" needs the manager block to be a whole number of state blocks:
        # with --mamba-block-size <= --block-size an mbs-token manager block
        # ends inside (or exactly on) a state block, and a 48-token manager
        # block over 16-token states selects states 2, 5, 8, ... -- one state
        # each, ending where the manager block ends (writes publish the state
        # that really covers the block's prefix).
        #
        # The other direction is the misplacement: a *larger* state block
        # (--mamba-block-size above --block-size) covers several manager blocks,
        # so one state would be published under several prefix keys.
        assert self._manager_block_size % meta.block_size == 0, (
            f"state group {meta.group_idx}: manager block size "
            f"{self._manager_block_size} is not a multiple of the state block "
            f"size {meta.block_size}; a state covers the prefix ending at its "
            f"block, so a larger state block would be published under several "
            f"manager block keys (a silently misplaced state)"
        )
        spec = spec_name(self._tp_rank, meta)
        block_views = []
        for name in meta.layer_names:
            try:
                block_views.append(state_kv_view(kv_caches[name], meta.page_size_bytes))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"state layer {name!r}: {exc}") from exc
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
    def _attn_token_indices(
        self,
        group: AttentionTransferGroup,
        manager_block_idxes: Sequence[int],
        block_table: List[int],
    ) -> List[List[int]]:
        """Map manager blocks to flat token slots of one attention group.

        Three-tier hierarchy:
          manager block (KVCM unit) -> global token idx
          -> group block (block_table unit, group.block_size tokens)
          -> kernel physical block (tensor unit; ratio physical per group block).
        """
        slots, _ = self._attn_slots(group, manager_block_idxes, block_table)
        return slots

    def _attn_slots(
        self,
        group: AttentionTransferGroup,
        manager_block_idxes: Sequence[int],
        block_table: List[int],
    ) -> Tuple[List[List[int]], List[List[bool]]]:
        """``_attn_token_indices`` plus, per token, "is this a null slot?".

        A windowed group's block table carries vLLM's null block (id 0) for the
        positions outside the window: those slots alias page 0, which is the
        engine's shared zero page. Both callers have to know (the loader must
        not write them, the capture must not publish their bytes), and the
        block table entry is the only trustworthy source -- a group block is
        null exactly when its table entry is 0.
        """
        mbs = self._manager_block_size
        gbs = group.block_size
        kbs = group.kernel_block_size
        ratio = gbs // kbs
        out: List[List[int]] = []
        null: List[List[bool]] = []
        for mb in manager_block_idxes:
            idxs = []
            nulls = []
            base = mb * mbs
            for i in range(mbs):
                tok = base + i
                logical = tok // gbs
                assert logical < len(block_table), (
                    f"group block {logical} out of range (len={len(block_table)})"
                )
                off = tok % gbs
                phys = block_table[logical] * ratio + off // kbs
                idxs.append(phys * kbs + off % kbs)
                nulls.append(block_table[logical] == 0)
            out.append(idxs)
            null.append(nulls)
        return out, null

    def _state_block_ids(
        self,
        group: StateTransferGroup,
        manager_block_idxes: Sequence[int],
        block_table: List[int],
    ) -> List[int]:
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
                f"group block {logical} out of range (len={len(block_table)})"
            )
            out.append(block_table[logical])
        return out

    # ------------------------------------------------------------------ #
    # Load / save
    # ------------------------------------------------------------------ #
    def _iter_task_chunks(
        self, plans: List[TransferPlan], per_task: Callable[[TransferGroup], int]
    ) -> Iterator[
        Tuple[
            TransferGroup,
            List[Optional[str]],
            Optional[List[List[int]]],
            Optional[List[int]],
            Optional[List[Optional[int]]],
            Optional[List[Optional[NullBlockRow]]],
        ]
    ]:
        """Slice each plan's blocks into per-task chunks.

        ``per_task`` resolves the batch for one group: the data transfer
        manager shrinks it to fit the group's staging pool (a task stages its
        whole batch as one contiguous run), so the configured
        ``block_per_save_task``/``block_per_load_task`` is an upper bound.

        The chunk is ``(group, uris, token_indices, block_ids, staged_slots,
        null_rows)`` -- the two trailing fields are the windowed-group extras
        (see TransferPlan); a group that has none yields None for both, which
        is what makes the save and load tasks one signature.

        Pure generator: the task index is the consumer's concern (enumerate
        there); this only reads the plans."""
        for plan in plans:
            n = len(plan.uris)
            batch = per_task(plan.group)
            for i in range(0, n, batch):
                end = min(n, i + batch)
                yield (
                    plan.group,
                    plan.uris[i:end],
                    plan.token_indices[i:end]
                    if plan.token_indices is not None
                    else None,
                    plan.block_ids[i:end] if plan.block_ids is not None else None,
                    plan.staged_slots[i:end]
                    if plan.staged_slots is not None
                    else None,
                    plan.null_rows[i:end] if plan.null_rows is not None else None,
                )

    def _settle_session_captures(
        self,
        save_req: SaveRequest,
        plans: List[TransferPlan],
        done_cb: Callable[[List[Any]], None],
    ) -> Callable[[List[Any]], None]:
        """Settle this session's captured slots once its tasks report.

        Every windowed save hands its blocks to a session (the ledger's
        ``take``) and this is where they come to rest: a block that was
        written returns its slot, a block that failed keeps it -- the bytes are
        the only copy the connector will ever hold (the HBM window that
        produced them has moved on, so a later capture abstains) and a later
        session announces the block again, so the retry must find them staged.
        A request that retired before the callback ran releases the slot
        instead (nothing will publish it).

        The verdict list is flat, group-major, ``num_blocks`` per group (the
        same layout create_save_done_callback reads). It can be short when a
        task never reported: the session's slots then stay in flight, which is
        exactly where the ledger keeps them until something settles.
        """
        windowed = [
            (group_idx, plan)
            for group_idx, plan in enumerate(plans)
            if getattr(plan.group, "retention", None) is not None
            and plan.staged_slots
        ]
        if not windowed:
            return done_cb
        num_blocks = len(save_req.manager_block_idxes)

        def wrapper(flat: List[Any]) -> None:
            # Settle *first* and report afterwards: the report goes to the
            # coordinator (a socket that can fail), and an exception there
            # would otherwise skip the settle for good -- MultiResult calls its
            # callback once, so this session's slots would stay in flight for
            # the process's life, eating the arena of unrelated requests. The
            # finally keeps the report ordered even when the settle raises.
            try:
                if len(flat) < len(plans) * num_blocks:
                    # A task never reported (an exception escaped the executor):
                    # its slots stay in flight until something settles them.
                    return
                outcomes: Dict[Tuple[str, int], bool] = {}
                for group_idx, plan in windowed:
                    for pos, slot in enumerate(plan.staged_slots):
                        if slot is None:
                            continue
                        block = save_req.manager_block_idxes[pos]
                        ok = flat[group_idx * num_blocks + pos]
                        if ok is None:
                            continue
                        outcomes[(plan.group.spec_name, block)] = bool(ok)
                self._captures.settle(
                    save_req.req_id, save_req.session_id, outcomes
                )
                failed = [key for key, ok in outcomes.items() if not ok]
                if failed:
                    logger.warning(
                        "req:%s session %d: %d windowed block(s) failed to write; "
                        "their captured bytes stay staged for a retry",
                        save_req.req_id,
                        save_req.session_id,
                        len(failed),
                    )
            finally:
                done_cb(flat)

        return wrapper

    def _windowed_save_annotations(
        self,
        req_id: str,
        session_id: int,
        group: AttentionTransferGroup,
        manager_block_idxes: Sequence[int],
        uris: List[Optional[str]],
    ) -> Dict[str, Any]:
        """Staging slots for a windowed group's save, block by block.

        The slot the capture holds, or None where it holds nothing; what None
        *means* is the task's business, and the location tells it which case
        this is: an announced location with no bytes must not be published (the
        block fails, withdrawing the claim), while a block the announcement
        already described as window-less wrote nothing by design and abstains.

        Handing a block over *moves* its entry from ``_staged`` to
        ``_in_flight``: the task owns the slot from here on and returns it when
        its transfer lands. A block is announced once, so a second plan for the
        same block means the announcement was repeated -- it gets no slot (and
        no location to write to), rather than a slot two tasks would both
        release.
        """
        slots: List[Optional[int]] = []
        for block in manager_block_idxes:
            key = (group.spec_name, block)
            capture = self._captures.take(req_id, session_id, key)
            if capture is None:
                if self._captures.taken_by(req_id, session_id, key):
                    logger.warning(
                        "req:%s windowed block %d (group %s) is planned twice; "
                        "its bytes stay with the first session and this block is "
                        "failed instead of handing the same slot out twice",
                        req_id,
                        block,
                        group.spec_name,
                    )
                slots.append(None)
                continue
            slots.append(capture.slot)
        missing = [
            i for i, slot in enumerate(slots) if slot is None and uris[i] is not None
        ]
        if missing:
            logger.warning(
                "save group %s: %d/%d announced blocks have no captured bytes "
                "(withheld capture?); failing them instead of publishing a hole",
                group.spec_name,
                len(missing),
                len(manager_block_idxes),
            )
        return {"staged_slots": slots}

    def _windowed_load_annotations(
        self,
        group: AttentionTransferGroup,
        manager_block_idxes: Sequence[int],
        token_indices: List[List[int]],
        null_masks: List[List[bool]],
    ) -> Dict[str, Any]:
        """Null-slot disposition of a windowed group's load, block by block.

        Rows pointing at vLLM's null block are invisible to attention, but
        writing them would scribble on the shared zero page: they are aliased
        onto the block's first real row instead (and the kernel's token slots
        are rewritten to match, so the real bytes land on the real slots). A
        block whose every row is null has nothing to load at all.
        """
        annotations: List[Optional[NullBlockRow]] = []
        for i, _block in enumerate(manager_block_idxes):
            nulls = null_masks[i]
            if not any(nulls):
                annotations.append(None)
                continue
            rows = [r for r, is_null in enumerate(nulls) if is_null]
            source = next((r for r, is_null in enumerate(nulls) if not is_null), None)
            if source is not None:
                for row in rows:
                    token_indices[i][row] = token_indices[i][source]
            annotations.append(NullBlockRow(rows=rows, source_row=source))
        return {"null_rows": annotations}

    def _plan_group_transfers(
        self,
        locations: List[Dict[str, Any]],
        manager_block_idxes: Sequence[int],
        block_ids_per_group: List[List[int]],
        req_id: str,
        *,
        for_save: bool = False,
        session_id: int = 0,
    ) -> Optional[List[TransferPlan]]:
        """Build the per-group TransferPlans for a set of manager blocks.

        ``uris`` is positionally aligned with ``manager_block_idxes`` and may
        contain ``None`` where a block carries no data for that group's spec
        (hybrid per-block spec coverage, see ``build_spec_groups``). Deciding
        what a hole means is the transfer task's job: for attention groups a
        hole is a failure, for state groups it must agree with the block
        table (a null state block). Returns None only if the manager returned
        the wrong number of locations altogether.

        Windowed groups get their own annotations in the same pass, and which
        ones depends on the direction: a *save* plan points each block at the
        staging slot the eager capture filled (``staged_slots``, with a
        withdrawn location for the blocks whose capture was withheld), a *load*
        plan carries the rows whose target slot is vLLM's null block
        (``null_rows``). Both are read off the block table snapshot the
        instructions carry; the late-gather path of full-attention groups is
        untouched.
        """
        num_blocks = len(manager_block_idxes)
        if len(locations) != num_blocks:
            logger.warning(
                "%d locations for %d blocks, skip transfer", len(locations), num_blocks
            )
            return None
        # One pass over the locations: per block, map spec name -> uri. The
        # per-group extraction below then reads by name instead of rescanning
        # every location's spec list for every group.
        per_block_uri_maps = [
            {s["name"]: s["uri"] for s in location.get("location_specs", [])}
            for location in locations
        ]
        self._check_block_table_covers(block_ids_per_group)
        plans = []
        for group in self._kvcache_info.groups:
            uris = [m.get(group.spec_name) for m in per_block_uri_maps]
            # block_ids_per_group is indexed by the vLLM group index.
            block_table = block_ids_per_group[group.group_idx]
            if isinstance(group, AttentionTransferGroup):
                if for_save and group.retention is not None:
                    # A staged (windowed) save reads its bytes out of the
                    # staging arena, not out of vLLM's slots, so it does not
                    # translate manager blocks through the block table at all:
                    # after a preemption the resume's table can be shorter than
                    # the prefix whose bytes the capture still holds, and the
                    # translation's range check would abort a save that needs
                    # nothing from the table (the captured block keys are the
                    # old prefix's, and so are its bytes).
                    plans.append(
                        TransferPlan(
                            group=group,
                            uris=uris,
                            token_indices=[],
                            **self._windowed_save_annotations(
                                req_id, session_id, group, manager_block_idxes, uris
                            ),
                        )
                    )
                    continue
                token_indices, null_masks = self._attn_slots(
                    group, manager_block_idxes, block_table
                )
                extra: Dict[str, Any] = {}
                if group.retention is not None:
                    # The save side of a windowed group returned above; what is
                    # left here is the load, whose rows come from the table.
                    extra = self._windowed_load_annotations(
                        group, manager_block_idxes, token_indices, null_masks
                    )
                plans.append(
                    TransferPlan(
                        group=group,
                        uris=uris,
                        token_indices=token_indices,
                        **extra,
                    )
                )
            else:
                # KVCacheInfo groups are only attention or state groups;
                # ty cannot infer that the complement is StateTransferGroup.
                plans.append(
                    TransferPlan(
                        group=group,
                        uris=uris,
                        block_ids=self._state_block_ids(
                            group,  # ty: ignore[invalid-argument-type]
                            manager_block_idxes,
                            block_table,
                        ),
                    )
                )
        return plans

    def _check_block_table_covers(self, block_ids_per_group: List[List[int]]) -> None:
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
            f"alignment between vLLM and this connector is broken"
        )

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        meta: TairKvCacheConnectorMetadata,
        **kwargs: Any,
    ) -> None:
        for load_req in meta.to_load_requests:
            if not load_req.need_load_locations:
                continue
            num_blocks = len(load_req.manager_block_idxes)
            plans = self._plan_group_transfers(
                load_req.need_load_locations,
                load_req.manager_block_idxes,
                load_req.all_block_ids,
                load_req.req_id,
            )

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
                    f"invalid blocks"
                )
                assert only.group_idx == 0, (
                    f"single vLLM block table but the transferred group is "
                    f"group {only.group_idx}; group-index alignment broken"
                )
                table = load_req.all_block_ids[0]
                gbs = only.block_size
                report_ids = [
                    table[(mb * self._manager_block_size) // gbs]
                    for mb in load_req.manager_block_idxes
                ]
            done_cb = self._data_transfer.create_load_done_callback(
                load_req.req_id,
                self._tp_rank,
                meta.epoch,
                copy.copy(report_ids),
                num_blocks,
                report_failures=report_failures,
            )

            if plans is None:
                # Nothing submitted; report the whole load as failed.
                mr = MultiResult(1, done_cb)
                mr.submit_result(0, [False] * num_blocks * self._num_groups)
                continue

            chunks = list(
                self._iter_task_chunks(plans, self._data_transfer.load_task_blocks)
            )
            multi_result = MultiResult(len(chunks), done_cb)
            for task_idx, chunk in enumerate(chunks):
                self._data_transfer.submit_task(
                    self._data_transfer.load_task, multi_result, task_idx, *chunk
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self, meta: TairKvCacheConnectorMetadata) -> None:
        """Capture the windowed groups' new blocks, then write the sessions.

        Capture first and unconditionally: those blocks are freed by vLLM's
        next schedule step, so this hook -- the tail of the step whose forward
        produced them -- is the only moment their bytes are still in HBM. The
        writes that follow are the sessions an earlier step's HTTP call
        produced; a windowed group's task takes its bytes from the capture
        (``staged_slots``) instead of gathering, so the announce -> session lag
        never touches HBM, while full-attention groups keep the late gather.
        """
        if meta.to_capture_requests:
            self._capture_blocks(meta)
        if not meta.to_save_requests:
            for req_id, blocks in meta.to_release_captures:
                self._release_captures_for(req_id, blocks)
            return
        ready_event = self._device_mod.Event()
        ready_event.record(self._device_mod.current_stream())

        for save_req in meta.to_save_requests:
            num_blocks = len(save_req.manager_block_idxes)
            plans = self._plan_group_transfers(
                save_req.target_locations,
                save_req.manager_block_idxes,
                save_req.all_block_ids,
                save_req.req_id,
                for_save=True,
                session_id=save_req.session_id,
            )

            done_cb = self._data_transfer.create_save_done_callback(
                save_req.req_id, self._tp_rank, save_req.write_session_id, num_blocks
            )
            if plans is not None:
                done_cb = self._settle_session_captures(save_req, plans, done_cb)

            if plans is None:
                mr = MultiResult(1, done_cb)
                mr.submit_result(0, [False] * num_blocks * self._num_groups)
                continue

            chunks = list(
                self._iter_task_chunks(plans, self._data_transfer.save_task_blocks)
            )
            multi_result = MultiResult(len(chunks), done_cb)
            for task_idx, chunk in enumerate(chunks):
                self._data_transfer.submit_task(
                    self._data_transfer.save_task,
                    multi_result,
                    task_idx,
                    *chunk,
                    ready_event,
                )
            if self._tp_rank == 0:
                # One session in flight; the coordinator's completion event
                # for this req decrements it (see get_finished).
                self._pending_saves[save_req.req_id] = (
                    self._pending_saves.get(save_req.req_id, 0) + 1
                )

        # The queued releases come *after* every plan of this batch has taken
        # its captures: a release queued for a block an older session left
        # behind must not be able to withdraw the bytes this step's session is
        # about to write (the scheduler re-checks its protection when it
        # flushes, and this is the same rule one layer down).
        for req_id, blocks in meta.to_release_captures:
            self._release_captures_for(req_id, blocks)

    # ------------------------------------------------------------------ #
    # Eager capture (windowed groups)
    # ------------------------------------------------------------------ #
    def _windowed_metas(self) -> List[AttentionGroupMeta]:
        return [
            meta
            for meta in self._group_metas
            if isinstance(meta, AttentionGroupMeta) and meta.retention is not None
        ]

    def _windowed_groups(self) -> List[AttentionTransferGroup]:
        return [
            g
            for g in self._kvcache_info.groups
            if isinstance(g, AttentionTransferGroup) and g.retention is not None
        ]

    def _capture_blocks(self, meta: TairKvCacheConnectorMetadata) -> None:
        """Copy every request-step of this step, on one step-level budget.

        The capture waits a bounded time for arena room (see
        ``CAPTURE_POOL_WAIT_SECONDS``); when the arena is held by in-flight
        writes, *every* request of the step would pay that wait and the engine
        thread pays the sum of them. One timeout is therefore the whole step's
        budget: the remaining captures are skipped (they would only time out
        too) and their blocks fail closed like a withheld capture -- announced
        location, no bytes, so the block is not published.
        """
        captures = meta.to_capture_requests
        for index, capture in enumerate(captures):
            try:
                if not self._capture_request(capture):
                    skipped = len(captures) - index - 1
                    if skipped:
                        logger.warning(
                            "windowed capture budget for this step is spent "
                            "(%.0f ms); skipping %d further capture(s), whose "
                            "blocks are not published",
                            CAPTURE_POOL_WAIT_SECONDS * 1000,
                            skipped,
                        )
                    break
            except Exception:
                # Degrade, never crash the engine mid-step: the block's save
                # verdict becomes a failure (its location claimed the spec and
                # no bytes were captured), so nothing is published for it.
                logger.exception(
                    "req:%s windowed capture failed, failing %d blocks",
                    capture.req_id,
                    len(capture.manager_block_idxes),
                )

    def _capture_request(self, capture: CaptureRequest) -> bool:
        """Copy this request-step's windowed blocks into the staging arena.

        Returns False when the request-step was withheld for lack of arena room
        within the capture deadline (the caller then stops capturing for this
        step), True otherwise.

        All-or-nothing per request-step: a block's publish verdict is the AND
        over its groups, so a partially captured step would fail the blocks
        anyway -- releasing the runs again keeps the arena free for the steps
        that can be captured whole. Blocks whose windowed slots are already
        null are left out of the runs (abstain: their bytes do not exist).
        """
        groups = self._windowed_groups()
        if not groups:
            return True
        blocks = capture.manager_block_idxes
        if not blocks:
            return True
        metas = self._windowed_metas()
        per_group: List[Tuple[AttentionTransferGroup, List[List[int]], List[int]]] = []
        for group in groups:
            table = capture.all_block_ids[group.group_idx]
            slots, _null_masks = self._attn_slots(group, blocks, table)
            # The *shared* predicate, on the announced snapshot: what the
            # scheduler announced as "has window data" is exactly what this
            # capture copies -- the two must never disagree, since the
            # announcement is what a later hit trusts.
            captureable = [
                i
                for i in range(len(blocks))
                if windowed_block_has_data(
                    metas, capture.all_block_ids, blocks[i], self._manager_block_size
                )
            ]
            if captureable:
                per_group.append((group, slots, captureable))
        if not per_group:
            return True
        # Take every run before writing anything down: a failure must leave no
        # half-held slot behind, so the runs of the groups that did fit are
        # released again when a later group does not fit (or raises).
        #
        # One deadline for the whole request-step: the budget is what a *step*
        # may spend holding the engine thread, not what each bucket may spend
        # on its own (a handful of buckets would otherwise stall the step for
        # buckets x CAPTURE_POOL_WAIT_SECONDS while the slots trickle free).
        deadline = time.monotonic() + CAPTURE_POOL_WAIT_SECONDS
        taken: List[Tuple[AttentionTransferGroup, List[int], Any, int]] = []
        withheld: Optional[AttentionTransferGroup] = None
        try:
            for group, slots, captureable in per_group:
                start = self._data_transfer.capture_blocks(
                    group,
                    [slots[i] for i in captureable],
                    timeout=max(deadline - time.monotonic(), 0.0),
                )
                if start is None:
                    withheld = group
                    break
                taken.append(
                    (group, captureable, self._data_transfer.pool_of(group), start)
                )
        except BaseException:
            self._release_runs(taken)
            raise
        if withheld is not None:
            self._release_runs(taken)
            logger.warning(
                "req:%s windowed capture of %d blocks withheld: the staging "
                "arena of %s has no free run within %.0f ms (held by in-flight "
                "writes); those blocks are failed, not published",
                capture.req_id,
                len(blocks),
                withheld.spec_name,
                CAPTURE_POOL_WAIT_SECONDS * 1000,
            )
            return False
        for group, captureable, pool, start in taken:
            for offset, i in enumerate(captureable):
                # A repeat capture for a block already staged (the save
                # watermark moved backwards when preemption shrank the
                # allocation): the ledger supersedes the old entry -- it has no
                # session reference, a session *moves* the entry into the
                # in-flight map when it takes it -- and returns its slot.
                superseded = self._captures.stage(
                    capture.req_id,
                    (group.spec_name, blocks[i]),
                    _Capture(pool=pool, slot=start + offset),
                )
                if superseded is not None:
                    logger.warning(
                        "req:%s block %d of %s was captured twice (the "
                        "announcement repeated); releasing its previous slot %d",
                        capture.req_id,
                        blocks[i],
                        group.spec_name,
                        superseded,
                    )
        logger.info(
            "req:%s captured %d windowed blocks (%d groups) before the window "
            "moved",
            capture.req_id,
            sum(len(captureable) for _, captureable, _, _ in taken),
            len(taken),
        )
        return True

    @staticmethod
    def _release_runs(taken: Sequence[Tuple[Any, List[int], Any, int]]) -> None:
        """Give back the runs of a capture that will not be kept."""
        for _group, captureable, pool, start in taken:
            pool.release(start, len(captureable))

    def _release_captures_for(self, req_id: str, blocks: Sequence[int]) -> None:
        """Return the captured slots of blocks no session is going to write.

        The manager reported them as already cached when the session came back,
        so the bytes the eager capture copied for them are never published (the
        session does not even mention them): holding them until the request
        retires would pin one arena slot per already-cached block for as long
        as the request runs. Entries already handed to a session are the task's
        to release and are left alone.
        """
        released = self._captures.release_staged(req_id, blocks)
        if released:
            logger.info(
                "req:%s released %d captured block slots the manager already "
                "holds (nothing to publish for them)",
                req_id,
                released,
            )

    def _release_staged(self, req_id: str) -> None:
        """Return the capture slots the request never handed to a session.

        Called when the request retires: whatever no session referenced (an
        announcement that was canceled, a step that was never re-announced) must
        not pin the arena for the rest of the process's life. A block that *did*
        reach a session is that session's to release -- it returns its slot when
        the transfer lands, success or failure -- because releasing in-flight
        slots here would hand live bytes to the next capture before the SDK read
        them and free the same slot twice (the pool's free list would then hold
        the range twice and dispatch it to two transfers; review_r1 S1).
        """
        released = self._captures.retire(req_id)
        in_flight = self._in_flight.get(req_id)
        logger.debug(
            "req:%s released %d un-taken captured block slots (%d in flight "
            "with their sessions)",
            req_id,
            released,
            len(in_flight or ()),
        )

    def on_save_finished(
        self, write_session_id: str, save_context: SaveContext
    ) -> None:
        for block_idx in range(len(save_context.locations)):
            fully_saved = all(
                save_context.result_per_rank[rank][block_idx]
                for rank in range(self._tp_size)
            )
            save_context.success_mask.append(fully_saved)
        logger.debug(
            "finish_write_cache mask:%s session:%s",
            save_context.success_mask,
            write_session_id,
        )
        try:
            self._manager_client.finish_write_cache(
                {
                    "trace_id": "finish_%s" % write_session_id[:8],
                    "instance_id": self._extra_config.instance_id,
                    "write_session_id": write_session_id,
                    "success_blocks": {
                        "bool_masks": {"values": save_context.success_mask}
                    },
                }
            )
        except Exception as e:
            logger.warning(
                "finish_write_cache failed, session: %s, error: %s", write_session_id, e
            )

    def get_finished(
        self, finished_req_ids: set, meta: TairKvCacheConnectorMetadata
    ) -> Tuple[Optional[set], Optional[set]]:
        """Report request completion to vLLM.

        A request is reported once its last save session has settled on the
        coordinator (tp0 aggregates the per-rank verdicts into
        finish_write_cache). The scheduler's FinishRequest -- sent when its
        own ledger says the request is settled -- either reports immediately
        or defers until the last session lands."""
        for finish_req in meta.to_finish_requests:
            # The request is done on the scheduler side (every announced
            # session was handed over), so no further write can reference its
            # captures: release whatever a session never took.
            self._release_staged(finish_req.req_id)
        if self._tp_rank != 0:
            for finish_req in meta.to_finish_requests:
                self._pending_saves.pop(finish_req.req_id, None)
                self._finish_pending.discard(finish_req.req_id)
            return None, None

        finished_saving = []
        finished_saving_tasks, finished_loading_tasks = (
            self._coordinator_server.get_finished_tasks()
        )
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
