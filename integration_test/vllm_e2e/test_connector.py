"""A verification wrapper around the production KVCM vLLM connector.

This connector is injected via vLLM's ``kv_connector_module_path`` and subclasses
the production ``TairKvCacheConnector`` without modifying it. Its purpose is to
independently capture the KV data that lives in vLLM's paged KV cache so that the
test driver can verify the connector's save/load translation layer.

Why this catches translation bugs
---------------------------------
The production connector is built around *per-group* transfer. Every
``kv_cache_group`` (a ``FullAttentionSpec`` group for pure-attention models, or
several ``MambaSpec`` groups plus one ``FullAttentionSpec`` group for hybrid
models) is a self-contained transfer unit with its own block table and its own
translation:

    KVCM manager block idx  ->  global token idx  ->  group logical block
        (step 1, connector-only)     (step 2/3, shared with vLLM)

Step 1 is connector-only logic. A bug there makes *save* gather from the wrong
physical slots and *load* scatter to the wrong physical slots. Because save and
load share the same translation, a transport-level round trip still "matches"
(the bug is symmetric).

To break the symmetry we capture KV data using ONLY the position -> physical
slot mapping that vLLM itself uses (its slot_mapping kernel), a pure function of
the group's block table and the token position. This reference is independent of
the connector's step-1 logic, so a step-1 bug makes the captured data diverge
from what the connector saved/loaded.

Group kinds
-----------
* Attention groups -> ``torch.Tensor`` of shape
  ``[2, num_blocks, kernel_block_size, num_kv_heads, head_size]``. Captured
  per-token using the three-tier mapping (group logical block -> kernel physical
  block) because the scheduler's group block size may exceed the kernel block
  size.
* Mamba/linear/gdn groups -> ``list[Tensor]`` (e.g. ``[conv_state, ssm_state]``).
  The state is stored **per group block**; we capture the whole state slice for
  the group block that each manager block maps to (mirroring the connector's
  ``_state_block_ids``: a manager block's *last* token selects the block).

Capture points
--------------
* Reference (save path): in ``wait_for_save`` we read the KV of the saved token
  range straight out of the paged cache (the forward pass has completed and the
  slots are not modified by the parent's async gather).
* Loaded (load path): loads are async, so the load step has no forward pass and
  the worker does not yet know the request's token ids. We record the load's
  per-group block tables in ``start_load_kv`` and emit the capture in a later
  ``wait_for_save`` once the token ids have arrived. The loaded KV persists in
  the paged cache (its blocks are allocated to the request).

Captures are written to ``$KVCM_E2E_CAPTURE_DIR`` as ``.pt`` files named
``{ref|loaded}_tp{rank}_{token_hash}.pt`` so the out-of-process driver can match
reference vs loaded by content (the captured token ids).
"""

import hashlib
import os
import threading
import typing

import torch

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.vllm.metadata import TairKvCacheConnectorMetadata
from kv_cache_manager.py_connector.vllm.v1_connector import (
    TairKvCacheConnector, attn_kv_views)
from kv_cache_manager.py_connector.vllm.connector_worker import ConnectorWorker
from kv_cache_manager.py_connector.vllm.vllm_common import AttentionGroupMeta

CAPTURE_DIR_ENV = "KVCM_E2E_CAPTURE_DIR"


def _worker_attr(name):
    """Property forwarding to the ConnectorWorker's attributes: the role
    split moved them off the shell, and the capture hooks run on the
    worker-role instance."""
    return property(lambda self: getattr(self.connector_worker, name))


class VerifyingConnector(TairKvCacheConnector):
    """Production connector + independent per-group KV capture for e2e."""

    _tp_rank = _worker_attr("_tp_rank")
    _device_mod = _worker_attr("_device_mod")
    _device = _worker_attr("_device")
    _kv_caches = _worker_attr("_kv_caches")
    _data_transfer = _worker_attr("_data_transfer")

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        super().register_kv_caches(kv_caches)

        self._capture_dir = os.environ.get(CAPTURE_DIR_ENV, "")
        if self._capture_dir:
            os.makedirs(self._capture_dir, exist_ok=True)

        # Snapshot the static per-group description into a capture-friendly form.
        # Each entry: (group_idx, is_attention, layer_names, group_block_size,
        # kernel_block_size). kernel_block_size is read straight off the tensor.
        self._cap_groups = []
        for meta in self._group_metas:
            if isinstance(meta, AttentionGroupMeta):
                ref = kv_caches[meta.layer_names[0]]
                kernel_bs = attn_kv_views(ref)[0][0].shape[1]
            else:
                kernel_bs = 0
            self._cap_groups.append(
                (meta.group_idx, isinstance(meta, AttentionGroupMeta),
                 list(meta.layer_names), meta.block_size, kernel_bs))

        # Track completion of async load scatters. The parent's load task already
        # CPU-synchronizes its own scatter before reporting the task result, so a
        # threading.Event set from the done callback is sufficient to know the
        # scatter is globally visible.
        self._load_done_events: dict[str, list[threading.Event]] = {}
        self._load_events_lock = threading.Lock()

        # Loads are async and their step has no forward pass, so the worker does
        # not yet have the request's token ids. Record the load's per-group block
        # tables here and emit the capture once the token ids arrive.
        # req_id -> (manager_block_idxes, block_ids_per_group)
        self._pending_loaded: dict[str, tuple[list, list]] = {}

        orig_factory = self._data_transfer.create_load_done_callback

        def tracking_factory(req_id, *args, **kwargs):
            orig_cb = orig_factory(req_id, *args, **kwargs)
            evt = threading.Event()
            with self._load_events_lock:
                self._load_done_events.setdefault(req_id, []).append(evt)

            def cb(task_results):
                try:
                    orig_cb(task_results)
                finally:
                    evt.set()

            return cb

        self._data_transfer.create_load_done_callback = tracking_factory
        logger.warning(
            "VerifyingConnector enabled, capture_dir=%s tp_rank=%s groups=%s "
            "vllm_bs=%s manager_bs=%s",
            self._capture_dir, self._tp_rank,
            [(g[0], "attn" if g[1] else "state", g[3], g[4]) for g in self._cap_groups],
            self._vllm_block_size, self._manager_block_size,
        )

    # ------------------------------------------------------------------ #
    # Load hook: record pending loaded captures
    # ------------------------------------------------------------------ #
    def start_load_kv(self, forward_context, **kwargs) -> None:
        meta = typing.cast(TairKvCacheConnectorMetadata, self._get_connector_metadata())
        load_reqs = [
            (lr.req_id, list(lr.manager_block_idxes),
             [list(b) for b in lr.all_block_ids])
            for lr in meta.to_load_requests
            if lr.all_block_ids and lr.need_load_locations
        ]

        super().start_load_kv(forward_context, **kwargs)

        if getattr(self, "_capture_dir", "") and load_reqs:
            for req_id, mbis, bpg in load_reqs:
                self._pending_loaded[req_id] = (mbis, bpg)
            logger.warning(
                "VerifyingConnector recorded %d pending loaded capture(s)",
                len(load_reqs))

    # ------------------------------------------------------------------ #
    # Scheduler-side: ship token snapshots for the worker's captures
    # ------------------------------------------------------------------ #
    def build_connector_meta(self, scheduler_output):
        meta = super().build_connector_meta(scheduler_output)
        # The captures below identify a block by its token content; the
        # worker no longer mirrors token streams (self-contained
        # instructions), so hand it the live streams from the ledger.
        scheduler = self.connector_scheduler
        meta.token_snapshots = {
            req_id: ledger.vllm_request.all_token_ids
            for req_id, ledger in scheduler._tracked.items()
        }
        return meta

    # ------------------------------------------------------------------ #
    # Save hook: reference captures + emit pending loaded captures
    # ------------------------------------------------------------------ #
    def wait_for_save(self):
        meta = typing.cast(TairKvCacheConnectorMetadata, self._get_connector_metadata())

        if getattr(self, "_capture_dir", "") and getattr(self, "_kv_caches", None):
            try:
                self._capture_refs(meta)
                self._capture_pending_loaded(meta)
            except Exception as e:  # never break inference for a capture error
                logger.warning("VerifyingConnector capture failed: %s", e, exc_info=True)

        super().wait_for_save()

    def _capture_refs(self, meta: TairKvCacheConnectorMetadata):
        if not meta.to_save_requests:
            return
        # Make all forward-pass KV writes visible before reading the paged cache.
        self._device_mod.synchronize()
        tokens = getattr(meta, "token_snapshots", {})
        for save_req in meta.to_save_requests:
            token_ids = tokens.get(save_req.req_id)
            if token_ids is None or not save_req.all_block_ids:
                continue
            self._capture_range(
                kind="ref",
                token_ids=token_ids,
                block_ids_per_group=save_req.all_block_ids,
                manager_block_idxes=save_req.manager_block_idxes,
            )

    def _capture_pending_loaded(self, meta):
        if not self._pending_loaded:
            return
        done = []
        tokens = getattr(meta, "token_snapshots", {})
        for req_id, (mbis, bpg) in self._pending_loaded.items():
            token_ids = tokens.get(req_id)
            if token_ids is None:
                # token ids have not arrived on this worker yet; wait for a
                # later step in which the request is scheduled.
                continue
            with self._load_events_lock:
                evts = list(self._load_done_events.get(req_id, []))
            for evt in evts:
                evt.wait(timeout=120)
            self._device_mod.synchronize()
            self._capture_range(
                kind="loaded",
                token_ids=token_ids,
                block_ids_per_group=bpg,
                manager_block_idxes=mbis,
            )
            done.append(req_id)
        for req_id in done:
            del self._pending_loaded[req_id]

    # ------------------------------------------------------------------ #
    # Capture helpers
    # ------------------------------------------------------------------ #
    def _capture_range(self, kind, token_ids, block_ids_per_group, manager_block_idxes):
        if not manager_block_idxes or not block_ids_per_group:
            return
        # One record per manager block. Saves are batched incrementally while
        # loads arrive all-at-once, so per-block records let the driver match
        # reference vs loaded captures by each block's token content.
        for b in manager_block_idxes:
            self._capture_block(kind, token_ids, block_ids_per_group, b)

    def _attn_token_slot(self, pos, block_table, group_bs, kernel_bs):
        """Map a global token position to its flat slot in an attention group.

        Mirrors vLLM's own slot_mapping kernel expressed with the three-tier
        block hierarchy (group logical block -> kernel physical block). Works for
        pure-attention groups (group_bs == kernel_bs, ratio 1) and hybrid
        attention groups (group block larger than kernel block). Independent of
        the connector's step-1 (manager-block) logic, which is what we verify.
        """
        ratio = group_bs // kernel_bs
        logical = pos // group_bs
        off = pos % group_bs
        physical = block_table[logical] * ratio + off // kernel_bs
        return physical * kernel_bs + off % kernel_bs

    def _capture_block(self, kind, token_ids, block_ids_per_group, manager_block_idx):
        mbs = self._manager_block_size

        # Global token positions covered by this manager block.
        positions = list(range(manager_block_idx * mbs, (manager_block_idx + 1) * mbs))
        if positions[-1] >= len(token_ids):
            positions = [p for p in positions if p < len(token_ids)]
        if not positions:
            return

        captured_token_ids = [token_ids[p] for p in positions]
        kv_by_layer = {}

        for group_idx, is_attention, layer_names, group_bs, kernel_bs in self._cap_groups:
            block_table = block_ids_per_group[group_idx]
            if is_attention:
                slots = [self._attn_token_slot(p, block_table, group_bs, kernel_bs)
                         for p in positions]
                slot_tensor = torch.tensor(slots, dtype=torch.long, device=self._device)
                for layer_name in layer_names:
                    kv_cache = self._kv_caches[layer_name]
                    # Normalize the layout (packed 4-D or split K/V 5-D) into
                    # token-major views via the production helper and gather the
                    # whole per-token vector by (block, token) advanced indexing
                    # -- split K/V views are non-contiguous, so flattening them
                    # first would copy the entire cache tensor. Split views are
                    # concatenated on the content dim, so a capture is
                    # comparable across save/load within one run.
                    parts = []
                    for v in attn_kv_views(kv_cache)[0]:
                        blk = slot_tensor // kernel_bs
                        tok = slot_tensor % kernel_bs
                        parts.append(v[blk, tok].reshape(len(slots), -1))
                    gathered = (parts[0] if len(parts) == 1
                                else torch.cat(parts, dim=-1)).contiguous()
                    kv_by_layer[layer_name] = gathered.cpu()
            else:
                # State stored once per group block; the manager block's last
                # token selects the block (mirrors _state_block_ids). vLLM's
                # mamba "align" mode materializes states only at segment
                # boundaries -- interior blocks hold the null block (id 0) and
                # carry no state to capture (the connector skips them too).
                logical = ((manager_block_idx + 1) * mbs - 1) // group_bs
                block_id = block_table[logical]
                if block_id == 0:
                    continue
                for layer_name in layer_names:
                    states = self._kv_caches[layer_name]  # list[Tensor]
                    kv_by_layer[layer_name] = [s[block_id].detach().cpu() for s in states]

        token_hash = hashlib.sha256(
            torch.tensor(captured_token_ids, dtype=torch.int64).numpy().tobytes()
        ).hexdigest()[:16]
        path = os.path.join(self._capture_dir, f"{kind}_tp{self._tp_rank}_{token_hash}.pt")
        torch.save({"token_ids": captured_token_ids, "kv": kv_by_layer}, path)
        logger.warning(
            "VerifyingConnector captured %s block=%d tokens=%d..%d tp=%s -> %s",
            kind, manager_block_idx, positions[0], positions[-1], self._tp_rank, path)


class MutatedWorkerCore(ConnectorWorker):
    """Off-by-one in the attention token translation (see MutatedConnector)."""

    def _attn_token_indices(self, group, manager_block_idxes, block_table):
        out = super()._attn_token_indices(group, manager_block_idxes, block_table)
        return [[slot - 1 for slot in block] for block in out]


class MutatedConnector(VerifyingConnector):
    """Meta-test connector: injects an off-by-one into the attention token
    translation (every gathered/scattered slot shifted by -1).

    The shift is symmetric between save and load, so with contiguous block
    tables a transport round trip cancels it in the interior of the loaded
    range (slot(t)-1 == slot(t-1)); the leak is at the boundary: the last
    loaded token's true slot is never written and keeps stale (uninitialized)
    data. The capture-based verification reads the cache through vLLM's own
    slot mapping and must observe that divergence -- the mutation e2e test
    asserts that verification FAILS with this connector.

    -1 (not +1) keeps every shifted slot in bounds: vLLM reserves physical
    block 0 as the null block, so real slots are >= kernel_block_size and
    slot-1 >= 0, while slot+1 of the cache's last block would read/write out
    of bounds. Only reachable through the test-side ``kv_connector_module_path``
    injection; never part of the production wheel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The translation lives on the ConnectorWorker since the role split;
        # swap the worker-role instance (state included) for the mutated
        # subclass. Scheduler-role instances have no worker to mutate.
        if self.connector_worker is None:
            return
        mutated = MutatedWorkerCore.__new__(MutatedWorkerCore)
        mutated.__dict__.update(self.connector_worker.__dict__)
        self.connector_worker = mutated
