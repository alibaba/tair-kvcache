from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class SaveRequest:
    req_id: str
    # CacheLocation dicts returned by the manager (one per manager block to save),
    # each carrying location_specs for every registered spec name.
    target_locations: List[dict]
    # Manager block indices (into the request's token stream) being saved.
    manager_block_idxes: List[int]
    write_session_id: str
    # Per-group block table snapshot taken by the scheduler when the save
    # instruction is handed to the worker: the gather needs it to translate
    # manager blocks into physical slots, and the scheduler's ledger is the
    # only place it exists. The worker keeps no request mirror.
    all_block_ids: List[List[int]] = field(default_factory=list)
    # The scheduler's id of the write session this request belongs to. The
    # worker does not read it (its own bookkeeping is per block); the scheduler
    # uses it to retire the session's capture protection when the request is
    # handed over (see _collect_save_instructions).
    session_id: int = 0


@dataclass
class LoadRequest:
    req_id: str
    manager_block_idxes: List[int]
    need_load_locations: List[dict]
    # Per-group block tables: all_block_ids[group_idx] is the list of local
    # block ids for that kv_cache_group (filled by the scheduler once vLLM
    # allocated the blocks). Length 1 for pure-attention models.
    all_block_ids: List[List[int]] = field(default_factory=list)


@dataclass
class FinishRequest:
    req_id: str


@dataclass
class CaptureRequest:
    """Copy one request's newly completed manager blocks out of HBM *now*.

    Windowed groups (see vllm_common.WindowRetention) lose their blocks to
    vLLM's window sliding in the *next* schedule step, so the scheduler asks
    the worker to copy the bytes in the step whose forward produced them --
    before the write session (and with it the remote addresses) exists. The
    worker keeps the captured bytes in its pinned staging arena and the save
    task writes them once the session lands, so the announce -> session lag
    never touches HBM again.

    ``all_block_ids`` is the scheduler's block-table snapshot, taken in the
    same scheduler loop iteration: the physical ids are still valid at capture
    time (the free happens in the next schedule), which is what makes this
    snapshot -- not a later ledger read -- the trustworthy null/data source.
    """

    req_id: str
    # Manager blocks to capture, in ascending order.
    manager_block_idxes: List[int]
    all_block_ids: List[List[int]] = field(default_factory=list)


@dataclass
class TairKvCacheConnectorMetadata(KVConnectorMetadata):
    """Scheduler -> worker metadata for one engine step.

    Five instruction kinds, each consumed by its own worker hook:
    to_load_requests (start_load_kv), to_capture_requests + to_save_requests +
    to_release_captures (wait_for_save) and to_finish_requests (get_finished).
    Every instruction is self-contained -- the worker keeps no per-request
    mirror of scheduler state."""

    def __init__(self, epoch: int) -> None:
        self.epoch = epoch
        self.to_load_requests: List[LoadRequest] = []
        self.to_capture_requests: List[CaptureRequest] = []
        self.to_save_requests: List[SaveRequest] = []
        self.to_finish_requests: List[FinishRequest] = []
        self._release_captures: List[Tuple[str, List[int]]] = []

    def add_load_request(self, request: LoadRequest) -> None:
        self.to_load_requests.append(request)

    def add_capture_request(self, request: CaptureRequest) -> None:
        self.to_capture_requests.append(request)

    def add_save_request(self, save_request: SaveRequest) -> None:
        self.to_save_requests.append(save_request)

    def add_finish_request(self, finish_request: FinishRequest) -> None:
        self.to_finish_requests.append(finish_request)

    @property
    def to_release_captures(self) -> List[Tuple[str, List[int]]]:
        """(req_id, block indices) whose eager captures no session will write.

        Filled by the scheduler when a write session reports that the manager
        already holds those blocks: those captured bytes are not going to be
        published, so the worker gives their arena slots back now instead of
        holding them until the request retires."""
        return self._release_captures

    def release_captures(
        self,
        req_id: str,
        block_idxes: List[int],
        block_uris: Optional[Dict[int, List[str]]] = None,
    ) -> None:
        """Return the captures of blocks no write session will publish.

        ``block_uris`` carries, per manager block, the cache keys the manager
        reported for it (only where the release came from an "already cached"
        answer): the worker uses them to tell a *durable* copy from another
        request's uncommitted allocation before dropping captured bytes (see
        ConnectorWorker._release_captures_for).
        """
        self._release_captures.append((req_id, list(block_idxes), block_uris))

    def __repr__(self) -> str:
        return (
            f"TairKvCacheConnectorMetadata(epoch={self.epoch}, "
            f"load={len(self.to_load_requests)}, "
            f"capture={len(self.to_capture_requests)}, "
            f"save={len(self.to_save_requests)}, "
            f"finish={len(self.to_finish_requests)}, "
            f"release={len(self._release_captures)})"
        )
