from dataclasses import dataclass, field
from typing import List

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
class TairKvCacheConnectorMetadata(KVConnectorMetadata):
    """Scheduler -> worker metadata for one engine step.

    Three instruction kinds, each consumed by its own worker hook:
    to_load_requests (start_load_kv), to_save_requests (wait_for_save) and
    to_finish_requests (get_finished). Every instruction is self-contained
    -- the worker keeps no per-request mirror of scheduler state."""

    def __init__(self, epoch: int):
        self.epoch = epoch
        self.to_load_requests: List[LoadRequest] = []
        self.to_save_requests: List[SaveRequest] = []
        self.to_finish_requests: List[FinishRequest] = []

    def add_load_request(self, request: LoadRequest):
        self.to_load_requests.append(request)

    def add_save_request(self, save_request: SaveRequest):
        self.to_save_requests.append(save_request)

    def add_finish_request(self, finish_request: FinishRequest):
        self.to_finish_requests.append(finish_request)

    def __repr__(self):
        return (f"TairKvCacheConnectorMetadata(epoch={self.epoch}, "
                f"load={len(self.to_load_requests)}, "
                f"save={len(self.to_save_requests)}, "
                f"finish={len(self.to_finish_requests)})")
