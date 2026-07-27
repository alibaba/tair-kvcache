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


@dataclass
class LoadRequest:
    req_id: str
    manager_block_idxes: List[int]
    need_load_locations: List[dict]
    # Per-group block tables: all_block_ids[group_idx] is the list of local block
    # ids for that kv_cache_group. Length 1 for pure-attention models.
    all_block_ids: List[List[int]] = field(default_factory=list)


@dataclass
class FinishRequest:
    req_id: str


@dataclass
class ReqStateToWorker:
    """Scheduler -> worker per-request state delta."""

    req_id: str
    has_saved_block_num: int
    new_tokens_ids: list = field(default_factory=list)
    # Per-group new local block ids (indexed by kv_cache_group).
    new_block_ids_per_group: List[List[int]] = field(default_factory=list)
    resumed_from_preemption: bool = False
    is_delta: bool = True


@dataclass
class TairKvCacheConnectorMetadata(KVConnectorMetadata):
    """Scheduler -> worker metadata for one engine step."""

    def __init__(self, epoch: int):
        self.epoch = epoch
        self.requests: List[ReqStateToWorker] = []
        self.to_load_requests: List[LoadRequest] = []
        self.to_save_requests: List[SaveRequest] = []
        self.to_finish_requests: List[FinishRequest] = []

    def add_req_state_to_worker(self, request: ReqStateToWorker):
        self.requests.append(request)

    def add_load_request(self, request: LoadRequest):
        self.to_load_requests.append(request)

    def add_save_request(self, save_request: SaveRequest):
        self.to_save_requests.append(save_request)

    def add_finish_request(self, finish_request: FinishRequest):
        self.to_finish_requests.append(finish_request)

    def __repr__(self):
        return (f"TairKvCacheConnectorMetadata(epoch={self.epoch}, "
                f"requests={len(self.requests)}, load={len(self.to_load_requests)}, "
                f"save={len(self.to_save_requests)}, finish={len(self.to_finish_requests)})")
