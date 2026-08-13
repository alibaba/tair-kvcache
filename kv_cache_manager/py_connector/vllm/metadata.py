from dataclasses import dataclass, field
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class SaveRequest:
    req_id: str
    target_locations: list[dict]
    manager_block_idxes: list
    write_session_id: str


@dataclass()
class LoadRequest:
    req_id: str
    manager_block_idxes: list
    need_load_locations: list[dict]
    local_block_ids: list = field(default_factory=list)


@dataclass()
class FinishRequest:
    req_id: str


@dataclass
class ReqStateToWorker:
    """发送给工作节点的请求状态数据结构"""

    req_id: str
    has_saved_block_num: int
    new_tokens_ids: list = field(default_factory=list)
    new_local_block_ids: list = field(default_factory=list)
    resumed_from_preemption: bool = False
    is_delta: bool = True

@dataclass
class TairKvCacheConnectorMetadata(KVConnectorMetadata):
    """TairKvCacheConnector的元数据类，用于在调度器和工作节点之间传递状态"""
    requests: list[ReqStateToWorker]

    def __init__(self, epoch: int):
        """
       初始化元数据

       Args:
           epoch: 当前epoch编号
       """
        self.epoch = epoch
        self.requests: list[ReqStateToWorker] = []
        self.to_load_requests: list[LoadRequest] = []
        self.to_save_requests: list[SaveRequest] = []
        self.to_finish_requests: list[FinishRequest] = []

    def add_req_state_to_worker(self, request: ReqStateToWorker):
        self.requests.append(request)

    def add_load_request(self, request: LoadRequest):
        self.to_load_requests.append(request)

    def add_save_request(self, save_request: SaveRequest):
        self.to_save_requests.append(save_request)

    def add_finish_request(self, finish_request: FinishRequest):
        self.to_finish_requests.append(finish_request)

    def __repr__(self):
        return f"TairKvCacheConnectorMetadata(requests={self.requests})"

