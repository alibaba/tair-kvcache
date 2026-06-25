import orjson
import threading
import time
from dataclasses import dataclass, field
from typing import List, Any, Union, Dict

import zmq

from kv_cache_manager.py_connector.common.logger import logger


@dataclass
class SendBlockStartEvent:
    request_id: str
    write_session_id: str
    locations: list = field(default_factory=list)
    type: str = "SendBlockStartEvent"


@dataclass
class SendBlockFinishedEvent:
    request_id: str
    tp_rank: int
    write_session_id: str
    is_success_list: List[bool] = field(default_factory=list)
    remote_uris: List[str] = field(default_factory=list)
    type: str = "SendBlockFinishedEvent"


@dataclass
class LoadBlockFinishedEvent:
    request_id: str
    tp_rank: int
    epoch: int
    failed_block_idxs: List[int] = field(default_factory=list)
    type: str = "LoadBlockFinishedEvent"


CoordinateEventUnion = Union[SendBlockFinishedEvent, LoadBlockFinishedEvent, SendBlockStartEvent]


@dataclass
class CoordinateMessage:
    send_time: float
    content: CoordinateEventUnion
    create_time: float = field(default_factory=time.time)


msg_type_mapping = {
    "SendBlockStartEvent": SendBlockStartEvent,
    "SendBlockFinishedEvent": SendBlockFinishedEvent,
    "LoadBlockFinishedEvent": LoadBlockFinishedEvent,
}


class CoordinateMsgSerializer:
    @staticmethod
    def dumps(obj: CoordinateMessage) -> bytes:
        return orjson.dumps(obj)

    @staticmethod
    def loads(input_bytes: bytes) -> CoordinateMessage:
        msg_dict = orjson.loads(input_bytes)
        msg_type_str = msg_dict["content"]["type"]
        msg_type = msg_type_mapping[msg_type_str]
        content = msg_type(**msg_dict["content"])
        return CoordinateMessage(
            send_time=msg_dict["send_time"],
            create_time=msg_dict["create_time"],
            content=content
        )


@dataclass(frozen=True)
class RunningId:
    write_session_id: str
    request_id: str


@dataclass()
class LoadContext:
    finished_rank: set = field(default_factory=set)

    def add_new_rank(self, tp_rank):
        if tp_rank in self.finished_rank:
            return
        self.finished_rank.add(tp_rank)

    def get_size(self):
        return len(self.finished_rank)


@dataclass
class SaveContext:
    locations: list
    result_per_rank: Dict[int, List[Any]] = field(default_factory=dict)
    success_mask: List[bool] = field(default_factory=list)

    def add_new_rank(self, tp_rank, is_successes):
        if tp_rank in self.result_per_rank:
            return
        self.result_per_rank[tp_rank] = is_successes

    def get_size(self):
        return len(self.result_per_rank)


class TpCoordinatorServer:
    def __init__(self, host_ip: str, base_port: int, tp_world_size: int, on_finished_callback):
        self._host_ip = host_ip
        self._base_port = base_port
        self._tp_world_size = tp_world_size

        self._coordinator_running = True
        self._coordinator_thread = threading.Thread(target=self.coordinator_routine, daemon=True)
        self._coordinator_thread.start()

        self._finished_loading = []
        self._failed_loading_block_idxs = []
        self._finished_loading_lock = threading.Lock()
        self._finished_saving = []
        self._finished_saving_lock = threading.Lock()
        self._on_finished_callback = on_finished_callback

    def coordinator_routine(self):
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        port = self._base_port
        socket.bind("tcp://%s:%d" % (self._host_ip, port))
        logger.warning("TairKvCacheConnector coordinator started")

        running_load: Dict[RunningId, LoadContext] = {}
        running_save: Dict[RunningId, SaveContext] = {}
        # Buffer for SendBlockFinishedEvent that arrives before SendBlockStartEvent
        pending_save_finishes: Dict[RunningId, List[SendBlockFinishedEvent]] = {}

        def complete_save_if_ready(save_id: RunningId, write_session_id: str, request_id: str):
            save_context = running_save.get(save_id)
            if save_context is None:
                return
            if save_context.get_size() != self._tp_world_size:
                return
            save_context = running_save.pop(save_id)
            self._on_finished_callback(write_session_id, save_context)
            with self._finished_saving_lock:
                self._finished_saving.append(request_id)

        while self._coordinator_running:
            raw_msg = socket.recv()
            # logger.warning("[coordinator] received msg: %s", raw_msg)
            try:
                msg = CoordinateMsgSerializer.loads(raw_msg)
            except Exception as e:
                logger.warning("[coordinator] received msg load failed: %s, error: %s", raw_msg, e)
                continue

            if isinstance(msg.content, SendBlockStartEvent):
                content: SendBlockStartEvent = msg.content
                save_id = RunningId(content.write_session_id, content.request_id)
                if save_id not in running_save:
                    running_save[save_id] = SaveContext(content.locations)

                # Merge any buffered finish events that arrived before this start
                if save_id in pending_save_finishes:
                    for finish_event in pending_save_finishes.pop(save_id):
                        running_save[save_id].add_new_rank(finish_event.tp_rank, finish_event.is_success_list)

                complete_save_if_ready(save_id, content.write_session_id, content.request_id)

            elif isinstance(msg.content, SendBlockFinishedEvent):
                content: SendBlockFinishedEvent = msg.content
                save_id = RunningId(content.write_session_id, content.request_id)

                if save_id not in running_save:
                    # Start event hasn't arrived yet, buffer this finish
                    if save_id not in pending_save_finishes:
                        pending_save_finishes[save_id] = []
                    pending_save_finishes[save_id].append(content)
                else:
                    running_save[save_id].add_new_rank(content.tp_rank, content.is_success_list)
                    complete_save_if_ready(save_id, content.write_session_id, content.request_id)
            elif isinstance(msg.content, LoadBlockFinishedEvent):
                content: LoadBlockFinishedEvent = msg.content
                load_id = RunningId(str(content.epoch), content.request_id)
                if load_id not in running_load:
                    running_load[load_id] = LoadContext()
                running_load[load_id].add_new_rank(content.tp_rank)
                if running_load[load_id].get_size() == self._tp_world_size:
                    running_load.pop(load_id)

                    self._finished_loading_lock.acquire()
                    self._failed_loading_block_idxs.extend(content.failed_block_idxs)
                    self._finished_loading.append(content.request_id)
                    self._finished_loading_lock.release()
                    # logger.warning("[coordinator] all rank finished load_blocks_finished %s", load_id)
            else:
                logger.warning("[coordinator] received wrong msg: %s", msg)

    def get_finished_tasks(self):
        finished_saving = []
        finished_loading = []
        if len(self._finished_loading) > 0:
            self._finished_loading_lock.acquire()
            finished_loading = self._finished_loading.copy()
            self._finished_loading = []
            self._finished_loading_lock.release()
        if len(self._finished_saving) > 0:
            self._finished_saving_lock.acquire()
            finished_saving = self._finished_saving.copy()
            self._finished_saving = []
            self._finished_saving_lock.release()
        return finished_saving, finished_loading

    def get_failed_loading_block_idxs(self):
        if len(self._failed_loading_block_idxs) == 0:
            return set()
        self._finished_loading_lock.acquire()
        failed_loading_block_idxs = self._failed_loading_block_idxs.copy()
        self._failed_loading_block_idxs = []
        self._finished_loading_lock.release()
        return set(failed_loading_block_idxs)


class TpCoordinatorClient:
    def __init__(self, host_ip, port):
        self._host_ip = host_ip
        self._port = port
        self._lock = threading.Lock()

        self._zmq_context = zmq.Context()
        # all rank need report finishing (include rank0)
        socket = self._zmq_context.socket(zmq.PUSH)
        socket.connect("tcp://%s:%d" % (self._host_ip, self._port))
        self._socket = socket

    def send(self, param: bytes):
        with self._lock:
            self._socket.send(param)
