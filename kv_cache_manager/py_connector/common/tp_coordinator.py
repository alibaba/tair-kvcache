import orjson
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import zmq

import os

from kv_cache_manager.py_connector.common.logger import logger

_TRACE = os.environ.get("KVCM_TRACE") == "1"

#: How many completed load ids the query socket keeps. Long enough for a rank
#: that polls a few steps late, short enough that a leak cannot grow.
_HISTORY = 8192
#: The query socket lives next to the PUSH/PULL one. Both ends derive it.
_QUERY_PORT_OFFSET = 1


@dataclass
class SendBlockStartEvent:
    request_id: str
    write_session_id: str
    # Manager CacheLocation dicts, one per block to save.
    locations: List[Dict[str, Any]] = field(default_factory=list)
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


CoordinateEventUnion = Union[
    SendBlockFinishedEvent, LoadBlockFinishedEvent, SendBlockStartEvent
]


@dataclass
class CoordinateMessage:
    send_time: float
    content: CoordinateEventUnion
    create_time: float = field(default_factory=time.time)


msg_type_mapping: Dict[str, type] = {
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
            content=content,
        )


@dataclass(frozen=True)
class RunningId:
    write_session_id: str
    request_id: str


@dataclass()
class LoadContext:
    finished_rank: Set[int] = field(default_factory=set)

    def add_new_rank(self, tp_rank: int) -> None:
        if tp_rank in self.finished_rank:
            return
        self.finished_rank.add(tp_rank)

    def get_size(self) -> int:
        return len(self.finished_rank)


@dataclass
class SaveContext:
    # Manager CacheLocation dicts, one per block to save.
    locations: List[Dict[str, Any]]
    result_per_rank: Dict[int, List[bool]] = field(default_factory=dict)
    success_mask: List[bool] = field(default_factory=list)

    def add_new_rank(self, tp_rank: int, is_successes: List[bool]) -> None:
        if tp_rank in self.result_per_rank:
            return
        self.result_per_rank[tp_rank] = is_successes

    def get_size(self) -> int:
        return len(self.result_per_rank)


class TpCoordinatorServer:
    def __init__(
        self,
        host_ip: str,
        base_port: int,
        tp_world_size: int,
        on_finished_callback: Callable[[str, SaveContext], None],
    ) -> None:
        self._host_ip = host_ip
        self._base_port = base_port
        self._tp_world_size = tp_world_size

        self._coordinator_running = True
        self._coordinator_thread = threading.Thread(
            target=self.coordinator_routine, daemon=True
        )
        self._coordinator_thread.start()

        self._finished_loading = []
        # A completion a rank may *ask* for, instead of only being told once.
        # The worker whose output the engine consumes is not known to this
        # process (vLLM takes `output[0]` from the executor, and under pipeline
        # parallelism that need not be global rank 0), so a single consumable
        # list would deliver the barrier to a rank whose report is dropped.
        # Bounded: a stale entry costs a lookup, not a leak.
        self._loading_history: List[str] = []
        self._loading_history_lock = threading.Lock()
        self._failed_loading_block_idxs = []
        self._finished_loading_lock = threading.Lock()
        self._finished_saving = []
        self._finished_saving_lock = threading.Lock()
        self._on_finished_callback = on_finished_callback

    def coordinator_routine(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        port = self._base_port
        socket.bind("tcp://%s:%d" % (self._host_ip, port))
        query = context.socket(zmq.REP)
        query.bind("tcp://%s:%d" % (self._host_ip, port + _QUERY_PORT_OFFSET))
        logger.warning("TairKvCacheConnector coordinator started")

        def serve_queries() -> None:
            """Answers "which loads are complete?" for every rank.

            A PULL socket hands each message to exactly one consumer, so the
            completion a PULL-based report consumed could never be re-read by
            another rank. This socket is how a rank that was *not* the one the
            engine listens to still learns the global barrier fired.
            """
            while True:
                try:
                    query.recv()
                    with self._loading_history_lock:
                        payload = "\n".join(self._loading_history)
                    query.send(payload.encode())
                except Exception as exc:  # noqa: BLE001 - a dead query socket is logged
                    logger.warning("[coordinator] query socket failed: %s", exc)
                    return

        threading.Thread(target=serve_queries, daemon=True).start()

        running_load: Dict[RunningId, LoadContext] = {}
        running_save: Dict[RunningId, SaveContext] = {}
        # Buffer for SendBlockFinishedEvent that arrives before SendBlockStartEvent
        pending_save_finishes: Dict[RunningId, List[SendBlockFinishedEvent]] = {}

        def complete_save_if_ready(
            save_id: RunningId, write_session_id: str, request_id: str
        ) -> None:
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
                logger.warning(
                    "[coordinator] received msg load failed: %s, error: %s", raw_msg, e
                )
                continue

            if isinstance(msg.content, SendBlockStartEvent):
                content: SendBlockStartEvent = msg.content
                save_id = RunningId(content.write_session_id, content.request_id)
                if save_id not in running_save:
                    running_save[save_id] = SaveContext(content.locations)

                # Merge any buffered finish events that arrived before this start
                if save_id in pending_save_finishes:
                    for finish_event in pending_save_finishes.pop(save_id):
                        running_save[save_id].add_new_rank(
                            finish_event.tp_rank, finish_event.is_success_list
                        )

                complete_save_if_ready(
                    save_id, content.write_session_id, content.request_id
                )

            elif isinstance(msg.content, SendBlockFinishedEvent):
                content: SendBlockFinishedEvent = msg.content
                save_id = RunningId(content.write_session_id, content.request_id)

                if save_id not in running_save:
                    # Start event hasn't arrived yet, buffer this finish
                    if save_id not in pending_save_finishes:
                        pending_save_finishes[save_id] = []
                    pending_save_finishes[save_id].append(content)
                else:
                    running_save[save_id].add_new_rank(
                        content.tp_rank, content.is_success_list
                    )
                    complete_save_if_ready(
                        save_id, content.write_session_id, content.request_id
                    )
            elif isinstance(msg.content, LoadBlockFinishedEvent):
                content: LoadBlockFinishedEvent = msg.content
                load_id = RunningId(str(content.epoch), content.request_id)
                if load_id not in running_load:
                    running_load[load_id] = LoadContext()
                    if _TRACE:
                        logger.info(
                            "[coordinator] load %s first rank %d of %d",
                            load_id,
                            content.tp_rank,
                            self._tp_world_size,
                        )
                running_load[load_id].add_new_rank(content.tp_rank)
                if _TRACE:
                    logger.info(
                        "[coordinator] load %s ranks %d/%d (new %d, epoch %s)",
                        load_id,
                        running_load[load_id].get_size(),
                        self._tp_world_size,
                        content.tp_rank,
                        content.epoch,
                    )
                if running_load[load_id].get_size() == self._tp_world_size:
                    running_load.pop(load_id)

                    self._finished_loading_lock.acquire()
                    self._failed_loading_block_idxs.extend(content.failed_block_idxs)
                    self._finished_loading.append(content.request_id)
                    self._finished_loading_lock.release()
                    with self._loading_history_lock:
                        self._loading_history.append(content.request_id)
                        del self._loading_history[:-_HISTORY]
                    # logger.warning("[coordinator] all rank finished load_blocks_finished %s", load_id)
            else:
                logger.warning("[coordinator] received wrong msg: %s", msg)

    def get_finished_tasks(self) -> Tuple[List[str], List[str]]:
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

    def get_failed_loading_block_idxs(self) -> Set[int]:
        if len(self._failed_loading_block_idxs) == 0:
            return set()
        self._finished_loading_lock.acquire()
        failed_loading_block_idxs = self._failed_loading_block_idxs.copy()
        self._failed_loading_block_idxs = []
        self._finished_loading_lock.release()
        return set(failed_loading_block_idxs)


class TpCoordinatorClient:
    def __init__(self, host_ip: str, port: int) -> None:
        self._host_ip = host_ip
        self._port = port
        self._lock = threading.Lock()
        #: The query socket is kept, not re-created per call: every rank asking
        #: every step over a fresh TCP connection is a fan-in the coordinator
        #: serializes, and it was measured to stall the engine (see the worker's
        #: load poller). `query_count` exists so a run can show that queries
        #: scale with loads and not with steps.
        self._query_lock = threading.Lock()
        self._query_socket: Optional[zmq.Socket] = None
        self.query_count = 0

        self._zmq_context = zmq.Context()
        # all rank need report finishing (include rank0)
        socket = self._zmq_context.socket(zmq.PUSH)
        socket.connect("tcp://%s:%d" % (self._host_ip, self._port))
        self._socket = socket

    def send(self, param: bytes) -> None:
        with self._lock:
            self._socket.send(param)

    def _query_connection(self, timeout_ms: int) -> "zmq.Socket":
        """The one query socket this rank keeps open."""
        if self._query_socket is None:
            socket = self._zmq_context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            socket.connect(
                "tcp://%s:%d" % (self._host_ip, self._port + _QUERY_PORT_OFFSET)
            )
            self._query_socket = socket
        return self._query_socket

    def finished_loads(self, timeout_ms: int = 1000) -> Set[str]:
        """The loads the coordinator has folded completely, or an empty set.

        Called by the worker's load poller, never from a step. One socket is
        reused for the process's life: a fresh REQ socket per call means a fresh
        TCP connection per call, and thirty-two ranks doing that every step is
        the fan-in that stalled the engine.
        """
        with self._query_lock:
            socket = self._query_connection(timeout_ms)
            try:
                socket.send(b"loads")
                self.query_count += 1
                payload = socket.recv().decode()
            except Exception:  # noqa: BLE001 - no answer is not an error
                # A REQ socket that timed out still owes a reply, so it cannot be
                # reused; drop it and let the next call reconnect.
                socket.close(linger=0)
                self._query_socket = None
                return set()
            return {line for line in payload.split("\n") if line}
