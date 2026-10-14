import orjson
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Union

import zmq

from kv_cache_manager.py_connector.common.logger import logger


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
    # Union of the per-rank failed vLLM block ids. A rank whose loads all
    # succeed reports an empty list, so keeping only the last-arriving
    # event's list silently dropped the other ranks' failures whenever a
    # success event arrived last -- at tp >= 2 a partially failed load was
    # then never reported to vLLM, which went on to compute on unwritten
    # KV blocks.
    failed_block_idxs: List[int] = field(default_factory=list)

    def add_new_rank(self, tp_rank: int, failed_block_idxs: List[int]) -> None:
        if tp_rank in self.finished_rank:
            return
        self.finished_rank.add(tp_rank)
        self.failed_block_idxs.extend(failed_block_idxs)

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
        # The cross-rank AND of a session, once every rank reported; the
        # worker-side settle asks for it (see TpCoordinatorClient.ask_verdict):
        # a rank whose write is not the global verdict must keep its capture
        # until this answer says "committed", or drop it when the answer says
        # a rank failed. ``None`` means "not decided yet".
        self._verdicts: Dict[str, List[bool]] = {}
        # Ranks that already asked for a given session's verdict: a verdict is
        # only dropped once every rank has consumed it, because the coordinator
        # cannot reconstruct it and a rank that asks later must not be told
        # "undecided" forever (it would keep its captures until retirement).
        self._verdict_acks: Dict[str, Set[int]] = {}
        self._verdicts_lock = threading.Lock()

        self._coordinator_running = True
        self._coordinator_thread = threading.Thread(
            target=self.coordinator_routine, daemon=True
        )
        self._coordinator_thread.start()
        # The verdict query rides a second socket (REQ/REP): the event stream
        # above is one-way (PUSH/PULL), and a query needs an answer. A port
        # that cannot be bound simply disables the active query -- the
        # captures then wait for the manager's mask or the request's
        # retirement, which is the pre-query behaviour (slower, still sound).
        self._query_port = base_port + 1
        self._query_thread = threading.Thread(
            target=self.verdict_query_routine, daemon=True
        )
        self._query_thread.start()

        self._finished_loading = []
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
        logger.warning("TairKvCacheConnector coordinator started")

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
                load_ctx = running_load[load_id]
                load_ctx.add_new_rank(content.tp_rank, content.failed_block_idxs)
                if load_ctx.get_size() == self._tp_world_size:
                    running_load.pop(load_id)

                    self._finished_loading_lock.acquire()
                    self._failed_loading_block_idxs.extend(load_ctx.failed_block_idxs)
                    self._finished_loading.append(content.request_id)
                    self._finished_loading_lock.release()
                    # logger.warning("[coordinator] all rank finished load_blocks_finished %s", load_id)
            else:
                logger.warning("[coordinator] received wrong msg: %s", msg)

    def verdict_query_routine(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        try:
            socket.bind("tcp://%s:%d" % (self._host_ip, self._query_port))
        except zmq.ZMQError as e:
            logger.warning(
                "TairKvCacheConnector verdict query disabled (port %d: %s); "
                "captures will wait for the manager's mask instead",
                self._query_port,
                e,
            )
            socket.close(linger=0)
            context.term()
            return
        logger.warning("TairKvCacheConnector verdict query started")
        while self._coordinator_running:
            try:
                raw = socket.recv()
            except zmq.ZMQError:  # pragma: no cover - context teardown
                return
            try:
                query = orjson.loads(raw)
                session_id, tp_rank = query["session_id"], query["tp_rank"]
                cancel = bool(query.get("cancel", False))
            except Exception:  # pragma: no cover - protocol misuse
                socket.send(b"")
                continue
            if cancel:
                self.cancel_query(session_id, tp_rank)
                socket.send(b"")
                continue
            verdict = self.answer_query(session_id, tp_rank)
            # b"" = not decided yet (the querying rank rechecks next step).
            socket.send(orjson.dumps(verdict) if verdict is not None else b"")

    def cancel_query(self, session_id: str, tp_rank: int) -> None:
        """This rank will never ask for the session's verdict.

        A request can retire on a fast rank while slower ranks are still
        writing it: the retiring rank drops its local obligation, so without
        this the entry waits for an acknowledgement that cannot come any more.
        Cancelling counts as the rank's answer, which lets the entry go once
        every rank is done.
        """
        with self._verdicts_lock:
            if session_id not in self._verdicts:
                return
            acked = self._verdict_acks.setdefault(session_id, set())
            acked.add(tp_rank)
            if len(acked) >= self._tp_world_size:
                self._verdicts.pop(session_id, None)
                self._verdict_acks.pop(session_id, None)

    def answer_query(self, session_id: str, tp_rank: int) -> "Optional[List[bool]]":
        """The stored verdict, acknowledging it for this rank.

        Nothing is dropped until every rank has consumed the mask: the
        coordinator cannot rebuild one, and a rank that asks afterwards would
        be told "undecided" forever, keeping its captures until retirements and
        draining the arena.
        """
        with self._verdicts_lock:
            verdict = self._verdicts.get(session_id)
            if verdict is None:
                return None
            acked = self._verdict_acks.setdefault(session_id, set())
            acked.add(tp_rank)
            if len(acked) >= self._tp_world_size:
                self._verdicts.pop(session_id, None)
                self._verdict_acks.pop(session_id, None)
            return verdict

    def record_verdict(self, write_session_id: str, success_mask: List[bool]) -> None:
        with self._verdicts_lock:
            self._verdicts[write_session_id] = list(success_mask)
            self._verdict_acks.setdefault(write_session_id, set())
            if len(self._verdicts) > 4096:
                # A rank that never asks and never cancels (it died, or its
                # request was never retired) would pin masks forever. Below
                # the cap nothing is dropped before every rank had its answer;
                # above it the oldest entries go, fully acknowledged ones
                # first, and either way loudly.
                stale = [
                    session for session in self._verdicts if not self._verdict_acks.get(session)
                ] + [session for session in self._verdicts if self._verdict_acks.get(session)]
                for session in stale[: len(self._verdicts) - 4096]:
                    self._verdicts.pop(session, None)
                    self._verdict_acks.pop(session, None)
                logger.warning(
                    "TairKvCacheConnector: %d verdicts retained; evicting the "
                    "oldest (a rank neither asked nor cancelled)",
                    len(self._verdicts),
                )

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

        self._zmq_context = zmq.Context()
        # all rank need report finishing (include rank0)
        socket = self._zmq_context.socket(zmq.PUSH)
        socket.connect("tcp://%s:%d" % (self._host_ip, self._port))
        self._socket = socket
        # The verdict query (see TpCoordinatorServer.verdict_query_routine):
        # REQ/REP on the coordinator's second port, created lazily so a
        # deployment without it pays nothing.
        self._query_socket: Any = None

    def send(self, param: bytes) -> None:
        with self._lock:
            self._socket.send(param)

    def cancel_verdict(self, write_session_id: str, tp_rank: int) -> None:
        """Tell the coordinator this rank will not ask for this session.

        Best effort: if the query socket is unavailable the entry simply waits
        for a rank that never comes -- which is what the cap is for.
        """
        self._ask_query(
            {"session_id": write_session_id, "tp_rank": tp_rank, "cancel": True}
        )

    def _send_query(self, payload: Dict[str, Any], timeout_ms: int = 100) -> Any:
        """One request/response over the query socket (caller holds the lock)."""
        if self._query_socket is None:
            socket = self._zmq_context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            socket.connect("tcp://%s:%d" % (self._host_ip, self._port + 1))
            self._query_socket = socket
        self._query_socket.send(orjson.dumps(payload))
        return self._query_socket.recv()

    def _ask_query(self, payload: Dict[str, Any], timeout_ms: int = 100) -> Any:
        with self._lock:
            try:
                return self._send_query(payload, timeout_ms)
            except zmq.ZMQError:
                if self._query_socket is not None:
                    self._query_socket.close(linger=0)
                    self._query_socket = None
                return False

    def ask_verdict(
        self, write_session_id: str, tp_rank: int = 0, timeout_ms: int = 100
    ) -> "Union[None, List[bool], bool]":
        """The coordinator's global verdict for this session, if decided.

        Returns the per-block AND mask (a list of bools), ``None`` while some
        rank has not reported yet, or ``False`` when the coordinator cannot be
        asked (no query socket, or a timeout): the caller then keeps its bytes
        and lets the retirement (or the manager's mask) close the loop.
        """
        raw = self._ask_query(
            {"session_id": write_session_id, "tp_rank": tp_rank}, timeout_ms
        )
        if raw is False:
            return False
        if not raw:
            return None
        return list(orjson.loads(raw))
