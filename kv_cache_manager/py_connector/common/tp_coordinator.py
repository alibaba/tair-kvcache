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

#: How many consecutive ports the coordinator may try when the configured one
#: is taken (a transient collision, typically TIME_WAIT after a restart).
COORDINATOR_PORT_ATTEMPTS = 9

#: How long a coordinator routine sleeps between polls, so shutdown() can stop
#: it without interrupting a blocked receive.
COORDINATOR_POLL_MS = 100

#: How many pre-verdict cancellations the coordinator remembers. They only
#: exist for sessions whose verdict is still to come (a rank that died, or a
#: commit that never landed) and are dropped oldest-first beyond this.
VERDICT_CANCEL_LIMIT = 4096


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
        # Ranks that gave up on a session before its verdict existed (a fast
        # rank retired the request while slower ranks were still writing it):
        # the cancel must not be dropped, or the verdict waits for an
        # acknowledgement that will never come.
        self._verdict_cancels: Dict[str, Set[int]] = {}
        self._verdicts_lock = threading.Lock()

        self._finished_loading = []
        self._failed_loading_block_idxs = []
        self._finished_loading_lock = threading.Lock()
        self._finished_saving = []
        self._finished_saving_lock = threading.Lock()
        self._on_finished_callback = on_finished_callback

        # Both sockets are bound *here*, before any thread starts, so a port
        # that cannot be taken fails the connector's initialization instead of
        # killing a background thread: a coordinator that never bound used to
        # leave the whole instance silently half-dead (no aggregations, no
        # finish_write_cache, saves never committed) until someone read the
        # logs. The configured port is tried first; a neighbour is only a
        # fallback for a transient collision (the harness allocates a free
        # port, but TIME_WAIT happens), and the port the coordinator actually
        # took is what it announces and what the other ranks probe for.
        self._context = zmq.Context()
        self._event_socket = self._context.socket(zmq.PULL)
        # LINGER 0 on every socket: a peer that never comes (the normal case in
        # tests) must not block the context's teardown, which is what turned a
        # clean test run into a 300 s timeout.
        self._event_socket.setsockopt(zmq.LINGER, 0)
        # The verdict query rides a second socket (REQ/REP): the event stream
        # is one-way (PUSH/PULL), and a query needs an answer. The two are
        # bound as a *pair* -- event B, query B+1 -- because that is the pair
        # the clients look for; chosen independently, a taken neighbour would
        # make the announced pair a lie and send every event to a dead port.
        self._query_socket = self._context.socket(zmq.REP)
        self._query_socket.setsockopt(zmq.LINGER, 0)
        self._base_port = self._bind_pair(base_port)
        self._query_port = self._base_port + 1
        logger.warning(
            "TairKvCacheConnector coordinator started: transfer_client_config "
            "coordinator_base_port=%d verdict_query_port=%d (pair)",
            self._base_port,
            self._query_port,
        )
        self._coordinator_running = True
        self._coordinator_thread = threading.Thread(
            target=self.coordinator_routine, daemon=True
        )
        self._coordinator_thread.start()
        self._query_thread = threading.Thread(
            target=self.verdict_query_routine, daemon=True
        )
        self._query_thread.start()

    def _bind_pair(self, base_port: int) -> int:
        """Bind the event and query sockets on ``candidate``/``candidate+1``.

        Both or neither: the clients derive the query port from the event port
        (``event + 1``), so a pair bound from two independent choices would
        answer on a port they never probe and take every event to a dead one.
        Failing over the whole range is deliberate (fail-closed): the
        alternative is a coordinator whose background thread dies on a bind
        error while the connector looks healthy, losing every save it makes.
        """
        last_error: Optional[Exception] = None
        for candidate in range(base_port, base_port + COORDINATOR_PORT_ATTEMPTS):
            try:
                self._event_socket.bind("tcp://%s:%d" % (self._host_ip, candidate))
            except zmq.ZMQError as e:
                last_error = e
                continue
            try:
                self._query_socket.bind(
                    "tcp://%s:%d" % (self._host_ip, candidate + 1)
                )
            except zmq.ZMQError as e:
                # Give the event port back and try the next pair: a half-bound
                # pair is exactly what the clients cannot find.
                last_error = e
                self._event_socket.close(linger=0)
                self._event_socket = self._context.socket(zmq.PULL)
                self._event_socket.setsockopt(zmq.LINGER, 0)
                continue
            return candidate
        # Nothing bound: release what is left before failing, so a rejected
        # initialization does not leave a context (and its IO thread) behind.
        for socket in (self._event_socket, self._query_socket):
            try:
                socket.close(linger=0)
            except Exception:  # pragma: no cover - already closed
                pass
        try:
            self._context.term()
        except Exception:  # pragma: no cover - already terminated
            pass
        raise RuntimeError(
            "TairKvCacheConnector: cannot bind the coordinator's event/query "
            "port pairs on %s:%d-%d (%s); another process holds those ports"
            % (
                self._host_ip,
                base_port,
                base_port + COORDINATOR_PORT_ATTEMPTS,
                last_error,
            )
        )

    def coordinator_routine(self) -> None:
        socket = self._event_socket

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
        socket = self._query_socket
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
                # The verdict is not published yet: remember the cancel so
                # record_verdict can apply it the moment it is.
                self._verdict_cancels.setdefault(session_id, set()).add(tp_rank)
                if len(self._verdict_cancels) > VERDICT_CANCEL_LIMIT:
                    # A rank that dies before the verdict is ever published
                    # would otherwise pin these entries forever: drop the
                    # oldest half, loudly, and keep serving the rest.
                    for stale in list(self._verdict_cancels)[: VERDICT_CANCEL_LIMIT // 2]:
                        self._verdict_cancels.pop(stale, None)
                    logger.warning(
                        "TairKvCacheConnector: dropped %d cancellations whose "
                        "session's verdict never arrived",
                        VERDICT_CANCEL_LIMIT // 2,
                    )
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
            acked = self._verdict_acks.setdefault(
                write_session_id, self._verdict_cancels.pop(write_session_id, set())
            )
            if len(acked) >= self._tp_world_size:
                # Every rank already gave up on it: no one will ever ask.
                self._verdicts.pop(write_session_id, None)
                self._verdict_acks.pop(write_session_id, None)
                return
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

    def stop(self) -> None:
        """Ask the routines to end (see shutdown for the full teardown)."""
        self._coordinator_running = False

    def shutdown(self) -> None:
        """Stop and release the sockets and the context.

        Order matters: the routines are stopped and their sockets closed
        first (LINGER 0, so the closes are immediate even with an undeliverable
        message queued), and only then is the context terminated -- tearing a
        context down under a blocked recv is what turns a clean run into a
        ContextTerminated storm. Only used by tests -- production runs for the
        life of the engine.
        """
        self.stop()
        for socket in (self._event_socket, self._query_socket):
            try:
                socket.close(linger=0)
            except Exception:  # pragma: no cover - already closed
                pass
        try:
            self._context.term()
        except Exception:  # pragma: no cover - already terminated
            pass

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
        # LINGER 0: a dead coordinator must not make the context's teardown
        # wait for a message nobody will ever receive.
        socket.setsockopt(zmq.LINGER, 0)
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
        """One request/response over the query socket (caller holds the lock).

        The coordinator may have had to bind a neighbouring port (a transient
        collision); the first unanswered query probes the range once and
        reconnects both sockets to whichever port answers.
        """
        if self._query_socket is None:
            self._connect_query(self._port + 1, timeout_ms)
        self._query_socket.send(orjson.dumps(payload))
        try:
            return self._query_socket.recv()
        except zmq.ZMQError:
            self._resync_endpoint(timeout_ms)
            raise

    def shutdown(self) -> None:
        """Release the client's sockets (tests only)."""
        if self._query_socket is not None:
            self._query_socket.close(linger=0)
            self._query_socket = None
        self._socket.close(linger=0)
        try:
            self._zmq_context.term()
        except Exception:  # pragma: no cover - already terminated
            pass
        self._zmq_context.term()

    def _connect_query(self, query_port: int, timeout_ms: int) -> None:
        socket = self._zmq_context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.connect("tcp://%s:%d" % (self._host_ip, query_port))
        self._query_socket = socket
        self._query_port = query_port

    def _resync_endpoint(self, timeout_ms: int = 100) -> None:
        """Find the coordinator's actual port after a failed query.

        Called at most once per outage: the port that answered is remembered
        (and both sockets move to it), so later queries do not probe again, and
        a still-dead coordinator is handed back to the caller, whose backoff
        keeps the engine thread off it.
        """
        if self._query_socket is not None:
            self._query_socket.close(linger=0)
            self._query_socket = None
        for candidate in range(self._port, self._port + COORDINATOR_PORT_ATTEMPTS):
            probe = self._zmq_context.socket(zmq.REQ)
            probe.setsockopt(zmq.LINGER, 0)
            probe.setsockopt(zmq.RCVTIMEO, min(timeout_ms, 20))
            probe.connect("tcp://%s:%d" % (self._host_ip, candidate + 1))
            answered = False
            try:
                probe.send(orjson.dumps({"probe": True}))
                probe.recv()
                answered = True
            except zmq.ZMQError:
                pass
            finally:
                probe.close(linger=0)
            if not answered:
                continue
            try:
                self._socket.disconnect(
                    "tcp://%s:%d" % (self._host_ip, self._port)
                )
            except zmq.ZMQError:  # pragma: no cover - never connected
                pass
            self._socket.connect("tcp://%s:%d" % (self._host_ip, candidate))
            logger.warning(
                "TairKvCacheConnector: the coordinator answered on port %d "
                "(configured %d was taken); reconnected",
                candidate,
                self._port,
            )
            self._port = candidate
            self._connect_query(candidate + 1, timeout_ms)
            return
        self._connect_query(self._port + 1, timeout_ms)

    def _ask_query(self, payload: Dict[str, Any], timeout_ms: int = 100) -> Any:
        with self._lock:
            try:
                return self._send_query(payload, timeout_ms)
            except Exception as e:
                # Nothing from the query path may reach the engine thread: a
                # coordinator that is down (or a bug in the probing) has to
                # look like "cannot ask" so the caller backs off and keeps its
                # bytes. The message is kept for whoever debugs the outage.
                if self._query_socket is not None:
                    self._query_socket.close(linger=0)
                    self._query_socket = None
                logger.warning(
                    "TairKvCacheConnector: verdict query failed (%s: %s)",
                    type(e).__name__,
                    e,
                )
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
