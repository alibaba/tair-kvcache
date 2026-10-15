"""Unit tests for TpCoordinatorServer race condition tolerance."""

import threading
import time
import unittest
import unittest.mock
import socket as socket_module


from kv_cache_manager.py_connector.common.tp_coordinator import (
    CoordinateMessage,
    CoordinateMsgSerializer,
    LoadBlockFinishedEvent,
    SendBlockFinishedEvent,
    SendBlockStartEvent,
    TpCoordinatorClient,
    TpCoordinatorServer,
)


def _find_free_port():
    """Find a free TCP port."""
    with socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(server, timeout=5.0):
    """Poll until the server is ready to accept connections.

    Takes the server (not a port): the pair may have had to move off the
    requested port (a transient collision), and the port that must answer is
    the one it actually took.
    """
    port = server._base_port
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket_module.create_connection(("127.0.0.1", port), timeout=0.1):
                time.sleep(0.05)  # Extra time for coordinator to enter recv loop
                return True
        except (ConnectionRefusedError, socket_module.timeout):
            time.sleep(0.01)
    raise TimeoutError(f"Server did not start within {timeout}s")


class TestTpCoordinatorNormalOrder(unittest.TestCase):
    """Test normal message ordering: start before finish."""

    def setUp(self):
        self.port = _find_free_port()
        self.callback_results = []
        self.callback_event = threading.Event()

        def on_finished(write_session_id, save_context):
            self.callback_results.append((write_session_id, save_context))
            self.callback_event.set()

        self.server = TpCoordinatorServer("127.0.0.1", self.port, 2, on_finished)
        _wait_for_server(self.server)
        self.client = TpCoordinatorClient("127.0.0.1", self.port)

    def test_normal_order_callback_triggered(self):
        """Start event arrives before all finish events — normal flow."""
        # Send start
        start_msg = CoordinateMessage(
            time.time(),
            SendBlockStartEvent(
                request_id="req-1",
                write_session_id="session-1",
                locations=[{"location_specs": []}],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(start_msg))

        # Send finish from rank 0
        finish_msg_0 = CoordinateMessage(
            time.time(),
            SendBlockFinishedEvent(
                request_id="req-1",
                tp_rank=0,
                write_session_id="session-1",
                is_success_list=[True, True],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(finish_msg_0))

        # Send finish from rank 1
        finish_msg_1 = CoordinateMessage(
            time.time(),
            SendBlockFinishedEvent(
                request_id="req-1",
                tp_rank=1,
                write_session_id="session-1",
                is_success_list=[True, True],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(finish_msg_1))

        # Wait for callback
        self.assertTrue(self.callback_event.wait(timeout=5), "Callback not triggered")
        self.assertEqual(len(self.callback_results), 1)
        session_id, save_context = self.callback_results[0]
        self.assertEqual(session_id, "session-1")
        self.assertEqual(save_context.get_size(), 2)

    def tearDown(self):
        # One case, one lifecycle: the client goes first (its sockets and any
        # probe socket it opened die with it), then the server's shutdown
        # stops both routines, joins them, and only then releases the sockets
        # and the context -- so nothing of this case is still inside zmq when
        # the next case binds its own pair.
        self.client.shutdown()
        self.server.shutdown()


class TestTpCoordinatorOutOfOrder(unittest.TestCase):
    """Test out-of-order messages: finish before start."""

    def setUp(self):
        self.port = _find_free_port()
        self.callback_results = []
        self.callback_event = threading.Event()

        def on_finished(write_session_id, save_context):
            self.callback_results.append((write_session_id, save_context))
            self.callback_event.set()

        self.server = TpCoordinatorServer("127.0.0.1", self.port, 2, on_finished)
        _wait_for_server(self.server)
        self.client = TpCoordinatorClient("127.0.0.1", self.port)

    def test_finish_before_start_no_crash(self):
        """Finish events arrive before start — should not crash."""
        # Send finish from rank 0 BEFORE start
        finish_msg_0 = CoordinateMessage(
            time.time(),
            SendBlockFinishedEvent(
                request_id="req-1",
                tp_rank=0,
                write_session_id="session-1",
                is_success_list=[True],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(finish_msg_0))

        # Send finish from rank 1 BEFORE start
        finish_msg_1 = CoordinateMessage(
            time.time(),
            SendBlockFinishedEvent(
                request_id="req-1",
                tp_rank=1,
                write_session_id="session-1",
                is_success_list=[True],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(finish_msg_1))

        # Now send start — should merge buffered finishes and trigger callback
        start_msg = CoordinateMessage(
            time.time(),
            SendBlockStartEvent(
                request_id="req-1",
                write_session_id="session-1",
                locations=[{"location_specs": []}],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(start_msg))

        self.assertTrue(
            self.callback_event.wait(timeout=5), "Callback not triggered after start"
        )
        self.assertEqual(len(self.callback_results), 1)
        session_id, save_context = self.callback_results[0]
        self.assertEqual(session_id, "session-1")
        self.assertEqual(save_context.get_size(), 2)

    def test_mixed_order_multiple_ranks(self):
        """Some ranks finish before start, some after."""
        # Rank 0 finishes before start
        finish_msg_0 = CoordinateMessage(
            time.time(),
            SendBlockFinishedEvent(
                request_id="req-2",
                tp_rank=0,
                write_session_id="session-2",
                is_success_list=[True],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(finish_msg_0))

        # Send start
        start_msg = CoordinateMessage(
            time.time(),
            SendBlockStartEvent(
                request_id="req-2",
                write_session_id="session-2",
                locations=[],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(start_msg))

        # Rank 1 finishes after start
        finish_msg_1 = CoordinateMessage(
            time.time(),
            SendBlockFinishedEvent(
                request_id="req-2",
                tp_rank=1,
                write_session_id="session-2",
                is_success_list=[True],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(finish_msg_1))

        self.assertTrue(self.callback_event.wait(timeout=5), "Callback not triggered")
        self.assertEqual(len(self.callback_results), 1)
        _, save_context = self.callback_results[0]
        self.assertEqual(save_context.get_size(), 2)

    def tearDown(self):
        # One case, one lifecycle: the client goes first (its sockets and any
        # probe socket it opened die with it), then the server's shutdown
        # stops both routines, joins them, and only then releases the sockets
        # and the context -- so nothing of this case is still inside zmq when
        # the next case binds its own pair.
        self.client.shutdown()
        self.server.shutdown()


class TestTpCoordinatorIdempotent(unittest.TestCase):
    """Test duplicate finish messages are handled gracefully."""

    def setUp(self):
        self.port = _find_free_port()
        self.callback_results = []
        self.callback_event = threading.Event()

        def on_finished(write_session_id, save_context):
            self.callback_results.append((write_session_id, save_context))
            self.callback_event.set()

        self.server = TpCoordinatorServer("127.0.0.1", self.port, 2, on_finished)
        _wait_for_server(self.server)
        self.client = TpCoordinatorClient("127.0.0.1", self.port)

    def test_duplicate_finish_ignored(self):
        """Duplicate finish from same rank should be ignored (idempotent)."""
        # Send start
        start_msg = CoordinateMessage(
            time.time(),
            SendBlockStartEvent(
                request_id="req-3",
                write_session_id="session-3",
                locations=[],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(start_msg))

        # Send finish from rank 0 twice
        finish_msg = CoordinateMessage(
            time.time(),
            SendBlockFinishedEvent(
                request_id="req-3",
                tp_rank=0,
                write_session_id="session-3",
                is_success_list=[True],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(finish_msg))
        self.client.send(CoordinateMsgSerializer.dumps(finish_msg))

        # Send finish from rank 1
        finish_msg_1 = CoordinateMessage(
            time.time(),
            SendBlockFinishedEvent(
                request_id="req-3",
                tp_rank=1,
                write_session_id="session-3",
                is_success_list=[True],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(finish_msg_1))

        self.assertTrue(self.callback_event.wait(timeout=5), "Callback not triggered")
        self.assertEqual(len(self.callback_results), 1)
        _, save_context = self.callback_results[0]
        # Should be 2, not 3 (duplicate rank 0 ignored)
        self.assertEqual(save_context.get_size(), 2)

    def test_duplicate_finish_in_pending_buffer(self):
        """Duplicate finish events buffered before start should be deduplicated."""
        # Send finish from rank 0 twice BEFORE start (both go to pending buffer)
        finish_msg = CoordinateMessage(
            time.time(),
            SendBlockFinishedEvent(
                request_id="req-4",
                tp_rank=0,
                write_session_id="session-4",
                is_success_list=[True],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(finish_msg))
        self.client.send(CoordinateMsgSerializer.dumps(finish_msg))

        # Send finish from rank 1 BEFORE start
        finish_msg_1 = CoordinateMessage(
            time.time(),
            SendBlockFinishedEvent(
                request_id="req-4",
                tp_rank=1,
                write_session_id="session-4",
                is_success_list=[True],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(finish_msg_1))

        # Now send start — should merge and deduplicate
        start_msg = CoordinateMessage(
            time.time(),
            SendBlockStartEvent(
                request_id="req-4",
                write_session_id="session-4",
                locations=[],
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(start_msg))

        self.assertTrue(self.callback_event.wait(timeout=5), "Callback not triggered")
        self.assertEqual(len(self.callback_results), 1)
        _, save_context = self.callback_results[0]
        # Should be 2, not 3 (duplicate rank 0 deduplicated by add_new_rank)
        self.assertEqual(save_context.get_size(), 2)

    def tearDown(self):
        # One case, one lifecycle: the client goes first (its sockets and any
        # probe socket it opened die with it), then the server's shutdown
        # stops both routines, joins them, and only then releases the sockets
        # and the context -- so nothing of this case is still inside zmq when
        # the next case binds its own pair.
        self.client.shutdown()
        self.server.shutdown()


class TestLoadFailureAggregation(unittest.TestCase):
    """The load-finish aggregation must UNION the per-rank failed block ids.

    Rank asymmetry is the production case: one rank's storage reads fail
    (e.g. evicted files) while the other rank's succeed, and a block is only
    usable when every rank loaded it. Before the fix the coordinator kept
    only the last-arriving event's list, so a success event arriving last
    erased the other rank's failures and vLLM was never told about the
    invalid blocks."""

    def setUp(self):
        self.port = _find_free_port()
        self.server = TpCoordinatorServer("127.0.0.1", self.port, 2, lambda *a: None)
        _wait_for_server(self.server)
        self.client = TpCoordinatorClient("127.0.0.1", self.port)

    def _send(self, tp_rank, failed):
        msg = CoordinateMessage(
            time.time(),
            LoadBlockFinishedEvent(
                request_id="req-load",
                tp_rank=tp_rank,
                epoch=7,
                failed_block_idxs=failed,
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(msg))

    def _poll_failed(self, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            failed = self.server.get_failed_loading_block_idxs()
            if failed:
                return failed
            time.sleep(0.01)
        return set()

    def test_failures_survive_later_success_event(self):
        # rank 0 fails blocks 3 and 5, then rank 1's all-success event
        # completes the set -- the failures must still be reported.
        self._send(0, [3, 5])
        self._send(1, [])
        self.assertEqual(self._poll_failed(), {3, 5})

    def test_failures_from_both_ranks_are_unioned(self):
        self._send(0, [2])
        self._send(1, [4])
        self.assertEqual(self._poll_failed(), {2, 4})

    def test_failures_reported_when_success_arrives_first(self):
        # Arrival order must not matter.
        self._send(1, [])
        self._send(0, [9])
        self.assertEqual(self._poll_failed(), {9})

    def tearDown(self):
        self.client.shutdown()
        self.server.shutdown()


class TestProbeInstanceIdentity(unittest.TestCase):
    """A probe only counts when the answering pair is *this* instance's.

    Two connector instances on one host can be configured with the same
    coordinator base port: the first server takes the configured pair, the
    second falls back to the neighbouring one. The second instance's clients
    probe the configured pair first, and the first instance's query port
    answers a probe with its *own* identity -- before the identity check the
    probe accepted any reply (an old server even answers protocol errors with
    an empty body), so every event and verdict query of the second instance
    went to the first instance's coordinator and neither instance's sessions
    could complete."""

    def setUp(self):
        # Four consecutive free ports: instance A takes (p, p+1), instance B
        # -- configured with the same base port -- falls back to (p+2, p+3).
        self.holders = []
        probe_port = _find_free_port()
        for _ in range(12):
            with socket_module.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", probe_port))
                except OSError:
                    probe_port = _find_free_port()
                    continue
            self.base_port = probe_port
            break
        else:
            raise AssertionError("no free port strip found")
        for port in range(self.base_port, self.base_port + 4):
            holder = socket_module.socket()
            holder.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
            holder.bind(("127.0.0.1", port))
            holder.listen(1)
            self.holders.append(holder)
        self.addCleanup(lambda: [h.close() for h in self.holders])
        # A binds its pair after the holders release the ports (TIME_WAIT
        # insurance), B must then move one pair further.
        for holder in self.holders:
            holder.close()
        self.holders = []
        self.server_a = TpCoordinatorServer(
            "127.0.0.1", self.base_port, 2, lambda *a: None, "inst-A"
        )
        _wait_for_server(self.server_a)
        self.server_b = TpCoordinatorServer(
            "127.0.0.1", self.base_port, 2, lambda *a: None, "inst-B"
        )
        _wait_for_server(self.server_b)
        self.assertEqual(self.base_port, self.server_a._base_port)
        self.assertEqual(self.base_port + 2, self.server_b._base_port)

    def tearDown(self):
        for cleanup in (
            lambda: self.server_a.shutdown(),
            lambda: self.server_b.shutdown(),
        ):
            try:
                cleanup()
            except Exception:
                pass

    def test_a_foreign_coordinator_is_rejected_and_the_own_pair_found(self):
        # The client of instance B probes the configured pair first: server A
        # answers with inst-A's identity, which must NOT count as a hit, and
        # the search moves on to B's own pair.
        client = TpCoordinatorClient("127.0.0.1", self.base_port, "inst-B")
        self.addCleanup(client.shutdown)
        client.send(b"{}")
        self.assertEqual(
            self.base_port + 2,
            client._port,
            "the client settled on a coordinator that is not its instance's",
        )
        self.assertEqual(self.base_port + 3, client._query_port)

    def test_the_own_coordinator_still_resolves_immediately(self):
        # The identity check must not over-reject: a client of instance A
        # finds its own pair on the first probe and does not move.
        client = TpCoordinatorClient("127.0.0.1", self.base_port, "inst-A")
        self.addCleanup(client.shutdown)
        client.send(b"{}")
        self.assertEqual(self.base_port, client._port)

    def test_an_identityless_reply_is_not_a_hit(self):
        # The reply faces the client can meet: an old server answers a probe
        # with an empty protocol-error body, a foreign server with another
        # instance's identity, and a dead port with nothing at all.
        client = TpCoordinatorClient("127.0.0.1", self.base_port, "inst-B")
        self.addCleanup(client.shutdown)
        import orjson

        self.assertFalse(client._probe_reply_matches(b""))
        self.assertFalse(client._probe_reply_matches(b"not json"))
        self.assertFalse(
            client._probe_reply_matches(orjson.dumps({"probe": True}))
        )
        self.assertFalse(
            client._probe_reply_matches(
                orjson.dumps({"probe": True, "instance_id": "inst-A"})
            )
        )
        self.assertTrue(
            client._probe_reply_matches(
                orjson.dumps({"probe": True, "instance_id": "inst-B"})
            )
        )


    def test_a_query_resolves_the_pair_before_it_is_sent(self):
        # A worker's first interaction with the coordinator can be a query,
        # not an event: a late settle cancels before its done_cb sends the
        # completion event (and the pending recheck asks before that). The
        # configured pair here belongs to inst-A, so a query sent straight to
        # the configured query port would reach A's REP socket -- whose empty
        # protocol-error reply is indistinguishable from a successful cancel:
        # B's coordinator would keep the abandoned verdict and A would record
        # cross-instance state. The query must resolve and validate the pair
        # first, exactly as an event does.
        client = TpCoordinatorClient("127.0.0.1", self.base_port, "inst-B")
        self.addCleanup(client.shutdown)
        self.server_b.record_verdict("sess-q", [True])
        self.assertTrue(
            client.cancel_verdict("sess-q", 0),
            "a cancel the coordinator acknowledged came back as a failure",
        )
        self.assertEqual(
            self.base_port + 2,
            client._port,
            "the query went to a coordinator that is not this instance's",
        )
        self.assertTrue(client._endpoint_resolved)
        # The cancellation reached B: with rank 1's acknowledgement the
        # verdict is retired, exactly as a retirement expects.
        self.assertEqual([True], self.server_b.answer_query("sess-q", 1))
        self.assertNotIn("sess-q", self.server_b._verdicts)
        # And inst-A never saw the foreign session at all.
        self.assertEqual({}, self.server_a._verdict_cancels)


class TestProbeEngineIdentity(unittest.TestCase):
    """A probe only counts when the answering pair is *this engine's.

    Serving replicas share instance_id on purpose (the manager keys on it),
    so two engines of one instance on one host with the same configured base
    port answer a probe with indistinguishable instance identities: the
    second coordinator takes the neighbouring pair, but its clients accept
    the first replica's coordinator, send it every completion and load
    event, and poll their own coordinator forever. The handshake therefore
    also carries the engine's own identity (vLLM's per-run kv_transfer
    engine_id): the reply only counts when it names the client's engine."""

    def setUp(self):
        # Four consecutive free ports: engine A takes (p, p+1), engine B --
        # the same instance, the same configured base port -- falls back to
        # (p+2, p+3). The port-strip dance is TestProbeInstanceIdentity's:
        # TIME_WAIT neighbours must not eat into the strip.
        probe_port = _find_free_port()
        for _ in range(12):
            with socket_module.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", probe_port))
                except OSError:
                    probe_port = _find_free_port()
                    continue
            self.base_port = probe_port
            break
        else:
            raise AssertionError("no free port strip found")
        holders = []
        for port in range(self.base_port, self.base_port + 4):
            holder = socket_module.socket()
            holder.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
            holder.bind(("127.0.0.1", port))
            holder.listen(1)
            holders.append(holder)
        self.addCleanup(lambda: [h.close() for h in holders])
        for holder in holders:
            holder.close()
        # The same instance twice; only the engine identity differs.
        self.server_a = TpCoordinatorServer(
            "127.0.0.1", self.base_port, 2, lambda *a: None, "inst", "engine-A"
        )
        _wait_for_server(self.server_a)
        self.server_b = TpCoordinatorServer(
            "127.0.0.1", self.base_port, 2, lambda *a: None, "inst", "engine-B"
        )
        _wait_for_server(self.server_b)
        self.assertEqual(self.base_port, self.server_a._base_port)
        self.assertEqual(self.base_port + 2, self.server_b._base_port)

    def tearDown(self):
        for cleanup in (
            lambda: self.server_a.shutdown(),
            lambda: self.server_b.shutdown(),
        ):
            try:
                cleanup()
            except Exception:
                pass

    def test_a_same_instance_foreign_engine_is_rejected(self):
        # Replica B's client probes the configured pair first: replica A's
        # coordinator answers with the *same* instance identity, which must
        # NOT count as a hit, and the search moves on to B's own pair.
        client = TpCoordinatorClient(
            "127.0.0.1", self.base_port, "inst", "engine-B"
        )
        self.addCleanup(client.shutdown)
        client.send(b"{}")
        self.assertEqual(
            self.base_port + 2,
            client._port,
            "the client settled on a coordinator of another engine",
        )
        self.assertEqual(self.base_port + 3, client._query_port)

    def test_the_own_engine_resolves_immediately(self):
        # The engine check must not over-reject: replica A's client finds its
        # own pair on the first probe and does not move.
        client = TpCoordinatorClient(
            "127.0.0.1", self.base_port, "inst", "engine-A"
        )
        self.addCleanup(client.shutdown)
        client.send(b"{}")
        self.assertEqual(self.base_port, client._port)

    def test_the_engine_identity_must_match(self):
        # The reply faces the client can meet: a same-instance coordinator of
        # another engine, and an older coordinator whose reply carries no
        # engine identity at all (not provably ours, so not accepted).
        import orjson

        client = TpCoordinatorClient(
            "127.0.0.1", self.base_port, "inst", "engine-B"
        )
        self.addCleanup(client.shutdown)
        self.assertFalse(
            client._probe_reply_matches(
                orjson.dumps(
                    {"probe": True, "instance_id": "inst", "engine_id": "engine-A"}
                )
            )
        )
        self.assertFalse(
            client._probe_reply_matches(
                orjson.dumps({"probe": True, "instance_id": "inst"})
            )
        )
        self.assertTrue(
            client._probe_reply_matches(
                orjson.dumps(
                    {"probe": True, "instance_id": "inst", "engine_id": "engine-B"}
                )
            )
        )


class TestRequestFailedAggregation(unittest.TestCase):
    """A request-level load failure is ORed across the ranks, once per load.

    Block ids cannot address a multi-table block layout upstream, so a
    windowed multi-group instance marks the *request* failed instead (see
    create_load_done_callback): any rank's broken blocks fail the request,
    and the mark is handed over exactly once with the load's completion."""

    def setUp(self):
        self.port = _find_free_port()
        self.server = TpCoordinatorServer("127.0.0.1", self.port, 2, lambda *a: None)
        _wait_for_server(self.server)
        self.client = TpCoordinatorClient("127.0.0.1", self.port)

    def _send(self, tp_rank, request_failed):
        msg = CoordinateMessage(
            time.time(),
            LoadBlockFinishedEvent(
                request_id="req-f",
                tp_rank=tp_rank,
                epoch=11,
                request_failed=request_failed,
            ),
        )
        self.client.send(CoordinateMsgSerializer.dumps(msg))

    def _poll_failed_requests(self, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            failed = self.server.get_failed_loading_requests()
            if failed:
                return failed
            time.sleep(0.01)
        return set()

    def test_one_failed_rank_fails_the_request(self):
        # Rank 0's blocks failed, rank 1's succeeded: the request is failed
        # (the AND a load needs is broken by any rank).
        self._send(0, True)
        self._send(1, False)
        self.assertEqual(self._poll_failed_requests(), {"req-f"})
        # Handed over once: a later poll finds nothing more.
        self.assertEqual(set(), self.server.get_failed_loading_requests())

    def test_all_success_reports_no_failed_request(self):
        self._send(0, False)
        self._send(1, False)
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if self.server.get_finished_tasks()[1]:
                break
            time.sleep(0.01)
        self.assertEqual(set(), self.server.get_failed_loading_requests())

    def tearDown(self):
        self.client.shutdown()
        self.server.shutdown()


class TestForeignCoordinatorRefusal(unittest.TestCase):
    """Only this instance's coordinator may receive this client's traffic.

    This instance's coordinator can be unavailable while another instance's
    coordinator owns the configured port: the probe rejects the foreign
    identity, and nothing may be delivered there -- a foreign coordinator
    would aggregate the wrong instance's completions. Events are held back
    (bounded) until a coordinator of this instance answers, then flushed in
    order; verdict queries report "cannot ask" instead of polling the foreign
    coordinator forever."""

    def setUp(self):
        # A strip of free ports like TestProbeInstanceIdentity: instance A
        # owns (p, p+1); a later server of instance B falls back to (p+2,
        # p+3), which is inside the probe range of a client configured with
        # the same base port.
        self.holders = []
        probe_port = _find_free_port()
        for _ in range(12):
            with socket_module.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", probe_port))
                except OSError:
                    probe_port = _find_free_port()
                    continue
            self.base_port = probe_port
            break
        else:
            raise AssertionError("no free port strip found")
        for port in range(self.base_port, self.base_port + 4):
            holder = socket_module.socket()
            holder.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
            holder.bind(("127.0.0.1", port))
            holder.listen(1)
            self.holders.append(holder)
        for holder in self.holders:
            holder.close()
        self.seen_a = []
        self.seen_b = []
        # tp_size=1: one start + one finish completes a save, so a delivery
        # to a coordinator is directly observable through its callback.
        self.server_a = TpCoordinatorServer(
            "127.0.0.1", self.base_port, 1, self._record(self.seen_a), "inst-A"
        )
        _wait_for_server(self.server_a)
        # B's own coordinator does not exist yet: its client is configured
        # with the same base port A owns.
        self.server_b = None
        self.client = TpCoordinatorClient("127.0.0.1", self.base_port, "inst-B")
        # Teardown order as everywhere in this suite: the client first, then
        # the servers.
        self.addCleanup(self.server_a.shutdown)
        self.addCleanup(self._shutdown_b)
        self.addCleanup(self.client.shutdown)

    def _record(self, sink):
        def on_finished(write_session_id, save_context):
            sink.append(write_session_id)

        return on_finished

    def _shutdown_b(self):
        if self.server_b is not None:
            self.server_b.shutdown()

    def _save_events(self, request_id, write_session_id):
        return [
            CoordinateMsgSerializer.dumps(
                CoordinateMessage(
                    time.time(),
                    SendBlockStartEvent(
                        request_id=request_id,
                        write_session_id=write_session_id,
                        locations=[],
                    ),
                )
            ),
            CoordinateMsgSerializer.dumps(
                CoordinateMessage(
                    time.time(),
                    SendBlockFinishedEvent(
                        request_id=request_id,
                        tp_rank=0,
                        write_session_id=write_session_id,
                        is_success_list=[True],
                    ),
                )
            ),
        ]

    def test_events_are_deferred_until_the_own_coordinator_answers(self):
        # While only A answers, every event is held back: A's coordinator
        # must not aggregate them (the wrong instance), and the endpoint
        # stays unresolved so discovery is retried.
        for param in self._save_events("req-b1", "sess-b1"):
            self.client.send(param)
        self.assertFalse(self.client._endpoint_resolved,
                         "a foreign-only answer must not resolve the endpoint")
        self.assertEqual(len(self.client._pending_events), 2)
        # Give a wrongly-delivered event time to arrive and complete A's
        # aggregation: neither the callback nor the finished list may see it.
        time.sleep(0.2)
        self.assertEqual(self.seen_a, [])
        self.assertEqual(self.server_a.get_finished_tasks(), ([], []),
                         "the foreign coordinator aggregated deferred events")

        # B's own coordinator comes up on the fallback pair; the next event
        # finds it, flushes everything that was held back, in order, and only
        # then delivers itself.
        self.server_b = TpCoordinatorServer(
            "127.0.0.1", self.base_port, 1, self._record(self.seen_b), "inst-B"
        )
        _wait_for_server(self.server_b)
        for param in self._save_events("req-b2", "sess-b2"):
            self.client.send(param)
        self.assertEqual(self.client._port, self.base_port + 2,
                         "the client did not move to its instance's pair")
        self.assertTrue(self.client._endpoint_resolved)
        self.assertEqual(len(self.client._pending_events), 0)
        deadline = time.time() + 5.0
        while time.time() < deadline and sorted(self.seen_b) != [
            "sess-b1",
            "sess-b2",
        ]:
            time.sleep(0.01)
        # The deferred save completed on B's coordinator (order preserved:
        # its start arrived before its finish), and A still saw nothing.
        self.assertEqual(sorted(self.seen_b), ["sess-b1", "sess-b2"])
        self.assertEqual(self.seen_a, [])

    def test_a_dead_port_defers_too_and_retries_discovery(self):
        # The other failure shape: nothing answers at all (the coordinator is
        # not up yet -- a startup race). The event is held back rather than
        # queued on a socket whose port this instance may never own, and the
        # next event re-probes: the coordinator that comes up later -- on the
        # configured pair or, after a collision, the fallback one -- receives
        # everything, in order.
        self.server_a.shutdown()
        start_b3, finish_b3 = self._save_events("req-b3", "sess-b3")
        self.client.send(start_b3)
        self.assertFalse(self.client._endpoint_resolved)
        self.assertEqual(len(self.client._pending_events), 1)
        # B's own coordinator comes up (whichever pair it could bind).
        self.server_b = TpCoordinatorServer(
            "127.0.0.1", self.base_port, 1, self._record(self.seen_b), "inst-B"
        )
        _wait_for_server(self.server_b)
        # The second event discovers it and flushes the held-back one first.
        self.client.send(finish_b3)
        deadline = time.time() + 5.0
        while time.time() < deadline and not self.seen_b:
            time.sleep(0.01)
        self.assertEqual(self.seen_b, ["sess-b3"])
        self.assertTrue(self.client._endpoint_resolved)
        self.assertEqual(len(self.client._pending_events), 0)

    def test_the_defer_buffer_is_bounded(self):
        # An outage longer than the buffer must drop the oldest events loudly
        # rather than grow without bound; the coordinator-side caps settle
        # the loss, exactly as for a dead coordinator.
        import kv_cache_manager.py_connector.common.tp_coordinator as tp

        with unittest.mock.patch.object(tp, "DEFERRED_EVENT_LIMIT", 3):
            # r1: start+finish, r2: start fills the buffer, r2's finish drops
            # r1's start -- without it r1's save can never complete.
            for param in self._save_events("req-b1", "sess-b1"):
                self.client.send(param)
            self.client.send(self._save_events("req-b2", "sess-b2")[0])
            with self.assertLogs(tp.logger, level="WARNING"):
                self.client.send(self._save_events("req-b2", "sess-b2")[1])
            self.assertEqual(len(self.client._pending_events), 3)
        self.server_b = TpCoordinatorServer(
            "127.0.0.1", self.base_port, 1, self._record(self.seen_b), "inst-B"
        )
        _wait_for_server(self.server_b)
        self.client.send(self._save_events("req-b3", "sess-b3")[0])
        deadline = time.time() + 5.0
        while time.time() < deadline and not self.seen_b:
            time.sleep(0.01)
        # Only r2 completed: r1's start was dropped, so its finish -- flushed
        # ahead of it -- can never complete on the coordinator.
        self.assertEqual(self.seen_b, ["sess-b2"])
        self.assertEqual(self.seen_a, [])


if __name__ == "__main__":
    unittest.main()
