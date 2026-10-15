"""Unit tests for TpCoordinatorServer race condition tolerance."""

import threading
import time
import unittest
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


if __name__ == "__main__":
    unittest.main()
