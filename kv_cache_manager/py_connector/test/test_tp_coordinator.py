"""Unit tests for TpCoordinatorServer race condition tolerance."""

import threading
import time
import unittest
import socket as socket_module


from kv_cache_manager.py_connector.common.tp_coordinator import (
    CoordinateMessage,
    CoordinateMsgSerializer,
    SendBlockFinishedEvent,
    SendBlockStartEvent,
    StorageConfigsEvent,
    TpCoordinatorClient,
    TpCoordinatorServer,
)


def _find_free_port():
    """Find a free TCP port."""
    with socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(port, timeout=5.0):
    """Poll until server is ready to accept connections."""
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
        _wait_for_server(self.port)
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
        self.server._coordinator_running = False
        # Don't join - thread is blocked on recv() and will be cleaned up as daemon
        self.client._socket.close()
        self.client._zmq_context.term()


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
        _wait_for_server(self.port)
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
        self.server._coordinator_running = False
        # Don't join - thread is blocked on recv() and will be cleaned up as daemon
        self.client._socket.close()
        self.client._zmq_context.term()


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
        _wait_for_server(self.port)
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
        self.server._coordinator_running = False
        # Don't join - thread is blocked on recv() and will be cleaned up as daemon
        self.client._socket.close()
        self.client._zmq_context.term()


ROSTER = (
    '[{"type": "tair_mempool", "is_available": true, '
    '"global_unique_name": "pace_0", "storage_spec": {"root_path": "/pace/"}}]'
)


def _shutdown(client):
    """Closes both of the client's sockets, then its context.

    `Context.term()` blocks on a socket that is still open, and the query socket
    is created lazily by the first question -- so a test that only closes the
    PUSH one hangs instead of failing.
    """
    for open_socket in (client._socket, client._query_socket):
        if open_socket is not None:
            open_socket.close(linger=0)
    client._zmq_context.term()


class TestTpCoordinatorRosterChannel(unittest.TestCase):
    """The storage roster travels scheduler -> coordinator -> every worker.

    Only the registrant sees the `registerInstance` reply, and the engine's
    workers are in other processes and other pods, so the array reaches them
    through the coordinator. Two things have to hold for that to be safe: the
    string arrives byte for byte (the SDK parses it with its own schema), and a
    worker that asks before the registration lands gets an answer it can retry
    in instead of a wrong one.
    """

    def setUp(self):
        self.port = _find_free_port()
        self.server = TpCoordinatorServer("127.0.0.1", self.port, 1, lambda *a: None)
        _wait_for_server(self.port)
        self.client = TpCoordinatorClient("127.0.0.1", self.port)

    def tearDown(self):
        self.server._coordinator_running = False
        _shutdown(self.client)

    def _publish(self, roster):
        self.client.send(
            CoordinateMsgSerializer.dumps(
                CoordinateMessage(time.time(), StorageConfigsEvent(roster))
            )
        )

    def test_a_published_roster_comes_back_unchanged(self):
        self._publish(ROSTER)
        deadline = time.time() + 5
        while time.time() < deadline:
            if self.client.roster(timeout_ms=500) == ROSTER:
                break
            time.sleep(0.01)
        # Byte for byte: whitespace, key order and the fields no connector code
        # reads all have to survive, because the client parses this string.
        self.assertEqual(self.client.roster(timeout_ms=500), ROSTER)

    def test_a_worker_that_asks_before_the_registration_is_told_so(self):
        # Empty, not None: the coordinator answered, the roster is simply not
        # there yet, and that is a state to retry in rather than a broken channel.
        self.assertEqual(self.client.roster(timeout_ms=500), "")

    def test_an_unreachable_coordinator_is_not_an_empty_roster(self):
        """`None` and `""` are different facts: one retries, one investigates."""
        port = _find_free_port()
        absent = TpCoordinatorClient("127.0.0.1", port)
        try:
            self.assertIsNone(absent.roster(timeout_ms=200))
        finally:
            _shutdown(absent)

    def test_a_second_roster_cannot_replace_the_frozen_one(self):
        self._publish(ROSTER)
        self.assertEqual(self.client.roster(timeout_ms=500), ROSTER)

        # The layout is frozen at the first registration, so a different array
        # means something is wrong upstream of this channel. Keeping the first
        # one keeps every worker writing to the storage the manager allocates
        # from; taking the second one would silently move the writes.
        self._publish(ROSTER.replace("pace_0", "pace_1"))
        self.assertEqual(self.client.roster(timeout_ms=500), ROSTER)

    def test_the_load_history_answer_is_unchanged(self):
        """The second command shares the socket with the first: neither breaks."""
        self._publish(ROSTER)
        self.assertEqual(self.client.finished_loads(timeout_ms=500), set())
        self.assertEqual(self.client.roster(timeout_ms=500), ROSTER)


if __name__ == "__main__":
    unittest.main()
