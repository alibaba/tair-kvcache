"""Unit tests for TpCoordinatorServer race condition tolerance."""

import threading
import time
import unittest

import zmq

from kv_cache_manager.py_connector.common.tp_coordinator import (
    CoordinateMessage,
    CoordinateMsgSerializer,
    SendBlockFinishedEvent,
    SendBlockStartEvent,
    TpCoordinatorClient,
    TpCoordinatorServer,
)


def _find_free_port():
    """Find a free TCP port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


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
        self.client = TpCoordinatorClient("127.0.0.1", self.port)
        # Give coordinator thread time to bind
        time.sleep(0.1)

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
        self.client = TpCoordinatorClient("127.0.0.1", self.port)
        time.sleep(0.1)

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

        self.assertTrue(self.callback_event.wait(timeout=5), "Callback not triggered after start")
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
        self.client = TpCoordinatorClient("127.0.0.1", self.port)
        time.sleep(0.1)

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

    def tearDown(self):
        self.server._coordinator_running = False


if __name__ == "__main__":
    unittest.main()
