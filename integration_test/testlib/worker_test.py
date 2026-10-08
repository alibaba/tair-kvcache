import types
import unittest
from unittest import mock

from integration_test.testlib.worker import Worker


class WorkerReadinessTest(unittest.TestCase):
    def setUp(self):
        self.worker = Worker(0, types.SimpleNamespace(
            ip="127.0.0.1", rpc_port=10001, http_port=10002, admin_http_port=10003))

    @mock.patch("integration_test.testlib.worker.time.sleep")
    @mock.patch("integration_test.testlib.worker.socket.create_connection")
    def test_retries_until_every_listener_is_ready(self, connect, sleep):
        connect.side_effect = [ConnectionRefusedError(), mock.MagicMock(),
                               mock.MagicMock(), mock.MagicMock()]
        self.assertTrue(self.worker._wait_ready())
        sleep.assert_called_once_with(0.05)
        self.assertEqual({10001, 10002, 10003}, {call[0][0][1] for call in connect.call_args_list[1:]})

    @mock.patch("integration_test.testlib.worker.time.monotonic", side_effect=[0, 0, 31])
    @mock.patch("integration_test.testlib.worker.time.sleep")
    @mock.patch("integration_test.testlib.worker.socket.create_connection", side_effect=ConnectionRefusedError())
    def test_timeout_reports_failure_instead_of_accepting_daemon_launch(self, connect, sleep, clock):
        self.assertFalse(self.worker._wait_ready())
        self.assertEqual(1, connect.call_count)
        sleep.assert_called_once()

    @mock.patch("integration_test.testlib.worker.time.sleep")
    @mock.patch("integration_test.testlib.worker.socket.create_connection")
    def test_ready_worker_has_no_fixed_startup_delay(self, connect, sleep):
        self.assertTrue(self.worker._wait_ready())
        self.assertEqual(3, connect.call_count)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
