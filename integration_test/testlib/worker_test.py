import types
import unittest
from unittest import mock

from integration_test.testlib.worker import Worker


class WorkerReadinessTest(unittest.TestCase):
    def setUp(self):
        self.worker = Worker(0, types.SimpleNamespace(
            ip="127.0.0.1", rpc_port=10001, http_port=10002, admin_http_port=10003))
        leader_patch = mock.patch.object(self.worker, "_has_leader", return_value=True)
        self.leader = leader_patch.start()
        self.addCleanup(leader_patch.stop)

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

    @mock.patch("integration_test.testlib.worker.time.sleep")
    @mock.patch("integration_test.testlib.worker.socket.create_connection")
    def test_listening_worker_waits_for_election(self, connect, sleep):
        self.leader.side_effect = [False, True]
        self.assertTrue(self.worker._wait_ready())
        self.assertEqual(6, connect.call_count)
        sleep.assert_called_once_with(0.05)

    @mock.patch("integration_test.testlib.worker.time.monotonic", side_effect=[0, 0, 31])
    @mock.patch("integration_test.testlib.worker.time.sleep")
    @mock.patch("integration_test.testlib.worker.socket.create_connection")
    def test_no_leader_fails_at_deadline(self, connect, sleep, clock):
        self.leader.return_value = False
        self.assertFalse(self.worker._wait_ready())

    @mock.patch("integration_test.testlib.worker.http.client.HTTPConnection")
    def test_follower_with_discovered_leader_is_ready(self, connection_type):
        connection = connection_type.return_value
        response = connection.getresponse.return_value
        response.status = 200
        response.read.return_value = (b'{"header":{"status":{"code":"OK"}},'
                                      b'"self_node_id":"follower","leader_node_id":"leader"}')
        self.assertTrue(Worker._has_leader(self.worker))
        connection.request.assert_called_once_with(
            "POST", "/api/getClusterInfo", body='{"trace_id":"test-readiness"}',
            headers={"Content-Type": "application/json"})
        connection.close.assert_called_once()

    @mock.patch("integration_test.testlib.worker.http.client.HTTPConnection")
    def test_missing_leader_or_unavailable_api_is_not_ready(self, connection_type):
        connection = connection_type.return_value
        response = connection.getresponse.return_value
        response.status = 200
        for body in [b'{"header":{"status":{"code":"OK"}}}',
                     b'{"header":{"status":{"code":"SERVICE_NOT_READY"}},"leader_node_id":"n"}',
                     b'not-json']:
            response.read.return_value = body
            self.assertFalse(Worker._has_leader(self.worker))
        connection.request.side_effect = ConnectionRefusedError()
        self.assertFalse(Worker._has_leader(self.worker))
        self.assertEqual(4, connection.close.call_count)


if __name__ == "__main__":
    unittest.main()
