"""Unit tests for KvCacheManagerClient leader discovery."""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import requests

from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient


def _ok_response_json(data=None):
    """Build a standard OK response dict."""
    resp = {"header": {"status": {"code": "OK", "message": ""}}}
    if data:
        resp.update(data)
    return resp


def _cluster_info_response(host="10.0.0.1", port=8080):
    """Build a GetClusterInfo OK response with leader endpoint."""
    return {
        "header": {"status": {"code": "OK", "message": ""}},
        "self_node_id": "node-1",
        "leader_node_id": "node-0",
        "leader_endpoint": {
            "node_id": "node-0",
            "host": host,
            "meta_http_port": port,
            "meta_rpc_port": 50051,
        },
    }


def _not_leader_response():
    """Build a SERVER_NOT_LEADER response."""
    return {
        "header": {
            "status": {
                "code": "SERVER_NOT_LEADER",
                "message": "Current node is not the leader",
            }
        }
    }


def _make_mock_response(json_data, status_code=200):
    """Create a mock requests.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    return mock_resp


class TestLeaderDiscoveryInit(unittest.TestCase):
    """Tests for leader discovery during __init__."""

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_init_discovers_leader_and_switches(self, mock_post):
        """Init should discover leader via getClusterInfo and switch base_url."""
        mock_post.return_value = _make_mock_response(_cluster_info_response("10.0.0.99", 9090))

        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=True)
        try:
            self.assertEqual(client.base_url, "http://10.0.0.99:9090")
            self.assertEqual(client._discovery_url, "http://10.0.0.1:8080")
            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            self.assertIn("/api/getClusterInfo", call_url)
        finally:
            client.close()

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_init_discovery_includes_instance_id(self, mock_post):
        """Discovery request should include instance_id in the request body."""
        mock_post.return_value = _make_mock_response(_cluster_info_response("10.0.0.99", 9090))

        client = KvCacheManagerClient(
            "http://10.0.0.1:8080",
            instance_id="my-instance-123",
            auto_discover_leader=True,
        )
        try:
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            self.assertEqual(call_kwargs["json"]["instance_id"], "my-instance-123")
        finally:
            client.close()

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_init_already_leader_no_switch(self, mock_post):
        """If initial address is already the leader, base_url stays the same."""
        mock_post.return_value = _make_mock_response(
            _cluster_info_response("10.0.0.1", 8080)
        )

        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=True)
        try:
            self.assertEqual(client.base_url, "http://10.0.0.1:8080")
        finally:
            client.close()

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_init_404_keeps_original_url(self, mock_post):
        """Old server returning 404 should keep original base_url."""
        mock_post.return_value = _make_mock_response({}, status_code=404)

        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=True)
        try:
            self.assertEqual(client.base_url, "http://10.0.0.1:8080")
        finally:
            client.close()

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_init_connection_error_keeps_original_url(self, mock_post):
        """ConnectionError during init discovery should keep original base_url."""
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=True)
        try:
            self.assertEqual(client.base_url, "http://10.0.0.1:8080")
        finally:
            client.close()

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_init_empty_leader_endpoint_keeps_original(self, mock_post):
        """Response with empty leader_endpoint should keep original base_url."""
        resp_data = {
            "header": {"status": {"code": "OK", "message": ""}},
            "self_node_id": "node-1",
            "leader_node_id": "",
        }
        mock_post.return_value = _make_mock_response(resp_data)

        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=True)
        try:
            self.assertEqual(client.base_url, "http://10.0.0.1:8080")
        finally:
            client.close()

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_init_leader_endpoint_missing_host(self, mock_post):
        """leader_endpoint with missing host should keep original base_url."""
        resp_data = {
            "header": {"status": {"code": "OK", "message": ""}},
            "leader_endpoint": {"node_id": "node-0", "host": "", "meta_http_port": 8080},
        }
        mock_post.return_value = _make_mock_response(resp_data)

        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=True)
        try:
            self.assertEqual(client.base_url, "http://10.0.0.1:8080")
        finally:
            client.close()

    def test_init_auto_discover_disabled(self):
        """auto_discover_leader=False should not start discovery or background thread."""
        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=False)
        try:
            self.assertEqual(client.base_url, "http://10.0.0.1:8080")
            self.assertIsNone(client._refresh_thread)
            self.assertFalse(client._auto_discover_leader)
        finally:
            client.close()

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_init_starts_background_thread(self, mock_post):
        """auto_discover_leader=True should start a daemon background thread."""
        mock_post.return_value = _make_mock_response(_cluster_info_response())

        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=True)
        try:
            self.assertIsNotNone(client._refresh_thread)
            self.assertTrue(client._refresh_thread.is_alive())
            self.assertTrue(client._refresh_thread.daemon)
        finally:
            client.close()


class TestDiscoveryAlwaysUsesSeedUrl(unittest.TestCase):
    """Tests that discovery always queries _discovery_url, never the current base_url."""

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_discover_uses_discovery_url_not_base_url(self, mock_post):
        """After switching to leader, re-discovery should still query the seed url."""
        # Init: discover leader, switch base_url to 10.0.0.99:9090
        mock_post.return_value = _make_mock_response(_cluster_info_response("10.0.0.99", 9090))
        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=True)
        self.assertEqual(client.base_url, "http://10.0.0.99:9090")
        self.assertEqual(client._discovery_url, "http://10.0.0.1:8080")

        # Reset and re-discover — should only call discovery_url (10.0.0.1:8080)
        mock_post.reset_mock()
        called_urls = []

        def capture_url(*args, **kwargs):
            called_urls.append(args[0])
            return _make_mock_response(_cluster_info_response("10.0.0.50", 7070))

        mock_post.side_effect = capture_url

        try:
            result = client._discover_leader()
            self.assertTrue(result)
            self.assertEqual(client.base_url, "http://10.0.0.50:7070")
            # Only the seed discovery_url should have been called
            self.assertEqual(len(called_urls), 1)
            self.assertIn("10.0.0.1:8080", called_urls[0])
            self.assertNotIn("10.0.0.99", called_urls[0])
        finally:
            client.close()


class TestServerNotLeaderRetry(unittest.TestCase):
    """Tests for SERVER_NOT_LEADER auto-retry in _make_api_request."""

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_not_leader_retry_succeeds(self, mock_discovery_post):
        """SERVER_NOT_LEADER should trigger discovery and retry, then succeed."""
        # Init discovery: already on leader
        mock_discovery_post.return_value = _make_mock_response(
            _cluster_info_response("10.0.0.1", 8080)
        )
        client = KvCacheManagerClient(
            "http://10.0.0.1:8080",
            auto_discover_leader=True,
            leader_retry_count=2,
            leader_retry_base_interval_seconds=0.001,
        )

        call_count = [0]

        def mock_session_post(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_mock_response(_not_leader_response())
            return _make_mock_response(_ok_response_json({"result": "success"}))

        # After NOT_LEADER, discovery finds new leader
        mock_discovery_post.reset_mock()
        mock_discovery_post.return_value = _make_mock_response(
            _cluster_info_response("10.0.0.2", 8080)
        )

        try:
            client.session.post = mock_session_post
            result = client.register_instance({"trace_id": "test"})
            self.assertEqual(result["result"], "success")
            # First call returns NOT_LEADER, second succeeds
            self.assertEqual(call_count[0], 2)
        finally:
            client.close()

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_not_leader_retries_exhausted(self, mock_discovery_post):
        """When all retries exhausted with NOT_LEADER, should raise AssertionError."""
        mock_discovery_post.return_value = _make_mock_response(
            _cluster_info_response("10.0.0.1", 8080)
        )
        client = KvCacheManagerClient(
            "http://10.0.0.1:8080",
            auto_discover_leader=True,
            leader_retry_count=1,
            leader_retry_base_interval_seconds=0.001,
        )

        # discovery always returns the same node (which keeps being not-leader)
        mock_discovery_post.reset_mock()
        mock_discovery_post.return_value = _make_mock_response(
            _cluster_info_response("10.0.0.1", 8080)
        )

        def mock_session_post(url, **kwargs):
            return _make_mock_response(_not_leader_response())

        try:
            client.session.post = mock_session_post
            with self.assertRaises(AssertionError) as ctx:
                client.register_instance({"trace_id": "test"})
            self.assertIn("Current node is not the leader", str(ctx.exception))
        finally:
            client.close()

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_not_leader_with_discover_disabled(self, mock_discovery_post):
        """With auto_discover_leader=False, NOT_LEADER should not trigger retry."""
        client = KvCacheManagerClient(
            "http://10.0.0.1:8080",
            auto_discover_leader=False,
        )

        def mock_session_post(url, **kwargs):
            return _make_mock_response(_not_leader_response())

        try:
            client.session.post = mock_session_post
            with self.assertRaises(AssertionError) as ctx:
                client.register_instance({"trace_id": "test"})
            self.assertIn("Current node is not the leader", str(ctx.exception))
        finally:
            client.close()


class TestConnectionErrorHandling(unittest.TestCase):
    """Tests for ConnectionError handling and background refresh wakeup."""

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_connection_error_wakes_background_refresh(self, mock_discovery_post):
        """ConnectionError should set _refresh_event to wake background thread."""
        mock_discovery_post.return_value = _make_mock_response(
            _cluster_info_response("10.0.0.1", 8080)
        )
        client = KvCacheManagerClient(
            "http://10.0.0.1:8080",
            auto_discover_leader=True,
        )

        def mock_session_post(url, **kwargs):
            raise requests.ConnectionError("Connection refused")

        try:
            client.session.post = mock_session_post
            client._refresh_event.clear()

            with self.assertRaises(requests.ConnectionError):
                client.register_instance({"trace_id": "test"})

            # _refresh_event should have been set
            self.assertTrue(client._refresh_event.is_set())
        finally:
            client.close()

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_connection_error_no_wakeup_when_disabled(self, mock_discovery_post):
        """ConnectionError with discover disabled should not touch _refresh_event."""
        client = KvCacheManagerClient(
            "http://10.0.0.1:8080",
            auto_discover_leader=False,
        )

        def mock_session_post(url, **kwargs):
            raise requests.ConnectionError("Connection refused")

        try:
            client.session.post = mock_session_post
            with self.assertRaises(requests.ConnectionError):
                client.register_instance({"trace_id": "test"})
            # _refresh_event should NOT be set
            self.assertFalse(client._refresh_event.is_set())
        finally:
            client.close()


class TestBackgroundRefresh(unittest.TestCase):
    """Tests for background leader refresh thread."""

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_background_refresh_updates_base_url(self, mock_post):
        """Background refresh should update base_url when leader changes."""
        # Init: current node is leader
        mock_post.return_value = _make_mock_response(
            _cluster_info_response("10.0.0.1", 8080)
        )
        client = KvCacheManagerClient(
            "http://10.0.0.1:8080",
            auto_discover_leader=True,
            discovery_refresh_interval_seconds=60,  # long interval, we'll trigger manually
            min_discover_interval_seconds=0,  # disable throttle for test
        )
        self.assertEqual(client.base_url, "http://10.0.0.1:8080")

        try:
            # Change the leader
            mock_post.return_value = _make_mock_response(
                _cluster_info_response("10.0.0.2", 9090)
            )
            # Reset last discover time so min interval doesn't block
            client._last_discover_time = 0

            # Trigger urgent wakeup
            client._refresh_event.set()
            # Give background thread time to process
            time.sleep(0.2)

            self.assertEqual(client.base_url, "http://10.0.0.2:9090")
        finally:
            client.close()

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_min_interval_protection(self, mock_post):
        """Min interval should delay (not skip) re-discovery, then still execute it."""
        mock_post.return_value = _make_mock_response(
            _cluster_info_response("10.0.0.1", 8080)
        )
        client = KvCacheManagerClient(
            "http://10.0.0.1:8080",
            auto_discover_leader=True,
            discovery_refresh_interval_seconds=60,
            min_discover_interval_seconds=0.3,  # short but nonzero
        )

        try:
            # Change the leader response
            mock_post.return_value = _make_mock_response(
                _cluster_info_response("10.0.0.99", 9090)
            )

            # Trigger urgent wakeup — min interval not yet elapsed
            client._refresh_event.set()
            time.sleep(0.05)

            # base_url should NOT have changed yet (still waiting for min interval)
            self.assertEqual(client.base_url, "http://10.0.0.1:8080")

            # After min interval elapses, the thread should complete discovery
            time.sleep(0.5)
            self.assertEqual(client.base_url, "http://10.0.0.99:9090")
        finally:
            client.close()


class TestCloseLifecycle(unittest.TestCase):
    """Tests for proper cleanup on close()."""

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_close_stops_background_thread(self, mock_post):
        """close() should stop the background refresh thread."""
        mock_post.return_value = _make_mock_response(
            _cluster_info_response("10.0.0.1", 8080)
        )
        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=True)

        self.assertTrue(client._refresh_thread.is_alive())
        client.close()
        self.assertFalse(client._refresh_thread.is_alive())
        self.assertTrue(client._closed.is_set())

    def test_close_without_discovery(self):
        """close() should work fine when auto_discover_leader=False."""
        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=False)
        client.close()  # Should not raise
        self.assertTrue(client._closed.is_set())


class TestThreadSafety(unittest.TestCase):
    """Tests for thread-safe leader discovery dedup."""

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_concurrent_discover_dedup(self, mock_post):
        """Multiple threads calling _discover_leader should dedup via lock."""
        actual_discover_count = [0]
        original_post = mock_post

        def slow_post(*args, **kwargs):
            actual_discover_count[0] += 1
            time.sleep(0.05)  # simulate slow network
            return _make_mock_response(_cluster_info_response("10.0.0.99", 9090))

        mock_post.side_effect = slow_post

        client = KvCacheManagerClient(
            "http://10.0.0.1:8080",
            auto_discover_leader=True,
            discovery_refresh_interval_seconds=60,
        )

        try:
            # Reset for the concurrent test
            actual_discover_count[0] = 0
            client.base_url = "http://10.0.0.1:8080"  # reset to trigger discovery

            # Launch many threads that all call _discover_leader
            threads = []
            results = []
            lock = threading.Lock()

            def do_discover():
                result = client._discover_leader()
                with lock:
                    results.append(result)

            for _ in range(10):
                t = threading.Thread(target=do_discover)
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            # All should have succeeded
            self.assertTrue(all(results))
            # But the actual HTTP call should only have happened once (or very few times)
            # due to the snapshot-check dedup in _discover_leader
            self.assertLessEqual(actual_discover_count[0], 2)
        finally:
            client.close()


class TestGetClusterInfoPublicApi(unittest.TestCase):
    """Tests for the public get_cluster_info() method."""

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_get_cluster_info_returns_response(self, mock_discovery_post):
        """get_cluster_info() should return cluster info response."""
        mock_discovery_post.return_value = _make_mock_response(
            _cluster_info_response("10.0.0.1", 8080)
        )
        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=True)

        expected = _cluster_info_response("10.0.0.1", 8080)

        def mock_session_post(url, **kwargs):
            return _make_mock_response(expected)

        try:
            client.session.post = mock_session_post
            result = client.get_cluster_info({"trace_id": "test"})
            self.assertEqual(result["leader_node_id"], "node-0")
            self.assertIn("leader_endpoint", result)
        finally:
            client.close()


class TestNormalApiStillWorks(unittest.TestCase):
    """Regression tests: normal API calls should still work unchanged."""

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_register_instance_ok(self, mock_discovery_post):
        """register_instance should work normally with leader discovery enabled."""
        mock_discovery_post.return_value = _make_mock_response(
            _cluster_info_response("10.0.0.1", 8080)
        )
        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=True)

        ok_resp = _ok_response_json({"storage_configs": []})

        def mock_session_post(url, **kwargs):
            return _make_mock_response(ok_resp)

        try:
            client.session.post = mock_session_post
            result = client.register_instance({"trace_id": "t1"})
            self.assertEqual(result["header"]["status"]["code"], "OK")
        finally:
            client.close()

    def test_api_works_without_discovery(self):
        """API calls should work with auto_discover_leader=False."""
        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=False)

        ok_resp = _ok_response_json({"locations": []})

        def mock_session_post(url, **kwargs):
            return _make_mock_response(ok_resp)

        try:
            client.session.post = mock_session_post
            result = client.get_cache_location({"trace_id": "t1"})
            self.assertEqual(result["header"]["status"]["code"], "OK")
        finally:
            client.close()

    @patch("kv_cache_manager.py_connector.common.manager_client.requests.post")
    def test_check_response_false_returns_raw(self, mock_discovery_post):
        """check_response=False should return raw response even on error."""
        mock_discovery_post.return_value = _make_mock_response(
            _cluster_info_response("10.0.0.1", 8080)
        )
        client = KvCacheManagerClient("http://10.0.0.1:8080", auto_discover_leader=True)

        error_resp = {
            "header": {"status": {"code": "INTERNAL_ERROR", "message": "something broke"}}
        }

        def mock_session_post(url, **kwargs):
            return _make_mock_response(error_resp)

        try:
            client.session.post = mock_session_post
            result = client.register_instance({"trace_id": "t1"}, check_response=False)
            self.assertEqual(result["header"]["status"]["code"], "INTERNAL_ERROR")
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
