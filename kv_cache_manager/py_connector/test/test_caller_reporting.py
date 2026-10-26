"""Unit tests for affinity caller reporting on the HTTP control plane.

Pure logic: the compiled kvcm_py_client is replaced by a stub, so these tests
need neither a pace sidecar nor the pybind module.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kv_cache_manager.py_connector.common import caller_node
from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient


def _ok_response_json(data=None):
    resp = {"header": {"status": {"code": "OK", "message": ""}}}
    if data:
        resp.update(data)
    return resp


def _make_mock_response(json_data, status_code=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    return mock_resp


class CallerNodeStateMixin(unittest.TestCase):
    """Reset the module-wide resolution cache around every test."""

    def setUp(self):
        caller_node._node_id = ""
        caller_node._last_attempt = 0.0

    def tearDown(self):
        caller_node._node_id = ""
        caller_node._last_attempt = 0.0


class TestWithCallerNode(CallerNodeStateMixin):
    def test_attaches_caller_on_write_endpoint(self):
        with patch.object(caller_node, "_query_local_node_id", return_value="72"):
            payload = caller_node.with_caller_node(
                "/api/startWriteCache", {"trace_id": "t1"}
            )
        self.assertEqual(payload["caller"], {"node_id": "72"})
        self.assertEqual(payload["trace_id"], "t1")

    def test_attaches_caller_on_read_endpoint(self):
        with patch.object(caller_node, "_query_local_node_id", return_value="72"):
            payload = caller_node.with_caller_node(
                "/api/getCacheLocation", {"trace_id": "t1"}
            )
        self.assertEqual(payload["caller"], {"node_id": "72"})

    def test_skips_endpoints_without_a_caller_field(self):
        with patch.object(caller_node, "_query_local_node_id", return_value="72"):
            for endpoint in (
                "/api/registerInstance",
                "/api/finishWriteCache",
                "/api/getCacheLocationsByBackend",
                "/api/reportEvent",
            ):
                payload = caller_node.with_caller_node(endpoint, {"trace_id": "t1"})
                self.assertNotIn("caller", payload, endpoint)

    def test_unknown_node_id_keeps_payload_identical(self):
        original = {"trace_id": "t1"}
        with patch.object(caller_node, "_query_local_node_id", return_value=""):
            payload = caller_node.with_caller_node("/api/startWriteCache", original)
        self.assertIs(payload, original)

    def test_explicit_caller_wins(self):
        original = {"trace_id": "t1", "caller": {"node_id": "38"}}
        with patch.object(caller_node, "_query_local_node_id", return_value="72"):
            payload = caller_node.with_caller_node("/api/startWriteCache", original)
        self.assertEqual(payload["caller"], {"node_id": "38"})

    def test_input_mapping_is_not_mutated(self):
        original = {"trace_id": "t1"}
        with patch.object(caller_node, "_query_local_node_id", return_value="72"):
            caller_node.with_caller_node("/api/startWriteCache", original)
        self.assertEqual(original, {"trace_id": "t1"})

    def test_none_payload_passes_through(self):
        with patch.object(caller_node, "_query_local_node_id", return_value="72"):
            self.assertIsNone(caller_node.with_caller_node("/api/startWriteCache", None))


class TestResolutionCaching(CallerNodeStateMixin):
    def test_resolution_is_cached_across_calls(self):
        query = MagicMock(return_value="72")
        with patch.object(caller_node, "_query_local_node_id", query):
            for _ in range(5):
                caller_node.with_caller_node("/api/startWriteCache", {"trace_id": "t"})
        self.assertEqual(query.call_count, 1)

    def test_failed_lookup_backs_off_instead_of_probing_every_call(self):
        query = MagicMock(return_value="")
        with patch.object(caller_node, "_query_local_node_id", query):
            for _ in range(5):
                caller_node.with_caller_node("/api/startWriteCache", {"trace_id": "t"})
        self.assertEqual(query.call_count, 1)

    def test_empty_refresh_keeps_previous_identity(self):
        with patch.object(caller_node, "_query_local_node_id", return_value="72"):
            self.assertEqual(caller_node.local_caller_node_id(), "72")
        # Force the next refresh to be due, then answer with an empty id as a
        # restarting sidecar would.
        caller_node._last_attempt = 0.0
        with patch.object(caller_node, "_query_local_node_id", return_value=""):
            self.assertEqual(caller_node.local_caller_node_id(), "72")

    def test_changed_identity_is_adopted(self):
        with patch.object(caller_node, "_query_local_node_id", return_value="72"):
            self.assertEqual(caller_node.local_caller_node_id(), "72")
        caller_node._last_attempt = 0.0
        with patch.object(caller_node, "_query_local_node_id", return_value="38"):
            self.assertEqual(caller_node.local_caller_node_id(), "38")

    def test_missing_compiled_client_yields_empty_id(self):
        # A sys.modules entry of None makes the import raise ImportError,
        # which is what a wheel without the compiled client looks like.
        with patch.dict(
            "sys.modules", {"kv_cache_manager.client.pybind": None}
        ):
            self.assertEqual(caller_node._query_local_node_id(), "")

    def test_missing_pybind_export_yields_empty_id(self):
        module = MagicMock(spec=[])  # no local_node_id attribute
        package = SimpleNamespace(kvcm_py_client=module)
        with patch.dict("sys.modules", {"kv_cache_manager.client.pybind": package}):
            self.assertEqual(caller_node._query_local_node_id(), "")

    def test_resolver_exception_yields_empty_id(self):
        module = MagicMock()
        module.local_node_id.side_effect = RuntimeError("sidecar gone")
        package = SimpleNamespace(kvcm_py_client=module)
        with patch.dict("sys.modules", {"kv_cache_manager.client.pybind": package}):
            self.assertEqual(caller_node._query_local_node_id(), "")


class TestClientInjection(CallerNodeStateMixin):
    def _client(self):
        client = KvCacheManagerClient(
            "http://127.0.0.1:6382", auto_discover_leader=False
        )
        client.session.post = MagicMock(
            return_value=_make_mock_response(_ok_response_json())
        )
        return client

    def test_start_write_cache_reports_caller(self):
        client = self._client()
        try:
            with patch.object(caller_node, "_query_local_node_id", return_value="72"):
                client.start_write_cache({"trace_id": "t1", "instance_id": "i1"})
            sent = client.session.post.call_args.kwargs["json"]
            self.assertEqual(sent["caller"], {"node_id": "72"})
            self.assertEqual(sent["trace_id"], "t1")
        finally:
            client.close()

    def test_get_cache_location_reports_caller(self):
        client = self._client()
        try:
            with patch.object(caller_node, "_query_local_node_id", return_value="72"):
                client.get_cache_location({"trace_id": "t1", "instance_id": "i1"})
            sent = client.session.post.call_args.kwargs["json"]
            self.assertEqual(sent["caller"], {"node_id": "72"})
        finally:
            client.close()

    def test_register_instance_stays_affinity_neutral(self):
        client = self._client()
        try:
            with patch.object(caller_node, "_query_local_node_id", return_value="72"):
                client.register_instance({"trace_id": "t1"})
            sent = client.session.post.call_args.kwargs["json"]
            self.assertNotIn("caller", sent)
        finally:
            client.close()

    def test_no_caller_without_a_local_provider(self):
        client = self._client()
        try:
            with patch.object(caller_node, "_query_local_node_id", return_value=""):
                client.start_write_cache({"trace_id": "t1"})
            sent = client.session.post.call_args.kwargs["json"]
            self.assertEqual(sent, {"trace_id": "t1"})
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
