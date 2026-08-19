"""Unit tests for LocationQueryManager's query fan-out behavior.

The scheduler re-asks ``get_num_new_matched_tokens`` every step while an
async location query is in flight. ``get_locations_for_query`` must absorb
those re-asks: issuing one full ``getCacheLocation`` per engine step while
the answer is on the wire multiplies manager load ~7x per request in the
vLLM 0.22.1 e2e perf harness (24 requests -> 163 HTTP queries).
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from kv_cache_manager.py_connector.test.vllm_stubs import (  # noqa: F401  installs stubs
    _install_stubs,
)
from kv_cache_manager.py_connector.vllm.location_query_manager import (
    LocationQueryManager,
)


class GatedClient:
    """Fake manager client whose getCacheLocation blocks until released."""

    def __init__(self, locations=None):
        self.calls = 0
        self.lock = threading.Lock()
        self.gate = threading.Event()
        self.answer = {"locations": list(locations or ["loc0", "loc1"])}

    def get_cache_location(self, request):
        with self.lock:
            self.calls += 1
        self.gate.wait(timeout=10)
        return self.answer


class LocationQueryFanoutTest(unittest.TestCase):
    MBS_TOKENS = object()  # unused; offsets are in blocks here

    def setUp(self):
        self.client = GatedClient()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.req = SimpleNamespace(request_id="r1", prompt_token_ids=[1, 2, 3])

    def tearDown(self):
        self.executor.shutdown(wait=True)

    def _manager(self, async_mode=True):
        return LocationQueryManager(
            self.client, self.executor, "inst",
            async_get_cache_location=async_mode)

    def test_in_flight_reasks_do_not_fan_out(self):
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # submit
        for _ in range(5):  # scheduler re-asks while the query is on the wire
            self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        time.sleep(0.2)
        self.assertEqual(self.client.calls, 1,
                         "re-asks while in flight must not issue new queries")

        self.client.gate.set()
        deadline = time.time() + 10
        result = None
        while time.time() < deadline:
            result = lqm.get_locations_for_query(self.req, 0)
            if result is not None:
                break
            time.sleep(0.01)
        self.assertEqual(result, ["loc0", "loc1"])
        self.assertEqual(self.client.calls, 1,
                         "cached answer must be served without a new query")

    def test_stale_offset_reissues_once_answered(self):
        lqm = self._manager()
        self.client.gate.set()
        deadline = time.time() + 10
        while time.time() < deadline:
            if lqm.get_locations_for_query(self.req, 0) is not None:
                break
            time.sleep(0.01)
        self.assertEqual(self.client.calls, 1)
        # A new offset invalidates the cached answer and re-issues exactly one
        # query (the documented re-ask path for grown prefixes).
        self.assertIsNone(lqm.get_locations_for_query(self.req, 5))
        self.assertIsNone(lqm.get_locations_for_query(self.req, 5))  # in flight
        time.sleep(0.2)
        self.assertEqual(self.client.calls, 2)

    def test_sync_mode_answers_inline_and_caches(self):
        self.client.gate.set()
        lqm = self._manager(async_mode=False)
        self.assertEqual(lqm.get_locations_for_query(self.req, 0), ["loc0", "loc1"])
        self.assertEqual(lqm.get_locations_for_query(self.req, 0), ["loc0", "loc1"])
        self.assertEqual(self.client.calls, 1)

    def test_failed_query_drops_entry_for_retry(self):
        lqm = self._manager()

        def boom(request):
            raise RuntimeError("manager down")

        self.client.get_cache_location = boom
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        deadline = time.time() + 10
        while time.time() < deadline:  # failure pops the entry asynchronously
            with lqm._lock:
                if not lqm._entries:
                    break
            time.sleep(0.01)
        # The next ask re-issues instead of waiting forever on a dead entry.
        restore = GatedClient.get_cache_location
        self.client.get_cache_location = \
            lambda request: restore(self.client, request)
        self.client.gate.set()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        deadline = time.time() + 10
        result = None
        while time.time() < deadline:
            result = lqm.get_locations_for_query(self.req, 0)
            if result is not None:
                break
            time.sleep(0.01)
        self.assertEqual(result, ["loc0", "loc1"])


if __name__ == "__main__":
    unittest.main()
