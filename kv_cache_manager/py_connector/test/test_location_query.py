"""Unit tests for LocationQueryManager's query fan-out and supersession.

The scheduler re-asks ``get_num_new_matched_tokens`` every step while an
async location query is in flight. ``get_locations_for_query`` must absorb
those re-asks: issuing one full ``getCacheLocation`` per engine step while
the answer is on the wire multiplies manager load ~7x per request in the
vLLM 0.22.1 e2e perf harness (24 requests -> 163 HTTP queries).

The cache holds one slot per request. A re-ask at the same offset waits
for (or serves) the slot; a re-ask at a different offset *supersedes* it
-- the newest ask wins, and the old ask's answer may arrive late but must
never write into the new slot (late answers only beat older asks, never
newer ones).
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
    """Fake manager client: each call blocks on its own gate and answers
    from a per-call answer list (the last answer repeats for extra calls).
    ``open=True`` pre-sets every gate (sync-mode tests, where the fetch
    runs inline and nobody else can release it)."""

    def __init__(self, answers=None, open=False):
        self.calls = 0
        self.lock = threading.Lock()
        self.gates = []
        self.answers = list(answers or [["loc0", "loc1"]])
        self.open = open

    def get_cache_location(self, request):
        with self.lock:
            self.calls += 1
            idx = self.calls - 1
            gate = threading.Event()
            if self.open:
                gate.set()
            self.gates.append(gate)
        gate.wait(timeout=10)
        return {"locations": list(self.answers[min(idx, len(self.answers) - 1)])}


class LocationQueryFanoutTest(unittest.TestCase):
    def setUp(self):
        self.client = GatedClient()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.req = SimpleNamespace(request_id="r1", prompt_token_ids=[1, 2, 3])

    def tearDown(self):
        self.executor.shutdown(wait=True)

    def _manager(self, async_mode=True):
        return LocationQueryManager(
            self.client, self.executor, "inst",
            async_get_cache_location=async_mode)

    def _await_answer(self, lqm, offset):
        """Pump the hook until the slot answers; assert the client was hit."""
        deadline = time.time() + 10
        while time.time() < deadline:
            result = lqm.get_locations_for_query(self.req, offset)
            if result is not None:
                return result
            time.sleep(0.01)
        self.fail(f"query at offset {offset} never answered")

    def _gate(self, idx):
        """Wait for the idx-th call's gate to exist, then return it."""
        deadline = time.time() + 5
        while time.time() < deadline:
            with self.client.lock:
                if len(self.client.gates) > idx:
                    return self.client.gates[idx]
            time.sleep(0.005)
        self.fail(f"gate {idx} was never created (calls={self.client.calls})")

    # ------------------------------------------------------------------ #
    # Dedupe: same offset
    # ------------------------------------------------------------------ #
    def test_in_flight_reasks_do_not_fan_out(self):
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # submit
        for _ in range(5):  # scheduler re-asks while the query is on the wire
            self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        time.sleep(0.2)
        self.assertEqual(self.client.calls, 1,
                         "re-asks while in flight must not issue new queries")

        self._gate(0).set()
        self.assertEqual(self._await_answer(lqm, 0), ["loc0", "loc1"])
        self.assertEqual(self.client.calls, 1,
                         "cached answer must be served without a new query")

    # ------------------------------------------------------------------ #
    # Supersession: a different offset is a different query
    # ------------------------------------------------------------------ #
    def test_superseding_offset_reissues_once_answered(self):
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # submit
        self._gate(0).set()
        self.assertEqual(self._await_answer(lqm, 0), ["loc0", "loc1"])
        self.assertEqual(self.client.calls, 1)
        # A new offset invalidates the cached answer and re-issues exactly one
        # query (the documented re-ask path for grown prefixes).
        self.assertIsNone(lqm.get_locations_for_query(self.req, 5))
        self.assertIsNone(lqm.get_locations_for_query(self.req, 5))  # in flight
        time.sleep(0.2)
        self.assertEqual(self.client.calls, 2)

    def test_superseding_offset_starts_immediately(self):
        # A grown offset must issue its own RPC at once, not wait behind the
        # older offset's in-flight query -- and once superseded, the older
        # offset is no longer addressable: asking it again supersedes back.
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))   # RPC #1
        self.assertIsNone(lqm.get_locations_for_query(self.req, 8))   # RPC #2
        self.assertEqual(self.client.calls, 2)
        self.assertIsNone(lqm.get_locations_for_query(self.req, 8))   # same key: dedup
        self.assertEqual(self.client.calls, 2)
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))   # older offset
        self.assertEqual(self.client.calls, 3,                        # supersedes back
                         "asking the superseded offset must re-issue, not serve")

    def test_superseded_answer_is_not_served_at_its_offset(self):
        # offset 0 answered, then superseded by offset 8: even when offset 8
        # answered too, asking offset 0 again starts a fresh query instead of
        # serving the dead slot's answer.
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # submit
        self._gate(0).set()
        self.assertEqual(self._await_answer(lqm, 0), ["loc0", "loc1"])
        self.assertIsNone(lqm.get_locations_for_query(self.req, 8))   # supersede
        self._gate(1).set()
        self.assertEqual(self._await_answer(lqm, 8), ["loc0", "loc1"])
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))   # fresh RPC
        self._gate(2)  # wait until the fresh query reaches the client
        self.assertEqual(self.client.calls, 3)

    # ------------------------------------------------------------------ #
    # Late answers never beat newer asks
    # ------------------------------------------------------------------ #
    def test_late_answer_of_old_ask_does_not_write_new_slot(self):
        # Old ask answered first: its answer must be dropped, the new slot
        # stays in flight until its own answer lands.
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))   # RPC #1
        self.assertIsNone(lqm.get_locations_for_query(self.req, 8))   # RPC #2
        self._gate(0).set()                                    # old answers
        deadline = time.time() + 5
        while time.time() < deadline and self.client.calls < 2:
            time.sleep(0.01)
        time.sleep(0.1)
        # The new slot must still be in flight (not answered by the old ask).
        self.assertIsNone(lqm.get_locations_for_query(self.req, 8))
        self._gate(1).set()
        self.assertEqual(self._await_answer(lqm, 8), ["loc0", "loc1"])

    def test_late_answer_of_old_ask_does_not_overwrite_new_answer(self):
        # New ask answered first, then the old answer lands: it must not
        # overwrite the new slot's answer.
        lqm = self._manager()
        self.client.answers = [["old_answers"], ["new_answers"]]
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))   # RPC #1
        self.assertIsNone(lqm.get_locations_for_query(self.req, 8))   # RPC #2
        self._gate(1).set()                                    # new answers
        self.assertEqual(self._await_answer(lqm, 8), ["new_answers"])
        self._gate(0).set()                                    # old answers late
        time.sleep(0.2)
        self.assertEqual(lqm.get_locations_for_query(self.req, 8), ["new_answers"],
                         "the superseded ask must not overwrite the new slot")

    # ------------------------------------------------------------------ #
    # Failure and sync mode
    # ------------------------------------------------------------------ #
    def test_sync_mode_answers_inline_and_caches(self):
        self.client = GatedClient(open=True)
        lqm = self._manager(async_mode=False)
        self.assertEqual(lqm.get_locations_for_query(self.req, 0), ["loc0", "loc1"])
        self.assertEqual(lqm.get_locations_for_query(self.req, 0), ["loc0", "loc1"])
        self.assertEqual(self.client.calls, 1)

    def test_failed_query_drops_slot_for_retry(self):
        lqm = self._manager()

        def boom(request):
            raise RuntimeError("manager down")

        self.client.get_cache_location = boom
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        deadline = time.time() + 10  # failure pops the slot asynchronously
        while time.time() < deadline:
            with lqm._lock:
                if not lqm._queries.get("r1"):
                    break
            time.sleep(0.01)
        # The next ask re-issues instead of waiting forever on a dead slot.
        restore = GatedClient.get_cache_location
        self.client.get_cache_location = \
            lambda request: restore(self.client, request)
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # re-issue
        self._gate(0).set()
        self.assertEqual(self._await_answer(lqm, 0), ["loc0", "loc1"])

    # ------------------------------------------------------------------ #
    # Expiry: an old answer is a miss, never a hit
    # ------------------------------------------------------------------ #
    def _expire(self, lqm, req_id="r1"):
        """Age the request's slot far past every horizon."""
        with lqm._lock:
            lqm._queries[req_id].ask_time -= 3600.0

    def test_expired_answer_is_refetched_not_served(self):
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # submit
        self._gate(0).set()
        self.assertEqual(self._await_answer(lqm, 0), ["loc0", "loc1"])
        self._expire(lqm)
        # Expired hit: same key, but not served -- re-issued instead.
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        self._gate(1)  # the re-query reached the client
        self.assertEqual(self.client.calls, 2)
        self.assertEqual(lqm.stale_supersede_count, 1)
        self._gate(1).set()
        self.assertEqual(self._await_answer(lqm, 0), ["loc0", "loc1"])

    def test_fresh_answer_served_without_refetch(self):
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # submit
        self._gate(0).set()
        self.assertEqual(self._await_answer(lqm, 0), ["loc0", "loc1"])
        # Well inside the horizon: re-asks keep serving the cached answer.
        for _ in range(3):
            self.assertEqual(lqm.get_locations_for_query(self.req, 0),
                             ["loc0", "loc1"])
        self.assertEqual(self.client.calls, 1)
        self.assertEqual(lqm.stale_supersede_count, 0)

    def test_expired_answer_refetches_inline_in_sync_mode(self):
        self.client = GatedClient(answers=[["old"], ["new"]], open=True)
        lqm = self._manager(async_mode=False)
        self.assertEqual(lqm.get_locations_for_query(self.req, 0), ["old"])
        self._expire(lqm)
        self.assertEqual(lqm.get_locations_for_query(self.req, 0), ["new"])
        self.assertEqual(self.client.calls, 2)
        self.assertEqual(lqm.stale_supersede_count, 1)

    def test_expiry_holds_at_the_serve_boundary_not_consume(self):
        # Consumption pops the slot regardless of age, so expiry holds at
        # the serve boundary: consume only ever sees a fresh serve's slot.
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # submit
        self._gate(0).set()
        self._await_answer(lqm, 0)
        self._expire(lqm)
        lqm.store_result("r1", ["loc0"])     # the hook's clamp, post-answer
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # expired: miss
        self._gate(1).set()
        self._await_answer(lqm, 0)           # re-answered fresh
        self.assertEqual(lqm.consume_locations("r1"), (["loc0", "loc1"], 0))

    # ------------------------------------------------------------------ #
    # Consume / store / invalidate
    # ------------------------------------------------------------------ #
    def test_consume_returns_answers_and_empties(self):
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # submit
        self._gate(0).set()
        self.assertEqual(self._await_answer(lqm, 0), ["loc0", "loc1"])
        # The hook clamps the answer before the allocation consumes it.
        lqm.store_result("r1", ["loc0"])
        self.assertEqual(lqm.consume_locations("r1"), (["loc0"], 0))
        self.assertIsNone(lqm.consume_locations("r1"))

    def test_consumed_slot_discards_the_inflight_answer(self):
        # consume pops even an in-flight slot; the late answer must not
        # resurrect a slot for a request vLLM already moved past.
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        self.assertIsNone(lqm.consume_locations("r1"))   # pops the in-flight slot
        self._gate(0).set()
        time.sleep(0.2)
        with lqm._lock:
            self.assertNotIn("r1", lqm._queries)
        # The next ask is a fresh query.
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        self._gate(1)  # wait until the fresh query reaches the client
        self.assertEqual(self.client.calls, 2)

    def test_invalidate_drops_slot_and_discards_late_answer(self):
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        lqm.invalidate("r1")
        with lqm._lock:
            self.assertNotIn("r1", lqm._queries)
        self._gate(0).set()
        time.sleep(0.2)
        with lqm._lock:
            self.assertNotIn("r1", lqm._queries,
                             "a late answer must not resurrect the slot")


if __name__ == "__main__":
    unittest.main()
