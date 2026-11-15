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
from unittest.mock import patch

from kv_cache_manager.py_connector.test.vllm_stubs import (  # noqa: F401  installs stubs
    _install_stubs,
)
from kv_cache_manager.py_connector.vllm import location_query_manager
from kv_cache_manager.py_connector.vllm.location_query_manager import (
    LocationQueryManager,
)


class FakeClock:
    """Stands in for the module's ``time`` (only ``monotonic`` is used), so a
    test can state the manager's latency instead of sleeping it."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


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
            self.client,  # ty: ignore[invalid-argument-type]  (test double)
            self.executor,
            "inst",
            async_get_cache_location=async_mode,
        )

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

    def _await_landing(self, lqm, req_id="r1"):
        """Wait for the answer to reach the slot *without* asking the hook:
        an ask could supersede the very slot being watched."""
        deadline = time.time() + 10
        while time.time() < deadline:
            with lqm._lock:
                q = lqm._queries.get(req_id)
                if q is not None and q.locations is not None:
                    return
            time.sleep(0.005)
        self.fail("the manager's answer never landed in the slot")

    def _slow_answer(self, lqm, call_index, offset=0, seconds=1.1):
        """Drive one real slow RPC, including the hook call that issues it.

        The hook call also observes (and counts) the previous answer's expiry.
        Sleeping while the RPC is genuinely in flight proves the age gate fired
        because of manager latency, not because a test rewrote ``ask_time``.
        """
        issued = time.monotonic()
        self.assertIsNone(lqm.get_locations_for_query(self.req, offset))
        gate = self._gate(call_index)
        time.sleep(seconds)
        gate.set()
        self._await_landing(lqm)
        latency = time.monotonic() - issued
        self.assertGreater(latency, lqm._max_answer_age_s)
        return latency

    def _slow_manager(self, lqm, seconds):
        """Issue the first query and let the manager sit on it in real time."""
        return self._slow_answer(lqm, 0, seconds=seconds)

    # ------------------------------------------------------------------ #
    # Dedupe: same offset
    # ------------------------------------------------------------------ #
    def test_in_flight_reasks_do_not_fan_out(self):
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # submit
        for _ in range(5):  # scheduler re-asks while the query is on the wire
            self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        time.sleep(0.2)
        self.assertEqual(
            self.client.calls, 1, "re-asks while in flight must not issue new queries"
        )

        self._gate(0).set()
        self.assertEqual(self._await_answer(lqm, 0), ["loc0", "loc1"])
        self.assertEqual(
            self.client.calls, 1, "cached answer must be served without a new query"
        )

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
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # RPC #1
        self.assertIsNone(lqm.get_locations_for_query(self.req, 8))  # RPC #2
        self.assertEqual(self.client.calls, 2)
        self.assertIsNone(lqm.get_locations_for_query(self.req, 8))  # same key: dedup
        self.assertEqual(self.client.calls, 2)
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # older offset
        self.assertEqual(
            self.client.calls,
            3,  # supersedes back
            "asking the superseded offset must re-issue, not serve",
        )

    def test_superseded_answer_is_not_served_at_its_offset(self):
        # offset 0 answered, then superseded by offset 8: even when offset 8
        # answered too, asking offset 0 again starts a fresh query instead of
        # serving the dead slot's answer.
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # submit
        self._gate(0).set()
        self.assertEqual(self._await_answer(lqm, 0), ["loc0", "loc1"])
        self.assertIsNone(lqm.get_locations_for_query(self.req, 8))  # supersede
        self._gate(1).set()
        self.assertEqual(self._await_answer(lqm, 8), ["loc0", "loc1"])
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # fresh RPC
        self._gate(2)  # wait until the fresh query reaches the client
        self.assertEqual(self.client.calls, 3)

    # ------------------------------------------------------------------ #
    # Late answers never beat newer asks
    # ------------------------------------------------------------------ #
    def test_late_answer_of_old_ask_does_not_write_new_slot(self):
        # Old ask answered first: its answer must be dropped, the new slot
        # stays in flight until its own answer lands.
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # RPC #1
        self.assertIsNone(lqm.get_locations_for_query(self.req, 8))  # RPC #2
        self._gate(0).set()  # old answers
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
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # RPC #1
        self.assertIsNone(lqm.get_locations_for_query(self.req, 8))  # RPC #2
        self._gate(1).set()  # new answers
        self.assertEqual(self._await_answer(lqm, 8), ["new_answers"])
        self._gate(0).set()  # old answers late
        time.sleep(0.2)
        self.assertEqual(
            lqm.get_locations_for_query(self.req, 8),
            ["new_answers"],
            "the superseded ask must not overwrite the new slot",
        )

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

        self.client.get_cache_location = boom  # ty: ignore[invalid-assignment]
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        deadline = time.time() + 10  # failure pops the slot asynchronously
        while time.time() < deadline:
            with lqm._lock:
                if not lqm._queries.get("r1"):
                    break
            time.sleep(0.01)
        # The next ask re-issues instead of waiting forever on a dead slot.
        restore = GatedClient.get_cache_location
        self.client.get_cache_location = (  # ty: ignore[invalid-assignment]
            lambda request: restore(self.client, request)
        )
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
            self.assertEqual(lqm.get_locations_for_query(self.req, 0), ["loc0", "loc1"])
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
        lqm.store_result("r1", ["loc0"])  # the hook's clamp, post-answer
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # expired: miss
        self._gate(1).set()
        self._await_answer(lqm, 0)  # re-answered fresh
        self.assertEqual(lqm.consume_locations("r1"), (["loc0", "loc1"], 0))

    # ------------------------------------------------------------------ #
    # Expiry observation: only the fourth consecutive non-empty hit is ERROR
    # ------------------------------------------------------------------ #
    def test_real_slow_manager_errors_from_fourth_consecutive_expiry(self):
        lqm = self._manager()  # the shipped one-second horizon, untouched
        with patch.object(location_query_manager.logger, "error") as error:
            self._slow_answer(lqm, 0)  # initial answer, not yet an expiry
            for count in range(1, 6):
                self._slow_answer(lqm, count)
                self.assertEqual(error.call_count, max(0, count - 3))
                if count < 4:
                    key = lqm._queries["r1"].key
                    self.assertEqual(lqm._expired_streaks, {key: count})

        self.assertEqual(lqm._max_answer_age_s, 1.0)
        self.assertEqual(lqm.stale_supersede_count, 5)
        self.assertEqual(error.call_count, 2)
        for call, count in zip(error.call_args_list, [4, 5], strict=True):
            message = call.args[0] % call.args[1:]
            self.assertIn("req_id=r1", message)
            self.assertIn("computed_blocks=0", message)
            self.assertIn("offset=0", message)
            self.assertIn(f"count={count}", message)

    def test_real_slow_manager_fresh_answer_resets_expiry_streak(self):
        lqm = self._manager()
        with patch.object(location_query_manager.logger, "error") as error:
            self._slow_answer(lqm, 0)
            self._slow_answer(lqm, 1)  # first consecutive expiry

            # The second expiry is re-answered immediately. Observing that
            # fresh answer must break the streak before it reaches ERROR.
            self.client.open = True
            self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
            self._gate(2)
            self._await_landing(lqm)
            self.assertEqual(lqm.get_locations_for_query(self.req, 0), ["loc0", "loc1"])
            self.assertEqual(lqm._expired_streaks, {})
            self.client.open = False
            error.assert_not_called()
            time.sleep(1.1)  # let the fresh answer become genuinely expired

            # A completely new streak needs four expiries of its own. Without
            # the fresh reset, the second call here would already be #4.
            for count in range(1, 5):
                self._slow_answer(lqm, count + 2)
                self.assertEqual(error.call_count, int(count == 4))
            self.assertIn("count=4", error.call_args.args[0] % error.call_args.args[1:])

    def test_real_slow_manager_expiry_streaks_are_isolated_by_offset(self):
        lqm = self._manager()
        with patch.object(location_query_manager.logger, "error") as error:
            self._slow_answer(lqm, 0, offset=0)
            for count in range(1, 4):
                self._slow_answer(lqm, count, offset=0)
            old_key = lqm._queries["r1"].key
            self.assertEqual(lqm._expired_streaks, {old_key: 3})
            error.assert_not_called()

            # Merely changing offset retires offset 0's counter. The first
            # expiry at offset 8 is #1, never a continuation of the old #3.
            self._slow_answer(lqm, 4, offset=8)
            self.assertEqual(lqm._expired_streaks, {})
            self._slow_answer(lqm, 5, offset=8)
            new_key = lqm._queries["r1"].key
            self.assertNotEqual(new_key, old_key)
            self.assertEqual(lqm._expired_streaks, {new_key: 1})
            error.assert_not_called()

    def test_successful_consume_and_invalidate_clear_expiry_streak(self):
        lqm = self._manager()
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        self._gate(0).set()
        self.assertEqual(self._await_answer(lqm, 0), ["loc0", "loc1"])
        self._expire(lqm)
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        key = lqm._queries["r1"].key
        self.assertEqual(lqm._expired_streaks, {key: 1})
        self._gate(1).set()
        self._await_landing(lqm)
        self.assertEqual(lqm.consume_locations("r1"), (["loc0", "loc1"], 0))
        self.assertEqual(lqm._expired_streaks, {})

        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        self._gate(2).set()
        self.assertEqual(self._await_answer(lqm, 0), ["loc0", "loc1"])
        self._expire(lqm)
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
        key = lqm._queries["r1"].key
        self.assertEqual(lqm._expired_streaks, {key: 1})
        lqm.invalidate("r1")
        self.assertEqual(lqm._expired_streaks, {})
        self._gate(3).set()

    # ------------------------------------------------------------------ #
    # Empty answers: no pointer, hence no expiry
    # ------------------------------------------------------------------ #
    def test_empty_answer_served_when_manager_is_slower_than_the_horizon(self):
        # A delayed fake RPC verifies that an empty answer is not re-queried
        # merely because it became older than the freshness horizon.
        self.client.answers = [[]]
        lqm = self._manager()  # default horizon
        self.assertEqual(lqm._max_answer_age_s, 1.0)
        self._slow_manager(lqm, 1.3)  # delayed fake RPC
        # [] -- not None -- is what keeps the scheduler out of deferred.
        for _ in range(5):
            self.assertEqual(lqm.get_locations_for_query(self.req, 0), [])
        self.assertEqual(self.client.calls, 1, "an empty miss must not re-query")
        self.assertEqual(lqm.stale_supersede_count, 0)
        self.assertEqual(lqm.consume_locations("r1"), ([], 0))

    def test_nonempty_answer_at_the_same_latency_is_still_refetched(self):
        # The control for the test above: identical >1s latency, but the
        # answer carries pointers the manager may have reclaimed and
        # re-allocated, so the age gate must still fire.
        lqm = self._manager()
        self._slow_manager(lqm, 1.3)
        self.assertIsNone(lqm.get_locations_for_query(self.req, 0))  # expired
        self._gate(1)  # the re-query reached the client
        self.assertEqual(self.client.calls, 2)
        self.assertEqual(lqm.stale_supersede_count, 1)
        self._gate(1).set()

    def test_empty_answer_never_expires_however_old(self):
        # Same exemption on a clock the test owns: hours past the horizon,
        # no wall-clock margin involved.
        self.client.answers = [[]]
        lqm = self._manager()
        clock = FakeClock()
        with patch.object(location_query_manager, "time", clock):
            self.assertIsNone(lqm.get_locations_for_query(self.req, 0))
            clock.advance(3600.0)  # the manager took an hour to answer
            self._gate(0).set()
            self._await_landing(lqm)
            self.assertEqual(lqm.get_locations_for_query(self.req, 0), [])
            self.assertEqual(self.client.calls, 1)
            self.assertEqual(lqm.stale_supersede_count, 0)

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
        self.assertIsNone(lqm.consume_locations("r1"))  # pops the in-flight slot
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
            self.assertNotIn(
                "r1", lqm._queries, "a late answer must not resurrect the slot"
            )


if __name__ == "__main__":
    unittest.main()
