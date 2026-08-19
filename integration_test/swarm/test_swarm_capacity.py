"""Capacity-pressure closed loop: writable and masked cold writes, C3 order."""
import unittest

from integration_test.swarm.swarm_test_base import SwarmScenarioTest


class SwarmCapacityTest(SwarmScenarioTest):
    def test_capacity_pressure_produces_writable_and_masked_evictions(self):
        run = self.run_scenario("v6d_capacity_pressure", expectations="v6d_capacity_pressure",
                                name_hint="swarm-cap")
        report = run.report
        deployment = report["behaviors"]["v6d-a"]
        cache = report["cache"]["v6d-a"]

        # Capacity pressure is real: the cache stayed at its bound and evictions
        # were driven by insert demand, not by a random spill probability.
        self.assertGreater(cache["victims_selected"], 0)
        self.assertGreater(cache["backpressure_waits"], 0,
                           "materialisation must actually have waited for capacity")
        # Every materialisation that waited eventually got its capacity: no
        # insert was abandoned at its deadline.
        self.assertEqual(cache["backpressure_timeouts"], 0)
        self.assertEqual(report["behaviors"]["v6d-a"]["turns"]["insert_failed_backpressure"], 0)
        # Hitting a key that is currently being evicted is a benign lifecycle
        # race, reported separately from real capacity pressure.
        self.assertIn("insert_skipped_evicting", report["behaviors"]["v6d-a"]["turns"])
        # Moments where every resident object was leased are reported as an
        # observation, not treated as an error.
        self.assertIn("no_victim_waits", cache)
        self.assertLessEqual(cache["used_bytes_total"], cache["capacity_bytes_total"])

        writable = 0
        masked = 0
        cold_allocations = 0
        for process in deployment["processes"]:
            eviction = process["eviction"]
            writable += eviction["writable_items"]
            masked += eviction["masked_items"]
            cold_allocations += eviction["cold_allocations_confirmed"]
            # Nothing may be removed locally before its write session closed,
            # and every removal must be followed by a BLOCK_DELETE.
            self.assertEqual(eviction["start_write_unknown"], 0)
            self.assertEqual(eviction["finish_write_unknown"], 0)
            self.assertEqual(eviction["local_removed"],
                             eviction["writable_items"] + eviction["masked_items"])
            self.assertEqual(process["reporter"]["block_delete_items"], eviction["local_removed"])
        self.assertGreater(writable, 0, "some eviction must actually write a cold copy")
        self.assertGreater(masked, 0, "some eviction must be masked by the replica threshold")
        # A masked item must not create a new cold allocation.
        self.assertEqual(cold_allocations, writable)
        self.assertEqual(deployment["expected_locations"]["cold_confirmed"], writable)
        self.assertGreater(deployment["expected_locations"]["cold_confirmed_bytes"], 0)

        c3 = next(check for check in report["invariants"]["checks"]
                  if check["check_name"] == "C3_capacity_pressure_eviction")
        self.assertEqual(c3["status"], "PASS")
        self.assertEqual(c3["counters"]["local_removals_after_write_close"],
                         c3["counters"]["deletes_after_local_removal"])
        self.assertEqual(c3["counters"]["cold_allocations_confirmed"], writable)

        # Normal shutdown never deletes cold allocations and never uses
        # RemoveCache.
        cleanup = report["cleanup"]["behaviors"]["v6d-a"]
        self.assertFalse(cleanup["remove_cache_called"])
        self.assertGreater(cleanup["confirmed_cold_allocations"], 0)
        self.assertEqual(report["cleanup"]["preflight"]["remove_cache_calls"], 1,
                         "only preflight may remove its own temporary cold key")


if __name__ == "__main__":
    unittest.main()
