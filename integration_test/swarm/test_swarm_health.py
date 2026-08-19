"""Independent health probe, both standalone and under business load."""
import unittest

from integration_test.swarm.swarm_test_base import SwarmScenarioTest


class SwarmHealthProbeTest(SwarmScenarioTest):
    def test_health_probe_only_needs_no_v6d_state(self):
        run = self.run_scenario("health_probe_only", expectations="health_probe_only",
                                name_hint="swarm-health")
        report = run.report
        self.assertNotIn("v6d-a", report["behaviors"])
        probe = report["behaviors"]["health-a"]
        self.assertEqual(probe["type"], "health_probe")
        self.assertEqual(probe["streams"], 3)
        self.assertGreater(probe["probes"], 0)
        self.assertEqual(probe["failed"], 0)
        # The probe owns its own transport context on the admin endpoint only.
        contexts = report["transport"]["contexts"]
        health_contexts = [c for c in contexts if c["behavior_id"] == "health-a"]
        self.assertEqual(len(health_contexts), 1)
        self.assertEqual({e["role"] for e in health_contexts[0]["endpoints"]}, {"admin"})
        # health_probe contributes no cache and no workload shape.
        self.assertNotIn("health-a", report["cache"])
        self.assertNotIn("health-a", report["workload_shape"])
        # Every CheckHealth uses the control lane.
        for entry in report["rpc"]["by_api_phase"]:
            if entry["api"] == "CheckHealth":
                self.assertEqual(entry["lane"], "control")

    def test_health_probe_progresses_under_business_saturation_pressure(self):
        run = self.run_scenario("v6d_async_pressure", expectations="v6d_async_pressure",
                                name_hint="swarm-async")
        report = run.report
        probe = report["behaviors"]["health-a"]
        self.assertGreater(probe["probes"], 20)
        self.assertEqual(probe["deadline_exceeded"], 0)

        # Far more in-flight business RPCs than Executor workers, and the
        # control lane still never waited behind them.
        self.assertEqual(report["run_config"]["runtime"]["workers"], 1)
        business = report["runtime"]["admission"]["business"]
        control = report["runtime"]["admission"]["control"]
        self.assertGreater(business["in_flight_peak"], report["run_config"]["runtime"]["workers"],
                           "network waiting must not be bounded by worker count")
        self.assertEqual(control["rejected"], 0)
        self.assertGreater(control["acquired"], 0)

        # Heartbeats and leader discovery kept their own deadlines.
        for process in report["behaviors"]["v6d-a"]["processes"]:
            self.assertGreater(process["reporter"]["heartbeats_sent"], 1)
            self.assertEqual(process["reporter"]["heartbeats_failed"], 0)
            self.assertGreater(process["reporter"]["leader_polls"], 0)
            self.assertEqual(process["reporter"]["leader_poll_failures"], 0)


if __name__ == "__main__":
    unittest.main()
