"""Drain, teardown, residual reporting and failure-path exit codes."""
import json
import os
import unittest

from integration_test.swarm import evaluator, runner
from integration_test.swarm.swarm_test_base import SwarmScenarioTest


class SwarmDrainTest(SwarmScenarioTest):
    def test_drain_order_flush_host_down_and_residual_reporting(self):
        run = self.run_scenario("v6d_capacity_pressure", expectations="v6d_capacity_pressure",
                                name_hint="swarm-drain")
        report = run.report
        deployment = report["behaviors"]["v6d-a"]
        cleanup = report["cleanup"]["behaviors"]["v6d-a"]

        # Drain flushes what is still local, then reports HOST_DOWN, and every
        # process really goes down.
        self.assertEqual(cleanup["host_down_attempted"], deployment["process_count"])
        self.assertEqual(cleanup["host_down_succeeded"], deployment["process_count"])
        self.assertEqual(cleanup["unflushed_local_objects"], 0)
        self.assertEqual(cleanup["unflushed_local_bytes"], 0)
        self.assertGreater(cleanup["shutdown_flush_objects"], 0)

        # No hot location is left in an unresolved state and no expected
        # location stays pending.
        expected = deployment["expected_locations"]
        self.assertEqual(expected["hot_pending_create"], 0)
        self.assertEqual(expected["hot_pending_delete"], 0)
        self.assertEqual(expected["hot_unknown"], 0)
        self.assertEqual(expected["hot_confirmed"], 0,
                         "HOST_DOWN retires every hot location the reporter owned")
        self.assertEqual(expected["unresolved_preview"], [])

        # Cold allocations survive a normal shutdown on purpose and are reported
        # as residue for the fixture to clean up.
        self.assertGreater(cleanup["confirmed_cold_allocations"], 0)
        self.assertTrue(report["run"]["drain_complete"])
        self.assertTrue(report["run"]["quiesced"])

        # Threads, connections and timers are released by the time the report is
        # written: only the reporting thread plus the fixed I/O pools remain.
        self.assertLessEqual(report["runtime"]["resource_usage"]["threads"], 16)

        # Drained runs still emit both the JSON facts and the human summary.
        self.assertTrue(os.path.exists(os.path.join(self.swarm_workdir, "summary.md")))
        with open(os.path.join(self.swarm_workdir, "summary.md")) as handle:
            summary = handle.read()
        self.assertIn("metadata-only", summary)
        self.assertIn("C3_capacity_pressure_eviction", summary)

    def test_registration_failure_aborts_initialize_with_a_bounded_exit_code(self):
        fixture = self.make_fixture(name_hint="swarm-badgroup")
        # Preflight is disabled so the failure surfaces in initialize: no
        # process can register into a group that does not exist on the server.
        config_path = fixture.render_config(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios", "v6d_normal.json"),
            out_name="bad_group.json",
            overrides={
                "preflight": False,
                "runtime": {"warmup": "1s", "steady": "1s", "drain_timeout": "5s"},
                "target": {"instance_groups": {fixture.instance_group: {"quota_bytes": fixture.quota_bytes},
                                               "swarm-nonexistent-group": {"quota_bytes": 1}}},
                "behaviors": [{"id": "v6d-a", "type": "v6d_deployment", "transport": "http",
                               "config": {"instance_group": "swarm-nonexistent-group"}}],
            })
        # render_config replaces behaviors wholesale, so rebuild the full list.
        with open(config_path) as handle:
            config = json.load(handle)
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios", "v6d_normal.json")
        with open(base_path) as handle:
            base = json.load(handle)
        config["behaviors"] = base["behaviors"]
        config["behaviors"][0]["config"]["instance_group"] = "swarm-nonexistent-group"
        config["behaviors"][0]["config"]["instance_id"] = fixture.instance_id_prefix + "-bad"
        with open(config_path, "w") as handle:
            json.dump(config, handle, indent=2)

        run = runner.run_swarm(config_path, timeout_seconds=240)
        # Exit code 4 means initialize failed, distinct from a configuration or
        # preflight problem.
        self.assertEqual(run.exit_code, 4, run.describe())
        self.assertIsNotNone(run.report)
        self.assertFalse(run.report["run"]["initialize_ok"])
        self.assertGreater(run.report["behaviors"]["v6d-a"]["register_failures"], 0)
        for process in run.report["behaviors"]["v6d-a"]["processes"]:
            self.assertFalse(process["registered"])
        # No session was ever admitted, so no workload traffic was produced.
        self.assertEqual(run.report["behaviors"]["v6d-a"]["sessions"]["admitted"], 0)
        self.assertEqual(run.report["behaviors"]["v6d-a"]["turns"]["turns"], 0)

    def test_unreachable_endpoint_fails_preflight_with_a_bounded_exit_code(self):
        fixture = self.make_fixture(name_hint="swarm-dead")
        config_path = fixture.render_config(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios", "health_probe_only.json"),
            out_name="dead_endpoint.json",
            overrides={"target": {"endpoints": {"admin_http": "http://127.0.0.1:9"}}})
        run = runner.run_swarm(config_path, timeout_seconds=180)
        # Exit code 3 means "the precondition failed", not "the workload failed".
        self.assertEqual(run.exit_code, 3, run.describe())
        self.assertIsNotNone(run.report, "a failed preflight must still produce a report")
        preflight = run.report["cleanup"]["preflight"]
        self.assertFalse(preflight["passed"])
        self.assertEqual(preflight["failure_stage"], "admin_endpoint_check_health")
        # The transport error is classified and the raw error is preserved.
        errors = {}
        for entry in run.report["rpc"]["by_api_phase"]:
            if entry["api"] == "CheckHealth":
                errors.update(entry["transport_errors"])
        self.assertTrue(errors, "the failed CheckHealth must be classified")
        self.assertTrue(set(errors) & {"connect", "disconnect", "timeout"}, errors)
        # And the evaluator fails closed on such a run.
        expectations = evaluator.load_expectations(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "expectations", "health_probe_only.json"))
        evaluation = evaluator.evaluate(run, expectations)
        self.assertFalse(evaluation.ok)
        self.assertIn("preflight did not pass", evaluation.describe())


if __name__ == "__main__":
    unittest.main()
