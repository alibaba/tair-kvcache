"""HTTP normal-workload closed loop against a real KVCM deployment."""
import unittest

from integration_test.swarm.swarm_test_base import SwarmScenarioTest


class SwarmHttpTest(SwarmScenarioTest):
    def test_http_normal_workload_closed_loop(self):
        run = self.run_scenario("v6d_normal", expectations="v6d_normal", name_hint="swarm-http")
        report = run.report

        # Several processes share one instance_id, each with its own reporter
        # identity and its own transport context.
        deployment = report["behaviors"]["v6d-a"]
        self.assertEqual(deployment["process_count"], 3)
        instance_ids = {process["reporter_identity"].split("|")[0] for process in deployment["processes"]}
        self.assertEqual(len(instance_ids), 1, "all processes must share one instance_id")
        hosts = {process["host_ip_port"] for process in deployment["processes"]}
        self.assertEqual(len(hosts), 3, "each process needs a unique reporter address")
        for process in deployment["processes"]:
            self.assertTrue(process["registered"])
            self.assertGreater(process["reporter"]["block_add_confirmed"], 0)
            self.assertEqual(process["reporter"]["block_add_unknown"], 0)
            self.assertEqual(process["reporter"]["block_delete_unknown"], 0)
            self.assertIn("ST_EVENT_REPORT_L2", process["reporter_identity"])

        # Every process owns a separate transport context; meta and admin never
        # share a socket, and there is no per-process network thread.
        contexts = [c for c in report["transport"]["contexts"] if c["behavior_id"] == "v6d-a"]
        self.assertEqual(len(contexts), 3)
        for context in contexts:
            self.assertEqual(context["kind"], "http")
            roles = {endpoint["role"] for endpoint in context["endpoints"]}
            self.assertEqual(roles, {"meta"})
        health_contexts = [c for c in report["transport"]["contexts"] if c["behavior_id"] == "health-a"]
        self.assertEqual(len(health_contexts), 1)
        self.assertEqual({e["role"] for e in health_contexts[0]["endpoints"]}, {"admin"})
        self.assertLessEqual(report["transport"]["io_threads"], 4)

        # warmup and steady are separate buckets over continuous state.
        self.assertIn("warmup", report["phases"])
        self.assertIn("steady", report["phases"])
        steady_lookups = [entry for entry in report["rpc"]["by_api_phase"]
                          if entry["api"] == "GetCacheLocationsByBackend" and entry["phase"] == "steady"]
        warmup_lookups = [entry for entry in report["rpc"]["by_api_phase"]
                          if entry["api"] == "GetCacheLocationsByBackend" and entry["phase"] == "warmup"]
        self.assertTrue(steady_lookups, "steady must carry lookups")
        self.assertTrue(warmup_lookups, "warmup must carry lookups")

        # Per-group facts prove the Full/Mamba key shapes really differ.
        groups = {group["id"]: group for group in deployment["groups"]}
        self.assertEqual(groups["full-0"]["lookup_selector"], "prefix")
        self.assertEqual(groups["full-1"]["lookup_selector"], "coverage")
        self.assertEqual(groups["mamba-0"]["lookup_selector"], "coverage")
        self.assertEqual(groups["full-0"]["spec_name"], "v6d_4096")
        self.assertEqual(groups["mamba-0"]["spec_name"], "v6d_1024")
        self.assertGreater(groups["full-0"]["objects"], groups["mamba-0"]["objects"],
                           "Mamba keys are sparse compared with Full Attention")

        # Cold backends are derived from the registration response.
        self.assertIn("ST_NFS", deployment["cold_backends"])
        self.assertIn("nfs_01", deployment["storage_configs"])

        # Query planning: one independent batch per Full Attention group, at most
        # one merged Mamba COVERAGE batch per turn, and per-key specs.
        turns = deployment["turns"]
        full_groups = sum(1 for group in deployment["groups"] if group["kind"] == "full_attention")
        self.assertEqual(full_groups, 2)
        self.assertGreaterEqual(turns["hot_lookup_batches"], turns["turns"] * full_groups)
        self.assertGreater(turns["mamba_coverage_batches"], 0)
        self.assertLessEqual(turns["mamba_coverage_batches"], turns["turns"])
        self.assertGreater(turns["cold_lookup_batches"], 0, "hot misses must fall back to the cold tier")
        self.assertGreater(turns["materialized"], 0)
        self.assertGreater(turns["sealed"], 0)

        # Connections stay inside the configured, lazily grown pool bounds.
        limits = report["run_config"]["runtime"]["limits"]
        pool_cap = limits["http_connections_per_endpoint"] + limits["http_control_connections_per_endpoint"]
        for context in contexts:
            for endpoint in context["endpoints"]:
                self.assertLessEqual(endpoint["connections_created"], pool_cap)
                self.assertGreater(endpoint["connections_reused"], 0, "connections must be reused")

        # Turn concurrency per process is bounded by the cache capacity and the
        # bound was never exhausted in this scenario.
        for process in deployment["processes"]:
            self.assertGreater(process["turn_capacity"]["capacity_bytes"], 0)
            self.assertEqual(process["turn_capacity"]["timeouts"], 0)
            self.assertLessEqual(
                process["turn_capacity"]["peak_in_use_bytes"],
                process["turn_capacity"]["capacity_bytes"],
            )
        shape = report["workload_shape"]["v6d-a"]
        self.assertGreater(shape["worst_case_turn_working_set_bytes"], 0)
        self.assertGreaterEqual(shape["local_cache_capacity_bytes"], shape["worst_case_turn_working_set_bytes"])


if __name__ == "__main__":
    unittest.main()
