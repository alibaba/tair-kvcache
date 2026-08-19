"""The generator runs from JSON alone, with no Python in the loop.

The fixture only prepares the isolated environment; the run itself is driven by
the binary reading a single JSON file, exactly as a deployment validation run
would do it.
"""
import json
import os
import unittest

from integration_test.swarm import runner
from integration_test.swarm.swarm_test_base import SwarmScenarioTest


class SwarmDirectJsonTest(SwarmScenarioTest):
    def test_hand_written_json_drives_a_complete_run(self):
        fixture = self.make_fixture(name_hint="swarm-direct")
        # A configuration written by hand, not rendered from a CI scenario.
        config = {
            "name": "hand-written-direct",
            "seed": 991,
            "runtime": {
                "warmup": "1s",
                "steady": "4s",
                "drain_timeout": "10s",
                "workers": 2,
                "limits": {"max_in_flight_business_rpcs": 128, "max_in_flight_control_rpcs": 16},
            },
            "target": {
                "endpoints": {
                    "meta_http": self.meta_http,
                    "meta_grpc": self.meta_grpc,
                    "admin_http": self.admin_http,
                },
                "instance_groups": {fixture.instance_group: {"quota_bytes": fixture.quota_bytes}},
            },
            "behaviors": [
                {
                    "id": "v6d-direct",
                    "type": "v6d_deployment",
                    "transport": "http",
                    "config": {
                        "process_count": 2,
                        "instance_group": fixture.instance_group,
                        "instance_id": fixture.instance_id_prefix + "-direct",
                        "local_cache": {"capacity_bytes": 262144},
                        "session_arrival": {"rate": 8, "mode": "even"},
                        "session_affinity": 0.5,
                        "limits": {"max_active_sessions": 64},
                        "heartbeat_interval": "2s",
                        "shared_prefix_pool": {"root_count": 4, "prefix_tokens": {"min": 64, "max": 96}},
                        "groups": [
                            {"id": "full-0", "kind": "full_attention", "block_size": 16,
                             "object_size": 4096, "lookup_selector": "prefix"}
                        ],
                        "session_classes": [
                            {"name": "chat", "weight": 1.0, "turns": {"min": 2, "max": 4},
                             "turn_interval": "60ms", "initial_tokens": {"min": 128, "max": 256},
                             "new_tokens_per_turn": 32, "rewrite_tail_tokens": 0,
                             "shared_prefix_probability": 0.5}
                        ],
                    },
                }
            ],
            "evidence": {
                "output_json": os.path.join(self.swarm_workdir, "direct_report.json"),
                "violations_jsonl": os.path.join(self.swarm_workdir, "direct_violations.jsonl"),
            },
        }
        os.makedirs(self.swarm_workdir, exist_ok=True)
        config_path = os.path.join(self.swarm_workdir, "direct.json")
        with open(config_path, "w") as handle:
            json.dump(config, handle, indent=2)

        # Local validation is pure: it neither connects nor sends an RPC.
        code, stdout, stderr = runner.validate_only(config_path)
        self.assertEqual(code, 0, stderr)
        self.assertIn("is valid", stdout)

        run = runner.run_swarm(config_path, timeout_seconds=240)
        self.assertEqual(run.exit_code, 0, run.describe())
        self.assertIsNotNone(run.report)
        self.assertEqual(run.report["run"]["name"], "hand-written-direct")
        self.assertEqual(run.report["rpc"]["success"], run.report["rpc"]["total"])
        self.assertGreater(run.report["behaviors"]["v6d-direct"]["turns"]["turns"], 0)
        # Scalar duration and integer forms normalise to closed intervals.
        effective = run.report["run_config"]["behaviors"]["v6d-direct"]["config"]
        session_class = effective["session_classes"][0]
        self.assertEqual(session_class["turn_interval"]["min"], session_class["turn_interval"]["max"])
        self.assertEqual(session_class["new_tokens_per_turn"]["min"], 32)
        self.assertEqual(session_class["new_tokens_per_turn"]["max"], 32)
        # Advanced defaults are materialised into the effective configuration.
        self.assertEqual(effective["min_replica_count"], 2)
        self.assertEqual(effective["eviction_batch_size"], 128)
        self.assertIn("write_timeout", effective)
        self.assertIn("turn_deadline", effective)

    def test_invalid_configuration_is_rejected_locally_without_touching_the_server(self):
        fixture = self.make_fixture(name_hint="swarm-invalid")
        os.makedirs(self.swarm_workdir, exist_ok=True)
        config_path = os.path.join(self.swarm_workdir, "invalid.json")
        with open(config_path, "w") as handle:
            json.dump({
                "name": "invalid",
                "seed": 1,
                "runtime": {"warmup": "1s", "steady": "1s", "drain_timeout": "1s", "workers": 1},
                "target": {
                    "endpoints": {"meta_http": "https://127.0.0.1:1", "meta_grpc": "127.0.0.1:2",
                                  "admin_http": "http://127.0.0.1:3"},
                    "instance_groups": {fixture.instance_group: {"quota_bytes": 1}},
                },
                "behaviors": [{"id": "h", "type": "health_probe", "transport": "http",
                               "config": {"interval": "1s", "mystery": 1}}],
                "evidence": {"output_json": "r.json", "violations_jsonl": "v.jsonl"},
            }, handle)
        code, _stdout, stderr = runner.validate_only(config_path)
        self.assertEqual(code, 2)
        self.assertIn("HTTPS/TLS endpoints are not supported", stderr)
        self.assertIn("unknown configuration field", stderr)
        self.assertFalse(os.path.exists("r.json"))


if __name__ == "__main__":
    unittest.main()
