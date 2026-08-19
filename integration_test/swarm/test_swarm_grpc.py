"""gRPC normal-workload closed loop against a real KVCM deployment."""
import unittest

from integration_test.swarm.swarm_test_base import SwarmScenarioTest


class SwarmGrpcTest(SwarmScenarioTest):
    def test_grpc_normal_workload_closed_loop(self):
        run = self.run_scenario(
            "v6d_normal",
            expectations="v6d_normal",
            name_hint="swarm-grpc",
            transport_override="grpc",
        )
        report = run.report
        contexts = [c for c in report["transport"]["contexts"] if c["behavior_id"] == "v6d-a"]
        self.assertEqual(len(contexts), 3)
        for context in contexts:
            self.assertEqual(context["kind"], "grpc")
            # One channel per unique endpoint, with concurrent RPCs multiplexed.
            for endpoint in context["endpoints"]:
                self.assertEqual(endpoint["channels"], 1)
        # The gRPC completion-queue threads are a small fixed pool.
        self.assertLessEqual(report["transport"]["io_threads"], 4)
        self.assertGreater(report["behaviors"]["v6d-a"]["turns"]["turns"], 0)


if __name__ == "__main__":
    unittest.main()
