"""Transport contract test: every API the generator uses works over both
plaintext HTTP and insecure gRPC, against a real KVCM deployment.

The generator's own run exercises RegisterInstance, ReportEvent,
GetCacheLocationsByBackend, StartWriteCache, FinishWriteCache, GetClusterInfo
and CheckHealth; preflight additionally exercises RemoveCache on its own
temporary cold key. The report is the evidence that each of them completed with
the correct request/response types and on the correct service endpoint.
"""
import unittest

from integration_test.swarm.swarm_test_base import SwarmScenarioTest


ALL_APIS = {
    "RegisterInstance",
    "ReportEvent",
    "GetCacheLocationsByBackend",
    "StartWriteCache",
    "FinishWriteCache",
    "GetClusterInfo",
    "CheckHealth",
    "RemoveCache",
}


class SwarmTransportContractTest(SwarmScenarioTest):
    def _assert_contract(self, report, transport):
        observed = {}
        workload_lanes = {}
        for entry in report["rpc"]["by_api_phase"]:
            bucket = observed.setdefault(entry["api"], {"total": 0, "success": 0, "lanes": set()})
            bucket["total"] += entry["total"]
            bucket["success"] += entry["success"]
            bucket["lanes"].add(entry["lane"])
            # Preflight deliberately runs every step on the control lane; lane
            # assignment is a property of the workload behaviors.
            if entry["phase"] != "preflight":
                workload_lanes.setdefault(entry["api"], set()).add(entry["lane"])
        missing = ALL_APIS - set(observed)
        self.assertFalse(missing, "APIs never exercised over %s: %s" % (transport, sorted(missing)))
        for api in ALL_APIS:
            self.assertEqual(observed[api]["total"], observed[api]["success"],
                             "%s failed at least once over %s" % (api, transport))

        # Lane assignment: business carries the workload, control carries
        # heartbeat, leader discovery, health probe and cleanup.
        self.assertEqual(workload_lanes["GetCacheLocationsByBackend"], {"business"})
        self.assertEqual(workload_lanes["CheckHealth"], {"control"})
        self.assertEqual(workload_lanes["GetClusterInfo"], {"control"})
        self.assertIn("business", workload_lanes["ReportEvent"])
        self.assertIn("control", workload_lanes["ReportEvent"])
        self.assertEqual(workload_lanes["StartWriteCache"] - {"business", "control"}, set())
        self.assertNotIn("RemoveCache", workload_lanes,
                         "RemoveCache is reachable from preflight only")

        # meta and admin never share a socket: each context serves one role per
        # endpoint and the admin endpoint is only used by CheckHealth.
        for context in report["transport"]["contexts"]:
            for endpoint in context["endpoints"]:
                self.assertIn(endpoint["role"], ("meta", "admin"))
                self.assertEqual(context["kind"], transport)
        admin_endpoints = {endpoint["endpoint"]
                           for context in report["transport"]["contexts"]
                           for endpoint in context["endpoints"] if endpoint["role"] == "admin"}
        meta_endpoints = {endpoint["endpoint"]
                          for context in report["transport"]["contexts"]
                          for endpoint in context["endpoints"] if endpoint["role"] == "meta"}
        if transport == "http":
            self.assertFalse(admin_endpoints & meta_endpoints)

        # No retry is hidden inside the transport: leader refreshes are counted.
        for process in report["behaviors"]["v6d-a"]["processes"]:
            self.assertEqual(process["reporter"]["not_leader_retry_failures"], 0)

    def test_every_api_works_over_http(self):
        run = self.run_scenario("v6d_normal", expectations="v6d_normal", name_hint="swarm-contract-http")
        self._assert_contract(run.report, "http")

    def test_every_api_works_over_grpc(self):
        run = self.run_scenario(
            "v6d_normal",
            expectations="v6d_normal",
            name_hint="swarm-contract-grpc",
            transport_override="grpc",
        )
        self._assert_contract(run.report, "grpc")


if __name__ == "__main__":
    unittest.main()
