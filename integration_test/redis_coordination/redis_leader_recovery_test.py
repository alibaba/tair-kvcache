import json
import logging
import os
import subprocess
import time
import unittest
import urllib.request

from integration_test.redis_coordination.redis_fault_proxy import RedisFaultProxy
from integration_test.testlib.redis_server import RedisServer
from integration_test.testlib.test_base import TestBase


class RedisLeaderRecoveryTest(TestBase, unittest.TestCase):
    CLUSTER = "redis_leader_recovery"
    LEASE_MS = 2500

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.clean_workdir()
        self.redis = RedisServer(self.get_workdir())
        self.addCleanup(self.redis.stop)
        self.redis.start()
        self.proxy = RedisFaultProxy(self.redis.port)
        self.addCleanup(self.proxy.stop)
        self.proxy.start()
        self.prepare_test_resource(2, work_dir=self.get_workdir())
        # Always release held replies before stopping workers, including on an
        # assertion failure, so cleanup cannot strand a coordination RPC.
        self.addCleanup(self.stop_worker)
        self.addCleanup(self.proxy.restore)

    def _start(self, worker_id, through_proxy=True):
        uri = self.proxy.uri if through_proxy else self.redis.uri
        uri += f"?cluster_name={self.CLUSTER}&timeout_ms=100&retry_count=3&client_max_pool_size=2"
        self.assertTrue(self.worker_manager.start_worker(worker_id, **{
            "kvcm.coordination.uri": "'" + uri + "'",
            "kvcm.leader_elector.node_id": self._node(worker_id),
            "kvcm.leader_elector.lease_ms": self.LEASE_MS,
            "kvcm.leader_elector.loop_interval_ms": 100,
            "kvcm.service.advertised_host": "127.0.0.1",
        }))
        self._wait(lambda: self._status(self._post(worker_id, "checkHealth")) == "OK")

    def _post(self, worker_id, endpoint, data=None, meta=False):
        worker = self.worker_manager.get_worker(worker_id)
        port = worker.env.http_port if meta else worker.env.admin_http_port
        body = dict(data or {}, trace_id="redis_fault_" + endpoint)
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/{endpoint}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.loads(response.read())

    @staticmethod
    def _status(response):
        return response.get("header", {}).get("status", {}).get("code")

    def _wait(self, predicate, timeout=15):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            try:
                last = predicate()
                if last:
                    return last
            except (OSError, ValueError) as error:
                last = str(error)
            time.sleep(0.05)
        self.fail(f"condition did not become true; last result={last}")

    def _wait_ready(self, worker_id):
        def ready():
            health = self._post(worker_id, "checkHealth")
            return (health.get("is_leader") is True
                    and self._status(self._post(worker_id, "listStorage")) == "OK"
                    and self._status(self._report_event(worker_id)) == "OK")
        self._wait(ready)
        self.assertEqual(self._node(worker_id), self.redis.command("GET", self._lock_key()))
        self.assertGreater(self.redis.command("PTTL", self._lock_key()), 0)

    def _wait_closed(self, worker_id):
        self._wait(lambda: self._status(self._post(worker_id, "listStorage")) == "SERVER_NOT_LEADER")
        self.assertFalse(self._post(worker_id, "checkHealth").get("is_leader", False))
        self.assertEqual("SERVER_NOT_LEADER", self._status(self._report_event(worker_id)))
        # Both discovery and health remain usable after the business gates close.
        self.assertEqual("OK", self._status(self._post(worker_id, "getClusterInfo", meta=True)))

    def _report_event(self, worker_id):
        # An empty event batch needs no registered instance or storage backend,
        # but exercises the actual MetaService leader-only request guard.
        return self._post(worker_id, "reportEvent", {
            "instance_id": "redis_fault_probe",
            "host_ip_port": "127.0.0.1:12345",
            "events": [],
        }, meta=True)

    def _assert_single_serving_node(self, expected):
        serving = [worker_id for worker_id in (0, 1)
                   if self._status(self._post(worker_id, "listStorage")) == "OK"]
        self.assertEqual([expected], serving)

    def _log(self, worker_id=0):
        path = os.path.join(self.worker_manager.get_worker(worker_id).env.workdir,
                            "logs", "kv_cache_manager.log")
        with open(path) as log:
            return log.read()

    def _recoveries(self, worker_id=0):
        return self._log(worker_id).count("Server promoted to leader, starting recover")

    def _node(self, worker_id):
        return f"{self.CLUSTER}_node_{worker_id}"

    def _lock_key(self):
        return f"kvcm_{self.CLUSTER}_lock:_TAIR_KVCM_LEADER_KEY"

    def test_external_lock_deletion_during_recovery_waits_for_cleanup_before_reacquiring(self):
        # The driver blocks lifecycle callbacks while using real Redis Lua and
        # an independent DEL connection. No production recovery hook is needed.
        driver = os.path.abspath(os.path.join(os.path.dirname(__file__), "leader_lock_deletion_test_driver"))
        env = dict(os.environ, KVCM_TEST_REDIS_URI=self.redis.uri + "?cluster_name=external_deletion")
        result = subprocess.run([driver], cwd=self.get_workdir(), env=env,
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_single_node_recovers_after_repeated_external_lock_deletion(self):
        self._start(0)
        self._wait_ready(0)
        recoveries = self._recoveries()
        self.assertEqual("OK", self._status(self._post(0, "updateLeaderElectorConfig", {
            "campaign_delay_time_ms": 0,
        })))
        deletions = 0

        def delete_lock():
            nonlocal deletions
            # Count actual deletions rather than repeated DELs of an absent key.
            deletions += self.redis.command("DEL", self._lock_key())
            return deletions >= 10

        self._wait(delete_lock)
        # Injection is now stopped. The key can reappear only after a new
        # acquisition, so an old open gate cannot make this check pass early.
        self._wait(lambda: self.redis.command("GET", self._lock_key()) == self._node(0))
        self._wait_ready(0)
        self.assertGreater(self._recoveries(), recoveries)
        self.assertIn("Server cleanup completed", self._log())
        # Readiness must survive subsequent successful renewal cycles too.
        for _ in range(3):
            expiration = int(self._post(0, "getManagerClusterInfo")["self_leader_expiration_time"])
            self._wait(lambda: int(self._post(0, "getManagerClusterInfo")["self_leader_expiration_time"]) > expiration)
            self._wait_ready(0)

    def test_single_node_recovers_after_repeated_connection_loss(self):
        self._start(0)
        self._wait_ready(0)
        for round_id in range(2):
            recoveries = self._recoveries()
            self.proxy.disconnect()
            self._wait_closed(0)
            self._wait(lambda: self.redis.command("GET", self._lock_key()) is None)
            self.proxy.restore()
            self._wait_ready(0)
            self.assertEqual(recoveries + 1, self._recoveries(), f"round={round_id}")
        self.assertEqual(2, self._log().count("Server cleanup completed"))

    def test_partitioned_leader_yields_to_connected_follower(self):
        self._start(0)
        self._wait_ready(0)
        self._start(1, through_proxy=False)
        recoveries = self._recoveries()
        self.proxy.disconnect()
        self._wait_closed(0)
        self._wait_ready(1)
        self.proxy.restore()
        # Wait until the formerly isolated node has observed the new holder.
        self._wait(lambda: self._post(0, "getManagerClusterInfo").get("leader_node_id") == self._node(1))
        self._assert_single_serving_node(1)
        self.assertEqual(recoveries, self._recoveries())

    def test_lost_successful_renew_reply_is_retried_within_same_lease(self):
        self._start(0)
        self._wait_ready(0)
        recoveries = self._recoveries()
        expiration = int(self._post(0, "getManagerClusterInfo")["self_leader_expiration_time"])
        self.proxy.arm_renew_reply("drop")
        self.assertTrue(self.proxy.fault_entered.wait(timeout=5))
        self.assertTrue(self.proxy.reply_replayed.wait(timeout=5))
        self.assertNotEqual(self.proxy.fault_connection, self.proxy.replay_connection)
        self._wait(lambda: "pipeline fail, connection not ok, try count[0]" in self._log())
        self._wait(lambda: int(self._post(0, "getManagerClusterInfo")["self_leader_expiration_time"]) > expiration)
        self._wait_ready(0)
        self.assertEqual(recoveries, self._recoveries())
        self.assertNotIn("renew lock failed", self._log())
        self.assertNotIn("Server cleanup completed", self._log())

    def test_late_successful_renew_reply_cannot_revive_old_leader_or_delete_new_lock(self):
        self._start(0)
        self._wait_ready(0)
        self._start(1, through_proxy=False)
        self.proxy.arm_renew_reply("hold")
        self.assertTrue(self.proxy.fault_entered.wait(timeout=5))
        self._wait_closed(0)
        self._wait_ready(1)
        self._wait(lambda: "Server cleanup completed" in self._log())
        self.proxy.restore()
        # This is the old renewal's compensating Unlock, executed by real Lua.
        self.assertTrue(self.proxy.unlock_mismatch.wait(timeout=5))
        self.assertEqual(self._node(1), self.redis.command("GET", self._lock_key()))
        self._assert_single_serving_node(1)

        self.assertEqual("OK", self._status(self._post(1, "updateLeaderElectorConfig", {
            "campaign_delay_time_ms": 10000,
        })))
        self.assertEqual("OK", self._status(self._post(1, "leaderDemote")))
        self._wait_ready(0)
        self._assert_single_serving_node(0)
        self.assertEqual(2, self._recoveries())

    def test_redis_restart_reloads_scripts_and_recovers_business_gate(self):
        self._start(0)
        self._wait_ready(0)
        recoveries = self._recoveries()
        original_pid = self.redis.process.pid
        self.redis.stop()
        self._wait_closed(0)
        self._wait(lambda: "Server cleanup completed" in self._log())
        self.redis.start()
        self.assertNotEqual(original_pid, self.redis.process.pid)
        self._wait_ready(0)
        self.assertGreater(self.proxy.noscript_count, 0)
        self.assertEqual(recoveries + 1, self._recoveries())
        self.assertIn("Script not loaded in Redis, reloading", self._log())


if __name__ == "__main__":
    unittest.main()
