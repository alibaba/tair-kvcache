import json
import logging
import os
import time
import unittest
import urllib.error
import urllib.request

from integration_test.testlib.redis_server import RedisServer
from integration_test.testlib.test_base import TestBase


class KVCMHttpClient(object):
    def __init__(self, base_url):
        self.base_url = base_url

    def close(self):
        pass

    def _post(self, endpoint, data, check_response=True):
        body = json.dumps(data).encode()
        request = urllib.request.Request(
            self.base_url + endpoint,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                response_data = json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            raise AssertionError(f"Request to {endpoint} failed with status code {e.code}") from e
        except urllib.error.URLError as e:
            raise AssertionError(f"Request to {endpoint} failed: {e}") from e

        if check_response:
            status = response_data.get("header", {}).get("status", {})
            if status.get("code") != "OK":
                raise AssertionError(f"Request to {endpoint} failed: {status.get('message')}")
        return response_data

    def check_health(self, data, check_response=True):
        return self._post("/api/checkHealth", data, check_response)

    def get_manager_cluster_info(self, data, check_response=True):
        return self._post("/api/getManagerClusterInfo", data, check_response)

    def leader_demote(self, data, check_response=True):
        return self._post("/api/leaderDemote", data, check_response)

    def update_leader_elector_config(self, data, check_response=True):
        return self._post("/api/updateLeaderElectorConfig", data, check_response)

    def get_instance_group(self, data, check_response=True):
        return self._post("/api/getInstanceGroup", data, check_response)

    def create_instance_group(self, data, check_response=True):
        return self._post("/api/createInstanceGroup", data, check_response)

    def register_instance(self, data, check_response=True):
        return self._post("/api/registerInstance", data, check_response)

    def start_write_cache(self, data, check_response=True):
        return self._post("/api/startWriteCache", data, check_response)

    def finish_write_cache(self, data, check_response=True):
        return self._post("/api/finishWriteCache", data, check_response)

    def get_cache_location(self, data, check_response=True):
        return self._post("/api/getCacheLocation", data, check_response)


class RedisClusterIsolationTest(TestBase, unittest.TestCase):
    CLUSTER_A = "redis_it_cluster_a"
    CLUSTER_B = "redis_it_cluster_b"
    CLUSTERS = (CLUSTER_A, CLUSTER_B)
    CLUSTER_WORKERS = {
        CLUSTER_A: (0, 1),
        CLUSTER_B: (2, 3),
    }

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.clean_workdir()
        self._redis = RedisServer(self.get_workdir())
        self.addCleanup(self._redis.stop)
        self._redis.start()
        self.prepare_test_resource(4, work_dir=self.get_workdir())
        self.addCleanup(self.stop_worker)
        self._startup_configs = self._write_startup_configs()
        self.start_worker()

    def start_worker(self, **kwargs):
        for cluster in self.CLUSTERS:
            for node_idx, worker_id in enumerate(self.CLUSTER_WORKERS[cluster]):
                worker_kwargs = dict(kwargs)
                worker_kwargs.update({
                    "kvcm.registry_storage.uri": self._shell_arg(self._cluster_redis_uri(cluster)),
                    "kvcm.coordination.uri": self._shell_arg(self._cluster_redis_uri(cluster)),
                    "kvcm.leader_elector.node_id": f"{cluster}_node_{node_idx}",
                    "kvcm.leader_elector.lease_ms": 5000,
                    "kvcm.leader_elector.loop_interval_ms": 100,
                    "kvcm.service.advertised_host": "127.0.0.1",
                    "kvcm.service.custom_info": f"{cluster}-worker-{node_idx}",
                    "kvcm.startup_config": self._shell_arg(self._startup_configs[cluster]),
                })
                self.assertTrue(self.worker_manager.start_worker(worker_id, **worker_kwargs))

    def test_two_clusters_share_one_redis_and_stay_isolated(self):
        admin_clients = {i: self._admin_client(i) for i in range(4)}
        meta_clients = {i: self._meta_client(i) for i in range(4)}
        for client in list(admin_clients.values()) + list(meta_clients.values()):
            self.addCleanup(client.close)

        leaders = {}
        for cluster in self.CLUSTERS:
            leaders[cluster] = self._wait_for_cluster_leader(cluster, admin_clients)
            self._assert_cluster_info(cluster, leaders[cluster], admin_clients)

        written_instances = []
        for cluster in self.CLUSTERS:
            leader_id = leaders[cluster]
            admin_client = admin_clients[leader_id]
            meta_client = meta_clients[leader_id]

            self._wait_for_instance_group(admin_client, self._group_name(cluster, 0))
            self._create_instance_group(admin_client, cluster, 1)

            for group_idx in (0, 1):
                instance_id = self._instance_id(cluster, group_idx)
                self._write_cache_meta(meta_client, instance_id, self._group_name(cluster, group_idx), group_idx)
                written_instances.append(instance_id)

        self._assert_redis_key_isolation(written_instances)

    def test_leader_demote_only_affects_its_cluster(self):
        admin_clients = {i: self._admin_client(i) for i in range(4)}
        for client in admin_clients.values():
            self.addCleanup(client.close)

        leader_a = self._wait_for_cluster_leader(self.CLUSTER_A, admin_clients)
        leader_b = self._wait_for_cluster_leader(self.CLUSTER_B, admin_clients)
        follower_a = self._other_worker(self.CLUSTER_A, leader_a)

        update_resp = admin_clients[leader_a].update_leader_elector_config({
            "trace_id": "trace_demote_cluster_a_config",
            "campaign_delay_time_ms": 10000,
        })
        self.assertEqual(update_resp["header"]["status"]["code"], "OK")

        demote_resp = admin_clients[leader_a].leader_demote({
            "trace_id": "trace_demote_cluster_a",
        })
        self.assertEqual(demote_resp["header"]["status"]["code"], "OK")

        self._wait_for_cluster_leader(self.CLUSTER_A, admin_clients, expected_leader=follower_a)
        self._assert_cluster_leader(self.CLUSTER_B, leader_b, admin_clients)

        self.assertEqual(
            self._node_id(self.CLUSTER_A, follower_a),
            self._redis.command("GET", f"kvcm_{self.CLUSTER_A}_lock:_TAIR_KVCM_LEADER_KEY"),
        )
        self.assertEqual(
            self._node_id(self.CLUSTER_B, leader_b),
            self._redis.command("GET", f"kvcm_{self.CLUSTER_B}_lock:_TAIR_KVCM_LEADER_KEY"),
        )

    def _write_startup_configs(self):
        result = {}
        for cluster in self.CLUSTERS:
            path = os.path.join(self.workdir, f"{cluster}_startup.json")
            with open(path, "w") as f:
                json.dump({
                    "storage_config": self._storage_config(cluster),
                    "instance_group": self._instance_group(cluster, 0),
                }, f)
            result[cluster] = path
        return result

    def _storage_config(self, cluster):
        return {
            "type": "file",
            "global_unique_name": self._storage_name(cluster),
            "storage_spec": {
                "root_path": os.path.join(self.workdir, cluster, "nfs") + "/",
                "key_count_per_file": 8,
            },
        }

    def _instance_group(self, cluster, group_idx, for_admin=False):
        return {
            "name": self._group_name(cluster, group_idx),
            "storage_candidates": [self._storage_name(cluster)],
            "global_quota_group_name": f"{cluster}_quota_group",
            "max_instance_count": 10,
            "quota": {
                "capacity": 30000000000,
                "quota_config": [
                    {
                        "storage_type": 4 if for_admin else "file",
                        "capacity": 30000000000,
                    },
                ],
            },
            "cache_config": {
                "reclaim_strategy": {
                    "storage_unique_name": self._storage_name(cluster),
                    "reclaim_policy": 1,
                    "trigger_strategy": {
                        "used_percentage": 0.8,
                    },
                    "delay_before_delete_ms": 1000,
                },
                "data_storage_strategy" if for_admin else "cache_prefer_strategy": 2,
                "meta_indexer_config": {
                    "max_key_count": 1000000,
                    "mutex_shard_num": 16,
                    "batch_key_size": 16,
                    "meta_storage_backend_config": {
                        "storage_type": "redis",
                        "storage_uri": self._redis.uri,
                    },
                    "meta_cache_policy_config": {
                        "type": "LRU",
                        "capacity": 10000,
                        "cache_shard_bits": 0,
                        "high_pri_pool_ratio": 0.0,
                    },
                },
            },
            "user_data": f"{cluster}-group-{group_idx}",
            "version": 1,
        }

    def _create_instance_group(self, admin_client, cluster, group_idx):
        resp = admin_client.create_instance_group({
            "trace_id": f"trace_create_{cluster}_{group_idx}",
            "instance_group": self._instance_group(cluster, group_idx, for_admin=True),
        })
        self.assertEqual(resp["header"]["status"]["code"], "OK")

    def _write_cache_meta(self, meta_client, instance_id, instance_group, group_idx):
        block_base = 100000 + group_idx * 1000 + (0 if self.CLUSTER_A in instance_id else 10000)
        block_keys = [block_base, block_base + 1]
        token_ids = [block_base + 10, block_base + 11]

        register_resp = meta_client.register_instance({
            "trace_id": f"trace_register_{instance_id}",
            "instance_group": instance_group,
            "instance_id": instance_id,
            "block_size": 128,
            "model_deployment": {
                "model_name": f"{instance_id}_model",
                "dtype": "FP16",
                "use_mla": False,
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": 1,
            },
            "location_spec_infos": [
                {
                    "name": "tp0",
                    "size": 1024,
                },
            ],
        })
        self.assertEqual(register_resp["header"]["status"]["code"], "OK")

        start_resp = meta_client.start_write_cache({
            "trace_id": f"trace_start_{instance_id}",
            "instance_id": instance_id,
            "block_keys": block_keys,
            "token_ids": token_ids,
            "write_timeout_seconds": 30,
        })
        self.assertIn("write_session_id", start_resp)
        self.assertEqual(2, len(start_resp["locations"]))

        finish_resp = meta_client.finish_write_cache({
            "trace_id": f"trace_finish_{instance_id}",
            "instance_id": instance_id,
            "write_session_id": start_resp["write_session_id"],
            "success_blocks": {
                "bool_masks": {
                    "values": [True, True],
                },
            },
        })
        self.assertEqual(finish_resp["header"]["status"]["code"], "OK")

        location_resp = meta_client.get_cache_location({
            "trace_id": f"trace_location_{instance_id}",
            "query_type": "QT_PREFIX_MATCH",
            "block_keys": block_keys,
            "instance_id": instance_id,
            "block_mask": {
                "offset": 0,
            },
        })
        self.assertEqual(2, len(location_resp["locations"]))

    def _wait_for_cluster_leader(self, cluster, admin_clients, timeout=30, expected_leader=None):
        worker_ids = self.CLUSTER_WORKERS[cluster]
        deadline = time.time() + timeout
        last_states = {}
        while time.time() < deadline:
            leaders = []
            for worker_id in worker_ids:
                try:
                    resp = admin_clients[worker_id].check_health({
                        "trace_id": f"trace_wait_{cluster}_{worker_id}_{int(time.time())}",
                    }, check_response=False)
                    last_states[worker_id] = resp
                    if (resp.get("header", {}).get("status", {}).get("code") == "OK"
                            and resp.get("is_health") is True
                            and resp.get("is_leader") is True):
                        leaders.append(worker_id)
                except Exception as e:
                    last_states[worker_id] = str(e)
            if len(leaders) == 1 and (expected_leader is None or leaders[0] == expected_leader):
                return leaders[0]
            time.sleep(0.5)
        self.fail(f"cluster {cluster} did not settle to exactly one leader, last states: {last_states}")

    def _wait_for_instance_group(self, admin_client, group_name, timeout=30):
        deadline = time.time() + timeout
        last_resp = None
        while time.time() < deadline:
            last_resp = admin_client.get_instance_group({
                "trace_id": f"trace_wait_group_{group_name}_{int(time.time())}",
                "name": group_name,
            }, check_response=False)
            if last_resp.get("header", {}).get("status", {}).get("code") == "OK":
                return
            time.sleep(0.5)
        self.fail(f"instance group {group_name} was not loaded, last response: {last_resp}")

    def _assert_cluster_info(self, cluster, leader_id, admin_clients):
        leader_client = admin_clients[leader_id]
        cluster_resp = leader_client.get_manager_cluster_info({
            "trace_id": f"trace_cluster_info_{cluster}",
        })
        self.assertEqual(cluster_resp["leader_node_id"], cluster_resp["self_node_id"])
        leader_ep = cluster_resp["leader_endpoint"]
        node_idx = self.CLUSTER_WORKERS[cluster].index(leader_id)
        self.assertEqual(leader_ep["node_id"], f"{cluster}_node_{node_idx}")
        self.assertEqual(leader_ep["host"], "127.0.0.1")
        self.assertEqual(leader_ep["custom_info"], f"{cluster}-worker-{node_idx}")

        follower_id = next(worker_id for worker_id in self.CLUSTER_WORKERS[cluster] if worker_id != leader_id)
        follower_resp = admin_clients[follower_id].get_manager_cluster_info({
            "trace_id": f"trace_follower_cluster_info_{cluster}",
        })
        self.assertEqual(follower_resp["leader_node_id"], cluster_resp["leader_node_id"])
        self.assertEqual(follower_resp["leader_endpoint"]["admin_http_port"], leader_ep["admin_http_port"])

    def _assert_cluster_leader(self, cluster, expected_leader_id, admin_clients):
        for worker_id in self.CLUSTER_WORKERS[cluster]:
            resp = admin_clients[worker_id].get_manager_cluster_info({
                "trace_id": f"trace_assert_leader_{cluster}_{worker_id}",
            })
            self.assertEqual(resp["leader_node_id"], self._node_id(cluster, expected_leader_id))
            self.assertEqual(resp["leader_endpoint"]["admin_http_port"],
                             self.worker_manager.get_worker(expected_leader_id).env.admin_http_port)

    def _assert_redis_key_isolation(self, written_instances):
        all_keys = set(self._redis.keys("*"))
        self.assertNotIn("kvcm_lock:_TAIR_KVCM_LEADER_KEY", all_keys)

        for cluster in self.CLUSTERS:
            registry_prefix = f"kvcache_registry:{cluster}:"
            self.assertIn(registry_prefix + "storage", all_keys)
            self.assertIn(registry_prefix + "instance_group", all_keys)
            self.assertIn(registry_prefix + "instance", all_keys)
            self.assertIn(f"kvcm_{cluster}_lock:_TAIR_KVCM_LEADER_KEY", all_keys)
            self.assertEqual(2, len(self._redis.keys(f"kvcm_{cluster}_kv:_TAIR_KVCM_NODE_INFO_*")))

            group_fields = self._redis.hgetall(registry_prefix + "instance_group")
            self.assertEqual({
                self._group_name(cluster, 0),
                self._group_name(cluster, 1),
            }, set(group_fields.keys()))

            instance_fields = self._redis.hgetall(registry_prefix + "instance")
            expected_instances = {
                self._instance_id(cluster, 0),
                self._instance_id(cluster, 1),
            }
            self.assertTrue(expected_instances.issubset(set(instance_fields.keys())))

        for instance_id in written_instances:
            cache_keys = self._redis.keys(f"kvcache:instance_{instance_id}:cache_*")
            self.assertGreaterEqual(len(cache_keys), 2)

        a_groups = self._redis.hgetall(f"kvcache_registry:{self.CLUSTER_A}:instance_group")
        b_groups = self._redis.hgetall(f"kvcache_registry:{self.CLUSTER_B}:instance_group")
        self.assertTrue(all(name.startswith(self.CLUSTER_A) for name in a_groups.keys()))
        self.assertTrue(all(name.startswith(self.CLUSTER_B) for name in b_groups.keys()))

    def _admin_client(self, worker_id):
        worker = self.worker_manager.get_worker(worker_id)
        return KVCMHttpClient(f"http://localhost:{worker.env.admin_http_port}")

    def _meta_client(self, worker_id):
        worker = self.worker_manager.get_worker(worker_id)
        return KVCMHttpClient(f"http://localhost:{worker.env.http_port}")

    def _cluster_redis_uri(self, cluster):
        return f"{self._redis.uri}?cluster_name={cluster}"

    def _node_id(self, cluster, worker_id):
        node_idx = self.CLUSTER_WORKERS[cluster].index(worker_id)
        return f"{cluster}_node_{node_idx}"

    def _other_worker(self, cluster, worker_id):
        return next(candidate for candidate in self.CLUSTER_WORKERS[cluster] if candidate != worker_id)

    @staticmethod
    def _shell_arg(value):
        return "'" + str(value).replace("'", "'\\''") + "'"

    @staticmethod
    def _storage_name(cluster):
        return f"{cluster}_nfs"

    @staticmethod
    def _group_name(cluster, group_idx):
        return f"{cluster}_group_{group_idx}"

    @staticmethod
    def _instance_id(cluster, group_idx):
        return f"{cluster}_group_{group_idx}_instance"


if __name__ == "__main__":
    unittest.main()
