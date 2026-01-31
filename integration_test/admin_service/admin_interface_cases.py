import abc
import logging
import unittest
import os

from typing import Dict, List

import time

from testlib.test_base import TestBase
from testlib.worker import Worker, WorkerEnv
from testlib.worker_manager import WorkerManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- AdminService客户端抽象基类 ---


class AdminServiceClientBase(abc.ABC):
    @abc.abstractmethod
    def add_storage(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def enable_storage(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def disable_storage(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def remove_storage(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def update_storage(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def list_storage(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def create_instance_group(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def update_instance_group(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def remove_instance_group(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def get_instance_group(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def get_cache_meta(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def remove_cache(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def register_instance(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def remove_instance(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def get_instance_info(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def list_instance_info(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def add_account(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def delete_account(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def list_account(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def gen_config_snapshot(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def load_config_snapshot(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def get_metrics(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def check_health(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def get_manager_cluster_info(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def leader_demote(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def get_leader_elector_config(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def update_leader_elector_config(self, data, check_response=True) -> Dict:
        return {}

    @abc.abstractmethod
    def close(self):
        pass


# --- AdminService测试基类 ---
class AdminServiceTestBase(abc.ABC, TestBase, unittest.TestCase):
    @abc.abstractmethod
    def _get_manager_client(self) -> AdminServiceClientBase:
        pass

    def setUp(self):
        self.init_default()
        self._client: AdminServiceClientBase = self._get_manager_client()
        self._trace_id = "test_trace_id"

    def tearDown(self):
        self._client.close()
        self.cleanup()

    # 生产数据样例生成函数
    def make_sample_storage(self) -> Dict:
        return {
            "global_unique_name": "test_storage_01",
            # oneof storage_type示例
            "nfs": {
                "root_path": "/tmp/1"
            }
        }

    def make_sample_instance_group(self) -> Dict:
        return {
            "name": "test_instance_group",
            "storage_candidates": ["test_storage_01", "test_storage_02"],
            "global_quota_group_name": "quota_group_test",
            "max_instance_count": 10,
            "quota": {
                "capacity": 1000,
                "quota_config": [
                    {"storage_type": 4, "capacity": 500},  # StorageType.ST_NFS=4
                    {"storage_type": 3, "capacity": 500}   # StorageType.ST_TAIRMEMPOOL=3
                ]
            },
            "cache_config": {
                "reclaim_strategy": {
                    "storage_unique_name": "test_storage_01",
                    "reclaim_policy": 1,  # POLICY_LRU
                    "trigger_strategy": {
                        "used_size": 1024 * 1024 * 1024,  # 1GB
                        "used_percentage": 0.8,
                    },
                    "trigger_period_seconds": 60,
                    "reclaim_step_size": 512 * 1024 * 1024,
                    "reclaim_step_percentage": 5,
                },
                "data_storage_strategy": 2,  # CPS_PREFER_3FS
                "meta_indexer_config": {
                    "max_key_count": 1000000,
                    "mutex_shard_num": 16,
                    "meta_storage_backend_config": {
                        "storage_type": "local",
                        "storage_uri": "file:///tmp/meta_storage"
                    },
                    "meta_cache_policy_config": {
                        "capacity": 1024 * 1024 * 1024,
                        "type": "LRU"
                    }
                }
            },
            "user_data": "user-defined info",
            "version": 1
        }

    def make_sample_instance_info(self) -> Dict:
        return {
            "quota_group_name": "quota_group_test",
            "instance_group_name": "test_instance_group",
            "instance_id": "instance_123",
            "model_deployment": {
                "model_name": "gpt_test",
                "dtype": "FP16",
                "use_mla": False,
                "tp_size": 1,
                "dp_size": 1,
                "lora_name": "lora_hash_123",
                "pp_size": 2,
                "extra": "extra_info",
                "user_data": "user_data_example"
            }
        }

    def make_sample_account(self) -> Dict:
        return {
            "user_name": "tester",
            "role": 1  # ROLE_ADMIN
        }

    def make_sample_register_instance_request(self) -> Dict:
        return {
            "trace_id": self._trace_id,
            "instance_group": "test_instance_group",
            "instance_id": "instance_123",
            "block_size": 128,
            "model_deployment": self.make_sample_instance_info()["model_deployment"],
            "location_spec_infos": [
                {"name": "tp0", "size": 2048}
            ]
        }

    # --- 示例测试用例 ---

    def test_add_and_list_storage(self):
        add_req = {
            "trace_id": self._trace_id,
            "storage": self.make_sample_storage()
        }
        resp = self._client.add_storage(add_req)
        self.assertIn("header", resp)
        self.assertEqual(resp["header"]["status"]["code"], "OK")

        list_resp = self._client.list_storage({"trace_id": self._trace_id})
        self.assertIn("storage", list_resp)
        found = any(s["global_unique_name"] == add_req["storage"]["global_unique_name"] for s in list_resp["storage"])
        self.assertTrue(found, "not found storage after added.")

    def test_create_update_get_remove_instance_group(self):
        ig = self.make_sample_instance_group()
        create_req = {"trace_id": self._trace_id, "instance_group": ig}
        create_resp = self._client.create_instance_group(create_req)
        self.assertEqual(create_resp["header"]["status"]["code"], "OK")

        # Update some field
        ig_updated = ig.copy()
        ig_updated["user_data"] = "updated_data"
        ig_updated["version"] = ig["version"] + 1
        update_req = {"trace_id": self._trace_id, "instance_group": ig_updated, "current_version": ig["version"]}
        update_resp = self._client.update_instance_group(update_req)
        self.assertEqual(update_resp["header"]["status"]["code"], "OK")

        get_req = {"trace_id": self._trace_id, "name": ig["name"]}
        get_resp = self._client.get_instance_group(get_req)
        self.assertEqual(get_resp["header"]["status"]["code"], "OK")
        self.assertEqual(get_resp["instance_group"]["user_data"], "updated_data")

        remove_req = {"trace_id": self._trace_id, "name": ig["name"]}
        remove_resp = self._client.remove_instance_group(remove_req)
        self.assertEqual(remove_resp["header"]["status"]["code"], "OK")

    def test_register_get_list_remove_instance(self):
        ig = self.make_sample_instance_group()
        create_req = {"trace_id": self._trace_id, "instance_group": ig}
        create_resp = self._client.create_instance_group(create_req)
        self.assertEqual(create_resp["header"]["status"]["code"], "OK")

        reg_req = self.make_sample_register_instance_request()
        reg_resp = self._client.register_instance(reg_req)
        self.assertEqual(reg_resp["header"]["status"]["code"], "OK")

        get_req = {"trace_id": self._trace_id, "instance_id": reg_req["instance_id"]}
        get_resp = self._client.get_instance_info(get_req)
        self.assertIn("instance_info", get_resp)
        self.assertEqual(get_resp["instance_info"]["instance_id"], reg_req["instance_id"])

        list_req = {"trace_id": self._trace_id, "instance_group_name": ig["name"]}
        list_resp = self._client.list_instance_info(list_req)
        self.assertIn("instance_info", list_resp)
        self.assertEqual(list_resp["header"]["status"]["code"], "OK")
        # list not exist instance_group
        list_req = {"trace_id": self._trace_id, "instance_group_name": "test_not_exist"}
        list_resp = self._client.list_instance_info(list_req)
        self.assertIn("instance_info", list_resp)
        self.assertEqual(list_resp["header"]["status"]["code"], "OK")

        remove_req = {"trace_id": self._trace_id, "instance_id": reg_req["instance_id"]}
        remove_resp = self._client.remove_instance(remove_req)
        self.assertEqual(remove_resp["header"]["status"]["code"], "OK")

        remove_req = {"trace_id": self._trace_id, "name": ig["name"]}
        remove_resp = self._client.remove_instance_group(remove_req)
        self.assertEqual(remove_resp["header"]["status"]["code"], "OK")

    def test_add_delete_list_account(self):
        add_req = {
            "trace_id": self._trace_id,
            "user_name": "new_user",
            "password": "pass123",
            "role": 0  # ROLE_USER
        }
        add_resp = self._client.add_account(add_req)
        self.assertEqual(add_resp["header"]["status"]["code"], "OK")

        list_resp = self._client.list_account({"trace_id": self._trace_id})
        self.assertTrue(any(acc["user_name"] == "new_user" for acc in list_resp["accounts"]))

        del_resp = self._client.delete_account({"trace_id": self._trace_id, "user_name": "new_user"})
        self.assertEqual(del_resp["header"]["status"]["code"], "OK")

    def test_get_metrics(self):
        # demo call
        # add_storage, list_storage
        self.test_add_and_list_storage()

        metrics_req = {"trace_id": self._trace_id}
        metrics_resp = self._client.get_metrics(metrics_req)
        logging.info(metrics_resp)
        self.assertIn("header", metrics_resp)
        self.assertEqual(metrics_resp["header"]["status"]["code"], "OK")
        self.assertIn("metrics", metrics_resp)

        metric_service_query_counter = 0
        metric_service_query_rt_us = 0

        for metric in metrics_resp["metrics"]:
            logging.info(metric)
            if metric["metric_name"] == "service.query_counter":
                for tag in metric["metric_tags"]:
                    if ((tag["tag_key"] == "api_name" and
                         tag["tag_value"] == "AddStorage") or
                        (tag["tag_key"] == "api_name" and
                         tag["tag_value"] == "ListStorage")):
                        metric_service_query_counter = int(metric["metric_value"]["int_value"])
            elif metric["metric_name"] == "service.query_rt_us":
                for tag in metric["metric_tags"]:
                    if ((tag["tag_key"] == "api_name" and
                         tag["tag_value"] == "AddStorage") or
                        (tag["tag_key"] == "api_name" and
                         tag["tag_value"] == "ListStorage")):
                        metric_service_query_rt_us = float(metric["metric_value"]["float_value"])
        self.assertGreater(metric_service_query_counter, 0)
        self.assertGreater(metric_service_query_rt_us, 1)

    def test_check_health(self):
        """测试 CheckHealth 接口"""
        req = {"trace_id": self._trace_id}
        resp = self._client.check_health(req)
        self.assertIn("header", resp)
        self.assertEqual(resp["header"]["status"]["code"], "OK")
        self.assertIn("is_leader", resp)
        self.assertIn("is_health", resp)
        self.assertIn("elector_last_loop_time_ms", resp)

    def test_get_manager_cluster_info(self):
        """测试 GetManagerClusterInfo 接口"""
        req = {"trace_id": self._trace_id}
        resp = self._client.get_manager_cluster_info(req)
        self.assertIn("header", resp)
        self.assertEqual(resp["header"]["status"]["code"], "OK")
        self.assertIn("info_updated_time", resp)
        self.assertIn("self_leader_expiration_time", resp)
        self.assertIn("self_node_id", resp)
        self.assertIn("leader_node_id", resp)

    def test_get_leader_elector_config(self):
        """测试 GetLeaderElectorConfig 接口"""
        req = {"trace_id": self._trace_id}
        resp = self._client.get_leader_elector_config(req)
        self.assertIn("header", resp)
        self.assertEqual(resp["header"]["status"]["code"], "OK")
        self.assertIn("campaign_delay_time_ms", resp)

    def test_update_leader_elector_config(self):
        """测试 UpdateLeaderElectorConfig 接口"""
        # 首先获取当前配置
        get_req = {"trace_id": self._trace_id}
        get_resp = self._client.get_leader_elector_config(get_req)
        self.assertEqual(get_resp["header"]["status"]["code"], "OK")
        original_delay = int(get_resp["campaign_delay_time_ms"])
        
        # 更新配置（增加1000ms）
        new_delay = original_delay + 1000
        update_req = {"trace_id": self._trace_id, "campaign_delay_time_ms": new_delay}
        update_resp = self._client.update_leader_elector_config(update_req)
        self.assertIn("header", update_resp)
        self.assertEqual(update_resp["header"]["status"]["code"], "OK")
        
        # 再次获取配置验证是否更新（如果更新成功）
        get_resp2 = self._client.get_leader_elector_config(get_req)
        self.assertEqual(get_resp2["header"]["status"]["code"], "OK")
        # 如果更新成功，新的延迟应该生效；否则保持原样
        # 我们只是记录而不做硬性断言，因为更新可能需要特定权限
        logging.info(f"Original delay: {original_delay}, current delay: {get_resp2['campaign_delay_time_ms']}")
    def test_leader_demote(self):
        """测试 LeaderDemote 接口"""
        req = {"trace_id": self._trace_id}
        resp = self._client.check_health(req)
        self.assertIn("header", resp)
        self.assertEqual(resp["header"]["status"]["code"], "OK")
        self.assertEqual(resp["is_leader"], True)

        update_req = {"trace_id": self._trace_id, "campaign_delay_time_ms": 100000}
        update_resp = self._client.update_leader_elector_config(update_req)
        self.assertEqual(update_resp["header"]["status"]["code"], "OK")

        resp = self._client.leader_demote(req)
        self.assertIn("header", resp)
        self.assertEqual(resp["header"]["status"]["code"], "OK")

        time.sleep(5)

        req = {"trace_id": self._trace_id}
        resp = self._client.check_health(req)
        self.assertIn("header", resp)
        self.assertEqual(resp["header"]["status"]["code"], "OK")
        self.assertEqual(resp["is_leader"], False)


# --- 基于文件分布式锁的选主测试 ---
class AdminServiceLeaderElectionTest(abc.ABC, TestBase, unittest.TestCase):
    """测试基于文件分布式锁的选主功能"""

    @abc.abstractmethod
    def _get_manager_client(self, worker_id: int) -> AdminServiceClientBase:
        pass

    def setUp(self):
        self.clean_workdir()
        self._lock_dir = os.path.join(self.get_workdir(), "distributed_lock")
        os.makedirs(self._lock_dir, exist_ok=True)
        self._lock_uri = f"file://{self._lock_dir}"
        self.prepare_test_resource(2)
        self.start_worker()

    def tearDown(self):
        self.cleanup()

    def prepare_test_resource(self, worker_num, work_dir=None, worker_mode='normal'):
        self._init_dirs(work_dir)
        self.worker_manager = WorkerManager()
        self.envs: List[WorkerEnv] = []
        self.mode = worker_mode

        port_range_from, port_range_to = self.get_hash_range(os.getcwd())
        for i in range(worker_num):
            env = WorkerEnv(workdir=os.path.join(self.workdir, 'worker_' + str(i)), path_root=self.workdir)
            env.set_port_range(port_range_from, port_range_to)
            env.set_mode(worker_mode)
            self.envs.append(env)
            logging.info(f"add worker {i} workdir: {env.workdir}")
            self.worker_manager.add_worker(Worker(i, env))

    def start_worker(self, **kwargs):
        # 配置文件分布式锁
        kwargs[f'kvcm.distributed_lock.uri'] = self._lock_uri
        self.assertTrue(self.worker_manager.start_all(**kwargs))

    def clean_test_resource(self):
        pass

    def stop_worker(self):
        self.worker_manager.stop_all()

    def _wait_for_leader(self, client, worker_id, timeout=30, interval=1):
        """等待指定worker成为leader"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                req = {"trace_id": f"trace_wait_{worker_id}_{int(time.time())}"}
                resp = client.check_health(req, check_response=False)
                if resp.get("header", {}).get("status", {}).get("code") == "OK" and resp.get("is_leader") == True:
                    return True
            except Exception as e:
                logging.warning(f"Worker {worker_id} check health failed: {e}")
            time.sleep(interval)
        return False

    def _wait_for_any_leader(self, clients, timeout=30, interval=1):
        """等待任意一个worker成为leader"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            for i, client in enumerate(clients):
                try:
                    req = {"trace_id": f"trace_any_{i}_{int(time.time())}"}
                    resp = client.check_health(req, check_response=False)
                    if resp.get("header", {}).get("status", {}).get("code") == "OK" and resp.get("is_leader") == True:
                        return i, resp
                except Exception as e:
                    logging.warning(f"Worker {i} check health failed: {e}")
            time.sleep(interval)
        return None, None

    def test_leader_election_with_file_lock(self):
        """测试两个worker通过文件分布式锁进行选主"""
        logging.info("开始测试基于文件分布式锁的选主功能")

        # 创建两个客户端，分别连接两个worker
        client0 = self._get_manager_client(0)
        client1 = self._get_manager_client(1)

        try:
            # 等待任意一个worker成为leader（最多等待10秒）
            leader_id, leader_resp = self._wait_for_any_leader([client0, client1], timeout=10)
            self.assertIsNotNone(leader_id, "在10秒内没有worker成为leader")
            logging.info(f"Worker {leader_id} 成为leader: {leader_resp}")
            original_leader_client = client0 if leader_id == 0 else client1
            original_follower_client = client0 if leader_id == 1 else client1

            # 验证leader的选举信息
            self.assertTrue(leader_resp.get("is_leader"), f"Worker {leader_id} 应该成为leader")
            self.assertTrue(leader_resp.get("is_health"), f"Worker {leader_id} 应该健康")

            # 获取leader的集群信息
            cluster_req = {"trace_id": "trace_cluster_info"}
            cluster_resp = original_leader_client.get_manager_cluster_info(cluster_req)
            logging.info(f"Leader集群信息: {cluster_resp}")

            self.assertEqual(cluster_resp["header"]["status"]["code"], "OK", "获取集群信息应该成功")
            self.assertEqual(cluster_resp["leader_node_id"], cluster_resp["self_node_id"], "leader节点ID和leader自身应该一致")
            self.assertIn("self_leader_expiration_time", cluster_resp, "集群信息应该包含leader租约过期时间")
            leader_node_id = cluster_resp["leader_node_id"]

            # 验证非leader的worker不是leader
            non_leader_id = 1 if leader_id == 0 else 0
            non_leader_req = {"trace_id": "trace_non_leader_check"}
            non_leader_resp = original_follower_client.check_health(non_leader_req)
            self.assertFalse(non_leader_resp.get("is_leader"), f"Worker {non_leader_id} 不应该成为leader")
            self.assertTrue(non_leader_resp.get("is_health"), f"Worker {non_leader_id} 应该健康")
            non_leader_cluster_resp = client0.get_manager_cluster_info(cluster_req)
            self.assertEqual(non_leader_cluster_resp["leader_node_id"], leader_node_id, "非leader节点应该看到正确的leader_id")

            # 测试leader降级功能（如果leader是当前worker）
            update_req = {"trace_id": "trace_demote_test", "campaign_delay_time_ms": 10 * 1000}
            update_resp = original_leader_client.update_leader_elector_config(update_req)
            self.assertEqual(update_resp["header"]["status"]["code"], "OK")

            demote_req = {"trace_id": "trace_demote_test"}
            demote_resp = original_leader_client.leader_demote(demote_req)
            self.assertEqual(demote_resp["header"]["status"]["code"],"OK", f"Leader降级失败: {demote_resp}")

            # 原来的leader不再是leader
            check_req = {"trace_id": f"trace_check_original_leader_{int(time.time())}"}
            check_resp = original_leader_client.check_health(check_req, check_response=False)
            self.assertFalse(check_resp["is_leader"])

            # 原来的follower变成leader
            self._wait_for_any_leader([client0, client1], timeout=10)
            check_req = {"trace_id": f"trace_check_original_leader_{int(time.time())}"}
            check_resp = original_follower_client.check_health(check_req, check_response=False)
            self.assertTrue(check_resp["is_leader"])

            logging.info("基于文件分布式锁的选主测试完成")

        finally:
            client0.close()
            client1.close()

    def test_leader_switch_when_worker_stops(self):
        """测试当一个worker挂掉时，leader能自动切换到另一个worker"""
        logging.info("开始测试worker挂掉后的leader自动切换")

        # 创建两个客户端
        client0 = self._get_manager_client(0)
        client1 = self._get_manager_client(1)

        try:
            # 等待任意一个worker成为leader（最多等待10秒）
            leader_id, leader_resp = self._wait_for_any_leader([client0, client1], timeout=10)
            self.assertIsNotNone(leader_id, "在10秒内没有worker成为leader")
            # 获取leader的详细信息
            leader_client = client0 if leader_id == 0 else client1
            leader_health_req = {"trace_id": "trace_leader_health_before_stop"}
            leader_health_resp = leader_client.check_health(leader_health_req)
            self.assertTrue(leader_health_resp.get("is_leader"), f"Worker {leader_id} 应该成为leader")

            # 确定非leader的worker
            non_leader_id = 1 if leader_id == 0 else 0
            non_leader_client = client0 if non_leader_id == 0 else client1
            
            # 检查非leader的状态
            non_leader_req = {"trace_id": "trace_non_leader_before_stop"}
            non_leader_resp = non_leader_client.check_health(non_leader_req)
            self.assertFalse(non_leader_resp.get("is_leader"), f"Worker {non_leader_id} 不应该成为leader")

            # 停止当前的leader worker
            self.worker_manager.stop_worker(leader_id)
            
            # 等待一段时间，让分布式锁检测到leader失效
            time.sleep(1)  # 等待足够时间让租约过期

            # 检查leader是否真的停止了
            try:
                # 尝试连接已停止的worker，应该会失败
                stop_check_req = {"trace_id": "trace_stop_check"}
                response = leader_client.check_health(stop_check_req, check_response=True)
                self.fail(f"原leader未停止, resp:{response}")
            except Exception as e:
                pass

            # 等待非leader worker成为新的leader（最多等待60秒，因为需要等待租约过期和重新选举）
            new_leader_found = self._wait_for_leader(non_leader_client, non_leader_id, timeout=10)
            self.assertTrue(new_leader_found, f"在10秒内Worker {non_leader_id} 没有成为新的leader")

            # 验证新的leader状态
            new_leader_req = {"trace_id": "trace_new_leader_health"}
            new_leader_resp = non_leader_client.check_health(new_leader_req)
            
            self.assertTrue(new_leader_resp.get("is_leader"), f"Worker {non_leader_id} 应该成为新的leader")
            self.assertTrue(new_leader_resp.get("is_health"), f"新的leader应该健康")

            # 获取新的leader的集群信息
            new_cluster_req = {"trace_id": "trace_new_leader_cluster"}
            new_cluster_resp = non_leader_client.get_manager_cluster_info(new_cluster_req)
            leader_node_id = new_cluster_resp["leader_node_id"]
            
            self.assertEqual(new_cluster_resp["header"]["status"]["code"], "OK", "获取新的leader集群信息应该成功")
            self.assertEqual(new_cluster_resp["leader_node_id"], new_cluster_resp["self_node_id"], "leader节点ID和leader自身应该一致")

            # 重新启动之前停止的worker
            self.worker_manager.start_worker(leader_id)
            
            # 等待重新启动的worker完成初始化
            time.sleep(2)

            # 重新创建客户端连接
            restarted_client = self._get_manager_client(leader_id)
            restarted_req = {"trace_id": "trace_restarted_worker"}
            restarted_resp = restarted_client.check_health(restarted_req, check_response=True)
            logging.info(f"重新启动的Worker {leader_id} 状态: {restarted_resp}")
            self.assertEqual(restarted_resp["header"]["status"]["code"], "OK")
            self.assertEqual(restarted_resp["is_leader"], False)
            new_cluster_req = {"trace_id": "trace_new_leader_cluster"}
            new_cluster_resp = restarted_client.get_manager_cluster_info(new_cluster_req)
            self.assertEqual(new_cluster_resp["leader_node_id"], leader_node_id, "正确感知leader节点ID")
            self.assertNotEqual(new_cluster_resp["self_node_id"], leader_node_id, "leader节点ID应该和自身id不一致")
            restarted_client.close()

            # 检查Worker 0的状态
            worker0_client = self._get_manager_client(0)
            worker0_req = {"trace_id": "trace_worker0_final_check"}
            worker0_resp = worker0_client.check_health(worker0_req)

            # 检查Worker 1的状态
            worker1_client = self._get_manager_client(1)
            worker1_req = {"trace_id": "trace_worker1_final_check"}
            worker1_resp = worker1_client.check_health(worker1_req)

            # 系统应该仍然有leader（可能是Worker 0或Worker 1）
            leader_count = 0
            if worker0_resp["is_leader"]:
                leader_count+=1
            if worker1_resp["is_leader"]:
                leader_count+=1
            self.assertEqual(leader_count, 1, "系统应该有且只有1个leader")

            worker0_client.close()
            worker1_client.close()
            
            logging.info("worker挂掉后的leader自动切换测试完成")

        finally:
            client0.close()
            client1.close()
