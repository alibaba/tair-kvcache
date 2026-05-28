# -*- coding: utf-8 -*-

"""Integration test for server recover fault tolerance.

Scenario:
1. Start server with local registry persistence enabled.
2. Add storage, instance group, and register instance_01; write+read a block.
3. Stop server, corrupt the persisted registry file (only instance_01).
4. Restart server — it should start successfully despite the corrupted data.
5. Register a new instance_02; write+read a block on instance_02 (proves service is usable).
6. Stop server, fix instance_01 in the persisted registry file (preserve instance_02).
7. Restart server — all data is correct, both instances should work immediately.
"""

import json
import logging
import os
import unittest
from typing import Dict

from integration_test.admin_service.http_interface_test import AdminServiceHttpClient
from integration_test.meta_service.http_interface_test import MetaServiceHttpClient
from integration_test.testlib.test_base import TestBase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class RecoverFaultToleranceTest(TestBase, unittest.TestCase):

    def setUp(self):
        self.clean_workdir()
        self.prepare_test_resource(1)
        self._registry_path = os.path.join(self.get_workdir(), "registry_data.json")
        self._start_server_with_persistence()
        self._admin_client, self._meta_client = self._connect_clients()
        self._trace_id = "recover_itest"
        self._storage_name = "recover_storage_01"
        self._instance_group_name = "recover_group_01"
        self._resp_dict = {}

    def tearDown(self):
        self._admin_client.close()
        self._meta_client.close()
        self.cleanup()

    def _start_server_with_persistence(self, **extra_kwargs):
        kwargs = {
            "kvcm.registry_storage.uri": f"local://{self._registry_path}",
        }
        kwargs.update(extra_kwargs)
        self.assertTrue(self.worker_manager.start_all(**kwargs))

    def _restart_server(self, **extra_kwargs):
        self.worker_manager.stop_worker(0)
        kwargs = {
            "kvcm.registry_storage.uri": f"local://{self._registry_path}",
        }
        kwargs.update(extra_kwargs)
        self.assertTrue(self.worker_manager.start_worker(0, **kwargs))
        self._admin_client.close()
        self._meta_client.close()
        self._admin_client, self._meta_client = self._connect_clients()

    def _connect_clients(self):
        admin_http_port = self.worker_manager.get_worker(0).env.admin_http_port
        http_port = self.worker_manager.get_worker(0).env.http_port
        admin_url = f"http://localhost:{admin_http_port}"
        meta_url = f"http://localhost:{http_port}"
        logging.info(f"connecting: admin={admin_url}, meta={meta_url}")
        return AdminServiceHttpClient(admin_url), MetaServiceHttpClient(meta_url)

    def test_recover_with_corrupted_registry(self):
        """Server tolerates corrupted registry data and recovers when fixed."""

        # --- Step 1: Normal setup ---
        self._add_storage()
        self._create_instance_group()
        self._register_instance("instance_01")
        self._write_and_read_block("instance_01", block_key=0)
        logging.info("Step 1 OK: instance_01 write+read succeeded")

        # Save the good registry content
        self.assertTrue(os.path.exists(self._registry_path),
                        "registry persistence file should exist")
        with open(self._registry_path, "r") as f:
            good_content = f.read()
        logging.info(f"Good registry content length: {len(good_content)}")

        # --- Step 2: Corrupt the registry file (only instance_01) ---
        corrupted = json.loads(good_content)
        for key in list(corrupted.keys()):
            if "instance_01" in key:
                corrupted[key] = "THIS_IS_CORRUPTED_DATA"
        with open(self._registry_path, "w") as f:
            json.dump(corrupted, f)
        logging.info("Step 2: Registry file corrupted (instance_01 broken)")

        # --- Step 3: Restart server — should start despite corruption ---
        self._restart_server()
        logging.info("Step 3 OK: Server restarted with corrupted registry")

        # --- Step 4: New instance_02 should work fine (registry persists storage/group) ---
        self._register_instance("instance_02")
        self._write_and_read_block("instance_02", block_key=100)
        logging.info("Step 4 OK: instance_02 write+read succeeded on corrupted server")

        # --- Step 5: Fix the registry file (read current file to preserve instance_02) ---
        self.worker_manager.stop_worker(0)
        with open(self._registry_path, "r") as f:
            current_content = json.loads(f.read())
        # Restore only instance_01 entries from the good content
        good_data = json.loads(good_content)
        for key in list(good_data.keys()):
            if "instance_01" in key:
                current_content[key] = good_data[key]
        with open(self._registry_path, "w") as f:
            json.dump(current_content, f)
        logging.info("Step 5: Registry file fixed (instance_01 restored, instance_02 preserved)")

        # --- Step 6: Restart — all data is correct, both instances should work immediately ---
        self._restart_server()
        logging.info("Step 6: Server restarted with fully correct registry")

        # Both instances should be usable right away
        self._write_and_read_block("instance_01", block_key=200)
        logging.info("Step 6 OK: instance_01 write+read succeeded after fix")

        self._write_and_read_block("instance_02", block_key=201)
        logging.info("Step 6 OK: instance_02 write+read also succeeded")

    # --- Helpers ---

    def _add_storage(self):
        req = {
            "trace_id": self._trace_id,
            "storage": {
                "global_unique_name": self._storage_name,
                "nfs": {
                    "root_path": f"{self.get_workdir()}/{self._storage_name}/",
                },
            },
        }
        self._admin_client.add_storage(req, check_response=False)

    def _create_instance_group(self):
        req = {
            "trace_id": self._trace_id,
            "instance_group": {
                "name": self._instance_group_name,
                "storage_candidates": [self._storage_name],
                "global_quota_group_name": "quota_group_test",
                "max_instance_count": 8,
                "quota": {
                    "capacity": 1024 * 32,
                    "quota_config": [
                        {"storage_type": 4, "capacity": 1024 * 16},
                    ],
                },
                "cache_config": {
                    "reclaim_strategy": {
                        "storage_unique_name": self._storage_name,
                        "reclaim_policy": 1,
                        "trigger_strategy": {"used_percentage": 3.2},
                    },
                    "data_storage_strategy": 2,
                    "meta_indexer_config": {
                        "max_key_count": 64,
                        "mutex_shard_num": 16,
                        "meta_storage_backend_config": {
                            "storage_type": "dummy",
                            "storage_uri": f"file://{self.get_workdir()}/meta_{self._instance_group_name}",
                        },
                        "meta_cache_policy_config": {
                            "capacity": 1024 * 1024 * 1024,
                            "type": "LRU",
                        },
                        "persist_metadata_interval_time_ms": 0,
                    },
                },
                "user_data": "",
                "version": 1,
            },
        }
        self._admin_client.create_instance_group(req, check_response=False)

    def _register_instance(self, instance_id):
        req = {
            "trace_id": f"{self._trace_id}_{instance_id}",
            "instance_group": self._instance_group_name,
            "instance_id": instance_id,
            "block_size": 128,
            "model_deployment": {
                "model_name": "test_model",
                "dtype": "FP8",
                "use_mla": False,
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": 1,
            },
            "location_spec_infos": [{"name": "tp0", "size": 1024}],
        }
        self._meta_client.register_instance(req)

    def _write_and_read_block(self, instance_id, block_key):
        trace_id = f"{self._trace_id}_{instance_id}_blk_{block_key}"

        # start write
        start_req = {
            "trace_id": trace_id,
            "instance_id": instance_id,
            "block_keys": [block_key],
            "token_ids": [block_key + 1000],
            "write_timeout_seconds": 30,
        }
        resp = self._meta_client.start_write_cache(start_req)
        write_session_id = resp["write_session_id"]
        self.assertIsNotNone(write_session_id)
        self.assertNotEqual(write_session_id, "")

        # finish write
        finish_req = {
            "trace_id": trace_id,
            "instance_id": instance_id,
            "write_session_id": write_session_id,
            "success_blocks": {"bool_masks": {"values": [True]}},
        }
        self._meta_client.finish_write_cache(finish_req)

        # read (get cache location)
        get_req = {
            "trace_id": trace_id,
            "query_type": "QT_PREFIX_MATCH",
            "block_keys": [block_key],
            "instance_id": instance_id,
            "block_mask": {"offset": 0},
        }
        resp = self._meta_client.get_cache_location(get_req)
        self.assertGreater(len(resp["locations"]), 0,
                           f"block {block_key} should be readable on {instance_id}")


if __name__ == "__main__":
    unittest.main()
