# -*- coding: utf-8 -*-


import abc
import logging
import time
import unittest

from typing import Dict

from integration_test.admin_service.http_interface_test import \
    AdminServiceHttpClient
from integration_test.meta_service.http_interface_test import \
    MetaServiceHttpClient
from integration_test.testlib.test_base import TestBase


class ReclaimingTest(abc.ABC, TestBase, unittest.TestCase):
    """HTTP version of the AdminService tests"""

    def setUp(self):
        self.init_default()
        self._admin_client, self._client = self._get_manager_client()
        self._trace_id = "reclaiming_itest_trace_id"
        self._storage_name = "test_storage_01"
        self._instance_group_name = "test_group_01"
        self._instance_id = "test_instance_01"
        self._model_name = "test_model"
        self._resp_dict = dict()

    def tearDown(self):
        self._admin_client.close()
        self._client.close()
        self.cleanup()

    def test_reclaiming_00(self):
        """Test basic reclaiming functionality."""
        # add storage
        add_storage_req = {
            "trace_id": self._trace_id,
            "storage": self._make_dummy_storage(),
        }
        self._admin_client.add_storage(add_storage_req)

        # add ins group
        # start with the trigger would not happen
        ig = self._make_dummy_instance_group()
        create_ig_req = {
            "trace_id": self._trace_id,
            "instance_group": ig,
        }
        self._admin_client.create_instance_group(create_ig_req)

        # register instance
        reg_ins_data_req = self._make_dummy_ins_req()
        self._client.register_instance(reg_ins_data_req)

        # write 16 blocks
        for i in range(16):
            self._write(i)

        # start write another 1 block
        # since no reclaimer would be triggered, the writing should fail
        # because of max key count is reached for the indexer
        self._start_write_expect_fail(16)

        # make the trigger happen
        curr_ver = ig["version"]
        ig["version"] = curr_ver + 1
        # location spec info size = 1024
        ig[
            "cache_config"
        ][
            "reclaim_strategy"
        ][
            "trigger_strategy"
        ][
            "used_percentage"
        ] = 0.1
        update_ig_req = {
            "trace_id": self._trace_id + "_update_ig",
            "instance_group": ig,
            "current_version": curr_ver,
        }
        self._admin_client.update_instance_group(update_ig_req)

        # 2 sec is enough to make sure the reclaiming happen
        time.sleep(2)

        # now the writing should success
        self._write(16)

    def test_reclaiming_01(self):
        """Test start-writing -> reclaiming -> finish-writing."""
        # add storage
        add_storage_req = {
            "trace_id": self._trace_id,
            "storage": self._make_dummy_storage(),
        }
        self._admin_client.add_storage(add_storage_req)

        # add ins group
        # start with the trigger would not happen
        ig = self._make_dummy_instance_group()
        create_ig_req = {
            "trace_id": self._trace_id,
            "instance_group": ig,
        }
        self._admin_client.create_instance_group(create_ig_req)

        # register instance
        reg_ins_data_req = self._make_dummy_ins_req()
        self._client.register_instance(reg_ins_data_req)

        # start write 16 blocks but not finish write them
        for i in range(16):
            # 0~15
            self._start_write(i)

        # start write another 1 block with key=16
        # since no reclaimer would be triggered, the writing should fail
        # because of max key count is reached for the indexer
        self._start_write_expect_fail(16)

        # make the trigger happen
        curr_ver = ig["version"]
        ig["version"] = curr_ver + 1
        # location spec info size = 1024
        ig[
            "cache_config"
        ][
            "reclaim_strategy"
        ][
            "trigger_strategy"
        ][
            "used_percentage"
        ] = 0.1
        update_ig_req = {
            "trace_id": self._trace_id + "_update_ig",
            "instance_group": ig,
            "current_version": curr_ver,
        }
        self._admin_client.update_instance_group(update_ig_req)

        # 2 sec is enough to make sure the reclaiming happen
        time.sleep(2)

        # start write block with key=16 again, which should still fail
        # since all the blocks within 0~15 are not finish writing and
        # should not be reclaimed
        self._start_write_expect_fail(16)

        # now finish write block 0~15
        for i in range(16):
            # no verify because the location could have been reclaimed
            self._finish_write_with_verify(i, verify=False)

        time.sleep(2)
        # at least one block in 0~15 should be reclaimed already
        # which give room to block 16
        # now the writing of key=16 should success
        self._write(16)

    def _get_manager_client(self):
        self._admin_http_port = self.worker_manager.get_worker(
            0).env.admin_http_port
        self._admin_http_url = f"http://localhost:{self._admin_http_port}"
        self._http_port = self.worker_manager.get_worker(0).env.http_port
        self._http_url = f"http://localhost:{self._http_port}"
        logging.info(
            f"admin http url: {self._admin_http_url}, http url: {self._http_url}")
        return (
            AdminServiceHttpClient(self._admin_http_url),
            MetaServiceHttpClient(self._http_url),
        )

    def _write(self, blk_key):
        logging.info(f"write block key: {blk_key}")
        trace_id = f"{self._trace_id}_blk_key_{blk_key}"

        # start write cache
        self._start_write(blk_key)

        # finish write cache
        self._finish_write_with_verify(blk_key)

    def _start_write_expect_fail(self, blk_key):
        logging.info(f"start write expecting failure, block key: {blk_key}")
        trace_id = f"{self._trace_id}_blk_key_{blk_key}"
        start_write_req = {
            "trace_id": trace_id,
            "instance_id": self._instance_id,
            "block_keys": [blk_key, ],
            "token_ids": [blk_key + 100, ],
            "write_timeout_seconds": 30,
        }
        resp = self._client.start_write_cache(start_write_req,
                                              check_response=False)
        self.assertNotEqual(resp['header']['status']['code'], "OK")

    def _start_write(self, blk_key):
        logging.info(f"start write, block key: {blk_key}")
        trace_id = f"{self._trace_id}_blk_key_{blk_key}"
        start_write_req = {
            "trace_id": trace_id,
            "instance_id": self._instance_id,
            "block_keys": [blk_key, ],
            "token_ids": [blk_key + 100, ],
            "write_timeout_seconds": 30,
        }

        resp = self._client.start_write_cache(start_write_req)

        write_session_id = resp["write_session_id"]
        self.assertIsNotNone(write_session_id)
        self.assertNotEqual(write_session_id, "")

        start_write_locations = resp["locations"]
        self.assertIsNotNone(start_write_locations)
        self.assertGreater(len(start_write_locations), 0)

        self._resp_dict[blk_key] = resp
        logging.info(
            f"block key: {blk_key} start write OK with write session id: {write_session_id}")

    def _finish_write_with_verify(self, blk_key, verify=True):
        # finish write cache
        trace_id = f"{self._trace_id}_blk_key_{blk_key}"
        resp = self._resp_dict[blk_key]
        write_session_id = resp["write_session_id"]
        start_write_locations = resp["locations"]

        finish_write_req = {
            "trace_id": trace_id,
            "instance_id": self._instance_id,
            "write_session_id": write_session_id,
            "success_blocks": {
                "bool_masks": {
                    "values": [True],
                }
            }
        }
        self._client.finish_write_cache(finish_write_req)
        if not verify:
            return

        # get cache location to verify it was added correctly
        get_location_req = {
            "trace_id": trace_id,
            "query_type": "QT_PREFIX_MATCH",
            "block_keys": [blk_key, ],
            "instance_id": self._instance_id,
            "block_mask": {
                "offset": 0,
            },
        }
        resp = self._client.get_cache_location(get_location_req)
        get_location_locations = resp["locations"]

        # verify
        self.assertEqual(
            len(start_write_locations),
            len(get_location_locations),
            "number of locations from startWriteCache and getCacheLocation should match",
        )
        # compare each location
        for i, (start_loc, get_loc) in enumerate(
                zip(start_write_locations, get_location_locations)):
            self.assertEqual(
                start_loc,
                get_loc,
                f"location {i} from startWriteCache and getCacheLocation should match",
            )

    def _make_dummy_storage(self) -> Dict:
        return {
            "global_unique_name": self._storage_name,
            "nfs": {
                "root_path": f"/tmp/{self._storage_name}",
            }
        }

    def _make_dummy_instance_group(self) -> Dict:
        return {
            "name": self._instance_group_name,
            "storage_candidates": [
                self._storage_name,
            ],
            "global_quota_group_name": "quota_group_test",
            "max_instance_count": 8,
            "quota": {
                "capacity": 1024 * 32,
                "quota_config": [
                    # StorageType.ST_TAIRMEMPOOL=3
                    {"storage_type": 3, "capacity": 1024 * 16},
                    # StorageType.ST_NFS=4
                    {"storage_type": 4, "capacity": 1024 * 16},
                ],
            },
            "cache_config": {
                "reclaim_strategy": {
                    "storage_unique_name": self._storage_name,
                    "reclaim_policy": 1,  # POLICY_LRU
                    "trigger_strategy": {
                        # make sure not trigger by percentage
                        "used_percentage": 3.2,
                    },
                    "delay_before_delete_ms": 100,
                },
                "data_storage_strategy": 2,  # CPS_PREFER_3FS
                "meta_indexer_config": {
                    "max_key_count": 16,  # start with 16 max key
                    "mutex_shard_num": 16,
                    "meta_storage_backend_config": {
                        "storage_type": "local",
                        "storage_uri": f"file://{self.get_workdir()}/meta_storage",
                    },
                    "meta_cache_policy_config": {
                        "capacity": 1024 * 1024 * 1024,
                        "type": "LRU",
                    }
                }
            },
            "user_data": "user-defined info",
            "version": 1,
        }

    def _make_dummy_ins_req(self) -> Dict:
        return {
            "trace_id": self._trace_id,
            "instance_group": self._instance_group_name,
            "instance_id": self._instance_id,
            "block_size": 128,
            "model_deployment": self._make_dummy_model_deployment(),
            "location_spec_infos": [
                {
                    "name": "tp0",
                    "size": 1024,
                },
            ],
        }

    def _make_dummy_model_deployment(self):
        return {
            "model_name": self._model_name,
            "dtype": "FP8",
            "use_mla": False,
            "tp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
        }


if __name__ == "__main__":
    unittest.main()
