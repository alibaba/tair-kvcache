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

    def test_persist_recover_00(self):
        """Test e2e persist/recover: cache locations and metadata
        survive a normal server restart.

        The meta indexer is configured with the dummy backend with
        filesystem persistence enabled so that

        1. cache locations are always flushed to disk, and
        2. metadata like key_count and storage usage accounting data are
           flushed to disk *before* every metadata READ.

        After a controlled server restart the instance group is
        re-registered (which reinitialise the MetaIndexer from the
        persisted file), and the test verifies:

        1. All block meta written before the restart are still
           addressable.
        2. key_count was recovered (not reset to zero), so the capacity
           limit is enforced correctly.
        """
        add_storage_req = {
            "trace_id": self._trace_id,
            "storage": self._make_dummy_storage(),
        }
        self._admin_client.add_storage(add_storage_req)

        # add instance group
        # start with the reclaiming trigger would not happen
        ig = self._make_dummy_instance_group()
        create_ig_req = {
            "trace_id": self._trace_id,
            "instance_group": ig,
        }
        self._admin_client.create_instance_group(create_ig_req)

        # register instance
        reg_ins_data_req = self._make_dummy_ins_req()
        self._client.register_instance(reg_ins_data_req)

        # write 8 blocks (half of max_key_count=16)
        write_count = 8
        for i in range(write_count):
            self._write(i)

        # --- restart ---
        self.worker_manager.stop_worker(0)
        self.assertTrue(self.worker_manager.start_worker(0))

        # reconnect clients: update_ports() assigns fresh ports on every
        # start
        self._admin_client.close()
        self._client.close()
        self._admin_client, self._client = self._get_manager_client()

        # the registry is in-memory only, so re-add the storage and
        # instance group after restart
        # crucially the same storage_uri is used so that
        # MetaIndexer.Init -> RecoverMetaData will reload key_count and
        # storage_usage_data from the persisted file
        self._admin_client.add_storage(add_storage_req)
        create_ig_req["trace_id"] = self._trace_id + "_restart"
        self._admin_client.create_instance_group(create_ig_req)
        self._client.register_instance(reg_ins_data_req)

        # 1. verify that all blocks written before the restart are still
        #    addressable (cache locations was persisted and recovered)
        for i in range(write_count):
            get_location_req = {
                "trace_id": f"{self._trace_id}_verify_{i}",
                "query_type": "QT_PREFIX_MATCH",
                "block_keys": [i],
                "instance_id": self._instance_id,
                "block_mask": {"offset": 0},
            }
            resp = self._client.get_cache_location(get_location_req)
            self.assertGreater(
                len(resp["locations"]),
                0,
                f"block {i} should be accessible after restart",
            )

        # 2. verify key_count was recovered (not reset to zero)
        #    write the remaining 8 blocks (keys 8-15) to reach
        #    max_key_count=16
        for i in range(write_count, 16):
            self._write(i)
        # key_count is now:
        # 8 (recovered) + 8 (just written) = 16 = max_key_count
        # so the next write must be rejected
        self._start_write_expect_fail(16)

    def test_persist_recover_01(self):
        """Test e2e persist/recover: storage usage data survives a
        normal server restart.

        After writing all 16 blocks (filling key_count to
        max_key_count=16), each block contributes 1024 bytes to
        StorageUsageData for the NFS storage type.  StorageUsageData is
        persisted to the dummy backend together with key_count.

        After a controlled server restart the instance group is
        re-registered, which reloads StorageUsageData from the persisted
        file.  The test verifies recovery by observing the reclaimer's
        byte-usage trigger:

        * If StorageUsageData IS recovered (grp_used_byte_sz_ > 0):
          the group byte-usage ratio is 0.5 which exceeds the 0.1
          threshold → the reclaimer fires → some blocks are freed → a
          new write succeeds.

        * If StorageUsageData is NOT recovered (grp_used_byte_sz_ == 0):
          cache_reclaimer.cc line 480 returns early with water-level
          = false even though key_count was recovered; the reclaimer
          does NOT trigger → the write still fails.
        """
        add_storage_req = {
            "trace_id": self._trace_id,
            "storage": self._make_dummy_storage(),
        }
        self._admin_client.add_storage(add_storage_req)

        # add instance group
        # start with the reclaiming trigger would not happen
        ig = self._make_dummy_instance_group()
        create_ig_req = {
            "trace_id": self._trace_id,
            "instance_group": ig,
        }
        self._admin_client.create_instance_group(create_ig_req)

        # register instance
        reg_ins_data_req = self._make_dummy_ins_req()
        self._client.register_instance(reg_ins_data_req)

        # write all 16 blocks to fill key_count to max_key_count=16
        # each block also adds 1024 bytes to StorageUsageData (NFS type)
        for i in range(16):
            self._write(i)

        # key_count is now at max; the next write must be rejected
        self._start_write_expect_fail(16)

        # --- restart ---
        self.worker_manager.stop_worker(0)
        self.assertTrue(self.worker_manager.start_worker(0))

        # reconnect clients after restart
        self._admin_client.close()
        self._client.close()
        self._admin_client, self._client = self._get_manager_client()

        # re-register storage and instance group using the same storage_uri
        # so that MetaIndexer.Init -> RecoverMetaData reloads both key_count
        # and storage_usage_data from the persisted file
        self._admin_client.add_storage(add_storage_req)
        create_ig_req["trace_id"] = self._trace_id + "_restart"
        self._admin_client.create_instance_group(create_ig_req)
        self._client.register_instance(reg_ins_data_req)

        # key_count is recovered to 16 = max, so write must still fail
        self._start_write_expect_fail(16)

        # lower the reclaim trigger so the reclaimer fires ONLY if
        # storage_usage_data was recovered:
        #   not recovered → grp_used_byte_sz_ == 0 → trigger checker
        #                   early-return false → reclaimer does NOT run
        #                   → write fails
        #   recovered     → grp_used_byte_sz_ == 16 * 1024 = 16384 bytes
        #                   group ratio = 16384 / 32768 = 0.5 > 0.1
        #                   → reclaimer fires → blocks freed → write
        #                   succeeds
        curr_ver = ig["version"]
        ig["version"] = curr_ver + 1
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

        # 2 seconds is enough for the background reclaimer to run
        time.sleep(2)

        # write should succeed only if reclaiming happened, which
        # requires storage_usage_data to have survived the restart
        self._write(16)

    def test_persist_recover_02(self):
        """Test metadata persistence-recovery backward compatibility
        handling.

        Scenario
        --------
        1. Setup: Fill test_instance_01 with 16 blocks but do not
           trigger reclaiming.
        2. Restart the server with intentionally crafted version 0
           metadata file, that is, only key_count is included.
        3. Re-register test_instance_01, lower the threshold for
           test_group_01; make the reclaimer fires.
        4. Assert: a new write to test_instance_01 should success, which
           means the reclaiming worked as expected, thus the backward
           compatibility is properly handled.
        """
        # add storage
        add_storage_req = {
            "trace_id": self._trace_id,
            "storage": self._make_dummy_storage(),
        }
        self._admin_client.add_storage(add_storage_req)

        # add instance group; reclaim trigger is intentionally too high
        # to fire at this point
        ig = self._make_dummy_instance_group()
        create_ig_req = {
            "trace_id": self._trace_id,
            "instance_group": ig,
        }
        self._admin_client.create_instance_group(create_ig_req)

        # register test_instance_01
        reg_ins_01_req = self._make_dummy_ins_req()
        self._client.register_instance(reg_ins_01_req)

        # write 16 keys into test_instance_01
        for i in range(16):
            self._write(i)

        # storage quota is full
        self._start_write_expect_fail(16)

        # restart the server
        self.worker_manager.stop_worker(0)
        meta_storage_backend_config = ig[
            "cache_config"
        ][
            "meta_indexer_config"
        ][
            "meta_storage_backend_config"
        ]
        self._make_v0_persist_data(meta_storage_backend_config)
        self.assertTrue(self.worker_manager.start_worker(0))

        # reconnect clients: update_ports() assigns fresh ports on every
        # start
        self._admin_client.close()
        self._client.close()
        self._admin_client, self._client = self._get_manager_client()

        # re-add the storage and instance group after restart
        # re-register test_instance_01 to bring the recovered indexer
        # back online
        self._admin_client.add_storage(add_storage_req)
        create_ig_req["trace_id"] = self._trace_id + "_restart"
        self._admin_client.create_instance_group(create_ig_req)
        self._client.register_instance(reg_ins_01_req)

        # fire the reclaimer for test_group_01
        curr_ver = ig["version"]
        ig["version"] = curr_ver + 1
        ig[
            "cache_config"
        ][
            "reclaim_strategy"
        ][
            "trigger_strategy"
        ][
            "used_percentage"
        ] = 0.1
        self._admin_client.update_instance_group({
            "trace_id": self._trace_id + "_update_ig",
            "instance_group": ig,
            "current_version": curr_ver,
        })

        time.sleep(2)

        # this write can succeed only when v0 meta is properly handled
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
                "root_path": f"{self.get_workdir()}/{self._storage_name}/",
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
                        "storage_type": "dummy",
                        "storage_uri": f"file://{self.get_workdir()}/meta_storage_{self._instance_group_name}",
                    },
                    "meta_cache_policy_config": {
                        "capacity": 1024 * 1024 * 1024,
                        "type": "LRU",
                    },
                    "persist_metadata_interval_time_ms": 0,
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

    def _make_v0_persist_data(self, meta_storage_backend_conf):
        if meta_storage_backend_conf["storage_type"] == "dummy":
            # rewrite the persisted metadata file
            import json
            d = {}

            persist_path = meta_storage_backend_conf["storage_uri"]
            persist_path = persist_path.removeprefix("file://")
            persist_path = "_".join((persist_path, self._instance_id,))

            with open(persist_path, 'r') as f:
                meta_key = "__metadata__"
                key = "__storage_usage_data__"
                d = json.load(f)
                d1 = json.loads(d[meta_key])
                if key in d1:
                    del d1[key]
                d[meta_key] = json.dumps(d1)

            with open(persist_path, 'w') as f:
                json.dump(d, f)

        elif meta_storage_backend_conf["storage_type"] == "local":
            pass
        elif meta_storage_backend_conf["storage_type"] == "redis":
            # TODO
            pass


if __name__ == "__main__":
    unittest.main()
