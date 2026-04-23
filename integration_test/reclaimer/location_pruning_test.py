# -*- coding: utf-8 -*-


import abc
import logging
import os
import os.path
import time
import unittest

from urllib.parse import urlparse

from integration_test.admin_service.http_interface_test import \
    AdminServiceHttpClient
from integration_test.meta_service.http_interface_test import \
    MetaServiceHttpClient
from integration_test.testlib.test_base import TestBase


class LocationPruningTest(abc.ABC, TestBase, unittest.TestCase):
    """Various location pruning policy tests"""

    def setUp(self):
        self.init_default()
        self._admin_client, self._client = self._get_manager_client()
        self._trace_id = "loc_pruning_itest_trace_id"
        self._storage_name = "test_storage_01"
        self._instance_group_name = "test_group_01"
        self._instance_id = "test_instance_01"
        self._model_name = "test_model"
        self._resp_dict = dict()

    def tearDown(self):
        self._admin_client.close()
        self._client.close()
        self.cleanup()

    def test_aggressive_location_prune_contiguous(self):
        """Verify aggressive prune on a contiguous suffix.

        When the underlying data for some cache blocks becomes
        unavailable (MightExist() returns false), the aggressive prune
        policy should clean ALL invalid locations in a single query or
        write pass -- rather than only pruning the first invalid one and
        requiring O(N) round-trips.

        Scenario (prefix = A B C D E, data for B C D E lost):
        1. Write A B C D E, confirm all 5 queryable.
        2. Delete the data files backing B C D E.
        3. Prefix-match query returns only [A].
           (aggressive: B C D E pruned in one pass)
        4. Re-write succeeds for all of B C D E in one pass.
        5. Prefix-match query returns [A B C D E] again.
        """

        self._make_dummy_storage()
        self._make_dummy_instance_group()
        self._make_dummy_instance()

        # ---------- step 1: write blocks A B C D E --------------------
        block_keys = [100, 101, 102, 103, 104]
        token_ids = [200, 201, 202, 203, 204]
        locations = self._write_blocks(block_keys, token_ids)
        self.assertEqual(len(locations), 5,
                         "initial write should allocate 5 locations")

        resp = self._prefix_query(block_keys)
        self.assertEqual(
            len(resp["locations"]), 5,
            "all 5 blocks should be queryable after initial write")

        # ---------- step 2: simulate data loss for B C D E ------------
        LocationPruningTest._delete_cache_locations(locations, range(1, 5))

        # ---------- step 3: prefix-match query ------------------------
        resp = self._prefix_query(block_keys)
        locations = resp["locations"]
        self.assertEqual(len(locations), 1,
                         "only block A should be returned; "
                         f"got {len(locations)} locations")

        # --------- step 4: wait for the async metadata prune ----------
        time.sleep(2)

        # --------- step 5: re-write the full chain --------------------
        # A still valid; B C D E need writing
        resp = self._start_write_blocks(block_keys, token_ids)
        write_session_id = resp["write_session_id"]
        locations = resp["locations"]
        self.assertEqual(len(locations), 4,
                         "re-write should allocate 4 locations for B C D E; "
                         f"got {len(locations)} locations")
        self._verify_block_keys(locations,
                                [block_keys[i] for i in (1, 2, 3, 4,)])
        # simulate the actual location data write
        LocationPruningTest._touch_cache_locations(locations)
        self._finish_write_blocks(write_session_id, 4)

        # --------- step 6: verify full recovery -----------------------
        resp = self._prefix_query(block_keys)
        self.assertEqual(len(resp["locations"]), 5,
                         "all 5 blocks should be queryable after re-write")

    def test_aggressive_location_prune_contiguous_wo_query(self):
        """Verify aggressive prune on a contiguous suffix with
        start_write_cache only.

        When the underlying data for some cache blocks becomes
        unavailable (MightExist() returns false), the aggressive prune
        policy should clean ALL invalid locations in a single write pass
        -- rather than only pruning the first invalid one and requiring
        O(N) round-trips.

        Scenario (prefix = A B C D E, data for B C D E lost):
        1. Write A B C D E, confirm all 5 queryable.
        2. Delete the data files backing B C D E.
        3. Send start_write_cache for B C D E, the server is expected to
           return all these locations to write, with same path URI (but
           different location_id).
        4. The original locations are deleted from the meta indexer,
           together with the file specified by URI; and since the new
           locations share the same URIs, they may need to be re-created.
        5. Prefix-match query returns [A B C D E] again.
        """

        self._make_dummy_storage()
        self._make_dummy_instance_group()
        self._make_dummy_instance()

        # ---------- step 1: write blocks A B C D E --------------------
        block_keys = [100, 101, 102, 103, 104]
        token_ids = [200, 201, 202, 203, 204]
        locations = self._write_blocks(block_keys, token_ids)
        self.assertEqual(len(locations), 5,
                         "initial write should allocate 5 locations")

        resp = self._prefix_query(block_keys)
        self.assertEqual(
            len(resp["locations"]), 5,
            "all 5 blocks should be queryable after initial write")

        # ---------- step 2: simulate data loss for B C D E ------------
        LocationPruningTest._delete_cache_locations(locations, range(1, 5))

        # ---------- step 3: re-write the full chain -------------------
        # A still valid; B C D E need writing
        resp = self._start_write_blocks(block_keys, token_ids)
        write_session_id = resp["write_session_id"]
        locations = resp["locations"]
        self.assertEqual(len(locations), 4,
                         "re-write should allocate 4 locations for B C D E; "
                         f"got {len(locations)} locations")
        self._verify_block_keys(locations,
                                [block_keys[i] for i in (1, 2, 3, 4,)])

        # --------- step 4: wait for the async metadata prune ----------
        time.sleep(2)
        # simulate the actual location data write
        LocationPruningTest._touch_cache_locations(locations)
        self._finish_write_blocks(write_session_id, len(locations))

        # --------- step 5: verify full recovery -----------------------
        resp = self._prefix_query(block_keys)
        self.assertEqual(len(resp["locations"]), 5,
                         "all 5 blocks should be queryable after re-write")

    def test_aggressive_location_prune_non_contiguous(self):
        """Verify aggressive prune with non-contiguous data loss.

        When stale blocks are interleaved with valid ones (e.g., B and D
        lost while A, C, E intact), the aggressive prune policy should
        clean ALL invalid locations in a single query or write pass --
        rather than only pruning the first invalid one and requiring
        O(N) round-trips, and the write allocate locations only for the
        stale blocks.

        Scenario (prefix = A B C D E, data for B and D lost):
        1. Write A B C D E, confirm all 5 queryable.
        2. Delete the data files backing B and D only.
        3. Prefix-match query returns only [A] (prefix truncated
           at the first stale block).
        4. Re-write allocates exactly 2 locations (for B and D).
        5. Prefix-match query returns [A B C D E] again.
        """

        self._make_dummy_storage()
        self._make_dummy_instance_group()
        self._make_dummy_instance()

        # ---------- step 1: write blocks A B C D E --------------------
        block_keys = [200, 201, 202, 203, 204]
        token_ids = [300, 301, 302, 303, 304]
        locations = self._write_blocks(block_keys, token_ids)
        self.assertEqual(len(locations), 5,
                         "initial write should allocate 5 locations")

        resp = self._prefix_query(block_keys)
        self.assertEqual(
            len(resp["locations"]), 5,
            "all 5 blocks should be queryable after initial write")

        # ---------- step 2: simulate data loss for B and D ------------
        LocationPruningTest._delete_cache_locations(locations, [1, 3])

        # ---------- step 3: prefix-match query ------------------------
        resp = self._prefix_query(block_keys)
        locations = resp["locations"]
        self.assertEqual(len(locations), 1,
                         "only block A should be returned; "
                         f"got {len(locations)} locations")

        # --------- step 4: wait for the async metadata prune ----------
        time.sleep(2)

        # --------- step 5: re-write the full chain --------------------
        # A, C, E still valid; B and D need writing
        resp = self._start_write_blocks(block_keys, token_ids)
        write_session_id = resp["write_session_id"]
        locations = resp["locations"]
        self.assertEqual(len(locations), 2,
                         "re-write should allocate 2 locations for B and D; "
                         f"got {len(locations)} locations")
        self._verify_block_keys(locations, [block_keys[i] for i in (1, 3,)])
        # simulate the actual location data write
        LocationPruningTest._touch_cache_locations(locations)
        self._finish_write_blocks(write_session_id, 2)

        # --------- step 6: verify full recovery -----------------------
        resp = self._prefix_query(block_keys)
        self.assertEqual(len(resp["locations"]), 5,
                         "all 5 blocks should be queryable after re-write")

    def test_aggressive_location_prune_non_contiguous_wo_query(self):
        """Verify aggressive prune with non-contiguous data loss with
        start_write_cache only.

        When stale blocks are interleaved with valid ones (e.g., B and D
        lost while A, C, E intact), the aggressive prune policy should
        clean ALL invalid locations in a single write pass -- rather
        than only pruning the first invalid one and requiring O(N)
        round-trips, and allocate locations only for the stale blocks.

        Scenario (prefix = A B C D E, data for B and D lost):
        1. Write A B C D E, confirm all 5 queryable.
        2. Delete the data files backing B and D only.
        3. Send start_write_cache for B D, the server is expected to
           return all these locations to write, with same path URI (but
           different location_id).
        4. The original locations are deleted from the meta indexer,
           together with the file specified by URI; and since the new
           locations share the same URIs, they may need to be re-created.
        5. Prefix-match query returns [A B C D E] again.
        """

        self._make_dummy_storage()
        self._make_dummy_instance_group()
        self._make_dummy_instance()

        # ---------- step 1: write blocks A B C D E --------------------
        block_keys = [200, 201, 202, 203, 204]
        token_ids = [300, 301, 302, 303, 304]
        locations = self._write_blocks(block_keys, token_ids)
        self.assertEqual(len(locations), 5,
                         "initial write should allocate 5 locations")

        resp = self._prefix_query(block_keys)
        self.assertEqual(
            len(resp["locations"]), 5,
            "all 5 blocks should be queryable after initial write")

        # ---------- step 2: simulate data loss for B and D ------------
        LocationPruningTest._delete_cache_locations(locations, [1, 3])

        # --------- step 3: re-write the full chain --------------------
        # A, C, E still valid; B and D need writing
        resp = self._start_write_blocks(block_keys, token_ids)
        write_session_id = resp["write_session_id"]
        locations = resp["locations"]
        self.assertEqual(len(locations), 2,
                         "re-write should allocate 2 locations for B and D; "
                         f"got {len(locations)} locations")
        self._verify_block_keys(locations, [block_keys[i] for i in (1, 3,)])

        # --------- step 4: wait for the async metadata prune ----------
        time.sleep(2)
        # simulate the actual location data write
        LocationPruningTest._touch_cache_locations(locations)
        self._finish_write_blocks(write_session_id, len(locations))

        # --------- step 5: verify full recovery -----------------------
        resp = self._prefix_query(block_keys)
        self.assertEqual(len(resp["locations"]), 5,
                         "all 5 blocks should be queryable after re-write")

    def test_aggressive_location_prune_all_missing(self):
        """Verify aggressive prune when all blocks are missing.

        When the underlying data for all the cache blocks become
        unavailable (MightExist() returns false), the aggressive prune
        policy should clean ALL invalid locations in a single query or
        write pass -- rather than only pruning the first invalid one and
        requiring O(N) round-trips.

        Scenario (prefix = A B C D E, all data lost):
        1. Write A B C D E, confirm all 5 queryable.
        2. Delete data files for all 5 blocks.
        3. Prefix-match query returns empty result.
        4. Re-write allocates 5 fresh locations.
        5. Prefix-match query returns [A B C D E] again.
        """

        self._make_dummy_storage()
        self._make_dummy_instance_group()
        self._make_dummy_instance()

        # --------- step 1: write blocks A B C -------------------------
        block_keys = [300, 301, 302, 303, 304]
        token_ids = [400, 401, 402, 403, 404]
        locations = self._write_blocks(block_keys, token_ids)
        self.assertEqual(len(locations), 5,
                         "initial write should allocate 5 locations")

        resp = self._prefix_query(block_keys)
        self.assertEqual(len(resp["locations"]), 5,
                         "all 5 blocks should be queryable after initial write")

        # --------- step 2: simulate data loss for A B C D E -----------
        LocationPruningTest._delete_cache_locations(locations, range(0, 5))

        # --------- step 3: prefix-match query -------------------------
        resp = self._prefix_query(block_keys)
        locations = resp["locations"]
        self.assertEqual(len(locations), 0,
                         "no blocks should be returned when all data is lost; "
                         f"got {len(locations)} locations")

        # --------- step 4: wait for async metadata prune --------------
        time.sleep(2)

        # --------- step 5: re-write all blocks ------------------------
        resp = self._start_write_blocks(block_keys, token_ids)
        write_session_id = resp["write_session_id"]
        locations = resp["locations"]
        self.assertEqual(len(locations), 5,
                         "re-write should allocate 5 locations for A B C D E; "
                         f"got {len(locations)} locations")
        self._verify_block_keys(locations, block_keys)
        # simulate the actual location data write
        LocationPruningTest._touch_cache_locations(locations)
        self._finish_write_blocks(write_session_id, 5)

        # --------- step 6: verify full recovery -----------------------
        resp = self._prefix_query(block_keys)
        self.assertEqual(len(resp["locations"]), 5,
                         "all 5 blocks should be queryable after re-write")

    def test_aggressive_location_prune_all_missing_wo_query(self):
        """Verify aggressive prune when all blocks are missing with
        start_write_cache only.

        When the underlying data for all the cache blocks become
        unavailable (MightExist() returns false), the aggressive prune
        policy should clean ALL invalid locations in a single write pass
        -- rather than only pruning the first invalid one and requiring
        O(N) round-trips.

        Scenario (prefix = A B C D E, all data lost):
        1. Write A B C D E, confirm all 5 queryable.
        2. Delete data files for all 5 blocks.
        3. Send start_write_cache for A B C D E, the server is expected
           to return all these locations to write, with same path URI
           (but different location_id).
        4. The original locations are deleted from the meta indexer,
           together with the file specified by URI; and since the new
           locations share the same URIs, they may need to be re-created.
        5. Prefix-match query returns [A B C D E] again.
        """

        self._make_dummy_storage()
        self._make_dummy_instance_group()
        self._make_dummy_instance()

        # --------- step 1: write blocks A B C -------------------------
        block_keys = [300, 301, 302, 303, 304]
        token_ids = [400, 401, 402, 403, 404]
        locations = self._write_blocks(block_keys, token_ids)
        self.assertEqual(len(locations), 5,
                         "initial write should allocate 5 locations")

        resp = self._prefix_query(block_keys)
        self.assertEqual(len(resp["locations"]), 5,
                         "all 5 blocks should be queryable after initial write")

        # --------- step 2: simulate data loss for A B C D E -----------
        LocationPruningTest._delete_cache_locations(locations, range(0, 5))

        # --------- step 3: re-write all blocks ------------------------
        resp = self._start_write_blocks(block_keys, token_ids)
        write_session_id = resp["write_session_id"]
        locations = resp["locations"]
        self.assertEqual(len(locations), 5,
                         "re-write should allocate 5 locations for A B C D E; "
                         f"got {len(locations)} locations")
        self._verify_block_keys(locations, block_keys)

        # --------- step 4: wait for async metadata prune --------------
        time.sleep(2)
        # simulate the actual location data write
        LocationPruningTest._touch_cache_locations(locations)
        self._finish_write_blocks(write_session_id, len(locations))

        # --------- step 5: verify full recovery -----------------------
        resp = self._prefix_query(block_keys)
        self.assertEqual(len(resp["locations"]), 5,
                         "all 5 blocks should be queryable after re-write")

    def _get_manager_client(self):
        worker = self.worker_manager.get_worker(0)
        self._admin_http_port = worker.env.admin_http_port
        self._admin_http_url = f"http://localhost:{self._admin_http_port}"
        self._http_port = worker.env.http_port
        self._http_url = f"http://localhost:{self._http_port}"
        logging.info(f"admin http url: {self._admin_http_url}, "
                     f"http url: {self._http_url}")
        return (
            AdminServiceHttpClient(self._admin_http_url),
            MetaServiceHttpClient(self._http_url),
        )

    def _write_blocks(self, block_keys, token_ids):
        resp = self._start_write_blocks(block_keys, token_ids)
        write_session_id = resp["write_session_id"]
        locations = resp["locations"]

        LocationPruningTest._touch_cache_locations(locations)

        self._finish_write_blocks(write_session_id, len(locations))
        return locations

    def _start_write_blocks(self, blk_keys, token_ids):
        return self._client.start_write_cache({
            "trace_id": self._trace_id,
            "instance_id": self._instance_id,
            "block_keys": blk_keys,
            "token_ids": token_ids,
            "write_timeout_seconds": 30,
        })

    def _finish_write_blocks(self, write_session_id, loc_sz):
        return self._client.finish_write_cache({
            "trace_id": self._trace_id,
            "instance_id": self._instance_id,
            "write_session_id": write_session_id,
            "success_blocks": {
                "bool_masks": {"values": [True] * loc_sz},
            },
        })

    def _prefix_query(self, block_keys):
        return self._client.get_cache_location({
            "trace_id": self._trace_id,
            "query_type": "QT_PREFIX_MATCH",
            "block_keys": block_keys,
            "instance_id": self._instance_id,
            "block_mask": {"offset": 0},
        })

    @staticmethod
    def _touch_cache_locations(locations):
        """simulate the cache data write"""
        for loc in locations:
            for spec in loc.get("location_specs", []):
                file_path = urlparse(spec["uri"]).path
                try:
                    # the data file can be deleted by a slow pruning
                    # after being touched here, which is acceptable
                    # (this is indistinguishable from being reclaimed,
                    # if the reclaimer were being enabled)
                    os.utime(file_path)
                except FileNotFoundError as e:
                    # if here, there are 2 possibilities:
                    # 1. initial write; create it
                    # 2. the data file might have been deleted by
                    #    the location pruning process; create it again
                    logging.info(f"FileNotFoundError: {e}, path: {file_path}")
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'x') as _:
                        pass

    @staticmethod
    def _delete_cache_locations(locations, indices):
        """simulate the cache data lost event"""
        for i in indices:
            loc = locations[i]
            for spec in loc.get("location_specs", []):
                file_path = urlparse(spec["uri"]).path
                os.remove(file_path)

    def _verify_block_keys(self, locations, block_keys):
        self.assertEqual(len(locations), len(block_keys))
        for (loc, key) in zip(locations, block_keys):
            for spec in loc.get("location_specs", []):
                file_path = urlparse(spec["uri"]).path
                self.assertEqual(int(os.path.basename(file_path), base=16), key)

    def _make_dummy_storage(self):
        dummy_root_path = f"{self.get_workdir()}/{self._storage_name}/data/"
        add_storage_req = {
            "trace_id": self._trace_id + "-add_storage",
            "storage": {
                "global_unique_name": self._storage_name,
                "dummy": {
                    "root_path": dummy_root_path,
                    "key_count_per_file": 1,
                }
            },
        }
        self._admin_client.add_storage(add_storage_req)

    def _make_dummy_instance_group(self):
        create_ig_req = {
            "trace_id": self._trace_id + "-add_instance_group",
            "instance_group": {
                "name": self._instance_group_name,
                "storage_candidates": [
                    self._storage_name,
                ],
                "global_quota_group_name": "quota_group_test",
                "max_instance_count": 8,
                "quota": {
                    "capacity": 1024 * 32,
                    "quota_config": [
                        # StorageType.ST_DUMMY=6
                        {"storage_type": 6, "capacity": 1024 * 32},
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
            },
        }
        self._admin_client.create_instance_group(create_ig_req)

    def _make_dummy_instance(self):
        reg_ins_req = {
            "trace_id": self._trace_id + "-add_instance",
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
        self._client.register_instance(reg_ins_req)

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
