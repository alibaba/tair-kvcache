    # -*- coding: utf-8 -*-

"""Integration tests for multi-location lifecycle scenarios.

These tests verify that the Meta indexer correctly handles single-key
with multiple coexisting locations, leveraging the LocationSpecGroup
mechanism to produce stable multi-location states:

- A key can have multiple locations when written via different spec groups.
- Query merges specs from all valid locations of the same storage backend.
- Partial data loss (one location's file deleted) still allows query via
  the remaining location.
- Full data loss (all locations' files deleted) breaks prefix match.
- Rewrite after partial/full loss correctly produces new locations.

This is complementary to location_pruning_test.py which focuses on the
pruning policy itself. Here we focus on the per-location granularity
behavior when multiple locations stably coexist for the same key.
"""

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


class MultiLocationTest(abc.ABC, TestBase, unittest.TestCase):
    """Tests for single-key with multiple locations via LocationSpecGroup."""

    # Two spec groups that partition the location_spec_infos
    GROUP_A = "GroupA"  # contains spec "tp0"
    GROUP_B = "GroupB"  # contains spec "tp1"

    def setUp(self):
        self.init_default()
        self._admin_client, self._client = self._get_manager_client()
        self._trace_id = "multi_loc_itest_trace_id"
        self._storage_name = "test_storage_ml"
        self._instance_group_name = "test_group_ml"
        self._instance_id = "test_instance_ml"
        self._model_name = "test_model_ml"

    def tearDown(self):
        self._admin_client.close()
        self._client.close()
        self.cleanup()

    def test_two_groups_produce_two_locations_per_key(self):
        """Writing with GroupA then GroupB produces two coexisting locations.

        Scenario:
        1. Write blocks [A, B, C] with GroupA → each key gets location_1.
        2. Write blocks [A, B, C] with GroupB → each key gets location_2.
        3. Query returns all 3 blocks with merged specs (tp0 + tp1).

        This verifies that LocationSpecGroup correctly allows multiple
        locations to coexist for the same key without triggering prune.
        """
        self._make_dummy_storage()
        self._make_dummy_instance_group()
        self._make_dummy_instance_with_groups()

        block_keys = [500, 501, 502]
        token_ids = [600, 601, 602]

        # Write with GroupA → location_1 per key (contains tp0)
        locs_a = self._write_blocks_with_group(block_keys, token_ids,
                                               self.GROUP_A)
        self.assertEqual(len(locs_a), 3)
        for loc in locs_a:
            spec_names = [s["name"] for s in loc["location_specs"]]
            self.assertIn("tp0", spec_names)
            self.assertNotIn("tp1", spec_names)

        # Write with GroupB → location_2 per key (contains tp1)
        locs_b = self._write_blocks_with_group(block_keys, token_ids,
                                               self.GROUP_B)
        self.assertEqual(len(locs_b), 3)
        for loc in locs_b:
            spec_names = [s["name"] for s in loc["location_specs"]]
            self.assertIn("tp1", spec_names)
            self.assertNotIn("tp0", spec_names)

        # Query: should return all 3 blocks, each with merged specs
        resp = self._prefix_query(block_keys)
        self.assertEqual(len(resp["locations"]), 3,
                         "all 3 blocks should be queryable")
        for loc in resp["locations"]:
            spec_names = [s["name"] for s in loc["location_specs"]]
            self.assertIn("tp0", spec_names,
                          "merged result should contain tp0")
            self.assertIn("tp1", spec_names,
                          "merged result should contain tp1")

    def test_partial_location_loss_still_queryable(self):
        """When one location's data is lost, query still works via the other.

        Scenario:
        1. Write [A, B, C] with GroupA → location_1 (tp0).
        2. Write [A, B, C] with GroupB → location_2 (tp1).
        3. Delete location_1's data files (tp0 gone).
        4. Wait for prune of stale location_1.
        5. Query still returns all 3 blocks (via location_2, tp1 only).
        6. The returned specs should only contain tp1.
        """
        self._make_dummy_storage()
        self._make_dummy_instance_group()
        self._make_dummy_instance_with_groups()

        block_keys = [600, 601, 602]
        token_ids = [700, 701, 702]

        locs_a = self._write_blocks_with_group(block_keys, token_ids,
                                               self.GROUP_A)
        locs_b = self._write_blocks_with_group(block_keys, token_ids,
                                               self.GROUP_B)

        # Delete all GroupA location files
        self._delete_cache_locations(locs_a, list(range(len(locs_a))))

        # Wait for async prune to detect and clean stale locations
        time.sleep(2)

        # Query: trigger prune detection via SelectForMatch
        resp = self._prefix_query(block_keys)
        self.assertEqual(len(resp["locations"]), 3,
                         "blocks should still be queryable via GroupB location")

        # Verify only tp1 specs remain (tp0 was pruned)
        for loc in resp["locations"]:
            spec_names = [s["name"] for s in loc["location_specs"]]
            self.assertIn("tp1", spec_names,
                          "tp1 should survive since GroupB data is intact")

        # Rewrite with GroupA → should allocate new locations
        new_locs_a = self._write_blocks_with_group(block_keys, token_ids,
                                                   self.GROUP_A)
        self.assertEqual(len(new_locs_a), 3,
                         "all blocks need new GroupA locations")

        # Query: full coverage restored
        resp = self._prefix_query(block_keys)
        self.assertEqual(len(resp["locations"]), 3)
        for loc in resp["locations"]:
            spec_names = [s["name"] for s in loc["location_specs"]]
            self.assertIn("tp0", spec_names)
            self.assertIn("tp1", spec_names)

    def test_all_locations_lost_breaks_prefix(self):
        """When all locations' data is lost, prefix match breaks.

        Scenario:
        1. Write [A, B, C] with GroupA and GroupB (two locations each).
        2. Delete ALL data files for block B (both locations).
        3. Wait for prune.
        4. Prefix query returns only [A] (prefix breaks at B).
        """
        self._make_dummy_storage()
        self._make_dummy_instance_group()
        self._make_dummy_instance_with_groups()

        block_keys = [700, 701, 702]
        token_ids = [800, 801, 802]

        locs_a = self._write_blocks_with_group(block_keys, token_ids,
                                               self.GROUP_A)
        locs_b = self._write_blocks_with_group(block_keys, token_ids,
                                               self.GROUP_B)

        # Delete ALL data for block B (index 1) across both locations
        self._delete_cache_locations(locs_a, [1])
        self._delete_cache_locations(locs_b, [1])

        # Wait for prune
        time.sleep(2)

        # Prefix query should break at B
        resp = self._prefix_query(block_keys)
        self.assertEqual(len(resp["locations"]), 1,
                         "prefix should break at B; only A should match")

    # ===== Helper methods =====

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

    def _write_blocks_with_group(self, block_keys, token_ids, group_name):
        """Write blocks with a specific LocationSpecGroup."""
        resp = self._start_write_blocks(block_keys, token_ids,
                                        group_name=group_name)
        write_session_id = resp["write_session_id"]
        locations = resp["locations"]
        self._touch_cache_locations(locations)
        self._finish_write_blocks(write_session_id, len(locations))
        return locations

    def _start_write_blocks(self, block_keys, token_ids, group_name=None):
        req = {
            "trace_id": self._trace_id,
            "instance_id": self._instance_id,
            "block_keys": block_keys,
            "token_ids": token_ids,
            "write_timeout_seconds": 30,
        }
        if group_name:
            req["location_spec_group_names"] = [group_name] * len(block_keys)
        return self._client.start_write_cache(req)

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
        """Simulate the cache data write by creating data files."""
        for loc in locations:
            for spec in loc.get("location_specs", []):
                file_path = urlparse(spec["uri"]).path
                try:
                    os.utime(file_path)
                except FileNotFoundError:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'x') as _:
                        pass

    @staticmethod
    def _delete_cache_locations(locations, indices):
        """Simulate data loss by removing location data files."""
        for i in indices:
            loc = locations[i]
            for spec in loc.get("location_specs", []):
                file_path = urlparse(spec["uri"]).path
                if os.path.exists(file_path):
                    os.remove(file_path)

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
                "global_quota_group_name": "quota_group_ml",
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
                            "used_percentage": 3.2,
                        },
                    },
                    "data_storage_strategy": 2,  # CPS_PREFER_3FS
                    "meta_indexer_config": {
                        "max_key_count": 32,
                        "mutex_shard_num": 16,
                        "meta_storage_backend_config": {
                            "storage_type": "dummy",
                            "storage_uri": (
                                f"file://{self.get_workdir()}"
                                f"/meta_storage_{self._instance_group_name}"
                            ),
                        },
                        "persist_metadata_interval_time_ms": 0,
                    }
                },
                "user_data": "multi-location test",
                "version": 1,
            },
        }
        self._admin_client.create_instance_group(create_ig_req)

    def _make_dummy_instance_with_groups(self):
        """Register instance with two specs and two groups.

        - tp0 belongs to GroupA
        - tp1 belongs to GroupB
        Writing with different groups produces separate locations.
        """
        reg_ins_req = {
            "trace_id": self._trace_id + "-add_instance",
            "instance_group": self._instance_group_name,
            "instance_id": self._instance_id,
            "block_size": 128,
            "model_deployment": self._make_dummy_model_deployment(),
            "location_spec_infos": [
                {"name": "tp0", "size": 1024},
                {"name": "tp1", "size": 1024},
            ],
            "location_spec_groups": [
                {"name": self.GROUP_A, "spec_names": ["tp0"]},
                {"name": self.GROUP_B, "spec_names": ["tp1"]},
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
