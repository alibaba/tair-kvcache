# -*- coding: utf-8 -*-

import json
import logging
import os
import time
import unittest
from urllib.parse import urlparse

from integration_test.admin_service.http_interface_test import (
    AdminServiceHttpClient,
)
from integration_test.meta_service.http_interface_test import (
    MetaServiceHttpClient,
)
from integration_test.testlib.test_base import TestBase


class CacheGarbageCollectorTest(TestBase, unittest.TestCase):
    """End-to-end coverage for background Location collection."""

    _GRACE_MS = 3_600_000

    def setUp(self):
        self.init_default()
        self._admin_client, self._client = self._get_manager_clients()
        self._trace_id = "cache_gc_e2e"
        self._storage_name = "cache_gc_nfs"
        self._group_name = "cache_gc_group"
        self._block_key = 101

    def tearDown(self):
        self._admin_client.close()
        self._client.close()
        self.cleanup()

    def test_orphan_writing_is_collected_after_restart(self):
        """Only the expired orphan is deleted after write sessions are lost."""
        self._create_topology()

        instance_ids = ("cache_gc_instance_a", "cache_gc_instance_b")
        for instance_id in instance_ids:
            self._register_instance(instance_id)
            response = self._start_write(instance_id, self._block_key)
            self.assertTrue(response.get("locations"))

        # Restart loses both in-memory write sessions. Backdate only instance A
        # in the authoritative dummy file so instance B remains inside grace.
        self.worker_manager.stop_worker(0)
        self._backdate_persisted_writing(instance_ids[0], self._block_key)
        self.assertTrue(
            self.worker_manager.start_worker(
                0,
                **{
                    "kvcm.cache_gc.enabled": "true",
                    "kvcm.cache_gc.scan_interval_ms": 25,
                    "kvcm.cache_gc.round_pause_ms": 100,
                    "kvcm.cache_gc.scan_batch_size": 8,
                    "kvcm.cache_gc.orphan_writing_grace_period_ms": self._GRACE_MS,
                    "kvcm.cache_gc.max_inflight_delete_requests": 2,
                },
            )
        )
        self._admin_client.close()
        self._client.close()
        self._admin_client, self._client = self._get_manager_clients()

        self._create_topology()
        for instance_id in instance_ids:
            self._register_instance(instance_id)

        self._wait_metric_at_least(
            "cache_gc.delete_result_count",
            1,
            tags={"status": "0"},
            timeout_s=10,
        )

        # The expired location in A was removed, so the same block can receive
        # a fresh location. B has the same block key but a young WRITING
        # location and therefore remains blocked.
        expired_response = self._start_write(instance_ids[0], self._block_key)
        self.assertTrue(
            expired_response.get("locations"),
            "expired orphan WRITING should be removed and become writable",
        )
        young_response = self._start_write(instance_ids[1], self._block_key)
        self.assertFalse(
            young_response.get("locations"),
            "young WRITING in another instance must not be collected",
        )
        self.assertEqual(
            1,
            self._metric_value("cache_gc.delete_target_count"),
            "only the expired instance should enter the delete request",
        )

        # Exercise the real leader-demotion wiring while GC is enabled. The
        # worker must remain healthy as a standby and completed rounds must stop
        # advancing after GC has been joined and metadata cleanup has run.
        self._admin_client.update_leader_elector_config(
            {
                "trace_id": f"{self._trace_id}_demote_config",
                "campaign_delay_time_ms": 100_000,
            }
        )
        self._admin_client.leader_demote(
            {"trace_id": f"{self._trace_id}_demote"}
        )
        self._wait_until_standby()
        rounds_after_demote = self._metric_value("cache_gc.scan_round_count")
        time.sleep(0.3)
        self.assertEqual(
            rounds_after_demote,
            self._metric_value("cache_gc.scan_round_count"),
            "GC rounds must stop after leader demotion",
        )

    def test_cached_metadata_gc_uses_persistent_backend(self):
        """Dual-backend GC finds and removes an orphan through persistent metadata."""
        self._create_topology(meta_storage_type="cached")
        instance_id = "cache_gc_cached_instance"
        self._register_instance(instance_id)
        response = self._start_write(instance_id, self._block_key)
        self.assertTrue(response.get("locations"))

        self.worker_manager.stop_worker(0)
        self._backdate_persisted_writing(instance_id, self._block_key)
        self.assertTrue(
            self.worker_manager.start_worker(
                0,
                **{
                    "kvcm.cache_gc.enabled": "true",
                    "kvcm.cache_gc.scan_interval_ms": 25,
                    "kvcm.cache_gc.round_pause_ms": 100,
                    "kvcm.cache_gc.scan_batch_size": 8,
                    "kvcm.cache_gc.orphan_writing_grace_period_ms": self._GRACE_MS,
                    "kvcm.cache_gc.max_inflight_delete_requests": 2,
                },
            )
        )
        self._admin_client.close()
        self._client.close()
        self._admin_client, self._client = self._get_manager_clients()

        self._create_topology(meta_storage_type="cached")
        self._register_instance(instance_id)
        self._wait_metric_at_least(
            "cache_gc.delete_result_count",
            1,
            tags={"status": "0"},
            timeout_s=10,
        )

        rewritten = self._start_write(instance_id, self._block_key)
        self.assertTrue(
            rewritten.get("locations"),
            "persistent orphan should be collected in cached metadata mode",
        )

    def test_missing_serving_data_is_collected_without_query(self):
        """A missing ordinary SERVING Location is found by the background probe."""
        self.worker_manager.stop_worker(0)
        self.assertTrue(
            self.worker_manager.start_worker(
                0,
                **{
                    "kvcm.cache_gc.enabled": "true",
                    "kvcm.cache_gc.scan_interval_ms": 25,
                    "kvcm.cache_gc.round_pause_ms": 100,
                    "kvcm.cache_gc.scan_batch_size": 8,
                    "kvcm.cache_gc.orphan_writing_grace_period_ms": self._GRACE_MS,
                    "kvcm.cache_gc.max_inflight_delete_requests": 2,
                },
            )
        )
        self._admin_client.close()
        self._client.close()
        self._admin_client, self._client = self._get_manager_clients()

        self._create_topology(data_storage_type="dummy")
        instance_id = "cache_gc_serving_instance"
        self._register_instance(instance_id)
        response = self._start_write(instance_id, self._block_key)
        locations = response.get("locations", [])
        self.assertEqual(1, len(locations))
        self._touch_location_files(locations)
        self._finish_write(
            instance_id,
            response["write_session_id"],
            len(locations),
        )
        self._remove_location_files(locations)

        # Do not issue GetCacheLocation: cleanup must be driven by GC rather
        # than the existing query-path MightExist pruning.
        self._wait_metric_at_least(
            "cache_gc.candidate_count",
            1,
            tags={"reason": "storage_missing"},
            timeout_s=10,
        )
        self._wait_metric_at_least(
            "cache_gc.delete_result_count",
            1,
            tags={"status": "0"},
            timeout_s=10,
        )
        self.assertEqual(1, self._metric_value("cache_gc.delete_target_count"))

        rewritten = self._start_write(instance_id, self._block_key)
        self.assertTrue(
            rewritten.get("locations"),
            "missing SERVING metadata should be removed and become writable",
        )

    def _create_topology(
        self, meta_storage_type="dummy", data_storage_type="nfs"
    ):
        metadata_uri = (
            f"file://{self.get_workdir()}/cache_gc_metadata"
        )
        if meta_storage_type == "cached":
            metadata_uri += "?persistent_type=dummy&cache_type=local"
        storage_spec = {
            "root_path": os.path.join(
                self.get_workdir(), self._storage_name
            )
            + "/",
        }
        storage_type_id = 4
        if data_storage_type == "dummy":
            storage_spec["key_count_per_file"] = 1
            storage_type_id = 6
        self._admin_client.add_storage(
            {
                "trace_id": self._trace_id,
                "storage": {
                    "global_unique_name": self._storage_name,
                    data_storage_type: storage_spec,
                },
            }
        )
        self._admin_client.create_instance_group(
            {
                "trace_id": self._trace_id,
                "instance_group": {
                    "name": self._group_name,
                    "storage_candidates": [self._storage_name],
                    "global_quota_group_name": "cache_gc_quota",
                    "max_instance_count": 8,
                    "quota": {
                        "capacity": 1024 * 1024,
                        "quota_config": [
                            {
                                "storage_type": storage_type_id,
                                "capacity": 1024 * 1024,
                            }
                        ],
                    },
                    "cache_config": {
                        "reclaim_strategy": {
                            "storage_unique_name": self._storage_name,
                            "reclaim_policy": 1,
                            "trigger_strategy": {"used_percentage": 2.0},
                            "delay_before_delete_ms": 0,
                        },
                        "data_storage_strategy": 2,
                        "meta_indexer_config": {
                            "max_key_count": 128,
                            "mutex_shard_num": 16,
                            "meta_storage_backend_config": {
                                "storage_type": meta_storage_type,
                                "storage_uri": metadata_uri,
                            },
                            "meta_cache_policy_config": {
                                "capacity": 1024 * 1024,
                                "type": "LRU",
                            },
                            "persist_metadata_interval_time_ms": 0,
                        },
                    },
                    "version": 1,
                },
            }
        )

    def _register_instance(self, instance_id):
        self._client.register_instance(
            {
                "trace_id": f"{self._trace_id}_{instance_id}",
                "instance_group": self._group_name,
                "instance_id": instance_id,
                "block_size": 128,
                "model_deployment": {
                    "model_name": "cache_gc_model",
                    "dtype": "FP8",
                    "use_mla": False,
                    "tp_size": 1,
                    "dp_size": 1,
                    "pp_size": 1,
                },
                "location_spec_infos": [{"name": "tp0", "size": 1024}],
            }
        )

    def _start_write(self, instance_id, block_key):
        return self._client.start_write_cache(
            {
                "trace_id": f"{self._trace_id}_{instance_id}_{block_key}",
                "instance_id": instance_id,
                "block_keys": [block_key],
                "token_ids": [block_key + 100],
                "write_timeout_seconds": 1800,
            }
        )

    def _finish_write(self, instance_id, write_session_id, location_count):
        return self._client.finish_write_cache(
            {
                "trace_id": f"{self._trace_id}_{instance_id}_finish",
                "instance_id": instance_id,
                "write_session_id": write_session_id,
                "success_blocks": {
                    "bool_masks": {"values": [True] * location_count},
                },
            }
        )

    @staticmethod
    def _touch_location_files(locations):
        for location in locations:
            for spec in location.get("location_specs", []):
                path = urlparse(spec["uri"]).path
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a", encoding="utf-8"):
                    pass

    @staticmethod
    def _remove_location_files(locations):
        for location in locations:
            for spec in location.get("location_specs", []):
                path = urlparse(spec["uri"]).path
                if os.path.exists(path):
                    os.remove(path)

    def _backdate_persisted_writing(self, instance_id, block_key):
        path = os.path.join(
            self.get_workdir(), f"cache_gc_metadata_{instance_id}"
        )
        self.assertTrue(os.path.exists(path), f"metadata file missing: {path}")
        with open(path, "r", encoding="utf-8") as source:
            persisted = json.load(source)

        fields = json.loads(persisted[str(block_key)])
        location_fields = [
            name for name in fields if name.startswith("L#")
        ]
        self.assertEqual(1, len(location_fields))
        location = json.loads(fields[location_fields[0]])
        self.assertEqual(2, location["status"])
        location["create_time"] = int(time.time() * 1_000_000) - (
            self._GRACE_MS + 60_000
        ) * 1000
        fields[location_fields[0]] = json.dumps(
            location, separators=(",", ":")
        )
        persisted[str(block_key)] = json.dumps(
            fields, separators=(",", ":")
        )
        with open(path, "w", encoding="utf-8") as destination:
            json.dump(persisted, destination, separators=(",", ":"))

    def _get_manager_clients(self):
        worker = self.worker_manager.get_worker(0)
        return (
            AdminServiceHttpClient(
                f"http://localhost:{worker.env.admin_http_port}"
            ),
            MetaServiceHttpClient(f"http://localhost:{worker.env.http_port}"),
        )

    def _wait_metric_at_least(
        self, metric_name, expected, tags=None, timeout_s=5
    ):
        deadline = time.monotonic() + timeout_s
        last_value = None
        while time.monotonic() < deadline:
            last_value = self._metric_value(metric_name, tags)
            if last_value is not None and last_value >= expected:
                return
            time.sleep(0.05)
        self.fail(
            f"{metric_name} with tags {tags or {}} did not reach "
            f"{expected}; last value: {last_value}"
        )

    def _wait_until_standby(self, timeout_s=5):
        deadline = time.monotonic() + timeout_s
        last_response = None
        while time.monotonic() < deadline:
            last_response = self._admin_client.check_health(
                {"trace_id": f"{self._trace_id}_standby"},
                check_response=False,
            )
            if not last_response.get("is_leader", True):
                return
            time.sleep(0.05)
        self.fail(f"worker did not finish leader demotion: {last_response}")

    def _metric_value(self, metric_name, tags=None):
        expected_tags = tags or {}
        metrics = self._admin_client.get_metrics(
            {"trace_id": f"{self._trace_id}_metrics"}
        )["metrics"]
        values = []
        for metric in metrics:
            if metric.get("metric_name") != metric_name:
                continue
            actual_tags = {
                tag["tag_key"]: tag["tag_value"]
                for tag in metric.get("metric_tags", [])
            }
            if actual_tags != expected_tags:
                continue
            metric_value = metric.get("metric_value", {})
            for value_type in ("int_value", "float_value", "double_value"):
                if value_type in metric_value:
                    values.append(float(metric_value[value_type]))
                    break
        if not values:
            return None
        self.assertEqual(
            1,
            len(values),
            f"expected one {metric_name} with tags {expected_tags}",
        )
        return values[0]


if __name__ == "__main__":
    unittest.main()
