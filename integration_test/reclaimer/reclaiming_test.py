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
        self._write_expect_accepted_after_reclaim(16)

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
            self._finish_write_with_verify(i, verify_readback=False)

        time.sleep(2)
        # at least one block in 0~15 should be reclaimed already
        # which give room to block 16
        # now the writing of key=16 should success
        self._write_expect_accepted_after_reclaim(16)

    def test_reclaiming_02(self):
        """Test that CLS_WRITING keys left over after a server restart
        should not permanently block reclaimer progress.

        Scenario
        --------
        1. Setup: Fill test_instance_01 with 8 start_write_cache calls
           (keys 0-7) without ever calling finish_write_cache.  The keys
           are now in CLS_WRITING state and their write sessions exist
           only in server memory.
        2. Restart the server immediately.  The meta indexer recovers
           keys 0-7 from the local-file backend (CLS_WRITING status
           preserved), but the in-memory write session table is gone.
        3. Re-register test_instance_01 (loads recovered indexer) and
           verify finish_write_cache with the pre-restart session IDs
           must all fail, because the server no longer knows those
           sessions.
        4. Write 8 new keys (8-15) into test_instance_01.
        5. Lower the threshold for test_group_01; make the reclaimer
           fires.
        6. Assert: a new write to test_instance_01 (key id = 16) must
           succeed once the reclaimer has freed a slot.
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

        # start writing keys 0-7 into test_instance_01 without finishing
        # them; the keys are recorded in CLS_WRITING state in the
        # indexer
        for i in range(8):
            self._start_write(i)

        # restart the server immediately before any finish_write_cache
        # is called; the write session table is lost, but the meta
        # indexer is flushed to disk (local-file backend) so keys 0-7
        # survive the restart with CLS_WRITING status
        self.worker_manager.stop_worker(0)
        self.assertTrue(
            self.worker_manager.start_worker(
                # configure batching_size=8 so the reclaimer always
                # builds a batch contains test_instance_01's all old
                # CLS_WRITING keys due to the LRU access time
                0, **{"kvcm.cache_reclaimer.del_batch_size": 8}
            )
        )

        # reconnect clients: update_ports() assigns fresh ports on every
        # start
        self._admin_client.close()
        self._client.close()
        self._admin_client, self._client = self._get_manager_client()

        # re-add the storage and instance group after restart; the
        # registry is in-memory only and was lost; using the same
        # storage_uri causes MetaIndexer to reload keys 0-7 (CLS_WRITING)
        # from the storage backend
        # re-register test_instance_01 to bring the recovered indexer
        # back online
        self._admin_client.add_storage(add_storage_req)
        create_ig_req["trace_id"] = self._trace_id + "_restart"
        self._admin_client.create_instance_group(create_ig_req)
        self._client.register_instance(reg_ins_01_req)

        # the server lost all write session state on restart, so
        # finish_write_cache with the pre-restart session IDs must fail
        for i in range(8):
            self._finish_write_expect_fail(i)

        # write 8 new keys (8-15) into test_instance_01
        for i in range(8, 16):
            self._write(i)
        # now the quota is full
        self._start_write_expect_fail(16)

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

        # the orphaned CLS_WRITING should not stop the evicting
        # new write should now be accepted
        self._write_expect_accepted_after_reclaim(16)

    def test_no_over_eviction_with_delay(self):
        """In-flight byte credit prevents repeated delayed batches."""
        self._assert_no_over_eviction_with_delay(
            group_capacity=1024 * 20,
            type_capacity=1024 * 30,
            max_key_count=30,
        )

    def test_no_over_eviction_with_delay_key_count(self):
        """Predicted-key credit prevents repeated delayed batches."""
        self._assert_no_over_eviction_with_delay(
            group_capacity=1024 * 1024,
            type_capacity=1024 * 1024,
            max_key_count=20,
        )

    def test_no_over_eviction_same_group_multiple_instances(self):
        """A Group-wide credit stops admission before the next instance."""
        self.worker_manager.stop_worker(0)
        self.assertTrue(
            self.worker_manager.start_worker(
                0, **{"kvcm.cache_reclaimer.del_batch_size": 4}
            )
        )
        self._admin_client.close()
        self._client.close()
        self._admin_client, self._client = self._get_manager_client()

        self._admin_client.add_storage({
            "trace_id": self._trace_id,
            "storage": self._make_dummy_storage(),
        })
        instance_group = self._make_dummy_instance_group()
        instance_group["quota"]["capacity"] = 1024 * 20
        instance_group["quota"]["quota_config"] = [
            {"storage_type": 3, "capacity": 1024 * 30},
            {"storage_type": 4, "capacity": 1024 * 30},
        ]
        instance_group[
            "cache_config"
        ][
            "meta_indexer_config"
        ][
            "max_key_count"
        ] = 16
        instance_group[
            "cache_config"
        ][
            "reclaim_strategy"
        ][
            "delay_before_delete_ms"
        ] = 3000
        self._admin_client.create_instance_group({
            "trace_id": self._trace_id,
            "instance_group": instance_group,
        })

        instance_ids = ["test_instance_01", "test_instance_02"]
        for instance_id in instance_ids:
            self._instance_id = instance_id
            self._client.register_instance(self._make_dummy_ins_req())
            for block_key in range(6):
                self._write(block_key)

        current_version = instance_group["version"]
        instance_group["version"] = current_version + 1
        instance_group[
            "cache_config"
        ][
            "reclaim_strategy"
        ][
            "trigger_strategy"
        ][
            "used_percentage"
        ] = 0.5
        self._admin_client.update_instance_group({
            "trace_id": self._trace_id + "_update_ig",
            "instance_group": instance_group,
            "current_version": current_version,
        })

        self._wait_metric_value(
            "cache_reclaimer.pending_delete_handler_count", 1, timeout_s=5
        )
        self._wait_metric_value(
            "cache_reclaimer.pending_delete_handler_count", 0, timeout_s=10
        )
        surviving_blocks = sum(
            self._count_surviving_blocks(instance_id, range(6))
            for instance_id in instance_ids
        )
        self.assertEqual(
            self._metric_value("cache_reclaimer.delete_submit_count"),
            1,
            "Group credit should prevent admission from the next instance",
        )
        self.assertGreaterEqual(
            surviving_blocks,
            8,
            "Group credit should limit delayed reclaim to one four-key batch",
        )
        self.assertLess(
            surviving_blocks,
            12,
            "the delayed reclaim batch should still complete",
        )

    def test_no_progress_uses_polling_backoff(self):
        """Active writers keep a hot water level from spinning the cron."""
        self._admin_client.add_storage({
            "trace_id": self._trace_id,
            "storage": self._make_dummy_storage(),
        })
        instance_group = self._make_dummy_instance_group()
        instance_group["quota"]["capacity"] = 1024 * 20
        instance_group["quota"]["quota_config"] = [
            {"storage_type": 3, "capacity": 1024 * 30},
            {"storage_type": 4, "capacity": 1024 * 30},
        ]
        instance_group[
            "cache_config"
        ][
            "meta_indexer_config"
        ][
            "max_key_count"
        ] = 32
        self._admin_client.create_instance_group({
            "trace_id": self._trace_id,
            "instance_group": instance_group,
        })
        self._client.register_instance(self._make_dummy_ins_req())

        for block_key in range(12):
            self._start_write(block_key)

        current_version = instance_group["version"]
        instance_group["version"] = current_version + 1
        instance_group[
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
            "instance_group": instance_group,
            "current_version": current_version,
        })

        time.sleep(0.8)
        metrics = self._admin_client.get_metrics({
            "trace_id": self._trace_id + "_metrics",
        })["metrics"]
        backoff_count = max(
            (
                int(metric["metric_value"].get("int_value", 0))
                for metric in metrics
                if metric["metric_name"] ==
                "cache_reclaimer.reclaim_no_progress_backoff_count"
            ),
            default=0,
        )
        self.assertGreater(backoff_count, 0)
        self.assertLess(
            backoff_count,
            20,
            "no-progress rounds should follow the normal polling interval",
        )

    def _assert_no_over_eviction_with_delay(
        self, group_capacity, type_capacity, max_key_count
    ):
        """Write 12 blocks, then trigger reclaim with a five-second delay.

        A four-key batch is sufficient to move either configured water level
        from 60% to 40%. The pending request must therefore credit the water
        level immediately and prevent a second batch during the delay window.
        The Future terminal state must release all temporary accounting.
        """
        self.worker_manager.stop_worker(0)
        self.assertTrue(
            self.worker_manager.start_worker(
                0, **{"kvcm.cache_reclaimer.del_batch_size": 4}
            )
        )
        self._admin_client.close()
        self._client.close()
        self._admin_client, self._client = self._get_manager_client()

        self._admin_client.add_storage({
            "trace_id": self._trace_id,
            "storage": self._make_dummy_storage(),
        })

        instance_group = self._make_dummy_instance_group()
        instance_group["quota"]["capacity"] = group_capacity
        instance_group["quota"]["quota_config"] = [
            {"storage_type": 3, "capacity": type_capacity},
            {"storage_type": 4, "capacity": type_capacity},
        ]
        instance_group[
            "cache_config"
        ][
            "meta_indexer_config"
        ][
            "max_key_count"
        ] = max_key_count
        instance_group[
            "cache_config"
        ][
            "reclaim_strategy"
        ][
            "delay_before_delete_ms"
        ] = 5000
        self._admin_client.create_instance_group({
            "trace_id": self._trace_id,
            "instance_group": instance_group,
        })
        self._client.register_instance(self._make_dummy_ins_req())

        for block_key in range(12):
            self._write(block_key)

        current_version = instance_group["version"]
        instance_group["version"] = current_version + 1
        instance_group[
            "cache_config"
        ][
            "reclaim_strategy"
        ][
            "trigger_strategy"
        ][
            "used_percentage"
        ] = 0.5
        self._admin_client.update_instance_group({
            "trace_id": self._trace_id + "_update_ig",
            "instance_group": instance_group,
            "current_version": current_version,
        })

        expected_group_type_tags = {
            "instance_group": self._instance_group_name,
            "storage_type": "file",
        }
        expected_group_tags = {
            "instance_group": self._instance_group_name,
        }
        inflight_metrics = self._wait_metric_values([
            ("cache_reclaimer.delete_submit_count", {}, 1),
            ("cache_reclaimer.delete_complete_count", {}, 0),
            ("cache_reclaimer.pending_delete_handler_count", {}, 1),
            ("cache_reclaimer.pending_location_count", {}, 4),
            ("cache_reclaimer.pending_delete_bytes", {}, 4 * 1024),
            ("cache_reclaimer.credited_delete_bytes", {}, 4 * 1024),
            ("cache_reclaimer.predicted_deleted_key_count", {}, 4),
            (
                "cache_reclaimer.pending_location_count",
                expected_group_type_tags,
                4,
            ),
            (
                "cache_reclaimer.pending_delete_bytes",
                expected_group_type_tags,
                4 * 1024,
            ),
            (
                "cache_reclaimer.credited_delete_bytes",
                expected_group_type_tags,
                4 * 1024,
            ),
            (
                "cache_reclaimer.predicted_deleted_key_count",
                expected_group_tags,
                4,
            ),
        ])
        logging.info("delayed reclaim in-flight metrics: %s", inflight_metrics)
        self._wait_surviving_block_count(
            self._instance_id,
            range(12),
            expected_count=8,
            timeout_s=2,
        )

        completed_metrics = self._wait_metric_values(
            [
                ("cache_reclaimer.delete_submit_count", {}, 1),
                ("cache_reclaimer.delete_complete_count", {}, 1),
                ("cache_reclaimer.pending_delete_handler_count", {}, 0),
                ("cache_reclaimer.pending_location_count", {}, 0),
                ("cache_reclaimer.pending_delete_bytes", {}, 0),
                ("cache_reclaimer.credited_delete_bytes", {}, 0),
                ("cache_reclaimer.predicted_deleted_key_count", {}, 0),
                (
                    "cache_reclaimer.pending_location_count",
                    expected_group_type_tags,
                    0,
                ),
                (
                    "cache_reclaimer.pending_delete_bytes",
                    expected_group_type_tags,
                    0,
                ),
                (
                    "cache_reclaimer.credited_delete_bytes",
                    expected_group_type_tags,
                    0,
                ),
                (
                    "cache_reclaimer.predicted_deleted_key_count",
                    expected_group_tags,
                    0,
                ),
            ],
            timeout_s=8,
        )
        logging.info("delayed reclaim completed metrics: %s", completed_metrics)

        surviving_blocks = self._count_surviving_blocks(
            self._instance_id, range(12)
        )

        logging.info(
            "delayed reclaim survivors: %d/12 (capacity=%d, max_keys=%d)",
            surviving_blocks,
            group_capacity,
            max_key_count,
        )
        self.assertGreaterEqual(
            surviving_blocks,
            8,
            "in-flight credit should limit delayed reclaim to one batch",
        )
        self.assertLess(
            surviving_blocks,
            12,
            "the delayed reclaim batch should still complete",
        )

    def _wait_metric_value(
        self, metric_name, expected_value, tags=None, timeout_s=2
    ):
        deadline = time.monotonic() + timeout_s
        last_value = None
        while time.monotonic() < deadline:
            last_value = self._metric_value(metric_name, tags)
            if last_value == expected_value:
                return
            time.sleep(0.05)
        self.fail(
            f"metric {metric_name} with tags {tags or {}} did not become "
            f"{expected_value}; last value: {last_value}"
        )

    def _metric_value(self, metric_name, tags=None):
        metrics = self._admin_client.get_metrics({
            "trace_id": self._trace_id + "_metrics",
        })["metrics"]
        return self._metric_value_from_metrics(metrics, metric_name, tags)

    def _metric_value_from_metrics(self, metrics, metric_name, tags=None):
        expected_tags = tags or {}
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
            for value_type in (
                "int_value",
                "float_value",
                "double_value",
            ):
                if value_type in metric_value:
                    values.append(float(metric_value[value_type]))
                    break
        self.assertEqual(
            len(values),
            1,
            f"expected one {metric_name} metric with tags {expected_tags}, "
            f"got {len(values)}",
        )
        return values[0]

    def _wait_metric_values(self, expected_metrics, timeout_s=2):
        deadline = time.monotonic() + timeout_s
        expected_values = {
            f"{metric_name} with tags {tags}": expected_value
            for metric_name, tags, expected_value in expected_metrics
        }
        last_values = {}
        while time.monotonic() < deadline:
            metrics = self._admin_client.get_metrics({
                "trace_id": self._trace_id + "_metrics",
            })["metrics"]
            last_values = {}
            matched = True
            for metric_name, tags, expected_value in expected_metrics:
                metric_description = f"{metric_name} with tags {tags}"
                try:
                    actual_value = self._metric_value_from_metrics(
                        metrics, metric_name, tags
                    )
                except self.failureException as error:
                    last_values[metric_description] = str(error)
                    matched = False
                    continue
                last_values[metric_description] = actual_value
                if actual_value != expected_value:
                    matched = False
            if matched:
                return last_values
            time.sleep(0.05)
        self.fail(
            "metrics did not reach a matching snapshot; "
            f"expected values: {expected_values}; last values: {last_values}"
        )

    def _count_surviving_blocks(self, instance_id, block_keys):
        surviving_blocks = 0
        for block_key in block_keys:
            response = self._client.get_cache_location({
                "trace_id": f"{self._trace_id}_check_{block_key}",
                "query_type": "QT_PREFIX_MATCH",
                "block_keys": [block_key],
                "instance_id": instance_id,
                "block_mask": {"offset": 0},
            }, check_response=False)
            if response.get("header", {}).get("status", {}).get("code") != "OK":
                continue
            if any(response.get("locations", [])):
                surviving_blocks += 1
        return surviving_blocks

    def _wait_surviving_block_count(
        self, instance_id, block_keys, expected_count, timeout_s
    ):
        deadline = time.monotonic() + timeout_s
        last_count = None
        while time.monotonic() < deadline:
            last_count = self._count_surviving_blocks(
                instance_id, block_keys
            )
            if last_count == expected_count:
                return
            time.sleep(0.05)
        self.fail(
            f"surviving block count did not become {expected_count}; "
            f"last count: {last_count}"
        )

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
        self._write_expect_accepted_after_reclaim(16)

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

    def _write(self, blk_key, verify_readback=True):
        logging.info(f"write block key: {blk_key}")
        trace_id = f"{self._trace_id}_blk_key_{blk_key}"

        # start write cache
        self._start_write(blk_key)

        # finish write cache
        self._finish_write_with_verify(blk_key, verify_readback=verify_readback)

    def _write_expect_accepted_after_reclaim(self, blk_key):
        """Assert a write is accepted after reclaim opens capacity.

        The reclaimer remains active in these scenarios, so the new location may
        be evicted before an immediate get_cache_location readback.
        """
        self._write(blk_key, verify_readback=False)

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

    def _finish_write_expect_fail(self, blk_key):
        logging.info(f"finish write expecting failure, block key: {blk_key}")
        trace_id = f"{self._trace_id}_blk_key_{blk_key}"
        resp = self._resp_dict[blk_key]
        finish_write_req = {
            "trace_id": trace_id,
            "instance_id": self._instance_id,
            "write_session_id": resp["write_session_id"],
            "success_blocks": {
                "bool_masks": {
                    "values": [True],
                }
            },
        }
        resp = self._client.finish_write_cache(finish_write_req,
                                               check_response=False)
        self.assertNotEqual(resp["header"]["status"]["code"], "OK")

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

    def _finish_write_with_verify(self, blk_key, verify_readback=True):
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
        if not verify_readback:
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


if __name__ == "__main__":
    unittest.main()
