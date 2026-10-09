"""Sustained multi-Instance metadata traffic through the real HTTP server.

The NFS backend supplies Location URIs; this test does not write KV payloads.
Increase KVCM_GROUP_LRU_TRAFFIC_KEYS / SECONDS for dedicated validation runs.
Results are emitted as JSON, not written into the repository or a PR report.
"""

import bisect
import concurrent.futures
import json
import os
import threading
import time
import unittest

from integration_test.reclaimer import reclaiming_test


class GroupLruTrafficTest(reclaiming_test.ReclaimingTest):
    def setUp(self):
        super().setUp()
        self.seed_size = max(10000, int(os.environ.get("KVCM_GROUP_LRU_TRAFFIC_KEYS", "12000")))
        self.duration = max(2.0, float(os.environ.get("KVCM_GROUP_LRU_TRAFFIC_SECONDS", "4")))

    def _reset_traffic(self, ratio, seed_counts):
        self.worker_manager.stop_worker(0)
        self._admin_client.close()
        self._client.close()
        self.assertTrue(self.worker_manager.start_worker(0, **{
            "kvcm.logger.log_level": 3,
            "kvcm.cache_gc.enabled": "false",
            "kvcm.cache_reclaimer.key_sampling_size_total": 100,
            "kvcm.cache_reclaimer.key_sampling_size_per_task": 100,
            "kvcm.cache_reclaimer.del_batch_size": 100,
            "kvcm.cache_reclaimer.group_lru_min_sampling_ratio": ratio,
            "kvcm.metrics.report_interval_ms": 100,
            "kvcm.metrics.enable_prometheus": "true",
        }))
        self._admin_client, self._client = self._get_manager_client()
        self._admin_client.add_storage({"trace_id": self._trace_id, "storage": self._make_dummy_storage()})
        group = self._make_dummy_instance_group()
        group["quota"] = {"capacity": self.seed_size * 16 * 1024, "quota_config": []}
        indexer = group["cache_config"]["meta_indexer_config"]
        indexer["max_key_count"] = self.seed_size * 16
        indexer["mutex_shard_num"] = 1024
        indexer["meta_storage_backend_config"] = {
            "storage_type": "local",
            "storage_uri": "local://?capacity=1024&num_shard_bits=6&sample_times=64",
        }
        strategy = group["cache_config"]["reclaim_strategy"]
        strategy["instance_reclaim_budget_policy"] = "GROUP_LRU"
        strategy["delay_before_delete_ms"] = 0
        strategy["trigger_strategy"]["used_percentage"] = 3.2
        self._admin_client.create_instance_group({"trace_id": self._trace_id, "instance_group": group})
        self.group = group
        self.known_access = {name: {} for name in seed_counts}
        self.next_key = {name: 0 for name in seed_counts}
        self.http_latencies_ms = []
        self.latency_lock = threading.Lock()
        self.thread_sessions = threading.local()
        self.requests = 0
        self.written = {name: 0 for name in seed_counts}
        self.read_hits = {name: 0 for name in seed_counts}
        self.hot_keys = {name: [] for name in seed_counts}
        for name, count in seed_counts.items():
            self._instance_id = name
            self._client.register_instance(self._make_dummy_ins_req())
            self._append(name, count)
        self.seed_counts = seed_counts
        self.pressure_target_keys = None
        self.inactive_remaining_timeline = []

    def _post(self, endpoint, data):
        begin = time.monotonic()
        session = getattr(self.thread_sessions, "session", self._client.session)
        response = session.post(self._http_url + endpoint, json=data, timeout=15)
        with self.latency_lock:
            self.http_latencies_ms.append((time.monotonic() - begin) * 1000)
            self.requests += 1
        response.raise_for_status()
        result = response.json()
        self.assertEqual("OK", result["header"]["status"]["code"], result)
        return result

    def _append(self, name, count):
        for offset in range(0, count, 256):
            size = min(256, count - offset)
            keys = list(range(self.next_key[name], self.next_key[name] + size))
            result = self._post("/api/startWriteCache", {
                "trace_id": self._trace_id, "instance_id": name,
                "block_keys": keys, "token_ids": [key + 100 for key in keys], "write_timeout_seconds": 30,
            })
            self.assertTrue(result.get("write_session_id"))
            self.assertEqual(size, len(result.get("locations", [])))
            self._post("/api/finishWriteCache", {
                "trace_id": self._trace_id, "instance_id": name,
                "write_session_id": result["write_session_id"],
                "success_blocks": {"bool_masks": {"values": [True] * size}},
            })
            timestamp = time.monotonic_ns()
            self.known_access[name].update((key, timestamp) for key in keys)
            self.next_key[name] += size
            self.written[name] += size

    def _read(self, name, keys, record_access):
        present = set()
        for offset in range(0, len(keys), 256):
            batch = keys[offset:offset + 256]
            result = self._post("/api/getCacheLocation", {
                "trace_id": self._trace_id, "instance_id": name, "query_type": "QT_BATCH_GET",
                "block_keys": batch, "block_mask": {"offset": 0},
            })
            locations = result.get("locations", [])
            self.assertEqual(len(batch), len(locations))
            # Batch Get preserves positional spec placeholders for misses;
            # a nonempty location_specs list alone is not a cache hit.
            hits = [key for key, location in zip(batch, locations)
                    if any(spec.get("uri") for spec in location.get("location_specs", []))]
            present.update(hits)
            if record_access:
                timestamp = time.monotonic_ns()
                for key in hits:
                    self.known_access[name][key] = timestamp
                self.read_hits[name] += len(hits)
        return present

    def _set_target(self, target_keys=None):
        if target_keys is not None:
            self.pressure_target_keys = target_keys
        current_version = self.group["version"]
        self.group["version"] += 1
        self.group["cache_config"]["reclaim_strategy"]["trigger_strategy"]["used_percentage"] = (
            3.2 if target_keys is None else (target_keys + 0.5) * 1024 / self.group["quota"]["capacity"]
        )
        self._admin_client.update_instance_group({
            "trace_id": self._trace_id, "instance_group": self.group, "current_version": current_version,
        })

    def _settle(self):
        # Stop water-level reclaim before verification; reading survivors
        # must not change which keys are chosen by an in-progress workload.
        self._set_target()
        deadline = time.monotonic() + 20
        stable_since = time.monotonic()
        previous = None
        while time.monotonic() < deadline:
            submitted = self._metric_value("cache_reclaimer.group_lru_submitted_block_count")
            pending = self._metric_value("cache_reclaimer.pending_delete_handler_count")
            if pending or submitted != previous:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= 1:
                return
            previous = submitted
            time.sleep(0.1)
        self.fail("reclaimer did not settle after disabling pressure")

    def _snapshot(self):
        return {name: self._read(name, list(keys), False) for name, keys in self.known_access.items()}

    def _instance_key_count(self, name):
        tags = {"instance_group": self.group["name"], "instance_id": name}
        metrics = self._admin_client.get_metrics({"trace_id": self._trace_id + "_usage"})["metrics"]
        if not any(metric.get("metric_name") == "cache_manager_instance.key_count" and
                   {tag["tag_key"]: tag["tag_value"] for tag in metric.get("metric_tags", [])} == tags
                   for metric in metrics):
            return None  # The asynchronous recorder has not published this Instance yet.
        return int(self._metric_value_from_metrics(metrics, "cache_manager_instance.key_count", tags))

    @staticmethod
    def _percentile(values, q):
        if not values:
            return 0
        values = sorted(values)
        return values[min(len(values) - 1, int((len(values) - 1) * q))]

    def _report(self, scenario, ratio, survivors, elapsed, workload_requests, workload_written, workload_latencies):
        deleted_times, retained_times = [], []
        counts = {}
        for name, accesses in self.known_access.items():
            retained = survivors[name]
            counts[name] = {
                "written": len(accesses), "remaining": len(retained), "deleted": len(accesses) - len(retained),
                "hot_missing": len(set(self.hot_keys[name]) - retained), "read_hits": self.read_hits[name],
                "foreground_written_during_pressure": workload_written[name],
            }
            for key, timestamp in accesses.items():
                (retained_times if key in retained else deleted_times).append(timestamp)
        retained_times.sort()
        inversions = sum(bisect.bisect_left(retained_times, timestamp) for timestamp in deleted_times)
        pairs = len(deleted_times) * len(retained_times)
        metrics = {}
        for field in ("sampled_key_count", "candidate_count", "selected_block_count", "submitted_block_count",
                      "delete_request_count", "plan_count", "partial_plan_count", "deadline_count"):
            metrics[field] = self._metric_value("cache_reclaimer.group_lru_" + field)
        age_metrics = {}
        response = self._admin_client.session.get(self._admin_http_url + "/metrics", timeout=15)
        response.raise_for_status()
        prometheus_samples = {line.rsplit(None, 1)[0]: float(line.rsplit(None, 1)[1]) for line in response.text.splitlines()
                              if line and not line.startswith("#")}
        for kind in ("lru", "create"):
            ages = {stat: self._metric_value(f"cache_reclaimer.reclaim_batch_{kind}_age_{stat}_us")
                    for stat in ("min", "max", "avg")}
            age_metrics[kind] = ages
            for stat, value in ages.items():
                exported = f"kvcm_cache_reclaimer_reclaim_batch_{kind}_age_{stat}_us"
                if metrics["submitted_block_count"]:
                    self.assertIn(exported, prometheus_samples)
                    # The exporter currently uses six significant digits.
                    self.assertAlmostEqual(value, prometheus_samples[exported], delta=max(1, abs(value) * 5e-6))
            if metrics["submitted_block_count"]:
                self.assertGreater(ages["min"], 0, age_metrics)
                self.assertLessEqual(ages["min"], ages["avg"], age_metrics)
                self.assertLessEqual(ages["avg"], ages["max"], age_metrics)
        result = {
            "scenario": scenario, "min_sampling_ratio": ratio, "initial_keys": sum(self.seed_counts.values()),
            "pressure_target_keys": self.pressure_target_keys,
            "pressure_seconds": round(elapsed, 3), "pressure_http_requests": workload_requests,
            "pressure_http_qps": round(workload_requests / elapsed, 2),
            "pressure_http_p50_ms": self._percentile(workload_latencies, 0.5),
            "pressure_http_p99_ms": self._percentile(workload_latencies, 0.99),
            "instances": counts, "metrics": metrics, "last_accepted_batch_age_us": age_metrics,
            "prometheus_age_metrics_verified": bool(metrics["submitted_block_count"]),
            "inactive_remaining_timeline": getattr(self, "inactive_remaining_timeline", []),
            "final_snapshot_lru_inversion_fraction": inversions / pairs if pairs else 0,
        }
        print("GROUP_LRU_TRAFFIC_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        self.assertEqual(len(deleted_times), metrics["submitted_block_count"],
                         "Completed deletes must reconcile with the final URI-based key inventory")
        return result

    def _traffic(self, rates, duration=None, pattern="steady", stop_when=None):
        begin = time.monotonic()
        initial_requests = self.requests
        initial_latencies = len(self.http_latencies_ms)
        initial_written = dict(self.written)
        duration = self.duration if duration is None else duration
        stopped = threading.Event()
        def produce(name, rate):
            # One independent, bounded HTTP session per active Instance.
            # Reclaimer runs concurrently with these different-rate writers.
            client = reclaiming_test.MetaServiceHttpClient(self._http_url)
            self.thread_sessions.session = client.session
            cycle = 0
            try:
                while time.monotonic() - begin < duration and not stopped.is_set():
                    if self.hot_keys[name]:
                        self._read(name, self.hot_keys[name], True)
                    burst = 4 if pattern == "bursty" and cycle % 8 == 0 else 1
                    self._append(name, rate * burst)
                    cycle += 1
                    time.sleep(0.02)
            finally:
                client.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(rates)) as executor:
            futures = [executor.submit(produce, name, rate) for name, rate in rates.items()]
            try:
                while stop_when is not None and time.monotonic() - begin < duration:
                    if all(future.done() for future in futures) or stop_when():
                        break
                    time.sleep(0.1)
            finally:
                if stop_when is not None:
                    stopped.set()
            for future in futures:
                future.result()
        return (time.monotonic() - begin, self.requests - initial_requests,
                {name: self.written[name] - initial_written[name] for name in self.written},
                self.http_latencies_ms[initial_latencies:])

    def test_traffic_deployment_cutover_sampling_ratios(self):
        results = {}
        for ratio in (1, 5, 10):
            with self.subTest(ratio=ratio):
                self._reset_traffic(ratio, {"old_large": self.seed_size, "new_small": self.seed_size // 10})
                self._set_target(self.seed_size)
                statistics = self._traffic({"new_small": 32})
                self._settle()
                survivors = self._snapshot()
                result = self._report("deployment_cutover", ratio, survivors, *statistics)
                results[ratio] = result
                self.assertGreater(result["instances"]["old_large"]["deleted"], 0)
                self.assertGreater(len(survivors["old_large"]), 0, "Keep cold data available throughout this comparison")
                if ratio > 1:
                    self.assertEqual(0, result["instances"]["new_small"]["deleted"], result)
        self.assertGreater(results[1]["instances"]["new_small"]["deleted"], 0,
                           "The 1:1 control must reproduce premature newer-key eviction")

    def test_traffic_inactive_large_instance_drains_to_zero(self):
        for initial_new in (0, self.seed_size // 10):
            with self.subTest(initial_new=initial_new):
                # Register the new Instance only after the old one is populated.
                self._reset_traffic(10, {"old_large": self.seed_size, "new_active": initial_new})
                # The Instance usage recorder updates every five seconds,
                # independently of the metrics reporter's 100 ms interval.
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and self._instance_key_count("old_large") != self.seed_size:
                    time.sleep(0.1)
                self.assertEqual(self.seed_size, self._instance_key_count("old_large"))
                self._set_target(self.seed_size)
                begin = time.monotonic()
                self.inactive_remaining_timeline = [{"seconds": 0, "remaining": self.seed_size}]

                def drained():
                    # Observe usage, not GetCacheLocation: old keys must stay untouched.
                    remaining = self._instance_key_count("old_large")
                    self.assertIsNotNone(remaining, "Missing metrics must not be interpreted as zero keys")
                    previous = self.inactive_remaining_timeline[-1]["remaining"]
                    self.assertLessEqual(remaining, previous)
                    if remaining != previous:
                        self.inactive_remaining_timeline.append({
                            "seconds": round(time.monotonic() - begin, 3), "remaining": remaining,
                        })
                    return remaining == 0

                statistics = self._traffic({"new_active": 256}, duration=max(60, self.duration * 4), stop_when=drained)
                self._settle()
                survivors = self._snapshot()
                scenario = "inactive_large_new_empty" if initial_new == 0 else "inactive_large_new_small"
                result = self._report(scenario, 10, survivors, *statistics)
                self.assertEqual(0, len(survivors["old_large"]), result)
                self.assertEqual(0, self.inactive_remaining_timeline[-1]["remaining"], result)
                self.assertEqual(0, result["instances"]["old_large"]["read_hits"])
                self.assertEqual(0, result["instances"]["old_large"]["foreground_written_during_pressure"])
                self.assertGreater(len(survivors["new_active"]), self.seed_size // 2, result)

    def test_traffic_inactive_small_instance_drains_to_zero(self):
        self._reset_traffic(10, {"old_small": self.seed_size // 10, "active_large": self.seed_size})
        self._set_target(self.seed_size)
        statistics = self._traffic({"active_large": 128}, duration=max(6, self.duration))
        self._settle()
        survivors = self._snapshot()
        result = self._report("inactive_small_drains", 10, survivors, *statistics)
        self.assertEqual(0, len(survivors["old_small"]), result)
        self.assertGreater(len(survivors["active_large"]), 0)

    def test_traffic_all_active_different_rates_and_hot_reads(self):
        for pattern in ("steady", "bursty"):
            with self.subTest(pattern=pattern):
                counts = {"fast": self.seed_size // 2, "medium": self.seed_size // 3,
                          "slow": self.seed_size - self.seed_size // 2 - self.seed_size // 3}
                self._reset_traffic(10, counts)
                for name in counts:
                    self.hot_keys[name] = list(range(64))
                    self.assertEqual(set(self.hot_keys[name]), self._read(name, self.hot_keys[name], True))
                self._set_target(self.seed_size * 4 // 5)
                statistics = self._traffic({"fast": 128, "medium": 32, "slow": 8}, pattern=pattern)
                self._settle()
                survivors = self._snapshot()
                result = self._report("all_active_" + pattern, 10, survivors, *statistics)
                self.assertGreater(sum(item["deleted"] for item in result["instances"].values()), 0)
                for name in counts:
                    self.assertGreater(result["instances"][name]["foreground_written_during_pressure"], 0)
                    self.assertEqual(0, result["instances"][name]["hot_missing"], result)
                self.assertLess(result["final_snapshot_lru_inversion_fraction"], 0.05, result)

    def test_traffic_without_pressure_does_not_clear_inactive_instance(self):
        self._reset_traffic(10, {"inactive": self.seed_size // 10, "active": self.seed_size})
        statistics = self._traffic({"active": 32}, duration=2)
        self._settle()
        survivors = self._snapshot()
        result = self._report("no_pressure", 10, survivors, *statistics)
        self.assertEqual(self.seed_size // 10, len(survivors["inactive"]), result)
        self.assertEqual(0, result["metrics"]["submitted_block_count"])


def load_tests(loader, tests, pattern):
    # Reuse helpers without re-running the inherited small smoke cases here.
    return unittest.TestSuite(GroupLruTrafficTest(name) for name in loader.getTestCaseNames(GroupLruTrafficTest)
                              if name.startswith("test_traffic_"))


if __name__ == "__main__":
    unittest.main()
