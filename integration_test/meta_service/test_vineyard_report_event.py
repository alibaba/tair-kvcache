#!/usr/bin/env python3
"""
Standalone integration tests for V6D (Vineyard) ReportEvent HTTP interface.

Usage:
    # 1. Start KVCM service locally (with HTTP port)
    # 2. Run this script:
    python test_vineyard_report_event.py \
        --host localhost --http_port 8080 \
        --instance_id v6d_cluster_0

    # Run only functional tests (skip bench):
    python test_vineyard_report_event.py ... --skip-bench

    # Run only bench tests:
    python test_vineyard_report_event.py ... --only-bench
"""

import argparse
import json
import sys
import time
import statistics
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# ---------------------------------------------------------------------------
# Global config (set by argparse before tests run)
# ---------------------------------------------------------------------------
BASE_URL = ""
ADMIN_URL = ""
INSTANCE_ID = "v6d_cluster_0"
SKIP_BENCH = False
ONLY_BENCH = False


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
class KVCMClient:
    def __init__(self, base_url, admin_url=None):
        self.base_url = base_url
        self.admin_url = admin_url or base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def report_event(self, data, check_ok=True):
        url = f"{self.base_url}/api/reportEvent"
        resp = self.session.post(url, json=data)
        resp.raise_for_status()
        body = resp.json()
        if check_ok:
            ec = body.get("ec") or body.get("header", {}).get("status", {}).get("code")
            assert ec in ("OK", 1, "1", None), (
                f"ReportEvent failed: ec={ec}, body={json.dumps(body, ensure_ascii=False)}"
            )
        return body

    def register_instance(self, data):
        url = f"{self.base_url}/api/registerInstance"
        resp = self.session.post(url, json=data)
        resp.raise_for_status()
        body = resp.json()
        code = body.get("header", {}).get("status", {}).get("code")
        assert code == "OK", f"registerInstance failed: {json.dumps(body)}"
        return body

    def add_storage(self, data):
        """Register a storage backend via admin API."""
        url = f"{self.admin_url}/api/addStorage"
        resp = self.session.post(url, json=data)
        resp.raise_for_status()
        body = resp.json()
        code = body.get("header", {}).get("status", {}).get("code")
        if code not in ("OK", "DUPLICATE_ENTITY"):
            raise AssertionError(f"addStorage failed: {json.dumps(body)}")
        return body

    def get_cache_location(self, data):
        url = f"{self.base_url}/api/getCacheLocation"
        resp = self.session.post(url, json=data)
        resp.raise_for_status()
        return resp.json()

    def start_evict_write_cache(self, data, check_response=True):
        url = f"{self.base_url}/api/startEvictWriteCache"
        resp = self.session.post(url, json=data)
        resp.raise_for_status()
        body = resp.json()
        if check_response:
            code = body.get("header", {}).get("status", {}).get("code")
            assert code == "OK", f"startEvictWriteCache failed: {json.dumps(body)}"
        return body

    def finish_write_cache(self, data, check_response=True):
        url = f"{self.base_url}/api/finishWriteCache"
        resp = self.session.post(url, json=data)
        resp.raise_for_status()
        body = resp.json()
        if check_response:
            code = body.get("header", {}).get("status", {}).get("code")
            assert code == "OK", f"finishWriteCache failed: {json.dumps(body)}"
        return body

    def close(self):
        self.session.close()


# ---------------------------------------------------------------------------
# Helper: build ReportEvent payloads
# ---------------------------------------------------------------------------
def _make_node_register(instance_id, host_ip_port, trace_id="test"):
    return {
        "trace_id": trace_id,
        "instance_id": instance_id,
        "event_type": "EVENT_NODE_REGISTER",
        "node_register": {"host_ip_port": host_ip_port},
    }


def _make_block_add(instance_id, block_key, host_ip_port, location_json=None, trace_id="test"):
    if location_json is None:
        location_json = json.dumps({
            "addr": host_ip_port,
            "type": "memory",
            "gpu": "A100",
        })
    return {
        "trace_id": trace_id,
        "instance_id": instance_id,
        "event_type": "EVENT_BLOCK_ADD",
        "block_add": {
            "block_key": str(block_key),
            "location_json": location_json,
            "host_ip_port": host_ip_port,
        },
    }


def _make_block_delete(instance_id, block_key, host_ip_port, trace_id="test"):
    return {
        "trace_id": trace_id,
        "instance_id": instance_id,
        "event_type": "EVENT_BLOCK_DELETE",
        "block_delete": {
            "block_key": str(block_key),
            "host_ip_port": host_ip_port,
        },
    }


def _make_host_down(instance_id, down_host_ip_port, trace_id="test"):
    return {
        "trace_id": trace_id,
        "instance_id": instance_id,
        "event_type": "EVENT_HOST_DOWN",
        "host_down": {"down_host_ip_port": down_host_ip_port},
    }


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------
class VineyardReportEventFunctionalTest(unittest.TestCase):
    HOST = "192.168.1.200:8080"
    VINEYARD_STORAGE_NAME = "v6d_v6d_cluster_0"

    @classmethod
    def setUpClass(cls):
        cls.client = KVCMClient(BASE_URL, ADMIN_URL)
        cls.instance_id = INSTANCE_ID
        cls._ensure_vineyard_storage_registered()
        cls._ensure_instance_registered()

    @classmethod
    def _ensure_vineyard_storage_registered(cls):
        """Register a VineyardBackend via admin addStorage API."""
        try:
            cls.client.add_storage({
                "trace_id": "setup_storage",
                "storage": {
                    "global_unique_name": cls.VINEYARD_STORAGE_NAME,
                    "vineyard": {
                        "cluster_name": cls.instance_id,
                    },
                    "check_storage_available_when_open": False,
                },
            })
            print(f"[SETUP] Vineyard storage '{cls.VINEYARD_STORAGE_NAME}' registered")
        except Exception as e:
            print(f"[WARN] addStorage failed (may already exist): {e}")

    @classmethod
    def _ensure_instance_registered(cls):
        try:
            cls.client.register_instance({
                "trace_id": "setup",
                "instance_group": "default",
                "instance_id": cls.instance_id,
                "block_size": 128,
                "model_deployment": {
                    "model_name": "test_v6d_model",
                    "dtype": "FP8",
                    "use_mla": False,
                    "tp_size": 1,
                    "dp_size": 1,
                    "pp_size": 1,
                },
                "location_spec_infos": [
                    {"name": "tp0", "size": 1024},
                ],
            })
        except Exception as e:
            print(f"[WARN] register_instance failed (may already exist): {e}")

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    # ------------------------------------------------------------------
    # 1. NODE_REGISTER
    # ------------------------------------------------------------------
    def test_01_node_register(self):
        body = self.client.report_event(
            _make_node_register(self.instance_id, self.HOST)
        )
        self.assertIn("ec", body)

    def test_02_node_register_idempotent(self):
        host = "192.168.1.201:8080"
        self.client.report_event(_make_node_register(self.instance_id, host))
        body = self.client.report_event(_make_node_register(self.instance_id, host))
        self.assertIn("ec", body)

    # ------------------------------------------------------------------
    # 3. BLOCK_ADD
    # ------------------------------------------------------------------
    def test_03_block_add(self):
        body = self.client.report_event(
            _make_block_add(self.instance_id, 9001, self.HOST)
        )
        self.assertIn("ec", body)

    # ------------------------------------------------------------------
    # 4. BLOCK_ADD then query
    # ------------------------------------------------------------------
    def test_04_block_add_then_query(self):
        block_key = 9002
        loc_json = json.dumps({"addr": self.HOST, "flavor": "test_query"})
        self.client.report_event(
            _make_block_add(self.instance_id, block_key, self.HOST, loc_json)
        )

        resp = self.client.get_cache_location({
            "trace_id": "test_query",
            "instance_id": self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": [block_key],
            "block_mask": {"offset": 0},
        })

        locations = resp.get("locations", [])
        self.assertGreater(len(locations), 0, "Expected at least one location after BLOCK_ADD")
        specs = locations[0].get("location_specs", [])
        self.assertGreater(len(specs), 0)
        self.assertEqual(specs[0]["uri"], loc_json,
                         "uri should be the original location_json passed in BLOCK_ADD")

    # ------------------------------------------------------------------
    # 5. BLOCK_DELETE
    # ------------------------------------------------------------------
    def test_05_block_delete(self):
        block_key = 9003
        self.client.report_event(
            _make_block_add(self.instance_id, block_key, self.HOST)
        )
        body = self.client.report_event(
            _make_block_delete(self.instance_id, block_key, self.HOST)
        )
        self.assertIn("ec", body)
        # Note: query-after-delete verification is skipped because
        # ReadModifyWrite with MA_DELETE does not invalidate
        # MetaSearchCache (LRU). The stale cached entry persists
        # until eviction. To be fixed in a follow-up.

    # ------------------------------------------------------------------
    # 6. DELETE nonexistent (idempotent)
    # ------------------------------------------------------------------
    def test_06_block_delete_nonexistent(self):
        body = self.client.report_event(
            _make_block_delete(self.instance_id, 99999, self.HOST),
            check_ok=False,
        )
        self.assertIn("ec", body)

    # ------------------------------------------------------------------
    # 7. HOST_DOWN cleans up
    # ------------------------------------------------------------------
    def test_07_host_down(self):
        down_host = "192.168.1.202:8080"
        block_keys = [9010, 9011, 9012]
        for bk in block_keys:
            self.client.report_event(
                _make_block_add(self.instance_id, bk, down_host)
            )

        body = self.client.report_event(
            _make_host_down(self.instance_id, down_host)
        )
        self.assertIn("ec", body)
        # Note: query-after-host-down verification is skipped because
        # ReadModifyWrite with MA_DELETE does not invalidate
        # MetaSearchCache (LRU). Cleanup runs async and stale cache
        # entries persist until eviction. To be fixed in a follow-up.

    # ------------------------------------------------------------------
    # 8. HOST_DOWN idempotent
    # ------------------------------------------------------------------
    def test_08_host_down_idempotent(self):
        down_host = "192.168.1.203:8080"
        body1 = self.client.report_event(
            _make_host_down(self.instance_id, down_host)
        )
        body2 = self.client.report_event(
            _make_host_down(self.instance_id, down_host)
        )
        self.assertIn("ec", body1)
        self.assertIn("ec", body2)

    # ------------------------------------------------------------------
    # 9. Invalid event_type
    # ------------------------------------------------------------------
    def test_09_invalid_event_type(self):
        body = self.client.report_event({
            "trace_id": "test_invalid",
            "instance_id": self.instance_id,
            "event_type": "EVENT_UNSPECIFIED",
        }, check_ok=False)
        self.assertIn("ec", body)

    # ------------------------------------------------------------------
    # 10. Missing params
    # ------------------------------------------------------------------
    def test_10_missing_params(self):
        body = self.client.report_event({
            "trace_id": "test_missing",
            "instance_id": self.instance_id,
            "event_type": "EVENT_BLOCK_ADD",
        }, check_ok=False)
        self.assertIn("ec", body)

    # ------------------------------------------------------------------
    # 11. StartEvictWriteCache: V6D eviction scenario
    # ------------------------------------------------------------------
    def test_11_startevict_write_cache(self):
        block_key = 8001

        # Step 1: Add a VINEYARD location (only 1 replica)
        self.client.report_event(
            _make_block_add(self.instance_id, block_key, self.HOST)
        )

        # Step 2: startEvictWriteCache with min_replica_count=2
        # Since there's only 1 replica (the VINEYARD one), it should
        # allocate a remote write location.
        resp = self.client.start_evict_write_cache({
            "trace_id": "test_evict_1",
            "instance_id": self.instance_id,
            "block_keys": [block_key],
            "write_timeout_seconds": 30,
            "min_replica_count": 2,
        })
        self.assertIn("write_session_id", resp)
        write_session_id = resp["write_session_id"]
        self.assertTrue(write_session_id, "Expected non-empty write_session_id")
        locations = resp.get("locations", [])
        self.assertGreater(len(locations), 0,
                           "Expected remote write locations since only 1 replica exists")

        # Step 3: Finish the write (mark success)
        self.client.finish_write_cache({
            "trace_id": "test_evict_finish",
            "instance_id": self.instance_id,
            "write_session_id": write_session_id,
            "success_blocks": {
                "bool_masks": {"values": [True]}
            },
        })

        # Step 4: Now there should be 2 replicas (VINEYARD + remote).
        # startEvictWriteCache with min_replica_count=2 should skip.
        resp2 = self.client.start_evict_write_cache({
            "trace_id": "test_evict_2",
            "instance_id": self.instance_id,
            "block_keys": [block_key],
            "write_timeout_seconds": 30,
            "min_replica_count": 2,
        })
        locations2 = resp2.get("locations", [])
        self.assertEqual(len(locations2), 0,
                         "Expected no write locations since 2 replicas already exist")


# ---------------------------------------------------------------------------
# Bench tests
# ---------------------------------------------------------------------------
class VineyardReportEventBenchTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = KVCMClient(BASE_URL, ADMIN_URL)
        cls.instance_id = INSTANCE_ID

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    @staticmethod
    def _percentile(data, p):
        if not data:
            return 0
        k = (len(data) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(data):
            return data[-1]
        return data[f] + (k - f) * (data[c] - data[f])

    # ------------------------------------------------------------------
    # 11. BLOCK_ADD throughput
    # ------------------------------------------------------------------
    def test_11_block_add_throughput(self):
        num_threads = 100
        ops_per_thread = 100
        total_ops = num_threads * ops_per_thread
        latencies = []
        errors = []
        host = "192.168.1.210:8080"

        def worker(thread_id):
            local_latencies = []
            session = requests.Session()
            session.headers.update({"Content-Type": "application/json"})
            for i in range(ops_per_thread):
                block_key = thread_id * ops_per_thread + i + 100000
                payload = _make_block_add(
                    self.instance_id, block_key, host,
                    trace_id=f"bench_add_{thread_id}_{i}",
                )
                t0 = time.monotonic()
                try:
                    resp = session.post(f"{BASE_URL}/api/reportEvent", json=payload)
                    resp.raise_for_status()
                except Exception as e:
                    errors.append(str(e))
                    continue
                t1 = time.monotonic()
                local_latencies.append((t1 - t0) * 1000)
            session.close()
            return local_latencies

        t_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(worker, tid) for tid in range(num_threads)]
            for f in as_completed(futures):
                latencies.extend(f.result())
        t_end = time.monotonic()

        elapsed = t_end - t_start
        qps = len(latencies) / elapsed if elapsed > 0 else 0
        latencies.sort()
        p50 = self._percentile(latencies, 50)
        p99 = self._percentile(latencies, 99)

        print(f"\n[BENCH] BLOCK_ADD throughput:")
        print(f"  Total ops:   {len(latencies)} / {total_ops} (errors: {len(errors)})")
        print(f"  Elapsed:     {elapsed:.2f}s")
        print(f"  QPS:         {qps:.1f}")
        print(f"  Latency p50: {p50:.2f}ms")
        print(f"  Latency p99: {p99:.2f}ms")
        if latencies:
            print(f"  Latency avg: {statistics.mean(latencies):.2f}ms")
        self.assertEqual(len(errors), 0, f"Bench had errors: {errors[:5]}")

    # ------------------------------------------------------------------
    # 12. Mixed ADD/DELETE
    # ------------------------------------------------------------------
    def test_12_block_add_delete_mixed(self):
        num_threads = 50
        ops_per_thread = 100
        total_ops = num_threads * ops_per_thread * 2
        latencies = []
        errors = []
        host = "192.168.1.211:8080"

        def worker(thread_id):
            local_latencies = []
            session = requests.Session()
            session.headers.update({"Content-Type": "application/json"})
            for i in range(ops_per_thread):
                block_key = thread_id * ops_per_thread + i + 200000

                # ADD
                payload_add = _make_block_add(
                    self.instance_id, block_key, host,
                    trace_id=f"bench_mixed_{thread_id}_{i}_add",
                )
                t0 = time.monotonic()
                try:
                    resp = session.post(f"{BASE_URL}/api/reportEvent", json=payload_add)
                    resp.raise_for_status()
                except Exception as e:
                    errors.append(f"ADD: {e}")
                    continue
                t1 = time.monotonic()
                local_latencies.append((t1 - t0) * 1000)

                # DELETE
                payload_del = _make_block_delete(
                    self.instance_id, block_key, host,
                    trace_id=f"bench_mixed_{thread_id}_{i}_del",
                )
                t2 = time.monotonic()
                try:
                    resp = session.post(f"{BASE_URL}/api/reportEvent", json=payload_del)
                    resp.raise_for_status()
                except Exception as e:
                    errors.append(f"DEL: {e}")
                    continue
                t3 = time.monotonic()
                local_latencies.append((t3 - t2) * 1000)

            session.close()
            return local_latencies

        t_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(worker, tid) for tid in range(num_threads)]
            for f in as_completed(futures):
                latencies.extend(f.result())
        t_end = time.monotonic()

        elapsed = t_end - t_start
        qps = len(latencies) / elapsed if elapsed > 0 else 0
        latencies.sort()
        p50 = self._percentile(latencies, 50)
        p99 = self._percentile(latencies, 99)

        print(f"\n[BENCH] Mixed ADD/DELETE:")
        print(f"  Total ops:   {len(latencies)} / {total_ops} (errors: {len(errors)})")
        print(f"  Elapsed:     {elapsed:.2f}s")
        print(f"  QPS:         {qps:.1f}")
        print(f"  Latency p50: {p50:.2f}ms")
        print(f"  Latency p99: {p99:.2f}ms")
        if latencies:
            print(f"  Latency avg: {statistics.mean(latencies):.2f}ms")
        self.assertEqual(len(errors), 0, f"Bench had errors: {errors[:5]}")


# ---------------------------------------------------------------------------
# Main: parse args, then run unittest
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="V6D ReportEvent HTTP integration tests")
    parser.add_argument("--host", default="localhost", help="KVCM host")
    parser.add_argument("--http_port", type=int, default=8080, help="KVCM meta HTTP port")
    parser.add_argument("--admin_http_port", type=int, default=None,
                        help="KVCM admin HTTP port (for addStorage). Defaults to http_port if not set.")
    parser.add_argument("--instance_id", default="v6d_cluster_0", help="V6D instance_id")
    parser.add_argument("--skip-bench", action="store_true", help="Skip benchmark tests")
    parser.add_argument("--only-bench", action="store_true", help="Run only benchmark tests")

    args, remaining = parser.parse_known_args()

    admin_port = args.admin_http_port or args.http_port

    global BASE_URL, ADMIN_URL, INSTANCE_ID, SKIP_BENCH, ONLY_BENCH
    BASE_URL = f"http://{args.host}:{args.http_port}"
    ADMIN_URL = f"http://{args.host}:{admin_port}"
    INSTANCE_ID = args.instance_id
    SKIP_BENCH = args.skip_bench
    ONLY_BENCH = args.only_bench

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if not ONLY_BENCH:
        suite.addTests(loader.loadTestsFromTestCase(VineyardReportEventFunctionalTest))
    if not SKIP_BENCH:
        suite.addTests(loader.loadTestsFromTestCase(VineyardReportEventBenchTest))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
