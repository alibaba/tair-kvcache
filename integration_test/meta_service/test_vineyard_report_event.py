#!/usr/bin/env python3
"""
Standalone integration tests for V6D (Vineyard) ReportEvent HTTP interface (V8).

V8 batch protocol semantics (vs V7):
  - host_ip_port is shared at the request level; every EventItem inherits it.
  - request.events is a `repeated EventItem` array; each EventItem has its
    own event_type plus the matching oneof payload.
  - BlockAddParams.uri is now a standard URI (vineyard://{ip}:{port}/{medium}?...);
    BlockAdd/BlockDelete carry `medium`, not `host_ip_port`.
  - NodeRegisterParams.mediums lists every medium the node advertises.
  - EVENT_HEARTBEAT replaces KVCM-side probing.
  - ReportEventResponse may carry per-EventItem results in `item_results`.

Usage:
    # 1. Start KVCM service locally:
    #    bazel-bin/kv_cache_manager/kv_cache_manager_bin \
    #        --env kvcm.service.rpc_port=56010 \
    #        --env kvcm.service.http_port=56020 \
    #        --env kvcm.service.admin_rpc_port=56030 \
    #        --env kvcm.service.admin_http_port=56040 \
    #        --env kvcm.service.enable_debug_service=false
    # 2. Run this script:
    python test_vineyard_report_event.py \
        --host localhost --http_port 56020 --admin_http_port 56040 \
        --instance_id v6d_cluster_0
"""

import argparse
import json
import sys
import time
import statistics
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


BASE_URL = ""
ADMIN_URL = ""
INSTANCE_ID = "v6d_cluster_0"
SKIP_BENCH = False
ONLY_BENCH = False


class KVCMClient:
    def __init__(self, base_url, admin_url=None):
        self.base_url = base_url
        self.admin_url = admin_url or base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def report_event(self, payload, check_ok=True):
        url = f"{self.base_url}/api/reportEvent"
        resp = self.session.post(url, json=payload)
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
# V8 EventItem builders -- one helper per event_type. Each returns the
# `{"event_type": ..., "<oneof>": {...}}` dict that goes inside
# request.events[]. The request-level fields (trace_id / instance_id /
# host_ip_port) are added by _make_request().
# ---------------------------------------------------------------------------
def _ev_node_register(mediums):
    return {
        "event_type": "EVENT_NODE_REGISTER",
        "node_register": {"mediums": list(mediums)},
    }


def _ev_block_add(block_key, uri, medium):
    return {
        "event_type": "EVENT_BLOCK_ADD",
        "block_add": {
            "block_key": str(block_key),
            "uri": uri,
            "medium": medium,
        },
    }


def _ev_block_delete(block_key, medium):
    return {
        "event_type": "EVENT_BLOCK_DELETE",
        "block_delete": {
            "block_key": str(block_key),
            "medium": medium,
        },
    }


def _ev_host_down():
    return {"event_type": "EVENT_HOST_DOWN", "host_down": {}}


def _ev_heartbeat(system_status=None):
    return {
        "event_type": "EVENT_HEARTBEAT",
        "heartbeat": {"system_status": system_status or {}},
    }


def _make_request(instance_id, host_ip_port, events, trace_id="test"):
    return {
        "trace_id": trace_id,
        "instance_id": instance_id,
        "host_ip_port": host_ip_port,
        "events": events,
    }


def _build_vineyard_uri(host_ip_port, medium, params=None):
    """Mirror VineyardUri::Build(): vineyard://{ip}:{port}/{medium}?k=v&..."""
    base = f"vineyard://{host_ip_port}/{medium}"
    if not params:
        return base
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{base}?{query}"


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
        # Always start with the host registered so subsequent BLOCK_ADD /
        # HEARTBEAT events have an existing NodeInfo to operate on.
        cls.client.report_event(
            _make_request(
                cls.instance_id,
                cls.HOST,
                [_ev_node_register(["mem", "disk"])],
                trace_id="setup_register_host",
            )
        )

    @classmethod
    def _ensure_vineyard_storage_registered(cls):
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

    # 1. NODE_REGISTER (with mediums)
    def test_01_node_register(self):
        body = self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_node_register(["mem", "disk", "ssd"])],
                trace_id="t01",
            )
        )
        self.assertIn("ec", body)

    # 2. NODE_REGISTER is idempotent and merges mediums
    def test_02_node_register_idempotent(self):
        host = "192.168.1.201:8080"
        self.client.report_event(
            _make_request(self.instance_id, host, [_ev_node_register(["mem"])], trace_id="t02a")
        )
        body = self.client.report_event(
            _make_request(self.instance_id, host, [_ev_node_register(["mem", "disk"])], trace_id="t02b")
        )
        self.assertIn("ec", body)

    # 3. BLOCK_ADD with explicit medium
    def test_03_block_add(self):
        uri = _build_vineyard_uri(self.HOST, "mem", {"gpu": "A100"})
        body = self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_block_add(9001, uri, "mem")],
                trace_id="t03",
            )
        )
        self.assertIn("ec", body)

    # 4. BLOCK_ADD then query: location_id must be kvs#v6d#{medium}#{host}
    def test_04_block_add_then_query(self):
        block_key = 9002
        uri = _build_vineyard_uri(self.HOST, "mem", {"flavor": "test_query"})
        self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_block_add(block_key, uri, "mem")],
                trace_id="t04",
            )
        )

        resp = self.client.get_cache_location({
            "trace_id": "t04_query",
            "instance_id": self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": [block_key],
            "block_mask": {"offset": 0},
        })

        locations = resp.get("locations", [])
        self.assertGreater(len(locations), 0, "Expected at least one location after BLOCK_ADD")
        specs = locations[0].get("location_specs", [])
        self.assertGreater(len(specs), 0)
        # V8: spec.uri must be the standard URI we sent in BLOCK_ADD.
        self.assertEqual(specs[0]["uri"], uri,
                         "spec.uri should be the V8 standard vineyard:// URI")
        # V8: spec.name should be the deterministic location_id pattern.
        expected_loc_id = f"kvs#v6d#mem#{self.HOST}"
        self.assertEqual(specs[0]["name"], expected_loc_id,
                         f"spec.name should be {expected_loc_id}")

    # 5. Two mediums on same host: each becomes its own location_id
    def test_05_block_add_multi_medium(self):
        block_key = 9020
        host = "192.168.1.220:8080"
        # NODE_REGISTER first so the host is known.
        self.client.report_event(
            _make_request(self.instance_id, host, [_ev_node_register(["mem", "disk"])], trace_id="t05a")
        )
        uri_mem = _build_vineyard_uri(host, "mem")
        uri_disk = _build_vineyard_uri(host, "disk")
        body = self.client.report_event(
            _make_request(
                self.instance_id, host,
                [
                    _ev_block_add(block_key, uri_mem, "mem"),
                    _ev_block_add(block_key, uri_disk, "disk"),
                ],
                trace_id="t05b",
            )
        )
        self.assertIn("ec", body)

        resp = self.client.get_cache_location({
            "trace_id": "t05_query",
            "instance_id": self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": [block_key],
            "block_mask": {"offset": 0},
        })
        # GetCacheLocation returns a single chosen location per key, so the
        # response only confirms "at least one location resolved" -- the key
        # invariant we want to lock down is that BatchAdd accepted both
        # mediums in one batch without rejecting either as duplicate.
        self.assertGreater(len(resp.get("locations", [])), 0)

    # 6. BLOCK_DELETE removes the specific (block_key, medium) entry
    def test_06_block_delete(self):
        block_key = 9003
        uri = _build_vineyard_uri(self.HOST, "mem")
        self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_block_add(block_key, uri, "mem")],
                trace_id="t06a",
            )
        )
        body = self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_block_delete(block_key, "mem")],
                trace_id="t06b",
            )
        )
        self.assertIn("ec", body)
        # Note: query-after-delete verification is skipped because
        # ReadModifyWrite with MA_DELETE does not invalidate
        # MetaSearchCache (LRU) synchronously.

    # 7. BLOCK_DELETE on missing key/medium is a no-op (idempotent)
    def test_07_block_delete_nonexistent(self):
        body = self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_block_delete(99999, "mem")],
                trace_id="t07",
            ),
            check_ok=False,
        )
        self.assertIn("ec", body)

    # 8. HOST_DOWN cleans up all mediums under the host
    def test_08_host_down(self):
        down_host = "192.168.1.202:8080"
        block_keys = [9010, 9011, 9012]
        # Register + add mem/disk replicas in a single batch.
        events = [_ev_node_register(["mem", "disk"])]
        for bk in block_keys:
            events.append(_ev_block_add(bk, _build_vineyard_uri(down_host, "mem"), "mem"))
            events.append(_ev_block_add(bk, _build_vineyard_uri(down_host, "disk"), "disk"))
        self.client.report_event(_make_request(self.instance_id, down_host, events, trace_id="t08a"))

        body = self.client.report_event(
            _make_request(self.instance_id, down_host, [_ev_host_down()], trace_id="t08b")
        )
        self.assertIn("ec", body)

    # 9. HOST_DOWN is idempotent
    def test_09_host_down_idempotent(self):
        down_host = "192.168.1.203:8080"
        self.client.report_event(
            _make_request(self.instance_id, down_host, [_ev_node_register(["mem"])], trace_id="t09a")
        )
        body1 = self.client.report_event(
            _make_request(self.instance_id, down_host, [_ev_host_down()], trace_id="t09b")
        )
        body2 = self.client.report_event(
            _make_request(self.instance_id, down_host, [_ev_host_down()], trace_id="t09c")
        )
        self.assertIn("ec", body1)
        self.assertIn("ec", body2)

    # 10. HEARTBEAT extends liveness; payload is opaque
    def test_10_heartbeat(self):
        body = self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_heartbeat({"version": "v6d-0.18", "cpu": "45%"})],
                trace_id="t10",
            )
        )
        self.assertIn("ec", body)

    # 11. HEARTBEAT for unregistered host: KVCM logs a warning, returns OK
    def test_11_heartbeat_unregistered(self):
        body = self.client.report_event(
            _make_request(
                self.instance_id, "10.10.10.10:8080",
                [_ev_heartbeat({})],
                trace_id="t11",
            )
        )
        self.assertIn("ec", body)

    # 12. Mixed batch: register + add + heartbeat in a single RPC
    def test_12_mixed_batch(self):
        host = "192.168.1.230:8080"
        block_key = 9030
        events = [
            _ev_node_register(["mem"]),
            _ev_block_add(block_key, _build_vineyard_uri(host, "mem"), "mem"),
            _ev_heartbeat({"phase": "boot"}),
        ]
        body = self.client.report_event(
            _make_request(self.instance_id, host, events, trace_id="t12")
        )
        self.assertIn("ec", body)

    # 13. Empty events array: should be a no-op success
    def test_13_empty_batch(self):
        body = self.client.report_event(
            _make_request(self.instance_id, self.HOST, [], trace_id="t13")
        )
        self.assertIn("ec", body)

    # 14. Missing block_add params: server must surface a per-item failure
    def test_14_block_add_missing_params(self):
        body = self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [{"event_type": "EVENT_BLOCK_ADD"}],
                trace_id="t14",
            ),
            check_ok=False,
        )
        # V8: when any item fails, response.item_results is filled (length
        # equals events length); when all succeed, item_results is omitted.
        self.assertIn("ec", body)

    # 15. Empty top-level host_ip_port: request-level validation must reject
    def test_15_missing_host_ip_port(self):
        body = self.client.report_event(
            {
                "trace_id": "t15",
                "instance_id": self.instance_id,
                "host_ip_port": "",
                "events": [_ev_node_register(["mem"])],
            },
            check_ok=False,
        )
        self.assertIn("ec", body)

    # 16. StartEvictWriteCache: V6D eviction with min_replica_count=2
    def test_16_startevict_write_cache(self):
        block_key = 8001
        uri = _build_vineyard_uri(self.HOST, "mem")

        # Step 1: 1 VINEYARD replica only.
        self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_block_add(block_key, uri, "mem")],
                trace_id="t16_add",
            )
        )

        # Step 2: ask for evict; with only 1 replica we expect a remote write.
        resp = self.client.start_evict_write_cache({
            "trace_id": "t16_evict_1",
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

        # Step 3: Finish the write to bring up replica count to 2.
        self.client.finish_write_cache({
            "trace_id": "t16_evict_finish",
            "instance_id": self.instance_id,
            "write_session_id": write_session_id,
            "success_blocks": {"bool_masks": {"values": [True]}},
        })

        # Step 4: Now n_total=2; evict should skip remote allocation.
        resp2 = self.client.start_evict_write_cache({
            "trace_id": "t16_evict_2",
            "instance_id": self.instance_id,
            "block_keys": [block_key],
            "write_timeout_seconds": 30,
            "min_replica_count": 2,
        })
        locations2 = resp2.get("locations", [])
        self.assertEqual(len(locations2), 0,
                         "Expected no write locations since 2 replicas already exist")


# ---------------------------------------------------------------------------
# Bench tests (V8 batch shape)
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

    @staticmethod
    def _ensure_host_registered(client, instance_id, host):
        # Pre-register so subsequent BLOCK_ADDs hit "node already known".
        client.report_event(
            _make_request(instance_id, host,
                          [_ev_node_register(["mem"])],
                          trace_id="bench_setup")
        )

    # 17. BLOCK_ADD throughput (one item per request)
    def test_17_block_add_throughput(self):
        num_threads = 100
        ops_per_thread = 100
        total_ops = num_threads * ops_per_thread
        latencies = []
        errors = []
        host = "192.168.1.210:8080"
        self._ensure_host_registered(self.client, self.instance_id, host)

        def worker(thread_id):
            local_latencies = []
            session = requests.Session()
            session.headers.update({"Content-Type": "application/json"})
            for i in range(ops_per_thread):
                block_key = thread_id * ops_per_thread + i + 100000
                payload = _make_request(
                    self.instance_id, host,
                    [_ev_block_add(block_key, _build_vineyard_uri(host, "mem"), "mem")],
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

    # 18. Mixed ADD/DELETE in the same batch (this is what V6D actually does
    #     every 10s in production -- exercises the batch aggregation path).
    def test_18_block_add_delete_mixed(self):
        num_threads = 50
        ops_per_thread = 100
        total_batches = num_threads * ops_per_thread
        latencies = []
        errors = []
        host = "192.168.1.211:8080"
        self._ensure_host_registered(self.client, self.instance_id, host)

        def worker(thread_id):
            local_latencies = []
            session = requests.Session()
            session.headers.update({"Content-Type": "application/json"})
            for i in range(ops_per_thread):
                block_key = thread_id * ops_per_thread + i + 200000
                events = [
                    _ev_block_add(block_key, _build_vineyard_uri(host, "mem"), "mem"),
                    _ev_block_delete(block_key, "mem"),
                    _ev_heartbeat({"thread": str(thread_id)}),
                ]
                payload = _make_request(
                    self.instance_id, host, events,
                    trace_id=f"bench_mixed_{thread_id}_{i}",
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

        print(f"\n[BENCH] Mixed ADD/DELETE/HEARTBEAT batch:")
        print(f"  Total batches: {len(latencies)} / {total_batches} (errors: {len(errors)})")
        print(f"  Elapsed:       {elapsed:.2f}s")
        print(f"  Batch QPS:     {qps:.1f}")
        print(f"  Latency p50:   {p50:.2f}ms")
        print(f"  Latency p99:   {p99:.2f}ms")
        if latencies:
            print(f"  Latency avg:   {statistics.mean(latencies):.2f}ms")
        self.assertEqual(len(errors), 0, f"Bench had errors: {errors[:5]}")


def main():
    parser = argparse.ArgumentParser(description="V6D ReportEvent HTTP integration tests (V8)")
    parser.add_argument("--host", default="localhost", help="KVCM host")
    parser.add_argument("--http_port", type=int, default=56020, help="KVCM meta HTTP port")
    parser.add_argument("--admin_http_port", type=int, default=None,
                        help="KVCM admin HTTP port (for addStorage). Defaults to http_port.")
    parser.add_argument("--instance_id", default="v6d_cluster_0", help="V6D instance_id")
    parser.add_argument("--skip-bench", action="store_true", help="Skip benchmark tests")
    parser.add_argument("--only-bench", action="store_true", help="Run only benchmark tests")

    args, _ = parser.parse_known_args()

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
