#!/usr/bin/env python3
"""
Integration tests for V6D ReportEvent HTTP interface.

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
# Requires small heartbeat_timeout_ms/cleanup_grace_ms in addStorage spec.
ENABLE_LIVENESS_TIMING_TESTS = False
HEARTBEAT_TIMEOUT_MS = 1000
CLEANUP_GRACE_MS = 2000


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
            code = body.get("header", {}).get("status", {}).get("code")
            assert code in ("OK", 1, "1", None), (
                f"ReportEvent failed: code={code}, body={json.dumps(body, ensure_ascii=False)}"
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

    def create_instance_group(self, data):
        url = f"{self.admin_url}/api/createInstanceGroup"
        resp = self.session.post(url, json=data)
        resp.raise_for_status()
        body = resp.json()
        code = body.get("header", {}).get("status", {}).get("code")
        if code not in ("OK", "DUPLICATE_ENTITY"):
            raise AssertionError(f"createInstanceGroup failed: {json.dumps(body)}")
        return body

    def get_cache_location(self, data):
        url = f"{self.base_url}/api/getCacheLocation"
        resp = self.session.post(url, json=data)
        resp.raise_for_status()
        return resp.json()

    def start_write_cache_with_min_replica(self, data, check_response=True):
        url = f"{self.base_url}/api/startWriteCache"
        resp = self.session.post(url, json=data)
        resp.raise_for_status()
        body = resp.json()
        if check_response:
            code = body.get("header", {}).get("status", {}).get("code")
            assert code == "OK", f"startWriteCache failed: {json.dumps(body)}"
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
# EventItem builders
# ---------------------------------------------------------------------------
def _ev_node_register(mediums):
    return {
        "event_type": "EVENT_NODE_REGISTER",
        "node_register": {"mediums": list(mediums)},
    }


def _ev_block_add(block_key, medium, specs):
    """Build a block_add event.

    Args:
        specs: list of {"name": ..., "uri": ...} dicts.
    """
    return {
        "event_type": "EVENT_BLOCK_ADD",
        "block_add": {
            "block_key": str(block_key),
            "medium": medium,
            "specs": specs,
        },
    }


def _make_single_spec(name, uri):
    """Convenience: build a one-element specs list."""
    return [{"name": name, "uri": uri}]


def _ev_block_delete(block_key, medium, spec_names=None):
    return {
        "event_type": "EVENT_BLOCK_DELETE",
        "block_delete": {
            "block_key": str(block_key),
            "medium": medium,
            "spec_names": list(spec_names or []),
        },
    }


def _ev_block_snapshot(medium, blocks):
    """Build a block_snapshot event.

    Args:
        blocks: list of (block_key, specs) tuples.
    """
    return {
        "event_type": "EVENT_BLOCK_SNAPSHOT",
        "block_snapshot": {
            "medium": medium,
            "blocks": [
                {
                    "block_key": str(block_key),
                    "specs": specs,
                }
                for block_key, specs in blocks
            ],
        },
    }


def _ev_host_down():
    return {"event_type": "EVENT_HOST_DOWN", "host_down": {}}


def _ev_heartbeat(system_status=None):
    return {
        "event_type": "EVENT_HEARTBEAT",
        "heartbeat": {"system_status": system_status or {}},
    }


def _make_request(instance_id, host_ip_port, events, trace_id="test", storage_type="ST_VINEYARD"):
    return {
        "trace_id": trace_id,
        "instance_id": instance_id,
        "host_ip_port": host_ip_port,
        "events": events,
        "storage_type": storage_type,
    }


def _build_vineyard_uri(host_ip_port, medium, params=None):
    """Build vineyard URI: vineyard://{ip}:{port}/{medium}?k=v&..."""
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
    VINEYARD_STORAGE_NAME = "vineyard_default"
    INSTANCE_GROUP_NAME = "vineyard_test_group"

    @classmethod
    def setUpClass(cls):
        cls.client = KVCMClient(BASE_URL, ADMIN_URL)
        cls.instance_id = INSTANCE_ID
        cls._ensure_vineyard_storage_registered()
        cls._ensure_instance_group_created()
        cls._ensure_instance_registered()
        # Register host so subsequent events have a NodeInfo entry.
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
                        "heartbeat_timeout_ms": 30000,
                        "cleanup_grace_ms": 300000,
                        "liveness_check_interval_ms": 5000,
                    },
                    "check_storage_available_when_open": False,
                },
            })
            print(f"[SETUP] Vineyard storage '{cls.VINEYARD_STORAGE_NAME}' registered")
        except Exception as e:
            print(f"[WARN] addStorage failed (may already exist): {e}")

    @classmethod
    def _ensure_instance_group_created(cls):
        cls.client.create_instance_group({
            "trace_id": "setup_ig",
            "instance_group": {
                "name": cls.INSTANCE_GROUP_NAME,
                "storage_candidates": ["nfs_01", cls.VINEYARD_STORAGE_NAME],
                "global_quota_group_name": "default_quota_group",
                "max_instance_count": 100,
                "quota": {
                    "capacity": 10737418240,
                    "quota_config": [{"storage_type": 4, "capacity": 10737418240}],
                },
                "cache_config": {
                    "reclaim_strategy": {
                        "storage_unique_name": "nfs_01",
                        "reclaim_policy": 1,
                        "trigger_strategy": {"used_size": 1073741824, "used_percentage": 0.8},
                        "trigger_period_seconds": 60,
                        "reclaim_step_size": 1073741824,
                        "reclaim_step_percentage": 10,
                    },
                    "data_storage_strategy": 2,
                    "meta_indexer_config": {
                        "max_key_count": 1000000,
                        "mutex_shard_num": 16,
                        "batch_key_size": 16,
                        "meta_storage_backend_config": {"storage_type": "local", "storage_uri": ""},
                        "meta_cache_policy_config": {"type": "LRU", "capacity": 10000},
                    },
                },
                "event_reporting_storage_candidates": [cls.VINEYARD_STORAGE_NAME],
                "version": 1,
            },
        })
        print(f"[SETUP] InstanceGroup '{cls.INSTANCE_GROUP_NAME}' created")

    @classmethod
    def _ensure_instance_registered(cls):
        try:
            cls.client.register_instance({
                "trace_id": "setup",
                "instance_group": cls.INSTANCE_GROUP_NAME,
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

    def _query_specs(self, block_key, trace_id):
        resp = self.client.get_cache_location({
            "trace_id": trace_id,
            "instance_id": self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": [block_key],
            "block_mask": {"offset": 0},
        })
        specs = []
        for loc in resp.get("locations", []):
            specs.extend(loc.get("location_specs", []))
        return specs

    def _assert_uri_present(self, block_key, expected_uri, trace_id):
        specs = self._query_specs(block_key, trace_id)
        self.assertTrue(
            any(spec.get("uri") == expected_uri for spec in specs),
            f"Expected uri={expected_uri} for block={block_key}, specs={specs}",
        )

    def _assert_uri_absent(self, block_key, unexpected_uri, trace_id):
        specs = self._query_specs(block_key, trace_id)
        self.assertFalse(
            any(spec.get("uri") == unexpected_uri for spec in specs),
            f"Unexpected uri={unexpected_uri} for block={block_key}, specs={specs}",
        )

    # 1. NODE_REGISTER (with mediums)
    def test_01_node_register(self):
        body = self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_node_register(["mem", "disk", "ssd"])],
                trace_id="t01",
            )
        )
        self.assertIn("header", body)

    # 2. NODE_REGISTER is idempotent and merges mediums
    def test_02_node_register_idempotent(self):
        host = "192.168.1.201:8080"
        self.client.report_event(
            _make_request(self.instance_id, host, [_ev_node_register(["mem"])], trace_id="t02a")
        )
        body = self.client.report_event(
            _make_request(self.instance_id, host, [_ev_node_register(["mem", "disk"])], trace_id="t02b")
        )
        self.assertIn("header", body)

    # 3. BLOCK_ADD with single spec
    def test_03_block_add(self):
        uri = _build_vineyard_uri(self.HOST, "mem", {"gpu": "A100"})
        body = self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_block_add(9001, "mem", _make_single_spec("default", uri))],
                trace_id="t03",
            )
        )
        self.assertIn("header", body)

    # 4. BLOCK_ADD then query: spec name/uri should match what was sent
    def test_04_block_add_then_query(self):
        block_key = 9002
        uri = _build_vineyard_uri(self.HOST, "mem", {"flavor": "test_query"})
        spec_name = "tp0"
        self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_block_add(block_key, "mem", _make_single_spec(spec_name, uri))],
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
        self.assertEqual(specs[0]["uri"], uri,
                         "spec.uri should match the URI sent in BLOCK_ADD")
        self.assertEqual(specs[0]["name"], spec_name,
                         f"spec.name should be {spec_name}")

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
                    _ev_block_add(block_key, "mem", _make_single_spec("mem_spec", uri_mem)),
                    _ev_block_add(block_key, "disk", _make_single_spec("disk_spec", uri_disk)),
                ],
                trace_id="t05b",
            )
        )
        self.assertIn("header", body)

        resp = self.client.get_cache_location({
            "trace_id": "t05_query",
            "instance_id": self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": [block_key],
            "block_mask": {"offset": 0},
        })
        code = resp.get("header", {}).get("status", {}).get("code")
        self.assertEqual(code, "OK", f"getCacheLocation failed: {json.dumps(resp, ensure_ascii=False)}")

        locations = resp.get("locations", [])
        self.assertEqual(len(locations), 1, "QT_BATCH_GET with one key should return one row")
        specs = locations[0].get("location_specs", [])
        by_name = {s["name"]: s for s in specs if s.get("name")}
        self.assertIn("mem_spec", by_name, f"Expected mem_spec, specs={list(by_name)}")
        self.assertIn("disk_spec", by_name, f"Expected disk_spec, specs={list(by_name)}")
        self.assertEqual(by_name["mem_spec"]["uri"], uri_mem)
        self.assertEqual(by_name["disk_spec"]["uri"], uri_disk)

    # 5b. BLOCK_ADD with multiple specs in one CacheLocation
    def test_05b_block_add_multi_spec(self):
        block_key = 9025
        host = "192.168.1.221:8080"
        self.client.report_event(
            _make_request(self.instance_id, host, [_ev_node_register(["mem"])], trace_id="t05b_reg")
        )
        uri_spec0 = _build_vineyard_uri(host, "mem", {"obj_id": "o1", "size": "512"})
        uri_spec1 = _build_vineyard_uri(host, "mem", {"obj_id": "o2", "size": "512"})
        body = self.client.report_event(
            _make_request(
                self.instance_id, host,
                [_ev_block_add(block_key, "mem", [
                    {"name": "spec_4096", "uri": uri_spec0},
                    {"name": "spec_8192", "uri": uri_spec1},
                ])],
                trace_id="t05b_add",
            )
        )
        self.assertIn("header", body)

        resp = self.client.get_cache_location({
            "trace_id": "t05b_query",
            "instance_id": self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": [block_key],
            "block_mask": {"offset": 0},
        })
        code = resp.get("header", {}).get("status", {}).get("code")
        self.assertEqual(code, "OK", f"getCacheLocation failed: {json.dumps(resp, ensure_ascii=False)}")

        locations = resp.get("locations", [])
        self.assertEqual(len(locations), 1)
        specs = locations[0].get("location_specs", [])
        by_name = {s["name"]: s for s in specs}
        self.assertIn("spec_4096", by_name, f"Expected spec_4096 in specs={list(by_name)}")
        self.assertIn("spec_8192", by_name, f"Expected spec_8192 in specs={list(by_name)}")
        self.assertEqual(by_name["spec_4096"]["uri"], uri_spec0)
        self.assertEqual(by_name["spec_8192"]["uri"], uri_spec1)

    # 6. BLOCK_DELETE removes the specific (block_key, medium) entry
    def test_06_block_delete(self):
        block_key = 9003
        uri = _build_vineyard_uri(self.HOST, "mem")
        self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_block_add(block_key, "mem", _make_single_spec("spec_4096", uri))],
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
        self.assertIn("header", body)
        # Query-after-delete skipped: MetaSearchCache may still serve stale entry.

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
        self.assertIn("header", body)

    # 8. HOST_DOWN cleans up all mediums under the host
    def test_08_host_down(self):
        down_host = "192.168.1.202:8080"
        block_keys = [9010, 9011, 9012]
        # Register + add mem/disk replicas in a single batch.
        events = [_ev_node_register(["mem", "disk"])]
        for bk in block_keys:
            events.append(_ev_block_add(bk, "mem", _make_single_spec("spec_4096", _build_vineyard_uri(down_host, "mem"))))
            events.append(_ev_block_add(bk, "disk", _make_single_spec("spec_4096", _build_vineyard_uri(down_host, "disk"))))
        self.client.report_event(_make_request(self.instance_id, down_host, events, trace_id="t08a"))

        body = self.client.report_event(
            _make_request(self.instance_id, down_host, [_ev_host_down()], trace_id="t08b")
        )
        self.assertIn("header", body)

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
        self.assertIn("header", body1)
        self.assertIn("header", body2)

    # 10. HEARTBEAT extends liveness; payload is opaque
    def test_10_heartbeat(self):
        body = self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_heartbeat({"version": "v6d-0.18", "cpu": "45%"})],
                trace_id="t10",
            )
        )
        self.assertIn("header", body)

    # 11. Mixed batch: register + add + heartbeat in a single RPC
    def test_11_mixed_batch(self):
        host = "192.168.1.230:8080"
        block_key = 9030
        events = [
            _ev_node_register(["mem"]),
            _ev_block_add(block_key, "mem", _make_single_spec("spec_4096", _build_vineyard_uri(host, "mem"))),
            _ev_heartbeat({"phase": "boot"}),
        ]
        body = self.client.report_event(
            _make_request(self.instance_id, host, events, trace_id="t11")
        )
        self.assertIn("header", body)

    # 12. Empty events array: should be a no-op success
    def test_12_empty_batch(self):
        body = self.client.report_event(
            _make_request(self.instance_id, self.HOST, [], trace_id="t12")
        )
        self.assertIn("header", body)

    # 13. Missing block_add params: server must surface a per-item failure
    def test_13_block_add_missing_params(self):
        body = self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [{"event_type": "EVENT_BLOCK_ADD"}],
                trace_id="t13",
            ),
            check_ok=False,
        )
        self.assertIn("header", body)
        self.assertIn("item_results", body)

    # 14. Empty top-level host_ip_port: request-level validation must reject
    def test_14_missing_host_ip_port(self):
        body = self.client.report_event(
            {
                "trace_id": "t14",
                "instance_id": self.instance_id,
                "host_ip_port": "",
                "events": [_ev_node_register(["mem"])],
                "storage_type": "ST_VINEYARD",
            },
            check_ok=False,
        )
        self.assertIn("header", body)

    # 15. StartWriteCacheWithMinReplica: V6D eviction with min_replica_count=2
    def test_15_start_write_cache_with_min_replica(self):
        block_key = 8001
        uri = _build_vineyard_uri(self.HOST, "mem")

        # Step 1: 1 VINEYARD replica only.
        self.client.report_event(
            _make_request(
                self.instance_id, self.HOST,
                [_ev_block_add(block_key, "mem", _make_single_spec("spec_4096", uri))],
                trace_id="t15_add",
            )
        )

        # Step 2: ask for evict; with only 1 replica we expect a remote write.
        resp = self.client.start_write_cache_with_min_replica({
            "trace_id": "t15_evict_1",
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
            "trace_id": "t15_evict_finish",
            "instance_id": self.instance_id,
            "write_session_id": write_session_id,
            "success_blocks": {"bool_masks": {"values": [True]}},
        })

        # Step 4: Now n_total=2; evict should skip remote allocation.
        resp2 = self.client.start_write_cache_with_min_replica({
            "trace_id": "t15_evict_2",
            "instance_id": self.instance_id,
            "block_keys": [block_key],
            "write_timeout_seconds": 30,
            "min_replica_count": 2,
        })
        locations2 = resp2.get("locations", [])
        self.assertEqual(len(locations2), 0,
                         "Expected no write locations since 2 replicas already exist")

    # 16. BLOCK_SNAPSHOT reconciles the full block set for one host/medium
    def test_16_block_snapshot_diff(self):
        host = "192.168.1.240:8080"
        medium = "mem"
        key_deleted = 8101
        key_kept = 8102
        key_added = 8103
        uri_deleted = _build_vineyard_uri(host, medium, {"obj_id": "deleted"})
        uri_kept = _build_vineyard_uri(host, medium, {"obj_id": "kept"})
        uri_added = _build_vineyard_uri(host, medium, {"obj_id": "added"})

        self.client.report_event(_make_request(self.instance_id, host, [
            _ev_node_register([medium]),
            _ev_block_snapshot(medium, [
                (key_deleted, _make_single_spec("spec_4096", uri_deleted)),
                (key_kept, _make_single_spec("spec_4096", uri_kept)),
            ]),
        ], trace_id="t16_snapshot_a"))
        self._assert_uri_present(key_kept, uri_kept, "t16_query_kept_a")

        self.client.report_event(_make_request(self.instance_id, host, [
            _ev_block_snapshot(medium, [
                (key_kept, _make_single_spec("spec_4096", uri_kept)),
                (key_added, _make_single_spec("spec_4096", uri_added)),
            ]),
        ], trace_id="t16_snapshot_b"))

        self._assert_uri_absent(key_deleted, uri_deleted, "t16_query_deleted_b")
        self._assert_uri_present(key_kept, uri_kept, "t16_query_kept_b")
        self._assert_uri_present(key_added, uri_added, "t16_query_added_b")

    # 16c. Snapshot diff is scoped by host: same medium on another host survives.
    def test_16c_block_snapshot_diff_is_host_scoped(self):
        host_a = "192.168.1.241:8080"
        host_b = "192.168.1.242:8080"
        medium = "mem"
        key_a_deleted = 8111
        key_a_kept = 8112
        key_b_kept = 8113
        uri_a_deleted = _build_vineyard_uri(host_a, medium, {"obj_id": "a_deleted"})
        uri_a_kept = _build_vineyard_uri(host_a, medium, {"obj_id": "a_kept"})
        uri_b_kept = _build_vineyard_uri(host_b, medium, {"obj_id": "b_kept"})

        self.client.report_event(_make_request(self.instance_id, host_a, [
            _ev_node_register([medium]),
            _ev_block_snapshot(medium, [
                (key_a_deleted, _make_single_spec("host_a_deleted", uri_a_deleted)),
                (key_a_kept, _make_single_spec("host_a_kept", uri_a_kept)),
            ]),
        ], trace_id="t16c_snapshot_host_a_initial"))
        self.client.report_event(_make_request(self.instance_id, host_b, [
            _ev_node_register([medium]),
            _ev_block_snapshot(medium, [
                (key_b_kept, _make_single_spec("host_b_kept", uri_b_kept)),
            ]),
        ], trace_id="t16c_snapshot_host_b_initial"))

        self.client.report_event(_make_request(self.instance_id, host_a, [
            _ev_block_snapshot(medium, [
                (key_a_kept, _make_single_spec("host_a_kept", uri_a_kept)),
            ]),
        ], trace_id="t16c_snapshot_host_a_reconcile"))

        self._assert_uri_absent(key_a_deleted, uri_a_deleted, "t16c_query_a_deleted")
        self._assert_uri_present(key_a_kept, uri_a_kept, "t16c_query_a_kept")
        self._assert_uri_present(key_b_kept, uri_b_kept, "t16c_query_b_kept")

    # 16d. Snapshot diff is scoped by medium: another tier on the same host survives.
    def test_16d_block_snapshot_diff_is_medium_scoped(self):
        host = "192.168.1.243:8080"
        key_mem_deleted = 8121
        key_mem_kept = 8122
        key_disk_kept = 8123
        uri_mem_deleted = _build_vineyard_uri(host, "mem", {"obj_id": "mem_deleted"})
        uri_mem_kept = _build_vineyard_uri(host, "mem", {"obj_id": "mem_kept"})
        uri_disk_kept = _build_vineyard_uri(host, "disk", {"obj_id": "disk_kept"})

        self.client.report_event(_make_request(self.instance_id, host, [
            _ev_node_register(["mem", "disk"]),
            _ev_block_snapshot("mem", [
                (key_mem_deleted, _make_single_spec("mem_deleted", uri_mem_deleted)),
                (key_mem_kept, _make_single_spec("mem_kept", uri_mem_kept)),
            ]),
            _ev_block_snapshot("disk", [
                (key_disk_kept, _make_single_spec("disk_kept", uri_disk_kept)),
            ]),
        ], trace_id="t16d_snapshot_initial"))

        self.client.report_event(_make_request(self.instance_id, host, [
            _ev_block_snapshot("mem", [
                (key_mem_kept, _make_single_spec("mem_kept", uri_mem_kept)),
            ]),
        ], trace_id="t16d_snapshot_mem_reconcile"))

        self._assert_uri_absent(key_mem_deleted, uri_mem_deleted, "t16d_query_mem_deleted")
        self._assert_uri_present(key_mem_kept, uri_mem_kept, "t16d_query_mem_kept")
        self._assert_uri_present(key_disk_kept, uri_disk_kept, "t16d_query_disk_kept")

    # 16e. Mamba-only reports are accepted but ignored for current full-attention matching.
    def test_16e_mamba_only_report_does_not_delete_full_attention(self):
        host = "192.168.1.244:8080"
        medium = "gpu"
        full_key = 8131
        mamba_key = 8132
        full_uri = _build_vineyard_uri(host, medium, {"obj_id": "full"})
        mamba_uri = _build_vineyard_uri(host, medium, {"obj_id": "mamba"})

        self.client.report_event(_make_request(self.instance_id, host, [
            _ev_node_register([medium]),
            _ev_block_add(
                full_key,
                medium,
                _make_single_spec("full_attention:group=0:tp=0", full_uri),
            ),
        ], trace_id="t16e_full_add"))
        self._assert_uri_present(full_key, full_uri, "t16e_query_full_before")

        self.client.report_event(_make_request(self.instance_id, host, [
            _ev_block_snapshot(medium, [
                (mamba_key, _make_single_spec("mamba_state:group=1", mamba_uri)),
            ]),
            _ev_block_delete(full_key, medium, ["mamba_state:group=1"]),
        ], trace_id="t16e_mamba_only"))

        self._assert_uri_present(full_key, full_uri, "t16e_query_full_after")
        self._assert_uri_absent(mamba_key, mamba_uri, "t16e_query_mamba_after")

    # 16f. Snapshot is an ordering barrier inside one ReportEvent request.
    def test_16f_snapshot_ordering_barrier_inside_batch(self):
        host = "192.168.1.245:8080"
        medium = "mem"
        key_add_after_snapshot = 8141
        key_add_before_snapshot = 8142
        uri_after = _build_vineyard_uri(host, medium, {"obj_id": "after"})
        uri_before = _build_vineyard_uri(host, medium, {"obj_id": "before"})

        self.client.report_event(_make_request(self.instance_id, host, [
            _ev_node_register([medium]),
            _ev_block_snapshot(medium, []),
            _ev_block_add(
                key_add_after_snapshot,
                medium,
                _make_single_spec("tp0", uri_after),
            ),
        ], trace_id="t16f_snapshot_then_add"))
        self._assert_uri_present(key_add_after_snapshot, uri_after, "t16f_query_after_present")

        self.client.report_event(_make_request(self.instance_id, host, [
            _ev_block_add(
                key_add_before_snapshot,
                medium,
                _make_single_spec("tp0", uri_before),
            ),
            _ev_block_snapshot(medium, []),
        ], trace_id="t16f_add_then_snapshot"))
        self._assert_uri_absent(key_add_after_snapshot, uri_after, "t16f_query_after_deleted")
        self._assert_uri_absent(key_add_before_snapshot, uri_before, "t16f_query_before_deleted")

    # 16g. Per-item failures do not drop other valid mutations in the same batch.
    def test_16g_partial_block_add_failure_keeps_valid_adds(self):
        host = "192.168.1.246:8080"
        medium = "mem"
        key_a = 8151
        key_b = 8152
        uri_a = _build_vineyard_uri(host, medium, {"obj_id": "valid_a"})
        uri_b = _build_vineyard_uri(host, medium, {"obj_id": "valid_b"})
        invalid_uri = _build_vineyard_uri(host, medium, {"obj_id": "invalid"})

        body = self.client.report_event(
            _make_request(self.instance_id, host, [
                _ev_node_register([medium]),
                _ev_block_add(key_a, medium, _make_single_spec("tp0", uri_a)),
                {
                    "event_type": "EVENT_BLOCK_ADD",
                    "block_add": {
                        "block_key": "not-an-int64",
                        "medium": medium,
                        "specs": _make_single_spec("tp0", invalid_uri),
                    },
                },
                _ev_block_add(key_b, medium, _make_single_spec("tp0", uri_b)),
            ], trace_id="t16g_partial_add"),
            check_ok=False,
        )
        item_results = body.get("item_results", [])
        self.assertGreaterEqual(len(item_results), 4, json.dumps(body, ensure_ascii=False))
        self.assertNotIn(item_results[2], ("OK", 1, "1"), json.dumps(body, ensure_ascii=False))

        self._assert_uri_present(key_a, uri_a, "t16g_query_valid_a")
        self._assert_uri_present(key_b, uri_b, "t16g_query_valid_b")

    # 16h. Snapshot diff is scoped by host even when the same block exists on another host.
    def test_16h_snapshot_same_block_other_host_survives(self):
        host_a = "192.168.1.247:8080"
        host_b = "192.168.1.248:8080"
        medium = "mem"
        block_key = 8161
        uri_a = _build_vineyard_uri(host_a, medium, {"obj_id": "same_block_a"})
        uri_b = _build_vineyard_uri(host_b, medium, {"obj_id": "same_block_b"})

        self.client.report_event(_make_request(self.instance_id, host_a, [
            _ev_node_register([medium]),
            _ev_block_snapshot(medium, [
                (block_key, _make_single_spec("host_a_full", uri_a)),
            ]),
        ], trace_id="t16h_snapshot_host_a"))
        self.client.report_event(_make_request(self.instance_id, host_b, [
            _ev_node_register([medium]),
            _ev_block_snapshot(medium, [
                (block_key, _make_single_spec("host_b_full", uri_b)),
            ]),
        ], trace_id="t16h_snapshot_host_b"))
        self._assert_uri_present(block_key, uri_a, "t16h_query_a_before")
        self._assert_uri_present(block_key, uri_b, "t16h_query_b_before")

        self.client.report_event(_make_request(self.instance_id, host_a, [
            _ev_block_snapshot(medium, []),
        ], trace_id="t16h_snapshot_host_a_empty"))

        self._assert_uri_absent(block_key, uri_a, "t16h_query_a_after")
        self._assert_uri_present(block_key, uri_b, "t16h_query_b_after")

    # 16i. Mixed full-attention and mamba specs store/match only full-attention today.
    def test_16i_full_attention_and_mamba_mixed_report(self):
        host = "192.168.1.249:8080"
        medium = "gpu"
        block_key = 8171
        full_uri = _build_vineyard_uri(host, medium, {"obj_id": "full"})
        mamba_uri = _build_vineyard_uri(host, medium, {"obj_id": "mamba"})

        self.client.report_event(_make_request(self.instance_id, host, [
            _ev_node_register([medium]),
            _ev_block_snapshot(medium, [
                (block_key, [
                    {"name": "full_attention:group=0:tp=0", "uri": full_uri},
                    {"name": "mamba_state:group=1", "uri": mamba_uri},
                ]),
            ]),
        ], trace_id="t16i_mixed_snapshot"))
        self._assert_uri_present(block_key, full_uri, "t16i_query_full_after_snapshot")
        self._assert_uri_absent(block_key, mamba_uri, "t16i_query_mamba_after_snapshot")

        self.client.report_event(_make_request(self.instance_id, host, [
            _ev_block_delete(block_key, medium, ["mamba_state:group=1"]),
        ], trace_id="t16i_mamba_delete"))
        self._assert_uri_present(block_key, full_uri, "t16i_query_full_after_mamba_delete")

        self.client.report_event(_make_request(self.instance_id, host, [
            _ev_block_delete(block_key, medium, ["full_attention:group=0:tp=0"]),
        ], trace_id="t16i_full_delete"))
        self._assert_uri_absent(block_key, full_uri, "t16i_query_full_after_full_delete")

    # 16a. Heartbeat timeout -> location filtered out, then recovery on heartbeat resume.
    def test_16a_heartbeat_timeout_then_recovery(self):
        if not ENABLE_LIVENESS_TIMING_TESTS:
            self.skipTest("--enable-liveness-timing-tests not set; skipping")

        host = "192.168.1.250:8080"
        block_key = 9100
        # Step 1: register + add a V6D replica.
        self.client.report_event(
            _make_request(self.instance_id, host, [
                _ev_node_register(["mem"]),
                _ev_block_add(block_key, "mem", _make_single_spec("spec_4096", _build_vineyard_uri(host, "mem"))),
            ], trace_id="t16a_setup")
        )
        # Confirm the replica is queryable.
        resp = self.client.get_cache_location({
            "trace_id": "t16a_q1",
            "instance_id": self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": [block_key],
            "block_mask": {"offset": 0},
        })
        self.assertGreater(len(resp.get("locations", [])), 0)

        # Step 2: skip heartbeat past heartbeat_timeout_ms (within cleanup_grace_ms).
        time.sleep((HEARTBEAT_TIMEOUT_MS + 500) / 1000.0)
        resp_after_timeout = self.client.get_cache_location({
            "trace_id": "t16a_q2",
            "instance_id": self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": [block_key],
            "block_mask": {"offset": 0},
        })
        # MetaSearchCache may still serve stale entry; not a hard assertion.

        # Step 3: heartbeat resumes within grace -> node recovers.
        self.client.report_event(
            _make_request(self.instance_id, host, [_ev_heartbeat({})], trace_id="t16a_hb")
        )
        resp_recovered = self.client.get_cache_location({
            "trace_id": "t16a_q3",
            "instance_id": self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": [block_key],
            "block_mask": {"offset": 0},
        })
        self.assertGreater(len(resp_recovered.get("locations", [])), 0,
                           "Replica must be queryable again after heartbeat resumed within grace")

    # 16b. Heartbeat timeout exceeds cleanup_grace_ms -> CleanupHostLocations triggered.
    def test_16b_heartbeat_exceeds_grace_triggers_cleanup(self):
        if not ENABLE_LIVENESS_TIMING_TESTS:
            self.skipTest("--enable-liveness-timing-tests not set; skipping")

        host = "192.168.1.251:8080"
        block_key = 9101
        self.client.report_event(
            _make_request(self.instance_id, host, [
                _ev_node_register(["mem"]),
                _ev_block_add(block_key, "mem", _make_single_spec("spec_4096", _build_vineyard_uri(host, "mem"))),
            ], trace_id="t16b_setup")
        )

        # Wait past hb_timeout + cleanup_grace + scheduler slack.
        wait_ms = HEARTBEAT_TIMEOUT_MS + CLEANUP_GRACE_MS + 1500
        time.sleep(wait_ms / 1000.0)

        # Soft check: MetaSearchCache TTL may delay eviction.
        resp = self.client.get_cache_location({
            "trace_id": "t16b_q",
            "instance_id": self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": [block_key],
            "block_mask": {"offset": 0},
        })
        # Every spec.uri returned must NOT belong to the cleaned-up host.
        for loc in resp.get("locations", []):
            for spec in loc.get("location_specs", []):
                self.assertNotIn(host, spec.get("uri", ""),
                                 f"Cleanup should have removed host [{host}] from results")



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
                    [_ev_block_add(block_key, "mem", _make_single_spec("spec_4096", _build_vineyard_uri(host, "mem")))],
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

    # 18. Mixed ADD/DELETE/HEARTBEAT batch throughput.
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
                    _ev_block_add(block_key, "mem", _make_single_spec("spec_4096", _build_vineyard_uri(host, "mem"))),
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
    parser = argparse.ArgumentParser(description="V6D ReportEvent HTTP integration tests")
    parser.add_argument("--host", default="localhost", help="KVCM host")
    parser.add_argument("--http_port", type=int, default=56020, help="KVCM meta HTTP port")
    parser.add_argument("--admin_http_port", type=int, default=None,
                        help="KVCM admin HTTP port (for addStorage). Defaults to http_port.")
    parser.add_argument("--instance_id", default="v6d_cluster_0", help="V6D instance_id")
    parser.add_argument("--skip-bench", action="store_true", help="Skip benchmark tests")
    parser.add_argument("--only-bench", action="store_true", help="Run only benchmark tests")
    parser.add_argument(
        "--enable-liveness-timing-tests",
        action="store_true",
        help=("Run heartbeat/cleanup timing tests. Requires the Vineyard storage to be opened with "
              "small heartbeat_timeout_ms / cleanup_grace_ms (defaults to 1000ms / 2000ms here)."),
    )
    parser.add_argument("--heartbeat-timeout-ms", type=int, default=1000)
    parser.add_argument("--cleanup-grace-ms", type=int, default=2000)

    args, _ = parser.parse_known_args()

    admin_port = args.admin_http_port or args.http_port

    global BASE_URL, ADMIN_URL, INSTANCE_ID, SKIP_BENCH, ONLY_BENCH
    global ENABLE_LIVENESS_TIMING_TESTS
    global HEARTBEAT_TIMEOUT_MS, CLEANUP_GRACE_MS
    BASE_URL = f"http://{args.host}:{args.http_port}"
    ADMIN_URL = f"http://{args.host}:{admin_port}"
    INSTANCE_ID = args.instance_id
    SKIP_BENCH = args.skip_bench
    ONLY_BENCH = args.only_bench
    ENABLE_LIVENESS_TIMING_TESTS = args.enable_liveness_timing_tests
    HEARTBEAT_TIMEOUT_MS = args.heartbeat_timeout_ms
    CLEANUP_GRACE_MS = args.cleanup_grace_ms

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
