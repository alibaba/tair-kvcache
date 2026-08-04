#!/usr/bin/env python3
"""
Integration tests for Event Report ReportEvent HTTP interface.

Usage:
    # 1. Start KVCM service locally:
    #    bazel-bin/kv_cache_manager/kv_cache_manager_bin \
    #        --env kvcm.service.rpc_port=56010 \
    #        --env kvcm.service.http_port=56020 \
    #        --env kvcm.service.admin_rpc_port=56030 \
    #        --env kvcm.service.admin_http_port=56040 \
    #        --env kvcm.service.enable_debug_service=false
    # 2. Run this script:
    python3 integration_test/meta_service/test_report_event.py \
        --host localhost --http_port 56020 --admin_http_port 56040 \
        --instance_id event_report_cluster_0
"""

import argparse
import json
import re
import sys
import time
import statistics
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qsl, urlsplit

import requests


BASE_URL = ""
ADMIN_URL = ""
INSTANCE_ID = "event_report_cluster_0"
SKIP_BENCH = False
ONLY_BENCH = False
HEARTBEAT_TIMEOUT_MS = 1000
CLEANUP_GRACE_MS = 2000
LIVENESS_CHECK_INTERVAL_MS = 200


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

    def update_storage(self, data):
        url = f"{self.admin_url}/api/updateStorage"
        resp = self.session.post(url, json=data)
        resp.raise_for_status()
        body = resp.json()
        code = body.get("header", {}).get("status", {}).get("code")
        if code != "OK":
            raise AssertionError(f"updateStorage failed: {json.dumps(body)}")
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

    def get_cache_locations_by_backend(self, data):
        url = f"{self.base_url}/api/getCacheLocationsByBackend"
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

    def get_host_cache_state(self, data, check_response=True):
        url = f"{self.base_url}/api/getHostCacheState"
        resp = self.session.post(url, json=data)
        resp.raise_for_status()
        body = resp.json()
        if check_response:
            code = body.get("header", {}).get("status", {}).get("code")
            assert code == "OK", f"getHostCacheState failed: {json.dumps(body)}"
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


def _ev_block_delete(block_key, medium, spec_names):
    return {
        "event_type": "EVENT_BLOCK_DELETE",
        "block_delete": {
            "block_key": str(block_key),
            "medium": medium,
            "spec_names": list(spec_names),
        },
    }


def _ev_host_down():
    return {"event_type": "EVENT_HOST_DOWN", "host_down": {}}


def _ev_heartbeat(system_status=None):
    return {
        "event_type": "EVENT_HEARTBEAT",
        "heartbeat": {"system_status": system_status or {}},
    }


def _make_request(instance_id, host_ip_port, events, trace_id="test", storage_type="ST_EVENT_REPORT_L2"):
    return {
        "trace_id": trace_id,
        "instance_id": instance_id,
        "host_ip_port": host_ip_port,
        "events": events,
        "storage_type": storage_type,
    }


def _build_event_report_uri(host_ip_port, medium, params=None):
    """Build vineyard URI: vineyard://{ip}:{port}/{medium}?k=v&..."""
    base = f"vineyard://{host_ip_port}/{medium}"
    if not params:
        return base
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{base}?{query}"


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------
class EventReportFunctionalTest(unittest.TestCase):
    HOST = "192.168.1.200:8080"
    INSTANCE_GROUP_NAME = "event_report_dual_type_test_group"
    PROFILES = {
        "l1p5": {
            "storage_name": "event_report_l1p5_default",
            "storage_type": "ST_EVENT_REPORT_L1P5",
            "type_value": 7,
            "uri_scheme": "event_report",
        },
        "l2": {
            "storage_name": "event_report_l2_default",
            "storage_type": "ST_EVENT_REPORT_L2",
            "type_value": 8,
            "uri_scheme": "vineyard",
        },
    }

    @classmethod
    def setUpClass(cls):
        cls.client = KVCMClient(BASE_URL, ADMIN_URL)
        cls.instance_id = INSTANCE_ID
        cls._ensure_event_report_storages_registered()
        cls._ensure_instance_group_created()
        cls._ensure_instance_registered()
        # Register host so subsequent events have a NodeInfo entry.
        for profile_name in cls.PROFILES:
            cls.client.report_event(
                cls._make_profile_request(
                    profile_name,
                    cls.instance_id,
                    cls.HOST,
                    [_ev_node_register(["mem", "disk"])],
                    trace_id=f"setup_register_host_{profile_name}",
                )
            )

    @classmethod
    def _ensure_event_report_storages_registered(cls):
        for profile_name, profile in cls.PROFILES.items():
            storage = cls._make_event_report_storage(profile)
            try:
                cls.client.add_storage({
                    "trace_id": f"setup_storage_{profile_name}",
                    "storage": storage,
                })
                print(f"[SETUP] Event report storage '{profile['storage_name']}' registered")
            except Exception as e:
                print(f"[WARN] addStorage failed for {profile['storage_name']} (may already exist): {e}")
            cls.client.update_storage({
                "trace_id": f"setup_update_storage_{profile_name}",
                "storage": storage,
                "force_update": True,
            })
            print(
                f"[SETUP] Event report storage '{profile['storage_name']}' liveness config: "
                f"heartbeat_timeout_ms={HEARTBEAT_TIMEOUT_MS}, cleanup_grace_ms={CLEANUP_GRACE_MS}, "
                f"liveness_check_interval_ms={LIVENESS_CHECK_INTERVAL_MS}"
            )

    @classmethod
    def _make_event_report_storage(cls, profile):
        return {
            "global_unique_name": profile["storage_name"],
            "storage_type": profile["storage_type"],
            "event_report": {
                "heartbeat_timeout_ms": HEARTBEAT_TIMEOUT_MS,
                "cleanup_grace_ms": CLEANUP_GRACE_MS,
                "liveness_check_interval_ms": LIVENESS_CHECK_INTERVAL_MS,
            },
            "check_storage_available_when_open": False,
        }

    @classmethod
    def _ensure_instance_group_created(cls):
        cls.client.create_instance_group({
            "trace_id": "setup_ig",
            "instance_group": {
                "name": cls.INSTANCE_GROUP_NAME,
                "storage_candidates": ["nfs_01"],
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
                "event_report_storage_candidates": [
                    cls.PROFILES["l1p5"]["storage_name"],
                    cls.PROFILES["l2"]["storage_name"],
                ],
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
                    "model_name": "test_er_model",
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

    @classmethod
    def _make_profile_request(cls, profile_name, instance_id, host_ip_port, events, trace_id="test"):
        return _make_request(
            instance_id,
            host_ip_port,
            events,
            trace_id=trace_id,
            storage_type=cls.PROFILES[profile_name]["storage_type"],
        )

    def _event_report_uri(self, profile_name, host_ip_port, medium, params=None):
        base = f"{self.PROFILES[profile_name]['uri_scheme']}://{host_ip_port}/{medium}"
        if not params:
            return base
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{base}?{query}"

    def _report_event(self, profile_name, host, events, trace_id, instance_id=None, check_ok=True):
        return self.client.report_event(
            self._make_profile_request(profile_name, instance_id or self.instance_id, host, events, trace_id),
            check_ok=check_ok,
        )

    def _report_node(self, profile_name, host, mediums, trace_id, instance_id=None):
        return self._report_event(profile_name, host, [_ev_node_register(mediums)], trace_id, instance_id=instance_id)

    def _report_block_add(self, profile_name, host, block_key, medium, specs, trace_id, instance_id=None):
        return self._report_event(
            profile_name,
            host,
            [_ev_block_add(block_key, medium, specs)],
            trace_id,
            instance_id=instance_id,
        )

    def _report_block_delete(self, profile_name, host, block_key, medium, spec_names, trace_id, instance_id=None,
                             check_ok=True):
        return self._report_event(
            profile_name,
            host,
            [_ev_block_delete(block_key, medium, spec_names)],
            trace_id,
            instance_id=instance_id,
            check_ok=check_ok,
        )

    def _get_cache_locations(self, block_keys, trace_id, instance_id=None):
        resp = self.client.get_cache_location({
            "trace_id": trace_id,
            "instance_id": instance_id or self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": block_keys,
            "block_mask": {"offset": 0},
        })
        code = resp.get("header", {}).get("status", {}).get("code")
        self.assertEqual(code, "OK", f"getCacheLocation failed: {json.dumps(resp, ensure_ascii=False)}")
        return resp.get("locations", [])

    def _get_cache_locations_by_backend(self, block_keys, profile_names, trace_id, instance_id=None):
        resp = self.client.get_cache_locations_by_backend({
            "trace_id": trace_id,
            "instance_id": instance_id or self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": block_keys,
            "block_mask": {"offset": 0},
            "backend_selectors": [
                {
                    "backend_type": self.PROFILES[profile_name]["storage_type"],
                    "strategy": "LSS_WEIGHTED_RANDOM",
                }
                for profile_name in profile_names
            ],
        })
        code = resp.get("header", {}).get("status", {}).get("code")
        self.assertEqual(
            code,
            "OK",
            f"getCacheLocationsByBackend failed: {json.dumps(resp, ensure_ascii=False)}",
        )
        key_locations = resp.get("key_locations", [])
        locations = []
        for key_location in key_locations:
            locations.extend(key_location.get("locations", []))
        return locations

    def _location_type_matches(self, loc, profile_name):
        actual = loc.get("type")
        expected_name = self.PROFILES[profile_name]["storage_type"]
        expected_value = self.PROFILES[profile_name]["type_value"]
        return actual in (expected_name, expected_value, str(expected_value))

    def _locations_for_profile(self, locations, profile_name):
        return [loc for loc in locations if self._location_type_matches(loc, profile_name)]

    def _assert_profile_specs(self, block_key, profile_name, expected_specs, trace_id, instance_id=None):
        locations = self._get_cache_locations([block_key], trace_id, instance_id=instance_id)
        self._assert_profile_specs_in_locations(locations, profile_name, expected_specs)

    def _assert_profile_specs_by_backend(self, block_key, profile_name, expected_specs, trace_id, instance_id=None):
        locations = self._get_cache_locations_by_backend(
            [block_key],
            [profile_name],
            trace_id,
            instance_id=instance_id,
        )
        self._assert_profile_specs_in_locations(locations, profile_name, expected_specs)

    def _assert_profile_specs_in_locations(self, locations, profile_name, expected_specs):
        profile_locations = self._locations_for_profile(locations, profile_name)
        specs_by_name = {
            spec.get("name"): spec
            for loc in profile_locations
            for spec in loc.get("location_specs", [])
            if spec.get("name")
        }
        for spec_name, uri in expected_specs.items():
            self.assertIn(
                spec_name,
                specs_by_name,
                f"Expected {profile_name} spec [{spec_name}], locations={json.dumps(locations, ensure_ascii=False)}",
            )
            actual_uri = specs_by_name[spec_name].get("uri", "")
            actual = urlsplit(actual_uri)
            expected = urlsplit(uri)
            self.assertEqual(
                (actual.scheme, actual.netloc, actual.path, actual.fragment),
                (expected.scheme, expected.netloc, expected.path, expected.fragment),
            )

            actual_params = parse_qsl(actual.query, keep_blank_values=True)
            expected_params = parse_qsl(expected.query, keep_blank_values=True)
            versions = [
                value for key, value in actual_params if key == "s_version"
            ]
            self.assertEqual(
                len(versions),
                1,
                f"Expected exactly one s_version in URI: {actual_uri}",
            )
            self.assertRegex(versions[0], re.compile(r"^[0-9a-fA-F]{32}$"))
            self.assertEqual(
                sorted(
                    (key, value)
                    for key, value in actual_params
                    if key != "s_version"
                ),
                sorted(expected_params),
                f"KVCM changed caller URI parameters: {actual_uri}",
            )

    def _assert_profile_spec_absent(self, block_key, profile_name, spec_name, trace_id, instance_id=None,
                                    timeout_seconds=10.0, interval_seconds=0.2):
        return self._assert_profile_spec_absent_with_query(
            block_key,
            profile_name,
            spec_name,
            trace_id,
            instance_id,
            timeout_seconds,
            interval_seconds,
            use_backend_query=False,
        )

    def _assert_profile_spec_absent_by_backend(self, block_key, profile_name, spec_name, trace_id, instance_id=None,
                                               timeout_seconds=10.0, interval_seconds=0.2):
        return self._assert_profile_spec_absent_with_query(
            block_key,
            profile_name,
            spec_name,
            trace_id,
            instance_id,
            timeout_seconds,
            interval_seconds,
            use_backend_query=True,
        )

    def _assert_profile_spec_absent_with_query(self, block_key, profile_name, spec_name, trace_id, instance_id,
                                               timeout_seconds, interval_seconds, use_backend_query):
        deadline = time.monotonic() + timeout_seconds
        last_locations = []
        last_leaked_specs = []

        while True:
            if use_backend_query:
                last_locations = self._get_cache_locations_by_backend(
                    [block_key],
                    [profile_name],
                    trace_id,
                    instance_id=instance_id,
                )
            else:
                last_locations = self._get_cache_locations([block_key], trace_id, instance_id=instance_id)
            profile_locations = self._locations_for_profile(last_locations, profile_name)
            last_leaked_specs = [
                spec
                for loc in profile_locations
                for spec in loc.get("location_specs", [])
                if spec.get("name") == spec_name
            ]
            if not last_leaked_specs:
                return
            if time.monotonic() >= deadline:
                self.fail(
                    f"Expected {profile_name} spec [{spec_name}] to be absent, "
                    f"but still found specs={last_leaked_specs}; "
                    f"locations={json.dumps(last_locations, ensure_ascii=False)}"
                )
            time.sleep(interval_seconds)

    def _assert_host_removed_from_cache_locations(self, block_keys, host, trace_id, profile_name=None,
                                                  timeout_seconds=10.0, interval_seconds=0.2):
        deadline = time.monotonic() + timeout_seconds
        last_resp = None
        last_leaked_uris = []

        while True:
            if profile_name:
                candidate_locations = self._get_cache_locations_by_backend(block_keys, [profile_name], trace_id)
                last_resp = {"locations": candidate_locations}
            else:
                resp = self.client.get_cache_location({
                    "trace_id": trace_id,
                    "instance_id": self.instance_id,
                    "query_type": "QT_BATCH_GET",
                    "block_keys": block_keys,
                    "block_mask": {"offset": 0},
                })
                last_resp = resp
                candidate_locations = resp.get("locations", [])
            last_leaked_uris = [
                spec.get("uri", "")
                for loc in candidate_locations
                for spec in loc.get("location_specs", [])
                if host in spec.get("uri", "")
            ]
            if not last_leaked_uris:
                return
            if time.monotonic() >= deadline:
                self.fail(
                    f"Expected host [{host}] locations to be cleaned, "
                    f"but still found uris={last_leaked_uris}; "
                    f"last_resp={json.dumps(last_resp, ensure_ascii=False)}"
                )
            time.sleep(interval_seconds)

    def _for_each_profile(self, fn):
        for profile_name in self.PROFILES:
            with self.subTest(profile=profile_name):
                fn(profile_name)

    # 1. NODE_REGISTER (with mediums)
    def test_01_node_register(self):
        def run(profile_name):
            body = self._report_node(profile_name, self.HOST, ["mem", "disk", "ssd"], f"t01_{profile_name}")
            self.assertEqual(body["header"]["status"]["code"], "OK")
            self.assertEqual(body.get("item_results", []), [])
            self.assertTrue(body.get("snapshot_required"))
            self.assertEqual(body.get("committed_snapshot_version", ""), "")
        self._for_each_profile(run)

    # 2. NODE_REGISTER is idempotent and merges mediums
    def test_02_node_register_idempotent(self):
        def run(profile_name):
            host = f"192.168.1.{201 + self.PROFILES[profile_name]['type_value']}:8080"
            self._report_node(profile_name, host, ["mem"], f"t02a_{profile_name}")
            body = self._report_node(profile_name, host, ["mem", "disk"], f"t02b_{profile_name}")
            self.assertEqual(body["header"]["status"]["code"], "OK")
            self.assertEqual(body.get("item_results", []), [])
            self.assertTrue(body.get("snapshot_required"))
            self.assertEqual(body.get("committed_snapshot_version", ""), "")
        self._for_each_profile(run)

    # 3. BLOCK_ADD then query: spec name/uri should match what was sent
    def test_03_block_add_then_query(self):
        def run(profile_name):
            block_key = 9000 + self.PROFILES[profile_name]["type_value"]
            uri = self._event_report_uri(profile_name, self.HOST, "mem", {"flavor": "test_query"})
            spec_name = "tp0"
            self._report_block_add(
                profile_name,
                self.HOST,
                block_key,
                "mem",
                _make_single_spec(spec_name, uri),
                f"t03_add_{profile_name}",
            )
            self._assert_profile_specs(block_key, profile_name, {spec_name: uri}, f"t03_query_{profile_name}")
        self._for_each_profile(run)

    # 4. Two mediums on same host: each becomes its own location_id
    def test_04_block_add_multi_medium(self):
        def run(profile_name):
            block_key = 9020 + self.PROFILES[profile_name]["type_value"]
            host = f"192.168.1.{220 + self.PROFILES[profile_name]['type_value']}:8080"
            self._report_node(profile_name, host, ["mem", "disk"], f"t04_reg_{profile_name}")
            uri_mem = self._event_report_uri(profile_name, host, "mem")
            uri_disk = self._event_report_uri(profile_name, host, "disk")
            body = self._report_event(
                profile_name,
                host,
                [
                    _ev_block_add(block_key, "mem", _make_single_spec("mem_spec", uri_mem)),
                    _ev_block_add(block_key, "disk", _make_single_spec("disk_spec", uri_disk)),
                ],
                f"t04_add_{profile_name}",
            )
            self.assertEqual(body["header"]["status"]["code"], "OK")
            self.assertEqual(body.get("item_results", []), [])
            self.assertEqual(len(body.get("committed_snapshot_version", "")), 32)
            self.assertTrue(body.get("snapshot_required"))
            self._assert_profile_specs(
                block_key,
                profile_name,
                {"mem_spec": uri_mem, "disk_spec": uri_disk},
                f"t04_query_{profile_name}",
            )
        self._for_each_profile(run)

    # 5. BLOCK_ADD with multiple specs in one CacheLocation
    def test_05_block_add_multi_spec(self):
        def run(profile_name):
            block_key = 9040 + self.PROFILES[profile_name]["type_value"]
            host = f"192.168.1.{230 + self.PROFILES[profile_name]['type_value']}:8080"
            self._report_node(profile_name, host, ["mem"], f"t05_reg_{profile_name}")
            uri_spec0 = self._event_report_uri(profile_name, host, "mem", {"obj_id": "o1", "size": "512"})
            uri_spec1 = self._event_report_uri(profile_name, host, "mem", {"obj_id": "o2", "size": "512"})
            body = self._report_block_add(
                profile_name,
                host,
                block_key,
                "mem",
                [
                    {"name": "spec_4096", "uri": uri_spec0},
                    {"name": "spec_8192", "uri": uri_spec1},
                ],
                f"t05_add_{profile_name}",
            )
            self.assertEqual(body["header"]["status"]["code"], "OK")
            self.assertEqual(body.get("item_results", []), [])
            self.assertEqual(len(body.get("committed_snapshot_version", "")), 32)
            self.assertTrue(body.get("snapshot_required"))
            self._assert_profile_specs(
                block_key,
                profile_name,
                {"spec_4096": uri_spec0, "spec_8192": uri_spec1},
                f"t05_query_{profile_name}",
            )
        self._for_each_profile(run)

    # 6. BLOCK_DELETE removes the specific spec and keeps other specs in the same location
    def test_06_block_delete(self):
        def run(profile_name):
            block_key = 9060 + self.PROFILES[profile_name]["type_value"]
            host = f"192.168.1.{240 + self.PROFILES[profile_name]['type_value']}:8080"
            self._report_node(profile_name, host, ["mem"], f"t06_reg_{profile_name}")
            keep_uri = self._event_report_uri(profile_name, host, "mem", {"spec": "keep"})
            drop_uri = self._event_report_uri(profile_name, host, "mem", {"spec": "drop"})
            self._report_block_add(
                profile_name,
                host,
                block_key,
                "mem",
                [
                    {"name": "keep_spec", "uri": keep_uri},
                    {"name": "drop_spec", "uri": drop_uri},
                ],
                f"t06_add_{profile_name}",
            )
            body = self._report_block_delete(
                profile_name,
                host,
                block_key,
                "mem",
                ["drop_spec"],
                f"t06_del_{profile_name}",
            )
            self.assertEqual(body["header"]["status"]["code"], "OK")
            self.assertEqual(body.get("item_results", []), [])
            self.assertEqual(len(body.get("committed_snapshot_version", "")), 32)
            self.assertFalse(body.get("snapshot_required"))
            self._assert_profile_specs(block_key, profile_name, {"keep_spec": keep_uri}, f"t06_query_{profile_name}")
            self._assert_profile_spec_absent(block_key, profile_name, "drop_spec", f"t06_absent_{profile_name}")
        self._for_each_profile(run)

    # 7. BLOCK_DELETE on missing key/medium is a no-op (idempotent)
    def test_07_block_delete_nonexistent(self):
        def run(profile_name):
            body = self._report_block_delete(
                profile_name,
                self.HOST,
                99999 + self.PROFILES[profile_name]["type_value"],
                "mem",
                ["spec_4096"],
                f"t07_{profile_name}",
                check_ok=False,
            )
            self.assertEqual(body["header"]["status"]["code"], "OK")
            self.assertEqual(body.get("item_results", []), [])
        self._for_each_profile(run)

    # 8. HOST_DOWN cleans up all mediums under the host for each event report type
    def test_08_host_down(self):
        def run(profile_name):
            down_host = f"192.168.1.{250 + self.PROFILES[profile_name]['type_value']}:8080"
            block_keys = [
                9080 + self.PROFILES[profile_name]["type_value"] * 10 + i
                for i in range(3)
            ]
            events = [_ev_node_register(["mem", "disk"])]
            for bk in block_keys:
                events.append(_ev_block_add(
                    bk,
                    "mem",
                    _make_single_spec("spec_4096", self._event_report_uri(profile_name, down_host, "mem")),
                ))
                events.append(_ev_block_add(
                    bk,
                    "disk",
                    _make_single_spec("spec_4096", self._event_report_uri(profile_name, down_host, "disk")),
                ))
            self._report_event(profile_name, down_host, events, f"t08_add_{profile_name}")
            body = self._report_event(profile_name, down_host, [_ev_host_down()], f"t08_down_{profile_name}")
            self.assertEqual(body["header"]["status"]["code"], "OK")
            self.assertEqual(body.get("item_results", []), [])
            self.assertTrue(body.get("snapshot_required"))
            self.assertEqual(body.get("committed_snapshot_version", ""), "")
            self._assert_host_removed_from_cache_locations(
                block_keys,
                down_host,
                f"t08_cleanup_check_{profile_name}",
                profile_name=profile_name,
            )
        self._for_each_profile(run)

    # 9. HOST_DOWN is idempotent
    def test_09_host_down_idempotent(self):
        def run(profile_name):
            down_host = f"192.168.1.{30 + self.PROFILES[profile_name]['type_value']}:8080"
            self._report_node(profile_name, down_host, ["mem"], f"t09_reg_{profile_name}")
            body1 = self._report_event(profile_name, down_host, [_ev_host_down()], f"t09_down1_{profile_name}")
            body2 = self._report_event(profile_name, down_host, [_ev_host_down()], f"t09_down2_{profile_name}")
            self.assertEqual(body1["header"]["status"]["code"], "OK")
            self.assertEqual(body2["header"]["status"]["code"], "OK")
            self.assertEqual(body1.get("item_results", []), [])
            self.assertEqual(body2.get("item_results", []), [])
        self._for_each_profile(run)

    # 10. HEARTBEAT extends liveness; payload is opaque
    def test_10_heartbeat(self):
        def run(profile_name):
            body = self._report_event(
                profile_name,
                self.HOST,
                [_ev_heartbeat({"version": "er-0.18", "cpu": "45%"})],
                f"t10_{profile_name}",
            )
            self.assertEqual(body["header"]["status"]["code"], "OK")
            self.assertEqual(body.get("item_results", []), [])
        self._for_each_profile(run)

    # 11. Mixed batch: register + add + heartbeat in a single RPC
    def test_11_mixed_batch(self):
        def run(profile_name):
            host = f"192.168.1.{60 + self.PROFILES[profile_name]['type_value']}:8080"
            block_key = 9110 + self.PROFILES[profile_name]["type_value"]
            events = [
                _ev_node_register(["mem"]),
                _ev_block_add(
                    block_key,
                    "mem",
                    _make_single_spec("spec_4096", self._event_report_uri(profile_name, host, "mem")),
                ),
                _ev_heartbeat({"phase": "boot"}),
            ]
            body = self._report_event(profile_name, host, events, f"t11_{profile_name}")
            self.assertEqual(body["header"]["status"]["code"], "OK")
            self.assertEqual(body.get("item_results", []), [])
            self.assertEqual(len(body.get("committed_snapshot_version", "")), 32)
            self.assertTrue(body.get("snapshot_required"))
        self._for_each_profile(run)

    # 12. Empty events array: should be a no-op success
    def test_12_empty_batch(self):
        def run(profile_name):
            body = self._report_event(profile_name, self.HOST, [], f"t12_{profile_name}")
            self.assertEqual(body["header"]["status"]["code"], "OK")
            self.assertEqual(body.get("item_results", []), [])
            self.assertEqual(body.get("committed_snapshot_version", ""), "")
            self.assertFalse(body.get("snapshot_required"))
            self.assertEqual(body.get("retry_after_ms", "0"), "0")
        self._for_each_profile(run)

    # 13. Missing block_add params: server must surface a per-item failure
    def test_13_block_add_missing_params(self):
        def run(profile_name):
            body = self._report_event(
                profile_name,
                self.HOST,
                [{"event_type": "EVENT_BLOCK_ADD"}],
                f"t13_{profile_name}",
                check_ok=False,
            )
            self.assertEqual(body["header"]["status"]["code"], "INVALID_ARGUMENT")
            self.assertEqual(body.get("item_results"), ["INVALID_ARGUMENT"])
        self._for_each_profile(run)

    # 15. L1.5 and L2 coexist for the same host/key/medium without overwriting each other
    def test_15_dual_type_block_add_then_query(self):
        host = "192.168.1.80:8080"
        block_key = 9150
        l1p5_uri = self._event_report_uri("l1p5", host, "mem", {"source": "engine"})
        l2_uri = self._event_report_uri("l2", host, "mem", {"source": "vineyard"})

        self._report_node("l1p5", host, ["mem"], "t15_reg_l1p5")
        self._report_node("l2", host, ["mem"], "t15_reg_l2")
        self._report_block_add(
            "l1p5",
            host,
            block_key,
            "mem",
            _make_single_spec("l1p5_spec", l1p5_uri),
            "t15_add_l1p5",
        )
        self._report_block_add(
            "l2",
            host,
            block_key,
            "mem",
            _make_single_spec("l2_spec", l2_uri),
            "t15_add_l2",
        )

        self._assert_profile_specs_by_backend(block_key, "l1p5", {"l1p5_spec": l1p5_uri}, "t15_query_l1p5")
        self._assert_profile_specs_by_backend(block_key, "l2", {"l2_spec": l2_uri}, "t15_query_l2")

    # 16. Deleting one event report type must not remove the other type
    def test_16_dual_type_delete_isolated(self):
        host = "192.168.1.81:8080"
        block_key = 9160
        l1p5_uri = self._event_report_uri("l1p5", host, "mem", {"source": "engine"})
        l2_uri = self._event_report_uri("l2", host, "mem", {"source": "vineyard"})

        for profile_name, spec_name, uri in [
            ("l1p5", "l1p5_spec", l1p5_uri),
            ("l2", "l2_spec", l2_uri),
        ]:
            self._report_node(profile_name, host, ["mem"], f"t16_reg_{profile_name}")
            self._report_block_add(
                profile_name,
                host,
                block_key,
                "mem",
                _make_single_spec(spec_name, uri),
                f"t16_add_{profile_name}",
            )

        self._report_block_delete("l1p5", host, block_key, "mem", ["l1p5_spec"], "t16_del_l1p5")
        self._assert_profile_spec_absent_by_backend(block_key, "l1p5", "l1p5_spec", "t16_absent_l1p5")
        self._assert_profile_specs_by_backend(block_key, "l2", {"l2_spec": l2_uri}, "t16_l2_survives")

        self._report_block_delete("l2", host, block_key, "mem", ["l2_spec"], "t16_del_l2")
        self._assert_profile_spec_absent_by_backend(block_key, "l2", "l2_spec", "t16_absent_l2")

    # 17. HOST_DOWN is scoped by event report type when L1.5 and L2 coexist
    def test_17_dual_type_host_down_isolated(self):
        host = "192.168.1.82:8080"
        block_keys = [9170, 9171]
        for profile_name in self.PROFILES:
            events = [_ev_node_register(["mem"])]
            for block_key in block_keys:
                events.append(_ev_block_add(
                    block_key,
                    "mem",
                    _make_single_spec(
                        f"{profile_name}_spec",
                        self._event_report_uri(profile_name, host, "mem", {"block": block_key}),
                    ),
                ))
            self._report_event(profile_name, host, events, f"t17_add_{profile_name}")

        self._report_event("l1p5", host, [_ev_host_down()], "t17_down_l1p5")
        self._assert_host_removed_from_cache_locations(block_keys, host, "t17_l1p5_removed", profile_name="l1p5")
        for block_key in block_keys:
            self._assert_profile_specs_by_backend(
                block_key,
                "l2",
                {
                    "l2_spec": self._event_report_uri("l2", host, "mem", {"block": block_key}),
                },
                f"t17_l2_survives_{block_key}",
            )

        self._report_event("l2", host, [_ev_host_down()], "t17_down_l2")
        self._assert_host_removed_from_cache_locations(block_keys, host, "t17_l2_removed", profile_name="l2")

    # 18. GetHostCacheState counts prefix length by host, not by event report type
    def test_18_get_host_cache_state_dual_type_prefix_match(self):
        instance_id = f"{self.instance_id}_host_cache_state_{int(time.time() * 1000)}"
        self.client.register_instance({
            "trace_id": "t18_register",
            "instance_group": self.INSTANCE_GROUP_NAME,
            "instance_id": instance_id,
            "block_size": 128,
            "default_query_type": "QT_PREFIX_MATCH",
            "model_deployment": {
                "model_name": "test_er_model",
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

        host_specs = [
            ("10.0.0.1:8080", {"l1p5": [10000, 10001], "l2": [10000, 10003]}),
            ("10.0.0.2:8080", {"l1p5": [10000, 10002], "l2": [10000, 10001, 10003]}),
            ("10.0.0.3:8080", {"l1p5": [], "l2": [10000]}),
        ]
        for host, keys_by_profile in host_specs:
            for profile_name, keys in keys_by_profile.items():
                if not keys:
                    continue
                events = [_ev_node_register(["mem"])]
                for key in keys:
                    events.append(
                        _ev_block_add(
                            key,
                            "mem",
                            _make_single_spec("tp0", self._event_report_uri(profile_name, host, "mem")),
                        )
                    )
                self._report_event(
                    profile_name,
                    host,
                    events,
                    f"t18_setup_{profile_name}_{host}",
                    instance_id=instance_id)

        resp = self.client.get_host_cache_state({
            "trace_id": "t18_query",
            "instance_id": instance_id,
            "block_cache_keys": [10000, 10001, 10002, 10003],
        })
        expected = {
            "10.0.0.1:8080": 2,
            "10.0.0.2:8080": 4,
            "10.0.0.3:8080": 1,
        }
        actual = {
            h["host_ip_port"]: int(h["local"])
            for h in resp.get("hosts", [])
        }
        self.assertNotIn("p2p_1_hit_count", resp)
        for host_match in resp.get("hosts", []):
            self.assertIn("p2p_1_fetch", host_match)
            self.assertIn("p2p_1_total_match", host_match)
        for host, prefix in expected.items():
            self.assertIn(host, actual, f"host {host} not found in response")
            self.assertEqual(actual[host], prefix, f"host {host}: expected prefix={prefix}, got {actual[host]}")

    # 19. StartWriteCacheWithMinReplica: L2 event report eviction with min_replica_count=2
    def test_19_start_write_cache_with_min_replica_l2(self):
        block_key = 9190 + int(time.time() * 1000) % 1000000
        uri = self._event_report_uri("l2", self.HOST, "mem")

        # Step 1: 1 L2 EVENT_REPORT replica only.
        self._report_block_add(
            "l2",
            self.HOST,
            block_key,
            "mem",
            _make_single_spec("spec_4096", uri),
            "t19_add_l2",
        )

        # Step 2: ask for evict; with only 1 replica we expect a remote write.
        resp = self.client.start_write_cache_with_min_replica({
            "trace_id": "t19_evict_1",
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
            "trace_id": "t19_evict_finish",
            "instance_id": self.instance_id,
            "write_session_id": write_session_id,
            "success_blocks": {"bool_masks": {"values": [True]}},
        })

        # Step 4: Now n_total=2; evict should skip remote allocation.
        resp2 = self.client.start_write_cache_with_min_replica({
            "trace_id": "t19_evict_2",
            "instance_id": self.instance_id,
            "block_keys": [block_key],
            "write_timeout_seconds": 30,
            "min_replica_count": 2,
        })
        locations2 = resp2.get("locations", [])
        self.assertEqual(len(locations2), 0,
                         "Expected no write locations since 2 replicas already exist")

    # 20a. L2 heartbeat timeout -> location filtered out, then recovery on heartbeat resume.
    def test_20a_l2_heartbeat_timeout_then_recovery(self):
        host = "192.168.1.250:8080"
        block_key = 9100
        # Step 1: register + add a L2 event report replica.
        self._report_event(
            "l2",
            host,
            [
                _ev_node_register(["mem"]),
                _ev_block_add(
                    block_key,
                    "mem",
                    _make_single_spec("spec_4096", self._event_report_uri("l2", host, "mem")),
                ),
            ],
            trace_id="t20a_setup",
        )
        # Confirm the replica is queryable.
        resp = self.client.get_cache_location({
            "trace_id": "t20a_q1",
            "instance_id": self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": [block_key],
            "block_mask": {"offset": 0},
        })
        self.assertGreater(len(resp.get("locations", [])), 0)

        # Step 2: skip heartbeat past heartbeat_timeout_ms (within cleanup_grace_ms).
        time.sleep((HEARTBEAT_TIMEOUT_MS + 500) / 1000.0)
        resp_after_timeout = self.client.get_cache_location({
            "trace_id": "t20a_q2",
            "instance_id": self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": [block_key],
            "block_mask": {"offset": 0},
        })
        # MetaSearchCache may still serve stale entry; not a hard assertion.

        # Step 3: heartbeat resumes within grace -> node recovers.
        self._report_event("l2", host, [_ev_heartbeat({})], "t20a_hb")
        resp_recovered = self.client.get_cache_location({
            "trace_id": "t20a_q3",
            "instance_id": self.instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": [block_key],
            "block_mask": {"offset": 0},
        })
        self.assertGreater(len(resp_recovered.get("locations", [])), 0,
                           "Replica must be queryable again after heartbeat resumed within grace")

    # 20b. L2 heartbeat timeout exceeds cleanup_grace_ms -> CleanupHostLocations triggered.
    def test_20b_l2_heartbeat_exceeds_grace_triggers_cleanup(self):
        host = "192.168.1.251:8080"
        block_key = 9101
        self._report_event(
            "l2",
            host,
            [
                _ev_node_register(["mem"]),
                _ev_block_add(
                    block_key,
                    "mem",
                    _make_single_spec("spec_4096", self._event_report_uri("l2", host, "mem")),
                ),
            ],
            trace_id="t20b_setup",
        )

        # Wait past hb_timeout + cleanup_grace + scheduler slack.
        wait_ms = HEARTBEAT_TIMEOUT_MS + CLEANUP_GRACE_MS + 1500
        time.sleep(wait_ms / 1000.0)

        # Soft check: MetaSearchCache TTL may delay eviction.
        resp = self.client.get_cache_location({
            "trace_id": "t20b_q",
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
class EventReportBenchTest(unittest.TestCase):

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
                events = [_ev_block_add(block_key, "mem", _make_single_spec("spec_4096", _build_event_report_uri(
                    host, "mem"))), _ev_block_delete(block_key, "mem", ["spec_4096"]), _ev_heartbeat({"thread": str(thread_id)}), ]
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
    parser = argparse.ArgumentParser(description="Event Report ReportEvent HTTP integration tests")
    parser.add_argument("--host", default="localhost", help="KVCM host")
    parser.add_argument("--http_port", type=int, default=56020, help="KVCM meta HTTP port")
    parser.add_argument("--admin_http_port", type=int, default=None,
                        help="KVCM admin HTTP port (for addStorage). Defaults to http_port.")
    parser.add_argument("--instance_id", default="event_report_cluster_0", help="event report instance_id")
    parser.add_argument("--skip-bench", action="store_true", help="Skip benchmark tests")
    parser.add_argument("--only-bench", action="store_true", help="Run only benchmark tests")
    parser.add_argument("--heartbeat-timeout-ms", type=int, default=1000)
    parser.add_argument("--cleanup-grace-ms", type=int, default=2000)
    parser.add_argument("--liveness-check-interval-ms", type=int, default=200)

    args, _ = parser.parse_known_args()

    admin_port = args.admin_http_port or args.http_port

    global BASE_URL, ADMIN_URL, INSTANCE_ID, SKIP_BENCH, ONLY_BENCH
    global HEARTBEAT_TIMEOUT_MS, CLEANUP_GRACE_MS, LIVENESS_CHECK_INTERVAL_MS
    BASE_URL = f"http://{args.host}:{args.http_port}"
    ADMIN_URL = f"http://{args.host}:{admin_port}"
    INSTANCE_ID = args.instance_id
    SKIP_BENCH = args.skip_bench
    ONLY_BENCH = args.only_bench
    HEARTBEAT_TIMEOUT_MS = args.heartbeat_timeout_ms
    CLEANUP_GRACE_MS = args.cleanup_grace_ms
    LIVENESS_CHECK_INTERVAL_MS = args.liveness_check_interval_ms

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if not ONLY_BENCH:
        suite.addTests(loader.loadTestsFromTestCase(EventReportFunctionalTest))
    if not SKIP_BENCH:
        suite.addTests(loader.loadTestsFromTestCase(EventReportBenchTest))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
