"""
Integration tests for cache affinity (write placement + read replication hint).

Test 1 (write affinity):
    StartWriteCache with caller.node_id = local IP → NFS backend's prefer_local
    pipeline matches the caller in the candidates list → returned LocationSpec
    carries node_id = local IP and URI contains preferred_node=<local IP>.

Test 2 (read replication hint):
    A *remote* caller (node_id != local IP) reads the same block repeatedly.
    Each remote read feeds the FrequencySketch; once the count reaches the
    configured replication_hot_threshold the server emits a ReplicationHint
    in the GetCacheLocation response.
"""

import concurrent.futures
import grpc
import os
import threading
import urllib.request
import json
import logging
import socket
import time
import unittest

from google.protobuf.json_format import MessageToDict, ParseDict

from kv_cache_manager.protocol.protobuf.meta_service_pb2 import (
    RegisterInstanceRequest,
    GetCacheMetaRequest,
    ReplicateCacheRequest,
    GetCacheLocationRequest,
    StartWriteCacheRequest,
    FinishWriteCacheRequest,
    RemoveCacheRequest,
)
from kv_cache_manager.protocol.protobuf.meta_service_pb2_grpc import MetaServiceStub
from testlib.test_base import TestBase

# The local IP that NFS backend registers as its node_id via NetUtil::GetLocalIp().
LOCAL_IP = socket.gethostbyname(socket.gethostname())

# A fake remote node id that is guaranteed to differ from LOCAL_IP so that reads
# from this caller are always "remote" (any_local = false in the strategy).
REMOTE_CALLER_NODE_ID = "remote_inference_node_99"

INSTANCE_ID = "affinity_integ_instance"
TRACE_ID = "affinity_integ_trace"
BLOCK_KEY = 200


def _make_strategy_json(replication_hot_threshold=2):
    """Build affinity strategy JSON for local_replica with write + read enabled."""
    return json.dumps({
        "type": "local_replica",
        "write": {
            "ops": {
                "prefer_local": {"on_miss": "passthrough"},
                "limit": 2,
            }
        },
        "read": {
            "on_miss": {
                "enabled": True,
                "replication_hot_threshold": replication_hot_threshold,
                "caller_capacity_threshold": 0.99,
                "caller_capacity_buffer": 0.01,
            }
        }
    })


class AffinityReplicationTest(TestBase, unittest.TestCase):
    """Real-server control-plane tests; physical data uses affinity_piggyback."""

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.clean_workdir()
        self.prepare_test_resource(1)
        self.addCleanup(self.cleanup)
        # Enable affinity so that the server honours strategy JSON and feeds
        # the FrequencySketch / write pipeline.
        self.start_worker(**{"kvcm.affinity.enabled": "true"})
        self._timeout = 10
        self._connect()
        # The synchronous warm-up in StartMetricsPullLoop runs before the NFS
        # backend DoOpen completes, so the node table is empty at that point.
        # Wait for at least one async metrics-pull cycle (interval = 5 s) so
        # that the NFS backend's node_id (local IP) is present in the affinity
        # node table when the write test fires.
        time.sleep(6)

    def _connect(self):
        address = f"{self.envs[0].ip}:{self.envs[0].rpc_port}"
        self._channel = grpc.insecure_channel(address)
        self.addCleanup(self._channel.close)
        grpc.channel_ready_future(self._channel).result(timeout=self._timeout)
        self._stub = MetaServiceStub(self._channel)

    # ------------------------------------------------------------------ helpers

    def _call(self, method_name, request_cls, data):
        """Issue a gRPC call and return the response as a dict."""
        request = ParseDict(data, request_cls())
        method = getattr(self._stub, method_name)
        response = method(request, timeout=self._timeout)
        return MessageToDict(
            response,
            including_default_value_fields=True,
            preserving_proto_field_name=True,
        )

    def _register_instance(self, strategy_json=None, instance_id=INSTANCE_ID, specs=None, instance_group="default"):
        data = {
            "trace_id": TRACE_ID,
            "instance_group": instance_group,
            "instance_id": instance_id,
            "block_size": 128,
            "model_deployment": {
                "model_name": "test_model",
                "dtype": "FP8",
                "use_mla": False,
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": 1,
            },
            "location_spec_infos": specs or [
                {"name": "tp0", "size": 1024},
            ],
            "affinity_strategy_json": strategy_json or _make_strategy_json(),
        }
        resp = self._call("RegisterInstance", RegisterInstanceRequest, data)
        self.assertEqual(resp["header"]["status"]["code"], "OK", resp)

    def _start_write(self, block_keys, caller_node_id=None, is_replication=False,
                     replication_target_node_id=None, instance_id=INSTANCE_ID, min_replica_count=1):
        start_data = {
            "trace_id": TRACE_ID,
            "instance_id": instance_id,
            "block_keys": block_keys,
            "token_ids": [456] * len(block_keys),
            "write_timeout_seconds": 30,
            "is_replication": is_replication,
            "min_replica_count": min_replica_count,
        }
        if caller_node_id:
            start_data["caller"] = {"node_id": caller_node_id}
        if replication_target_node_id:
            start_data["replication_target_node_id"] = replication_target_node_id
        resp = self._call("StartWriteCache", StartWriteCacheRequest, start_data)
        return resp

    def _finish_write(self, session_id, block_count, successes=None, instance_id=INSTANCE_ID):
        finish_data = {
            "trace_id": TRACE_ID,
            "instance_id": instance_id,
            "write_session_id": session_id,
            "success_blocks": {
                "bool_masks": {"values": successes if successes is not None else [True] * block_count},
            },
        }
        resp = self._call("FinishWriteCache", FinishWriteCacheRequest, finish_data)
        self.assertEqual(resp["header"]["status"]["code"], "OK", resp)

    def _write_block(self, block_keys=None, caller_node_id=None, instance_id=INSTANCE_ID):
        block_keys = block_keys or [BLOCK_KEY]
        resp = self._start_write(block_keys, caller_node_id, instance_id=instance_id)
        self.assertEqual(resp["header"]["status"]["code"], "OK", resp)
        session_id = resp["write_session_id"]
        self.assertTrue(session_id)
        self._finish_write(session_id, len(block_keys), instance_id=instance_id)
        return resp

    def _get_cache_location(self, caller_node_id, block_keys=None, instance_id=INSTANCE_ID, capabilities=1):
        data = {
            "trace_id": TRACE_ID,
            "instance_id": instance_id,
            "query_type": "QT_PREFIX_MATCH",
            "block_keys": block_keys or [BLOCK_KEY],
            "caller": {"node_id": caller_node_id, "replication_capabilities": capabilities},
        }
        return self._call("GetCacheLocation", GetCacheLocationRequest, data)

    def _assert_miss(self, response):
        self.assertEqual(response["header"]["status"]["code"], "OK", response)
        self.assertFalse(any(loc.get("location_specs") for loc in response.get("locations", [])), response)
        self.assertFalse(response.get("hints"), response)

    @staticmethod
    def _limited_strategy(**limits):
        strategy = json.loads(_make_strategy_json())
        strategy["replica_limits"] = limits
        return json.dumps(strategy)

    def _http_call(self, path, data, admin=False):
        port = self.envs[0].admin_http_port if admin else self.envs[0].http_port
        request = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (port, path),
            json.dumps(data).encode("utf8"), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            return json.load(response)

    def _replicas(self, key, instance_id=INSTANCE_ID):
        response = self._call("GetCacheMeta", GetCacheMetaRequest, {
            "trace_id": TRACE_ID, "instance_id": instance_id,
            "block_keys": [key], "detail_level": 2,
        })
        self.assertEqual("OK", response["header"]["status"]["code"], response)
        self.assertEqual(1, len(response["replica_locations"]), response)
        return response["replica_locations"][0].get("locations", [])

    def _wait_no_replicas(self, key, instance_id=INSTANCE_ID):
        deadline = time.monotonic() + 10
        while True:
            replicas = self._replicas(key, instance_id)
            if not replicas:
                return
            self.assertLess(time.monotonic(), deadline, replicas)
            time.sleep(0.05)

    def _assert_ok(self, response):
        self.assertEqual("OK", response["header"]["status"]["code"], response)

    # ------------------------------------------------------------------- tests

    def test_write_affinity_local_ip_in_location(self):
        """StartWriteCache with caller.node_id = LOCAL_IP should produce
        LocationSpecs whose node_id equals LOCAL_IP and whose URI carries
        the preferred_node=<LOCAL_IP> query parameter.

        Why LOCAL_IP: NFS backend registers itself with node_id = local IP.
        The prefer_local pipeline op only matches when the caller's node_id
        exists in the candidates list (i.e. the NFS-reported nodes). A fake
        node_id that doesn't appear in the list simply falls through via
        on_miss=passthrough, producing no preferred placement.
        """
        self._register_instance()

        resp = self._start_write([BLOCK_KEY], caller_node_id=LOCAL_IP)
        self.assertEqual(resp["header"]["status"]["code"], "OK", resp)
        session_id = resp["write_session_id"]
        self.assertTrue(session_id, "write_session_id should be non-empty")

        locations = resp.get("locations", [])
        self.assertGreater(len(locations), 0, "should return at least one location")

        spec = locations[0]["location_specs"][0]
        # NFS backend sets node_id = preferred_node_ids[0] = caller's node_id
        self.assertEqual(
            spec.get("node_id", ""), LOCAL_IP,
            f"spec.node_id should equal LOCAL_IP ({LOCAL_IP}), got: {spec}",
        )
        # NFS backend also appends preferred_node=<ip> to the URI
        uri = spec.get("uri", "")
        self.assertIn(
            f"preferred_node={LOCAL_IP}", uri,
            f"URI should contain preferred_node={LOCAL_IP}, got: {uri}",
        )
        logging.info("Write affinity OK: node_id=%s, uri=%s", spec["node_id"], uri)

        self._finish_write(session_id, 1)

    def test_remote_reads_trigger_replication_hint(self):
        """A remote caller reads the same block twice (threshold=2).
        The first read increments the sketch 0→1, no hint.
        The second read increments 1→2, reaching threshold → ReplicationHint.
        """
        self._register_instance()
        self._write_block()

        # Read #1 — sketch count 0→1, below threshold → no hint
        resp1 = self._get_cache_location(REMOTE_CALLER_NODE_ID)
        self.assertEqual(resp1["header"]["status"]["code"], "OK", resp1)
        locations = resp1.get("locations", [])
        self.assertGreater(len(locations), 0, "block should be found after write")
        hints1 = resp1.get("hints", [])
        self.assertEqual(
            len(hints1), 0,
            f"First read should NOT produce hints, got: {hints1}",
        )

        # Read #2 — sketch count 1→2, reaches threshold → hint emitted
        resp2 = self._get_cache_location(REMOTE_CALLER_NODE_ID)
        self.assertEqual(resp2["header"]["status"]["code"], "OK", resp2)
        hints2 = resp2.get("hints", [])
        self.assertGreater(
            len(hints2), 0,
            "Second read should trigger a ReplicationHint (threshold=2)",
        )

        hint = hints2[0]
        self.assertEqual(
            hint["block_key"], str(BLOCK_KEY),
            f"hint.block_key should be {BLOCK_KEY}",
        )
        self.assertEqual(
            hint["target_node_id"], REMOTE_CALLER_NODE_ID,
            "hint.target_node_id should be the remote caller",
        )
        self.assertTrue(
            hint.get("source_uri", ""),
            "hint.source_uri should be non-empty",
        )
        logging.info("ReplicationHint OK: %s", hint)

    def test_non_strict_write_remote_caller_succeeds(self):
        """Normal write (is_replication=false) with a caller whose node_id does
        NOT match the local NFS node should still succeed — the prefer_local
        pipeline misses but on_miss=passthrough lets it through."""
        self._register_instance()

        resp = self._start_write([BLOCK_KEY + 100], caller_node_id=REMOTE_CALLER_NODE_ID)
        self.assertEqual(resp["header"]["status"]["code"], "OK", resp)
        session_id = resp["write_session_id"]
        self.assertTrue(session_id, "write_session_id should be non-empty")

        locations = resp.get("locations", [])
        self.assertGreater(len(locations), 0, "should return at least one location")
        logging.info("Non-strict remote write OK: session_id=%s", session_id)

        self._finish_write(session_id, 1)

    def test_strict_write_remote_caller_fails(self):
        """A strict replication write to a node the backend cannot serve fails."""
        # Use on_miss=abort so that prefer_local aborts when caller is remote.
        abort_strategy = json.dumps({
            "type": "local_replica",
            "write": {
                "ops": {
                    "prefer_local": {"on_miss": "abort"},
                }
            },
            "read": {
                "on_miss": {
                    "enabled": True,
                    "replication_hot_threshold": 2,
                    "caller_capacity_threshold": 0.99,
                    "caller_capacity_buffer": 0.01,
                }
            }
        })
        self._register_instance(strategy_json=abort_strategy)

        resp = self._start_write(
            [BLOCK_KEY + 200],
            caller_node_id=REMOTE_CALLER_NODE_ID,
            is_replication=True,
            replication_target_node_id=REMOTE_CALLER_NODE_ID,
        )
        status_code = resp["header"]["status"]["code"]
        self.assertNotEqual(
            status_code, "OK",
            f"Strict write with non-matching caller should fail, but got OK: {resp}",
        )
        logging.info("Strict remote write correctly failed: status=%s", status_code)

    def test_unfinished_and_failed_writes_are_not_readable_then_retry(self):
        self._register_instance()
        started = self._start_write([BLOCK_KEY], LOCAL_IP)
        self.assertEqual(started["header"]["status"]["code"], "OK", started)
        self._assert_miss(self._get_cache_location(REMOTE_CALLER_NODE_ID))
        self._finish_write(started["write_session_id"], 1, [False])
        self._assert_miss(self._get_cache_location(REMOTE_CALLER_NODE_ID))
        self._write_block(caller_node_id=LOCAL_IP)
        visible = self._get_cache_location(LOCAL_IP)
        self.assertEqual(visible["header"]["status"]["code"], "OK", visible)
        self.assertTrue(visible.get("locations"), visible)

    def test_partial_batch_publication_stops_prefix_at_failed_block(self):
        self._register_instance()
        keys = [BLOCK_KEY, BLOCK_KEY + 1, BLOCK_KEY + 2]
        started = self._start_write(keys, LOCAL_IP)
        self.assertEqual(started["header"]["status"]["code"], "OK", started)
        self._finish_write(started["write_session_id"], 3, [True, False, True])
        response = self._get_cache_location(LOCAL_IP, keys)
        self.assertEqual(response["header"]["status"]["code"], "OK", response)
        hits = [loc for loc in response.get("locations", []) if loc.get("location_specs")]
        self.assertEqual(len(hits), 1, response)
        self._assert_miss(self._get_cache_location(LOCAL_IP, [keys[1]]))
        tail = self._get_cache_location(LOCAL_IP, [keys[2]])
        self.assertEqual(tail["header"]["status"]["code"], "OK", tail)
        self.assertTrue(tail.get("locations"), tail)

    def test_local_reads_do_not_generate_replication_hints(self):
        self._register_instance()
        self._write_block(caller_node_id=LOCAL_IP)
        for _ in range(8):
            response = self._get_cache_location(LOCAL_IP)
            self.assertEqual(response["header"]["status"]["code"], "OK", response)
            self.assertTrue(response.get("locations"), response)
            self.assertFalse(response.get("hints"), response)
            for location in response["locations"]:
                for spec in location["location_specs"]:
                    self.assertEqual(spec["node_id"], LOCAL_IP, response)

    def test_hint_suppression_does_not_block_another_key(self):
        self._register_instance()
        for key in [BLOCK_KEY, BLOCK_KEY + 1]:
            self._write_block([key], LOCAL_IP)
            first = self._get_cache_location(REMOTE_CALLER_NODE_ID, [key])
            self.assertEqual(first["header"]["status"]["code"], "OK", first)
            self.assertFalse(first.get("hints"), first)
            second = self._get_cache_location(REMOTE_CALLER_NODE_ID, [key])
            self.assertEqual(second["header"]["status"]["code"], "OK", second)
            self.assertEqual(len(second.get("hints", [])), 1, second)
            self.assertEqual(second["hints"][0]["block_key"], str(key), second)
            for _ in range(3):
                suppressed = self._get_cache_location(REMOTE_CALLER_NODE_ID, [key])
                self.assertEqual(suppressed["header"]["status"]["code"], "OK", suppressed)
                self.assertFalse(suppressed.get("hints"), suppressed)

    def test_instance_isolation_for_same_block_key(self):
        self._register_instance()
        self._write_block(caller_node_id=LOCAL_IP)
        other = INSTANCE_ID + "_isolated"
        self._register_instance(instance_id=other)
        for _ in range(4):
            self._assert_miss(self._get_cache_location(REMOTE_CALLER_NODE_ID, instance_id=other))
        original = self._get_cache_location(LOCAL_IP)
        self.assertEqual(original["header"]["status"]["code"], "OK", original)
        self.assertTrue(original.get("locations"), original)

    def test_legacy_caller_can_read_without_replication_hints(self):
        self._register_instance()
        self._write_block(caller_node_id=LOCAL_IP)
        for _ in range(4):
            response = self._get_cache_location("")
            self.assertEqual(response["header"]["status"]["code"], "OK", response)
            self.assertTrue(response.get("locations"), response)
            self.assertFalse(response.get("hints"), response)

    def test_noop_strategy_preserves_read_write_without_hints(self):
        self._register_instance(strategy_json=json.dumps({"type": "noop"}))
        self._write_block(caller_node_id=LOCAL_IP)
        for _ in range(4):
            response = self._get_cache_location(REMOTE_CALLER_NODE_ID)
            self.assertEqual(response["header"]["status"]["code"], "OK", response)
            self.assertTrue(response.get("locations"), response)
            self.assertFalse(response.get("hints"), response)

    def test_remove_then_rewrite_restores_query_visibility(self):
        self._register_instance()
        self._write_block(caller_node_id=LOCAL_IP)
        removed = self._call("RemoveCache", RemoveCacheRequest, {
            "trace_id": TRACE_ID, "instance_id": INSTANCE_ID, "block_keys": [BLOCK_KEY],
        })
        self.assertEqual(removed["header"]["status"]["code"], "OK", removed)
        deadline = time.monotonic() + 10
        while True:
            response = self._get_cache_location(LOCAL_IP)
            if not any(loc.get("location_specs") for loc in response.get("locations", [])):
                self._assert_miss(response)
                break
            self.assertLess(time.monotonic(), deadline, response)
            time.sleep(0.05)
        self._write_block(caller_node_id=LOCAL_IP)
        response = self._get_cache_location(LOCAL_IP)
        self.assertEqual(response["header"]["status"]["code"], "OK", response)
        self.assertTrue(response.get("locations"), response)

    def test_multi_spec_publication_returns_all_components(self):
        self._register_instance(specs=[{"name": "tp0", "size": 1024}, {"name": "tp1", "size": 2048}])
        self._write_block(caller_node_id=LOCAL_IP)
        response = self._get_cache_location(LOCAL_IP)
        self.assertEqual(response["header"]["status"]["code"], "OK", response)
        specs = [spec for loc in response.get("locations", []) for spec in loc["location_specs"]]
        self.assertEqual({spec["name"] for spec in specs}, {"tp0", "tp1"}, response)
        self.assertTrue(all(spec["node_id"] == LOCAL_IP for spec in specs), response)
        self.assertFalse(response.get("hints"), response)

        # The remote hint must preserve every spec name/URI through protobuf.
        self.assertFalse(self._get_cache_location(REMOTE_CALLER_NODE_ID).get("hints"))
        remote = self._get_cache_location(REMOTE_CALLER_NODE_ID)
        self.assertEqual(len(remote.get("hints", [])), 1, remote)
        hint = remote["hints"][0]
        self.assertEqual(hint["target_node_id"], REMOTE_CALLER_NODE_ID)
        self.assertFalse(hint.get("source_uri"), hint)
        self.assertEqual(
            {item["spec_name"]: item["uri"] for item in hint["source_specs"]},
            {item["name"]: item["uri"] for loc in remote["locations"] for item in loc["location_specs"]},
        )

    def test_remote_frequency_is_independent_between_populated_instances(self):
        instances = [INSTANCE_ID, INSTANCE_ID + "_second"]
        for instance in instances:
            self._register_instance(instance_id=instance)
            self._write_block(caller_node_id=LOCAL_IP, instance_id=instance)
            first = self._get_cache_location(REMOTE_CALLER_NODE_ID, instance_id=instance)
            self.assertEqual(first["header"]["status"]["code"], "OK", first)
            self.assertFalse(first.get("hints"), first)
        for instance in instances:
            second = self._get_cache_location(REMOTE_CALLER_NODE_ID, instance_id=instance)
            self.assertEqual(len(second.get("hints", [])), 1, second)

    def test_hint_suppression_is_independent_between_populated_instances(self):
        for instance in [INSTANCE_ID, INSTANCE_ID + "_second"]:
            self._register_instance(strategy_json=_make_strategy_json(1), instance_id=instance)
            self._write_block(caller_node_id=LOCAL_IP, instance_id=instance)
            first = self._get_cache_location(REMOTE_CALLER_NODE_ID, instance_id=instance)
            self.assertEqual(len(first.get("hints", [])), 1, first)
            repeated = self._get_cache_location(REMOTE_CALLER_NODE_ID, instance_id=instance)
            self.assertEqual(repeated["header"]["status"]["code"], "OK", repeated)
            self.assertFalse(repeated.get("hints"), repeated)

    def test_concurrent_writing_reservations_obey_instance_budget_and_release_on_abort(self):
        self._register_instance(self._limited_strategy(max_instance_bytes=2048))
        barrier = threading.Barrier(8)

        def reserve(key):
            barrier.wait(timeout=10)
            return key, self._start_write([key], LOCAL_IP)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(reserve, range(3000, 3008)))
        accepted = [(key, response) for key, response in results
                    if response["header"]["status"]["code"] == "OK"]
        self.assertEqual(2, len(accepted), results)
        for key, response in results:
            if response["header"]["status"]["code"] != "OK":
                self.assertFalse(response.get("write_session_id"), response)
                self._wait_no_replicas(key)
            self._assert_miss(self._get_cache_location(LOCAL_IP, [key]))
        first_key, first = accepted[0]
        second_key, second = accepted[1]
        self._finish_write(first["write_session_id"], 1, [False])
        self._wait_no_replicas(first_key)
        self._finish_write(second["write_session_id"], 1)
        self._write_block([3010], LOCAL_IP)
        rejected = self._start_write([3011], LOCAL_IP)
        self.assertNotEqual("OK", rejected["header"]["status"]["code"], rejected)
        self.assertEqual("CLS_SERVING", self._replicas(second_key)[0]["status"])

    def test_multi_spec_budget_counts_every_component_and_failed_batch_leaves_no_reservations(self):
        self._register_instance(self._limited_strategy(max_instance_bytes=3072),
                                specs=[{"name": "tp0", "size": 1024}, {"name": "tp1", "size": 2048}])
        rejected = self._start_write([3100, 3101], LOCAL_IP)
        self.assertNotEqual("OK", rejected["header"]["status"]["code"], rejected)
        for key in (3100, 3101):
            self._wait_no_replicas(key)
        self._write_block([3100], LOCAL_IP)
        rejected = self._start_write([3101], LOCAL_IP)
        self.assertNotEqual("OK", rejected["header"]["status"]["code"], rejected)
        replica = self._replicas(3100)[0]
        self.assertEqual({"tp0", "tp1"}, {spec["name"] for spec in replica["location_specs"]})
        self.assertEqual("CLS_SERVING", replica["status"])

    def test_per_key_budget_rollback_preserves_existing_replica_and_frees_new_batch_key(self):
        self._register_instance(self._limited_strategy(max_replicas_per_key=1, max_instance_bytes=4096))
        self._write_block([3200], LOCAL_IP)
        original = self._replicas(3200)
        rejected = self._start_write([3200, 3201], LOCAL_IP, min_replica_count=2)
        self.assertNotEqual("OK", rejected["header"]["status"]["code"], rejected)
        self.assertEqual(original, self._replicas(3200))
        self._wait_no_replicas(3201)
        self._write_block([3201], LOCAL_IP)
        self.assertEqual(1, len(self._replicas(3201)))

    def test_capability_upgrade_does_not_inherit_legacy_heat_or_suppression(self):
        self._register_instance(specs=[{"name": "tp0", "size": 1024}, {"name": "tp1", "size": 2048}])
        self._write_block(caller_node_id=LOCAL_IP)
        for capabilities in (0, 8):
            for _ in range(4):
                response = self._get_cache_location(REMOTE_CALLER_NODE_ID, capabilities=capabilities)
                self._assert_ok(response)
                self.assertTrue(response["locations"])
                self.assertFalse(response["hints"])
                self.assertEqual(0, response["replication_capabilities"])
        first = self._get_cache_location(REMOTE_CALLER_NODE_ID, capabilities=1)
        self.assertFalse(first["hints"], first)
        second = self._get_cache_location(REMOTE_CALLER_NODE_ID, capabilities=1)
        self.assertEqual(1, second["replication_capabilities"])
        self.assertEqual(1, len(second["hints"]), second)
        self.assertEqual({"tp0", "tp1"}, {source["spec_name"] for source in second["hints"][0]["source_specs"]})
        self.assertFalse(self._get_cache_location(REMOTE_CALLER_NODE_ID)["hints"])

    def test_legacy_single_spec_still_receives_compatible_source_uri(self):
        self._register_instance()
        self._write_block(caller_node_id=LOCAL_IP)
        self.assertFalse(self._get_cache_location(REMOTE_CALLER_NODE_ID, capabilities=0)["hints"])
        response = self._get_cache_location(REMOTE_CALLER_NODE_ID, capabilities=0)
        self.assertEqual(1, len(response["hints"]), response)
        self.assertTrue(response["hints"][0]["source_uri"])
        self.assertEqual(0, response["replication_capabilities"])

    def test_explicit_target_is_independent_of_caller_and_missing_target_is_rejected(self):
        self._register_instance()
        missing = self._start_write([3300], LOCAL_IP, is_replication=True)
        self.assertEqual("INVALID_ARGUMENT", missing["header"]["status"]["code"], missing)
        self._wait_no_replicas(3300)
        response = self._start_write([3300], REMOTE_CALLER_NODE_ID, is_replication=True,
                                     replication_target_node_id=LOCAL_IP)
        self._assert_ok(response)
        for spec in response["locations"][0]["location_specs"]:
            self.assertEqual(LOCAL_IP, spec["node_id"])
        self._finish_write(response["write_session_id"], 1)
        duplicate = self._start_write([3300], REMOTE_CALLER_NODE_ID, is_replication=True,
                                      replication_target_node_id=LOCAL_IP)
        self._assert_ok(duplicate)
        self.assertFalse(duplicate["locations"], duplicate)
        self._finish_write(duplicate["write_session_id"], 0)
        self.assertEqual(1, len(self._replicas(3300)))

    def test_server_copy_batch_keeps_item_results_and_cleans_unsupported_copies(self):
        for transport in ("grpc", "http"):
            with self.subTest(transport=transport):
                instance = INSTANCE_ID + "_" + transport
                self._register_instance(self._limited_strategy(max_instance_bytes=2048), instance_id=instance)
                source = self._write_block([3400], LOCAL_IP, instance_id=instance)
                sources = [{"spec_name": spec["name"], "uri": spec["uri"]}
                           for spec in source["locations"][0]["location_specs"]]
                request = {"trace_id": TRACE_ID, "instance_id": instance, "items": [
                    {"block_key": 3400, "source_specs": sources, "target_node_id": LOCAL_IP},
                    {"block_key": 3401, "source_specs": sources, "target_node_id": ""},
                    {"block_key": 3402, "source_specs": sources, "target_node_id": LOCAL_IP},
                ]}
                response = (self._call("ReplicateCache", ReplicateCacheRequest, request) if transport == "grpc"
                            else self._http_call("/api/replicateCache", request))
                self._assert_ok(response)
                self.assertEqual(["OK", "INVALID_ARGUMENT", "UNSUPPORTED"],
                                 [result["code"] for result in response["results"]], response)
                self.assertEqual([True, False, False],
                                 [result.get("already_exists", False) for result in response["results"]], response)
                self.assertEqual(1, len(self._replicas(3400, instance)))
                for key in (3401, 3402):
                    self._wait_no_replicas(key, instance)
                # The one failed allocation must release the remaining 1024-byte budget.
                self._write_block([3402], LOCAL_IP, instance_id=instance)

    def test_malformed_named_sources_release_session_and_allow_exact_budget_retry(self):
        self._register_instance(self._limited_strategy(max_instance_bytes=1024, max_replicas_per_key=1))
        request = {"trace_id": TRACE_ID, "instance_id": INSTANCE_ID, "block_key": 3500,
                   "target_node_id": LOCAL_IP, "source_specs": [
                       {"spec_name": "tp0", "uri": "file://nfs_01/source?size=1024"},
                       {"spec_name": "tp0", "uri": "file://nfs_01/other?size=1024"},
                   ]}
        response = self._call("ReplicateCache", ReplicateCacheRequest, request)
        self.assertEqual("INVALID_ARGUMENT", response["header"]["status"]["code"], response)
        self._wait_no_replicas(3500)
        self._write_block([3500], LOCAL_IP)
        self.assertEqual("CLS_SERVING", self._replicas(3500)[0]["status"])

    def test_get_meta_preserves_individual_replica_ids_and_publication_states(self):
        self._register_instance(self._limited_strategy(max_replicas_per_key=2))
        first = self._write_block([3600], LOCAL_IP)
        second = self._start_write([3600], LOCAL_IP, min_replica_count=2)
        self._assert_ok(second)
        replicas = self._replicas(3600)
        self.assertEqual(2, len(replicas), replicas)
        self.assertEqual({"CLS_SERVING", "CLS_WRITING"}, {r["status"] for r in replicas})
        self.assertEqual(2, len({r["id"] for r in replicas}))
        self.assertTrue(all(r["id"] and int(r["create_time"]) > 0 for r in replicas))
        self._finish_write(second["write_session_id"], 1)
        published = self._replicas(3600)
        self.assertEqual({r["id"] for r in replicas}, {r["id"] for r in published})
        self.assertTrue(all(r["status"] == "CLS_SERVING" for r in published))
        self.assertEqual({first["locations"][0]["location_specs"][0]["uri"],
                          second["locations"][0]["location_specs"][0]["uri"]},
                         {r["location_specs"][0]["uri"] for r in published})

    def test_instance_budget_and_node_ownership_survive_controlled_restart(self):
        storage = {"global_unique_name": "affinity_persist_nfs",
                   "nfs": {"root_path": os.path.join(self.workdir, "nfs") + "/"}}
        group = {
            "name": "affinity_persist_group", "storage_candidates": ["affinity_persist_nfs"],
            "max_instance_count": 8, "global_quota_group_name": "quota_group_test",
            "quota": {"capacity": 1024 * 1024, "quota_config": []}, "version": 1,
            "cache_config": {
                "reclaim_strategy": {"storage_unique_name": "affinity_persist_nfs", "reclaim_policy": 1,
                                     "trigger_strategy": {"used_percentage": 3.2}, "delay_before_delete_ms": 0},
                "data_storage_strategy": 2,
                "meta_indexer_config": {
                    "max_key_count": 1024, "mutex_shard_num": 16,
                    "meta_storage_backend_config": {"storage_type": "dummy",
                        "storage_uri": "file://" + os.path.join(self.workdir, "persistent_meta")},
                    "persist_metadata_interval_time_ms": 0,
                },
            },
        }

        def register():
            self._assert_ok(self._http_call("/api/addStorage", {"trace_id": TRACE_ID, "storage": storage}, admin=True))
            self._assert_ok(self._http_call("/api/createInstanceGroup",
                {"trace_id": TRACE_ID, "instance_group": group}, admin=True))
            self._register_instance(self._limited_strategy(max_instance_bytes=1024), instance_group=group["name"])

        register()
        self._write_block([3700], LOCAL_IP)
        original = self._replicas(3700)
        self._channel.close()
        self.worker_manager.stop_worker(0)
        self.assertTrue(self.worker_manager.start_worker(0))
        self._connect()
        register()
        self.assertEqual(original, self._replicas(3700))
        self.assertTrue(all(spec["node_id"] == LOCAL_IP for spec in original[0]["location_specs"]))
        rejected = self._start_write([3701], LOCAL_IP)
        self.assertNotEqual("OK", rejected["header"]["status"]["code"], rejected)
        self._wait_no_replicas(3701)
        self._assert_ok(self._call("RemoveCache", RemoveCacheRequest, {
            "trace_id": TRACE_ID, "instance_id": INSTANCE_ID, "block_keys": [3700]}))
        self._wait_no_replicas(3700)
        self._write_block([3701], LOCAL_IP)



if __name__ == "__main__":
    unittest.main()
