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

import grpc
import json
import logging
import socket
import time
import unittest

from google.protobuf.json_format import MessageToDict, ParseDict

from kv_cache_manager.protocol.protobuf.meta_service_pb2 import (
    RegisterInstanceRequest,
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
        # Enable affinity so that the server honours strategy JSON and feeds
        # the FrequencySketch / write pipeline.
        self.start_worker(**{"kvcm.affinity.enabled": "true"})
        address = f"{self.envs[0].ip}:{self.envs[0].rpc_port}"
        self._channel = grpc.insecure_channel(address)
        self._stub = MetaServiceStub(self._channel)
        self._timeout = 10
        # The synchronous warm-up in StartMetricsPullLoop runs before the NFS
        # backend DoOpen completes, so the node table is empty at that point.
        # Wait for at least one async metrics-pull cycle (interval = 5 s) so
        # that the NFS backend's node_id (local IP) is present in the affinity
        # node table when the write test fires.
        time.sleep(6)

    def tearDown(self):
        self._channel.close()
        self.cleanup()

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

    def _register_instance(self, strategy_json=None, instance_id=INSTANCE_ID, specs=None):
        data = {
            "trace_id": TRACE_ID,
            "instance_group": "default",
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

    def _start_write(self, block_keys, caller_node_id=None, is_replication=False, instance_id=INSTANCE_ID):
        start_data = {
            "trace_id": TRACE_ID,
            "instance_id": instance_id,
            "block_keys": block_keys,
            "token_ids": [456] * len(block_keys),
            "write_timeout_seconds": 30,
            "is_replication": is_replication,
        }
        if caller_node_id:
            start_data["caller"] = {"node_id": caller_node_id}
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

    def _get_cache_location(self, caller_node_id, block_keys=None, instance_id=INSTANCE_ID):
        data = {
            "trace_id": TRACE_ID,
            "instance_id": instance_id,
            "query_type": "QT_PREFIX_MATCH",
            "block_keys": block_keys or [BLOCK_KEY],
            "caller": {"node_id": caller_node_id},
        }
        return self._call("GetCacheLocation", GetCacheLocationRequest, data)

    def _assert_miss(self, response):
        self.assertEqual(response["header"]["status"]["code"], "OK", response)
        self.assertFalse(any(loc.get("location_specs") for loc in response.get("locations", [])), response)
        self.assertFalse(response.get("hints"), response)

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
        """Strict write (is_replication=true) with on_miss=abort strategy and a
        caller whose node_id does NOT match any backend node should fail —
        the pipeline aborts (no preferred nodes), so the backend receives an
        empty preferred list under strict mode and returns EC_ERROR."""
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


if __name__ == "__main__":
    unittest.main()
