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
    """End-to-end tests for write-affinity placement and read replication hints."""

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

    def _register_instance(self, strategy_json=None):
        data = {
            "trace_id": TRACE_ID,
            "instance_group": "default",
            "instance_id": INSTANCE_ID,
            "block_size": 128,
            "model_deployment": {
                "model_name": "test_model",
                "dtype": "FP8",
                "use_mla": False,
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": 1,
            },
            "location_spec_infos": [
                {"name": "tp0", "size": 1024},
            ],
            "affinity_strategy_json": strategy_json or _make_strategy_json(),
        }
        resp = self._call("RegisterInstance", RegisterInstanceRequest, data)
        self.assertEqual(resp["header"]["status"]["code"], "OK", resp)

    def _start_write(self, block_keys, caller_node_id=None, is_replication=False):
        start_data = {
            "trace_id": TRACE_ID,
            "instance_id": INSTANCE_ID,
            "block_keys": block_keys,
            "token_ids": [456] * len(block_keys),
            "write_timeout_seconds": 30,
            "is_replication": is_replication,
        }
        if caller_node_id:
            start_data["caller"] = {"node_id": caller_node_id}
        resp = self._call("StartWriteCache", StartWriteCacheRequest, start_data)
        return resp

    def _finish_write(self, session_id, block_count):
        finish_data = {
            "trace_id": TRACE_ID,
            "instance_id": INSTANCE_ID,
            "write_session_id": session_id,
            "success_blocks": {
                "bool_masks": {"values": [True] * block_count},
            },
        }
        resp = self._call("FinishWriteCache", FinishWriteCacheRequest, finish_data)
        self.assertEqual(resp["header"]["status"]["code"], "OK", resp)

    def _write_block(self, block_keys=None, caller_node_id=None):
        block_keys = block_keys or [BLOCK_KEY]
        resp = self._start_write(block_keys, caller_node_id)
        self.assertEqual(resp["header"]["status"]["code"], "OK", resp)
        session_id = resp["write_session_id"]
        self.assertTrue(session_id)
        self._finish_write(session_id, len(block_keys))
        return resp

    def _get_cache_location(self, caller_node_id, block_keys=None):
        data = {
            "trace_id": TRACE_ID,
            "instance_id": INSTANCE_ID,
            "query_type": "QT_PREFIX_MATCH",
            "block_keys": block_keys or [BLOCK_KEY],
            "caller": {"node_id": caller_node_id},
        }
        return self._call("GetCacheLocation", GetCacheLocationRequest, data)

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


if __name__ == "__main__":
    unittest.main()
