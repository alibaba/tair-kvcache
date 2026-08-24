import os
import time
import unittest
from array import array
from unittest.mock import patch

from kv_cache_manager.py_connector.dashtrace._proto import (
    kvcm_optimizer_ingest_minimal_pb2,
)
from kv_cache_manager.py_connector.dashtrace.kvcm_shadow_forwarder import (
    KVCMShadowForwarder,
    KVCMShadowForwarderConfig,
)
from kv_cache_manager.py_connector.dashtrace.observed_request import ObservedRequest


def _observation(sequence: int, tokens: list[int]) -> ObservedRequest:
    return ObservedRequest(
        sequence=sequence,
        timestamp_ns=sequence,
        trace_id=f"trace-{sequence}",
        instance_id="instance-a",
        token_ids=array("q", tokens),
    )


class _GrpcChannel:
    def __init__(self):
        self.paths = []
        self.requests = []
        self.closed = False

    def unary_unary(self, path, **_):
        self.paths.append(path)

        def call(request, timeout):
            del timeout
            self.requests.append(request)
            return kvcm_optimizer_ingest_minimal_pb2.TraceObservationBatchResponse(
                header=kvcm_optimizer_ingest_minimal_pb2.CommonResponseHeader(
                    status=kvcm_optimizer_ingest_minimal_pb2.Status(
                        code=kvcm_optimizer_ingest_minimal_pb2.OK
                    )
                ),
                accepted_count=len(request.observations),
                last_accepted_sequence=request.observations[-1].sequence,
            )

        return call

    def close(self):
        self.closed = True


class KVCMShadowForwarderTest(unittest.TestCase):
    def _forwarder(self, channel: _GrpcChannel, **overrides):
        values = dict(
            enabled=True,
            grpc_target="127.0.0.1:6381",
            instance_id="instance-a",
            batch_size=8,
            batch_wait_ms=10,
            max_batch_tokens=100,
            max_owned_tokens=100,
            producer_id="producer-a",
        )
        values.update(overrides)
        return KVCMShadowForwarder(
            KVCMShadowForwarderConfig(**values),
            grpc_channel_factory=lambda *_args, **_kwargs: channel,
        )

    def test_disabled_forwarder_is_a_noop(self):
        forwarder = KVCMShadowForwarder(KVCMShadowForwarderConfig())
        self.assertFalse(forwarder.submit_observation(_observation(1, [1])))
        forwarder.close()

    def test_rejects_request_larger_than_batch_limit(self):
        channel = _GrpcChannel()
        forwarder = self._forwarder(channel, max_batch_tokens=2)
        self.assertFalse(forwarder.submit_observation(_observation(1, [1, 2, 3])))
        forwarder.close()
        self.assertEqual(channel.requests, [])

    def test_owned_token_limit_includes_worker_owned_requests(self):
        channel = _GrpcChannel()
        forwarder = self._forwarder(
            channel, batch_size=8, batch_wait_ms=100, max_owned_tokens=3
        )
        self.assertTrue(forwarder.submit_observation(_observation(1, [1, 2])))
        time.sleep(0.02)
        self.assertFalse(forwarder.submit_observation(_observation(2, [3, 4])))
        forwarder.close()

    def test_sends_ordered_batch_and_validates_ack(self):
        channel = _GrpcChannel()
        forwarder = self._forwarder(channel)
        self.assertTrue(forwarder.submit_observation(_observation(1, [1, 2])))
        self.assertTrue(forwarder.submit_observation(_observation(2, [3])))
        deadline = time.monotonic() + 1
        while not channel.requests and time.monotonic() < deadline:
            time.sleep(0.01)
        forwarder.close()

        self.assertEqual(
            channel.paths,
            [
                "/kv_cache_manager.proto.optimizer.OptimizerEventStreamService/ReportTraceBatch"
            ],
        )
        self.assertEqual(len(channel.requests), 1)
        request = channel.requests[0]
        self.assertEqual(request.producer_id, "producer-a")
        self.assertEqual([item.sequence for item in request.observations], [1, 2])
        self.assertEqual(list(request.observations[0].token_ids), [1, 2])
        self.assertTrue(channel.closed)

    def test_batch_size_one_still_uses_formal_batch_rpc(self):
        channel = _GrpcChannel()
        forwarder = self._forwarder(channel, batch_size=1)
        self.assertTrue(forwarder.submit_observation(_observation(1, [1])))
        deadline = time.monotonic() + 1
        while not channel.requests and time.monotonic() < deadline:
            time.sleep(0.01)
        forwarder.close()
        self.assertEqual(len(channel.requests[0].observations), 1)

    def test_token_limit_splits_batches_without_reordering(self):
        channel = _GrpcChannel()
        forwarder = self._forwarder(channel, max_batch_tokens=3)
        for sequence in range(1, 4):
            self.assertTrue(
                forwarder.submit_observation(_observation(sequence, [1, 2]))
            )
        deadline = time.monotonic() + 1
        while len(channel.requests) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        forwarder.close()
        self.assertEqual(
            [item.sequence for request in channel.requests for item in request.observations],
            [1, 2, 3],
        )

    def test_enabled_config_requires_direct_target(self):
        with self.assertRaisesRegex(ValueError, "DASHTRACE_KVCM_GRPC_TARGET"):
            KVCMShadowForwarderConfig(enabled=True, instance_id="instance-a").validate()

    def test_from_env_reads_only_formal_transport_keys(self):
        env = {
            "DASHTRACE_ONLINE_MRC_ENABLED": "true",
            "DASHTRACE_KVCM_GRPC_TARGET": "kvcm:6381",
            "DASHTRACE_INSTANCE_ID": "instance-a",
            "DASHTRACE_ONLINE_MRC_BATCH_SIZE": "16",
        }
        with patch.dict(os.environ, env, clear=True):
            config = KVCMShadowForwarderConfig.from_env()
        self.assertTrue(config.enabled)
        self.assertEqual(config.grpc_target, "kvcm:6381")
        self.assertEqual(config.instance_id, "instance-a")
        self.assertEqual(config.batch_size, 16)


if __name__ == "__main__":
    unittest.main()
