import json
import threading
import time
import unittest
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from kv_cache_manager.py_connector.dashtrace._proto import kvcm_meta_minimal_pb2
from kv_cache_manager.py_connector.dashtrace._proto import (
    kvcm_optimizer_ingest_minimal_pb2,
)
from kv_cache_manager.py_connector.dashtrace.kvcm_shadow_forwarder import (
    KVCMShadowForwarder,
    KVCMShadowForwarderConfig,
    attach_kvcm_shadow_forwarder,
)


class _Handler(BaseHTTPRequestHandler):
    requests = []
    lock = threading.Lock()
    response_code = "OK"

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        with self.lock:
            self.requests.append((self.path, body))
        payload = json.dumps({"header": {"status": {"code": self.response_code}}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


class _Tracer:
    def __init__(self):
        self.records = []

    def record(self, request_id, token_ids, **kwargs):
        self.records.append((request_id, list(token_ids), kwargs))


@dataclass
class _Endpoint:
    ip: str
    port: int


class _Discovery:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.closed = False

    def get_one_endpoint(self):
        return self.endpoint

    def refresh(self):
        return True

    def close(self):
        self.closed = True


class _GrpcChannel:
    def __init__(self):
        self.requests = []
        self.batch_requests = []
        self.closed = False

    def unary_unary(self, path, request_serializer, response_deserializer):
        self.path = path

        if path.endswith("/ReportTraceBatch"):
            def report_batch(request, timeout):
                self.batch_requests.append((request, timeout))
                return kvcm_optimizer_ingest_minimal_pb2.TraceObservationBatchResponse(
                    header=kvcm_optimizer_ingest_minimal_pb2.CommonResponseHeader(
                        status=kvcm_optimizer_ingest_minimal_pb2.Status(
                            code=kvcm_optimizer_ingest_minimal_pb2.OK
                        )
                    ),
                    accepted_count=len(request.observations),
                    last_accepted_sequence=request.observations[-1].sequence,
                )

            return report_batch

        def call(request, timeout):
            self.requests.append((request, timeout))
            return kvcm_meta_minimal_pb2.GetCacheLocationResponse(
                header=kvcm_meta_minimal_pb2.CommonResponseHeader(
                    status=kvcm_meta_minimal_pb2.Status(
                        code=kvcm_meta_minimal_pb2.OK
                    )
                )
            )

        return call

    def close(self):
        self.closed = True


class KVCMShadowForwarderTest(unittest.TestCase):
    def setUp(self):
        _Handler.requests = []
        _Handler.response_code = "OK"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2)

    def _forwarder(self, **overrides):
        values = {
            "enabled": True,
            "base_url": f"http://127.0.0.1:{self.server.server_port}",
            "instance_id": "shadow-instance",
            "queue_capacity": 8,
            "timeout_ms": 1000,
            "sample_ratio": 1.0,
            "max_qps": 0.0,
        }
        values.update(overrides)
        return KVCMShadowForwarder(KVCMShadowForwarderConfig(**values))

    def test_forwards_requests_in_order_with_tokens(self):
        forwarder = self._forwarder()
        try:
            self.assertTrue(forwarder.submit("request-1", [1, 2, 3]))
            self.assertTrue(forwarder.submit("request-2", [4, 5]))
            deadline = time.time() + 2
            while len(_Handler.requests) < 2 and time.time() < deadline:
                time.sleep(0.01)
        finally:
            forwarder.close()

        self.assertEqual(2, len(_Handler.requests))
        self.assertEqual("/api/getCacheLocation", _Handler.requests[0][0])
        self.assertEqual(["request-1", "request-2"], [item[1]["trace_id"] for item in _Handler.requests])
        self.assertEqual([1, 2, 3], _Handler.requests[0][1]["token_ids"])
        self.assertEqual("shadow-instance", _Handler.requests[0][1]["instance_id"])
        self.assertEqual("QT_PREFIX_MATCH", _Handler.requests[0][1]["query_type"])

    def test_disabled_forwarder_never_queues(self):
        config = KVCMShadowForwarderConfig(enabled=False)
        forwarder = KVCMShadowForwarder(config)
        self.assertFalse(forwarder.submit("request", [1]))
        self.assertEqual(0, forwarder.queue_depth)

    def test_rejects_request_larger_than_batch_token_limit(self):
        forwarder = self._forwarder(max_batch_tokens=2)
        try:
            self.assertFalse(forwarder.submit("request", [1, 2, 3]))
            self.assertEqual(0, forwarder.queue_depth)
        finally:
            forwarder.close()

    def test_rejects_when_owned_token_budget_is_exhausted(self):
        channel = _GrpcChannel()
        config = KVCMShadowForwarderConfig(
            enabled=True,
            grpc_target="127.0.0.1:6381",
            instance_id="shadow-instance",
            batch_size=32,
            batch_wait_ms=100,
            max_owned_tokens=3,
        )
        forwarder = KVCMShadowForwarder(
            config, grpc_channel_factory=lambda *_args, **_kwargs: channel
        )
        try:
            self.assertTrue(forwarder.submit("request-1", [1, 2, 3]))
            self.assertFalse(forwarder.submit("request-2", [4]))
        finally:
            forwarder.close()

    def test_service_discovery_endpoint_is_used(self):
        discovery = _Discovery(_Endpoint("127.0.0.1", self.server.server_port))
        config = KVCMShadowForwarderConfig(
            enabled=True,
            service_discovery_url="spectrum://v-test?port=6382",
            instance_id="shadow-instance",
            timeout_ms=1000,
        )
        forwarder = KVCMShadowForwarder(config, lambda _: discovery)
        try:
            self.assertTrue(forwarder.submit("request-sd", [9, 10]))
            deadline = time.time() + 2
            while len(_Handler.requests) < 1 and time.time() < deadline:
                time.sleep(0.01)
        finally:
            forwarder.close()

        self.assertEqual(1, len(_Handler.requests))
        self.assertEqual("request-sd", _Handler.requests[0][1]["trace_id"])
        self.assertTrue(discovery.closed)

    def test_grpc_transport_preserves_order_and_tokens(self):
        channel = _GrpcChannel()
        config = KVCMShadowForwarderConfig(
            enabled=True,
            grpc_target="127.0.0.1:6381",
            instance_id="shadow-instance",
            timeout_ms=1000,
        )
        forwarder = KVCMShadowForwarder(
            config, grpc_channel_factory=lambda *_args, **_kwargs: channel
        )
        try:
            self.assertTrue(forwarder.submit("request-grpc-1", [1, 2, 3]))
            self.assertTrue(forwarder.submit("request-grpc-2", [4, 5]))
            deadline = time.time() + 2
            while len(channel.requests) < 2 and time.time() < deadline:
                time.sleep(0.01)
        finally:
            forwarder.close()

        self.assertEqual(
            "/kv_cache_manager.proto.meta.MetaService/GetCacheLocation",
            channel.path,
        )
        self.assertEqual(
            ["request-grpc-1", "request-grpc-2"],
            [item[0].trace_id for item in channel.requests],
        )
        self.assertEqual([1, 2, 3], list(channel.requests[0][0].token_ids))
        self.assertTrue(channel.closed)

    def test_grpc_batch_transport_sends_one_ordered_acked_batch(self):
        channel = _GrpcChannel()
        config = KVCMShadowForwarderConfig(
            enabled=True,
            grpc_target="127.0.0.1:6381",
            instance_id="shadow-instance",
            timeout_ms=1000,
            batch_size=3,
            batch_wait_ms=50,
            producer_id="producer-test",
        )
        forwarder = KVCMShadowForwarder(
            config, grpc_channel_factory=lambda *_args, **_kwargs: channel
        )
        try:
            for index in range(3):
                self.assertTrue(forwarder.submit(f"request-{index}", [index, 9]))
            deadline = time.time() + 2
            while len(channel.batch_requests) < 1 and time.time() < deadline:
                time.sleep(0.01)
        finally:
            forwarder.close()

        self.assertEqual(1, len(channel.batch_requests))
        request = channel.batch_requests[0][0]
        self.assertEqual("producer-test", request.producer_id)
        self.assertEqual([0, 1, 2], [item.sequence for item in request.observations])
        self.assertEqual(
            ["request-0", "request-1", "request-2"],
            [item.trace_id for item in request.observations],
        )
        self.assertEqual([0, 9], list(request.observations[0].token_ids))

    def test_grpc_batch_respects_total_token_limit_without_reordering(self):
        channel = _GrpcChannel()
        config = KVCMShadowForwarderConfig(
            enabled=True,
            grpc_target="127.0.0.1:6381",
            instance_id="shadow-instance",
            timeout_ms=1000,
            batch_size=3,
            batch_wait_ms=50,
            max_batch_tokens=3,
        )
        forwarder = KVCMShadowForwarder(
            config, grpc_channel_factory=lambda *_args, **_kwargs: channel
        )
        try:
            self.assertTrue(forwarder.submit("request-1", [1, 2]))
            self.assertTrue(forwarder.submit("request-2", [3, 4]))
            deadline = time.time() + 2
            while len(channel.batch_requests) < 2 and time.time() < deadline:
                time.sleep(0.01)
        finally:
            forwarder.close()

        self.assertEqual(2, len(channel.batch_requests))
        self.assertEqual(
            ["request-1", "request-2"],
            [call[0].observations[0].trace_id for call in channel.batch_requests],
        )

    def test_attach_preserves_tracer_and_is_idempotent(self):
        forwarder = self._forwarder()
        tracer = _Tracer()
        try:
            self.assertIs(tracer, attach_kvcm_shadow_forwarder(tracer, forwarder))
            self.assertIs(tracer, attach_kvcm_shadow_forwarder(tracer, forwarder))
            tracer.record(request_id="request-3", token_ids=[7, 8], request_type="text")
            deadline = time.time() + 2
            while len(_Handler.requests) < 1 and time.time() < deadline:
                time.sleep(0.01)
        finally:
            forwarder.close()

        self.assertEqual([("request-3", [7, 8], {"request_type": "text"})], tracer.records)
        self.assertEqual(1, len(_Handler.requests))

    def test_invalid_enabled_config_fails_fast(self):
        with self.assertRaises(ValueError):
            KVCMShadowForwarder(
                KVCMShadowForwarderConfig(
                    enabled=True,
                    base_url="",
                    instance_id="shadow-instance",
                )
            )

    def test_service_discovery_only_config_is_valid(self):
        config = KVCMShadowForwarderConfig(
            enabled=True,
            service_discovery_url="spectrum://v-test?port=6382",
            instance_id="shadow-instance",
        )
        config.validate()

    def test_online_mrc_enable_switch_overrides_legacy_switch(self):
        with patch.dict(
            "os.environ",
            {
                "DASHTRACE_ONLINE_MRC_ENABLED": "false",
                "DASHTRACE_KVCM_ENABLED": "true",
            },
            clear=True,
        ):
            self.assertFalse(KVCMShadowForwarderConfig.from_env().enabled)

        with patch.dict(
            "os.environ",
            {"DASHTRACE_KVCM_ENABLED": "true"},
            clear=True,
        ):
            self.assertTrue(KVCMShadowForwarderConfig.from_env().enabled)

    def test_generic_instance_id_overrides_legacy_instance_id(self):
        with patch.dict(
            "os.environ",
            {
                "DASHTRACE_INSTANCE_ID": "generic-instance",
                "DASHTRACE_KVCM_INSTANCE_ID": "legacy-instance",
            },
            clear=True,
        ):
            self.assertEqual(
                "generic-instance",
                KVCMShadowForwarderConfig.from_env().instance_id,
            )


if __name__ == "__main__":
    unittest.main()
