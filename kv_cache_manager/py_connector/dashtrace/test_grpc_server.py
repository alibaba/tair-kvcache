import os
import struct
import sys
import types
import unittest
from unittest import mock

from kv_cache_manager.py_connector.dashtrace._proto import (
    kserve_minimal_pb2 as pb2,
)
from kv_cache_manager.py_connector.dashtrace.grpc_server import (
    _metrics_payload,
    _stream_responses,
)


class GrpcServerTest(unittest.TestCase):
    def test_metrics_payload_merges_dashtrace_and_colocated_kvcm(self):
        prometheus = types.SimpleNamespace(
            CONTENT_TYPE_LATEST="text/plain; version=0.0.4",
            generate_latest=lambda: b"dashtrace_metric 1\n",
        )
        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = b"kvcm_metric 2\n"
        response.__enter__.return_value = response
        with mock.patch.dict(sys.modules, {"prometheus_client": prometheus}), mock.patch.dict(
            os.environ,
            {"DASHTRACE_KVCM_METRICS_URL": "http://127.0.0.1:6492/metrics"},
        ), mock.patch("urllib.request.urlopen", return_value=response):
            payload, content_type = _metrics_payload()

        self.assertEqual(b"dashtrace_metric 1\nkvcm_metric 2\n", payload)
        self.assertEqual("text/plain; version=0.0.4", content_type)

    def test_stream_response_matches_dashserving_contract(self):
        request = pb2.ModelInferRequest(model_name="request-model", id="request-1")
        with mock.patch.dict(
            os.environ, {"DASHTRACE_RESPONSE_MODEL_NAME": "served-model"}
        ):
            first, last = _stream_responses(request)

        self.assertEqual("served-model", first.model_name)
        self.assertEqual("request-1", first.id)
        self.assertEqual(["finish_reason", "generated_ids"], [
            output.name for output in first.outputs
        ])
        self.assertEqual((2,), struct.unpack("<q", first.raw_output_contents[0]))
        self.assertEqual((0,), struct.unpack("<i", first.raw_output_contents[1]))
        self.assertEqual((0,), struct.unpack("<q", last.raw_output_contents[0]))
        self.assertEqual(
            (0, 100, 151645), struct.unpack("<3i", last.raw_output_contents[1])
        )
        self.assertEqual(9, last.parameters["prompt_token_num"].int64_param)
        self.assertEqual("served-model", last.parameters["model_name"].string_param)


if __name__ == "__main__":
    unittest.main()
