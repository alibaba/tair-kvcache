"""Minimal no-model KServe v2 server for replicated DashTrace traffic."""

from __future__ import annotations

import os
import signal
import struct
import threading
import urllib.request
from concurrent import futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import grpc

from kv_cache_manager.py_connector.dashtrace._proto import (
    kserve_minimal_pb2 as pb2,
)
from kv_cache_manager.py_connector.dashtrace._proto import (
    kserve_minimal_pb2_grpc as pb2_grpc,
)
from kv_cache_manager.py_connector.dashtrace.instance_bootstrap import (
    start_instance_bootstrap,
)
from kv_cache_manager.py_connector.dashtrace.request_observer import (
    get_request_observer,
)


def _metrics_payload() -> tuple[bytes, str]:
    """Return DashTrace metrics plus the colocated KVCM Prometheus page."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    payload = generate_latest()
    upstream_url = os.environ.get("DASHTRACE_KVCM_METRICS_URL", "").strip()
    if upstream_url:
        try:
            with urllib.request.urlopen(upstream_url, timeout=0.5) as response:
                if response.status != 200:
                    raise RuntimeError(f"unexpected status {response.status}")
                upstream = response.read()
            payload = payload.rstrip(b"\n") + b"\n" + upstream.lstrip(b"\n")
        except Exception as error:  # noqa: BLE001 - local metrics stay available
            print(f"DashTrace KVCM metrics merge failed: {error}", flush=True)
    return payload, CONTENT_TYPE_LATEST


def _response_chunk(
    request: pb2.ModelInferRequest,
    *,
    finish_reason: int,
    generated_ids: tuple[int, ...],
) -> pb2.ModelInferResponse:
    model_name = os.environ.get(
        "DASHTRACE_RESPONSE_MODEL_NAME", request.model_name
    )
    response = pb2.ModelInferResponse(
        model_name=model_name,
        model_version="",
        id=request.id,
    )
    response.parameters["incremental_output"].int64_param = 1
    response.parameters["prompt_cached_token_num"].int64_param = 0
    response.parameters["prompt_token_num"].int64_param = 9
    response.parameters["model_name"].string_param = model_name

    finish = response.outputs.add(name="finish_reason", datatype="INT64")
    finish.shape.append(1)
    response.raw_output_contents.append(struct.pack("<q", finish_reason))

    tokens = response.outputs.add(name="generated_ids", datatype="INT32")
    tokens.shape.extend((1, len(generated_ids)))
    response.raw_output_contents.append(
        struct.pack(f"<{len(generated_ids)}i", *generated_ids)
    )
    return response


def _stream_responses(
    request: pb2.ModelInferRequest,
) -> tuple[pb2.ModelInferResponse, pb2.ModelInferResponse]:
    # Keep this two-chunk contract aligned with the production DashTrace worker.
    # DashServing treats a single empty terminal chunk as EngineAbort.
    return (
        _response_chunk(request, finish_reason=2, generated_ids=(0,)),
        _response_chunk(
            request,
            finish_reason=0,
            generated_ids=(0, 100, 151645),
        ),
    )


class DashTraceInferenceService(pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(self, observer=None):
        self._observer = observer or get_request_observer()

    def ServerLive(self, _request, _context):
        return pb2.ServerLiveResponse(live=True)

    def ServerReady(self, _request, _context):
        return pb2.ServerReadyResponse(ready=True)

    def ModelReady(self, _request, _context):
        return pb2.ModelReadyResponse(ready=True)

    def ServerMetadata(self, _request, _context):
        return pb2.ServerMetadataResponse(
            name="kvcm-dashtrace", version="1", extensions=["model_stream_infer"]
        )

    def ModelMetadata(self, request, _context):
        return pb2.ModelMetadataResponse(
            name=request.name,
            versions=[request.version] if request.version else ["1"],
            platform="kvcm-dashtrace",
        )

    def ModelInfer(self, request, _context):
        self._observer.observe(request)
        return _stream_responses(request)[-1]

    def ModelStreamInfer(self, request_iterator, _context):
        for request in request_iterator:
            self._observer.observe(request)
            for response in _stream_responses(request):
                yield pb2.ModelStreamInferResponse(infer_response=response)


class _HealthHandler(BaseHTTPRequestHandler):
    instance_ready = threading.Event()
    instance_ready.set()

    def do_GET(self):
        if self.path == "/metrics":
            try:
                payload, content_type = _metrics_payload()
            except Exception as error:  # pragma: no cover - runtime dependency guard
                payload = f"metrics unavailable: {error}\n".encode()
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
            else:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path not in {"/", "/health", "/liveness", "/readiness"}:
            self.send_error(404)
            return
        if self.path == "/readiness":
            if not self.instance_ready.is_set():
                payload = b"KVCM instance not ready\n"
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            health_url = os.environ.get("DASHTRACE_KVCM_HEALTH_URL", "")
            if health_url:
                try:
                    with urllib.request.urlopen(health_url, timeout=0.5) as response:
                        if response.status != 200:
                            raise RuntimeError(f"unexpected status {response.status}")
                except Exception as error:
                    payload = f"KVCM not ready: {error}\n".encode()
                    self.send_response(503)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
        payload = b"ok\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


def serve() -> None:
    port = int(os.environ.get("DASHTRACE_GRPC_PORT", "8090"))
    health_port = int(os.environ.get("DASHTRACE_HEALTH_PORT", "8601"))
    workers = int(os.environ.get("DASHTRACE_GRPC_WORKERS", "4"))
    max_message = int(
        os.environ.get("DASHTRACE_GRPC_MAX_MESSAGE_BYTES", str(100 * 1024 * 1024))
    )

    stopping = threading.Event()
    instance_ready = threading.Event()
    _HealthHandler.instance_ready = instance_ready
    registration_thread = start_instance_bootstrap(instance_ready, stopping)

    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        options=(
            ("grpc.max_receive_message_length", max_message),
            ("grpc.max_send_message_length", max_message),
        ),
    )
    pb2_grpc.add_GRPCInferenceServiceServicer_to_server(
        DashTraceInferenceService(), grpc_server
    )
    grpc_server.add_insecure_port(f"[::]:{port}")

    health_server = ThreadingHTTPServer(("0.0.0.0", health_port), _HealthHandler)
    health_thread = threading.Thread(
        target=health_server.serve_forever,
        name="dashtrace-health",
        daemon=True,
    )
    health_thread.start()

    def stop(_signum, _frame):
        if stopping.is_set():
            return
        stopping.set()
        health_server.shutdown()
        grpc_server.stop(grace=5)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    grpc_server.start()
    print(
        f"DashTrace KServe server listening on grpc={port} health={health_port}",
        flush=True,
    )
    grpc_server.wait_for_termination()
    if registration_thread is not None:
        registration_thread.join(timeout=1)


if __name__ == "__main__":
    serve()
