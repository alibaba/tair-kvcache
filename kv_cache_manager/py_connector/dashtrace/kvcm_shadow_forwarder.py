"""Non-blocking DashTrace-to-KVCM observation forwarder.

The inference coroutine only performs a deterministic sampling decision and a
non-blocking queue insertion.  A single background I/O loop sends requests in
FIFO order so LiteHit observes the same request ordering as DashTrace.

With batching disabled it calls ``GetCacheLocation`` for compatibility.  With
batching enabled it uses KVCM's lightweight optimizer observation endpoint,
which only normalizes block keys and publishes ordered events.  Neither mode
writes cache data, and inference never waits for either response.
"""

from __future__ import annotations

import atexit
import hashlib
import http.client
import json
import logging
import os
import queue
import socket
import threading
import time
import uuid
from array import array
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence
from urllib.parse import urlsplit

from kv_cache_manager.py_connector.dashtrace._proto import kvcm_meta_minimal_pb2
from kv_cache_manager.py_connector.dashtrace._proto import (
    kvcm_optimizer_ingest_minimal_pb2,
)
from kv_cache_manager.py_connector.dashtrace.observed_request import ObservedRequest

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class KVCMShadowForwarderConfig:
    enabled: bool = False
    base_url: str = ""
    grpc_target: str = ""
    service_discovery_url: str = ""
    instance_id: str = ""
    queue_capacity: int = 4096
    timeout_ms: int = 500
    sample_ratio: float = 1.0
    max_qps: float = 0.0
    batch_size: int = 1
    batch_wait_ms: int = 2
    max_batch_tokens: int = 262144
    max_owned_tokens: int = 1048576
    producer_id: str = ""

    @classmethod
    def from_env(cls) -> "KVCMShadowForwarderConfig":
        enabled_name = (
            "DASHTRACE_ONLINE_MRC_ENABLED"
            if "DASHTRACE_ONLINE_MRC_ENABLED" in os.environ
            else "DASHTRACE_KVCM_ENABLED"
        )
        return cls(
            enabled=_env_bool(enabled_name),
            base_url=os.environ.get("DASHTRACE_KVCM_BASE_URL", "").rstrip("/"),
            grpc_target=os.environ.get("DASHTRACE_KVCM_GRPC_TARGET", "").strip(),
            service_discovery_url=os.environ.get(
                "DASHTRACE_KVCM_SERVICE_DISCOVERY_URL", ""
            ),
            instance_id=os.environ.get(
                "DASHTRACE_INSTANCE_ID",
                os.environ.get("DASHTRACE_KVCM_INSTANCE_ID", ""),
            ),
            queue_capacity=_env_int("DASHTRACE_KVCM_QUEUE_CAPACITY", 4096),
            timeout_ms=_env_int("DASHTRACE_KVCM_TIMEOUT_MS", 500),
            sample_ratio=_env_float("DASHTRACE_KVCM_SAMPLE_RATIO", 1.0),
            max_qps=_env_float("DASHTRACE_KVCM_MAX_QPS", 0.0),
            batch_size=_env_int("DASHTRACE_ONLINE_MRC_BATCH_SIZE", 1),
            batch_wait_ms=_env_int("DASHTRACE_ONLINE_MRC_BATCH_WAIT_MS", 2),
            max_batch_tokens=_env_int(
                "DASHTRACE_ONLINE_MRC_MAX_BATCH_TOKENS", 262144
            ),
            max_owned_tokens=_env_int(
                "DASHTRACE_ONLINE_MRC_MAX_OWNED_TOKENS", 1048576
            ),
            producer_id=os.environ.get("DASHTRACE_ONLINE_MRC_PRODUCER_ID", ""),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.base_url and not self.grpc_target and not self.service_discovery_url:
            raise ValueError(
                "DASHTRACE_KVCM_BASE_URL, DASHTRACE_KVCM_GRPC_TARGET or "
                "DASHTRACE_KVCM_SERVICE_DISCOVERY_URL must be set"
            )
        if self.base_url:
            parsed = urlsplit(self.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("DASHTRACE_KVCM_BASE_URL must be an http(s) URL")
        if not self.instance_id:
            raise ValueError("DASHTRACE_KVCM_INSTANCE_ID must not be empty")
        if self.queue_capacity <= 0:
            raise ValueError("DASHTRACE_KVCM_QUEUE_CAPACITY must be positive")
        if self.timeout_ms <= 0:
            raise ValueError("DASHTRACE_KVCM_TIMEOUT_MS must be positive")
        if not 0.0 <= self.sample_ratio <= 1.0:
            raise ValueError("DASHTRACE_KVCM_SAMPLE_RATIO must be in [0, 1]")
        if self.max_qps < 0:
            raise ValueError("DASHTRACE_KVCM_MAX_QPS must be non-negative")
        if not 1 <= self.batch_size <= 256:
            raise ValueError("DASHTRACE_ONLINE_MRC_BATCH_SIZE must be in [1, 256]")
        if self.batch_wait_ms < 0:
            raise ValueError("DASHTRACE_ONLINE_MRC_BATCH_WAIT_MS must be non-negative")
        if self.max_batch_tokens <= 0:
            raise ValueError(
                "DASHTRACE_ONLINE_MRC_MAX_BATCH_TOKENS must be positive"
            )
        if self.max_owned_tokens <= 0:
            raise ValueError(
                "DASHTRACE_ONLINE_MRC_MAX_OWNED_TOKENS must be positive"
            )


class _NoopMetric:
    def labels(self, **_: str) -> "_NoopMetric":
        return self

    def inc(self, _: float = 1.0) -> None:
        pass

    def set(self, _: float) -> None:
        pass

    def observe(self, _: float) -> None:
        pass


def _make_metrics() -> tuple[Any, ...]:
    try:
        from prometheus_client import Counter, Gauge, Histogram

        submitted = Counter(
            "dashtrace_kvcm_shadow_submitted_total",
            "DashTrace requests accepted by the KVCM shadow queue.",
        )
        observed = Counter(
            "dashtrace_kvcm_shadow_observed_total",
            "DashTrace requests observed before sampling and queue admission.",
        )
        dropped = Counter(
            "dashtrace_kvcm_shadow_dropped_total",
            "DashTrace requests not forwarded to KVCM.",
            ["reason"],
        )
        requests = Counter(
            "dashtrace_kvcm_shadow_requests_total",
            "Completed DashTrace observations sent toward KVCM.",
            ["result"],
        )
        acknowledged = Counter(
            "dashtrace_kvcm_shadow_acknowledged_total",
            "Observations acknowledged as atomically admitted by KVCM.",
        )
        depth = Gauge(
            "dashtrace_kvcm_shadow_queue_depth",
            "Current number of requests waiting for KVCM shadow forwarding.",
        )
        latency = Histogram(
            "dashtrace_kvcm_shadow_request_latency_seconds",
            "KVCM shadow transport latency per unary RPC or batch.",
        )
        batch_size = Histogram(
            "dashtrace_kvcm_shadow_batch_size",
            "Number of observations in one KVCM transport request.",
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        )
        owned_tokens = Gauge(
            "dashtrace_kvcm_shadow_owned_tokens",
            "Compact token ids retained by the online MRC reporter.",
        )
        owned_bytes = Gauge(
            "dashtrace_kvcm_shadow_owned_token_bytes",
            "Bytes in compact int64 token arrays retained by the reporter.",
        )
        oldest_age = Gauge(
            "dashtrace_kvcm_shadow_oldest_age_seconds",
            "Age of the oldest observation in the current transport request.",
        )
        return (
            submitted,
            observed,
            dropped,
            requests,
            acknowledged,
            depth,
            latency,
            batch_size,
            owned_tokens,
            owned_bytes,
            oldest_age,
        )
    except (ImportError, ValueError):
        # ValueError also covers duplicate registration in unusual reload setups.
        noop = _NoopMetric()
        return (noop,) * 11


(
    _SUBMITTED_TOTAL,
    _OBSERVED_TOTAL,
    _DROPPED_TOTAL,
    _REQUESTS_TOTAL,
    _ACKNOWLEDGED_TOTAL,
    _QUEUE_DEPTH,
    _REQUEST_LATENCY,
    _BATCH_SIZE,
    _OWNED_TOKENS,
    _OWNED_TOKEN_BYTES,
    _OLDEST_AGE,
) = _make_metrics()


@dataclass(frozen=True)
class _PendingRequest:
    sequence: int
    timestamp_ns: int
    request_id: str
    instance_id: str
    token_ids: array
    enqueued_at: float


class KVCMShadowForwarder:
    """At-most-once, ordered shadow forwarding to KVCM."""

    _STOP = object()

    def __init__(
        self,
        config: KVCMShadowForwarderConfig,
        service_discovery_factory: Optional[Callable[[str], Any]] = None,
        grpc_channel_factory: Optional[Callable[..., Any]] = None,
    ):
        config.validate()
        self._config = config
        self._queue: queue.Queue[object] = queue.Queue(maxsize=config.queue_capacity)
        self._closed = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connection: Optional[http.client.HTTPConnection] = None
        self._grpc_channel: Optional[Any] = None
        self._grpc_get_cache_location: Optional[Callable[..., Any]] = None
        self._grpc_report_trace_batch: Optional[Callable[..., Any]] = None
        self._service_discovery: Optional[Any] = None
        self._next_send_time = 0.0
        self._ownership_lock = threading.Lock()
        self._owned_tokens = 0
        self._carry: Optional[_PendingRequest] = None
        self._legacy_sequence = 0
        self._legacy_sequence_lock = threading.Lock()
        self._producer_id = config.producer_id or (
            f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}"
        )
        if config.enabled:
            if config.grpc_target:
                channel_factory = grpc_channel_factory
                if channel_factory is None:
                    import grpc

                    channel_factory = grpc.insecure_channel
                self._grpc_channel = channel_factory(
                    config.grpc_target,
                    options=(
                        ("grpc.max_send_message_length", 100 * 1024 * 1024),
                        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                    ),
                )
                self._grpc_get_cache_location = self._grpc_channel.unary_unary(
                    "/kv_cache_manager.proto.meta.MetaService/GetCacheLocation",
                    request_serializer=kvcm_meta_minimal_pb2.GetCacheLocationRequest.SerializeToString,
                    response_deserializer=kvcm_meta_minimal_pb2.GetCacheLocationResponse.FromString,
                )
                if config.batch_size > 1:
                    self._grpc_report_trace_batch = self._grpc_channel.unary_unary(
                        "/kv_cache_manager.proto.optimizer.OptimizerEventStreamService/ReportTraceBatch",
                        request_serializer=kvcm_optimizer_ingest_minimal_pb2.TraceObservationBatchRequest.SerializeToString,
                        response_deserializer=kvcm_optimizer_ingest_minimal_pb2.TraceObservationBatchResponse.FromString,
                    )
            if config.service_discovery_url:
                if service_discovery_factory is None:
                    from kv_cache_manager.py_connector.common.service_discovery_factory import (
                        create_service_discovery,
                    )

                    service_discovery_factory = create_service_discovery
                self._service_discovery = service_discovery_factory(
                    config.service_discovery_url
                )
                if self._service_discovery is None:
                    raise ValueError(
                        "failed to create DASHTRACE_KVCM_SERVICE_DISCOVERY_URL"
                    )
            self._thread = threading.Thread(
                target=self._run,
                name="dashtrace-kvcm-shadow",
                daemon=True,
            )
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def submit(self, request_id: str, token_ids: Sequence[int]) -> bool:
        """Queue one request without blocking the caller."""
        compact = (
            token_ids
            if isinstance(token_ids, array) and token_ids.typecode == "q"
            else array("q", (int(token) for token in token_ids))
        )
        with self._legacy_sequence_lock:
            sequence = self._legacy_sequence
            self._legacy_sequence += 1
        return self.submit_observation(
            ObservedRequest(
                sequence=sequence,
                timestamp_ns=time.time_ns(),
                trace_id=str(request_id),
                instance_id=self._config.instance_id,
                token_ids=compact,
            )
        )

    def submit_observation(self, observation: ObservedRequest) -> bool:
        """Queue one shared observation without blocking or copying tokens."""
        if (
            not self._config.enabled
            or self._closed.is_set()
            or not observation.token_ids
        ):
            return False
        _OBSERVED_TOTAL.inc()
        if not self._sample(observation.trace_id):
            _DROPPED_TOTAL.labels(reason="sampled_out").inc()
            return False
        token_count = len(observation.token_ids)
        if token_count > self._config.max_batch_tokens:
            _DROPPED_TOTAL.labels(reason="request_too_large").inc()
            return False
        pending = _PendingRequest(
            sequence=observation.sequence,
            timestamp_ns=observation.timestamp_ns,
            request_id=observation.trace_id,
            instance_id=observation.instance_id,
            token_ids=observation.token_ids,
            enqueued_at=time.monotonic(),
        )
        with self._ownership_lock:
            if self._owned_tokens + token_count > self._config.max_owned_tokens:
                _DROPPED_TOTAL.labels(reason="token_capacity").inc()
                return False
            try:
                self._queue.put_nowait(pending)
            except queue.Full:
                _DROPPED_TOTAL.labels(reason="queue_full").inc()
                return False
            self._owned_tokens += token_count
            _OWNED_TOKENS.set(self._owned_tokens)
            _OWNED_TOKEN_BYTES.set(self._owned_tokens * 8)
        _SUBMITTED_TOTAL.inc()
        _QUEUE_DEPTH.set(self._queue.qsize())
        return True

    def close(self, timeout: float = 2.0) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._close_connection()
        if self._grpc_channel is not None:
            self._grpc_channel.close()
        if self._service_discovery is not None:
            self._service_discovery.close()

    def _sample(self, request_id: str) -> bool:
        ratio = self._config.sample_ratio
        if ratio >= 1.0:
            return True
        if ratio <= 0.0:
            return False
        digest = hashlib.blake2b(str(request_id).encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big") / float(1 << 64)
        return value < ratio

    def _run(self) -> None:
        while True:
            if self._carry is None:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if self._closed.is_set():
                        break
                    continue
            else:
                item = self._carry
                self._carry = None
            _QUEUE_DEPTH.set(self._queue.qsize())
            if item is self._STOP:
                break
            assert isinstance(item, _PendingRequest)
            batch, stop_after_batch = self._take_batch(item)
            self._pace(len(batch))
            started = time.monotonic()
            result = "ok"
            acknowledged = 0
            try:
                _OLDEST_AGE.set(max(time.monotonic() - batch[0].enqueued_at, 0.0))
                _BATCH_SIZE.observe(len(batch))
                acknowledged = self._send_batch(batch)
            except Exception as exc:  # noqa: BLE001 - background path must stay alive
                result = self._result_label(exc)
                _DROPPED_TOTAL.labels(reason=result).inc(len(batch))
                self._close_connection()
                logger.warning("DashTrace KVCM shadow request failed: %s", exc)
            finally:
                if acknowledged:
                    _ACKNOWLEDGED_TOTAL.inc(acknowledged)
                _REQUEST_LATENCY.observe(time.monotonic() - started)
                _REQUESTS_TOTAL.labels(result=result).inc()
                with self._ownership_lock:
                    self._owned_tokens -= sum(len(item.token_ids) for item in batch)
                    _OWNED_TOKENS.set(self._owned_tokens)
                    _OWNED_TOKEN_BYTES.set(self._owned_tokens * 8)
                for _ in batch:
                    self._queue.task_done()
                _OLDEST_AGE.set(0.0)
            if stop_after_batch:
                break

    def _take_batch(self, first: _PendingRequest) -> tuple[list[_PendingRequest], bool]:
        if self._grpc_report_trace_batch is None:
            return [first], False
        batch = [first]
        total_tokens = len(first.token_ids)
        deadline = time.monotonic() + self._config.batch_wait_ms / 1000.0
        while len(batch) < self._config.batch_size:
            timeout = max(0.0, deadline - time.monotonic())
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                break
            _QUEUE_DEPTH.set(self._queue.qsize())
            if item is self._STOP:
                return batch, True
            assert isinstance(item, _PendingRequest)
            if total_tokens + len(item.token_ids) > self._config.max_batch_tokens:
                self._carry = item
                break
            batch.append(item)
            total_tokens += len(item.token_ids)
        return batch, False

    def _pace(self, count: int = 1) -> None:
        if self._config.max_qps <= 0:
            return
        interval = count / self._config.max_qps
        now = time.monotonic()
        if now < self._next_send_time:
            time.sleep(self._next_send_time - now)
            now = time.monotonic()
        self._next_send_time = max(now, self._next_send_time) + interval

    def _new_connection(self) -> http.client.HTTPConnection:
        if self._service_discovery is not None:
            endpoint = self._service_discovery.get_one_endpoint()
            if endpoint is None:
                self._service_discovery.refresh()
                endpoint = self._service_discovery.get_one_endpoint()
            if endpoint is None:
                raise RuntimeError("service_discovery_empty")
            return http.client.HTTPConnection(
                endpoint.ip,
                endpoint.port,
                timeout=self._config.timeout_ms / 1000.0,
            )
        parsed = urlsplit(self._config.base_url)
        cls: type[http.client.HTTPConnection]
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return cls(parsed.hostname, port, timeout=self._config.timeout_ms / 1000.0)

    def _send(self, item: _PendingRequest) -> None:
        if self._grpc_get_cache_location is not None:
            self._send_grpc(item)
            return
        if self._connection is None:
            self._connection = self._new_connection()
        base_path = ""
        if self._config.base_url:
            base_path = urlsplit(self._config.base_url).path.rstrip("/")
        path = f"{base_path}/api/getCacheLocation"
        body = json.dumps(
            {
                "trace_id": item.request_id,
                "instance_id": item.instance_id,
                "query_type": "QT_PREFIX_MATCH",
                "token_ids": list(item.token_ids),
                "block_mask": {"offset": 0},
            },
            separators=(",", ":"),
        )
        self._connection.request(
            "POST",
            path,
            body=body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        response = self._connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f"http_{response.status}")
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("invalid_json") from exc
        status = decoded.get("header", {}).get("status", {})
        if status.get("code") != "OK":
            code = str(status.get("code") or "unknown").lower()
            raise RuntimeError(f"kvcm_{code}")

    def _send_batch(self, batch: list[_PendingRequest]) -> int:
        if self._grpc_report_trace_batch is None:
            for item in batch:
                self._send(item)
            return len(batch)

        request = kvcm_optimizer_ingest_minimal_pb2.TraceObservationBatchRequest(
            producer_id=self._producer_id
        )
        for item in batch:
            observation = request.observations.add(
                sequence=item.sequence,
                timestamp_ns=item.timestamp_ns,
                trace_id=item.request_id,
                instance_id=item.instance_id,
            )
            observation.token_ids.extend(item.token_ids)
        response = self._grpc_report_trace_batch(
            request, timeout=self._config.timeout_ms / 1000.0
        )
        if response.header.status.code != kvcm_optimizer_ingest_minimal_pb2.OK:
            code = str(response.header.status.code).lower()
            raise RuntimeError(f"kvcm_{code}")
        if (
            response.accepted_count != len(batch)
            or response.last_accepted_sequence != batch[-1].sequence
        ):
            raise RuntimeError("kvcm_incomplete_batch_ack")
        return int(response.accepted_count)

    def _send_grpc(self, item: _PendingRequest) -> None:
        request = kvcm_meta_minimal_pb2.GetCacheLocationRequest(
            trace_id=item.request_id,
            instance_id=item.instance_id,
            query_type=kvcm_meta_minimal_pb2.QT_PREFIX_MATCH,
        )
        request.token_ids.extend(item.token_ids)
        request.block_mask.offset = 0
        response = self._grpc_get_cache_location(
            request, timeout=self._config.timeout_ms / 1000.0
        )
        if response.header.status.code != kvcm_meta_minimal_pb2.OK:
            code = str(response.header.status.code).lower()
            raise RuntimeError(f"kvcm_{code}")

    def _close_connection(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None

    @staticmethod
    def _result_label(exc: Exception) -> str:
        message = str(exc)
        if message.startswith("http_") or message.startswith("kvcm_"):
            return message
        if isinstance(exc, TimeoutError):
            return "timeout"
        return "transport_error"


_singleton: Optional[KVCMShadowForwarder] = None
_singleton_lock = threading.Lock()


def get_kvcm_shadow_forwarder() -> KVCMShadowForwarder:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = KVCMShadowForwarder(KVCMShadowForwarderConfig.from_env())
                atexit.register(_singleton.close)
    return _singleton


def attach_kvcm_shadow_forwarder(
    tracer: Any,
    forwarder: Optional[KVCMShadowForwarder] = None,
) -> Any:
    """Attach forwarding to a DashTrace tracer without changing worker logic.

    The wrapper invokes the original ``record`` first, preserving DashTrace's
    current WAL semantics, then performs a non-blocking queue insertion.
    """
    if getattr(tracer, "_kvcm_shadow_forwarder_attached", False):
        return tracer
    target = forwarder or get_kvcm_shadow_forwarder()
    original: Callable[..., None] = tracer.record

    def record_and_forward(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        request_id = kwargs.get("request_id")
        token_ids = kwargs.get("token_ids")
        if request_id is None and args:
            request_id = args[0]
        if token_ids is None and len(args) > 1:
            token_ids = args[1]
        if request_id is not None and token_ids:
            target.submit(str(request_id), token_ids)

    tracer.record = record_and_forward
    tracer._kvcm_shadow_forwarder_attached = True
    return tracer
