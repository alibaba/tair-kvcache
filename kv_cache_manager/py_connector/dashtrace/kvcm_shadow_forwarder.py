"""Non-blocking ordered DashTrace-to-KVCM batch observation forwarder."""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import socket
import threading
import time
import uuid
from array import array
from dataclasses import dataclass
from typing import Any, Callable, Optional

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
    grpc_target: str = ""
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
        return cls(
            enabled=_env_bool("DASHTRACE_ONLINE_MRC_ENABLED"),
            grpc_target=os.environ.get("DASHTRACE_KVCM_GRPC_TARGET", "").strip(),
            instance_id=os.environ.get("DASHTRACE_INSTANCE_ID", ""),
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
        if not self.grpc_target:
            raise ValueError("DASHTRACE_KVCM_GRPC_TARGET must not be empty")
        if not self.instance_id:
            raise ValueError("DASHTRACE_INSTANCE_ID must not be empty")
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
        grpc_channel_factory: Optional[Callable[..., Any]] = None,
    ):
        config.validate()
        self._config = config
        self._queue: queue.Queue[object] = queue.Queue(maxsize=config.queue_capacity)
        self._closed = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._grpc_channel: Optional[Any] = None
        self._grpc_report_trace_batch: Optional[Callable[..., Any]] = None
        self._next_send_time = 0.0
        self._ownership_lock = threading.Lock()
        self._owned_tokens = 0
        self._carry: Optional[_PendingRequest] = None
        self._producer_id = config.producer_id or (
            f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}"
        )
        if config.enabled:
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
            self._grpc_report_trace_batch = self._grpc_channel.unary_unary(
                "/kv_cache_manager.proto.optimizer.OptimizerEventStreamService/ReportTraceBatch",
                request_serializer=kvcm_optimizer_ingest_minimal_pb2.TraceObservationBatchRequest.SerializeToString,
                response_deserializer=kvcm_optimizer_ingest_minimal_pb2.TraceObservationBatchResponse.FromString,
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
        if self._grpc_channel is not None:
            self._grpc_channel.close()

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

    def _send_batch(self, batch: list[_PendingRequest]) -> int:
        assert self._grpc_report_trace_batch is not None
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

    @staticmethod
    def _result_label(exc: Exception) -> str:
        message = str(exc)
        if message.startswith("kvcm_"):
            return message
        if isinstance(exc, TimeoutError):
            return "timeout"
        return "transport_error"
