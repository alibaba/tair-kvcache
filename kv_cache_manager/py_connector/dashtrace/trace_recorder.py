"""Bounded asynchronous JSONL trace recorder for DashTrace shadow traffic."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from kv_cache_manager.py_connector.dashtrace.observed_request import ObservedRequest

try:
    import orjson as _orjson
except ImportError:  # Keep the wheel usable in minimal development environments.
    _orjson = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TraceRecorderConfig:
    enabled: bool = True
    directory: str = "/home/admin/dashtrace/trace"
    queue_capacity: int = 8192
    segment_bytes: int = 256 * 1024 * 1024
    max_disk_bytes: int = 20 * 1024 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "TraceRecorderConfig":
        return cls(
            enabled=_env_bool("DASHTRACE_RECORD_ENABLED", True),
            directory=os.environ.get(
                "DASHTRACE_TRACE_DIR", "/home/admin/dashtrace/trace"
            ),
            queue_capacity=int(os.environ.get("DASHTRACE_TRACE_QUEUE_CAPACITY", 8192)),
            segment_bytes=int(
                os.environ.get("DASHTRACE_TRACE_SEGMENT_BYTES", 256 * 1024 * 1024)
            ),
            max_disk_bytes=int(
                os.environ.get("DASHTRACE_TRACE_MAX_DISK_BYTES", 20 * 1024**3)
            ),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.directory:
            raise ValueError("DASHTRACE_TRACE_DIR must not be empty")
        if self.queue_capacity <= 0:
            raise ValueError("DASHTRACE_TRACE_QUEUE_CAPACITY must be positive")
        if self.segment_bytes <= 0:
            raise ValueError("DASHTRACE_TRACE_SEGMENT_BYTES must be positive")
        if self.max_disk_bytes < self.segment_bytes:
            raise ValueError(
                "DASHTRACE_TRACE_MAX_DISK_BYTES must be at least one segment"
            )


class TraceRecorder:
    """Non-blocking trace recorder with bounded local-disk retention."""

    _STOP = object()

    def __init__(self, config: TraceRecorderConfig):
        config.validate()
        self._config = config
        self._queue: queue.Queue[object] = queue.Queue(config.queue_capacity)
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._file = None
        self._file_size = 0
        self._sequence = 0
        self.accepted = 0
        self.dropped = 0
        if config.enabled:
            Path(config.directory).mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(
                target=self._run, name="dashtrace-recorder", daemon=True
            )
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def record(
        self, trace_id: str, instance_id: str, token_ids: Sequence[int]
    ) -> bool:
        compact = (
            token_ids
            if isinstance(token_ids, array) and token_ids.typecode == "q"
            else array("q", (int(token) for token in token_ids))
        )
        return self.submit(
            ObservedRequest(
                sequence=0,
                timestamp_ns=time.time_ns(),
                trace_id=str(trace_id),
                instance_id=str(instance_id),
                token_ids=compact,
            )
        )

    def submit(self, observation: ObservedRequest) -> bool:
        """Queue an already-parsed observation without copying its token array."""
        if (
            not self._config.enabled
            or self._closed.is_set()
            or not observation.token_ids
        ):
            return False
        item = observation
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1
            return False
        self.accepted += 1
        return True

    def close(self, timeout: float = 5.0) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout)
        self._close_file()

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._closed.is_set():
                    break
                continue
            if item is self._STOP:
                break
            assert isinstance(item, ObservedRequest)
            self._write(item)
            self._queue.task_done()

    def _write(self, item: ObservedRequest) -> None:
        record = {
            "sequence": item.sequence,
            "timestamp_ns": item.timestamp_ns,
            "trace_id": item.trace_id,
            "instance_id": item.instance_id,
            "token_ids": list(item.token_ids),
        }
        if _orjson is not None:
            encoded = _orjson.dumps(record, option=_orjson.OPT_APPEND_NEWLINE)
        else:
            encoded = (json.dumps(record, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        if self._file is None or self._file_size + len(encoded) > self._config.segment_bytes:
            self._rotate()
        self._file.write(encoded)
        self._file_size += len(encoded)

    def _rotate(self) -> None:
        self._close_file()
        directory = Path(self._config.directory)
        while True:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            path = directory / f"trace-{stamp}-{self._sequence:06d}.jsonl"
            self._sequence += 1
            try:
                self._file = path.open("xb")
                break
            except FileExistsError:
                continue
        self._file_size = 0
        self._enforce_disk_limit()

    def _enforce_disk_limit(self) -> None:
        files = sorted(Path(self._config.directory).glob("trace-*.jsonl"))
        total = sum(path.stat().st_size for path in files)
        for path in files:
            if total <= self._config.max_disk_bytes:
                break
            if self._file is not None and path.name == Path(self._file.name).name:
                continue
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size

    def _close_file(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
