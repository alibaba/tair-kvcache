"""Extract request identity and token IDs, then fan out to trace and KVCM."""

from __future__ import annotations

import atexit
import logging
import struct
import threading
import time
from array import array
from collections.abc import Mapping, Sequence
from typing import Any

from kv_cache_manager.py_connector.dashtrace.kvcm_shadow_forwarder import (
    KVCMShadowForwarder,
    KVCMShadowForwarderConfig,
)
from kv_cache_manager.py_connector.dashtrace.observed_request import ObservedRequest
from kv_cache_manager.py_connector.dashtrace.trace_recorder import (
    TraceRecorder,
    TraceRecorderConfig,
)

_TOKEN_INPUT_NAMES = (
    "input_ids",
    "prompt_token_ids",
    "token_ids",
    "input_token_ids",
)

logger = logging.getLogger(__name__)


def _as_int_list(value: Any) -> list[int]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return []
    if isinstance(value, Sequence):
        try:
            return [int(item) for item in value]
        except (TypeError, ValueError):
            return []
    return []


def _tensor_values(tensor: Any, raw: Any = None) -> list[int]:
    contents = getattr(tensor, "contents", None) or tensor
    for attr in ("int64_contents", "int_contents", "uint64_contents"):
        values = _as_int_list(getattr(contents, attr, None))
        if values:
            return values

    if raw is None:
        raw = getattr(contents, "bytes_contents", None)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        try:
            raw = b"".join(raw)
        except TypeError:
            return []
    if not isinstance(raw, (bytes, bytearray)):
        return []
    datatype = str(getattr(tensor, "datatype", "")).upper()
    fmt = {"INT32": "i", "UINT32": "I", "INT64": "q", "UINT64": "Q"}.get(
        datatype
    )
    if fmt is None:
        return []
    size = struct.calcsize(fmt)
    if len(raw) % size:
        return []
    return list(struct.unpack(f"<{len(raw) // size}{fmt}", raw))


def extract_token_ids(request: Any) -> list[int]:
    """Extract already-tokenized input without loading a tokenizer."""
    get_input = getattr(request, "get_input", None)
    if callable(get_input):
        for name in _TOKEN_INPUT_NAMES:
            tensor = get_input(name)
            values = _tensor_values(tensor) if tensor is not None else []
            if values:
                return values

    inputs = getattr(request, "inputs", None) or []
    raw_inputs = getattr(request, "raw_input_contents", None) or []
    for index, tensor in enumerate(inputs):
        if getattr(tensor, "name", "") in _TOKEN_INPUT_NAMES:
            raw = raw_inputs[index] if index < len(raw_inputs) else None
            values = _tensor_values(tensor, raw)
            if values:
                return values

    parameters = getattr(request, "parameters", None)
    if isinstance(parameters, Mapping):
        for name in _TOKEN_INPUT_NAMES:
            values = _as_int_list(parameters.get(name))
            if values:
                return values
    return []


def extract_request_id(request: Any) -> str:
    for attr in ("id", "request_id", "trace_id"):
        value = getattr(request, attr, None)
        if value:
            return str(value)
    parameters = getattr(request, "parameters", None)
    if isinstance(parameters, Mapping):
        for name in ("request_id", "trace_id"):
            if parameters.get(name):
                return str(parameters[name])
    return "unknown"


class RequestObserver:
    """Fail-open fan-out; the inference request never waits for trace I/O."""

    def __init__(
        self,
        instance_id: str,
        recorder: TraceRecorder,
        forwarder: KVCMShadowForwarder,
    ):
        self._instance_id = instance_id
        self._recorder = recorder
        self._forwarder = forwarder
        self._sequence = 0
        self._sequence_lock = threading.Lock()

    def observe(self, request: Any) -> bool:
        token_ids = array("q", extract_token_ids(request))
        if not token_ids:
            return False
        request_id = extract_request_id(request)
        with self._sequence_lock:
            observation = ObservedRequest(
                sequence=self._sequence,
                timestamp_ns=time.time_ns(),
                trace_id=request_id,
                instance_id=self._instance_id,
                token_ids=token_ids,
            )
            self._sequence += 1
            try:
                recorded = self._recorder.submit(observation)
            except Exception as exc:  # noqa: BLE001 - observation is fail-open
                recorded = False
                logger.warning("DashTrace local recording failed open: %s", exc)
            try:
                reported = self._forwarder.submit_observation(observation)
            except Exception as exc:  # noqa: BLE001 - observation is fail-open
                reported = False
                logger.warning("DashTrace online MRC reporting failed open: %s", exc)
        return recorded or reported

    def close(self) -> None:
        try:
            self._recorder.close()
        finally:
            self._forwarder.close()


_observer: RequestObserver | None = None
_observer_lock = threading.Lock()


def get_request_observer() -> RequestObserver:
    global _observer
    if _observer is None:
        with _observer_lock:
            if _observer is None:
                forwarder_config = KVCMShadowForwarderConfig.from_env()
                forwarder_config.validate()
                recorder_config = TraceRecorderConfig.from_env()
                recorder_config.validate()
                if (
                    recorder_config.enabled or forwarder_config.enabled
                ) and not forwarder_config.instance_id:
                    raise ValueError(
                        "DASHTRACE_INSTANCE_ID must be set when recording or "
                        "online MRC reporting is enabled"
                    )
                observer = RequestObserver(
                    instance_id=forwarder_config.instance_id,
                    recorder=TraceRecorder(recorder_config),
                    forwarder=KVCMShadowForwarder(forwarder_config),
                )
                atexit.register(observer.close)
                _observer = observer
    return _observer
