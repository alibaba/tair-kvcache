"""Small Python facade for KVCM's exact-size object client."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import islice
from typing import Any, List, Optional, Tuple

KV_META_MAX_BATCH_ITEMS = 64
KV_META_MAX_ADDRESSES = 64
KV_META_MAX_ADDRESS_BYTES = 1024
KV_META_MAX_KEY_BYTES = 512
KV_META_MAX_INSTANCE_ID_BYTES = 512
KV_META_MAX_INSTANCE_GROUP_BYTES = 512
KV_META_MAX_USER_DATA_BYTES = 64 * 1024
KV_META_MAX_OBJECT_BYTES = 1024 * 1024 * 1024
KV_META_MAX_BATCH_BYTES = 4 * 1024 * 1024 * 1024
KV_META_MAX_CALL_TIMEOUT_MS = 600_000
KV_META_MAX_WRITE_TIMEOUT_SECONDS = 1800
KV_META_OBJECT_API_VERSION = 2

_MAX_UINT64 = (1 << 64) - 1
_MAX_INT32 = (1 << 31) - 1
_LOGGER = logging.getLogger(__name__)


def _utf8_size(value: str, description: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError(f"{description} must be valid UTF-8") from error


def _bounded_text(value: Any, description: str, max_bytes: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty string")
    if _utf8_size(value, description) > max_bytes:
        raise ValueError(f"{description} exceeds {max_bytes} UTF-8 bytes")
    return value


def _positive_int(value: Any, description: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{description} must be an integer")
    if not 0 < value <= maximum:
        raise ValueError(f"{description} must be in [1, {maximum}]")
    return value


def _snapshot_sequence(values: Any, description: str) -> Tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{description} must be a sequence")
    return tuple(values)


class KvMetaObjectMemory(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class KvMetaObjectBuffer:
    """One caller-owned contiguous object buffer."""

    key: str
    pointer: int
    nbytes: int
    memory: KvMetaObjectMemory = KvMetaObjectMemory.CPU
    owner: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _bounded_text(self.key, "KVMeta object key", KV_META_MAX_KEY_BYTES)
        _positive_int(self.pointer, "KVMeta object pointer", _MAX_UINT64)
        _positive_int(self.nbytes, "KVMeta object size", KV_META_MAX_OBJECT_BYTES)
        if self.nbytes > _MAX_UINT64 - self.pointer:
            raise ValueError("KVMeta object buffer address range exceeds uint64")
        try:
            memory = KvMetaObjectMemory(self.memory)
        except (TypeError, ValueError) as error:
            raise ValueError("KVMeta object memory must be 'cpu' or 'gpu'") from error
        object.__setattr__(self, "memory", memory)

    @classmethod
    def from_tensor(cls, key: str, tensor: Any) -> "KvMetaObjectBuffer":
        """Describe a contiguous CPU/CUDA/MUSA tensor without importing torch."""

        is_contiguous = getattr(tensor, "is_contiguous", None)
        data_ptr = getattr(tensor, "data_ptr", None)
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if not callable(is_contiguous) or not is_contiguous():
            raise ValueError("KVMeta tensor must be contiguous")
        if not callable(data_ptr) or not callable(numel) or not callable(element_size):
            raise TypeError("KVMeta tensor lacks the required tensor interface")

        device_type = getattr(getattr(tensor, "device", None), "type", None)
        if device_type == "cpu":
            memory = KvMetaObjectMemory.CPU
        elif device_type in ("cuda", "musa"):
            memory = KvMetaObjectMemory.GPU
        else:
            raise ValueError(f"KVMeta tensor device is unsupported: {device_type!r}")

        count = numel()
        width = element_size()
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("KVMeta tensor numel must be a non-negative integer")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("KVMeta tensor element size must be a positive integer")
        if count > _MAX_UINT64 // width:
            raise ValueError("KVMeta tensor byte size exceeds uint64")
        return cls(key, data_ptr(), count * width, memory, tensor)


@dataclass(frozen=True)
class KvMetaObjectClientConfig:
    addresses: Sequence[str]
    instance_id: str
    instance_group: str
    transfer_client_config: str = field(repr=False)
    user_data: str = field(default="", repr=False)
    call_timeout_ms: int = 3000
    write_timeout_seconds: int = 30
    max_object_bytes: int = KV_META_MAX_OBJECT_BYTES
    memory_base: int = 0
    memory_size: int = 0
    memory_fd: int = -1

    def __post_init__(self) -> None:
        if isinstance(self.addresses, (str, bytes)):
            raise TypeError("KVMeta addresses must be a sequence of endpoints")
        try:
            addresses = tuple(islice(iter(self.addresses), KV_META_MAX_ADDRESSES + 1))
        except TypeError as error:
            raise TypeError("KVMeta addresses must be a sequence of endpoints") from error
        if not addresses or len(addresses) > KV_META_MAX_ADDRESSES:
            raise ValueError(
                f"KVMeta addresses must contain 1 to {KV_META_MAX_ADDRESSES} endpoints"
            )
        for address in addresses:
            _bounded_text(address, "KVMeta address", KV_META_MAX_ADDRESS_BYTES)
        if len(set(addresses)) != len(addresses):
            raise ValueError("KVMeta addresses must be unique")
        object.__setattr__(self, "addresses", addresses)

        _bounded_text(
            self.instance_id, "KVMeta instance_id", KV_META_MAX_INSTANCE_ID_BYTES
        )
        _bounded_text(
            self.instance_group,
            "KVMeta instance_group",
            KV_META_MAX_INSTANCE_GROUP_BYTES,
        )
        if not isinstance(self.user_data, str):
            raise TypeError("KVMeta user_data must be a string")
        if _utf8_size(self.user_data, "KVMeta user_data") > KV_META_MAX_USER_DATA_BYTES:
            raise ValueError(
                f"KVMeta user_data exceeds {KV_META_MAX_USER_DATA_BYTES} UTF-8 bytes"
            )
        if not isinstance(self.transfer_client_config, str) or not self.transfer_client_config:
            raise ValueError("KVMeta transfer_client_config must be a non-empty string")
        _utf8_size(self.transfer_client_config, "KVMeta transfer_client_config")

        _positive_int(
            self.call_timeout_ms,
            "KVMeta call_timeout_ms",
            KV_META_MAX_CALL_TIMEOUT_MS,
        )
        _positive_int(
            self.write_timeout_seconds,
            "KVMeta write_timeout_seconds",
            KV_META_MAX_WRITE_TIMEOUT_SECONDS,
        )
        _positive_int(
            self.max_object_bytes,
            "KVMeta max_object_bytes",
            KV_META_MAX_OBJECT_BYTES,
        )

        for value, description, minimum, maximum in (
            (self.memory_base, "KVMeta memory_base", 0, _MAX_UINT64),
            (self.memory_size, "KVMeta memory_size", 0, _MAX_UINT64),
            (self.memory_fd, "KVMeta memory_fd", -1, _MAX_INT32),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
                or value > maximum
            ):
                raise ValueError(
                    f"{description} must be an integer in [{minimum}, {maximum}]"
                )
        if (self.memory_base == 0) != (self.memory_size == 0):
            raise ValueError(
                "KVMeta memory_base and memory_size must both be zero or positive"
            )
        if self.memory_base > 0 and self.memory_size > _MAX_UINT64 - self.memory_base:
            raise ValueError("KVMeta registered memory address range exceeds uint64")
        if self.memory_fd >= 0 and self.memory_base == 0:
            raise ValueError("KVMeta memory_fd requires a registered memory range")


def _safe_code_label(code: Any) -> str:
    if type(code) is int:
        return str(code)
    if isinstance(code, Enum):
        return f"{type(code).__name__}.{code.name}"
    return type(code).__name__


class KvMetaObjectClientError(RuntimeError):
    def __init__(
        self,
        operation: str,
        code: Any,
        *,
        unknown_outcome: bool = False,
        batch_index: int = 0,
        batch_count: int = 1,
        batch_start: int = 0,
        batch_size: int = 0,
        completed_items: int = 0,
    ) -> None:
        self.operation = operation
        self.code = code
        self.unknown_outcome = unknown_outcome
        self.batch_index = batch_index
        self.batch_count = batch_count
        self.batch_start = batch_start
        self.batch_size = batch_size
        self.completed_items = completed_items
        uncertainty = "; mutation outcome is unknown" if unknown_outcome else ""
        super().__init__(
            f"KVCM {operation} failed with client error {_safe_code_label(code)} "
            f"in service batch {batch_index + 1}/{batch_count}{uncertainty}"
        )


def _load_kvcm_pybind():
    try:
        from kv_cache_manager.client.pybind import kvcm_py_client
    except Exception as error:
        raise ImportError(
            "KvMetaObjectClient requires the kvcm_py_client wheel with KVMeta support"
        ) from error
    return kvcm_py_client


def _validate_pybind_module(pybind: Any) -> None:
    required = (
        "ClientErrorCode",
        "MemoryType",
        "RoleType",
        "KvMetaClientConfig",
        "KvMetaObjectClientConfig",
        "KvMetaObjectClient",
        "InitParams",
        "RegistSpan",
        "Iov",
        "BlockBuffer",
    )
    try:
        version = getattr(pybind, "KV_META_OBJECT_API_VERSION", None)
        complete = all(hasattr(pybind, name) for name in required)
        factory = (
            getattr(pybind.KvMetaObjectClient, "Create", None) if complete else None
        )
    except Exception as error:
        raise ImportError("installed kvcm_py_client cannot be inspected") from error
    if version != KV_META_OBJECT_API_VERSION:
        raise ImportError("installed kvcm_py_client has an incompatible object API")
    if not complete:
        raise ImportError("installed kvcm_py_client exports an incomplete object API")
    if not callable(factory):
        raise ImportError("installed kvcm_py_client has no object client factory")


def _matches_native_code(pybind: Any, code: Any, name: str, raw_value: int) -> bool:
    if type(code) is int:
        return code == raw_value
    try:
        expected = getattr(pybind.ClientErrorCode, name, None)
    except Exception:
        return False
    return expected is not None and type(code) is type(expected) and code == expected


def _is_ok_code(pybind: Any, code: Any) -> bool:
    return _matches_native_code(pybind, code, "ER_OK", 0)


def _mutation_outcome_is_unknown(pybind: Any, code: Any) -> bool:
    try:
        if _matches_native_code(pybind, code, "ER_INVALID_GRPCSTATUS", 2):
            return True
        members = tuple(pybind.ClientErrorCode.__members__.values())
    except Exception:
        return True
    if type(code) is int:
        return not any(getattr(member, "value", None) == code for member in members)
    return not any(type(code) is type(member) and code == member for member in members)


def _best_effort_close(client: Any, phase: str) -> None:
    if client is None:
        return
    for method_name in ("Close", "close"):
        close = getattr(client, method_name, None)
        if callable(close):
            try:
                close()
            except Exception as error:
                _LOGGER.warning(
                    "KVCM client cleanup failed during %s (exception_type=%s)",
                    phase,
                    type(error).__name__,
                )
            return


class KvMetaObjectClient:
    """Thread-safe synchronous client for variable-size opaque objects."""

    def __init__(
        self,
        config: KvMetaObjectClientConfig,
        *,
        registration_owner: Any = None,
        _object_client: Any = None,
        _pybind_module: Any = None,
    ) -> None:
        if not isinstance(config, KvMetaObjectClientConfig):
            raise TypeError("config must be a KvMetaObjectClientConfig")
        self.config = config
        # Python only serializes close itself. The native client owns operation
        # lifetime and drains calls already in flight from Close().
        self._close_condition = threading.Condition()
        self._closing = False
        self._closed = False
        self._registration_owner = registration_owner
        self._client = _object_client
        self._pybind = _pybind_module
        try:
            if self._pybind is None:
                self._pybind = _load_kvcm_pybind()
            _validate_pybind_module(self._pybind)
            if self._client is None:
                self._client = self._create_client()
            if not all(
                callable(getattr(self._client, name, None))
                for name in ("SaveObjects", "LoadObjects", "Remove", "Close")
            ):
                raise TypeError("native KVMeta object client is incomplete")
        except Exception:
            client, self._client = self._client, None
            _best_effort_close(client, "initialization rollback")
            self._registration_owner = None
            raise

    @staticmethod
    def _generated_trace_id(operation: str) -> str:
        return f"kvcm-py-{operation}-{uuid.uuid4().hex}"

    @classmethod
    def _resolve_trace_id(cls, operation: str, trace_id: Optional[str]) -> str:
        if trace_id is None:
            return cls._generated_trace_id(operation)
        if not isinstance(trace_id, str):
            raise TypeError("KVMeta trace_id must be a string")
        if not trace_id:
            raise ValueError("KVMeta trace_id must not be empty")
        _utf8_size(trace_id, "KVMeta trace_id")
        return trace_id

    @staticmethod
    def _batch_trace_id(base: str, index: int, count: int) -> str:
        return base if count == 1 else f"{base}:batch-{index + 1}-of-{count}"

    def _create_client(self):
        metadata = self._pybind.KvMetaClientConfig()
        metadata.addresses = list(self.config.addresses)
        metadata.instance_id = self.config.instance_id
        metadata.call_timeout_ms = self.config.call_timeout_ms

        init_params = self._pybind.InitParams()
        init_params.role_type = self._pybind.RoleType.WORKER
        init_params.self_location_spec_name = "value"
        if self.config.memory_base > 0:
            span = self._pybind.RegistSpan()
            span.base = self.config.memory_base
            span.size = self.config.memory_size
            if self.config.memory_fd >= 0:
                span.fd = self.config.memory_fd
            if hasattr(span, "owner"):
                span.owner = self._registration_owner
            init_params.regist_span = span

        native_config = self._pybind.KvMetaObjectClientConfig()
        native_config.metadata = metadata
        native_config.instance_group = self.config.instance_group
        native_config.user_data = self.config.user_data
        native_config.transfer_client_config = self.config.transfer_client_config
        native_config.transfer_init_params = init_params
        native_config.max_object_bytes = self.config.max_object_bytes
        native_config.write_timeout_seconds = self.config.write_timeout_seconds

        try:
            code, client = self._pybind.KvMetaObjectClient.Create(
                self._generated_trace_id("init"), native_config
            )
        except Exception as error:
            raise KvMetaObjectClientError("init", type(error).__name__) from error
        if client is None or not _is_ok_code(self._pybind, code):
            _best_effort_close(client, "failed native creation")
            raise KvMetaObjectClientError("init", code)
        return client

    def _client_snapshot(self):
        with self._close_condition:
            if self._closing or self._closed or self._client is None:
                raise RuntimeError("KvMetaObjectClient is closed")
            return self._client

    def _validate_buffers(self, objects: Any) -> Tuple[KvMetaObjectBuffer, ...]:
        materialized = _snapshot_sequence(objects, "KVMeta object buffers")
        if not materialized:
            raise ValueError("KVMeta object operation must not be empty")
        keys = set()
        for index, obj in enumerate(materialized):
            if not isinstance(obj, KvMetaObjectBuffer):
                raise TypeError(f"objects[{index}] must be a KvMetaObjectBuffer")
            if obj.nbytes > self.config.max_object_bytes:
                raise ValueError(f"objects[{index}] exceeds configured max_object_bytes")
            if obj.key in keys:
                raise ValueError("KVMeta object keys must be unique")
            keys.add(obj.key)
        return materialized

    @staticmethod
    def _partition(objects: Tuple[KvMetaObjectBuffer, ...]) -> List[Tuple[int, int]]:
        batches = []
        begin = 0
        while begin < len(objects):
            end = begin
            batch_bytes = 0
            while (
                end < len(objects)
                and end - begin < KV_META_MAX_BATCH_ITEMS
                and objects[end].nbytes <= KV_META_MAX_BATCH_BYTES - batch_bytes
            ):
                batch_bytes += objects[end].nbytes
                end += 1
            if end == begin:
                raise ValueError("KVMeta object cannot fit in a service batch")
            batches.append((begin, end))
            begin = end
        return batches

    def _native_buffers(self, objects: Tuple[KvMetaObjectBuffer, ...]) -> Tuple[Any, ...]:
        buffers = []
        for obj in objects:
            iov = self._pybind.Iov()
            iov.type = (
                self._pybind.MemoryType.GPU
                if obj.memory == KvMetaObjectMemory.GPU
                else self._pybind.MemoryType.CPU
            )
            iov.base = obj.pointer
            iov.size = obj.nbytes
            iov.ignore = False
            block = self._pybind.BlockBuffer()
            block.iovs = [iov]
            buffers.append(block)
        return tuple(buffers)

    def _invoke_buffers(
        self, operation: str, objects: Any, trace_id: Optional[str]
    ) -> None:
        materialized = self._validate_buffers(objects)
        batches = self._partition(materialized)
        keys = tuple(obj.key for obj in materialized)
        sizes = tuple(obj.nbytes for obj in materialized)
        native_buffers = self._native_buffers(materialized)
        base_trace = self._resolve_trace_id(operation, trace_id)
        client = self._client_snapshot()
        try:
            method = client.SaveObjects if operation == "save" else client.LoadObjects
        except Exception as error:
            raise KvMetaObjectClientError(
                operation,
                type(error).__name__,
                unknown_outcome=operation == "save",
                batch_count=len(batches),
                batch_size=batches[0][1],
            ) from error

        for batch_index, (begin, end) in enumerate(batches):
            try:
                code = method(
                    self._batch_trace_id(base_trace, batch_index, len(batches)),
                    list(keys[begin:end]),
                    list(sizes[begin:end]),
                    list(native_buffers[begin:end]),
                )
            except Exception as error:
                raise KvMetaObjectClientError(
                    operation,
                    type(error).__name__,
                    unknown_outcome=operation == "save",
                    batch_index=batch_index,
                    batch_count=len(batches),
                    batch_start=begin,
                    batch_size=end - begin,
                    completed_items=begin,
                ) from error
            if not _is_ok_code(self._pybind, code):
                raise KvMetaObjectClientError(
                    operation,
                    code,
                    unknown_outcome=(
                        operation == "save"
                        and _mutation_outcome_is_unknown(self._pybind, code)
                    ),
                    batch_index=batch_index,
                    batch_count=len(batches),
                    batch_start=begin,
                    batch_size=end - begin,
                    completed_items=begin,
                )

    @staticmethod
    def _buffers_from_tensors(keys: Any, tensors: Any) -> Tuple[KvMetaObjectBuffer, ...]:
        materialized_keys = _snapshot_sequence(keys, "KVMeta keys")
        materialized_tensors = _snapshot_sequence(tensors, "KVMeta tensors")
        if len(materialized_keys) != len(materialized_tensors):
            raise ValueError("KVMeta keys and tensors length mismatch")
        return tuple(
            KvMetaObjectBuffer.from_tensor(key, tensor)
            for key, tensor in zip(materialized_keys, materialized_tensors)
        )

    def save(
        self,
        keys: Sequence[str],
        tensors: Sequence[Any],
        *,
        trace_id: Optional[str] = None,
    ) -> None:
        self.save_buffers(self._buffers_from_tensors(keys, tensors), trace_id=trace_id)

    def load(
        self,
        keys: Sequence[str],
        tensors: Sequence[Any],
        *,
        trace_id: Optional[str] = None,
    ) -> None:
        self.load_buffers(self._buffers_from_tensors(keys, tensors), trace_id=trace_id)

    def save_buffers(
        self,
        objects: Sequence[KvMetaObjectBuffer],
        *,
        trace_id: Optional[str] = None,
    ) -> None:
        self._invoke_buffers("save", objects, trace_id)

    def load_buffers(
        self,
        objects: Sequence[KvMetaObjectBuffer],
        *,
        trace_id: Optional[str] = None,
    ) -> None:
        self._invoke_buffers("load", objects, trace_id)

    def remove(self, keys: Sequence[str], *, trace_id: Optional[str] = None) -> None:
        materialized = _snapshot_sequence(keys, "KVMeta remove keys")
        if not materialized:
            return
        unique = set()
        for index, key in enumerate(materialized):
            _bounded_text(key, f"keys[{index}]", KV_META_MAX_KEY_BYTES)
            if key in unique:
                raise ValueError("KVMeta remove keys must be unique")
            unique.add(key)

        batches = [
            (begin, min(begin + KV_META_MAX_BATCH_ITEMS, len(materialized)))
            for begin in range(0, len(materialized), KV_META_MAX_BATCH_ITEMS)
        ]
        base_trace = self._resolve_trace_id("remove", trace_id)
        client = self._client_snapshot()
        completed = 0
        for batch_index, (begin, end) in enumerate(batches):
            try:
                code = client.Remove(
                    self._batch_trace_id(base_trace, batch_index, len(batches)),
                    list(materialized[begin:end]),
                )
            except Exception as error:
                raise KvMetaObjectClientError(
                    "remove",
                    type(error).__name__,
                    unknown_outcome=True,
                    batch_index=batch_index,
                    batch_count=len(batches),
                    batch_start=begin,
                    batch_size=end - begin,
                    completed_items=completed,
                ) from error
            if not _is_ok_code(self._pybind, code):
                raise KvMetaObjectClientError(
                    "remove",
                    code,
                    unknown_outcome=_mutation_outcome_is_unknown(self._pybind, code),
                    batch_index=batch_index,
                    batch_count=len(batches),
                    batch_start=begin,
                    batch_size=end - begin,
                    completed_items=completed,
                )
            completed += end - begin

    def close(self) -> None:
        """Close once; native Close drains operations already in flight."""

        with self._close_condition:
            while self._closing:
                self._close_condition.wait()
            if self._closed:
                return
            self._closing = True
            client = self._client
            self._client = None
        try:
            client.Close()
        finally:
            # Native state must be released before the registered-memory owner.
            client = None
            self._registration_owner = None
            with self._close_condition:
                self._closed = True
                self._closing = False
                self._close_condition.notify_all()

    def __enter__(self) -> "KvMetaObjectClient":
        self._client_snapshot()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
            return
        try:
            self.close()
        except Exception as error:
            _LOGGER.warning(
                "KVCM close failed while preserving an active exception "
                "(exception_type=%s)",
                type(error).__name__,
            )
