"""Synchronous Python client for isolated, exact-size KVMeta objects.

The native :class:`KvMetaObjectClient` exposes caller-owned buffers.  This
module adds a small, framework-independent Python layer that validates a whole
operation, converts contiguous CPU/CUDA tensors to native IOVs, and splits a
logical operation into service-sized requests.  Importing the module does not
load the native extension; the extension is imported only when a client is
constructed.
"""

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
# Must match kKvMetaObjectClientApiVersion in the native public header.  This is
# an API/ABI capability marker, not the wheel package version.
KV_META_OBJECT_API_VERSION = 1

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
    # Snapshot once so validation, native buffers, and batch boundaries all
    # describe the same caller input even if a mutable list is later changed.
    return tuple(values)


class KvMetaObjectMemory(str, Enum):
    """Memory kind of one caller-owned exact-size object buffer."""

    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class KvMetaObjectBuffer:
    """A contiguous caller-owned buffer stored under one exact KVCM key.

    ``owner`` is retained through every synchronous native call.  Raw-pointer
    users remain responsible for keeping the underlying allocation valid.
    """

    key: str
    pointer: int
    nbytes: int
    memory: KvMetaObjectMemory = KvMetaObjectMemory.CPU
    owner: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _bounded_text(self.key, "KVMeta object key", KV_META_MAX_KEY_BYTES)
        _positive_int(self.pointer, "KVMeta object pointer", _MAX_UINT64)
        _positive_int(self.nbytes, "KVMeta object size", KV_META_MAX_OBJECT_BYTES)
        # Native backends commonly form an exclusive end address.  Require it
        # to remain representable instead of relying on pointer wraparound.
        if self.nbytes > _MAX_UINT64 - self.pointer:
            raise ValueError("KVMeta object buffer address range exceeds uint64")
        try:
            memory = KvMetaObjectMemory(self.memory)
        except (TypeError, ValueError) as error:
            raise ValueError("KVMeta object memory must be 'cpu' or 'gpu'") from error
        object.__setattr__(self, "memory", memory)

    @classmethod
    def from_tensor(cls, key: str, tensor: Any) -> "KvMetaObjectBuffer":
        """Describe a contiguous CPU/CUDA tensor without importing torch."""

        is_contiguous = getattr(tensor, "is_contiguous", None)
        if not callable(is_contiguous) or not is_contiguous():
            raise ValueError("KVMeta tensor must be contiguous")

        data_ptr = getattr(tensor, "data_ptr", None)
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if not callable(data_ptr) or not callable(numel) or not callable(element_size):
            raise TypeError("KVMeta tensor lacks the required tensor interface")

        device_type = getattr(getattr(tensor, "device", None), "type", None)
        if device_type == "cpu":
            memory = KvMetaObjectMemory.CPU
        elif device_type == "cuda":
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

        return cls(
            key=key,
            pointer=data_ptr(),
            nbytes=count * width,
            memory=memory,
            owner=tensor,
        )


@dataclass(frozen=True)
class KvMetaObjectClientConfig:
    """Configuration for the isolated KVMeta object client."""

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
            raise TypeError(
                "KVMeta addresses must be a sequence of endpoints"
            ) from error
        if not addresses:
            raise ValueError("KVMeta addresses must not be empty")
        if len(addresses) > KV_META_MAX_ADDRESSES:
            raise ValueError(
                f"KVMeta addresses exceed {KV_META_MAX_ADDRESSES} endpoints"
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
        if (
            not isinstance(self.transfer_client_config, str)
            or not self.transfer_client_config
        ):
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
    if isinstance(code, str) and code.isidentifier() and len(code) <= 128:
        return code
    return type(code).__name__


class KvMetaObjectClientError(RuntimeError):
    """A native object operation failed in one service-sized batch."""

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
        failed_batches: int = 1,
    ) -> None:
        self.operation = operation
        self.code = code
        self.unknown_outcome = unknown_outcome
        self.batch_index = batch_index
        self.batch_count = batch_count
        self.batch_start = batch_start
        self.batch_size = batch_size
        self.completed_items = completed_items
        self.failed_batches = failed_batches
        uncertainty = (
            "; mutation outcome is unknown and must not be retried blindly"
            if unknown_outcome
            else ""
        )
        super().__init__(
            f"KVCM {operation} failed with client error {_safe_code_label(code)} "
            f"in service batch {batch_index + 1}/{batch_count}{uncertainty}"
        )


def _load_kvcm_pybind():
    try:
        from kv_cache_manager.client.pybind import kvcm_py_client
    except Exception as error:
        raise ImportError(
            "KvMetaObjectClient requires the kvcm_py_client wheel with "
            "KVMeta object support"
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
        has_required_types = all(hasattr(pybind, name) for name in required)
        native_version = getattr(pybind, "KV_META_OBJECT_API_VERSION", None)
    except Exception:
        raise ImportError(
            "installed kvcm_py_client cannot be inspected safely"
        ) from None
    if not has_required_types:
        raise ImportError(
            "installed kvcm_py_client does not export the KVMeta object API"
        )
    if (
        type(native_version) is not int
        or native_version != KV_META_OBJECT_API_VERSION
    ):
        raise ImportError(
            "installed kvcm_py_client has an incompatible KVMeta object API version"
        )
    try:
        required_members = (
            (pybind.ClientErrorCode, ("ER_OK", "ER_INVALID_GRPCSTATUS")),
            (pybind.MemoryType, ("CPU", "GPU")),
            (pybind.RoleType, ("WORKER",)),
        )
        has_required_members = not any(
            not all(hasattr(enum_type, name) for name in names)
            for enum_type, names in required_members
        )
        has_factory = callable(getattr(pybind.KvMetaObjectClient, "Create", None))
    except Exception:
        raise ImportError(
            "installed kvcm_py_client cannot be inspected safely"
        ) from None
    if not has_required_members or not has_factory:
        raise ImportError(
            "installed kvcm_py_client exports an incomplete KVMeta object API"
        )


def _matches_native_code(pybind: Any, code: Any, name: str, raw_value: int) -> bool:
    # bool, floats, and foreign IntEnums compare equal to integer/native enum
    # values.  Only the documented raw integer or exact native enum type is
    # accepted, so malformed bindings cannot turn a failed write into success.
    if type(code) is int:
        return code == raw_value
    try:
        expected = getattr(getattr(pybind, "ClientErrorCode", object()), name, None)
        return (
            expected is not None
            and type(code) is type(expected)
            and code == expected
        )
    except Exception:
        return False


def _is_ok_code(pybind: Any, code: Any) -> bool:
    return _matches_native_code(pybind, code, "ER_OK", 0)


def _is_ambiguous_code(pybind: Any, code: Any) -> bool:
    return _matches_native_code(pybind, code, "ER_INVALID_GRPCSTATUS", 2)


def _is_known_native_code(pybind: Any, code: Any) -> bool:
    """Return whether *code* is a documented member of the native error enum."""

    try:
        members = tuple(pybind.ClientErrorCode.__members__.values())
        if type(code) is int:
            return any(
                type(getattr(member, "value", None)) is int
                and member.value == code
                for member in members
            )
        return any(type(code) is type(member) and code == member for member in members)
    except Exception:  # A malformed binding must fail closed after a mutation.
        return False


def _mutation_outcome_is_unknown(pybind: Any, code: Any) -> bool:
    # A transport-status failure is explicitly ambiguous.  A malformed or
    # unknown return value is equally unsafe: treating it as a confirmed
    # rejection could make an upper layer retry an already-committed mutation.
    return _is_ambiguous_code(pybind, code) or not _is_known_native_code(pybind, code)


def _best_effort_close(client: Any, phase: str) -> None:
    if client is None:
        return
    for method_name in ("close", "Close"):
        try:
            close = getattr(client, method_name, None)
        except Exception as error:
            _LOGGER.warning(
                "KVCM client cleanup hook lookup failed during %s "
                "(exception_type=%s)",
                phase,
                type(error).__name__,
            )
            continue
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
    """Thread-safe synchronous client for variable-size opaque objects.

    A logical Python operation is validated in full and then split by the
    native service limits (64 objects and 4 GiB per request).  Mutations are
    never automatically retried or rolled back here: keys may pre-exist, and a
    transport error can have an unknown commit outcome.
    """

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
        self._lock = threading.Lock()
        self._closed = False
        self._registration_owner = registration_owner
        # Assign an injected client before validating the binding module so an
        # initialization failure cannot leak an already-created native client.
        self._client = _object_client
        self._pybind = _pybind_module
        try:
            if self._pybind is None:
                self._pybind = _load_kvcm_pybind()
            _validate_pybind_module(self._pybind)
            if self._client is None:
                self._client = self._create_client()
            if not all(
                callable(getattr(self._client, method, None))
                for method in ("SaveObjects", "LoadObjects", "Remove")
            ):
                raise TypeError(
                    "native KVMeta object client is missing SaveObjects, "
                    "LoadObjects, or Remove"
                )
        except Exception:
            client, self._client = self._client, None
            self._closed = True
            try:
                _best_effort_close(client, "initialization rollback")
            finally:
                # Drop the native object before its registered-memory owner.
                client = None
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
            if self.config.memory_fd >= 0 and not hasattr(span, "fd"):
                raise ImportError(
                    "installed kvcm_py_client cannot register shared memory by fd"
                )
            if hasattr(span, "fd"):
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

        client = None
        try:
            code, client = self._pybind.KvMetaObjectClient.Create(
                self._generated_trace_id("init"), native_config
            )
        except Exception as error:
            raise KvMetaObjectClientError("init", type(error).__name__) from error
        if client is None or not _is_ok_code(self._pybind, code):
            _best_effort_close(client, "failed native creation")
            client = None
            raise KvMetaObjectClientError("init", code)
        return client

    def _check_open_locked(self) -> None:
        if self._closed or self._client is None:
            raise RuntimeError("KvMetaObjectClient is closed")

    def _validate_buffers(self, objects: Any) -> Tuple[KvMetaObjectBuffer, ...]:
        materialized = _snapshot_sequence(objects, "KVMeta object buffers")
        if not materialized:
            raise ValueError("KVMeta object operation must not be empty")
        keys = set()
        for index, obj in enumerate(materialized):
            if not isinstance(obj, KvMetaObjectBuffer):
                raise TypeError(f"objects[{index}] must be a KvMetaObjectBuffer")
            if obj.nbytes > self.config.max_object_bytes:
                raise ValueError(
                    f"objects[{index}] exceeds configured max_object_bytes "
                    f"({obj.nbytes} > {self.config.max_object_bytes})"
                )
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
            # Each validated object is <= 1 GiB, so one must always fit in the
            # 4-GiB service request. Keep this guard fail-closed if limits drift.
            if end == begin:
                raise ValueError("KVMeta object cannot fit in a service batch")
            batches.append((begin, end))
            begin = end
        return batches

    def _native_buffers(
        self, objects: Tuple[KvMetaObjectBuffer, ...]
    ) -> Tuple[Any, ...]:
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
        self,
        operation: str,
        objects: Any,
        trace_id: Optional[str],
    ) -> None:
        materialized = self._validate_buffers(objects)
        batches = self._partition(materialized)
        keys = tuple(obj.key for obj in materialized)
        sizes = tuple(obj.nbytes for obj in materialized)
        native_buffers = self._native_buffers(materialized)
        resolved_trace_id = self._resolve_trace_id(operation, trace_id)
        method_name = "SaveObjects" if operation == "save" else "LoadObjects"

        with self._lock:
            self._check_open_locked()
            for batch_index, (begin, end) in enumerate(batches):
                batch_trace = self._batch_trace_id(
                    resolved_trace_id, batch_index, len(batches)
                )
                try:
                    # Resolve the operation inside the exception boundary. A
                    # malformed/dynamically proxied binding can throw from
                    # attribute lookup even though it passed initialization
                    # inspection; such provider details must not escape.
                    native_method = getattr(self._client, method_name)
                    code = native_method(
                        batch_trace,
                        list(keys[begin:end]),
                        list(sizes[begin:end]),
                        list(native_buffers[begin:end]),
                    )
                except KvMetaObjectClientError:
                    raise
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
    def _buffers_from_tensors(
        keys: Any, tensors: Any
    ) -> Tuple[KvMetaObjectBuffer, ...]:
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
        """Save contiguous CPU/CUDA tensors under exact keys.

        This method intentionally matches RTP-LLM's multimodal writer protocol.
        Framework-specific stream synchronization remains the caller's job.
        """

        self.save_buffers(self._buffers_from_tensors(keys, tensors), trace_id=trace_id)

    def load(
        self,
        keys: Sequence[str],
        tensors: Sequence[Any],
        *,
        trace_id: Optional[str] = None,
    ) -> None:
        """Load exact objects into preallocated contiguous CPU/CUDA tensors."""

        self.load_buffers(self._buffers_from_tensors(keys, tensors), trace_id=trace_id)

    def save_tensors(
        self,
        keys: Sequence[str],
        tensors: Sequence[Any],
        *,
        trace_id: Optional[str] = None,
    ) -> None:
        """Explicit alias for :meth:`save`."""

        self.save(keys, tensors, trace_id=trace_id)

    def load_tensors(
        self,
        keys: Sequence[str],
        tensors: Sequence[Any],
        *,
        trace_id: Optional[str] = None,
    ) -> None:
        """Explicit alias for :meth:`load`."""

        self.load(keys, tensors, trace_id=trace_id)

    def save_buffers(
        self,
        objects: Sequence[KvMetaObjectBuffer],
        *,
        trace_id: Optional[str] = None,
    ) -> None:
        """Save raw exact-size buffers without automatic mutation retries."""

        self._invoke_buffers("save", objects, trace_id)

    def load_buffers(
        self,
        objects: Sequence[KvMetaObjectBuffer],
        *,
        trace_id: Optional[str] = None,
    ) -> None:
        """Load raw exact-size buffers into caller-owned memory."""

        self._invoke_buffers("load", objects, trace_id)

    def remove(self, keys: Sequence[str], *, trace_id: Optional[str] = None) -> None:
        """Remove exact keys, attempting every service batch once.

        Remove is metadata-idempotent for missing keys, but an ambiguous
        transport result is not retried inside this client because a new
        generation could appear between attempts.
        """

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
        resolved_trace_id = self._resolve_trace_id("remove", trace_id)

        first_failure = None
        first_cause = None
        failed_batches = 0
        confirmed_items = 0
        any_unknown = False
        with self._lock:
            self._check_open_locked()
            for batch_index, (begin, end) in enumerate(batches):
                code = None
                cause = None
                unknown = False
                try:
                    code = self._client.Remove(
                        self._batch_trace_id(
                            resolved_trace_id, batch_index, len(batches)
                        ),
                        list(materialized[begin:end]),
                    )
                    failed = not _is_ok_code(self._pybind, code)
                    unknown = failed and _mutation_outcome_is_unknown(
                        self._pybind, code
                    )
                except KvMetaObjectClientError:
                    raise
                except Exception as error:
                    code = type(error).__name__
                    cause = error
                    failed = True
                    unknown = True

                if not failed:
                    confirmed_items += end - begin
                    continue
                failed_batches += 1
                any_unknown = any_unknown or unknown
                if first_failure is None:
                    first_failure = (code, batch_index, begin, end)
                    first_cause = cause

        if first_failure is not None:
            code, batch_index, begin, end = first_failure
            error = KvMetaObjectClientError(
                "remove",
                code,
                unknown_outcome=any_unknown,
                batch_index=batch_index,
                batch_count=len(batches),
                batch_start=begin,
                batch_size=end - begin,
                completed_items=confirmed_items,
                failed_batches=failed_batches,
            )
            if first_cause is not None:
                raise error from first_cause
            raise error

    def close(self) -> None:
        """Wait for any in-flight call and release the native client once."""

        with self._lock:
            if self._closed:
                return
            client = self._client
            self._client = None
            self._closed = True
            try:
                for method_name in ("close", "Close"):
                    close = getattr(client, method_name, None)
                    if callable(close):
                        close()
                        break
            finally:
                # Ensure native destruction happens while registered memory is
                # still owned, including bindings without an explicit close.
                client = None
                self._registration_owner = None

    def __enter__(self) -> "KvMetaObjectClient":
        with self._lock:
            self._check_open_locked()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
            return
        try:
            self.close()
        except Exception as error:
            # A cleanup failure must not replace the exception that caused the
            # with-body to unwind.  Keep provider text out of the warning.
            _LOGGER.warning(
                "KVCM client close failed while preserving an active exception "
                "(exception_type=%s)",
                type(error).__name__,
            )
