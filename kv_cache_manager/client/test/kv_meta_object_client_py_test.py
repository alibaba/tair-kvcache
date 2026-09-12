import gc
import threading
import unittest
import weakref
from enum import IntEnum
from types import SimpleNamespace

from kv_cache_manager.client import (
    KV_META_OBJECT_API_VERSION,
    KvMetaObjectBuffer,
    KvMetaObjectClient,
    KvMetaObjectClientConfig,
    KvMetaObjectClientError,
    KvMetaObjectMemory,
)
from kv_cache_manager.client.kv_meta_object_client import (
    KV_META_MAX_BATCH_BYTES,
    KV_META_MAX_BATCH_ITEMS,
    KV_META_MAX_OBJECT_BYTES,
)


class _Code(IntEnum):
    ER_OK = 0
    ER_INVALID_GRPCSTATUS = 2
    ER_FAILED = 51


class _Memory(IntEnum):
    CPU = 0
    GPU = 1


class _Role(IntEnum):
    WORKER = 1


class _Value:
    pass


class _RegistSpan:
    def __init__(self):
        self.base = 0
        self.size = 0
        self.fd = -1
        self.owner = None


class _FakeClient:
    def __init__(self):
        self.calls = []
        self.results = {
            "SaveObjects": [],
            "LoadObjects": [],
            "Remove": [],
        }
        self.closed = False
        self.close_calls = 0

    def _result(self, method):
        results = self.results[method]
        return results.pop(0) if results else _Code.ER_OK

    def SaveObjects(self, trace_id, keys, sizes, buffers):
        self.calls.append(("SaveObjects", trace_id, keys, sizes, buffers))
        return self._result("SaveObjects")

    def LoadObjects(self, trace_id, keys, sizes, buffers):
        self.calls.append(("LoadObjects", trace_id, keys, sizes, buffers))
        return self._result("LoadObjects")

    def Remove(self, trace_id, keys):
        self.calls.append(("Remove", trace_id, keys))
        return self._result("Remove")

    def close(self):
        self.close_calls += 1
        self.closed = True


class _FakePybind:
    KV_META_OBJECT_API_VERSION = KV_META_OBJECT_API_VERSION
    ClientErrorCode = _Code
    MemoryType = _Memory
    RoleType = _Role
    KvMetaClientConfig = _Value
    KvMetaObjectClientConfig = _Value
    InitParams = _Value
    RegistSpan = _RegistSpan
    Iov = _Value
    BlockBuffer = _Value

    def __init__(self, client=None, create_code=_Code.ER_OK):
        self.client = client or _FakeClient()
        self.create_code = create_code
        self.create_calls = []
        self.KvMetaObjectClient = SimpleNamespace(Create=self._create)

    def _create(self, trace_id, config):
        self.create_calls.append((trace_id, config))
        return self.create_code, self.client


class _Tensor:
    def __init__(
        self,
        pointer=0x4000,
        count=12,
        width=2,
        device="cpu",
        contiguous=True,
    ):
        self._pointer = pointer
        self._count = count
        self._width = width
        self._contiguous = contiguous
        self.device = SimpleNamespace(type=device)

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._pointer

    def numel(self):
        return self._count

    def element_size(self):
        return self._width


def _config(**overrides):
    values = {
        "addresses": ("10.0.0.1:19001", "10.0.0.2:19001"),
        "instance_id": "rtp-emb-1",
        "instance_group": "epd-emb-only",
        "transfer_client_config": '{"block_size": 1}',
        "user_data": '{"owner": "rtp"}',
    }
    values.update(overrides)
    return KvMetaObjectClientConfig(**values)


def _client(config=None, native=None, pybind=None, registration_owner=None):
    native = native or _FakeClient()
    pybind = pybind or _FakePybind(native)
    return (
        KvMetaObjectClient(
            config or _config(),
            registration_owner=registration_owner,
            _object_client=native,
            _pybind_module=pybind,
        ),
        native,
        pybind,
    )


class KvMetaObjectClientConfigTest(unittest.TestCase):
    def test_config_normalizes_addresses_and_hides_sensitive_values(self):
        config = _config(addresses=["10.0.0.1:19001"])

        self.assertEqual(config.addresses, ("10.0.0.1:19001",))
        self.assertNotIn("block_size", repr(config))
        self.assertNotIn("owner", repr(config))

    def test_invalid_config_is_rejected_before_native_construction(self):
        cases = [
            ({"addresses": []}, ValueError),
            ({"addresses": "host:1"}, TypeError),
            ({"addresses": None}, TypeError),
            ({"addresses": [1]}, ValueError),
            ({"addresses": ["same", "same"]}, ValueError),
            ({"addresses": [f"host-{index}" for index in range(65)]}, ValueError),
            ({"addresses": ["x" * 1025]}, ValueError),
            ({"instance_id": ""}, ValueError),
            ({"instance_id": None}, ValueError),
            ({"instance_group": ""}, ValueError),
            ({"instance_group": "x" * 513}, ValueError),
            ({"transfer_client_config": ""}, ValueError),
            ({"transfer_client_config": None}, ValueError),
            ({"user_data": b"bytes"}, TypeError),
            ({"user_data": "x" * (64 * 1024 + 1)}, ValueError),
            ({"call_timeout_ms": 0}, ValueError),
            ({"call_timeout_ms": True}, TypeError),
            ({"call_timeout_ms": 1.0}, TypeError),
            ({"write_timeout_seconds": True}, TypeError),
            ({"write_timeout_seconds": 1801}, ValueError),
            ({"max_object_bytes": 0}, ValueError),
            ({"max_object_bytes": KV_META_MAX_OBJECT_BYTES + 1}, ValueError),
            ({"memory_base": True}, ValueError),
            ({"memory_size": -1}, ValueError),
            ({"memory_fd": 1 << 31}, ValueError),
            ({"memory_base": 1, "memory_size": 0}, ValueError),
            ({"memory_fd": 3}, ValueError),
            ({"memory_base": (1 << 64) - 1, "memory_size": 1}, ValueError),
        ]
        for overrides, error in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(error):
                    _config(**overrides)

    def test_utf8_limits_are_measured_in_bytes(self):
        with self.assertRaisesRegex(ValueError, "512 UTF-8 bytes"):
            _config(instance_id="界" * 171)
        with self.assertRaisesRegex(ValueError, "valid UTF-8"):
            _config(transfer_client_config="\ud800")

    def test_exact_config_boundaries_are_accepted_and_snapshotted(self):
        addresses = [f"host-{index}" for index in range(64)]
        config = _config(
            addresses=iter(addresses),
            instance_id="界" * 170 + "ab",
            instance_group="g" * 512,
            user_data="u" * (64 * 1024),
            call_timeout_ms=600_000,
            write_timeout_seconds=1800,
            max_object_bytes=KV_META_MAX_OBJECT_BYTES,
            memory_base=1,
            memory_size=(1 << 64) - 2,
            memory_fd=(1 << 31) - 1,
        )

        addresses[0] = "mutated"
        self.assertEqual(len(config.addresses), 64)
        self.assertEqual(config.addresses[0], "host-0")
        self.assertEqual(len(config.instance_id.encode("utf-8")), 512)


class KvMetaObjectBufferTest(unittest.TestCase):
    def test_tensor_adapter_derives_exact_size_memory_and_owner(self):
        cpu = _Tensor(pointer=0x1000, count=3, width=4)
        cuda = _Tensor(pointer=0x2000, count=7, width=2, device="cuda")

        cpu_buffer = KvMetaObjectBuffer.from_tensor("cpu", cpu)
        cuda_buffer = KvMetaObjectBuffer.from_tensor("cuda", cuda)

        self.assertEqual(
            (cpu_buffer.pointer, cpu_buffer.nbytes, cpu_buffer.memory),
            (0x1000, 12, KvMetaObjectMemory.CPU),
        )
        self.assertEqual(cuda_buffer.memory, KvMetaObjectMemory.GPU)
        self.assertEqual(cuda_buffer.nbytes, 14)
        self.assertIs(cpu_buffer.owner, cpu)
        self.assertIs(cuda_buffer.owner, cuda)

    def test_malformed_tensor_protocol_is_rejected(self):
        cases = [
            (_Tensor(contiguous=False), ValueError),
            (_Tensor(device="xpu"), ValueError),
            (_Tensor(count=0), ValueError),
            (_Tensor(count=-1), ValueError),
            (_Tensor(count=True), ValueError),
            (_Tensor(count=1 << 63, width=2), ValueError),
            (_Tensor(width=0), ValueError),
            (_Tensor(width=True), ValueError),
            (_Tensor(pointer=0), ValueError),
            (_Tensor(pointer=True), TypeError),
            (_Tensor(pointer=1.0), TypeError),
            (_Tensor(pointer=(1 << 64) - 1, count=1, width=1), ValueError),
            (object(), ValueError),
            (SimpleNamespace(is_contiguous=lambda: True), TypeError),
        ]
        for tensor, error in cases:
            with self.subTest(tensor=tensor):
                with self.assertRaises(error):
                    KvMetaObjectBuffer.from_tensor("key", tensor)

    def test_buffer_rejects_invalid_keys_ranges_and_memory(self):
        cases = [
            (("", 1, 1), {}, ValueError),
            (("界" * 171, 1, 1), {}, ValueError),
            (("key", 0, 1), {}, ValueError),
            (("key", True, 1), {}, TypeError),
            (("key", 1, 0), {}, ValueError),
            (("key", 1, True), {}, TypeError),
            (("key", 1, KV_META_MAX_OBJECT_BYTES + 1), {}, ValueError),
            (("key", (1 << 64) - 1, 1), {}, ValueError),
            (("key", 1, 1), {"memory": "xpu"}, ValueError),
        ]
        for args, kwargs, error in cases:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(error):
                    KvMetaObjectBuffer(*args, **kwargs)

    def test_exact_buffer_boundaries_and_string_memory_are_accepted(self):
        max_key = "界" * 170 + "ab"
        end_of_address_space = KvMetaObjectBuffer(
            max_key,
            (1 << 64) - 2,
            1,
            memory="gpu",
        )
        max_object = KvMetaObjectBuffer(
            "max-object",
            1,
            KV_META_MAX_OBJECT_BYTES,
        )

        self.assertEqual(len(end_of_address_space.key.encode("utf-8")), 512)
        self.assertEqual(end_of_address_space.memory, KvMetaObjectMemory.GPU)
        self.assertEqual(max_object.nbytes, KV_META_MAX_OBJECT_BYTES)


class KvMetaObjectClientTest(unittest.TestCase):
    def test_native_config_is_mapped_and_registered_memory_owner_is_retained(self):
        owner = object()
        pybind = _FakePybind()
        client = KvMetaObjectClient(
            _config(
                memory_base=0x1000,
                memory_size=4096,
                memory_fd=7,
                max_object_bytes=4096,
                call_timeout_ms=1234,
                write_timeout_seconds=45,
            ),
            registration_owner=owner,
            _pybind_module=pybind,
        )
        self.addCleanup(client.close)

        self.assertEqual(len(pybind.create_calls), 1)
        trace_id, native_config = pybind.create_calls[0]
        self.assertTrue(trace_id.startswith("kvcm-py-init-"))
        self.assertEqual(
            native_config.metadata.addresses, list(client.config.addresses)
        )
        self.assertEqual(native_config.metadata.instance_id, "rtp-emb-1")
        self.assertEqual(native_config.metadata.call_timeout_ms, 1234)
        self.assertEqual(native_config.instance_group, "epd-emb-only")
        self.assertEqual(native_config.max_object_bytes, 4096)
        self.assertEqual(native_config.write_timeout_seconds, 45)
        span = native_config.transfer_init_params.regist_span
        self.assertEqual((span.base, span.size, span.fd), (0x1000, 4096, 7))
        self.assertIs(span.owner, owner)
        self.assertEqual(native_config.transfer_init_params.role_type, _Role.WORKER)
        self.assertEqual(
            native_config.transfer_init_params.self_location_spec_name, "value"
        )

    def test_create_failure_closes_partial_native_client(self):
        native = _FakeClient()
        pybind = _FakePybind(native, create_code=_Code.ER_FAILED)

        with self.assertRaises(KvMetaObjectClientError) as raised:
            KvMetaObjectClient(_config(), _pybind_module=pybind)

        self.assertEqual(raised.exception.operation, "init")
        self.assertEqual(raised.exception.code, _Code.ER_FAILED)
        self.assertTrue(native.closed)
        self.assertEqual(native.close_calls, 1)

    def test_create_exception_is_sanitized_and_chained(self):
        pybind = _FakePybind()

        def fail_create(*_args):
            raise OSError("secret endpoint and credentials")

        pybind.KvMetaObjectClient = SimpleNamespace(Create=fail_create)
        with self.assertRaises(KvMetaObjectClientError) as raised:
            KvMetaObjectClient(_config(), _pybind_module=pybind)

        self.assertEqual(raised.exception.operation, "init")
        self.assertEqual(raised.exception.code, "OSError")
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertNotIn("secret", str(raised.exception))

    def test_malformed_create_result_fails_closed(self):
        pybind = _FakePybind()
        pybind.KvMetaObjectClient = SimpleNamespace(Create=lambda *_args: _Code.ER_OK)

        with self.assertRaises(KvMetaObjectClientError) as raised:
            KvMetaObjectClient(_config(), _pybind_module=pybind)

        self.assertEqual(raised.exception.operation, "init")
        self.assertEqual(raised.exception.code, "TypeError")

    def test_invalid_binding_closes_injected_native_client(self):
        native = _FakeClient()

        with self.assertRaisesRegex(ImportError, "does not export"):
            KvMetaObjectClient(
                _config(),
                _object_client=native,
                _pybind_module=SimpleNamespace(),
            )

        self.assertTrue(native.closed)

    def test_incompatible_binding_version_closes_injected_native_client(self):
        native = _FakeClient()
        pybind = _FakePybind(native)
        pybind.KV_META_OBJECT_API_VERSION = KV_META_OBJECT_API_VERSION + 1

        with self.assertRaisesRegex(ImportError, "incompatible.*API version"):
            KvMetaObjectClient(
                _config(),
                _object_client=native,
                _pybind_module=pybind,
            )

        self.assertTrue(native.closed)
        self.assertEqual(native.close_calls, 1)

    def test_incomplete_versioned_binding_is_rejected_before_creation(self):
        pybind = _FakePybind()
        pybind.MemoryType = SimpleNamespace(CPU=_Memory.CPU)

        with self.assertRaisesRegex(ImportError, "incomplete"):
            KvMetaObjectClient(_config(), _pybind_module=pybind)

        self.assertEqual(pybind.create_calls, [])

    def test_uninspectable_binding_fails_closed_and_releases_injected_client(self):
        class ExplodingPybind:
            def __getattribute__(self, _name):
                raise RuntimeError("provider capability detail")

        native = _FakeClient()
        with self.assertRaisesRegex(ImportError, "inspected safely") as raised:
            KvMetaObjectClient(
                _config(),
                _object_client=native,
                _pybind_module=ExplodingPybind(),
            )

        self.assertNotIn("provider capability detail", str(raised.exception))
        self.assertTrue(native.closed)
        self.assertEqual(native.close_calls, 1)

    def test_missing_native_operation_closes_created_client(self):
        native = _FakeClient()
        native.LoadObjects = None
        pybind = _FakePybind(native)

        with self.assertRaisesRegex(TypeError, "missing SaveObjects"):
            KvMetaObjectClient(_config(), _pybind_module=pybind)

        self.assertTrue(native.closed)
        self.assertEqual(native.close_calls, 1)

    def test_variable_sizes_and_cpu_cuda_iovs_reach_native_api(self):
        client, native, _ = _client(config=_config(max_object_bytes=64))
        self.addCleanup(client.close)
        tensors = [
            _Tensor(pointer=0x1000, count=3, width=4),
            _Tensor(pointer=0x2000, count=7, width=2, device="cuda"),
        ]

        client.save(["embedding", "position"], tensors, trace_id="save-trace")
        client.load_tensors(["embedding", "position"], tensors, trace_id="load-trace")

        save, load = native.calls
        self.assertEqual(
            save[:4],
            ("SaveObjects", "save-trace", ["embedding", "position"], [12, 14]),
        )
        self.assertEqual(
            load[:4],
            ("LoadObjects", "load-trace", ["embedding", "position"], [12, 14]),
        )
        self.assertEqual(save[4][0].iovs[0].type, _Memory.CPU)
        self.assertEqual(save[4][1].iovs[0].type, _Memory.GPU)
        self.assertEqual(save[4][0].iovs[0].base, 0x1000)
        self.assertEqual(save[4][0].iovs[0].size, 12)
        self.assertEqual(save[4][1].iovs[0].size, 14)
        self.assertFalse(save[4][0].iovs[0].ignore)

    def test_raw_buffer_aliases_preserve_exact_order_and_sizes(self):
        client, native, _ = _client(config=_config(max_object_bytes=64))
        self.addCleanup(client.close)
        owners = [object(), object(), object()]
        objects = [
            KvMetaObjectBuffer("one", 0x1000, 1, owner=owners[0]),
            KvMetaObjectBuffer("seven", 0x2000, 7, KvMetaObjectMemory.GPU, owners[1]),
            KvMetaObjectBuffer("thirteen", 0x3000, 13, owner=owners[2]),
        ]

        client.save_buffers(objects, trace_id="raw-save")
        client.load_buffers(objects, trace_id="raw-load")

        for call, operation, trace in zip(
            native.calls,
            ("SaveObjects", "LoadObjects"),
            ("raw-save", "raw-load"),
        ):
            self.assertEqual(call[0], operation)
            self.assertEqual(call[1], trace)
            self.assertEqual(call[2], ["one", "seven", "thirteen"])
            self.assertEqual(call[3], [1, 7, 13])
            self.assertEqual(
                [block.iovs[0].base for block in call[4]],
                [0x1000, 0x2000, 0x3000],
            )

    def test_count_and_byte_limits_split_one_logical_save(self):
        client, native, _ = _client()
        self.addCleanup(client.close)
        count_objects = [
            KvMetaObjectBuffer(f"count-{index}", 0x1000 + index, 1)
            for index in range(KV_META_MAX_BATCH_ITEMS + 1)
        ]
        client.save_buffers(count_objects, trace_id="count")
        self.assertEqual([len(call[2]) for call in native.calls], [64, 1])
        self.assertEqual(
            [call[1] for call in native.calls],
            ["count:batch-1-of-2", "count:batch-2-of-2"],
        )

        native.calls.clear()
        byte_objects = [
            KvMetaObjectBuffer(
                f"byte-{index}", 0x2000 + index, KV_META_MAX_OBJECT_BYTES
            )
            for index in range(KV_META_MAX_BATCH_BYTES // KV_META_MAX_OBJECT_BYTES + 1)
        ]
        client.save_buffers(byte_objects, trace_id="bytes")
        self.assertEqual([len(call[2]) for call in native.calls], [4, 1])

    def test_generated_traces_are_unique_and_batch_qualified(self):
        client, native, _ = _client()
        self.addCleanup(client.close)
        objects = [
            KvMetaObjectBuffer(f"key-{index}", 0x1000 + index, 1)
            for index in range(KV_META_MAX_BATCH_ITEMS + 1)
        ]

        client.save_buffers(objects)
        first_traces = [call[1] for call in native.calls]
        native.calls.clear()
        client.load_buffers(objects)
        second_traces = [call[1] for call in native.calls]

        self.assertRegex(first_traces[0], r"^kvcm-py-save-[0-9a-f]{32}:batch-1-of-2$")
        self.assertEqual(first_traces[1], first_traces[0].replace("batch-1", "batch-2"))
        self.assertRegex(second_traces[0], r"^kvcm-py-load-[0-9a-f]{32}:batch-1-of-2$")
        self.assertNotEqual(first_traces[0], second_traces[0])

    def test_whole_operation_is_validated_before_first_native_call(self):
        client, native, _ = _client()
        self.addCleanup(client.close)
        objects = [
            KvMetaObjectBuffer(f"key-{index}", 0x1000 + index, 1)
            for index in range(KV_META_MAX_BATCH_ITEMS + 1)
        ]
        objects[-1] = KvMetaObjectBuffer("key-0", 0x9999, 1)

        with self.assertRaisesRegex(ValueError, "unique"):
            client.save_buffers(objects)
        self.assertEqual(native.calls, [])

        with self.assertRaisesRegex(ValueError, "length mismatch"):
            client.save(["key"], [])
        self.assertEqual(native.calls, [])

    def test_save_failure_reports_partial_progress_and_never_auto_rolls_back(self):
        client, native, _ = _client()
        self.addCleanup(client.close)
        native.results["SaveObjects"] = [_Code.ER_OK, _Code.ER_FAILED]
        objects = [
            KvMetaObjectBuffer(f"key-{index}", 0x1000 + index, 1)
            for index in range(KV_META_MAX_BATCH_ITEMS + 1)
        ]

        with self.assertRaises(KvMetaObjectClientError) as raised:
            client.save_buffers(objects)

        error = raised.exception
        self.assertEqual(error.operation, "save")
        self.assertEqual(error.batch_index, 1)
        self.assertEqual(error.batch_count, 2)
        self.assertEqual(error.completed_items, 64)
        self.assertFalse(error.unknown_outcome)
        self.assertEqual(
            [call[0] for call in native.calls], ["SaveObjects", "SaveObjects"]
        )
        self.assertNotIn("Remove", [call[0] for call in native.calls])

    def test_ambiguous_mutations_and_native_exceptions_are_marked_unknown(self):
        client, native, _ = _client()
        self.addCleanup(client.close)
        native.results["SaveObjects"] = [_Code.ER_INVALID_GRPCSTATUS]
        with self.assertRaises(KvMetaObjectClientError) as save_error:
            client.save(["key"], [_Tensor()])
        self.assertTrue(save_error.exception.unknown_outcome)

        def fail_remove(*_args):
            raise OSError("endpoint and key must not leak")

        native.Remove = fail_remove
        with self.assertRaises(KvMetaObjectClientError) as remove_error:
            client.remove(["key"])
        self.assertTrue(remove_error.exception.unknown_outcome)
        self.assertIsInstance(remove_error.exception.__cause__, OSError)
        self.assertNotIn("endpoint", str(remove_error.exception))

    def test_load_failure_stops_before_later_batches_and_is_not_ambiguous(self):
        client, native, _ = _client()
        self.addCleanup(client.close)
        native.results["LoadObjects"] = [_Code.ER_FAILED]
        objects = [
            KvMetaObjectBuffer(f"key-{index}", 0x1000 + index, 1)
            for index in range(KV_META_MAX_BATCH_ITEMS + 1)
        ]

        with self.assertRaises(KvMetaObjectClientError) as raised:
            client.load_buffers(objects, trace_id="load")

        error = raised.exception
        self.assertEqual(error.operation, "load")
        self.assertEqual(error.batch_index, 0)
        self.assertEqual(error.batch_count, 2)
        self.assertEqual(error.batch_start, 0)
        self.assertEqual(error.batch_size, KV_META_MAX_BATCH_ITEMS)
        self.assertEqual(error.completed_items, 0)
        self.assertFalse(error.unknown_outcome)
        self.assertEqual(len(native.calls), 1)

    def test_native_load_exception_is_sanitized_and_not_ambiguous(self):
        client, native, _ = _client()
        self.addCleanup(client.close)

        def fail_load(*_args):
            raise OSError("secret object key")

        native.LoadObjects = fail_load
        with self.assertRaises(KvMetaObjectClientError) as raised:
            client.load(["key"], [_Tensor()])

        self.assertFalse(raised.exception.unknown_outcome)
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertNotIn("secret", str(raised.exception))

    def test_dynamic_native_method_lookup_failure_is_sanitized(self):
        client, native, _ = _client()
        self.addCleanup(client.close)

        class ExplodingDescriptor:
            def __get__(self, _instance, _owner):
                raise OSError("secret dynamically loaded provider path")

        original_save = type(native).SaveObjects
        type(native).SaveObjects = ExplodingDescriptor()
        self.addCleanup(setattr, type(native), "SaveObjects", original_save)

        with self.assertRaises(KvMetaObjectClientError) as raised:
            client.save(["key"], [_Tensor()])

        self.assertEqual(raised.exception.operation, "save")
        self.assertTrue(raised.exception.unknown_outcome)
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertNotIn("secret", str(raised.exception))

    def test_malformed_zero_like_codes_are_not_success(self):
        client, native, _ = _client()
        self.addCleanup(client.close)
        for malformed in (False, 0.0, "ER_OK"):
            native.results["SaveObjects"] = [malformed]
            with self.subTest(malformed=malformed):
                with self.assertRaises(KvMetaObjectClientError) as raised:
                    client.save(["key"], [_Tensor()])
                self.assertTrue(raised.exception.unknown_outcome)

    def test_unknown_integer_mutation_code_is_ambiguous_but_known_code_is_not(self):
        client, native, _ = _client()
        self.addCleanup(client.close)

        native.results["Remove"] = [987654]
        with self.assertRaises(KvMetaObjectClientError) as unknown:
            client.remove(["unknown"])
        self.assertTrue(unknown.exception.unknown_outcome)

        native.results["Remove"] = [_Code.ER_FAILED]
        with self.assertRaises(KvMetaObjectClientError) as known:
            client.remove(["known"])
        self.assertFalse(known.exception.unknown_outcome)

    def test_malformed_native_enum_registry_keeps_mutation_ambiguous(self):
        client, native, pybind = _client()
        self.addCleanup(client.close)

        class BrokenMembers:
            def values(self):
                raise RuntimeError("malformed native enum registry")

        class ExplodingLookup:
            def __getattribute__(self, _name):
                raise RuntimeError("malformed native enum lookup")

        registries = (
            SimpleNamespace(
                ER_OK=_Code.ER_OK,
                ER_INVALID_GRPCSTATUS=_Code.ER_INVALID_GRPCSTATUS,
                __members__=BrokenMembers(),
            ),
            ExplodingLookup(),
        )
        for index, registry in enumerate(registries):
            with self.subTest(registry=type(registry).__name__):
                pybind.ClientErrorCode = registry
                native.results["Remove"] = [_Code.ER_FAILED]
                with self.assertRaises(KvMetaObjectClientError) as raised:
                    client.remove([f"key-{index}"])

                self.assertTrue(raised.exception.unknown_outcome)

    def test_remove_attempts_every_batch_and_reports_aggregate_result(self):
        client, native, _ = _client()
        self.addCleanup(client.close)
        native.results["Remove"] = [
            _Code.ER_FAILED,
            _Code.ER_INVALID_GRPCSTATUS,
            _Code.ER_OK,
        ]
        keys = [f"key-{index}" for index in range(129)]

        with self.assertRaises(KvMetaObjectClientError) as raised:
            client.remove(keys, trace_id="remove")

        error = raised.exception
        self.assertEqual([len(call[2]) for call in native.calls], [64, 64, 1])
        self.assertEqual(error.batch_index, 0)
        self.assertEqual(error.failed_batches, 2)
        self.assertEqual(error.completed_items, 1)
        self.assertTrue(error.unknown_outcome)

    def test_remove_success_batches_all_keys_and_empty_remove_is_a_noop(self):
        client, native, _ = _client()
        self.addCleanup(client.close)
        keys = [f"key-{index}" for index in range(129)]

        client.remove(keys, trace_id="remove")
        client.remove([])

        self.assertEqual([len(call[2]) for call in native.calls], [64, 64, 1])
        self.assertEqual(
            [call[1] for call in native.calls],
            [
                "remove:batch-1-of-3",
                "remove:batch-2-of-3",
                "remove:batch-3-of-3",
            ],
        )
        self.assertEqual(
            [key for call in native.calls for key in call[2]],
            keys,
        )

    def test_trace_key_and_object_limit_validation_precedes_native_io(self):
        client, native, _ = _client(config=_config(max_object_bytes=8))
        self.addCleanup(client.close)
        invalid_calls = [
            lambda: client.save((key for key in ["a"]), [_Tensor()]),
            lambda: client.save(["a"], (_Tensor(),)),
            lambda: client.save(["a"], [_Tensor(count=9, width=1)]),
            lambda: client.save(["a"], [_Tensor()], trace_id=1),
            lambda: client.save(["a"], [_Tensor()], trace_id=""),
            lambda: client.remove(["a", "a"]),
            lambda: client.remove("a"),
            lambda: client.remove([1]),
            lambda: client.remove(["x" * 513]),
            lambda: client.remove(["a"], trace_id="\ud800"),
        ]
        for invoke in invalid_calls:
            with self.subTest(invoke=invoke):
                with self.assertRaises((TypeError, ValueError)):
                    invoke()
        self.assertEqual(native.calls, [])

    def test_operations_are_serialized_across_threads(self):
        save_entered = threading.Event()
        load_entered = threading.Event()
        release_save = threading.Event()

        class BlockingClient(_FakeClient):
            def SaveObjects(self, trace_id, keys, sizes, buffers):
                self.calls.append(("SaveObjects", trace_id, keys, sizes, buffers))
                save_entered.set()
                release_save.wait(timeout=5)
                return _Code.ER_OK

            def LoadObjects(self, trace_id, keys, sizes, buffers):
                self.calls.append(("LoadObjects", trace_id, keys, sizes, buffers))
                load_entered.set()
                return _Code.ER_OK

        native = BlockingClient()
        client, _, _ = _client(native=native)
        self.addCleanup(client.close)
        save_thread = threading.Thread(target=client.save, args=(["save"], [_Tensor()]))
        load_thread = threading.Thread(target=client.load, args=(["load"], [_Tensor()]))

        save_thread.start()
        self.assertTrue(save_entered.wait(timeout=2))
        load_thread.start()
        self.assertFalse(load_entered.wait(timeout=0.1))
        release_save.set()
        save_thread.join(timeout=2)
        load_thread.join(timeout=2)

        self.assertFalse(save_thread.is_alive())
        self.assertFalse(load_thread.is_alive())
        self.assertTrue(load_entered.is_set())
        self.assertEqual(
            [call[0] for call in native.calls], ["SaveObjects", "LoadObjects"]
        )

    def test_close_waits_for_inflight_call_and_is_idempotent(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingClient(_FakeClient):
            def SaveObjects(self, trace_id, keys, sizes, buffers):
                entered.set()
                release.wait(timeout=5)
                return _Code.ER_OK

        native = BlockingClient()
        client, _, _ = _client(native=native)
        save_thread = threading.Thread(target=client.save, args=(["key"], [_Tensor()]))
        close_thread = threading.Thread(target=client.close)

        save_thread.start()
        self.assertTrue(entered.wait(timeout=2))
        close_thread.start()
        close_thread.join(timeout=0.05)
        self.assertTrue(close_thread.is_alive())
        release.set()
        save_thread.join(timeout=2)
        close_thread.join(timeout=2)

        self.assertFalse(save_thread.is_alive())
        self.assertFalse(close_thread.is_alive())
        self.assertTrue(native.closed)
        client.close()
        self.assertEqual(native.close_calls, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            client.load(["key"], [_Tensor()])

    def test_close_failure_still_closes_state_and_drops_registration_owner(self):
        class WeakOwner:
            pass

        class FailingCloseClient(_FakeClient):
            def close(self):
                super().close()
                raise OSError("close failed")

        owner = WeakOwner()
        owner_ref = weakref.ref(owner)
        native = FailingCloseClient()
        pybind = _FakePybind(native)
        client = KvMetaObjectClient(
            _config(memory_base=0x1000, memory_size=4096),
            registration_owner=owner,
            _pybind_module=pybind,
        )
        # Do not let the fake binding's call recording retain its temporary
        # native config and registered-memory owner.
        pybind.create_calls.clear()
        del owner
        gc.collect()
        self.assertIsNotNone(owner_ref())

        with self.assertRaisesRegex(OSError, "close failed"):
            client.close()
        gc.collect()

        self.assertTrue(native.closed)
        self.assertEqual(native.close_calls, 1)
        self.assertIsNone(owner_ref())
        client.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            client.save(["key"], [_Tensor()])

    def test_context_manager_closes_client(self):
        client, native, _ = _client()
        with client as active:
            self.assertIs(active, client)
        self.assertTrue(native.closed)

    def test_context_close_failure_does_not_mask_body_exception(self):
        class FailingCloseClient(_FakeClient):
            def close(self):
                super().close()
                raise OSError("secret close detail")

        native = FailingCloseClient()
        client, _, _ = _client(native=native)

        with self.assertLogs(
            "kv_cache_manager.client.kv_meta_object_client", level="WARNING"
        ) as logs:
            with self.assertRaisesRegex(ValueError, "body failure"):
                with client:
                    raise ValueError("body failure")

        self.assertTrue(native.closed)
        self.assertEqual(native.close_calls, 1)
        self.assertNotIn("secret close detail", " ".join(logs.output))


class KvMetaObjectNativeBindingContractTest(unittest.TestCase):
    def test_real_binding_exports_exact_object_types_and_64_bit_iovs(self):
        from kv_cache_manager.client.pybind import kvcm_py_client

        required = (
            "KV_META_OBJECT_API_VERSION",
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
        self.assertTrue(all(hasattr(kvcm_py_client, name) for name in required))
        self.assertEqual(
            kvcm_py_client.KV_META_OBJECT_API_VERSION,
            KV_META_OBJECT_API_VERSION,
        )

        iov = kvcm_py_client.Iov()
        iov.type = kvcm_py_client.MemoryType.CPU
        iov.base = (1 << 64) - 1
        iov.size = KV_META_MAX_OBJECT_BYTES
        iov.ignore = False
        block = kvcm_py_client.BlockBuffer()
        block.iovs = [iov]

        self.assertEqual(block.iovs[0].base, (1 << 64) - 1)
        self.assertEqual(block.iovs[0].size, KV_META_MAX_OBJECT_BYTES)
        self.assertEqual(block.iovs[0].type, kvcm_py_client.MemoryType.CPU)

    def test_high_level_client_reaches_real_native_create_and_maps_error(self):
        from kv_cache_manager.client.pybind import kvcm_py_client

        with self.assertRaises(KvMetaObjectClientError) as raised:
            KvMetaObjectClient(
                _config(
                    addresses=("127.0.0.1:1",),
                    transfer_client_config="{invalid-json",
                )
            )

        self.assertEqual(raised.exception.operation, "init")
        self.assertEqual(
            raised.exception.code,
            kvcm_py_client.ClientErrorCode.ER_INVALID_CLIENT_CONFIG,
        )
        self.assertFalse(raised.exception.unknown_outcome)


if __name__ == "__main__":
    unittest.main()
