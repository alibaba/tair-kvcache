import threading
import time
import unittest
from enum import IntEnum
from types import SimpleNamespace

from kv_cache_manager.client import (
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
        self.closed = True


class _FakePybind:
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
            ({"addresses": ["same", "same"]}, ValueError),
            ({"addresses": ["x" * 1025]}, ValueError),
            ({"instance_id": ""}, ValueError),
            ({"instance_group": "x" * 513}, ValueError),
            ({"transfer_client_config": ""}, ValueError),
            ({"user_data": "x" * (64 * 1024 + 1)}, ValueError),
            ({"call_timeout_ms": 0}, ValueError),
            ({"call_timeout_ms": True}, TypeError),
            ({"write_timeout_seconds": 1801}, ValueError),
            ({"max_object_bytes": KV_META_MAX_OBJECT_BYTES + 1}, ValueError),
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
            (_Tensor(width=0), ValueError),
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
            (("key", 1, 0), {}, ValueError),
            (("key", (1 << 64) - 1, 1), {}, ValueError),
            (("key", 1, 1), {"memory": "xpu"}, ValueError),
        ]
        for args, kwargs, error in cases:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(error):
                    KvMetaObjectBuffer(*args, **kwargs)


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

    def test_invalid_binding_closes_injected_native_client(self):
        native = _FakeClient()

        with self.assertRaisesRegex(ImportError, "does not export"):
            KvMetaObjectClient(
                _config(),
                _object_client=native,
                _pybind_module=SimpleNamespace(),
            )

        self.assertTrue(native.closed)

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
        self.assertFalse(save[4][0].iovs[0].ignore)

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

    def test_malformed_zero_like_codes_are_not_success(self):
        client, native, _ = _client()
        self.addCleanup(client.close)
        for malformed in (False, 0.0, "ER_OK"):
            native.results["SaveObjects"] = [malformed]
            with self.subTest(malformed=malformed):
                with self.assertRaises(KvMetaObjectClientError):
                    client.save(["key"], [_Tensor()])

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

    def test_trace_key_and_object_limit_validation_precedes_native_io(self):
        client, native, _ = _client(config=_config(max_object_bytes=8))
        self.addCleanup(client.close)
        invalid_calls = [
            lambda: client.save((key for key in ["a"]), [_Tensor()]),
            lambda: client.save(["a"], (_Tensor(),)),
            lambda: client.save(["a"], [_Tensor(count=9, width=1)]),
            lambda: client.remove(["a", "a"]),
            lambda: client.remove(["a"], trace_id="\ud800"),
        ]
        for invoke in invalid_calls:
            with self.subTest(invoke=invoke):
                with self.assertRaises((TypeError, ValueError)):
                    invoke()
        self.assertEqual(native.calls, [])

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
        with self.assertRaisesRegex(RuntimeError, "closed"):
            client.load(["key"], [_Tensor()])

    def test_context_manager_closes_client(self):
        client, native, _ = _client()
        with client as active:
            self.assertIs(active, client)
        self.assertTrue(native.closed)


if __name__ == "__main__":
    unittest.main()
