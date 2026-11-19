"""Unit tests for the data-plane seam and the worker's rank-space hooks.

The seam must be invisible to GPU deployments -- an unset
``transfer_backend_module_path`` still yields the very ``DataTransferManager``
the worker always built -- and complete enough for another engine to replace
the swap hop: ``ConnectorWorker`` may only ever reach the data plane through
``self._backend``, and it must take its rank and world size from the two
overridable hooks.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# vllm_stubs must be imported before torch: in the open-source CI (no torch
# installed) it registers the MagicMock stand-in that the bare `import torch`
# below then resolves to.
from kv_cache_manager.py_connector.test.vllm_stubs import (  # noqa: F401 (stubs)
    STUBBED,
    make_connector,
)

import torch
from pydantic import ValidationError

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import FullAttentionSpec

from kv_cache_manager.py_connector.common.tp_coordinator import SaveContext
from kv_cache_manager.py_connector.vllm import connector_worker, v1_connector
from kv_cache_manager.py_connector.vllm.config import TairKvCacheConnectorExtraConfig
from kv_cache_manager.py_connector.vllm.data_transfer import DataTransferManager
from kv_cache_manager.py_connector.vllm.metadata import (
    LoadRequest,
    SaveRequest,
    TairKvCacheConnectorMetadata,
)
from kv_cache_manager.py_connector.vllm.transfer_backend import (
    MultiResult,
    TransferBackend,
    make_transfer_backend,
)
from kv_cache_manager.py_connector.vllm.transfer_types import (
    AttentionTransferGroup,
    KVLayout,
    TransferPlan,
)
from kv_cache_manager.py_connector.vllm.v1_connector import TairKvCacheConnector
from kv_cache_manager.py_connector.vllm.vllm_common import AttentionGroupMeta

#: Import path the factory resolves for the fake backend module.
BACKEND_MODULE = "kvcm_test_transfer_backend"


def _extra_config(**overrides):
    return TairKvCacheConnectorExtraConfig(
        manager_uri="http://127.0.0.1:1",
        coordinator_base_port=19000,
        instance_group="group",
        instance_id="instance",
        **overrides,
    )


def _fake_backend_module(make_backend):
    """Register a stand-in backend module under :data:`BACKEND_MODULE`.

    The factory resolves the configured path through ``importlib``, so the
    stand-in has to be a registered module rather than a patched attribute.
    """
    module = types.ModuleType(BACKEND_MODULE)
    if make_backend is not None:
        setattr(module, "make_backend", make_backend)
    return patch.dict(sys.modules, {BACKEND_MODULE: module})


class FakeBackend:
    """A backend with the six-method shape and no data movement."""

    def submit(self, fn, *args, **kwargs):
        return MagicMock()

    def save_task(self, *args, **kwargs):
        pass

    def load_task(self, *args, **kwargs):
        pass

    def create_save_done_callback(self, *args, **kwargs):
        return MagicMock()

    def create_load_done_callback(self, *args, **kwargs):
        return MagicMock()

    def shutdown(self):
        pass


def _attn_kv_cache(num_blocks=4, kernel_bs=8, heads=2, head_dim=4):
    """A packed 4-D KV page laid out as vLLM >= 0.26 allocates it: the tensor
    exposes (blocks, heads, kernel_block, 2*head_dim) while the bytes inside
    one block stay token-major."""
    base = torch.zeros(num_blocks, kernel_bs, heads, 2 * head_dim, dtype=torch.float16)
    return base.permute(0, 2, 1, 3)


def _attn_group(block_size=8):
    return AttentionTransferGroup(
        group_idx=0,
        spec_name="tp0_g0",
        layer_names=["l0"],
        block_size=block_size,
        per_block_bytes=1024,
        layer_num=1,
        kv_layout=KVLayout.PACKED_4D,
        kvcache_ptr_tensor_gpu=None,  # ty: ignore[invalid-argument-type]
        num_kv_ptrs=1,
        per_token_dim=16,
        kernel_block_size=block_size,
        block_stride=0,
    )


def _factory_kwargs(extra_config):
    return dict(
        extra_config=extra_config,
        kvcache_info=SimpleNamespace(
            tp_rank=0,
            world_size=1,
            groups=[SimpleNamespace(spec_name="tp0_g0", per_block_bytes=4096)],
            device=torch.device("cpu"),
            dtype=torch.uint8,
        ),
        manager_block_size=16,
        transfer_client=MagicMock(),
        coordinator_client=MagicMock(),
        device=torch.device("cpu"),
    )


def _bare_worker():
    """A ConnectorWorker with only what the hooks under test touch."""
    conn = make_connector(manager_block_size=8)
    conn._self_spec_names = {0: "tp0_g0"}
    conn._pending_saves = {}
    conn._extra_config = (  # ty: ignore[invalid-assignment]
        SimpleNamespace(block_per_save_task=8, block_per_load_task=8)
    )
    conn._device_mod = MagicMock()
    conn._backend = MagicMock(spec=TransferBackend)
    conn._plan_group_transfers = MagicMock(
        return_value=[
            TransferPlan(group=_attn_group(), uris=["u0", "u1"], token_indices=[[], []])
        ]
    )
    return conn


class TestExtraConfig(unittest.TestCase):
    def test_defaults_keep_the_single_stage_gpu_path(self):
        cfg = _extra_config()
        self.assertIsNone(cfg.transfer_backend_module_path)

    def test_unknown_keys_are_still_rejected(self):
        with self.assertRaises(ValidationError):
            _extra_config(transfer_backend="gpu")


class TestFactoryDefault(unittest.TestCase):
    def test_unset_path_builds_the_gpu_backend(self):
        kwargs = _factory_kwargs(_extra_config())
        backend = make_transfer_backend(**kwargs)
        try:
            self.assertIsInstance(backend, DataTransferManager)
        finally:
            backend.shutdown()

    def test_empty_string_path_also_means_the_gpu_backend(self):
        kwargs = _factory_kwargs(_extra_config(transfer_backend_module_path=""))
        backend = make_transfer_backend(**kwargs)
        try:
            self.assertIsInstance(backend, DataTransferManager)
        finally:
            backend.shutdown()

    def test_default_backend_wires_the_triton_kernels(self):
        backend = make_transfer_backend(**_factory_kwargs(_extra_config()))
        assert isinstance(backend, DataTransferManager)
        try:
            # Gather/scatter without the kernels wired in would degrade to a
            # silent no-op instead of moving bytes.
            self.assertTrue(
                backend._kernel.__name__.endswith("batch_gather_scatter_helper")
            )
        finally:
            backend.shutdown()

    def test_default_backend_implements_every_protocol_method(self):
        protocol = {name for name in dir(TransferBackend) if not name.startswith("_")}
        self.assertTrue(protocol, "the protocol must declare the data-plane surface")
        for name in protocol:
            self.assertTrue(callable(getattr(DataTransferManager, name, None)), name)


class TestFactoryCustomBackend(unittest.TestCase):
    def test_named_module_is_loaded_with_the_worker_context(self):
        made = {}

        def make_backend(**kwargs):
            made.update(kwargs)
            return FakeBackend()

        kwargs = _factory_kwargs(
            _extra_config(transfer_backend_module_path=BACKEND_MODULE)
        )
        with _fake_backend_module(make_backend):
            backend = make_transfer_backend(**kwargs)

        self.assertIsInstance(backend, FakeBackend)
        # The backend gets exactly the context the worker was built with.
        self.assertEqual(set(made), set(kwargs))
        for name, value in kwargs.items():
            self.assertIs(made[name], value, name)

    def test_missing_module_names_the_knob(self):
        kwargs = _factory_kwargs(
            _extra_config(transfer_backend_module_path="kvcm_absent_backend_module")
        )
        with self.assertRaises(ImportError) as ctx:
            make_transfer_backend(**kwargs)
        self.assertIn("transfer_backend_module_path", str(ctx.exception))

    def test_module_without_a_factory_is_rejected(self):
        kwargs = _factory_kwargs(
            _extra_config(transfer_backend_module_path=BACKEND_MODULE)
        )
        with _fake_backend_module(None):
            with self.assertRaises(TypeError) as ctx:
                make_transfer_backend(**kwargs)
        self.assertIn("make_backend", str(ctx.exception))

    def test_backend_missing_a_protocol_method_is_rejected(self):
        kwargs = _factory_kwargs(
            _extra_config(transfer_backend_module_path=BACKEND_MODULE)
        )
        with _fake_backend_module(lambda **_: SimpleNamespace(save_task=None)):
            with self.assertRaises(TypeError) as ctx:
                make_transfer_backend(**kwargs)
        self.assertIn("submit", str(ctx.exception))


class TestWorkerUsesBackend(unittest.TestCase):
    def test_wait_for_save_submits_through_the_backend(self):
        conn = _bare_worker()
        meta = TairKvCacheConnectorMetadata(epoch=0)
        meta.add_save_request(
            SaveRequest(
                req_id="r0",
                target_locations=[{}, {}],
                manager_block_idxes=[0, 1],
                write_session_id="sess",
            )
        )

        conn.wait_for_save(meta)

        conn._backend.create_save_done_callback.assert_called_once_with(
            "r0", 0, "sess", 2
        )
        conn._backend.submit.assert_called_once()
        fn, *args = conn._backend.submit.call_args.args
        self.assertIs(fn, conn._backend.save_task)
        self.assertIsInstance(args[0], MultiResult)
        # Its one session stays pending until the coordinator reports.
        self.assertEqual(conn._pending_saves, {"r0": 1})

    def test_start_load_kv_submits_through_the_backend(self):
        conn = _bare_worker()
        meta = TairKvCacheConnectorMetadata(epoch=0)
        meta.add_load_request(
            LoadRequest(
                req_id="r0",
                manager_block_idxes=[0, 1],
                need_load_locations=[{"location_specs": []}] * 2,
                all_block_ids=[[10, 11, 12]],
            )
        )

        conn.start_load_kv(MagicMock(), meta)

        conn._backend.create_load_done_callback.assert_called_once()
        conn._backend.submit.assert_called_once()
        fn, *args = conn._backend.submit.call_args.args
        self.assertIs(fn, conn._backend.load_task)
        self.assertIsInstance(args[0], MultiResult)

    def _register(self, conn):
        conn._backend = None  # built by the call under test
        conn._transfer_client = MagicMock()
        conn._coordinator_client = MagicMock()
        conn.register_kv_caches({"l0": _attn_kv_cache()})
        return conn._backend

    @unittest.skipIf("torch" in STUBBED, "needs a real KV cache tensor (stubbed in CI)")
    def test_register_kv_caches_builds_the_default_backend(self):
        conn = _bare_worker()
        conn._extra_config = _extra_config()
        backend = self._register(conn)
        try:
            self.assertIsInstance(backend, DataTransferManager)
        finally:
            backend.shutdown()

    @unittest.skipIf("torch" in STUBBED, "needs a real KV cache tensor (stubbed in CI)")
    def test_register_kv_caches_builds_the_configured_backend(self):
        conn = _bare_worker()
        conn._extra_config = _extra_config(transfer_backend_module_path=BACKEND_MODULE)
        built = FakeBackend()

        with _fake_backend_module(lambda **_: built):
            backend = self._register(conn)

        self.assertIs(backend, built)
        self.assertEqual(conn._kvcache_info.groups[0].spec_name, "tp0_g0")


class TestWorkerShutdown(unittest.TestCase):
    def test_shutdown_without_a_backend_is_a_noop(self):
        conn = _bare_worker()
        conn._backend = None
        conn.shutdown()  # engine teardown before register_kv_caches

    def test_shutdown_reaches_the_backend(self):
        conn = _bare_worker()
        conn.shutdown()
        conn._backend.shutdown.assert_called_once_with()
