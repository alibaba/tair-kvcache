"""Unit tests for the multi-version KV cache layout detection.

``attn_kv_views`` must recognize the three flash_attn layouts vLLM has shipped
(detected from the tensor shape, never from version strings) and reject
anything else:

* 4-D packed   ``(num_blocks, H, block, 2D)``   -- vLLM >= 0.26.0
* 5-D N-first  ``(num_blocks, 2, block, H, D)`` -- vLLM 0.23.0 - 0.25.x
* 5-D KV-first ``(2, num_blocks, block, H, D)`` -- vLLM <= 0.22.1

``_build_transfer_group`` must derive the transfer pointers / strides from the
normalized views, and ``ensure_hybrid_supported`` must fail fast when the
installed vLLM's scheduler rejects external KV loads for hybrid models
(vLLM <= 0.22.x).

Attention tests run without torch using FakeTensor for strided-view semantics.
State copy tests use real CPU tensors and skip when torch is unavailable.
"""

import sys
import types
from typing import Any
import unittest

from kv_cache_manager.py_connector.test.vllm_stubs import STUBBED, make_connector
from kv_cache_manager.py_connector.vllm.vllm_common import (
    AttentionGroupMeta,
    StateGroupMeta,
    state_kv_view,
)
import torch
from kv_cache_manager.py_connector.vllm.v1_connector import (
    attn_kv_views,
    ensure_hybrid_supported,
)
from kv_cache_manager.py_connector.vllm.transfer_types import KVLayout

ITEMSIZE = 2  # bf16/fp16
BASE_PTR = 1 << 20


class FakeTensor:
    """Minimal strided tensor: only what attn_kv_views / _build_transfer_group
    read (dim/shape/stride/permute/indexing/data_ptr)."""

    def __init__(self, shape, strides, offset=0, base=BASE_PTR):
        self.shape = tuple(shape)
        self._strides = tuple(strides)
        self._offset = offset
        self._base = base

    @classmethod
    def contiguous(cls, shape, base=BASE_PTR):
        strides, acc = [], 1
        for s in reversed(shape):
            strides.append(acc)
            acc *= s
        return cls(shape, tuple(reversed(strides)), base=base)

    def dim(self):
        return len(self.shape)

    def stride(self, i=None):
        return self._strides if i is None else self._strides[i]

    def data_ptr(self):
        return self._base + self._offset * ITEMSIZE

    def permute(self, *dims):
        return FakeTensor(
            [self.shape[d] for d in dims],
            [self._strides[d] for d in dims],
            self._offset,
            self._base,
        )

    def __getitem__(self, idx):
        if isinstance(idx, int):  # t[i]: drop dim 0
            return FakeTensor(
                self.shape[1:],
                self._strides[1:],
                self._offset + idx * self._strides[0],
                self._base,
            )
        if isinstance(idx, tuple) and idx[0] == slice(None) and isinstance(idx[1], int):
            # t[:, i]: drop dim 1
            return FakeTensor(
                self.shape[:1] + self.shape[2:],
                self._strides[:1] + self._strides[2:],
                self._offset + idx[1] * self._strides[1],
                self._base,
            )
        raise TypeError(f"unsupported index {idx!r}")


def packed_4d(n=10, h=4, b=16, d2=256, base=BASE_PTR):
    """vLLM >= 0.26.0: NHD memory is (n, b, h, d2) contiguous; the registered
    tensor is its (n, h, b, d2) permuted view."""
    return FakeTensor.contiguous([n, b, h, d2], base=base).permute(0, 2, 1, 3)


def kv_first_5d(n=10, b=16, h=4, d=128, base=BASE_PTR):
    """vLLM <= 0.22.1: (2, n, b, h, d) contiguous."""
    return FakeTensor.contiguous([2, n, b, h, d], base=base)


def n_first_5d(n=10, b=16, h=4, d=128, base=BASE_PTR):
    """vLLM 0.23.0 - 0.25.x: (n, 2, b, h, d) contiguous."""
    return FakeTensor.contiguous([n, 2, b, h, d], base=base)


class TestAttnKvViews(unittest.TestCase):
    def test_packed_4d(self):
        views, layout = attn_kv_views(packed_4d())
        self.assertIs(layout, KVLayout.PACKED_4D)
        self.assertEqual(len(views), 1)
        v = views[0]
        self.assertEqual(v.shape, (10, 16, 4, 256))  # (n, b, h, 2d)
        self.assertEqual(v.stride(), (16 * 4 * 256, 4 * 256, 256, 1))
        self.assertEqual(v.data_ptr(), BASE_PTR)  # storage base

    def test_kv_first_5d(self):
        views, layout = attn_kv_views(kv_first_5d())
        self.assertIs(layout, KVLayout.SPLIT_KV_5D_KV_FIRST)
        self.assertEqual(len(views), 2)
        k, v = views
        for view in (k, v):
            self.assertEqual(view.shape, (10, 16, 4, 128))
            self.assertEqual(view.stride(), (16 * 4 * 128, 4 * 128, 128, 1))
        self.assertEqual(k.data_ptr(), BASE_PTR)
        # V base = K base + num_blocks * block * h * d elements.
        self.assertEqual(v.data_ptr() - k.data_ptr(), 10 * 16 * 4 * 128 * ITEMSIZE)

    def test_n_first_5d(self):
        views, layout = attn_kv_views(n_first_5d())
        self.assertIs(layout, KVLayout.SPLIT_KV_5D_N_FIRST)
        self.assertEqual(len(views), 2)
        k, v = views
        for view in (k, v):
            self.assertEqual(view.shape, (10, 16, 4, 128))
            # K and V of one block are interleaved: the block stride covers
            # both halves while the inner page stays token-major.
            self.assertEqual(view.stride(), (2 * 16 * 4 * 128, 4 * 128, 128, 1))
        self.assertEqual(v.data_ptr() - k.data_ptr(), 16 * 4 * 128 * ITEMSIZE)

    def test_unrecognized_layouts_fail_fast(self):
        bad = [
            FakeTensor.contiguous([10, 16, 4]),  # 3-D
            FakeTensor.contiguous([10, 2, 16, 4, 128, 2]),  # 6-D
            FakeTensor.contiguous([10, 16, 2, 4, 128]),  # 5-D, K/V dim misplaced
        ]
        for t in bad:
            with self.subTest(shape=t.shape):
                with self.assertRaises(NotImplementedError):
                    attn_kv_views(t)

    def test_ambiguous_layout_fails_fast(self):
        # num_blocks == 2 in a KV-first shape is indistinguishable from a
        # two-block N-first shape; refusing beats guessing.
        with self.assertRaises(NotImplementedError):
            attn_kv_views(FakeTensor.contiguous([2, 2, 16, 4, 128]))


def _make_group_conn():
    conn = make_connector(manager_block_size=16)
    # list instead of the production dict: group_idx == position here.
    conn._self_spec_names = ["tp0_g0"]  # ty: ignore[invalid-assignment]
    conn._device = "cpu"
    return conn


def _attn_meta(layer_names, block_size=16):
    return AttentionGroupMeta(
        group_idx=0, layer_names=layer_names, block_size=block_size, per_block_bytes=0
    )


class TestBuildTransferGroup(unittest.TestCase):
    """Pointer construction per layout. Layer tensors get distinct bases so the
    interleaving [K0, V0, K1, V1, ...] is observable. The pointer list is
    captured by patching ``torch.tensor`` (works with both the stubbed and a
    real torch: no tensor math happens on the captured value)."""

    def _build(self, kv_caches):
        import unittest.mock as mock
        import kv_cache_manager.py_connector.vllm.connector_worker as wc

        conn = _make_group_conn()
        captured = []

        def fake_tensor(data, **kw):
            captured[:] = list(data)
            t = mock.MagicMock()
            t.to.return_value = t
            return t

        with mock.patch.object(wc.torch, "tensor", side_effect=fake_tensor):
            g = conn._build_attention_group(
                _attn_meta(list(kv_caches.keys())), kv_caches
            )
        return g, captured

    def test_packed_one_ptr_per_layer(self):
        kv = {"l0": packed_4d(base=BASE_PTR), "l1": packed_4d(base=2 * BASE_PTR)}
        g, ptrs = self._build(kv)
        self.assertEqual(g.num_kv_ptrs, 2)
        self.assertEqual(g.layer_num, 2)
        self.assertEqual(g.per_token_dim, 4 * 256)
        self.assertEqual(g.kernel_block_size, 16)
        self.assertEqual(g.block_stride, 0)  # flat
        self.assertEqual(ptrs, [BASE_PTR, 2 * BASE_PTR])

    def test_kv_first_two_ptrs_per_layer(self):
        kv = {"l0": kv_first_5d(base=BASE_PTR), "l1": kv_first_5d(base=2 * BASE_PTR)}
        g, ptrs = self._build(kv)
        self.assertEqual(g.num_kv_ptrs, 4)
        self.assertEqual(g.layer_num, 2)
        self.assertEqual(g.per_token_dim, 4 * 128)
        self.assertEqual(g.block_stride, 0)  # each half is flat token-major
        v_off = 10 * 16 * 4 * 128 * ITEMSIZE
        self.assertEqual(
            ptrs, [BASE_PTR, BASE_PTR + v_off, 2 * BASE_PTR, 2 * BASE_PTR + v_off]
        )

    def test_n_first_strided_blocks(self):
        kv = {"l0": n_first_5d(base=BASE_PTR)}
        g, ptrs = self._build(kv)
        self.assertEqual(g.num_kv_ptrs, 2)
        self.assertEqual(g.per_token_dim, 4 * 128)
        # K/V interleaved per block -> kernel must walk the strided path.
        self.assertEqual(g.block_stride, 2 * 16 * 4 * 128)
        v_off = 16 * 4 * 128 * ITEMSIZE
        self.assertEqual(ptrs, [BASE_PTR, BASE_PTR + v_off])

    def test_unrecognized_layout_fails_fast(self):
        with self.assertRaises(NotImplementedError):
            self._build({"l0": FakeTensor.contiguous([10, 16, 4])})


@unittest.skipIf("torch" in STUBBED, "requires real CPU tensors")
class TestStateKvViews(unittest.TestCase):
    """Opaque transfers must respect shared storage and leave other pages alone."""

    def test_shared_layer_views_copy_only_selected_page(self):
        # Two layers interleaved within each block, with a leading offset,
        # page padding and gaps between blocks. C exposes content bytes only.
        raw = torch.arange(224, dtype=torch.uint8)
        caches = {
            name: raw.view(torch.int8).as_strided(
                (3, 1, 1, 20), (64, 20, 20, 1), 16 + layer * 24
            )
            for layer, name in enumerate(("m0", "m1"))
        }
        conn = _make_group_conn()
        group = conn._build_state_group(
            StateGroupMeta(0, list(caches), 16, 48, page_size_bytes=24), caches
        )
        before = raw.clone()
        for layer, view in enumerate(group.block_view_tensors):
            self.assertEqual(view.stride(), (64, 1))
            for block in range(3):
                start = 16 + layer * 24 + block * 64
                self.assertTrue(torch.equal(view[block], before[start : start + 24]))

        # Exercise the same staging copy used by state save/load tasks.
        staged = torch.empty(24, dtype=torch.uint8)
        staged.copy_(group.block_view_tensors[1][2])
        group.block_view_tensors[1][1].copy_(staged)
        expected = before.clone()
        expected[104:128] = before[168:192]
        self.assertTrue(torch.equal(raw, expected))

    def test_legacy_typed_states_preserve_offset_stride_and_padding(self):
        raw = torch.arange(224, dtype=torch.uint8)
        conv = raw.view(torch.int16).as_strided((3, 2, 2), (32, 2, 1), 8)
        ssm = raw.view(torch.float32).as_strided((3, 3), (16, 1), 6)
        view = state_kv_view([conv, ssm], page_size_bytes=24)
        for block in range(3):
            start = 16 + block * 64
            self.assertTrue(torch.equal(view[block], raw[start : start + 24]))
        before = raw.clone()
        view[1].fill_(197)
        expected = before.clone()
        expected[80:104] = 197
        self.assertTrue(torch.equal(raw, expected))

    def test_legacy_contiguous_tuple(self):
        raw = torch.arange(72, dtype=torch.uint8)
        conv = raw.view(torch.int16).as_strided((3, 2, 2), (12, 2, 1))
        ssm = raw.view(torch.float32).as_strided((3, 3), (6, 1), 2)
        self.assertTrue(torch.equal(state_kv_view((conv, ssm), 24), raw.view(3, 24)))

    def test_refuse_invalid_state_layouts(self):
        raw = torch.zeros(20, dtype=torch.int8)
        with self.assertRaisesRegex(ValueError, "shape"):
            state_kv_view(raw, 24)
        # Content fits, but copying the padded page would escape the storage.
        with self.assertRaisesRegex(ValueError, "required end"):
            state_kv_view(raw.as_strided((1, 1, 1, 20), (24, 20, 20, 1)), 24)
        self.assertEqual(raw.untyped_storage().nbytes(), 20)
        with self.assertRaisesRegex(ValueError, "do not share storage"):
            state_kv_view([raw.view(1, 20), raw.clone().view(1, 20)], 20)

    def test_refuse_component_crossing_page_boundary(self):
        raw = torch.arange(64, dtype=torch.uint8)
        conv = raw.as_strided((2, 4), (24, 1))
        ssm = raw.as_strided((2, 8), (24, 1), 20)
        before = raw.clone()
        with self.assertRaisesRegex(ValueError, "component 1.*outside page"):
            state_kv_view([conv, ssm], 24)
        self.assertTrue(torch.equal(raw, before))
        self.assertEqual(raw.untyped_storage().nbytes(), 64)

    def test_refuse_strided_component_crossing_page_boundary(self):
        raw = torch.zeros(128, dtype=torch.uint8)
        conv = raw.as_strided((2, 2, 2), (64, 4, 1), 16)
        # Four elements occupy 15 bytes with these inner strides. Bounding
        # just numel would accept this component, which escapes the page.
        ssm = raw.as_strided((2, 2, 2), (64, 12, 2), 26)
        with self.assertRaisesRegex(ValueError, "component 1.*outside page"):
            state_kv_view((conv, ssm), 24)

    def test_single_block_preserves_offset_without_resizing_storage(self):
        raw = torch.arange(40, dtype=torch.uint8)
        cache = raw.view(torch.int8).as_strided((1, 1, 1, 20), (64, 20, 20, 1), 16)
        view = state_kv_view(cache, 24)
        self.assertTrue(torch.equal(view[0], raw[16:40]))
        self.assertEqual(raw.untyped_storage().nbytes(), 40)

    def test_empty_optional_component_fits_at_page_end(self):
        raw = torch.arange(48, dtype=torch.uint8)
        conv = raw.as_strided((2, 8), (24, 1))
        empty = raw.as_strided((2, 0), (24, 1), 24)
        self.assertTrue(torch.equal(state_kv_view([conv, empty], 24), raw.view(2, 24)))

    def test_refuse_invalid_content_and_block_geometry(self):
        raw = torch.zeros(64, dtype=torch.uint8)
        invalid = [
            (raw.as_strided((2, 1, 1, 10), (24, 20, 20, 2)), 24, "contiguous"),
            (raw.as_strided((2, 1, 1, 20), (20, 20, 20, 1)), 24, "block stride"),
            (torch.empty((0, 1, 1, 20), dtype=torch.uint8), 20, "block count"),
            (raw.view(1, 1, 1, 64), 24, "fit page"),
            (raw.view(1, 1, 1, 64), 0, "positive"),
        ]
        for cache, page_size, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    state_kv_view(cache, page_size)

    def test_refuse_invalid_component_types_and_storage(self):
        for cache in ([], (), [None]):
            with self.subTest(cache=cache):
                with self.assertRaises(TypeError):
                    state_kv_view(cache, 24)  # ty: ignore[invalid-argument-type]
        with self.assertRaisesRegex(ValueError, "block dimension"):
            state_kv_view([torch.tensor(0)], 24)
        with self.assertRaisesRegex(ValueError, "materialized"):
            state_kv_view(
                torch.empty((2, 1, 1, 20), dtype=torch.uint8, device="meta"), 24
            )

    def test_refuse_inconsistent_legacy_block_geometry(self):
        raw = torch.zeros(128, dtype=torch.uint8)
        conv = raw.as_strided((2, 4), (24, 1))
        for ssm in (
            raw.as_strided((3, 4), (24, 1), 4),
            raw.as_strided((2, 4), (32, 1), 4),
        ):
            with self.subTest(shape=ssm.shape, stride=ssm.stride()):
                with self.assertRaisesRegex(ValueError, "share block count"):
                    state_kv_view([conv, ssm], 24)

    def test_worker_error_identifies_state_layer(self):
        conn = _make_group_conn()
        with self.assertRaisesRegex(ValueError, "state layer 'm0'.*shape") as context:
            conn._build_state_group(
                StateGroupMeta(0, ["m0"], 16, 24, page_size_bytes=24),
                {"m0": torch.zeros(24, dtype=torch.uint8)},
            )
        self.assertIsInstance(context.exception.__cause__, ValueError)


class _BlockedScheduler:
    """Mimics vLLM <= 0.22.x: external loads are asserted away."""

    def _mamba_block_aligned_split(
        self,
        request,
        num_new_tokens,
        num_new_local_computed_tokens=0,
        num_external_computed_tokens=0,
    ):
        assert num_external_computed_tokens == 0, (
            "External KV connector is not verified yet"
        )


class _OpenScheduler:
    """Mimics vLLM >= 0.23.0: the split handles external tokens."""

    def _mamba_block_aligned_split(
        self,
        request,
        num_new_tokens,
        num_new_local_computed_tokens=0,
        num_external_computed_tokens=0,
    ):
        return num_new_tokens


# Method exists but inspect.getsource fails (frozen / bytecode-only vLLM):
# compiled from a string, so there is no source file to read.
_exec_ns = {}
exec(
    compile(
        "def _mamba_block_aligned_split(self, *a, **kw):\n    pass\n",
        "<kvcm-test-no-source>",
        "exec",
    ),
    _exec_ns,
)
_SourcelessScheduler = type(
    "_SourcelessScheduler",
    (),
    {"_mamba_block_aligned_split": _exec_ns["_mamba_block_aligned_split"]},
)


class TestHybridGate(unittest.TestCase):
    MOD = "vllm.v1.core.sched.scheduler"

    def _with_scheduler(self, cls):
        mod: Any = types.ModuleType(self.MOD)
        mod.Scheduler = cls
        old = sys.modules.get(self.MOD)
        sys.modules[self.MOD] = mod
        self.addCleanup(
            lambda: (
                sys.modules.pop(self.MOD, None),
                old and sys.modules.__setitem__(self.MOD, old),
            )
        )

    def test_old_vllm_hybrid_raises_gracefully(self):
        self._with_scheduler(_BlockedScheduler)
        with self.assertRaises(NotImplementedError) as ctx:
            ensure_hybrid_supported()
        # The message must tell the operator what to do.
        self.assertIn("vLLM >= 0.23.0", str(ctx.exception))
        self.assertIn("hybrid", str(ctx.exception))

    def test_new_vllm_hybrid_passes(self):
        self._with_scheduler(_OpenScheduler)
        ensure_hybrid_supported()  # must not raise

    def test_method_removed_does_not_block(self):
        # Future vLLM refactors _mamba_block_aligned_split away: the blocking
        # assert went with it, so hybrid must not be blocked.
        self._with_scheduler(object)  # no _mamba_block_aligned_split at all
        ensure_hybrid_supported()  # must not raise

    def test_sourceless_method_fails_closed(self):
        # Method exists but its source is unavailable: the vllm <= 0.22.x
        # blocking assert cannot be ruled out, so the gate must fail closed
        # with an actionable override hint.
        self._with_scheduler(_SourcelessScheduler)
        with self.assertRaises(NotImplementedError) as ctx:
            ensure_hybrid_supported()
        self.assertIn("force_hybrid_support", str(ctx.exception))

    def test_sourceless_method_force_override(self):
        self._with_scheduler(_SourcelessScheduler)
        ensure_hybrid_supported(force=True)  # must not raise

    def test_force_does_not_unblock_known_bad_vllm(self):
        # force only bypasses the *inconclusive* probe; a positively detected
        # blocking assert still raises.
        self._with_scheduler(_BlockedScheduler)
        with self.assertRaises(NotImplementedError):
            ensure_hybrid_supported(force=True)


if __name__ == "__main__":
    unittest.main()
