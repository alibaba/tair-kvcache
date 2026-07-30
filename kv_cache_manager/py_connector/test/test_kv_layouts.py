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

Runs without torch: a minimal FakeTensor models the strided-view semantics
(shape / stride / offset / data_ptr) that the code under test reads.
"""

import sys
import types
import unittest

from kv_cache_manager.py_connector.test.vllm_stubs import make_connector
from kv_cache_manager.py_connector.vllm.v1_connector import (
    attn_kv_views, ensure_hybrid_supported, GroupMeta)

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
        return FakeTensor([self.shape[d] for d in dims],
                          [self._strides[d] for d in dims],
                          self._offset, self._base)

    def __getitem__(self, idx):
        if isinstance(idx, int):  # t[i]: drop dim 0
            return FakeTensor(self.shape[1:], self._strides[1:],
                              self._offset + idx * self._strides[0], self._base)
        if isinstance(idx, tuple) and idx[0] == slice(None) and isinstance(idx[1], int):
            # t[:, i]: drop dim 1
            return FakeTensor(self.shape[:1] + self.shape[2:],
                              self._strides[:1] + self._strides[2:],
                              self._offset + idx[1] * self._strides[1], self._base)
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
        views = attn_kv_views(packed_4d())
        self.assertEqual(len(views), 1)
        v = views[0]
        self.assertEqual(v.shape, (10, 16, 4, 256))          # (n, b, h, 2d)
        self.assertEqual(v.stride(), (16 * 4 * 256, 4 * 256, 256, 1))
        self.assertEqual(v.data_ptr(), BASE_PTR)             # storage base

    def test_kv_first_5d(self):
        views = attn_kv_views(kv_first_5d())
        self.assertEqual(len(views), 2)
        k, v = views
        for view in (k, v):
            self.assertEqual(view.shape, (10, 16, 4, 128))
            self.assertEqual(view.stride(), (16 * 4 * 128, 4 * 128, 128, 1))
        self.assertEqual(k.data_ptr(), BASE_PTR)
        # V base = K base + num_blocks * block * h * d elements.
        self.assertEqual(v.data_ptr() - k.data_ptr(),
                         10 * 16 * 4 * 128 * ITEMSIZE)

    def test_n_first_5d(self):
        views = attn_kv_views(n_first_5d())
        self.assertEqual(len(views), 2)
        k, v = views
        for view in (k, v):
            self.assertEqual(view.shape, (10, 16, 4, 128))
            # K and V of one block are interleaved: the block stride covers
            # both halves while the inner page stays token-major.
            self.assertEqual(view.stride(), (2 * 16 * 4 * 128, 4 * 128, 128, 1))
        self.assertEqual(v.data_ptr() - k.data_ptr(),
                         16 * 4 * 128 * ITEMSIZE)

    def test_unrecognized_layouts_fail_fast(self):
        bad = [
            FakeTensor.contiguous([10, 16, 4]),           # 3-D
            FakeTensor.contiguous([10, 2, 16, 4, 128, 2]),  # 6-D
            FakeTensor.contiguous([10, 16, 2, 4, 128]),   # 5-D, K/V dim misplaced
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
    conn._self_spec_names = ["tp0_g0"]
    conn._device = "cpu"
    return conn


def _attn_meta(layer_names, block_size=16):
    return GroupMeta(group_idx=0, is_attention=True, layer_names=layer_names,
                     block_size=block_size, per_block_bytes=0)


class TestBuildTransferGroup(unittest.TestCase):
    """Pointer construction per layout. Layer tensors get distinct bases so the
    interleaving [K0, V0, K1, V1, ...] is observable. The pointer list is
    captured by patching ``torch.tensor`` (works with both the stubbed and a
    real torch: no tensor math happens on the captured value)."""

    def _build(self, kv_caches):
        import unittest.mock as mock
        import kv_cache_manager.py_connector.vllm.v1_connector as v1c
        conn = _make_group_conn()
        captured = []

        def fake_tensor(data, **kw):
            captured[:] = list(data)
            t = mock.MagicMock()
            t.to.return_value = t
            return t

        with mock.patch.object(v1c.torch, "tensor", side_effect=fake_tensor):
            g = conn._build_transfer_group(
                _attn_meta(list(kv_caches.keys())), kv_caches)
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
        self.assertEqual(ptrs, [BASE_PTR, BASE_PTR + v_off,
                                2 * BASE_PTR, 2 * BASE_PTR + v_off])

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


class _BlockedScheduler:
    """Mimics vLLM <= 0.22.x: external loads are asserted away."""

    def _mamba_block_aligned_split(self, request, num_new_tokens,
                                   num_new_local_computed_tokens=0,
                                   num_external_computed_tokens=0):
        assert num_external_computed_tokens == 0, (
            "External KV connector is not verified yet"
        )


class _OpenScheduler:
    """Mimics vLLM >= 0.23.0: the split handles external tokens."""

    def _mamba_block_aligned_split(self, request, num_new_tokens,
                                   num_new_local_computed_tokens=0,
                                   num_external_computed_tokens=0):
        return num_new_tokens


# Method exists but inspect.getsource fails (frozen / bytecode-only vLLM):
# compiled from a string, so there is no source file to read.
_exec_ns = {}
exec(compile("def _mamba_block_aligned_split(self, *a, **kw):\n    pass\n",
             "<kvcm-test-no-source>", "exec"), _exec_ns)
_SourcelessScheduler = type(
    "_SourcelessScheduler", (),
    {"_mamba_block_aligned_split": _exec_ns["_mamba_block_aligned_split"]})


class TestHybridGate(unittest.TestCase):
    MOD = "vllm.v1.core.sched.scheduler"

    def _with_scheduler(self, cls):
        mod = types.ModuleType(self.MOD)
        mod.Scheduler = cls
        old = sys.modules.get(self.MOD)
        sys.modules[self.MOD] = mod
        self.addCleanup(lambda: (sys.modules.pop(self.MOD, None),
                                 old and sys.modules.__setitem__(self.MOD, old)))

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
