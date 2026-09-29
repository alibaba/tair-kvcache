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
from typing import Any
import unittest

from kv_cache_manager.py_connector.test.vllm_stubs import make_connector
from kv_cache_manager.py_connector.vllm.vllm_common import AttentionGroupMeta
from kv_cache_manager.py_connector.vllm.v1_connector import (
    attn_kv_views,
    ensure_hybrid_supported,
)
from kv_cache_manager.py_connector.vllm.transfer_types import KVLayout

ITEMSIZE = 2  # bf16/fp16
BASE_PTR = 1 << 20


class FakeTensor:
    """Minimal strided tensor: only what attn_kv_views / _build_attention_group
    read (dim/shape/stride/permute/indexing/data_ptr/element_size/dtype)."""

    def __init__(
        self,
        shape,
        strides,
        offset=0,
        base=BASE_PTR,
        itemsize=ITEMSIZE,
        dtype="bfloat16",
    ):
        self.shape = tuple(shape)
        self._strides = tuple(strides)
        self._offset = offset
        self._base = base
        self._itemsize = itemsize
        self.dtype = dtype

    @classmethod
    def contiguous(cls, shape, base=BASE_PTR, itemsize=ITEMSIZE, dtype="bfloat16"):
        strides, acc = [], 1
        for s in reversed(shape):
            strides.append(acc)
            acc *= s
        return cls(
            shape,
            tuple(reversed(strides)),
            base=base,
            itemsize=itemsize,
            dtype=dtype,
        )

    def dim(self):
        return len(self.shape)

    def stride(self, i=None):
        return self._strides if i is None else self._strides[i]

    def element_size(self):
        return self._itemsize

    def data_ptr(self):
        return self._base + self._offset * self._itemsize

    def permute(self, *dims):
        return FakeTensor(
            [self.shape[d] for d in dims],
            [self._strides[d] for d in dims],
            self._offset,
            self._base,
            self._itemsize,
            self.dtype,
        )

    def unsqueeze(self, dim):
        # Torch gives the inserted size-1 dim the preceding dim's stride
        # (the whole-tensor extent for dim 0); any value is addressable, but
        # mirror torch so stride assertions see production values.
        shape = list(self.shape)
        strides = list(self._strides)
        new_stride = shape[0] * strides[0] if dim == 0 else strides[dim - 1]
        shape.insert(dim, 1)
        strides.insert(dim, new_stride)
        return FakeTensor(
            shape, strides, self._offset, self._base, self._itemsize, self.dtype
        )

    def __getitem__(self, idx):
        if isinstance(idx, int):  # t[i]: drop dim 0
            return FakeTensor(
                self.shape[1:],
                self._strides[1:],
                self._offset + idx * self._strides[0],
                self._base,
                self._itemsize,
                self.dtype,
            )
        if isinstance(idx, tuple) and idx[0] == slice(None) and isinstance(idx[1], int):
            # t[:, i]: drop dim 1
            return FakeTensor(
                self.shape[:1] + self.shape[2:],
                self._strides[:1] + self._strides[2:],
                self._offset + idx[1] * self._strides[1],
                self._base,
                self._itemsize,
                self.dtype,
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


def mla_3d(n=10, b=16, d=576, base=BASE_PTR):
    """MLA latent cache (all eras): (n, b, d) contiguous, one latent vector
    per token (num_kv_heads == 1, no K/V split)."""
    return FakeTensor.contiguous([n, b, d], base=base)


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

    def test_mla_3d(self):
        views, layout = attn_kv_views(mla_3d())
        self.assertIs(layout, KVLayout.MLA_3D)
        self.assertEqual(len(views), 1)
        v = views[0]
        # Single implicit head: (n, b, 1, d) with the original data_ptr.
        self.assertEqual(v.shape, (10, 16, 1, 576))
        self.assertEqual(v.stride(), (16 * 576, 576, 576, 1))
        self.assertEqual(v.data_ptr(), BASE_PTR)

    def test_mla_3d_padded_pages(self):
        # Padded MLA pages (page_size_padded): block stride != b*d keeps the
        # token-major inner page, exactly like the N-first split layout.
        d, b, n, pad = 576, 16, 10, 3 * 576
        t = FakeTensor([n, b, d], [b * d + pad, d, 1])
        views, layout = attn_kv_views(t)  # ty: ignore[invalid-argument-type]
        self.assertIs(layout, KVLayout.MLA_3D)
        v = views[0]
        self.assertEqual(v.shape, (10, 16, 1, 576))
        self.assertEqual(v.stride(), (b * d + pad, 576, 576, 1))

    def test_unrecognized_layouts_fail_fast(self):
        bad = [
            FakeTensor.contiguous([10, 16]),  # 2-D
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


class TestFakeTensorItemsize(unittest.TestCase):
    """The fixture's byte math backs every stride/size assertion below:
    data_ptr() must scale with the element size (uint8 fp8 rows vs bf16)."""

    def test_uint8_rows_are_byte_offset(self):
        t = FakeTensor.contiguous([10, 64, 656], itemsize=1)
        self.assertEqual(t.element_size(), 1)
        self.assertEqual(t[1].data_ptr() - t[0].data_ptr(), 64 * 656)

    def test_default_itemsize_is_two_bytes(self):
        t = FakeTensor.contiguous([10, 16, 576])
        self.assertEqual(t.element_size(), 2)
        self.assertEqual(t[1].data_ptr() - t[0].data_ptr(), 16 * 576 * 2)


def _make_group_conn():
    conn = make_connector(manager_block_size=16)
    conn._device = "cpu"
    return conn


def _attn_meta(layer_names, block_size=16, page_bytes=18432, manager_block_size=16):
    """One attention bucket whose spec page matches the MLA bf16 tensor used
    by most cases here (16 x 576 x 2 B)."""
    return AttentionGroupMeta(
        group_idx=0,
        layer_names=layer_names,
        block_size=block_size,
        per_block_bytes=page_bytes
        // block_size
        * manager_block_size
        * len(layer_names),
        page_bytes=page_bytes,
    )


class TestBuildTransferGroup(unittest.TestCase):
    """Pointer construction per layout. Layer tensors get distinct bases so the
    interleaving [K0, V0, K1, V1, ...] is observable. The pointer list is
    captured by patching ``torch.tensor`` (works with both the stubbed and a
    real torch: no tensor math happens on the captured value)."""

    def _build(self, kv_caches, block_size=16, page_bytes=18432, mbs=16):
        import unittest.mock as mock
        import kv_cache_manager.py_connector.vllm.connector_worker as wc

        conn = make_connector(manager_block_size=mbs)
        conn._device = "cpu"
        captured = []

        def fake_tensor(data, **kw):
            captured[:] = list(data)
            t = mock.MagicMock()
            t.to.return_value = t
            return t

        with mock.patch.object(wc.torch, "tensor", side_effect=fake_tensor):
            g = conn._build_attention_group(
                _attn_meta(
                    list(kv_caches.keys()),
                    block_size=block_size,
                    page_bytes=page_bytes,
                    manager_block_size=mbs,
                ),
                kv_caches,
            )
        return g, captured

    def test_packed_one_ptr_per_layer(self):
        kv = {"l0": packed_4d(base=BASE_PTR), "l1": packed_4d(base=2 * BASE_PTR)}
        g, ptrs = self._build(kv, page_bytes=32768)
        self.assertEqual(g.num_kv_ptrs, 2)
        self.assertEqual(g.layer_num, 2)
        self.assertEqual(g.per_token_dim, 4 * 256)
        self.assertEqual(g.kernel_block_size, 16)
        self.assertEqual(g.block_stride, 0)  # flat
        self.assertEqual(ptrs, [BASE_PTR, 2 * BASE_PTR])

    def test_kv_first_two_ptrs_per_layer(self):
        kv = {"l0": kv_first_5d(base=BASE_PTR), "l1": kv_first_5d(base=2 * BASE_PTR)}
        # Split K/V: two views of 16 x 512 elements -> a 32768-byte page.
        g, ptrs = self._build(kv, page_bytes=32768)
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
        g, ptrs = self._build(kv, page_bytes=32768)
        self.assertEqual(g.num_kv_ptrs, 2)
        self.assertEqual(g.per_token_dim, 4 * 128)
        # K/V interleaved per block -> kernel must walk the strided path.
        self.assertEqual(g.block_stride, 2 * 16 * 4 * 128)
        v_off = 16 * 4 * 128 * ITEMSIZE
        self.assertEqual(ptrs, [BASE_PTR, BASE_PTR + v_off])

    def test_mla_one_ptr_per_layer(self):
        # The latent cache is a single flat (n, b, d) per layer: one pointer,
        # per-token dim is the full latent vector, flat block stride.
        kv = {"l0": mla_3d(base=BASE_PTR), "l1": mla_3d(base=2 * BASE_PTR)}
        g, ptrs = self._build(kv)
        self.assertEqual(g.num_kv_ptrs, 2)
        self.assertEqual(g.layer_num, 2)
        self.assertEqual(g.per_token_dim, 576)
        self.assertEqual(g.kernel_block_size, 16)
        self.assertEqual(g.block_stride, 0)  # flat
        self.assertEqual(g.kv_layout, KVLayout.MLA_3D)
        self.assertEqual(ptrs, [BASE_PTR, 2 * BASE_PTR])

    def test_mla_padded_pages_strided(self):
        d, b, n, pad = 576, 16, 10, 3 * 576
        kv = {"l0": FakeTensor([n, b, d], [b * d + pad, d, 1])}
        g, _ = self._build(kv)
        self.assertEqual(g.per_token_dim, 576)
        self.assertEqual(g.block_stride, b * d + pad)  # skip the pad gap

    def test_unrecognized_layout_fails_fast(self):
        with self.assertRaises(NotImplementedError):
            self._build({"l0": FakeTensor.contiguous([10, 16])})

    def test_mla_fp8_ds_packed_uint8(self):
        # V3.2 fp8_ds_mla main layer: 656 B/token, one uint8 row per token,
        # flat pages -> itemsize 1 (the byte count is the element count).
        kv = {"l0": FakeTensor.contiguous([10, 64, 656], itemsize=1, dtype="uint8")}
        g, _ = self._build(kv, block_size=64, page_bytes=41984)
        self.assertEqual(g.kernel_block_size, 64)
        self.assertEqual(g.per_token_dim, 656)
        self.assertEqual(g.block_stride, 0)  # 64 * 656 == stride(0)
        self.assertEqual(g.dtype, "uint8")

    def test_mla_padded_fp8_ds_packed_pages(self):
        # V4 fp8_ds_mla (584 B/row): real 37376, the 64-byte alignment tail
        # lives at the end of the page, so the stride must skip it and the
        # spec page must stay the compact size.
        kv = {"l0": FakeTensor([10, 64, 584], [37440, 584, 1], itemsize=1)}
        g, _ = self._build(kv, block_size=256, page_bytes=37376)
        self.assertEqual(g.kernel_block_size, 64)
        self.assertEqual(g.per_token_dim, 584)
        self.assertEqual(g.block_stride, 37440)
        self.assertEqual(g.dtype, "bfloat16")  # the fixture's default dtype

    def test_mla_fp8_ds_c128_pages(self):
        # V4 c128a: two storage rows per page, 560 bytes of pad after them.
        kv = {"l0": FakeTensor([10, 2, 584], [1728, 584, 1], itemsize=1)}
        g, _ = self._build(kv, block_size=256, page_bytes=1168)
        self.assertEqual(g.kernel_block_size, 2)
        self.assertEqual(g.block_stride, 1728)

    def test_mla_indexer_padded_pages(self):
        # V4 indexer: 132 B/row, page padded 8448 -> 8640.
        kv = {"l0": FakeTensor([10, 64, 132], [8640, 132, 1], itemsize=1)}
        g, _ = self._build(kv, block_size=256, page_bytes=8448)
        self.assertEqual(g.kernel_block_size, 64)
        self.assertEqual(g.per_token_dim, 132)
        self.assertEqual(g.block_stride, 8640)

    def test_mla_indexer_flat_pages(self):
        # Real V3.2 indexer: no alignment, so the pages are contiguous and
        # the kernel takes the flat path (block_stride 0).
        kv = {"l0": FakeTensor.contiguous([10, 64, 132], itemsize=1)}
        g, _ = self._build(kv, block_size=64, page_bytes=8448)
        self.assertEqual(g.kernel_block_size, 64)
        self.assertEqual(g.per_token_dim, 132)
        self.assertEqual(g.block_stride, 0)

    def test_hnd_layout_rejected(self):
        # HND interleaves heads across tokens: the gather kernel needs NHD.
        kv = {"l0": FakeTensor([10, 16, 576], [18432, 1, 16], itemsize=2)}
        with self.assertRaises(AssertionError) as ctx:
            self._build(kv)
        self.assertIn("not token-major", str(ctx.exception))
        self.assertIn("VLLM_KV_CACHE_LAYOUT=NHD", str(ctx.exception))

    def test_layers_must_share_shape_and_stride(self):
        kv = {
            "l0": FakeTensor.contiguous([10, 16, 576]),
            "l1": FakeTensor([10, 16, 576], [16 * 576 + 576, 576, 1]),
        }
        with self.assertRaises(AssertionError) as ctx:
            self._build(kv)
        self.assertIn("share shape/stride", str(ctx.exception))

    def test_spec_page_bytes_must_match_the_tensor(self):
        # The gate reads the spec; the tensors are the backend's truth. A
        # mismatch (here: an X1-style 167936-byte spec against a 18432-byte
        # bf16 page) must fail at register_kv_caches, not silently mis-size
        # every location.
        kv = {"l0": mla_3d()}
        with self.assertRaises(NotImplementedError) as ctx:
            self._build(kv, page_bytes=167936)
        self.assertIn("167936", str(ctx.exception))
        self.assertIn("18432", str(ctx.exception))

    def test_backend_without_packed_scales_rejected(self):
        # A backend that stores the fp8 rows *without* the in-row scale
        # (512 B instead of the spec's 584) is a different layout than the
        # spec declares; refuse instead of transferring misaligned bytes.
        kv = {"l0": FakeTensor([10, 16, 512], [8192, 512, 1], itemsize=1)}
        with self.assertRaises(NotImplementedError) as ctx:
            self._build(kv, page_bytes=9344)
        self.assertIn("9344", str(ctx.exception))
        self.assertIn("8192", str(ctx.exception))

    def test_split_kv_page_counted_twice(self):
        # Regression: the cross-check must count *both* views of a split K/V
        # layout, otherwise every vLLM <= 0.25 page looks half-sized.
        kv = {"l0": kv_first_5d()}
        g, _ = self._build(kv, page_bytes=32768)
        self.assertEqual(g.num_kv_ptrs, 2)

    def test_per_block_bytes_account_for_the_staged_rows(self):
        # E1: the manager location's per-block bytes are exactly what the
        # staging view stages -- one row set per transfer pointer, a manager
        # block of rows, the bucket's per-token dim and element size.
        for kv, block_size, page_bytes, mbs, itemsize in [
            ({"l0": mla_3d()}, 16, 18432, 16, 2),
            ({"l0": FakeTensor.contiguous([10, 64, 656], itemsize=1)}, 64, 41984, 64, 1),
            (
                {"l0": FakeTensor.contiguous([10, 64, 132], itemsize=1)},
                64,
                8448,
                128,
                1,
            ),
        ]:
            with self.subTest(page_bytes=page_bytes):
                g, _ = self._build(
                    kv, block_size=block_size, page_bytes=page_bytes, mbs=mbs
                )
                self.assertEqual(
                    g.per_block_bytes,
                    g.num_kv_ptrs * mbs * g.per_token_dim * itemsize,
                )


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
