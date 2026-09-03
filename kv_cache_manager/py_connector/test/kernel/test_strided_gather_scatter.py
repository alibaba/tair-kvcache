"""GPU tests for the strided path of the batch gather/scatter Triton kernel.

The flat path (block_stride=0) is covered by test_batch_gather_scatter.py.
Here we cover the strided path added for vLLM's paged layout, where the flat
token index is decomposed as (kv_block, token_in_block) and the block starts
``block_stride`` elements apart -- including padded pages where
``block_stride > local_block_size * dims_per_token`` leaves a gap between
blocks that must be skipped, not walked.

Every case is checked element-wise against a naive torch reference that
performs the same (kv_block, token) decomposition with plain indexing.
"""

import unittest

import torch

from kv_cache_manager.py_connector.kernel.batch_gather_scatter_helper import (
    batch_gather_kv_caches,
    batch_scatter_kv_caches,
)


def _make_paged_caches(num_layers, num_blocks, local_block_size, dims_per_token,
                       pad_tokens, device, dtype, fill_random=True):
    """Per-layer paged caches shaped (num_blocks, padded_tokens, dims) where
    padded_tokens = local_block_size + pad_tokens. block_stride (in elements)
    is padded_tokens * dims_per_token."""
    caches = []
    for _ in range(num_layers):
        t = torch.randn(num_blocks, local_block_size + pad_tokens, dims_per_token,
                        device=device, dtype=dtype) if fill_random else \
            torch.zeros(num_blocks, local_block_size + pad_tokens, dims_per_token,
                        device=device, dtype=dtype)
        caches.append(t)
    return caches


def _ref_slot(cache, flat_token_idx, local_block_size):
    blk = flat_token_idx // local_block_size
    tok = flat_token_idx % local_block_size
    return cache[blk, tok, :]


class TestStridedGatherScatter(unittest.TestCase):
    # (local_block_size, pad_tokens, tokens_per_manager_block)
    CASES = [
        (16, 0, 16),   # strided == flat geometry (stride still exercised)
        (16, 4, 16),   # padded pages: gap between blocks
        (64, 0, 528),  # hybrid attention: manager block spans many kv blocks
        (64, 8, 48),   # padded + manager block not aligned to kv block
    ]

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("requires a GPU")
        torch.manual_seed(7)
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.num_layers = 3
        self.dims = 128
        self.num_kv_blocks = 64

    def _indices(self, num_manager_blocks, tokens_per_block, local_block_size):
        total_tokens = self.num_kv_blocks * local_block_size
        need = num_manager_blocks * tokens_per_block
        assert need <= total_tokens, "test setup: not enough kv slots"
        perm = torch.randperm(total_tokens)[:need]
        return perm.tolist()

    def test_gather_strided_matches_reference(self):
        for local_bs, pad, tokens_per_block in self.CASES:
            with self.subTest(local_bs=local_bs, pad=pad, tpb=tokens_per_block):
                caches = _make_paged_caches(
                    self.num_layers, self.num_kv_blocks, local_bs, self.dims,
                    pad, self.device, self.dtype)
                block_stride = caches[0].stride(0)
                self.assertEqual(block_stride, (local_bs + pad) * self.dims)
                ptrs = torch.tensor([c.data_ptr() for c in caches],
                                    device=self.device, dtype=torch.int64)
                num_mb = 4
                token_indices = self._indices(num_mb, tokens_per_block, local_bs)
                dst_block_indices = [2, 0, 3, 1]
                dst = torch.zeros(num_mb, self.num_layers, tokens_per_block,
                                  self.dims, device="cpu", dtype=self.dtype,
                                  pin_memory=True)
                batch_gather_kv_caches(
                    ptrs, dst, token_indices, dst_block_indices,
                    tokens_per_block, self.dims,
                    block_stride=block_stride, local_block_size=local_bs)
                torch.cuda.synchronize()

                caches_cpu = [c.cpu() for c in caches]
                for mb in range(num_mb):
                    for pos in range(tokens_per_block):
                        flat_idx = token_indices[mb * tokens_per_block + pos]
                        for layer in range(self.num_layers):
                            want = _ref_slot(caches_cpu[layer], flat_idx, local_bs)
                            got = dst[dst_block_indices[mb], layer, pos, :]
                            torch.testing.assert_close(
                                got, want,
                                msg=f"gather mismatch mb={mb} pos={pos} "
                                    f"layer={layer} flat={flat_idx}")

    def test_scatter_strided_matches_reference(self):
        for local_bs, pad, tokens_per_block in self.CASES:
            with self.subTest(local_bs=local_bs, pad=pad, tpb=tokens_per_block):
                caches = _make_paged_caches(
                    self.num_layers, self.num_kv_blocks, local_bs, self.dims,
                    pad, self.device, self.dtype, fill_random=False)
                # Sentinel in the padding region: scatter must never touch it.
                sentinel = 123.0
                if pad:
                    for c in caches:
                        c[:, local_bs:, :] = sentinel
                block_stride = caches[0].stride(0)
                ptrs = torch.tensor([c.data_ptr() for c in caches],
                                    device=self.device, dtype=torch.int64)
                num_mb = 4
                token_indices = self._indices(num_mb, tokens_per_block, local_bs)
                src_block_indices = [1, 3, 0, 2]
                src = torch.randn(num_mb, self.num_layers, tokens_per_block,
                                  self.dims, dtype=self.dtype).pin_memory()
                batch_scatter_kv_caches(
                    ptrs, src, token_indices, src_block_indices,
                    tokens_per_block, self.dims,
                    block_stride=block_stride, local_block_size=local_bs)
                torch.cuda.synchronize()

                caches_cpu = [c.cpu() for c in caches]
                for mb in range(num_mb):
                    for pos in range(tokens_per_block):
                        flat_idx = token_indices[mb * tokens_per_block + pos]
                        for layer in range(self.num_layers):
                            got = _ref_slot(caches_cpu[layer], flat_idx, local_bs)
                            want = src[src_block_indices[mb], layer, pos, :]
                            torch.testing.assert_close(
                                got, want,
                                msg=f"scatter mismatch mb={mb} pos={pos} "
                                    f"layer={layer} flat={flat_idx}")
                if pad:
                    for layer, c in enumerate(caches_cpu):
                        self.assertTrue(
                            bool((c[:, local_bs:, :] == sentinel).all()),
                            f"scatter wrote into the padding of layer {layer}")

    def test_gather_scatter_roundtrip_strided(self):
        """Scattering gathered data into zeroed caches must reproduce exactly
        the gathered slots (and only them)."""
        local_bs, pad, tokens_per_block = 64, 8, 48
        src_caches = _make_paged_caches(
            self.num_layers, self.num_kv_blocks, local_bs, self.dims,
            pad, self.device, self.dtype)
        dst_caches = _make_paged_caches(
            self.num_layers, self.num_kv_blocks, local_bs, self.dims,
            pad, self.device, self.dtype, fill_random=False)
        block_stride = src_caches[0].stride(0)
        src_ptrs = torch.tensor([c.data_ptr() for c in src_caches],
                                device=self.device, dtype=torch.int64)
        dst_ptrs = torch.tensor([c.data_ptr() for c in dst_caches],
                                device=self.device, dtype=torch.int64)
        num_mb = 3
        token_indices = self._indices(num_mb, tokens_per_block, local_bs)
        buf = torch.zeros(num_mb, self.num_layers, tokens_per_block, self.dims,
                          device="cpu", dtype=self.dtype, pin_memory=True)
        batch_gather_kv_caches(
            src_ptrs, buf, token_indices, list(range(num_mb)),
            tokens_per_block, self.dims,
            block_stride=block_stride, local_block_size=local_bs)
        torch.cuda.synchronize()
        batch_scatter_kv_caches(
            dst_ptrs, buf, token_indices, list(range(num_mb)),
            tokens_per_block, self.dims,
            block_stride=block_stride, local_block_size=local_bs)
        torch.cuda.synchronize()
        src_cpu = [c.cpu() for c in src_caches]
        dst_cpu = [c.cpu() for c in dst_caches]
        for flat_idx in token_indices:
            for layer in range(self.num_layers):
                torch.testing.assert_close(
                    _ref_slot(dst_cpu[layer], flat_idx, local_bs),
                    _ref_slot(src_cpu[layer], flat_idx, local_bs))


if __name__ == "__main__":
    unittest.main()
