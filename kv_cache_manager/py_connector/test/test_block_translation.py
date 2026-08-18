"""Unit tests for the connector's manager-block -> physical-slot translation.

Covers ``_attn_token_indices`` (attention groups: token-granular three-tier
mapping) and ``_state_block_ids`` (mamba/state groups: manager block's last
token selects the group block), verifying against an independent brute-force
reference implementation, token by token.
"""

import unittest

from kv_cache_manager.py_connector.test.vllm_stubs import make_connector
from kv_cache_manager.py_connector.vllm.transfer_types import (
    AttentionTransferGroup, KVLayout, StateTransferGroup)


def _make_group(group_bs, kernel_bs=0, is_attention=True):
    common = dict(group_idx=0, spec_name="tp0_g0", layer_names=["layer0"],
                  block_size=group_bs, per_block_bytes=0, layer_num=1)
    if is_attention:
        return AttentionTransferGroup(
            kv_layout=KVLayout.PACKED_4D, kvcache_ptr_tensor_gpu=None,
            num_kv_ptrs=1, per_token_dim=8, kernel_block_size=kernel_bs,
            block_stride=0, **common)
    return StateTransferGroup(
        block_view_tensors=[], page_size_bytes=0, **common)


def _ref_attn_token_indices(manager_bs, group_bs, kernel_bs, manager_block_idxes,
                            block_table):
    """Brute-force reference: walk every token of every manager block and map it
    through the block hierarchy step by step."""
    out = []
    for mb in manager_block_idxes:
        slots = []
        for tok in range(mb * manager_bs, (mb + 1) * manager_bs):
            group_block = tok // group_bs          # logical block in group table
            tok_in_group = tok - group_block * group_bs
            kernel_in_group = tok_in_group // kernel_bs
            tok_in_kernel = tok_in_group - kernel_in_group * kernel_bs
            physical = block_table[group_block] * (group_bs // kernel_bs) + kernel_in_group
            slots.append(physical * kernel_bs + tok_in_kernel)
        out.append(slots)
    return out


def _ref_state_block_ids(manager_bs, group_bs, manager_block_idxes, block_table):
    """Brute-force reference: the state covering a manager block is the state of
    the group block containing the manager block's last token."""
    out = []
    for mb in manager_block_idxes:
        last_token = (mb + 1) * manager_bs - 1
        out.append(block_table[last_token // group_bs])
    return out


class TestAttnTokenIndices(unittest.TestCase):
    # (manager_bs, group_bs, kernel_bs): ratio=1, ratio>1, manager != group.
    CASES = [
        (16, 16, 16),    # full attention default: all equal
        (32, 16, 16),    # preferred_block_size > vllm block size
        (528, 528, 64),  # hybrid: group block spans several kernel blocks
        (528, 528, 528), # hybrid with kernel == group
        (48, 16, 8),     # manager > group > kernel
    ]

    def test_against_reference(self):
        for manager_bs, group_bs, kernel_bs in self.CASES:
            with self.subTest(manager_bs=manager_bs, group_bs=group_bs,
                              kernel_bs=kernel_bs):
                conn = make_connector(manager_block_size=manager_bs)
                group = _make_group(group_bs, kernel_bs)
                # Enough non-trivially permuted blocks for 4 manager blocks.
                needed = 4 * manager_bs // group_bs + 1
                block_table = [(i * 7 + 3) % 97 for i in range(needed)]
                mbis = [0, 1, 3]
                got = conn._attn_token_indices(group, mbis, block_table)
                want = _ref_attn_token_indices(
                    manager_bs, group_bs, kernel_bs, mbis, block_table)
                self.assertEqual(got, want)

    def test_manual_example(self):
        # manager_bs=4, group_bs=2, kernel_bs=2; block_table maps logical
        # blocks 0..3 -> physical 5,2,9,0. Manager block 1 covers tokens 4..7 ->
        # logical blocks 2,3 -> physical 9,0 -> slots 18,19,0,1.
        conn = make_connector(manager_block_size=4)
        group = _make_group(group_bs=2, kernel_bs=2)
        got = conn._attn_token_indices(group, [1], [5, 2, 9, 0])
        self.assertEqual(got, [[18, 19, 0, 1]])

    def test_out_of_range_asserts(self):
        conn = make_connector(manager_block_size=16)
        group = _make_group(group_bs=16, kernel_bs=16)
        with self.assertRaises(AssertionError):
            conn._attn_token_indices(group, [1], [0])  # table too short


class TestStateBlockIds(unittest.TestCase):
    def test_against_reference(self):
        for manager_bs, group_bs in [(528, 528), (16, 16), (16, 32), (48, 16)]:
            with self.subTest(manager_bs=manager_bs, group_bs=group_bs):
                conn = make_connector(manager_block_size=manager_bs)
                group = _make_group(group_bs, is_attention=False)
                needed = 4 * manager_bs // group_bs + 1
                block_table = [(i * 11 + 5) % 89 for i in range(needed)]
                mbis = [0, 1, 3]
                got = conn._state_block_ids(group, mbis, block_table)
                want = _ref_state_block_ids(manager_bs, group_bs, mbis, block_table)
                self.assertEqual(got, want)

    def test_manual_example(self):
        # manager_bs=4, group_bs=8: manager blocks 0 and 1 both end inside group
        # block 0; manager block 2 ends in group block 1.
        conn = make_connector(manager_block_size=4)
        group = _make_group(group_bs=8, is_attention=False)
        got = conn._state_block_ids(group, [0, 1, 2], [7, 3])
        self.assertEqual(got, [7, 7, 3])

    def test_out_of_range_asserts(self):
        conn = make_connector(manager_block_size=16)
        group = _make_group(group_bs=16, is_attention=False)
        with self.assertRaises(AssertionError):
            conn._state_block_ids(group, [2], [0, 1])


if __name__ == "__main__":
    unittest.main()
