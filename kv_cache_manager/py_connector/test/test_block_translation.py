"""Unit tests for the connector's manager-block -> physical-slot translation.

Covers ``_attn_token_indices`` (attention groups: token-granular three-tier
mapping) and ``_state_block_ids`` (mamba/state groups: manager block's last
token selects the group block), verifying against an independent brute-force
reference implementation, token by token.
"""

import unittest
from typing import Any, Dict

# vllm_stubs must be imported before torch: in the open-source CI (no torch
# installed) it registers the MagicMock stand-in that the bare `import torch`
# below then resolves to.
from kv_cache_manager.py_connector.test.vllm_stubs import make_connector

import torch
from kv_cache_manager.py_connector.vllm.transfer_types import (
    AttentionTransferGroup,
    KVLayout,
    StateTransferGroup,
)


def _make_group(group_bs, kernel_bs=0, is_attention=True):
    common: Dict[str, Any] = dict(
        group_idx=0,
        spec_name="tp0_g0",
        layer_names=["layer0"],
        block_size=group_bs,
        per_block_bytes=0,
        layer_num=1,
    )
    if is_attention:
        return AttentionTransferGroup(
            kv_layout=KVLayout.PACKED_4D,
            # No real device pointers in the pure translation tests.
            kvcache_ptr_tensor_gpu=None,  # ty: ignore[invalid-argument-type]
            num_kv_ptrs=1,
            per_token_dim=8,
            kernel_block_size=kernel_bs,
            block_stride=0,
            dtype=torch.bfloat16,
            **common,
        )
    return StateTransferGroup(block_view_tensors=[], page_size_bytes=0, **common)


def _ref_attn_token_indices(
    manager_bs, group_bs, kernel_bs, manager_block_idxes, block_table
):
    """Brute-force reference: walk every token of every manager block and map it
    through the block hierarchy step by step."""
    out = []
    for mb in manager_block_idxes:
        slots = []
        for tok in range(mb * manager_bs, (mb + 1) * manager_bs):
            group_block = tok // group_bs  # logical block in group table
            tok_in_group = tok - group_block * group_bs
            kernel_in_group = tok_in_group // kernel_bs
            tok_in_kernel = tok_in_group - kernel_in_group * kernel_bs
            physical = (
                block_table[group_block] * (group_bs // kernel_bs) + kernel_in_group
            )
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
        (16, 16, 16),  # full attention default: all equal
        (32, 16, 16),  # preferred_block_size > vllm block size
        (528, 528, 64),  # hybrid: group block spans several kernel blocks
        (528, 528, 528),  # hybrid with kernel == group
        (48, 16, 8),  # manager > group > kernel
    ]

    def test_against_reference(self):
        for manager_bs, group_bs, kernel_bs in self.CASES:
            with self.subTest(
                manager_bs=manager_bs, group_bs=group_bs, kernel_bs=kernel_bs
            ):
                conn = make_connector(manager_block_size=manager_bs)
                group = _make_group(group_bs, kernel_bs)
                # Enough non-trivially permuted blocks for 4 manager blocks.
                needed = 4 * manager_bs // group_bs + 1
                block_table = [(i * 7 + 3) % 97 for i in range(needed)]
                mbis = [0, 1, 3]
                got = conn._attn_token_indices(group, mbis, block_table)
                want = _ref_attn_token_indices(
                    manager_bs, group_bs, kernel_bs, mbis, block_table
                )
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


class TestSlotInvariants(unittest.TestCase):
    """C10: the invariants every c == 1 slot mapping must satisfy.

    The gather/scatter kernel copies a whole manager block as one batch, so
    the mapping must give exactly one slot per token, never a duplicate, and
    within one physical page a single contiguous run (no holes, no page
    revisited) -- that is what makes the row-wise copy exact."""

    #: (manager_bs, group_bs, kernel_bs): the V3.2 MLA geometry, a manager
    #: block spanning two group blocks, the V4 indexer page shape, plain MLA,
    #: the hybrid geometry and small hand values.
    CASES = [
        (64, 64, 64),
        (128, 64, 64),
        (256, 256, 64),
        (16, 16, 16),
        (528, 528, 64),
        (4, 4, 2),
    ]

    def test_slot_invariants(self):
        for manager_bs, group_bs, kernel_bs in self.CASES:
            with self.subTest(
                manager_bs=manager_bs, group_bs=group_bs, kernel_bs=kernel_bs
            ):
                conn = make_connector(manager_block_size=manager_bs)
                group = _make_group(group_bs, kernel_bs)
                needed = 3 * manager_bs // group_bs + 1
                block_table = [(i * 7 + 1) % 61 for i in range(needed)]
                rows = conn._attn_token_indices(group, [0, 1, 2], block_table)
                ratio = group_bs // kernel_bs
                flat_limit = (max(block_table) + 1) * ratio * kernel_bs
                for row in rows:
                    self.assertEqual(len(row), manager_bs)
                    self.assertEqual(len(set(row)), len(row))
                    # Flat slot indexes stay inside the paged tensor.
                    self.assertTrue(all(slot < flat_limit for slot in row))
                    seen = set()
                    prev_page = None
                    run = 0
                    for slot in row:
                        page, offset = divmod(slot, kernel_bs)
                        if page != prev_page:
                            self.assertNotIn(page, seen)  # page not revisited
                            seen.add(page)
                            run = 0
                        # A page's slots are one contiguous run: the kernel
                        # copies rows, not holes.
                        self.assertEqual(offset, run)
                        run += 1
                        prev_page = page

    def test_hand_golden_permuted_table(self):
        # manager_bs=4, group_bs=4, kernel_bs=2 with a permuted table
        # (0->5, 1->2, ...): manager block 0 lands on pages 10, 11 (10,11,12,
        # 13) and block 1 on pages 4, 5 (8,9,10,11).
        conn = make_connector(manager_block_size=4)
        group = _make_group(group_bs=4, kernel_bs=2)
        table = [5, 2, 9, 0]
        self.assertEqual(
            conn._attn_token_indices(group, [0], table), [[20, 21, 22, 23]]
        )
        self.assertEqual(conn._attn_token_indices(group, [1], table), [[8, 9, 10, 11]])


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
