"""Tests for block_id_converter: conversion correctness and prefix preservation."""
import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schedule_simulator.schedule_emulator.block_id_converter import (
    convert_block_ids,
    _hash_block_group,
)


# ---------------------------------------------------------------------------
# 1. Basic conversion correctness
# ---------------------------------------------------------------------------

class TestConvertBlockIds:
    def test_same_block_size_returns_copy(self):
        """When source == target, return a copy of the original."""
        original = [10, 20, 30, 40]
        result = convert_block_ids(original, 256, 256)
        assert result == original
        assert result is not original  # must be a copy

    def test_merge_ratio_2(self):
        """2:1 merge (256 -> 512): 4 blocks -> 2 blocks."""
        blocks = [1, 2, 3, 4]
        result = convert_block_ids(blocks, 256, 512)
        assert len(result) == 2

    def test_merge_ratio_8(self):
        """8:1 merge (256 -> 2048): 16 blocks -> 2 blocks."""
        blocks = list(range(16))
        result = convert_block_ids(blocks, 256, 2048)
        assert len(result) == 2

    def test_merge_ratio_4(self):
        """4:1 merge (256 -> 1024): 12 blocks -> 3 blocks."""
        blocks = list(range(12))
        result = convert_block_ids(blocks, 256, 1024)
        assert len(result) == 3

    def test_trailing_incomplete_group_dropped(self):
        """Incomplete trailing group is dropped."""
        # 10 blocks with ratio 8 -> 1 complete group, 2 leftover dropped
        blocks = list(range(10))
        result = convert_block_ids(blocks, 256, 2048)
        assert len(result) == 1

    def test_fewer_than_ratio_returns_empty(self):
        """If fewer blocks than ratio, result is empty."""
        blocks = [1, 2, 3]
        result = convert_block_ids(blocks, 256, 2048)  # ratio=8
        assert result == []

    def test_empty_input(self):
        """Empty block_ids returns empty result."""
        result = convert_block_ids([], 256, 2048)
        assert result == []

    def test_exact_multiple(self):
        """Exactly ratio blocks -> exactly 1 output block."""
        blocks = list(range(8))
        result = convert_block_ids(blocks, 256, 2048)
        assert len(result) == 1

    def test_large_dataset_block_count(self):
        """Simulate real data: 350 blocks at 256 -> 43 blocks at 2048."""
        blocks = list(range(350))
        result = convert_block_ids(blocks, 256, 2048)
        assert len(result) == 350 // 8  # 43

    def test_output_is_int(self):
        """Each merged block_id should be an int."""
        blocks = list(range(8))
        result = convert_block_ids(blocks, 256, 2048)
        assert all(isinstance(x, int) for x in result)


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        """Same block_ids always produce the same merged result."""
        blocks = [100, 200, 300, 400, 500, 600, 700, 800]
        r1 = convert_block_ids(blocks, 256, 2048)
        r2 = convert_block_ids(blocks, 256, 2048)
        assert r1 == r2

    def test_different_input_different_output(self):
        """Different block_ids produce different merged results."""
        a = [1, 2, 3, 4, 5, 6, 7, 8]
        b = [1, 2, 3, 4, 5, 6, 7, 9]  # last block differs
        ra = convert_block_ids(a, 256, 2048)
        rb = convert_block_ids(b, 256, 2048)
        assert ra != rb

    def test_hash_group_deterministic(self):
        """_hash_block_group is deterministic."""
        group = [111, 222, 333, 444]
        h1 = _hash_block_group(group)
        h2 = _hash_block_group(group)
        assert h1 == h2


# ---------------------------------------------------------------------------
# 3. Prefix preservation (the KEY requirement)
# ---------------------------------------------------------------------------

class TestPrefixPreservation:
    def test_shared_prefix_preserved_after_merge(self):
        """Two requests sharing a prefix in source blocks share it after merge."""
        # Request A: 24 blocks (3 merged blocks at ratio 8)
        # Request B: same first 16 blocks + different last 8
        shared = list(range(100, 116))  # 16 shared source blocks
        req_a = shared + list(range(200, 208))
        req_b = shared + list(range(300, 308))

        merged_a = convert_block_ids(req_a, 256, 2048)
        merged_b = convert_block_ids(req_b, 256, 2048)

        assert len(merged_a) == 3
        assert len(merged_b) == 3
        # First 2 merged blocks should be identical (shared 16 source blocks)
        assert merged_a[:2] == merged_b[:2]
        # Third merged block should differ
        assert merged_a[2] != merged_b[2]

    def test_no_shared_prefix_stays_different(self):
        """Requests with no shared prefix remain different after merge."""
        req_a = list(range(0, 8))
        req_b = list(range(100, 108))
        merged_a = convert_block_ids(req_a, 256, 2048)
        merged_b = convert_block_ids(req_b, 256, 2048)
        assert merged_a != merged_b

    def test_full_prefix_match(self):
        """Request B is a prefix of A: after merge, B is still a prefix of A."""
        req_a = list(range(0, 24))  # 3 merged blocks
        req_b = list(range(0, 16))  # 2 merged blocks (prefix of A)
        merged_a = convert_block_ids(req_a, 256, 2048)
        merged_b = convert_block_ids(req_b, 256, 2048)
        assert len(merged_a) == 3
        assert len(merged_b) == 2
        assert merged_a[:2] == merged_b[:2]

    def test_prefix_preserved_across_many_requests(self):
        """Multiple requests sharing various prefix lengths."""
        base = list(range(1000, 1032))  # 32 source blocks = 4 merged

        # Create requests with 0, 1, 2, 3, 4 shared merged blocks
        requests = []
        for shared_merged in range(5):
            shared_source = shared_merged * 8
            unique = list(range(2000 + shared_merged * 100,
                                2000 + shared_merged * 100 + 32 - shared_source))
            req = base[:shared_source] + unique
            requests.append(req)

        merged = [convert_block_ids(r, 256, 2048) for r in requests]
        base_merged = convert_block_ids(base, 256, 2048)

        for i, m in enumerate(merged):
            shared_count = i
            # First shared_count blocks should match base
            assert m[:shared_count] == base_merged[:shared_count], (
                f"Request {i}: expected {shared_count} shared blocks"
            )

    def test_partial_group_does_not_create_false_match(self):
        """Partial prefix within a merged group does NOT match."""
        # A and B share first 4 source blocks but not a full group of 8
        req_a = list(range(0, 4)) + list(range(100, 104))  # 8 blocks
        req_b = list(range(0, 4)) + list(range(200, 204))  # 8 blocks
        merged_a = convert_block_ids(req_a, 256, 2048)
        merged_b = convert_block_ids(req_b, 256, 2048)
        assert len(merged_a) == 1
        assert len(merged_b) == 1
        # They differ because the second half of the group differs
        assert merged_a[0] != merged_b[0]


# ---------------------------------------------------------------------------
# 4. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_target_smaller_than_source_raises(self):
        with pytest.raises(ValueError, match="must be >="):
            convert_block_ids([1, 2], 2048, 256)

    def test_non_multiple_raises(self):
        with pytest.raises(ValueError, match="must be a multiple"):
            convert_block_ids([1, 2], 256, 300)

    def test_zero_source_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            convert_block_ids([1], 0, 256)

    def test_negative_target_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            convert_block_ids([1], 256, -1)


# ---------------------------------------------------------------------------
# 5. Integration: verify with real-like data dimensions
# ---------------------------------------------------------------------------

class TestRealDataDimensions:
    def test_h21_data_conversion(self):
        """Simulate h21_32_256k_full.jsonl: 350 blocks at 256 -> 2048."""
        # input_length=89502, 350 blocks at 256 tokens/block
        import random
        random.seed(42)
        block_ids = [random.randint(-(1 << 62), 1 << 62) for _ in range(350)]
        result = convert_block_ids(block_ids, 256, 2048)
        assert len(result) == 43  # 350 // 8

    def test_conversion_preserves_input(self):
        """Original block_ids list is not modified."""
        original = list(range(16))
        original_copy = list(original)
        convert_block_ids(original, 256, 2048)
        assert original == original_copy
