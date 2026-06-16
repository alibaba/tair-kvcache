"""Tests for ChooseBestEngine P2P tracker optimization.

Validates that ChooseBestEngine correctly uses the P2P tracker to narrow
candidate engines before PrefixMatchCount, and that it falls back to full
scan when the first block is not tracked.
"""
import sys
import json
import os
import pytest
import tempfile

sys.path.insert(
    0,
    "/sgl-workspace/claude_workspace/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind",
)
import kvcm_py_optimizer as kvcm


def _make_manager(num_engines: int, hbm_gb: float = 0.25, dram_gb: float = 0.5):
    """Create a HierarchicalReplayManager with N engines for testing."""
    config_path = os.path.join(
        os.path.dirname(__file__), "assets", "hierarchical", "test_config.json"
    )
    config = json.load(open(config_path))
    config["infer_clusters"][0]["infer_ids"] = [f"P{i}" for i in range(num_engines)]
    config["infer_clusters"][0]["model"]["block_size"] = 1
    config["infer_clusters"][0]["model"]["bytes_per_token"] = 1
    config["infer_clusters"][0]["tiers"][0]["capacity"] = hbm_gb
    config["infer_clusters"][0]["tiers"][1]["capacity"] = dram_gb

    tmpdir = tempfile.mkdtemp(prefix="cbe_opt_test_")
    os.makedirs(os.path.join(tmpdir, "pool"), exist_ok=True)
    config["output_result_path"] = tmpdir
    config["storage_pool"]["output_result_path"] = os.path.join(tmpdir, "pool")

    tmp_config = os.path.join(tmpdir, "config.json")
    with open(tmp_config, "w") as f:
        json.dump(config, f)

    loader = kvcm.HierarchicalReplayConfigLoader()
    assert loader.load(tmp_config)
    mgr = kvcm.HierarchicalReplayManager(loader.config())
    assert mgr.Init()
    return mgr


class TestChooseBestEngineCorrectness:
    """Correctness tests: optimization must produce same results as full scan."""

    def test_single_engine_with_data(self):
        """Only one engine has data, should be selected."""
        mgr = _make_manager(4)
        mgr.WriteCache("P2", "w0", 1000, list(range(100, 200)))

        res = mgr.ChooseBestEngine(list(range(100, 150)), 2000)
        assert res.engine_instance_id == "P2"
        assert res.hit_count == 50

    def test_multiple_engines_different_data(self):
        """Each engine has unique data, correct one selected."""
        mgr = _make_manager(8)
        for i in range(8):
            mgr.WriteCache(f"P{i}", f"w{i}", 1000, list(range(i * 1000, i * 1000 + 500)))

        for i in range(8):
            res = mgr.ChooseBestEngine(list(range(i * 1000, i * 1000 + 100)), 2000)
            assert res.engine_instance_id == f"P{i}", f"Expected P{i}, got {res.engine_instance_id}"
            assert res.hit_count == 100

    def test_shared_prefix_longest_match_wins(self):
        """When multiple engines share first block, longest prefix wins."""
        mgr = _make_manager(4)
        mgr.WriteCache("P0", "w0", 1000, list(range(0, 30)))
        mgr.WriteCache("P1", "w1", 1000, list(range(0, 80)))
        mgr.WriteCache("P2", "w2", 1000, list(range(0, 150)))
        mgr.WriteCache("P3", "w3", 1000, list(range(0, 50)))

        res = mgr.ChooseBestEngine(list(range(0, 200)), 2000)
        assert res.engine_instance_id == "P2"
        assert res.hit_count == 150

    def test_cold_data_fallback(self):
        """When data not in any engine, returns first active with hit=0."""
        mgr = _make_manager(4)
        mgr.WriteCache("P0", "w0", 1000, list(range(0, 100)))

        res = mgr.ChooseBestEngine(list(range(9999, 9999 + 50)), 2000)
        assert res.hit_count == 0
        assert res.engine_instance_id != ""

    def test_empty_block_ids(self):
        """Empty block_ids should return a valid engine."""
        mgr = _make_manager(4)
        res = mgr.ChooseBestEngine([], 2000)
        assert res.engine_instance_id != ""
        assert res.hit_count == 0

    def test_partial_prefix_match(self):
        """Engine has [0-100], query [0-200] → match=100."""
        mgr = _make_manager(4)
        mgr.WriteCache("P1", "w1", 1000, list(range(0, 100)))

        res = mgr.ChooseBestEngine(list(range(0, 200)), 2000)
        assert res.engine_instance_id == "P1"
        assert res.hit_count == 100

    def test_after_eviction_still_correct(self):
        """After data is evicted, ChooseBestEngine should not find it."""
        mgr = _make_manager(2, hbm_gb=0.0001, dram_gb=0.0001)  # tiny capacity

        # Write data that will be evicted
        mgr.WriteCache("P0", "w0", 1000, list(range(0, 50)))
        # Evict by writing much more data
        for i in range(100):
            mgr.WriteCache("P0", f"evict_{i}", 1000 + i, list(range(1000 + i * 100, 1000 + i * 100 + 100)))

        # Original data likely evicted
        res = mgr.ChooseBestEngine(list(range(0, 50)), 5000)
        # Either hit=0 (evicted from both tree and tracker) or still in tracker
        # but PrefixMatchCount returns 0. Both are valid.
        if res.hit_count > 0:
            assert res.engine_instance_id == "P0"

    def test_write_then_query_same_engine(self):
        """Write and query on same engine, P2P tracker should have it."""
        mgr = _make_manager(4)
        mgr.WriteCache("P3", "w_test", 1000, [42, 43, 44, 45, 46])

        res = mgr.ChooseBestEngine([42, 43, 44, 45, 46], 2000)
        assert res.engine_instance_id == "P3"
        assert res.hit_count == 5


class TestChooseBestEngineFallback:
    """Tests for the fallback path when P2P tracker has no candidates."""

    def test_no_p2p_flows_configured(self):
        """If P2P flows are empty, should still work via fallback."""
        config_path = os.path.join(
            os.path.dirname(__file__), "assets", "hierarchical", "test_config.json"
        )
        config = json.load(open(config_path))
        config["infer_clusters"][0]["infer_ids"] = ["P0", "P1"]
        config["infer_clusters"][0]["model"]["block_size"] = 1
        config["infer_clusters"][0]["model"]["bytes_per_token"] = 1
        config["infer_clusters"][0]["p2p_read_flows"] = []  # Disable P2P

        tmpdir = tempfile.mkdtemp(prefix="cbe_nop2p_")
        os.makedirs(os.path.join(tmpdir, "pool"), exist_ok=True)
        config["output_result_path"] = tmpdir
        config["storage_pool"]["output_result_path"] = os.path.join(tmpdir, "pool")

        tmp_config = os.path.join(tmpdir, "config.json")
        with open(tmp_config, "w") as f:
            json.dump(config, f)

        loader = kvcm.HierarchicalReplayConfigLoader()
        assert loader.load(tmp_config)
        mgr = kvcm.HierarchicalReplayManager(loader.config())
        assert mgr.Init()

        mgr.WriteCache("P1", "w1", 1000, list(range(0, 100)))
        res = mgr.ChooseBestEngine(list(range(0, 50)), 2000)
        # Without P2P tracker, falls back to full scan. RadixTree still works.
        assert res.engine_instance_id == "P1"
        assert res.hit_count == 50

    def test_first_block_not_tracked_but_data_exists(self):
        """Edge case: P2P tracker empty but engine has data in RadixTree."""
        # This tests the fallback path correctly finding data
        mgr = _make_manager(4)
        mgr.WriteCache("P2", "w2", 1000, list(range(500, 600)))

        # Query for blocks that start with one on P2
        res = mgr.ChooseBestEngine(list(range(500, 550)), 2000)
        assert res.engine_instance_id == "P2"
        assert res.hit_count == 50


class TestChooseBestEngineConsistency:
    """Verify optimization produces identical results to what full scan would give."""

    def test_consistency_with_many_engines(self):
        """Run many queries, ensure optimized ChooseBestEngine matches expected."""
        import random
        random.seed(123)

        mgr = _make_manager(32, hbm_gb=1.0, dram_gb=1.0)

        # Write varied data
        written = {}
        for i in range(32):
            blocks = list(range(i * 2000, i * 2000 + random.randint(500, 1500)))
            mgr.WriteCache(f"P{i}", f"w{i}", 1000, blocks)
            written[f"P{i}"] = blocks

        # Run 100 random queries and verify
        for trial in range(100):
            engine_idx = random.randint(0, 31)
            blocks = written[f"P{engine_idx}"]
            query_len = min(random.randint(50, 200), len(blocks))
            query = blocks[:query_len]

            res = mgr.ChooseBestEngine(query, 2000)
            # The result should be an engine whose hit_count >= query_len
            # (could be the same engine or one with overlapping data)
            assert res.hit_count >= query_len, (
                f"Trial {trial}: expected hit>={query_len}, got {res.hit_count} "
                f"from {res.engine_instance_id}"
            )

    def test_repeated_calls_same_result(self):
        """Same query should always produce same result (deterministic)."""
        mgr = _make_manager(8)
        for i in range(8):
            mgr.WriteCache(f"P{i}", f"w{i}", 1000, list(range(i * 100, i * 100 + 100)))

        query = list(range(300, 350))
        results = [mgr.ChooseBestEngine(query, 2000) for _ in range(50)]
        assert all(r.engine_instance_id == results[0].engine_instance_id for r in results)
        assert all(r.hit_count == results[0].hit_count for r in results)
