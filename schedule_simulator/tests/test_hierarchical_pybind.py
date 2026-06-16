"""
Tests for the extended pybind: HierarchicalReplayManager Python bindings.
"""
import os
import sys
import pytest

KVCM_SO_DIR = "/sgl-workspace/claude_workspace/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind"
if KVCM_SO_DIR not in sys.path:
    sys.path.insert(0, KVCM_SO_DIR)

try:
    import kvcm_py_optimizer as kvcm
    HAS_KVCM = True
except ImportError:
    HAS_KVCM = False

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "assets/hierarchical/test_config.json")

pytestmark = pytest.mark.skipif(not HAS_KVCM, reason="kvcm_py_optimizer not available")


@pytest.fixture
def manager():
    os.makedirs("/tmp/hierarchical_test_output/pool", exist_ok=True)
    os.makedirs("/tmp/hierarchical_test_output/infer", exist_ok=True)
    with open("/tmp/hierarchical_test_trace.jsonl", "w"):
        pass

    loader = kvcm.HierarchicalReplayConfigLoader()
    ok = loader.load(CONFIG_PATH)
    assert ok, f"Failed to load config from {CONFIG_PATH}"

    mgr = kvcm.HierarchicalReplayManager(loader.config())
    assert mgr.Init(), "HierarchicalReplayManager Init failed"
    return mgr


def test_bindings_exist():
    assert hasattr(kvcm, "HierarchicalReplayManager")
    assert hasattr(kvcm, "HierarchicalReplayConfig")
    assert hasattr(kvcm, "HierarchicalReplayConfigLoader")
    assert hasattr(kvcm, "HierarchicalGetCacheLocationRes")
    print("[bindings] All 4 hierarchical bindings present")


def test_config_loader():
    loader = kvcm.HierarchicalReplayConfigLoader()
    assert loader.load(CONFIG_PATH)
    config = loader.config()
    assert config.infer_scheduling_strategy() == "preserve_trace"
    print(f"[config] strategy={config.infer_scheduling_strategy()}")


def test_cold_read(manager):
    keys = [1, 2, 3, 4, 5]
    res = manager.GetCacheLocation("engine_0", "cold", 1000, keys, len(keys))
    assert res.engine_hit_length == 0
    assert res.peer_hit_length == 0
    assert res.storage_pool_hit_length == 0
    assert res.total_hit_length == 0
    print("[cold_read] All hits = 0")


def test_write_then_local_read(manager):
    keys = [10, 20, 30]
    manager.WriteCache("engine_0", "w1", 1000, keys)
    res = manager.GetCacheLocation("engine_0", "r1", 2000, keys, len(keys))
    assert res.engine_hit_length > 0, f"Expected local hits, got {res.engine_hit_length}"
    assert res.total_hit_length == len(keys)
    print(f"[local_read] engine={res.engine_hit_length}, total={res.total_hit_length}")


def test_cross_engine_read(manager):
    keys = [100, 200, 300]
    manager.WriteCache("engine_0", "w_cross", 1000, keys)
    res = manager.GetCacheLocation("engine_1", "r_cross", 2000, keys, len(keys))
    assert res.engine_hit_length == 0, "engine_1 should not have local hits"
    non_local = res.peer_hit_length + res.storage_pool_hit_length
    assert non_local > 0, f"Should get P2P or pool hits, got peer={res.peer_hit_length}, pool={res.storage_pool_hit_length}"
    print(f"[cross_engine] peer={res.peer_hit_length}, pool={res.storage_pool_hit_length}, total={res.total_hit_length}")


def test_write_cache_with_ttl(manager):
    keys = [1000, 2000, 3000]
    res = manager.WriteCacheWithTtlUs("engine_0", "ttl_w", 1000, keys, 5000000)
    assert res.trace_id == "ttl_w"
    print(f"[ttl_write] wrote {res.kvcm_write_length} blocks")


def test_multiple_writes_accumulate(manager):
    manager.WriteCache("engine_0", "w_a", 1000, [1, 2, 3])
    manager.WriteCache("engine_0", "w_b", 2000, [4, 5, 6])
    res = manager.GetCacheLocation("engine_0", "r_all", 3000, [1, 2, 3, 4, 5, 6], 6 * 256)
    assert res.total_hit_length == 6, f"Expected 6 total hits, got {res.total_hit_length}"
    print(f"[accumulate] total={res.total_hit_length}")


if __name__ == "__main__":
    test_bindings_exist()
    test_config_loader()
