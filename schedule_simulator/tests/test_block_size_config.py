"""
Test that page_size/block_size is correctly configured and passed through the system.
"""
import os
import sys
import tempfile
import json
import pytest

sys.path.insert(0, '/sgl-workspace/claude_workspace/schedule_simulator/src')

from schedule_simulator.schedule_emulator.types import SchedulerConfig, PlatformConfig
from schedule_simulator.schedule_emulator.hierarchical_config_builder import build_hierarchical_config


def test_default_page_size():
    """Test that default page_size (None) results in block_size=1 in config"""
    sc = SchedulerConfig(
        model="Qwen2.5-3B",
        hicache_storage_backend="hf3fs",
    )
    pc = PlatformConfig(device="H20")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = build_hierarchical_config(sc, pc, ["P0"], output_dir=tmpdir, enable_p2p=False)
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["infer_clusters"][0]["model"]["block_size"] == 1
        assert config["storage_pool"]["pools"][0]["model"]["block_size"] == 1


def test_custom_page_size_16():
    """Test that page_size=16 is correctly set in config"""
    sc = SchedulerConfig(
        model="Qwen2.5-3B",
        hicache_storage_backend="hf3fs",
        page_size=16,
    )
    pc = PlatformConfig(device="H20")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = build_hierarchical_config(sc, pc, ["P0"], output_dir=tmpdir, enable_p2p=False)
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["infer_clusters"][0]["model"]["block_size"] == 16
        assert config["storage_pool"]["pools"][0]["model"]["block_size"] == 16


def test_custom_page_size_256():
    """Test that page_size=256 is correctly set in config"""
    sc = SchedulerConfig(
        model="Qwen2.5-3B",
        hicache_storage_backend="hf3fs",
        page_size=256,
    )
    pc = PlatformConfig(device="H20")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = build_hierarchical_config(sc, pc, ["P0"], output_dir=tmpdir, enable_p2p=False)
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["infer_clusters"][0]["model"]["block_size"] == 256
        assert config["storage_pool"]["pools"][0]["model"]["block_size"] == 256


def test_page_size_with_p2p():
    """Test that page_size works correctly with P2P enabled"""
    sc = SchedulerConfig(
        model="Qwen2.5-3B",
        hicache_storage_backend="hf3fs",
        page_size=32,
    )
    pc = PlatformConfig(device="H20")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = build_hierarchical_config(sc, pc, ["P0", "P1", "P2"], output_dir=tmpdir, enable_p2p=True)
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["infer_clusters"][0]["model"]["block_size"] == 32
        assert config["storage_pool"]["pools"][0]["model"]["block_size"] == 32
        assert len(config["infer_clusters"][0]["infer_ids"]) == 3


def test_adapter_uses_page_size():
    """Test that HierarchicalCacheAdapter uses self.page_size for validation"""
    from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
    from schedule_simulator.schedule_emulator.base import GlobalValues
    
    class MockManager:
        def GetCacheLocation(self, *args, **kwargs):
            class Result:
                engine_hit_length = 0
                peer_hit_length = 0
                storage_pool_hit_length = 0
            return Result()
    
    adapter = HierarchicalCacheAdapter(
        manager=MockManager(),
        engine_instance_id="test_engine",
        platform_config=PlatformConfig(device="H20"),
        kv_cache_space_per_token=46080,
        page_size=16,
        global_values=GlobalValues(),
    )
    assert adapter.page_size == 16
    
    adapter2 = HierarchicalCacheAdapter(
        manager=MockManager(),
        engine_instance_id="test_engine",
        platform_config=PlatformConfig(device="H20"),
        kv_cache_space_per_token=46080,
        page_size=256,
        global_values=GlobalValues(),
    )
    assert adapter2.page_size == 256


if __name__ == "__main__":
    test_default_page_size()
    test_custom_page_size_16()
    test_custom_page_size_256()
    test_page_size_with_p2p()
    test_adapter_uses_page_size()
    print("All block_size config tests passed!")
