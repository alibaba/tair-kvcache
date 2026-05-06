
"""
Python版本的Optimizer测试
参考C++版的一部分测试 optimizer_manager_test.cc
使用unittest框架
"""

import os
import tempfile
import unittest
import copy
import json
from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer
# instance id与 配置文件 optimizer_startup_config_load.json 中一致
class OptimizerManagerPyTest(unittest.TestCase):
    def setUp(self):
        # 创建临时目录用于测试文件存储
        self.temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {self.temp_dir}")
        # 直接初始化 OptimizerManager
        test_srcdir = os.getenv("TEST_SRCDIR")
        if test_srcdir:
            config_file = os.path.join(
                test_srcdir, 
                "kv_cache_manager/kv_cache_manager/optimizer/test/testdata/optimizer_startup_config_load.json"
            )

        self.config_file = config_file

        self.config_loader = kvcm_py_optimizer.OptimizerConfigLoader()
        self.assertTrue(self.config_loader.load(config_file))
        self.config = self.config_loader.config()
        self.manager = kvcm_py_optimizer.OptimizerManager(self.config)
        
        self.assertIsNotNone(self.manager)
        self.manager.Init()
    def tearDown(self):
        # 清理临时目录
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_write_cache(self):
        instance_id = "3780643326877293460"
        trace_id = "test_trace_001"
        timestamp = 1234567890
        block_ids = [1, 2]
        token_ids = [10, 20, 30, 40, 50]
        res = self.manager.WriteCache(instance_id, trace_id, timestamp, block_ids, token_ids)

        self.assertEqual(res.trace_id, trace_id)
        self.assertEqual(res.kvcm_write_length, 2)
        self.assertEqual(res.kvcm_write_hit_length, 0)

    def test_get_cache_location(self):
        instance_id = "3780643326877293460"
        trace_id = "test_trace_002"
        write_timestamp = 1234567890
        write_block_ids = [1 ,2 ,3]
        write_token_ids = [60, 70, 80, 90, 100]
        self.manager.WriteCache(instance_id, trace_id, write_timestamp, write_block_ids, write_token_ids)

        read_timestamp = 1234567900
        read_block_ids = [1, 2, 4]
        read_token_ids = [60, 70, 110]
        mask_offset = 0
        res = self.manager.GetCacheLocation(instance_id, trace_id, read_timestamp, read_block_ids, read_token_ids, mask_offset)

        self.assertEqual(res.trace_id, trace_id)
        self.assertEqual(res.kvcm_hit_length, 2)

    def test_ttl_expiration(self):
        """测试 TTL: 写入显式传 ttl_seconds，过期前命中、过期后失效"""
        # 仅 TTL 用例要求测试配置为 ttl 策略，其他用例不受限
        with open(self.config_file, "r", encoding="utf-8") as fp:
            config_json = json.load(fp)
        policy_type = config_json["instance_groups"][0]["instances"][0]["eviction_policy_type"]
        if policy_type != "ttl":
            self.skipTest(f"test_ttl_expiration requires ttl policy, current policy is: {policy_type}")

        instance_id = "3780643326877293460"
        trace_id = "test_ttl_trace"
        write_timestamp = 1_000_000
        block_ids = [101, 102]
        ttl_seconds = 1

        self.manager.WriteCache(instance_id, trace_id, write_timestamp, block_ids, [], ttl_seconds)

        # 过期前查询（1 秒内）
        before_expire = self.manager.GetCacheLocation(
            instance_id, trace_id, write_timestamp + 500_000, block_ids, [], 0
        )
        self.assertEqual(before_expire.kvcm_hit_length, 2, "Should hit before TTL expiration")

        # TTL 是滑动窗口：上一次访问会刷新 last_access_time，因此要基于 before_expire 的时间继续推进
        after_expire_timestamp = (write_timestamp + 500_000) + ttl_seconds * 1_000_000 + 1
        after_expire = self.manager.GetCacheLocation(
            instance_id, trace_id, after_expire_timestamp, block_ids, [], 0
        )
        self.assertEqual(after_expire.kvcm_hit_length, 0, "Should miss after TTL expiration")

    def test_clear_cache(self):
        """测试清空缓存功能（保留统计）"""
        instance_id = "3780643326877293460"
        # 写入一些数据
        for i in range(5):
            self.manager.WriteCache(instance_id, f"trace_{i}", i * 1000, [i, i+1, i+2], [])
        
        # 查询应该有命中
        res = self.manager.GetCacheLocation(instance_id, "query_1", 10000, [0, 1, 2], [], 0)
        self.assertGreater(res.kvcm_hit_length, 0, "Should have cache hits before clear")
        
        # 清空缓存
        success = self.manager.ClearCache(instance_id)
        self.assertTrue(success, "Clear cache should succeed")
        
        # 查询应该没有命中
        res = self.manager.GetCacheLocation(instance_id, "query_2", 11000, [0, 1, 2], [], 0)
        self.assertEqual(res.kvcm_hit_length, 0, "Should have no cache hits after clear")

    def test_clear_all_caches(self):
        """测试清空所有实例缓存"""
        instance_id = "3780643326877293460"
        # 写入数据
        for i in range(3):
            self.manager.WriteCache(instance_id, f"trace_{i}", i * 1000, [i, i+1], [])
        
        # 清空所有缓存
        self.manager.ClearAllCaches()
        
        # 验证缓存已清空
        res = self.manager.GetCacheLocation(instance_id, "query_3", 5000, [0, 1, 2], [], 0)
        self.assertEqual(res.kvcm_hit_length, 0, "All caches should be cleared")

    def test_clear_cache_and_reset_stats(self):
        """测试清空缓存并重置统计"""
        instance_id = "3780643326877293460"
        # 写入数据
        for i in range(5):
            self.manager.WriteCache(instance_id, f"trace_{i}", i * 1000, [i, i+1], [])
        
        # 清空缓存并重置统计
        success = self.manager.ClearCacheAndResetStats(instance_id)
        self.assertTrue(success, "Clear cache and reset stats should succeed")
        
        # 验证缓存已清空
        res = self.manager.GetCacheLocation(instance_id, "query_4", 10000, [0, 1], [], 0)
        self.assertEqual(res.kvcm_hit_length, 0, "Cache should be cleared")

    def test_clear_all_caches_and_reset_stats(self):
        """测试清空所有缓存并重置统计"""
        instance_id = "3780643326877293460"
        # 写入数据
        for i in range(3):
            self.manager.WriteCache(instance_id, f"trace_{i}", i * 1000, [i*2, i*2+1], [])
        
        # 清空所有缓存并重置统计
        self.manager.ClearAllCachesAndResetStats()
        
        # 验证缓存已清空
        res = self.manager.GetCacheLocation(instance_id, "query_5", 5000, [0, 2, 4], [], 0)
        self.assertEqual(res.kvcm_hit_length, 0, "All caches should be cleared")

if __name__ == '__main__':
    unittest.main()