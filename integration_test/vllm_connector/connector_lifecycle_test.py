"""
vLLM Connector 生命周期集成测试

该模块测试 TairKvCacheConnector 的完整生命周期，包括：
- Scheduler 角色初始化
- Worker 角色初始化
- KV Cache 注册
- 缓存匹配查询
- 元数据构建
- 请求生命周期

测试运行方式：
    bazel test //integration_test/vllm_connector:connector_lifecycle_test
    bazel test //integration_test/vllm_connector:connector_lifecycle_test --test_output=all
"""

import unittest
from vllm_connector_cases import VllmConnectorTestBase
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole


class ConnectorSchedulerInitTest(VllmConnectorTestBase):
    """Scheduler 角色初始化测试"""
    
    def _init_connector(self):
        """初始化 Scheduler 角色的 Connector"""
        from kv_cache_manager.py_connector.vllm.v1_connector import TairKvCacheConnector
        
        config = self._create_test_vllm_config()
        self.scheduler_connector = TairKvCacheConnector(
            config, 
            KVConnectorRole.SCHEDULER
        )
    
    def _cleanup_connector(self):
        """清理 Connector"""
        if hasattr(self, 'scheduler_connector') and self.scheduler_connector:
            self.scheduler_connector.shutdown()
    
    def test_scheduler_init(self):
        """测试 Scheduler 角色初始化"""
        self.assertIsNotNone(self.scheduler_connector)
        self.assertEqual(self.scheduler_connector.role, KVConnectorRole.SCHEDULER)
    
    def test_scheduler_has_manager_client(self):
        """测试 Scheduler 有 manager client"""
        self.assertIsNotNone(self.scheduler_connector._manager_client)
    
    def test_scheduler_has_location_query_manager(self):
        """测试 Scheduler 有 location query manager"""
        self.assertIsNotNone(self.scheduler_connector._location_query_manager)
    
    def test_get_num_matched_tokens_no_cache(self):
        """测试查询无缓存匹配的情况"""
        # 创建一个 mock request
        request = self._create_mock_request(
            request_id="test_req_1",
            prompt_token_ids=list(range(1, 33)),  # 32 tokens
        )
        
        # 查询匹配的 tokens
        matched, is_async = self.scheduler_connector.get_num_new_matched_tokens(
            request, 
            num_computed_tokens=0
        )
        
        # 无缓存时应该返回 0 或 None（异步查询中）
        if matched is not None:
            self.assertEqual(matched, 0)
            self.assertFalse(is_async)
    
    def test_build_connector_meta_empty(self):
        """测试构建空的元数据"""
        # 创建空的 scheduler output
        scheduler_output = self._create_mock_scheduler_output()
        
        # 构建元数据
        meta = self.scheduler_connector.build_connector_meta(scheduler_output)
        
        # 验证元数据不为空
        self.assertIsNotNone(meta)
        # 验证元数据类型
        from kv_cache_manager.py_connector.vllm.metadata import TairKvCacheConnectorMetadata
        self.assertIsInstance(meta, TairKvCacheConnectorMetadata)


class ConnectorWorkerInitTest(VllmConnectorTestBase):
    """Worker 角色初始化测试"""
    
    def _init_connector(self):
        """初始化 Worker 角色的 Connector"""
        from kv_cache_manager.py_connector.vllm.v1_connector import TairKvCacheConnector
        
        # Worker 需要先有 Scheduler 注册 instance
        # 这里我们先创建 Scheduler 注册，然后创建 Worker
        config = self._create_test_vllm_config()
        
        # 先创建 Scheduler 来注册 instance
        self.scheduler_connector = TairKvCacheConnector(
            config, 
            KVConnectorRole.SCHEDULER
        )
        
        # 然后创建 Worker
        self.worker_connector = TairKvCacheConnector(
            config, 
            KVConnectorRole.WORKER
        )
    
    def _cleanup_connector(self):
        """清理 Connectors"""
        if hasattr(self, 'worker_connector') and self.worker_connector:
            self.worker_connector.shutdown()
        if hasattr(self, 'scheduler_connector') and self.scheduler_connector:
            self.scheduler_connector.shutdown()
    
    def test_worker_init(self):
        """测试 Worker 角色初始化"""
        self.assertIsNotNone(self.worker_connector)
        self.assertEqual(self.worker_connector.role, KVConnectorRole.WORKER)
    
    def test_worker_has_transfer_client(self):
        """测试 Worker 有 transfer client"""
        self.assertIsNotNone(self.worker_connector._transfer_client)
    
    def test_worker_has_coordinator_client(self):
        """测试 Worker 有 coordinator client"""
        self.assertIsNotNone(self.worker_connector._coordinator_client)
    
    def test_register_kv_caches(self):
        """测试 Worker 注册 KV Cache"""
        # 创建 mock KV caches
        kv_caches = self._create_mock_kv_caches(
            num_layers=4,
            num_blocks=100,
            block_size=16,
            num_kv_heads=8,
            head_size=64,
        )
        
        # 注册 KV caches
        self.worker_connector.register_kv_caches(kv_caches)
        
        # 验证注册成功
        self.assertIsNotNone(self.worker_connector._kv_caches)
        self.assertEqual(len(self.worker_connector._kv_caches), 4)
        
        # 验证 kvcache_info 被正确设置
        self.assertIsNotNone(self.worker_connector._kvcache_info)
    
    def test_bind_connector_metadata(self):
        """测试绑定元数据"""
        from kv_cache_manager.py_connector.vllm.metadata import TairKvCacheConnectorMetadata
        
        # 创建测试元数据
        meta = TairKvCacheConnectorMetadata(epoch=1)
        
        # 绑定元数据
        self.worker_connector.bind_connector_metadata(meta)
        
        # 验证元数据已绑定
        self.assertTrue(self.worker_connector.has_connector_metadata())


class ConnectorLifecycleTest(VllmConnectorTestBase):
    """完整生命周期测试"""
    
    def _init_connector(self):
        """初始化 Scheduler 和 Worker Connectors"""
        from kv_cache_manager.py_connector.vllm.v1_connector import TairKvCacheConnector
        
        config = self._create_test_vllm_config()
        
        # 创建 Scheduler
        self.scheduler_connector = TairKvCacheConnector(
            config, 
            KVConnectorRole.SCHEDULER
        )
        
        # 创建 Worker
        self.worker_connector = TairKvCacheConnector(
            config, 
            KVConnectorRole.WORKER
        )
        
        # 注册 KV caches
        kv_caches = self._create_mock_kv_caches()
        self.worker_connector.register_kv_caches(kv_caches)
    
    def _cleanup_connector(self):
        """清理 Connectors"""
        if hasattr(self, 'worker_connector') and self.worker_connector:
            self.worker_connector.shutdown()
        if hasattr(self, 'scheduler_connector') and self.scheduler_connector:
            self.scheduler_connector.shutdown()
    
    def test_scheduler_worker_communication(self):
        """测试 Scheduler 和 Worker 之间的通信"""
        # 1. Scheduler 构建元数据
        scheduler_output = self._create_mock_scheduler_output()
        meta = self.scheduler_connector.build_connector_meta(scheduler_output)
        
        # 2. Worker 绑定元数据
        self.worker_connector.bind_connector_metadata(meta)
        
        # 3. 验证元数据传递成功
        self.assertTrue(self.worker_connector.has_connector_metadata())
    
    def test_request_new_match_flow(self):
        """测试新请求的匹配流程"""
        # 创建请求
        request = self._create_mock_request(
            request_id="test_req_flow",
            prompt_token_ids=list(range(1, 65)),  # 64 tokens
        )
        
        # 1. 查询匹配
        matched, is_async = self.scheduler_connector.get_num_new_matched_tokens(
            request, 
            num_computed_tokens=0
        )
        
        # 2. 模拟分配 blocks
        if matched is not None and matched == 0:
            # 无远程缓存，分配本地 blocks
            block_ids = list(range(4))  # 假设分配 4 个 blocks
            blocks = self._create_mock_kv_cache_blocks(block_ids)
            
            self.scheduler_connector.update_state_after_alloc(
                request, 
                blocks, 
                num_external_tokens=0
            )
        
        # 3. 构建元数据并传递给 Worker
        # 创建 mock scheduled_new_req
        from unittest.mock import MagicMock
        scheduled_new_req = MagicMock()
        scheduled_new_req.req_id = request.request_id
        scheduled_new_req.block_ids = [[0, 1, 2, 3]]
        
        scheduler_output = self._create_mock_scheduler_output(
            new_reqs=[scheduled_new_req],
            num_scheduled_tokens={request.request_id: 64}
        )
        
        meta = self.scheduler_connector.build_connector_meta(scheduler_output)
        
        # 4. Worker 绑定元数据
        self.worker_connector.bind_connector_metadata(meta)
        
        # 5. 验证请求状态被跟踪
        self.assertIn(request.request_id, self.worker_connector._alive_requests)


if __name__ == '__main__':
    unittest.main()
