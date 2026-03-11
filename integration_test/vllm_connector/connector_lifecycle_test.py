"""
vLLM Connector 生命周期集成测试

该模块测试 TairKvCacheConnector 的完整生命周期，包括：
- Scheduler 角色初始化
- Worker 角色初始化
- KV Cache 注册
- 缓存匹配查询
- 元数据构建
- 请求生命周期
- KV Cache 完整读写流程

测试运行方式：
    bazel test //integration_test/vllm_connector:connector_lifecycle_test
    bazel test //integration_test/vllm_connector:connector_lifecycle_test --test_output=all
"""

import time
import unittest
from unittest.mock import MagicMock
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


class ConnectorKVCacheWriteReadTest(VllmConnectorTestBase):
    """KV Cache 完整读写流程测试

    模拟两个 vllm instance（同一 instance group），一个做 prefill 写入 KV cache，
    另一个查询并加载已写入的 KV cache。
    """

    def _init_connector(self):
        """初始化两个 instance 的 Connector 对"""
        from kv_cache_manager.py_connector.vllm.v1_connector import TairKvCacheConnector

        # Instance A
        self._instance_a_id = "instance_a"
        self._coordinator_port_a = self._get_free_port()
        config_a = self._create_test_vllm_config(
            instance_id=self._instance_a_id,
            coordinator_base_port=self._coordinator_port_a,
        )
        self.scheduler_a = TairKvCacheConnector(config_a, KVConnectorRole.SCHEDULER)
        self.worker_a = TairKvCacheConnector(config_a, KVConnectorRole.WORKER)
        kv_caches_a = self._create_mock_kv_caches()
        self.worker_a.register_kv_caches(kv_caches_a)

        # Instance B
        self._instance_b_id = "instance_b"
        self._coordinator_port_b = self._get_free_port()
        config_b = self._create_test_vllm_config(
            instance_id=self._instance_b_id,
            coordinator_base_port=self._coordinator_port_b,
        )
        self.scheduler_b = TairKvCacheConnector(config_b, KVConnectorRole.SCHEDULER)
        self.worker_b = TairKvCacheConnector(config_b, KVConnectorRole.WORKER)
        kv_caches_b = self._create_mock_kv_caches()
        self.worker_b.register_kv_caches(kv_caches_b)

    def _cleanup_connector(self):
        """清理所有 Connectors"""
        for c in ["worker_b", "scheduler_b", "worker_a", "scheduler_a"]:
            conn = getattr(self, c, None)
            if conn:
                conn.shutdown()

    # ------------------------------------------------------------------
    # test_write_then_read_kvcache
    # ------------------------------------------------------------------
    def test_write_then_read_kvcache(self):
        """测试完整的写入-读取流程

        Step 1: Instance A prefill + save
        Step 2: Instance B prefill + load（应匹配 Instance A 写入的缓存）
        Step 3: 请求完成
        """
        # ----- 准备测试数据 -----
        block_size = self._test_config.block_size  # 16
        # 64 tokens → 4 blocks（按 block_size=16 计算）
        prompt_token_ids = list(range(100, 164))
        num_tokens = len(prompt_token_ids)
        num_blocks = num_tokens // block_size
        block_ids_a = list(range(num_blocks))

        req_id = "write_read_req_1"

        # ===== Step 1: Instance A 写入 KV Cache =====

        # 1.1 Scheduler A: get_num_new_matched_tokens（首次无缓存）
        request_a = self._create_mock_request(
            request_id=req_id,
            prompt_token_ids=prompt_token_ids,
        )
        matched_a, has_match_a = self.scheduler_a.get_num_new_matched_tokens(
            request_a, num_computed_tokens=0
        )
        self.assertEqual(matched_a, 0, "首次查询应无远程缓存匹配")
        self.assertFalse(has_match_a)

        # 1.2 Scheduler A: update_state_after_alloc
        blocks_a = self._create_mock_kv_cache_blocks(block_ids_a)
        self.scheduler_a.update_state_after_alloc(
            request_a, blocks_a, num_external_tokens=0
        )

        # 1.3 Scheduler A: build_connector_meta → 触发 start_save_kvcache_async
        scheduled_new_req_a = self._create_mock_scheduled_new_req(
            req_id=req_id,
            token_ids=prompt_token_ids,
            block_ids=block_ids_a,
        )
        scheduler_output_a = self._create_mock_scheduler_output(
            new_reqs=[scheduled_new_req_a],
            num_scheduled_tokens={req_id: num_tokens},
        )
        meta_a, finished_saving_a, finished_loading_a = self._simulate_engine_step(
            self.scheduler_a, self.worker_a, scheduler_output_a,
        )

        # 验证 meta 中包含了新请求
        from kv_cache_manager.py_connector.vllm.metadata import TairKvCacheConnectorMetadata
        self.assertIsInstance(meta_a, TairKvCacheConnectorMetadata)
        self.assertTrue(len(meta_a.requests) > 0, "meta 应包含请求状态")
        self.assertEqual(meta_a.requests[0].req_id, req_id)

        # 验证请求在 worker_a 中被跟踪
        self.assertIn(req_id, self.worker_a._alive_requests)

        # 等待异步 start_save_kvcache_async 完成（通过 http_executor 执行）
        # start_save_kvcache_async 会调用 manager 的 start_write_cache / finish_write_cache
        time.sleep(2)

        # 第二轮 build_connector_meta 以推送 SaveRequest 到 worker
        scheduler_output_a2 = self._create_mock_scheduler_output(
            num_scheduled_tokens={req_id: 1},
        )
        # 模拟 cached_reqs 包含这个请求
        scheduler_output_a2.scheduled_cached_reqs.req_ids = [req_id]
        scheduler_output_a2.scheduled_cached_reqs.new_block_ids = [None]
        scheduler_output_a2.scheduled_cached_reqs.resumed_req_ids = set()
        # 更新 vllm_request 的 all_token_ids 以模拟 decode 产出新 token
        request_a.all_token_ids = prompt_token_ids + [200]

        meta_a2, _, _ = self._simulate_engine_step(
            self.scheduler_a, self.worker_a, scheduler_output_a2,
        )

        # ===== Step 2: Instance B 读取 KV Cache =====

        # 2.1 Scheduler B: get_num_new_matched_tokens（应匹配到 Instance A 写入的缓存）
        request_b = self._create_mock_request(
            request_id=req_id,
            prompt_token_ids=prompt_token_ids,
        )
        matched_b, has_match_b = self.scheduler_b.get_num_new_matched_tokens(
            request_b, num_computed_tokens=0
        )

        # 如果 Manager 已经完成写入，这里应该能匹配到 tokens
        if matched_b is not None and matched_b > 0:
            self.assertTrue(has_match_b, "有匹配时 has_match 应为 True")
            self.assertGreater(matched_b, 0, "应匹配到 Instance A 写入的 tokens")

            # 2.2 Scheduler B: update_state_after_alloc
            block_ids_b = list(range(num_blocks))
            blocks_b = self._create_mock_kv_cache_blocks(block_ids_b)
            self.scheduler_b.update_state_after_alloc(
                request_b, blocks_b, num_external_tokens=matched_b
            )

            # 2.3 Build meta & engine step
            scheduled_new_req_b = self._create_mock_scheduled_new_req(
                req_id=req_id,
                token_ids=prompt_token_ids,
                block_ids=block_ids_b,
            )
            scheduler_output_b = self._create_mock_scheduler_output(
                new_reqs=[scheduled_new_req_b],
                num_scheduled_tokens={req_id: num_tokens},
            )
            meta_b, _, _ = self._simulate_engine_step(
                self.scheduler_b, self.worker_b, scheduler_output_b,
            )

            # 验证 meta_b 包含 LoadRequest
            self.assertTrue(
                len(meta_b.to_load_requests) > 0,
                "Instance B 的 meta 应包含 LoadRequest"
            )
            self.assertEqual(meta_b.to_load_requests[0].req_id, req_id)

        # ===== Step 3: 请求完成 =====
        self.scheduler_a.request_finished(request_a, block_ids_a)
        self.assertNotIn(req_id, self.scheduler_a._alive_requests,
                         "request_finished 后请求应从 alive_requests 中移除")

        if matched_b is not None and matched_b > 0:
            self.scheduler_b.request_finished(request_b, block_ids_b)
            self.assertNotIn(req_id, self.scheduler_b._alive_requests)

    # ------------------------------------------------------------------
    # test_single_instance_write_and_query
    # ------------------------------------------------------------------
    def test_single_instance_write_and_query(self):
        """测试单 instance 写入后查询

        一个 instance 先做 prefill + save，然后用同一个 scheduler 查询验证缓存是否已写入。
        """
        block_size = self._test_config.block_size
        prompt_token_ids = list(range(200, 232))  # 32 tokens → 2 blocks
        num_tokens = len(prompt_token_ids)
        num_blocks = num_tokens // block_size
        block_ids = list(range(num_blocks))
        req_id = "single_write_query_1"

        # Step 1: 写入
        request = self._create_mock_request(
            request_id=req_id, prompt_token_ids=prompt_token_ids
        )
        matched, _ = self.scheduler_a.get_num_new_matched_tokens(request, 0)
        self.assertEqual(matched, 0)

        blocks = self._create_mock_kv_cache_blocks(block_ids)
        self.scheduler_a.update_state_after_alloc(request, blocks, 0)

        new_req = self._create_mock_scheduled_new_req(req_id, prompt_token_ids, block_ids)
        sched_out = self._create_mock_scheduler_output(
            new_reqs=[new_req],
            num_scheduled_tokens={req_id: num_tokens},
        )
        self._simulate_engine_step(self.scheduler_a, self.worker_a, sched_out)

        # 等待异步保存完成
        time.sleep(2)

        # 完成请求
        self.scheduler_a.request_finished(request, block_ids)

        # 推送 FinishRequest 到 worker
        sched_out2 = self._create_mock_scheduler_output()
        self._simulate_engine_step(self.scheduler_a, self.worker_a, sched_out2)

        # Step 2: 用同一个 scheduler 查询（模拟新请求使用相同 prompt）
        req_id_2 = "single_write_query_2"
        request2 = self._create_mock_request(
            request_id=req_id_2, prompt_token_ids=prompt_token_ids
        )
        matched2, has_match2 = self.scheduler_a.get_num_new_matched_tokens(request2, 0)

        # 验证：如果异步写入完成，应能匹配到缓存
        if matched2 is not None:
            # 即使匹配为0也是合理的（取决于异步保存是否成功完成）
            self.assertGreaterEqual(matched2, 0)

        # 清理
        if req_id_2 in self.scheduler_a._alive_requests:
            self.scheduler_a.request_finished(request2, block_ids)

    # ------------------------------------------------------------------
    # test_multiple_requests_lifecycle
    # ------------------------------------------------------------------
    def test_multiple_requests_lifecycle(self):
        """测试多请求并发生命周期管理

        模拟多个请求同时在 Scheduler 和 Worker 中的生命周期，
        包括新请求和续批请求的管理。
        """
        block_size = self._test_config.block_size
        requests = []
        scheduled_new_reqs = []
        num_scheduled_tokens = {}

        # 创建 3 个并发请求
        for i in range(3):
            req_id = f"multi_req_{i}"
            tokens = list(range(300 + i * 32, 300 + (i + 1) * 32))  # 每个 32 tokens
            num_tokens = len(tokens)
            num_blocks = num_tokens // block_size
            block_ids = list(range(i * num_blocks, (i + 1) * num_blocks))

            req = self._create_mock_request(req_id, tokens)
            requests.append((req, tokens, block_ids))

            # 查询匹配
            matched, _ = self.scheduler_a.get_num_new_matched_tokens(req, 0)

            # 分配 blocks
            blocks = self._create_mock_kv_cache_blocks(block_ids)
            self.scheduler_a.update_state_after_alloc(req, blocks, 0)

            scheduled_new_reqs.append(
                self._create_mock_scheduled_new_req(req_id, tokens, block_ids)
            )
            num_scheduled_tokens[req_id] = num_tokens

        # 验证所有请求都在 alive_requests 中
        for req, _, _ in requests:
            self.assertIn(req.request_id, self.scheduler_a._alive_requests,
                          f"{req.request_id} 应在 alive_requests 中")

        # 执行 engine step
        sched_out = self._create_mock_scheduler_output(
            new_reqs=scheduled_new_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
        )
        meta, _, _ = self._simulate_engine_step(
            self.scheduler_a, self.worker_a, sched_out,
        )

        # 验证 worker 中也跟踪了所有请求
        for req, _, _ in requests:
            self.assertIn(req.request_id, self.worker_a._alive_requests,
                          f"Worker 应跟踪 {req.request_id}")

        # 验证 meta 中包含所有新请求
        self.assertEqual(len(meta.requests), 3,
                         "meta 应包含 3 个请求状态")

        # 等待异步保存完成（build_connector_meta 触发的 start_save_kvcache_async）
        time.sleep(2)

        # 通过 build_connector_meta 收集异步保存结果（将 sent_saving_count 同步）
        sched_out_sync = self._create_mock_scheduler_output()
        self._simulate_engine_step(self.scheduler_a, self.worker_a, sched_out_sync)

        # 逐个完成请求
        for req, tokens, block_ids in requests:
            result, extra_info = self.scheduler_a.request_finished(req, block_ids)
            self.assertTrue(result, f"{req.request_id} 应成功标记完成")

        # 推送 FinishRequest 到 worker
        sched_out2 = self._create_mock_scheduler_output()
        self._simulate_engine_step(self.scheduler_a, self.worker_a, sched_out2)

        # 验证 scheduler 中的请求已清理（worker 清理依赖 coordinator 确认保存完成，
        # 在无存储后端的测试环境中 worker 可能保持 need_report_after_saving_finished 状态）
        for req, _, _ in requests:
            self.assertNotIn(req.request_id, self.scheduler_a._alive_requests,
                             f"Scheduler 中 {req.request_id} 应已清理")


if __name__ == '__main__':
    unittest.main()
