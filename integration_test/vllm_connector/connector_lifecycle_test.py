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

import unittest
from vllm_connector_cases import VllmConnectorTestBase
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole


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

        # 3. 构建元数据并通过完整引擎步骤传递给 Worker
        scheduled_new_req = self._create_mock_scheduled_new_req(
            req_id=request.request_id,
            token_ids=list(range(1, 65)),
            block_ids=[0, 1, 2, 3],
        )

        scheduler_output = self._create_mock_scheduler_output(
            new_reqs=[scheduled_new_req],
            num_scheduled_tokens={request.request_id: 64}
        )

        meta, _, _ = self._simulate_engine_step(
            self.scheduler_connector, self.worker_connector, scheduler_output)

        # 5. 通过 metadata 协议验证请求信息已正确传递
        self.assertTrue(len(meta.requests) > 0, "meta 应包含请求状态")
        self.assertEqual(meta.requests[0].req_id, request.request_id)


class ConnectorKVCacheWriteReadTest(VllmConnectorTestBase):
    """KV Cache 完整读写流程测试

    创建两套独立的 Scheduler+Worker Connector 对（A 和 B），共享同一个 instance_id。
    MetaIndexer 按 instance_id 隔离，因此写入和读取匹配必须在同一个 instance_id 上进行。
    两套 Connector 对使用不同的 coordinator 端口，模拟同一 instance group 内的两个引擎进程。
    """

    def _init_connector(self):
        """初始化两套 Connector 对，共享同一个 instance_id"""
        from kv_cache_manager.py_connector.vllm.v1_connector import TairKvCacheConnector

        self._shared_instance_id = "instance_a"

        # Connector 对 A
        self._coordinator_port_a = self._get_free_port()
        config_a = self._create_test_vllm_config(
            instance_id=self._shared_instance_id,
            coordinator_base_port=self._coordinator_port_a,
        )
        self.scheduler_a = TairKvCacheConnector(config_a, KVConnectorRole.SCHEDULER)
        self.worker_a = TairKvCacheConnector(config_a, KVConnectorRole.WORKER)
        kv_caches_a = self._create_mock_kv_caches()
        self.worker_a.register_kv_caches(kv_caches_a)

        # Connector 对 B（同一个 instance_id，不同 coordinator 端口）
        self._coordinator_port_b = self._get_free_port()
        config_b = self._create_test_vllm_config(
            instance_id=self._shared_instance_id,
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
        """测试完整的写入-读取流程（两套 Connector 对，同一 instance_id）

        两套 Scheduler+Worker 共享同一个 instance_id，Manager 侧使用同一个 MetaSearcher。
        Step 1: Connector 对 A prefill + save
        Step 2: Connector 对 B 用新请求查询（应匹配到 Step 1 写入的缓存）
        Step 3: 请求完成
        """
        # ----- 准备测试数据 -----
        block_size = self._test_config.block_size  # 16
        # 64 tokens → 4 blocks（按 block_size=16 计算）
        prompt_token_ids = list(range(100, 164))
        num_tokens = len(prompt_token_ids)
        num_blocks = num_tokens // block_size
        block_ids_a = list(range(num_blocks))

        write_req_id = "write_read_req_write"

        # ===== Step 1: Connector 对 A 写入 KV Cache =====

        # 1.1 Scheduler A: get_num_new_matched_tokens（首次无缓存）
        request_a = self._create_mock_request(
            request_id=write_req_id,
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

        # 1.3 Scheduler A: build_connector_meta
        scheduled_new_req_a = self._create_mock_scheduled_new_req(
            req_id=write_req_id,
            token_ids=prompt_token_ids,
            block_ids=block_ids_a,
        )
        scheduler_output_a = self._create_mock_scheduler_output(
            new_reqs=[scheduled_new_req_a],
            num_scheduled_tokens={write_req_id: num_tokens},
        )
        meta_a, finished_saving_a, finished_loading_a = self._simulate_engine_step(
            self.scheduler_a, self.worker_a, scheduler_output_a,
        )

        # 验证 meta 中包含了新请求
        from kv_cache_manager.py_connector.vllm.metadata import TairKvCacheConnectorMetadata
        self.assertIsInstance(meta_a, TairKvCacheConnectorMetadata)
        self.assertTrue(len(meta_a.requests) > 0, "meta 应包含请求状态")
        self.assertEqual(meta_a.requests[0].req_id, write_req_id)

        # 轮询直到异步 start_save_kvcache_async 的结果被 build_connector_meta 收集
        self._poll_engine_until_save_collected(
            self.scheduler_a, self.worker_a, expected_count=1)

        # 第二轮 build_connector_meta 处理 cached req（decode token）
        scheduler_output_a2 = self._create_mock_scheduler_output(
            num_scheduled_tokens={write_req_id: 1},
        )

        scheduler_output_a2.scheduled_cached_reqs.req_ids = [write_req_id]
        scheduler_output_a2.scheduled_cached_reqs.new_block_ids = [None]
        scheduler_output_a2.scheduled_cached_reqs.resumed_req_ids = set()
        request_a.all_token_ids = prompt_token_ids + [200]

        meta_a2, _, _ = self._simulate_engine_step(
            self.scheduler_a, self.worker_a, scheduler_output_a2,
        )

        # 完成写请求
        self.scheduler_a.request_finished(request_a, block_ids_a)

        # 推送 FinishRequest 到 worker
        sched_out_finish = self._create_mock_scheduler_output()
        self._simulate_engine_step(self.scheduler_a, self.worker_a, sched_out_finish)

        # ===== Step 2: Connector 对 B 用新请求读取 KV Cache =====

        read_req_id = "write_read_req_read"
        request_read = self._create_mock_request(
            request_id=read_req_id,
            prompt_token_ids=prompt_token_ids,
        )

        # 轮询直到 Manager 中缓存数据可查询（替代 time.sleep(5)）
        matched_b, has_match_b = self._poll_until_cache_queryable(
            self.scheduler_b, request_read, timeout_seconds=30)

        # 同一个 instance_id，Connector 对 B 应匹配到 A 写入的缓存
        self.assertGreater(matched_b, 0, "应匹配到之前写入的 tokens")
        self.assertTrue(has_match_b, "有匹配时 has_match 应为 True")

        # 2.2 update_state_after_alloc
        block_ids_read = list(range(num_blocks))
        blocks_read = self._create_mock_kv_cache_blocks(block_ids_read)
        self.scheduler_b.update_state_after_alloc(
            request_read, blocks_read, num_external_tokens=matched_b
        )

        # 2.3 Build meta & engine step → 验证 LoadRequest 生成
        scheduled_new_req_read = self._create_mock_scheduled_new_req(
            req_id=read_req_id,
            token_ids=prompt_token_ids,
            block_ids=block_ids_read,
        )
        scheduler_output_read = self._create_mock_scheduler_output(
            new_reqs=[scheduled_new_req_read],
            num_scheduled_tokens={read_req_id: num_tokens},
        )
        meta_read, _, _ = self._simulate_engine_step(
            self.scheduler_b, self.worker_b, scheduler_output_read,
        )

        # 验证 meta 包含 LoadRequest
        self.assertTrue(
            len(meta_read.to_load_requests) > 0,
            "读取请求的 meta 应包含 LoadRequest"
        )
        self.assertEqual(meta_read.to_load_requests[0].req_id, read_req_id)

        # ===== Step 3: 请求完成 =====
        self.scheduler_b.request_finished(request_read, block_ids_read)

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

        # 轮询直到异步保存被收集（替代 time.sleep(2)）
        self._poll_engine_until_save_collected(
            self.scheduler_a, self.worker_a, expected_count=1)

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

        # 轮询直到 Manager 中缓存数据可查询
        matched2, has_match2 = self._poll_until_cache_queryable(
            self.scheduler_a, request2, timeout_seconds=30)

        # 验证：异步写入完成后，应能匹配到缓存
        self.assertGreater(matched2, 0, "应匹配到之前写入的缓存")

        # 清理：直接调用 request_finished（内部已处理不存在的请求，返回 (False, {})）
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

        # 执行 engine step
        sched_out = self._create_mock_scheduler_output(
            new_reqs=scheduled_new_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
        )
        meta, _, _ = self._simulate_engine_step(
            self.scheduler_a, self.worker_a, sched_out,
        )

        # 验证 meta 中包含所有新请求
        self.assertEqual(len(meta.requests), 3,
                         "meta 应包含 3 个请求状态")

        # 轮询直到 3 个异步保存都被收集
        self._poll_engine_until_save_collected(
            self.scheduler_a, self.worker_a, expected_count=3)

        # 逐个完成请求
        for req, tokens, block_ids in requests:
            result, extra_info = self.scheduler_a.request_finished(req, block_ids)
            self.assertTrue(result, f"{req.request_id} 应成功标记完成")

        # 推送 FinishRequest 到 worker
        sched_out2 = self._create_mock_scheduler_output()
        self._simulate_engine_step(self.scheduler_a, self.worker_a, sched_out2)

        # 验证 scheduler 中的请求已清理：通过公开 API 返回值验证
        # request_finished() 对不在 _alive_requests 中的请求返回 (False, {})
        for req, _, block_ids in requests:
            result, _ = self.scheduler_a.request_finished(req, block_ids)
            self.assertFalse(result,
                             f"Scheduler 中 {req.request_id} 应已清理"
                             f"（request_finished 返回 False）")


if __name__ == '__main__':
    unittest.main()
