"""
vLLM Connector 集成测试基类

该模块提供 vLLM Connector 集成测试的基础设施，包括：
- 测试基类 VllmConnectorTestBase
- VllmConfig 创建工具
- Mock Request 创建工具

测试运行方式：
1. 使用 Bazel（推荐）：
   bazel test //integration_test/vllm_connector:connector_lifecycle_test

2. 运行特定测试方法：
   bazel test //integration_test/vllm_connector:connector_lifecycle_test --test_filter=test_method_name

3. 查看测试输出：
   bazel test //integration_test/vllm_connector:connector_lifecycle_test --test_output=all

参考文件：
- Connector 实现：kv_cache_manager/py_connector/vllm/v1_connector.py
- 测试基类：integration_test/testlib/test_base.py
- meta_service 测试：integration_test/meta_service/meta_interface_cases.py
"""

import abc
import os
import shutil
import tempfile
import time
import unittest
import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from testlib.test_base import TestBase
from mock_cuda import apply_cuda_patches, apply_distributed_patches
from integration_test.admin_service.http_interface_test import AdminServiceHttpClient


# 在导入 vLLM 之前应用 CUDA patches
apply_cuda_patches()
apply_distributed_patches()


@dataclass
class TestConnectorConfig:
    """测试用 Connector 配置"""
    manager_uri: str
    coordinator_base_port: int = 50000
    instance_group: str = "vllm_test"  # 通过 Admin API 创建的测试 instance group
    instance_id: str = "test_instance"
    preferred_block_size: int = 16
    block_size: int = 16
    num_layers: int = 4
    num_kv_heads: int = 8
    head_size: int = 64
    tp_size: int = 1


class VllmConnectorTestBase(abc.ABC, TestBase, unittest.TestCase):
    """vLLM Connector 测试基类

    该基类提供：
    - Manager 服务的启动和停止
    - 通过 Admin API 配置 NFS 存储后端和 instance group
    - VllmConfig 的创建
    - Mock KV Cache 的创建
    - Mock Request 的创建

    子类需要实现：
    - _init_connector(): 初始化 Connector
    - _cleanup_connector(): 清理 Connector
    """

    _STORAGE_NAME = "test_nfs_vllm"
    _INSTANCE_GROUP_NAME = "vllm_test"

    @classmethod
    def setUpClass(cls):
        """类级别的初始化，在所有测试之前执行一次"""
        # CUDA patches 已经在模块导入时应用
        pass

    def setUp(self):
        """每个测试方法之前执行"""
        # 启动 Manager 服务
        self.init_default()

        # 通过 Admin API 配置可用的存储后端
        self._setup_storage_backend()

        # 获取服务端口
        worker = self.worker_manager.get_worker(0)
        self._http_port = worker.env.http_port
        self._rpc_port = worker.env.rpc_port
        self._manager_uri = f"http://localhost:{self._http_port}"

        # 创建测试配置 - 使用通过 Admin API 创建的 instance group
        self._test_config = TestConnectorConfig(
            manager_uri=self._manager_uri,
            coordinator_base_port=self._get_free_port(),
            instance_group=self._INSTANCE_GROUP_NAME,
        )

        # 初始化 Connector（由子类实现）
        self._init_connector()

    def tearDown(self):
        """每个测试方法之后执行"""
        # 清理 Connector（由子类实现）
        self._cleanup_connector()
        # 清理存储后端临时目录
        self._cleanup_storage_backend()
        # 停止 Manager 服务
        self.cleanup()

    def _setup_storage_backend(self):
        """通过 Admin API 配置 NFS 存储后端和 instance group"""
        # 创建临时目录作为 NFS 存储根路径
        self._nfs_tmpdir = tempfile.mkdtemp(prefix="kvcm_vllm_test_")

        # 获取 Admin HTTP 端口，创建 Admin 客户端
        admin_http_port = self.worker_manager.get_worker(0).env.admin_http_port
        self._admin_client = AdminServiceHttpClient(
            f"http://localhost:{admin_http_port}"
        )

        # 1. 添加 NFS 存储
        self._admin_client.add_storage({
            "trace_id": "vllm_test_add_storage",
            "storage": {
                "global_unique_name": self._STORAGE_NAME,
                "nfs": {
                    "root_path": self._nfs_tmpdir + "/",
                },
            },
        })

        # 2. 创建 instance group
        self._admin_client.create_instance_group({
            "trace_id": "vllm_test_create_ig",
            "instance_group": {
                "name": self._INSTANCE_GROUP_NAME,
                "storage_candidates": [self._STORAGE_NAME],
                "global_quota_group_name": "vllm_test_quota",
                "max_instance_count": 100,
                "quota": {
                    "capacity": 10000000000,
                    "quota_config": [
                        {"storage_type": 4, "capacity": 10000000000},  # ST_NFS=4
                    ],
                },
                "cache_config": {
                    "reclaim_strategy": {
                        "storage_unique_name": self._STORAGE_NAME,
                        "reclaim_policy": 1,  # POLICY_LRU
                        "trigger_strategy": {"used_percentage": 0.95},
                        "delay_before_delete_ms": 1000,
                    },
                    "data_storage_strategy": 2,  # CPS_PREFER_3FS
                    "meta_indexer_config": {
                        "max_key_count": 1000000,
                        "mutex_shard_num": 16,
                        "meta_storage_backend_config": {
                            "storage_type": "local",
                            "storage_uri": "",
                        },
                        "meta_cache_policy_config": {
                            "type": "LRU",
                            "capacity": 10000,
                        },
                    },
                },
                "version": 1,
            },
        })

    def _cleanup_storage_backend(self):
        """清理临时 NFS 存储目录"""
        if hasattr(self, '_admin_client') and self._admin_client:
            self._admin_client.close()
        if hasattr(self, '_nfs_tmpdir') and self._nfs_tmpdir and os.path.exists(self._nfs_tmpdir):
            shutil.rmtree(self._nfs_tmpdir, ignore_errors=True)

    @abc.abstractmethod
    def _init_connector(self):
        """初始化 Connector，由子类实现"""
        pass

    @abc.abstractmethod
    def _cleanup_connector(self):
        """清理 Connector，由子类实现"""
        pass

    def _get_free_port(self) -> int:
        """获取一个空闲端口用于 coordinator"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def _get_test_extra_config(
            self,
            instance_id: Optional[str] = None,
            coordinator_base_port: Optional[int] = None,
    ) -> Dict[str, Any]:
        """创建测试用 kv_connector_extra_config

        Args:
            instance_id: 可选覆盖 instance_id，用于多 instance 测试
            coordinator_base_port: 可选覆盖 coordinator_base_port
        """
        return {
            "manager_uri": self._test_config.manager_uri,
            "coordinator_base_port": coordinator_base_port or self._test_config.coordinator_base_port,
            "instance_group": self._test_config.instance_group,
            "instance_id": instance_id or self._test_config.instance_id,
            "preferred_block_size": self._test_config.preferred_block_size,
            "async_get_cache_location": False,  # 同步查询以简化测试
        }

    def _get_test_model_deployment(self) -> Dict[str, Any]:
        """创建测试用 model deployment 配置"""
        return {
            "model_name": "test_model",
            "dtype": "float16",
            "use_mla": False,
            "tp_size": self._test_config.tp_size,
            "dp_size": 1,
            "pp_size": 1,
        }

    def _create_test_vllm_config(
            self,
            instance_id: Optional[str] = None,
            coordinator_base_port: Optional[int] = None,
    ):
        """创建测试用 VllmConfig

        Args:
            instance_id: 可选覆盖 instance_id，用于多 instance 测试
            coordinator_base_port: 可选覆盖 coordinator_base_port

        Returns:
            VllmConfig 对象（mock），配置了测试所需的参数

        注意：使用 MagicMock 避免 vLLM 访问 HuggingFace 下载模型配置
        """
        from unittest.mock import MagicMock
        from vllm.config import KVTransferConfig

        extra_config = self._get_test_extra_config(
            instance_id=instance_id,
            coordinator_base_port=coordinator_base_port,
        )

        # 创建 KVTransferConfig - 这个是真实的，包含 connector 配置
        kv_transfer_config = KVTransferConfig(
            kv_connector="TairKvCacheConnector",
            kv_role="kv_both",
            kv_connector_extra_config=extra_config,
        )

        # 使用 MagicMock 创建 VllmConfig 避免网络请求
        vllm_config = MagicMock()

        # Mock ModelConfig
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.model = "test_model"
        vllm_config.model_config.served_model_name = "test_model"
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.use_mla = False
        vllm_config.model_config.get_num_layers.return_value = self._test_config.num_layers
        vllm_config.model_config.get_num_kv_heads.return_value = self._test_config.num_kv_heads
        vllm_config.model_config.get_head_size.return_value = self._test_config.head_size
        vllm_config.model_config.hf_config = MagicMock()
        vllm_config.model_config.hf_config.num_hidden_layers = self._test_config.num_layers
        vllm_config.model_config.hf_config.num_attention_heads = self._test_config.num_kv_heads
        vllm_config.model_config.hf_config.hidden_size = self._test_config.head_size * self._test_config.num_kv_heads

        # Mock CacheConfig
        vllm_config.cache_config = MagicMock()
        vllm_config.cache_config.block_size = self._test_config.block_size
        vllm_config.cache_config.cache_dtype = "auto"
        vllm_config.cache_config.num_gpu_blocks = 100
        vllm_config.cache_config.num_cpu_blocks = 0

        # Mock ParallelConfig
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = self._test_config.tp_size
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.world_size = self._test_config.tp_size

        # Mock SchedulerConfig
        vllm_config.scheduler_config = MagicMock()
        vllm_config.scheduler_config.max_num_seqs = 16
        vllm_config.scheduler_config.max_num_batched_tokens = 64
        vllm_config.scheduler_config.max_model_len = 1000

        # Mock DeviceConfig
        vllm_config.device_config = MagicMock()
        vllm_config.device_config.device = "cpu"
        vllm_config.device_config.device_type = "cpu"

        # 设置真实的 KVTransferConfig
        vllm_config.kv_transfer_config = kv_transfer_config

        return vllm_config

    def _create_mock_kv_caches(
            self,
            num_layers: Optional[int] = None,
            num_blocks: int = 100,
            block_size: Optional[int] = None,
            num_kv_heads: Optional[int] = None,
            head_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """创建 CPU 上的 Mock KV Cache 张量
        
        Args:
            num_layers: 层数，默认使用配置中的值
            num_blocks: block 数量
            block_size: block 大小，默认使用配置中的值
            num_kv_heads: KV head 数量，默认使用配置中的值
            head_size: head 大小，默认使用配置中的值
        
        Returns:
            Dict[str, torch.Tensor]: layer name 到 KV cache tensor 的映射
        """
        num_layers = num_layers or self._test_config.num_layers
        block_size = block_size or self._test_config.block_size
        num_kv_heads = num_kv_heads or self._test_config.num_kv_heads
        head_size = head_size or self._test_config.head_size

        kv_caches = {}
        for i in range(num_layers):
            # Shape: [2, num_blocks, block_size, num_kv_heads, head_size]
            # 2 表示 key 和 value
            kv_caches[f"layer_{i}"] = torch.zeros(
                2, num_blocks, block_size, num_kv_heads, head_size,
                dtype=torch.float16,
                device="cpu",
            ).contiguous()
        return kv_caches

    def _create_mock_request(
            self,
            request_id: str = "test_request_1",
            prompt_token_ids: Optional[list] = None,
            max_tokens: int = 10,
    ):
        """创建 Mock vLLM Request 对象
        
        Args:
            request_id: 请求 ID
            prompt_token_ids: prompt token IDs，默认生成随机 tokens
            max_tokens: 最大生成 token 数
        
        Returns:
            Mock Request 对象
        """
        from unittest.mock import MagicMock

        if prompt_token_ids is None:
            # 生成一些测试用的 token IDs
            prompt_token_ids = list(range(1, 33))  # 32 tokens

        request = MagicMock()
        request.request_id = request_id
        request.prompt_token_ids = prompt_token_ids
        request.all_token_ids = prompt_token_ids.copy()

        # Mock sampling_params
        request.sampling_params = MagicMock()
        request.sampling_params.max_tokens = max_tokens

        return request

    def _create_mock_scheduler_output(
            self,
            new_reqs: Optional[list] = None,
            num_scheduled_tokens: Optional[Dict[str, int]] = None,
    ):
        """创建 Mock SchedulerOutput 对象
        
        Args:
            new_reqs: 新调度的请求列表
            num_scheduled_tokens: 请求 ID 到调度 token 数的映射
        
        Returns:
            Mock SchedulerOutput 对象
        """
        from unittest.mock import MagicMock

        scheduler_output = MagicMock()

        # scheduled_new_reqs
        if new_reqs is None:
            new_reqs = []
        scheduler_output.scheduled_new_reqs = new_reqs

        # scheduled_cached_reqs
        cached_reqs = MagicMock()
        cached_reqs.req_ids = []
        cached_reqs.new_block_ids = []
        cached_reqs.resumed_req_ids = set()
        scheduler_output.scheduled_cached_reqs = cached_reqs

        # num_scheduled_tokens
        if num_scheduled_tokens is None:
            num_scheduled_tokens = {}
        scheduler_output.num_scheduled_tokens = num_scheduled_tokens

        return scheduler_output

    def _create_mock_kv_cache_blocks(self, block_ids: list):
        """创建 Mock KVCacheBlocks 对象

        Args:
            block_ids: block ID 列表

        Returns:
            Mock KVCacheBlocks 对象
        """
        from unittest.mock import MagicMock

        blocks = MagicMock()
        blocks.get_block_ids = MagicMock(return_value=[block_ids])
        return blocks

    def _create_mock_scheduled_new_req(
            self,
            req_id: str,
            token_ids: List[int],
            block_ids: List[int],
    ):
        """创建 Mock 的 SchedulerOutput.scheduled_new_reqs 元素

        模拟 vllm 新调度请求的数据结构。

        Args:
            req_id: 请求 ID
            token_ids: token ID 列表
            block_ids: 分配的 block ID 列表

        Returns:
            Mock scheduled_new_req 对象
        """
        from unittest.mock import MagicMock

        scheduled_new_req = MagicMock()
        scheduled_new_req.req_id = req_id
        scheduled_new_req.block_ids = [block_ids]  # 外层 list 对应 KV cache groups
        scheduled_new_req.token_ids = token_ids
        return scheduled_new_req

    def _simulate_engine_step(
            self,
            scheduler_connector,
            worker_connector,
            scheduler_output,
    ):
        """模拟一个完整的 vllm 引擎步骤

        按照 vllm 引擎的实际调用顺序执行:
        build_connector_meta → bind → start_load → wait_for_save → get_finished → clear

        Args:
            scheduler_connector: Scheduler 角色的 Connector
            worker_connector: Worker 角色的 Connector
            scheduler_output: Mock SchedulerOutput 对象

        Returns:
            tuple: (meta, finished_saving, finished_loading, save_request_count)
                save_request_count 是本步骤中 to_save_requests 的数量，
                在 wait_for_save 之前读取（处理后会清空）。
        """
        # 1. Scheduler: build meta
        meta = scheduler_connector.build_connector_meta(scheduler_output)

        # 2. Worker: bind metadata
        worker_connector.bind_connector_metadata(meta)

        # 3. Worker: start_load_kv (需要 forward_context，但在测试中可以传 None)
        from unittest.mock import MagicMock
        mock_forward_ctx = MagicMock()
        worker_connector.start_load_kv(mock_forward_ctx)

        # 4. Worker: wait_for_save
        # 提前记录 save 请求数（_process_save_requests 处理后会清空 to_save_requests）
        save_request_count = len(meta.to_save_requests)
        worker_connector.wait_for_save()

        # 5. Worker: get_finished
        finished_req_ids = set()
        finished_saving, finished_loading = worker_connector.get_finished(finished_req_ids)

        # 6. Worker: get_block_ids_with_load_errors
        worker_connector.get_block_ids_with_load_errors()

        # 7. Worker: clear_connector_metadata
        worker_connector.clear_connector_metadata()

        return meta, finished_saving, finished_loading, save_request_count

    def _poll_engine_until_save_collected(
        self, scheduler, worker, expected_count=1,
        timeout_seconds=10, poll_interval=0.3,
    ):
        """轮询引擎步骤直到收集到预期数量的 SaveRequest

        模拟 vllm 引擎循环：反复执行 engine step，等待异步
        start_save_kvcache_async HTTP 完成后被 build_connector_meta 收集。

        Args:
            scheduler: Scheduler 角色的 Connector
            worker: Worker 角色的 Connector
            expected_count: 预期收集的 SaveRequest 数量
            timeout_seconds: 超时时间
            poll_interval: 轮询间隔
        """
        collected = 0
        deadline = time.time() + timeout_seconds
        while collected < expected_count:
            if time.time() > deadline:
                self.fail(
                    f"等待 SaveRequest 收集超时 ({timeout_seconds}s)，"
                    f"已收集 {collected}/{expected_count}")
            empty_out = self._create_mock_scheduler_output()
            _, _, _, save_count = self._simulate_engine_step(scheduler, worker, empty_out)
            collected += save_count
            if collected < expected_count:
                time.sleep(poll_interval)
        return collected

    def _simulate_engine_step_no_forward(
            self,
            scheduler_connector,
            worker_connector,
            scheduler_output,
    ):
        """模拟一个无 forward 工作的 vllm 引擎步骤（kv_connector_no_forward 路径）

        当 worker 没有 forward 工作时，vllm 调用 kv_connector_no_forward，
        内部使用 _get_kv_connector_output(scheduler_output, wait_for_save=False)。
        这意味着 wait_for_save 不会被调用。

        参考：vllm/v1/worker/kv_connector_model_runner_mixin.py:75-92
              vllm/v1/worker/gpu_model_runner.py:2682-2687

        Args:
            scheduler_connector: Scheduler 角色的 Connector
            worker_connector: Worker 角色的 Connector
            scheduler_output: Mock SchedulerOutput 对象

        Returns:
            tuple: (meta, finished_saving, finished_loading)
        """
        # 1. Scheduler: build meta
        meta = scheduler_connector.build_connector_meta(scheduler_output)

        # 2. Worker: bind metadata
        worker_connector.bind_connector_metadata(meta)

        # 3. Worker: start_load_kv
        from unittest.mock import MagicMock
        mock_forward_ctx = MagicMock()
        worker_connector.start_load_kv(mock_forward_ctx)

        # 4. 关键差异：跳过 wait_for_save（模拟 kv_connector_no_forward 路径）
        #    vllm 在 _get_kv_connector_output 中传入 wait_for_save=False

        # 5. Worker: get_finished
        finished_req_ids = set()
        finished_saving, finished_loading = worker_connector.get_finished(finished_req_ids)

        # 6. Worker: get_block_ids_with_load_errors
        worker_connector.get_block_ids_with_load_errors()

        # 7. Worker: clear_connector_metadata
        worker_connector.clear_connector_metadata()

        return meta, finished_saving, finished_loading

    def _poll_until_cache_queryable(
        self, scheduler, request, timeout_seconds=30, poll_interval=0.5,
    ):
        """轮询直到 Manager 中缓存数据可被查询到

        反复调用 get_num_new_matched_tokens 直到返回 matched > 0，
        表明完整的保存管道（NFS 写入 + finish_write_cache）已完成。

        Args:
            scheduler: Scheduler 角色的 Connector
            request: Mock Request 对象
            timeout_seconds: 超时时间
            poll_interval: 轮询间隔

        Returns:
            tuple: (matched, has_match) 查询结果
        """
        deadline = time.time() + timeout_seconds
        while True:
            matched, has_match = scheduler.get_num_new_matched_tokens(request, 0)
            if matched is not None and matched > 0:
                return matched, has_match
            if time.time() > deadline:
                self.fail(f"等待缓存可查询超时 ({timeout_seconds}s)")
            time.sleep(poll_interval)
