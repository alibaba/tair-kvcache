# vLLM Connector 集成测试框架

## 概述

vLLM Connector 集成测试框架旨在解决以下问题：

1. **测试环境复杂性**：传统测试需要启动完整的 vLLM 服务，配置复杂且耗时
2. **CUDA 依赖**：vLLM Connector 依赖 CUDA 操作，无法在 CPU-only 环境运行
3. **服务依赖**：需要启动 KV Cache Manager 服务进行端到端测试

本框架通过以下方式解决这些问题：

- **真实 Manager 服务**：使用 TestBase 框架自动启动/停止 KV Cache Manager 服务
- **CUDA Mock**：仅 Mock CUDA 操作（Stream/Event），保持其他代码真实执行
- **vLLM Config Mock**：使用 MagicMock 替代真实的 VllmConfig，避免 HuggingFace 网络请求
- **Bazel 集成**：通过 Bazel 管理测试依赖和执行

## 目录结构

```
integration_test/vllm_connector/
├── BUILD                          # Bazel 构建配置
├── mock_cuda.py                   # CUDA 操作 Mock
├── vllm_connector_cases.py        # 测试基类和工具方法
└── connector_lifecycle_test.py    # 生命周期测试用例
```

## 核心组件

### 1. mock_cuda.py - CUDA Mock

提供 CUDA 操作的 Mock 实现，使测试可以在 CPU 环境运行：

```python
from integration_test.vllm_connector.mock_cuda import apply_cuda_patches

# 在测试 setUp 中应用 patches
apply_cuda_patches()
```

Mock 内容包括：
- `MockCudaStream`：模拟 CUDA Stream
- `MockCudaEvent`：模拟 CUDA Event
- `torch.cuda.current_stream()`
- `torch.cuda.Stream()`
- vLLM 分布式函数（`get_tp_group`, `get_world_group` 等）

### 2. vllm_connector_cases.py - 测试基类

`VllmConnectorTestBase` 提供：

- **服务生命周期管理**：自动启动/停止 KV Cache Manager
- **配置生成**：创建测试用的 VllmConfig 和 extra_config
- **Mock 对象工厂**：创建 Request、KVCacheBlocks、SchedulerOutput 等

关键配置参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `instance_group` | `"default"` | 使用 KVCM 默认创建的 instance group |
| `instance_id` | `"test_instance"` | 测试实例 ID |
| `block_size` | `16` | KV Cache block 大小 |
| `async_get_cache_location` | `False` | 同步查询以简化测试 |

### 3. 测试用例结构

测试用例继承 `VllmConnectorTestBase` 和 `unittest.TestCase`：

```python
class MyConnectorTest(VllmConnectorTestBase, unittest.TestCase):
    def _init_connector(self):
        """初始化 Connector（可覆盖）"""
        # 创建 scheduler 和 worker connector
        pass
    
    def test_my_feature(self):
        """测试用例"""
        # 使用 self.scheduler_connector 和 self.worker_connector
        pass
```

## 运行测试

### 运行所有测试

```bash
bazel test //integration_test/vllm_connector:connector_lifecycle_test
```

### 运行单个测试

```bash
bazel test //integration_test/vllm_connector:connector_lifecycle_test \
    --test_filter=test_scheduler_init
```

### 查看详细输出

```bash
bazel test //integration_test/vllm_connector:connector_lifecycle_test \
    --test_output=all
```

### 强制重新运行（不使用缓存）

```bash
bazel test //integration_test/vllm_connector:connector_lifecycle_test \
    --cache_test_results=no
```

## 添加新测试

### 步骤 1：创建测试文件

在 `integration_test/vllm_connector/` 目录下创建新的测试文件：

```python
# my_feature_test.py
import unittest
from integration_test.vllm_connector.vllm_connector_cases import VllmConnectorTestBase

class MyFeatureTest(VllmConnectorTestBase, unittest.TestCase):
    def _init_connector(self):
        """初始化测试所需的 Connector"""
        vllm_config = self._create_test_vllm_config()
        
        # 创建 Scheduler Connector
        self.scheduler_connector = TairKvCacheConnector(
            vllm_config=vllm_config,
            role=KVConnectorRole.SCHEDULER,
        )
        
        # 如果需要 Worker Connector
        self.worker_connector = TairKvCacheConnector(
            vllm_config=vllm_config,
            role=KVConnectorRole.WORKER,
            rank=0,
        )
    
    def test_my_feature(self):
        """测试用例"""
        # 创建测试请求
        request = self._create_mock_request(
            request_id="test_req",
            prompt_token_ids=list(range(1, 65)),
        )
        
        # 测试逻辑
        result = self.scheduler_connector.some_method(request)
        
        # 断言
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
```

### 步骤 2：更新 BUILD 文件

在 `integration_test/vllm_connector/BUILD` 中添加新的测试目标：

```python
py_test(
    name = "my_feature_test",
    srcs = ["my_feature_test.py"],
    deps = [
        ":vllm_connector_cases",
        "//kv_cache_manager/py_connector/vllm:vllm_connector",
    ],
    data = [
        "//kv_cache_manager/service:kv_cache_manager_bin",
    ],
)
```

### 步骤 3：运行测试验证

```bash
bazel test //integration_test/vllm_connector:my_feature_test --test_output=all
```

## 注意事项

### 1. 服务启动等待

KV Cache Manager 服务启动需要时间，`TestBase` 已处理等待逻辑。如果遇到连接问题，检查：
- 端口是否被占用
- 服务是否正常启动（查看 worker 目录下的 `stderr` 和 `kv_cache_manager.log`）

### 2. Instance Group

测试默认使用 `"default"` instance group，这是 KVCM 启动时自动创建的。如需使用其他 group，需要先通过 Admin API 创建。

### 3. 异步查询模式

`async_get_cache_location` 配置控制缓存位置查询模式：
- `False`（推荐用于测试）：同步查询，`get_num_new_matched_tokens` 立即返回结果
- `True`：异步查询，需要多次调用等待结果

### 4. Mock VllmConfig

测试使用 MagicMock 创建 VllmConfig，避免访问 HuggingFace。如需修改模型参数，在 `_create_test_vllm_config()` 中调整：

```python
vllm_config.model_config.get_num_layers.return_value = 12
vllm_config.model_config.get_num_kv_heads.return_value = 8
vllm_config.model_config.get_head_size.return_value = 64
```

### 5. 端口冲突

测试框架自动分配端口，但并行运行多个测试可能导致端口冲突。`TestBase` 使用端口范围哈希来减少冲突概率。

### 6. 资源清理

测试结束后会自动清理资源。测试工作目录位于 bazel runfiles 目录中（`~/.cache/bazel/.../runfiles/kv_cache_manager/integration_test/test_xxx/`），不会污染源代码目录。

如果测试异常退出，可能需要手动清理：

```bash
# 清理残留进程
pkill -f kv_cache_manager_bin

# 清理 bazel 测试缓存（可选）
bazel clean --expunge
```

### 7. CUDA Mock 局限性

Mock 仅模拟基本的 Stream/Event 行为，不支持：
- 真实的 GPU 内存分配
- CUDA 计算操作
- 多 GPU 场景

如需测试这些功能，需要在有 GPU 的环境中运行。

## 常见问题

### Q: 测试显示 "instance group not found"

A: 确保使用 `"default"` 作为 instance_group，或在测试前创建自定义 group。

### Q: 测试超时

A: 增加超时时间：
```bash
bazel test //integration_test/vllm_connector:xxx --test_timeout=300
```

### Q: Bazel 使用旧的测试结果

A: 使用 `--cache_test_results=no` 或删除测试日志后重新运行。

### Q: 如何查看服务端日志

A: 查看 worker 目录下的日志文件：
```bash
find ~/.cache/bazel -name "kv_cache_manager.log" -path "*vllm_connector*" | xargs tail -100
```

## 参考资料

- [vLLM KVConnector API](https://docs.vllm.ai/)
- [Bazel Python 测试文档](https://bazel.build/reference/be/python#py_test)
- [integration_test/meta_service](../integration_test/meta_service/) - Meta Service 集成测试参考
