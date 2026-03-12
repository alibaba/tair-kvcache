# vLLM Connector 集成测试框架

## 概述

vLLM Connector 集成测试框架旨在解决以下问题：

1. **测试环境复杂性**：传统测试需要启动完整的 vLLM 服务，配置复杂且耗时
2. **CUDA 依赖**：vLLM Connector 依赖 CUDA 操作，无法在 CPU-only 环境运行
3. **服务依赖**：需要启动 KV Cache Manager 服务进行端到端测试

本框架通过以下方式解决这些问题：

- **真实 Manager 服务**：使用 TestBase 框架自动启动/停止 KV Cache Manager 服务
- **CUDA Mock**：仅 Mock CUDA 操作（Stream/Event），保持其他代码真实执行
- **Triton Mock**：在无 triton 环境下自动 mock triton 模块，使导入链不中断；有 triton 时使用真实模块
- **GPU 内核 Mock**：将 `batch_gather_scatter_helper` 中的 GPU 内核函数替换为空操作
- **vLLM Config Mock**：使用 MagicMock 替代真实的 VllmConfig，避免 HuggingFace 网络请求
- **Bazel 集成**：通过 Bazel 管理测试依赖和执行

## 目录结构

```
integration_test/vllm_connector/
├── BUILD                          # Bazel 构建配置
├── mock_cuda.py                   # CUDA/Triton/GPU 内核 Mock
├── vllm_connector_cases.py        # 测试基类和工具方法
└── connector_lifecycle_test.py    # 生命周期及读写流程测试用例
```

## 核心组件

### 1. mock_cuda.py - CUDA / Triton / GPU 内核 Mock

提供多层 Mock 实现，使测试可以在 CPU 环境运行。

**使用方式**：在导入 vLLM **之前**调用 `apply_cuda_patches()`，这已由 `vllm_connector_cases.py` 在模块加载时自动完成。

```python
from mock_cuda import apply_cuda_patches, apply_distributed_patches

# 必须在 import vllm 之前调用
apply_cuda_patches()
apply_distributed_patches()

# 之后才能安全 import vllm
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
```

#### Mock 层次

| 层次 | 内容 | 说明 |
|------|------|------|
| CUDA 基础 | `MockCudaStream`、`MockCudaEvent`、`torch.cuda.*` 函数 | 所有 CUDA 操作变为空操作 |
| Triton 模块 | `_TritonSubModule` + `_TritonImportFinder` | 仅在 triton 未安装时激活；通过 `sys.meta_path` 拦截所有 `triton.*` 导入 |
| GPU 内核 | `batch_scatter_kv_caches`、`batch_gather_kv_caches` | 替换为空函数，避免 CPU 环境调用 triton 内核 |
| 分布式 | `get_tensor_model_parallel_rank` | 固定返回 0，模拟单机环境 |

#### 调用顺序

`apply_cuda_patches()` 内部按严格顺序执行：

1. `_mock_triton()` — 必须最先执行，因为 `import torch` 可能触发 `torch._inductor` 加载 triton
2. `_mock_batch_gather_scatter()` — 在 `sys.modules` 中预注册空操作模块
3. `import torch` + 替换 `torch.cuda.*`

### 2. vllm_connector_cases.py - 测试基类

`VllmConnectorTestBase` 提供：

- **服务生命周期管理**：自动启动/停止 KV Cache Manager
- **配置生成**：创建测试用的 VllmConfig 和 extra_config，支持多 instance 场景
- **Mock 对象工厂**：创建 Request、KVCacheBlocks、SchedulerOutput 等
- **引擎步骤模拟**：`_simulate_engine_step()` 封装完整的 vllm 引擎单步调度

#### 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `instance_group` | `"default"` | 使用 KVCM 默认创建的 instance group |
| `instance_id` | `"test_instance"` | 测试实例 ID，可通过 `_create_test_vllm_config(instance_id=...)` 覆盖 |
| `block_size` | `16` | KV Cache block 大小 |
| `num_layers` | `4` | 模型层数 |
| `num_kv_heads` | `8` | KV head 数量 |
| `head_size` | `64` | Head 维度 |
| `async_get_cache_location` | `False` | 同步查询以简化测试 |

#### 工具方法

| 方法 | 说明 |
|------|------|
| `_create_test_vllm_config(instance_id, coordinator_base_port)` | 创建 Mock VllmConfig，支持指定 instance_id 和 coordinator 端口 |
| `_create_mock_request(request_id, prompt_token_ids)` | 创建 Mock vLLM Request 对象 |
| `_create_mock_scheduler_output(new_reqs, num_scheduled_tokens)` | 创建 Mock SchedulerOutput |
| `_create_mock_kv_caches(num_layers, num_blocks, ...)` | 创建 CPU 上的 Mock KV Cache 张量 |
| `_create_mock_kv_cache_blocks(block_ids)` | 创建 Mock KVCacheBlocks |
| `_create_mock_scheduled_new_req(req_id, token_ids, block_ids)` | 创建 Mock scheduled_new_req 元素 |
| `_simulate_engine_step(scheduler, worker, scheduler_output)` | 模拟完整的 vllm 引擎步骤（含 `get_block_ids_with_load_errors` 和 `clear_connector_metadata`） |
| `_poll_engine_until_save_collected(scheduler, worker, expected_count, ...)` | 轮询引擎步骤直到收集到预期数量的 SaveRequest（替代 `time.sleep`） |
| `_poll_until_cache_queryable(scheduler, request, ...)` | 轮询 `get_num_new_matched_tokens` 直到 Manager 中缓存数据可查询（替代 `time.sleep`） |
| `_get_free_port()` | 获取空闲端口用于 coordinator |

### 3. 测试用例

#### 测试类概览

| 测试类 | 覆盖范围 |
|--------|----------|
| `ConnectorSchedulerInitTest` | Scheduler 角色初始化、manager client、缓存查询、元数据构建 |
| `ConnectorWorkerInitTest` | Worker 角色初始化、transfer client、KV Cache 注册、元数据绑定 |
| `ConnectorLifecycleTest` | Scheduler-Worker 通信、新请求匹配流程 |
| `ConnectorKVCacheWriteReadTest` | 完整 KV Cache 读写流程、多 instance 协作、多请求并发生命周期 |

#### ConnectorKVCacheWriteReadTest 详解

该测试类创建两套独立的 Scheduler+Worker Connector 对（A 和 B），共享同一个 `instance_id`，使用不同的 coordinator 端口，模拟同一 instance group 内的两个引擎进程。

> **注意**：MetaIndexer 按 `instance_id` 隔离，不支持跨 instance 查询。写入和读取匹配必须在同一个 `instance_id` 上进行。两套 Connector 对共享 instance_id，因此 Manager 侧使用同一个 MetaSearcher。

**test_write_then_read_kvcache — 两套 Connector 对写入后读取**：

```
Connector 对 A (scheduler_a + worker_a)
─────────────────
get_num_new_matched_tokens → 0 (首次无缓存)
update_state_after_alloc
engine_step → start_save_kvcache_async (异步)
_poll_engine_until_save_collected (轮询收集 SaveRequest)
第二轮 engine_step (decode token, 处理 cached req)
request_finished (完成写请求)
engine_step → 推送 FinishRequest

Connector 对 B (scheduler_b + worker_b, 同一 instance_id):
_poll_until_cache_queryable → matched > 0 (轮询直到数据可查)
update_state_after_alloc
engine_step (含 LoadRequest)
request_finished
```

**test_single_instance_write_and_query — 单 instance 写后查询**：
- 同一个 instance 先写入再用新 request_id 查询，验证 getCacheLocation 接口
- 使用 `_poll_engine_until_save_collected` 等待异步保存收集
- 使用 `_poll_until_cache_queryable` 等待 Manager 数据可查

**test_multiple_requests_lifecycle — 多请求并发**：
- 3 个请求同时创建、分配、执行引擎步骤
- 验证请求生命周期通过公开 API 的正确管理
- 使用 `_poll_engine_until_save_collected` 等待 3 个异步保存收集完成
- 通过 `request_finished()` 返回值验证请求清理（而非直接访问内部状态）

## 运行测试

### 运行所有测试

```bash
bazel test //integration_test/vllm_connector:connector_lifecycle_test
```

### 运行特定测试类

```bash
bazel test //integration_test/vllm_connector:connector_lifecycle_test \
    --test_filter="ConnectorKVCacheWriteReadTest"
```

### 运行单个测试方法

```bash
bazel test //integration_test/vllm_connector:connector_lifecycle_test \
    --test_filter="test_write_then_read_kvcache"
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

### 步骤 1：创建测试类

在 `connector_lifecycle_test.py` 中添加新的测试类，或创建新文件：

```python
class MyFeatureTest(VllmConnectorTestBase):
    def _init_connector(self):
        """初始化 Connector"""
        from kv_cache_manager.py_connector.vllm.v1_connector import TairKvCacheConnector

        config = self._create_test_vllm_config()

        self.scheduler = TairKvCacheConnector(config, KVConnectorRole.SCHEDULER)
        self.worker = TairKvCacheConnector(config, KVConnectorRole.WORKER)

        kv_caches = self._create_mock_kv_caches()
        self.worker.register_kv_caches(kv_caches)

    def _cleanup_connector(self):
        """清理 Connector"""
        if hasattr(self, 'worker') and self.worker:
            self.worker.shutdown()
        if hasattr(self, 'scheduler') and self.scheduler:
            self.scheduler.shutdown()

    def test_my_feature(self):
        """测试用例"""
        request = self._create_mock_request(
            request_id="test_req",
            prompt_token_ids=list(range(1, 65)),
        )

        matched, has_match = self.scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(matched, 0)
```

**多 instance 测试**：使用 `instance_id` 和 `coordinator_base_port` 参数创建不同的 Connector：

```python
config_a = self._create_test_vllm_config(
    instance_id="instance_a",
    coordinator_base_port=self._get_free_port(),
)
config_b = self._create_test_vllm_config(
    instance_id="instance_b",
    coordinator_base_port=self._get_free_port(),
)
```

**模拟引擎步骤**：使用 `_simulate_engine_step()` 执行完整的引擎调度循环：

```python
new_req = self._create_mock_scheduled_new_req(req_id, token_ids, block_ids)
sched_out = self._create_mock_scheduler_output(
    new_reqs=[new_req],
    num_scheduled_tokens={req_id: num_tokens},
)
meta, finished_saving, finished_loading = self._simulate_engine_step(
    self.scheduler, self.worker, sched_out,
)
```

### 步骤 2：更新 BUILD 文件（如新建文件）

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

## 异步保存流程与测试要点

Connector 的保存流程是异步的，理解这个流程对编写正确的测试至关重要。

### 异步保存时序

```
build_connector_meta()
  ├─ 遍历 alive_requests
  ├─ 计算 target_save_num
  ├─ 如果有新的 block 需要保存：
  │   ├─ scheduled_saving_count += 1
  │   └─ http_executor.submit(start_save_kvcache_async)  ←── 异步提交
  ├─ 收集 _waiting_to_save_requests（上一轮异步保存的结果）
  │   └─ sent_saving_count += 1
  └─ 返回 meta
```

### 关键：`scheduled_saving_count` 与 `sent_saving_count` 的关系

`request_finished()` 检查 `scheduled_saving_count == sent_saving_count`：
- **相等**：所有保存已完成，立即移除请求
- **不相等**：设置 `need_report_after_saving_finished = True`，延迟移除

因此测试中需要：

1. **轮询等待异步 HTTP 完成**：使用 `_poll_engine_until_save_collected` 反复执行引擎步骤，直到 `build_connector_meta` 收集到预期数量的 SaveRequest
2. **之后才能 request_finished**：此时 `scheduled_saving_count == sent_saving_count`，请求可以被正常清理

```python
# 第一轮：触发保存
self._simulate_engine_step(scheduler, worker, sched_out)

# 轮询收集异步保存结果（替代 time.sleep）
self._poll_engine_until_save_collected(scheduler, worker, expected_count=1)

# 现在可以安全完成请求
scheduler.request_finished(request, block_ids)
```

如果需要等待完整保存管道完成后验证 Manager 数据可查，使用 `_poll_until_cache_queryable`：

```python
# 轮询直到 Manager 中数据可查（替代 time.sleep(5)）
matched, has_match = self._poll_until_cache_queryable(scheduler, probe_request)
```

### Worker 侧清理的局限

在无存储后端的测试环境中，Worker 侧的 `get_finished()` 通过 coordinator server 查询保存完成状态。由于没有真实的数据传输，coordinator 永远不会收到完成事件，因此 Worker 侧的请求可能不会被完全清理。测试断言应以 Scheduler 侧的公开 API 返回值为主，不直接访问内部状态。

## 注意事项

### 1. Mock 初始化顺序

`apply_cuda_patches()` **必须**在 `import vllm` 之前调用。原因链路：

```
import vllm
  → vllm.env_override
    → torch._inductor
      → import triton.backends.compiler  （需要 triton mock）
      → triton.__version__               （需要返回版本字符串）
      → inspect.signature(triton.language.core.view)  （需要可调用）
```

`vllm_connector_cases.py` 在模块加载时已处理此顺序：
```python
apply_cuda_patches()      # 行 37
apply_distributed_patches()  # 行 38
# 之后 connector_lifecycle_test.py 才 import vllm
```

### 2. 服务启动等待

KV Cache Manager 服务启动需要时间，`TestBase` 已处理等待逻辑。如果遇到连接问题，检查：
- 端口是否被占用
- 服务是否正常启动（查看 worker 目录下的 `stderr` 和 `kv_cache_manager.log`）

### 3. Instance Group

测试默认使用 `"default"` instance group，这是 KVCM 启动时自动创建的。如需使用其他 group，需要先通过 Admin API 创建。

### 4. 异步查询模式

`async_get_cache_location` 配置控制缓存位置查询模式：
- `False`（推荐用于测试）：同步查询，`get_num_new_matched_tokens` 立即返回结果
- `True`：异步查询，首次调用返回 `(None, False)`，需多次调用等待结果

### 5. Mock VllmConfig

测试使用 MagicMock 创建 VllmConfig，避免访问 HuggingFace。如需修改模型参数，通过 `_create_test_vllm_config()` 的 `TestConnectorConfig` 调整：

```python
self._test_config.num_layers = 12
self._test_config.num_kv_heads = 8
self._test_config.head_size = 64
```

或直接传入参数覆盖：
```python
config = self._create_test_vllm_config(
    instance_id="custom_instance",
    coordinator_base_port=50001,
)
```

### 6. 端口冲突

测试框架自动分配端口，但并行运行多个测试可能导致端口冲突。`TestBase` 使用端口范围哈希来减少冲突概率。`_get_free_port()` 使用 bind-then-release 方式获取空闲端口，存在 TOCTOU 窗口。

### 7. 资源清理

测试结束后会自动清理资源。手动清理方法见[开发文档的资源清理章节](README.md#资源清理)。

### 8. CUDA Mock 局限性

Mock 仅模拟基本的 Stream/Event 行为，不支持：
- 真实的 GPU 内存分配
- CUDA 计算操作
- 多 GPU 场景

`batch_gather_scatter_helper` 中的 GPU 内核被替换为空操作，因此 Worker 侧的实际数据传输（gather/scatter）不会执行。如需测试这些功能，需要在有 GPU 的环境中运行。

## 调试与日志

通用的调试方法（Manager Server 日志、TransferClient 日志、排查流程、资源清理）请参见[开发文档的调试章节](README.md#调试)。以下仅记录 vLLM Connector 集成测试的特定调试技巧。

### vLLM Connector 特定的日志查找

**快速查找 Manager 日志**：

```bash
# 找到 vllm_connector 测试的所有 Manager 日志
find ~/.cache/bazel -name "kv_cache_manager.log" -path "*vllm_connector*"

# 查看特定测试的 Manager 日志
find ~/.cache/bazel -path "*test_write_then_read_kvcache*/kv_cache_manager.log" | xargs cat
```

**启用 TransferClient DEBUG 日志**：

```bash
bazel test //integration_test/vllm_connector:connector_lifecycle_test \
    --test_env=KVCM_LOG_LEVEL=DEBUG
```

### Python Connector 日志

Connector 的 Python 日志通过 `kv_cache_manager/py_connector/common/logger.py` 配置，输出到 stderr（被 bazel 捕获到 test.log）。

**日志位置**：

```bash
# bazel test 的标准输出，包含所有 [KVCM] 前缀的日志
cat ~/.cache/bazel/.../testlogs/integration_test/vllm_connector/connector_lifecycle_test/test.log
```

**日志级别控制**：修改 `kv_cache_manager/py_connector/common/logger.py` 中的 `setLevel()`：

```python
# 默认 INFO，改为 DEBUG 可看到详细的 save/load 流程
logger.setLevel(logging.DEBUG)
handler.setLevel(logging.DEBUG)
```

**关键日志模式**：

```bash
# 查看数据传输、finish_write_cache、匹配结果
grep -E "start transfer|done transfer|finish_write_cache|matched_count|save task failed|ER_" test.log
```

### vLLM Connector 排查流程

典型的写入-查询问题排查顺序（通用排查流程见[开发文档](README.md#排查流程建议)）：

1. **Python Connector 日志**：确认 `start_write_cache` 返回了 locations、`save_task` 是否成功（`ER_OK` vs `ER_SDKALLOC_ERROR`）、`finish_write_cache` 是否报 Connection refused
2. **TransferClient 日志**：如果 `save_task` 失败，查看 `kv_cache_manager_client.log` 中的 `Alloc failed` 或 `DoPut` 错误
3. **Manager 日志**：确认 `FinishWriteCache` 是否被处理、`GetCacheLocation` 返回了多少 locations、`PrefixMatchBestLocationImpl` 的 return code（2 = key not found）

## 已知限制

### 测试性能

每个测试方法都会重启一次 KV Cache Manager 服务（约 2-3 秒），15 个测试总计约 40 秒。如需优化，可考虑在同一 TestClass 内通过 `setUpClass` 复用 Manager 实例（需解决测试间状态隔离）。

## 常见问题
### Q: `request_finished` 后请求仍未清理

A: 异步保存尚未完成（`scheduled_saving_count != sent_saving_count`）。使用 `_poll_engine_until_save_collected` 轮询引擎步骤直到异步保存结果被收集，然后再调用 `request_finished`。参见「异步保存流程与测试要点」章节。

### Q: 导入报错 `ModuleNotFoundError: No module named 'triton.xxx'`

A: 检查 `apply_cuda_patches()` 是否在 `import vllm` **之前**被调用。如果已安装 triton 但版本不兼容，可尝试 `pip install triton>=3.3.0`。

### Q: Worker 侧请求未被清理

A: 这是无存储后端测试环境的预期行为。Worker 的 `get_finished()` 依赖 coordinator server 报告保存/加载完成，而无真实数据传输时 coordinator 不会收到完成事件。测试断言应以 Scheduler 侧的公开 API 返回值为主。

## 参考资料

- [vLLM KVConnector API](https://docs.vllm.ai/)
- [Bazel Python 测试文档](https://bazel.build/reference/be/python#py_test)
- [integration_test/meta_service](../integration_test/meta_service/) - Meta Service 集成测试参考
