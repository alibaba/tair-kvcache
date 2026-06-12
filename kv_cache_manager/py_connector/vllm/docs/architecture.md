# TairKvCache vLLM Connector 架构文档

## 1. 概述

本模块通过实现 vLLM 的 **KV Connector V1** 接口，将 Tair KVCache Manager 作为外部 KV 缓存后端接入 vLLM，实现跨实例的 KV Cache 复用（Prefix Caching）。

核心能力：
- **Load**：当 vLLM 收到推理请求时，查询 Manager 是否存在可复用的前缀 KV Cache，若存在则从远程存储加载到本地 GPU 显存
- **Save**：推理过程中，将新生成的 KV Cache 异步写入远程存储，供后续请求复用
- **TP 协调**：在 Tensor Parallel 场景下，确保所有 rank 的 Save/Load 均完成后才向上层汇报

### 1.1 兼容的 vLLM 版本

- vLLM >= v0.11.0（通过 `inspect.signature` 和 `hasattr` 做了 v0.11.0 / v0.11.1 的兼容适配）

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          vLLM Engine                                    │
│  ┌──────────────────┐         ┌───────────────────────────────────┐    │
│  │   Scheduler       │         │   Worker 0 (TP rank 0)           │    │
│  │                   │  meta   │  ┌─────────────────────────────┐ │    │
│  │  TairKvCache      │ ──────→ │  │ TairKvCacheConnector        │ │    │
│  │  Connector        │         │  │ (role=WORKER)               │ │    │
│  │  (role=SCHEDULER) │         │  │                             │ │    │
│  │                   │         │  │ TpCoordinatorServer (ZMQ)   │ │    │
│  │  LocationQuery    │         │  │ DataTransferManager          │ │    │
│  │  Manager          │         │  │ TransferClient (C++ SDK)     │ │    │
│  └────────┬──────────┘         │  └──────────────┬──────────────┘ │    │
│           │                    └─────────────────┼────────────────┘    │
│           │ HTTP API                             │ ZMQ PUSH            │
│           │                    ┌─────────────────┼────────────────┐    │
│  ┌────────▼──────────┐         │   Worker 1 (TP rank 1)           │    │
│  │ Tair KVCache      │         │  ┌─────────────────────────────┐ │    │
│  │ Manager Service   │         │  │ TairKvCacheConnector        │ │    │
│  │ (registerInstance │         │  │ (role=WORKER)               │ │    │
│  │  getCacheLocation │         │  │ DataTransferManager          │ │    │
│  │  startWriteCache  │         │  │ TransferClient (C++ SDK)     │ │    │
│  │  finishWriteCache)│         │  └──────────────┬──────────────┘ │    │
│  └───────────────────┘         └─────────────────┼────────────────┘    │
│                                                   │                     │
│                                    ┌──────────────▼──────────────┐      │
│                                    │   Remote Storage (HF3FS)    │      │
│                                    └─────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 进程模型

vLLM V1 引擎采用 Scheduler-Worker 分离架构，Connector 在每个进程中各有一个实例，但职责不同：

| 进程 | 角色 | 核心职责 |
|------|------|----------|
| Scheduler 进程 | `KVConnectorRole.SCHEDULER` | 与 Manager HTTP 通信（查询缓存位置、申请写入、汇报完成）；构建 Metadata 传递给 Worker |
| Worker 进程 (rank 0) | `KVConnectorRole.WORKER` | 执行实际数据传输；运行 `TpCoordinatorServer`（ZMQ PULL）收集所有 rank 的完成事件 |
| Worker 进程 (rank N) | `KVConnectorRole.WORKER` | 执行实际数据传输；通过 `TpCoordinatorClient`（ZMQ PUSH）向 rank 0 报告完成 |

---

## 4. 文件与类说明

### 4.1 核心文件

| 文件 | 核心类 | 职责 |
|------|--------|------|
| `v1_connector.py` | `TairKvCacheConnector` | 主入口，继承 `KVConnectorBase_V1`，实现 vLLM 要求的所有 Scheduler/Worker 方法 |
| `v1_connector.py` | `ReqState` | 跟踪单个推理请求在 Connector 内部的状态（token_ids, block_ids, 保存进度等） |
| `config.py` | `TairKvCacheConnectorExtraConfig` | 从 `kv_connector_extra_config` 解析所有配置参数 |
| `metadata.py` | `TairKvCacheConnectorMetadata` | Scheduler→Worker 的元数据包，每个调度周期构建一次 |
| `metadata.py` | `SaveRequest`, `LoadRequest`, `FinishRequest`, `ReqStateToWorker` | Metadata 内部使用的数据结构 |
| `data_transfer.py` | `DataTransferManager` | 数据传输核心，负责 GPU↔CPU 的 gather/scatter 和调用 TransferClient 进行远程 IO |
| `data_transfer.py` | `MultiResult` | 分片任务聚合器，所有分片完成后触发回调 |
| `location_query_manager.py` | `LocationQueryManager` | 管理向 Manager 查询缓存位置的请求，支持异步查询和本地缓存 |

### 4.2 Common 模块（`py_connector/common/`）

| 文件 | 核心类 | 职责 |
|------|--------|------|
| `manager_client.py` | `KvCacheManagerClient` | Tair KVCache Manager 的 HTTP 客户端，封装所有 REST API 调用，支持 Leader 自动发现 |
| `tp_coordinator.py` | `TpCoordinatorServer` | ZMQ PULL Server（仅 rank 0），收集所有 rank 的完成事件并聚合 |
| `tp_coordinator.py` | `TpCoordinatorClient` | ZMQ PUSH Client（所有 rank），向 Server 发送事件消息 |
| `tp_coordinator.py` | `CoordinateMessage` 及事件类型 | ZMQ 消息的数据结构和序列化 |
| `types.py` | `KVCacheInfo` | GPU KV Cache 元信息的冻结数据类 |

### 4.3 Kernel 模块（`py_connector/kernel/`）

| 文件 | 核心函数 | 职责 |
|------|----------|------|
| `batch_gather_scatter_helper.py` | `batch_gather_kv_caches()` | Triton kernel：从 GPU 各层 KV Cache 批量 gather 到 CPU pinned memory（连续排列） |
| `batch_gather_scatter_helper.py` | `batch_scatter_kv_caches()` | Triton kernel：从 CPU pinned memory scatter 回 GPU 各层 KV Cache |
| `gather_scatter_helper.py` | `CopyBufferAllocator` | CPU pinned memory 池管理器，为 gather/scatter 提供预分配的缓冲区 |

---

## 5. 与 Tair KVCache Manager 的对接

### 5.1 Manager HTTP API

Connector 通过 `KvCacheManagerClient` 与 Manager 交互，使用以下 API：

| API | 调用方 | 用途 |
|-----|--------|------|
| `/api/registerInstance` | 初始化时（Scheduler + Worker） | 注册实例，上报模型部署信息和 location spec（每个 TP rank 的形状、大小） |
| `/api/getCacheLocation` | Scheduler（`LocationQueryManager`） | 前缀匹配查询：给定 token_ids，Manager 返回远程存储中已有的 KV Cache 位置 |
| `/api/startWriteCache` | Scheduler（`start_save_kvcache_async`） | 申请写入：Manager 返回写入目标位置和 write_session_id |
| `/api/finishWriteCache` | Worker rank 0（`on_save_finished`） | 汇报写入结果：告知 Manager 哪些 block 写成功、哪些失败 |

### 5.2 数据传输 SDK

Worker 进程通过 `kvcm_py_client.TransferClient`（C++ pybind 绑定）与远程存储交互：

| 方法 | 方向 | 底层实现 |
|------|------|----------|
| `TransferClient.LoadKvCaches(uris, buffers)` | 远程存储 → CPU buffer | `SdkWrapper::Get()`，走 HF3FS 等分布式文件系统 |
| `TransferClient.SaveKvCaches(uris, buffers)` | CPU buffer → 远程存储 | `SdkWrapper::Put()`，走 HF3FS 等分布式文件系统 |

### 5.3 Instance 注册流程

```
Connector.__init__()
  │
  ├── manager_client.register_instance({
  │       instance_group, instance_id,
  │       model_deployment: {model_name, dtype, use_mla, tp_size, dp_size, pp_size},
  │       block_size,
  │       location_spec_infos: [
  │         {name: "tp0", size: ...},
  │         {name: "tp1", size: ...},
  │         ...
  │       ]
  │   })
  │   └── 返回: storage_configs（存储后端配置，如 HF3FS mountpoint 等）
  │
  └── [仅 Worker] TransferClient.Create(transfer_client_config, init_params)
      └── 初始化 SDK，准备 IOV（IO Vector）缓冲区
```

`location_spec_infos` 告诉 Manager 每个 TP rank 需要存储的数据形状。形状计算：

```
per_tp_rank_shape = [num_layers, 1 if MLA else 2, manager_block_size, kv_head_num_per_tp, head_size]
per_spec_byte_size = prod(per_tp_rank_shape) * dtype.itemsize
```

---

## 6. 数据流详解

### 6.1 Load 流程（远程 KV Cache → GPU）

```
时间线: Scheduler 调度周期 N → Worker 执行

═══ Scheduler ═══

1. get_num_new_matched_tokens(request, num_computed_tokens)
   │
   ├── computed_manager_block_size = num_computed_tokens // manager_block_size
   │
   ├── location_query_manager.get_locations_for_query(request, computed_manager_block_size)
   │   │
   │   ├── [异步模式] 提交 HTTP 请求到线程池，返回 (False, [])
   │   │   └── 下一次调度周期再次调用时，结果已就绪，返回 (True, locations)
   │   │
   │   └── [同步模式] 阻塞等待 HTTP 响应，返回 (True, locations)
   │
   └── 创建 LoadRequest，加入 _waiting_to_load_requests

2. update_state_after_alloc(request, blocks)
   │
   └── 记录 vLLM 分配的 local_block_ids

3. build_connector_meta(scheduler_output)
   │
   ├── 将 LoadRequest（附带 local_block_ids）打包进 meta.to_load_requests
   └── 返回 TairKvCacheConnectorMetadata

═══ Worker (每个 rank 独立执行) ═══

4. bind_connector_metadata(meta)
   └── 同步 _alive_requests 状态

5. start_load_kv(forward_context)
   │
   ├── 对每个 LoadRequest:
   │   ├── generate_blocks_idx()          → 计算每个 manager block 对应的 GPU token 索引
   │   ├── get_self_uris()                → 提取当前 rank 的远程 URI
   │   ├── create_load_done_callback()    → 创建回调（完成后发 ZMQ 消息）
   │   │
   │   └── 分片提交到 io_executor 线程池:
   │       └── data_transfer.load_task(multi_result, task_idx, uris, indices)
   │
   └── [io_executor 线程中]

6. DataTransferManager.load_task()
   │
   ├── copy_buffer_allocator.alloc()          ← 从池分配 CPU pinned buffer
   ├── transfer_client.LoadKvCaches(uris, buffers)  ← C++ SDK: 远程存储 → CPU
   ├── batch_scatter_kv_caches(...)           ← Triton kernel: CPU → GPU HBM
   ├── copy_buffer_allocator.free()
   │
   └── multi_result.submit_result(task_idx, result)
       │
       └── [所有分片完成] → done_callback
           │
           └── coordinator_client.send(LoadBlockFinishedEvent)

═══ TpCoordinatorServer (rank 0 Worker) ═══

7. coordinator_routine()
   └── 收到所有 rank 的 LoadBlockFinishedEvent 后:
       ├── _finished_loading.append(request_id)
       └── _failed_loading_block_idxs.extend(failed_block_idxs)

═══ Worker rank 0 下一个调度周期 ═══

8. get_finished()
   └── coordinator_server.get_finished_tasks() → 返回给 vLLM
```

### 6.2 Save 流程（GPU → 远程 KV Cache）

```
═══ Scheduler ═══

1. build_connector_meta() 中判断: 哪些请求有新的 block 需要保存
   │
   └── http_executor.submit(start_save_kvcache_async, req_id, token_ids, target_save_num)

2. start_save_kvcache_async() [http_executor 线程]
   │
   ├── manager_client.start_write_cache({token_ids, instance_id, ...})
   │   └── 返回: {locations, write_session_id, block_mask}
   │
   ├── parse_block_mask_to_save_indices()  → 确定需要写入的 block 索引
   │
   ├── coordinator_client.send(SendBlockStartEvent)  → 通知 Server 新任务
   │
   └── SaveRequest 加入 _waiting_to_save_requests

3. 下一轮 build_connector_meta()
   └── 将 SaveRequest 打包进 meta.to_save_requests

═══ Worker (每个 rank 独立执行) ═══

4. wait_for_save()
   │
   ├── 记录 kvcache_ready_event (GPU event)
   │
   └── 对每个 SaveRequest:
       ├── generate_blocks_idx()
       ├── get_self_uris()
       ├── create_save_done_callback()
       │
       └── 分片提交:
           └── data_transfer.save_task(multi_result, task_idx, uris, indices, kvcache_ready_event)

5. DataTransferManager.save_task() [io_executor 线程]
   │
   ├── kvcache_ready_event.wait()           ← 等 GPU 计算完
   ├── batch_gather_kv_caches(...)          ← Triton kernel: GPU HBM → CPU pinned
   ├── copy_done_event.synchronize()
   ├── transfer_client.SaveKvCaches(uris, buffers)  ← C++ SDK: CPU → 远程存储
   ├── copy_buffer_allocator.free()
   │
   └── multi_result.submit_result()
       └── [所有分片完成] → done_callback
           └── coordinator_client.send(SendBlockFinishedEvent)

═══ TpCoordinatorServer (rank 0) ═══

6. coordinator_routine()
   └── 收到 SendBlockStartEvent → 创建 SaveContext
   └── 收到所有 rank 的 SendBlockFinishedEvent 后:
       ├── on_save_finished(write_session_id, save_context)
       │   ├── 汇总各 rank 的 is_success → 生成 success_mask
       │   └── manager_client.finish_write_cache({success_blocks, write_session_id})
       │
       └── _finished_saving.append(request_id)

═══ Worker rank 0 ═══

7. get_finished()
   └── coordinator_server.get_finished_tasks() → 返回给 vLLM
```

---

## 7. TP 协调机制 (TpCoordinator)

### 7.1 为什么需要协调

在 TP 场景下，同一个请求的 KV Cache 被切分到多个 GPU。每个 rank 独立执行 Save/Load，但 Manager 需要知道"整个请求"是否全部完成。Coordinator 负责跨 rank 的完成状态聚合。

### 7.2 通信模型

```
Scheduler          Worker rank 0        Worker rank 1       Worker rank N-1
  │                    │                     │                    │
  │ coord_client       │ coord_client        │ coord_client       │ coord_client
  │ (ZMQ PUSH)        │ (ZMQ PUSH)          │ (ZMQ PUSH)         │ (ZMQ PUSH)
  │                    │                     │                    │
  └────────────────────┼─────────────────────┼────────────────────┘
                       ▼                     ▼
              ┌──────────────────────────────────────┐
              │ TpCoordinatorServer (仅 rank 0)      │
              │ ZMQ PULL socket                      │
              │                                      │
              │ 对每个 (request_id, write_session_id)│
              │ 维护计数: collected_ranks              │
              │ 当 |collected_ranks| == tp_world_size │
              │   → 触发回调 / 标记完成               │
              └──────────────────────────────────────┘
```

### 7.3 消息类型

| 事件 | 发送方 | 触发时机 | Server 处理 |
|------|--------|----------|-------------|
| `SendBlockStartEvent` | Scheduler | Manager 返回写入位置后 | 创建 `SaveContext` 开始跟踪 |
| `SendBlockFinishedEvent` | 每个 rank Worker | 该 rank 数据传输完成 | `add_new_rank()`，集齐 tp_size 个后触发 `on_save_finished` 回调 |
| `LoadBlockFinishedEvent` | 每个 rank Worker | 该 rank 数据加载完成 | `add_new_rank()`，集齐后标记完成并记录失败 block |

### 7.4 两层聚合

```
单个 rank 内部:
  一次传输有 N 个分片（每 block_per_save/load_task 个 block 一个分片）
  MultiResult 等所有分片完成 → 触发 done_callback → 发 ZMQ 消息

跨 rank (Coordinator):
  一次传输有 tp_size 个 rank
  CoordinatorServer 等所有 rank → 触发 on_save_finished / 标记完成
```

---

## 8. GPU 数据搬运 (Triton Kernel)

### 8.1 问题

vLLM 的 KV Cache 是**每层一个独立 tensor**，在 GPU 显存中地址不连续。而远程存储 SDK 只能操作 CPU 内存，且要求数据连续排列（格式: `[num_blocks, num_layers × 2, tokens_per_block, hidden_dim]`）。

### 8.2 解决方案

使用 Triton JIT 编译的 GPU kernel 实现高效的跨 PCIe 批量 gather/scatter：

- **`batch_gather_kv_caches`** (Save): 从 GPU 64 个不连续 tensor → CPU pinned memory 连续 buffer
- **`batch_scatter_kv_caches`** (Load): 从 CPU pinned memory 连续 buffer → GPU 64 个不连续 tensor

关键优化：
- **单次 kernel launch**: 将所有层的操作合并到一个 kernel，避免 64 次 launch 开销
- **限制 SM 数量** (`sm_count=3`): 只使用少量 SM 处理搬运，把其余 SM 留给推理计算
- **Grid-stride loop**: 少量 SM 通过循环处理所有 block
- **PCIe 友好访问模式**: host memory 侧按大块连续读写，最大化 PCIe 带宽
- **指针间接寻址**: 通过指针数组动态定位各层 KV Cache 基址

### 8.3 CopyBufferAllocator

预分配一个 CPU pinned memory 池（默认 1024 个 slot），提供线程安全的阻塞式分配和释放，避免每次传输都 `malloc`。

---

## 9. 异步缓存位置查询 (LocationQueryManager)

### 9.1 问题

`getCacheLocation` 是一次 HTTP 调用，在调度器的关键路径上阻塞会严重影响吞吐量。

### 9.2 解决方案

- **异步查询**: 第一次调用时提交 HTTP 请求到线程池，立即返回 `(False, [])`，告知 vLLM "查询中，暂不调度"
- **本地缓存**: 查询结果缓存 1 秒（TTL），下次调度周期直接取结果
- **自动清理**: 后台线程定期清理过期缓存条目

```
调度周期 1: get_num_new_matched_tokens()
  → query_manager 发起异步 HTTP 请求
  → 返回 (None, False) → vLLM 不调度此请求

调度周期 2: get_num_new_matched_tokens()
  → query_manager 发现结果已就绪
  → 返回 (matched_count, True) → vLLM 正常调度
```

---

## 10. Metadata 传递机制

Scheduler 和 Worker 之间的所有状态同步通过 `TairKvCacheConnectorMetadata` 完成。vLLM 会在每个调度周期：

1. 调用 Scheduler 侧的 `build_connector_meta()` 构建 metadata
2. 自动序列化并传输到各 Worker 进程
3. 调用 Worker 侧的 `bind_connector_metadata()` 反序列化

Metadata 包含：

| 字段 | 内容 |
|------|------|
| `epoch` | 当前调度周期编号 |
| `requests` | `ReqStateToWorker` 列表：同步 token_ids、block_ids 增量到 Worker |
| `to_load_requests` | `LoadRequest` 列表：需要从远程加载的 block 信息 |
| `to_save_requests` | `SaveRequest` 列表：需要写入远程存储的 block 信息 |
| `to_finish_requests` | `FinishRequest` 列表：通知 Worker 清理已结束的请求 |

---

## 11. 请求生命周期

```
┌──────────────────────────────────────────────────────────────────┐
│                     Scheduler 进程                                │
│                                                                  │
│  get_num_new_matched_tokens()                                    │
│    → 查询远程缓存，创建 ReqState                                  │
│                                                                  │
│  update_state_after_alloc()                                      │
│    → 记录 vLLM 分配的物理 block IDs                               │
│                                                                  │
│  build_connector_meta()          ←── 每个调度周期调用一次          │
│    → 打包 LoadRequest / SaveRequest / FinishRequest              │
│    → 异步发起 startWriteCache                                     │
│                                                                  │
│  request_finished()                                              │
│    → 请求结束，等待未完成的 Save 全部完成后清理 ReqState            │
│                                                                  │
│  get_finished()                  ←── Worker rank 0               │
│    → 从 CoordinatorServer 获取已完成的 Save/Load                  │
│    → 返回给 vLLM                                                 │
└──────────────────────────────────────────────────────────────────┘
```

一个请求在 `_alive_requests` 中的状态流转：

```
创建 (get_num_new_matched_tokens)
  → 分配 block (update_state_after_alloc)
  → 调度执行 (build_connector_meta)
  → [多轮: 生成新 token → 异步 Save]
  → 请求结束 (request_finished)
     → 若所有 Save 已完成: 立即清理
     → 若仍有 Save 在途: 标记 need_report_after_saving_finished，等 Save 回调后清理
```

---

## 12. 配置参数

通过 vLLM 的 `kv_connector_extra_config` dict 传入，由 `TairKvCacheConnectorExtraConfig` 解析：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `manager_uri` | (必填) | Manager HTTP 服务地址 |
| `instance_id` | (必填) | 实例 ID，KV Cache 仅在相同 instance_id 内复用 |
| `instance_group` | (必填) | 实例组名称 |
| `coordinator_base_port` | (必填) | TpCoordinator ZMQ 端口 |
| `preferred_block_size` | 0 (使用 vLLM block_size) | Manager 侧的 block size |
| `sdk_thread_num` | 32 | TransferClient SDK 线程数 |
| `sdk_queue_size` | 1000 | TransferClient 任务队列大小 |
| `sdk_get_timeout_ms` | 5000 | Load 超时时间 (ms) |
| `sdk_put_timeout_ms` | 10000 | Save 超时时间 (ms) |
| `block_per_save_task` | 128 | 每个 Save 分片任务包含的 block 数 |
| `block_per_load_task` | 128 | 每个 Load 分片任务包含的 block 数 |
| `async_get_cache_location` | True | 是否异步查询缓存位置 |
| `write_timeout_seconds` | 30 | 写入超时时间 |
| `read_iov_block_size` | 0 | HF3FS 读 IOV block size |
| `write_iov_block_size` | 0 | HF3FS 写 IOV block size |
| `hf3fs_concurrent_io_block_count` | 32 | HF3FS 并发 IO block 数 |
| `auto_discover_leader` | False | 是否自动发现 Manager Leader |
| `leader_retry_count` | 1 | Leader 切换重试次数 |

---

## 13. 约束与限制

- **Instance 隔离**: KV Cache 仅在同一个 `instance_id` 内复用，跨 Instance 不匹配
- **Pipeline Parallel**: 当前仅支持 `pp_size == 1`（代码中有 assert）
- **MLA 支持**: 通过检测 `model_config.use_mla` 自动切换 KV Cache 形状（MLA 时 K/V 合并为 1 个维度）
- **vLLM 版本兼容**: 通过 `inspect.signature` 和 `try/except ImportError` 兼容 v0.11.0 和 v0.11.1
