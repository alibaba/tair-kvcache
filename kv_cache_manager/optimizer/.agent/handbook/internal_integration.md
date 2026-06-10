# 内部接入

当其他 C++ 或 Python 组件需要直接调用 Optimizer，而不是通过 CLI 启动时，使用这份文档。

## C++ 单 Manager API

头文件：

```cpp
#include "kv_cache_manager/optimizer/manager/optimizer_manager.h"
```

基本流程：

```cpp
kv_cache_manager::OptimizerConfig config = ...;
kv_cache_manager::OptimizerManager manager(config);
if (!manager.Init()) {
    // 处理初始化失败
}

manager.DirectRun();
manager.AnalyzeResults();
```

直接读写调用：

```cpp
manager.GetCacheLocation(
    instance_id,
    trace_id,
    timestamp_ns,
    block_ids,
    block_mask,
    input_len,
    true,
    true,
    "prefix_match");

manager.WriteCacheWithTtlUs(
    instance_id,
    trace_id,
    timestamp_ns,
    block_ids,
    ttl_us);
```

常用管理接口：

- `ClearCache(instance_id)`：清空单个 instance cache，保留统计。
- `ClearAllCaches()`：清空所有 instance cache，保留统计。
- `ClearCacheAndResetStats(instance_id)`：清空单个 instance cache 并重置统计。
- `ExportRadixTrees()`：导出 index 状态用于可视化。
- `TouchCacheKeysAtTier(...)`：在 hierarchical/storage-pool 相关 flow 中 touch 指定 tier。

## C++ Hierarchical API

头文件：

```cpp
#include "kv_cache_manager/optimizer/manager/hierarchical_replay_manager.h"
```

基本流程：

```cpp
kv_cache_manager::HierarchicalReplayConfig config = ...;
kv_cache_manager::HierarchicalReplayManager manager(config);
if (!manager.Init()) {
    // 处理初始化失败
}

manager.DirectRun();
manager.AnalyzeResults();
```

直接调用：

```cpp
manager.GetCacheLocation(
    engine_instance_id,
    trace_id,
    timestamp_ns,
    block_ids,
    input_len,
    "prefix_match");

manager.WriteCacheWithTtlUs(
    engine_instance_id,
    trace_id,
    timestamp_ns,
    block_ids,
    ttl_us);
```

只有语义包含 storage pool、P2P、scheduler、active window 或 cache drop 时，才使用 hierarchical API。

## Python Binding

模块：

```python
import kvcm_py_optimizer
```

典型流程：

```python
loader = kvcm_py_optimizer.OptimizerConfigLoader()
config = loader.load("/path/to/config.json")

manager = kvcm_py_optimizer.OptimizerManager(
    config,
    enable_lifecycle_tracking=False,
    enable_template_analysis=False,
    hit_rate_perspective=kvcm_py_optimizer.HitRatePerspective.KVCM_L3,
)
manager.Init()
manager.DirectRun()
manager.AnalyzeResults()
```

直接调用：

```python
manager.GetCacheLocation(
    "instance-a",
    "trace-1",
    timestamp_ns,
    block_ids,
    [],
    input_len,
    "prefix_match",
)
manager.WriteCache("instance-a", "trace-1", timestamp_ns + 1, block_ids, 0)
```

## 接入规则

- CLI 和嵌入式 API 必须保持相同 trace/config 语义。
- 不要绕过 `input_len`，token hit rate 依赖它。
- 除非显式使用 hierarchical P2P/storage pool，否则保持 instance 隔离。
- 新增 public method 前先写 task 和 plan。
- 如果 API 成为标准对外能力，需要同步更新文档和对应 skill。
