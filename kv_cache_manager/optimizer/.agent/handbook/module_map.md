# Optimizer 模块地图

这份文档是工作手册后的第二层入口，用来把概念映射到代码位置，避免为了找扩展点扫描整个仓库。

## 运行入口

| 入口 | 路径 | 职责 |
|---|---|---|
| `optimizer_main` | `main.cc` | 运行一个标准 optimizer config |
| `hierarchical_replay_main` | `hierarchical_replay_main.cc` | 运行 engine-local + storage-pool 联合回放 |
| `optimizer_run` | `analysis/script` | 单次回放、画图、导出 lifecycle 的 Python 包装入口 |
| `multi_infer_replay` | `analysis/script` | 按 pod 或 infer instance 并行回放并聚合 |
| `tradeoff` | `analysis/script` | 容量 Pareto sweep |

## 核心 C++ 模块

| 模块 | 路径 | 职责 | 扩展点 |
|---|---|---|---|
| 配置 | `config/` | JSON config 结构、枚举解析、校验 | 新字段需要同时补解析、序列化、校验、文档和测试 |
| Trace loader | `trace_loader/` | 标准 trace schema、类型解析、排序加载 | 只有新增标准回放语义时才添加 trace type |
| Manager | `manager/optimizer_manager.*` | 单 optimizer instance-group 仿真 API | 嵌入式 API、直接读写调用、lifecycle/template 开关 |
| Runner | `manager/optimizer_runner.*` | 回放顺序和 request/get/write 执行 | 新回放语义必须先有 schema 和计划文档 |
| Index | `index/` | Radix/hash index、prefix/batch 查询 | 新索引实现或查询行为 |
| 驱逐策略 | `eviction_policy/` | LRU、RandomLRU、LeafAwareLRU、TTL | 通过 policy factory 和 config enum 新增策略 |
| 统计 | `analysis/`、`analysis/tracker/` | 命中率、lifecycle、template-prefix 记录 | 新指标列、导出器、画图脚本 |
| Scheduler | `scheduler/` | infer 调度和 active window 过滤 | 新调度策略或在线窗口行为 |
| P2P | `p2p/` | peer presence tracker 和 peer 选择支持 | 新 peer 选择行为，必须明确作用范围 |
| Storage pool | `storage_pool/` | hash storage pool 仿真 | storage-pool-only 行为 |
| Tier flow | `tier_flow/` | tier 事件记录和层间流动语义 | 只有分析确实需要时才添加事件 reason |
| Python binding | `pybind/` | Python 直接 API | C++ API 稳定后再暴露稳定 Python API |

## 标准数据流

```text
标准 JSONL trace
  -> loader 解析 get/write/request
  -> runner 或 hierarchical manager 按时间顺序执行
  -> index query/write 更新 cache 状态
  -> eviction policy 执行容量约束
  -> stats tracker 记录 token/block 指标
  -> C++/Python 工具输出 CSV 和图
```

Hierarchical replay 会在 trace 加载和 stats 输出之间插入 scheduler、active-window 过滤、P2P 和 storage-pool 读写流。

## 继续阅读

- 配置和 trace 字段：[trace_schema_and_conversion.md](trace_schema_and_conversion.md)、[../../docs/strategy_config.md](../../docs/strategy_config.md)
- 运行流程：[external_usage.md](external_usage.md)
- 接入 API：[internal_integration.md](internal_integration.md)
- 功能扩展规则：[feature_development.md](feature_development.md)
