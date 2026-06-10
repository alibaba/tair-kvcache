# Optimizer 工作手册

这份手册是 Optimizer 工作的渐进式入口。先判断任务类型，确认必须由人判断的参数，再按需深入到具体字段、接口和扩展点。

## 第 0 层：选择任务路径

| 用户意图 | 优先使用的 skill | 继续阅读 |
|---|---|---|
| 跑一个 optimizer 配置并查看命中率 | [run-single-replay](../skills/run-single-replay/SKILL.md) | [external_usage.md](external_usage.md) |
| 按 pod 或推理实例独立回放并聚合 | [run-multi-infer-replay](../skills/run-multi-infer-replay/SKILL.md) | [external_usage.md](external_usage.md) |
| 联合模拟 engine 本地 cache、storage pool、P2P | [run-hierarchical-replay](../skills/run-hierarchical-replay/SKILL.md) | [../../docs/hierarchical_replay.md](../../docs/hierarchical_replay.md) |
| 画容量与命中率的 Pareto 图 | [run-pareto-analysis](../skills/run-pareto-analysis/SKILL.md) | [external_usage.md](external_usage.md) |
| 把外部 trace 转为 optimizer JSONL | [prepare-trace](../skills/prepare-trace/SKILL.md) | [trace_schema_and_conversion.md](trace_schema_and_conversion.md) |
| 在其他代码中接入 Optimizer | [integrate-optimizer-api](../skills/integrate-optimizer-api/SKILL.md) | [internal_integration.md](internal_integration.md) |
| 新增标准 optimizer 能力 | [add-optimizer-feature](../skills/add-optimizer-feature/SKILL.md) | [feature_development.md](feature_development.md) |
| 新增驱逐策略 | [add-eviction-policy](../skills/add-eviction-policy/SKILL.md) | [feature_development.md](feature_development.md) |

## 第 1 层：基本心智模型

Optimizer 是离线 KV cache 仿真器。它回放标准 JSONL trace，按配置执行索引查询、tier 流动、驱逐、调度、P2P 或 storage pool 行为，并输出命中率、IO、lifecycle 等分析文件。

标准回放分三类：

| 模式 | 入口 | 适用场景 | cache 归属 |
|---|---|---|---|
| KVCM/L3-only | `optimizer_main` 或 `optimizer_run` | 分析一个逻辑池或 KVCM/L3 回放 | 配置中的 optimizer `instance_id` |
| engine-local-only | `multi_infer_replay` | trace 已经知道请求打到哪个 pod 或推理实例 | 每个 pod 独立 cache |
| engine local + storage pool | `hierarchical_replay_main` | 需要本地多层 cache、storage pool、P2P、调度、active window 或 cache drop | engine instance 加显式 storage pool |

完整字段语义以 [../../docs/strategy_config.md](../../docs/strategy_config.md) 为准。本手册只负责路由工作，并指出哪些参数不能凭经验猜。

## 第 2 层：文档地图

| 文档 | 作用 |
|---|---|
| [module_map.md](module_map.md) | 模块职责和扩展点 |
| [external_usage.md](external_usage.md) | CLI/script 使用方式、输出和常见运行模式 |
| [trace_schema_and_conversion.md](trace_schema_and_conversion.md) | 标准 trace 字段和转换规则 |
| [config_decision_guide.md](config_decision_guide.md) | 需要人判断的配置参数 |
| [internal_integration.md](internal_integration.md) | C++/Python 接口接入方式 |
| [feature_development.md](feature_development.md) | 设计优先的功能开发流程 |
| [../tasks/README.md](../tasks/README.md) | 需求记录 |
| [../plans/README.md](../plans/README.md) | 实现前必须落盘的计划或设计文档 |
| [../../docs/strategy_config.md](../../docs/strategy_config.md) | 标准 config 和 trace 语义 |
| [../../docs/hierarchical_replay.md](../../docs/hierarchical_replay.md) | hierarchical replay 语义 |
| [../../docs/p2p_read.md](../../docs/p2p_read.md) | P2P read 设计 |
| [../../analysis/script/README.md](../../analysis/script/README.md) | 分析脚本 CLI 细节 |

## 第 3 层：必须确认的参数

运行或改动 Optimizer 前，从用户、部署配置或 trace 元数据中确认：

- `block_size`：每个 block 的 token 数。它会改变 trace key 和命中率语义。
- `bytes_per_token`：单 token KV cache 大小。容量换算必须依赖它。
- 容量：HBM/DRAM/L3 容量、单位、是 per pod 还是 global、是否需要扣除非 KV 内存。
- 回放模式：KVCM/L3-only、engine-local-only 或 hierarchical replay。
- 路由：保留 trace `instance_id`，还是模拟 scheduler 分配。
- 查询语义：`prefix_match` 或 `batch_get`。
- 写入和 touch 策略：`write_through`、`cascading`、`write_through_selective`、promote、下层 touch、selective threshold。
- 在线拓扑：扩缩容 trace 的 active window 和 cache drop 事件。
- 时间戳：原始 trace 是秒、微秒还是纳秒，转换到 ns 时不能经过 float。
- 输出目标：token hit rate、窗口命中率、Pareto 容量、lifecycle 或 IO。

如果用户不能直接给出某个值，需要解释这个值控制什么；只有在分析仍然有意义时，才可以使用明确标注的假设值。

## 第 4 层：完成标准

运行类任务：

- 输入 trace 是标准 optimizer JSONL，并按 `timestamp_ns` 排序。
- 完整 block key 和 `input_len` 满足 block-size 规则。
- config 记录了所有需要人判断的参数。
- 输出目录和历史实验隔离。
- 最终回复给出 token hit rate、关键输出路径，以及未完成的校验项。

代码改动类任务：

- `.agent/tasks/` 下有需求记录。
- `.agent/plans/` 下有实现前的计划或设计文档。
- 测试覆盖新的语义边界。
- 如果流程变化，更新用户可见文档或对应 skill。
- 如果是通用标准能力，按 optimizer 仿真开发准备到 `alibaba/tair-kvcache` 的 PR。
