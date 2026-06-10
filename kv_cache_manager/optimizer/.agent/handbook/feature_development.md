# 功能开发

Optimizer 功能应该容易理解、评审和扩展。新增能力必须先设计、再实现，并落到正确的模块边界。

## 必须遵守的顺序

1. 在 `kv_cache_manager/optimizer/.agent/tasks/` 记录需求。
2. 在 `kv_cache_manager/optimizer/.agent/plans/` 写计划或设计文档。
3. 停下来和用户讨论 plan 中的设计决策与设计细节，尤其是会影响 trace/config 语义、命中率口径、输出文件、兼容性或后续实验解释的部分。
4. 在 plan 中记录用户确认结果或需要修改的方向。
5. 只有用户确认后，才能在最小归属模块内实现计划中的能力。
6. 为行为和非法输入补聚焦测试。
7. 如果工作流变化，同步更新用户文档和对应 skill。
8. 实现完成后，如需复盘，在 `kv_cache_manager/optimizer/.agent/workflows/` 记录实际执行过程和逐文件改动。
9. 如果是通用能力，按 optimizer 仿真开发准备到 `https://github.com/alibaba/tair-kvcache` 的 PR。

使用 [../tasks/TEMPLATE.md](../tasks/TEMPLATE.md) 和 [../plans/TEMPLATE.md](../plans/TEMPLATE.md)。

## 功能类型路由

| 功能类型 | 从哪里开始 | 必须更新 |
|---|---|---|
| 新 config 字段 | `config/` | parser、writer、validation、文档、测试 |
| 新 trace 语义 | `trace_loader/` 和 runner/manager | schema 文档、转换文档、测试 |
| 新驱逐策略 | `eviction_policy/` | enum parser、factory、策略测试；如对用户可见则更新 skill |
| 新 tier flow 行为 | `index/`、`manager/`、`tier_flow/` | flow 文档、事件语义、命中率测试 |
| 新 scheduler 策略 | `scheduler/` | hierarchical 文档、active-window 行为测试 |
| 新 P2P 行为 | `p2p/`、`manager/hierarchical_replay_manager.*` | P2P 文档和 peer-selection 测试 |
| 新分析输出 | `analysis/`、`analysis/tracker/` | CSV 列文档；可行时补画图测试 |
| 新 Python/CLI 工作流 | `pybind/` 或 `analysis/script/` | README、skill、CLI 参数校验 |

## 设计评审清单

- 影响哪种回放模式？
- 是否改变 token hit-rate 语义？
- 是否改变 block counter，还是只改变派生图？
- 是否保持 instance 隔离？
- 是否需要 active-window 或 cache-drop 行为？
- 哪些 config 字段必须由人确认，不能猜？
- 哪些非法输入应该 fail fast？
- 输出文件或 CSV 列有什么变化？
- 哪些测试能证明新行为？

## Plan Review 规则

plan 不是实现后的总结，而是实现前的讨论材料。对新增能力而言，plan 至少要把以下内容列成“待用户确认的设计决策”：

- 能力作用范围，例如只影响某个 replay 入口，还是改变标准 trace/config。
- 新字段的命名、默认值、非法输入和兼容策略。
- 新行为如何影响命中率、block counter、IO 或生命周期统计。
- 是否复用现有 manager/index/eviction 结构，还是新增独立模块。
- 哪些参数必须由用户、部署配置或 trace 元数据确认，不能由 agent 猜测。
- 哪些输出是用户可见的稳定接口。

用户确认前，agent 只能继续阅读代码、补充 plan 或提出备选方案，不得开始编辑实现代码。若用户要求先做原型，也必须在 plan 中明确标记为“未确认假设版”。

## 标准开源 PR 标准

如果能力不绑定私有数据集、私有部署名或一次性实验脚本，就适合进入开源仓库。PR 描述需要明确这是 optimizer 仿真开发，并包含：

- 问题和动机
- 用户可见的 config/API 变化
- trace schema 变化，如有
- 行为示例
- 验证命令
- 已更新的文档或 skill
