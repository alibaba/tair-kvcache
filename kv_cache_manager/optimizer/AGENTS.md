# Optimizer 工作入口

这个目录是 KVCacheManager Optimizer 任务的局部入口。按任务需要逐层阅读，不要一开始就扫完整个目录。

## 渐进式阅读顺序

1. 先读 [.agent/handbook/README.md](.agent/handbook/README.md)，确定任务路径。
2. 执行固定流程时，优先使用 [.agent/skills/](.agent/skills/) 下的任务 skill。
3. 修改代码前，先在 [.agent/tasks/](.agent/tasks/) 记录需求，再在 [.agent/plans/](.agent/plans/) 写计划或设计文档。
4. plan 落盘后必须先和用户讨论设计决策与设计细节；用户确认后才能开始实现。不要把未确认的 agent 假设直接写成代码。
5. 只有任务需要字段、接口或扩展点细节时，再继续打开更深层参考。

## 不可违反的约束

- Instance 隔离仍是默认规则：KVCache 只在同一个 `instance_id` 内复用。
- P2P 和 storage pool 是明确的 hierarchical replay 能力，不要把普通 optimizer instance 默默解释成共享 cache。
- LiteHit 是 full-attention 专用的容量无关分析核：核心不接收容量，任何容量换算只能经过 `HitCurveProjector`；不支持 TTL、分层、admission 和 prefetch。
- LiteHit facts 是全有或全无的对账账本：任何畸形行、乱序时间戳或长度违约都必须 fail-fast，不允许静默丢行。
- 标准命中率是 token hit rate：`HitRate = HitTokens / InputTokens`。
- 标准 `get` 和 `request` trace 必须包含 `input_len`。
- `keys` 只包含完整 block key；不足一个 block 的尾部 token 不写入 `keys`，但仍计入 `input_len`。
- 任何新的 optimizer 需求都必须在 `kv_cache_manager/optimizer/.agent/tasks/` 下有 task 记录。
- 任何会改变代码、行为、流程或用户可见文档的需求，都必须在实现前于 `kv_cache_manager/optimizer/.agent/plans/` 下写计划或设计文档。
- plan 必须列出待用户确认的设计决策；这些决策被用户确认前，不得开始代码实现。
- workflow 记录实现后的事实过程，不能替代实现前的 plan review。

## 运行前必须确认的参数

不要猜这些参数，必须从用户、部署配置或 trace 元数据中确认：

- `block_size`
- `bytes_per_token`
- cache 容量和容量单位
- 回放模式：KVCM/L3-only、engine-local-only、hierarchical replay、LiteHit facts 或 LiteHit 在线
- 路由语义：保留 trace 路由，还是模拟 scheduler
- 查询语义：`prefix_match` 或 `batch_get`
- tier 写入模式、touch/promote 行为
- 扩缩容 trace 的 active window 或 cache drop 事件
- 原始 trace 的时间单位和时间戳精度
- 新能力是否应作为 optimizer 仿真能力提到开源仓库

## 标准能力 PR 说明

如果新增能力是通用能力，且适合进入开源仓库，应按 optimizer 仿真开发提 PR 到：

https://github.com/alibaba/tair-kvcache
