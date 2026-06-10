# Optimizer 计划文档

任何会改变 optimizer 代码、行为、流程或用户可见文档的需求，都必须在实现前先在这里写 plan 或设计文档。

## 与 Tasks 的关系

- `tasks/` 记录用户要什么，以及为什么要做。
- `plans/` 记录准备怎么做，以及如何验证。

每个 task 应链接到一个主要 plan。一个 plan 也应反向链接到对应 task。

## 文件命名

使用和 task 相同的短标题：

```text
YYYY-MM-DD-short-title.md
```

## 必填内容

从 [TEMPLATE.md](TEMPLATE.md) 开始。plan 必须包含：

- 当前状态
- 设计或行为方案
- 预计改动文件 / 模块
- 兼容性和迁移说明
- 校验计划
- 如适用，包含开源 PR 说明

文档类工作可以写轻量 plan，但实现前仍必须有 plan。
