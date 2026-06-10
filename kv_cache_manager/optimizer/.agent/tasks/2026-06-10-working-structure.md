# Optimizer 工作结构整理

状态：completed
日期：2026-06-10
Plan：../plans/2026-06-10-working-structure.md

## 需求

减少显式项目风格标签，把工作指南从 `docs/` 移出，让项目通过渐进式披露自然可用。同时要求后续每个新需求都记录在 `.agent/tasks/`，涉及实现或流程变化的需求都在 `.agent/plans/` 下有对应计划或设计文档。

## 范围

- 将工作指南内容放入 `.agent/handbook/`。
- 将可见文案改为中性的 handbook/workflow 表达。
- 新增 `.agent/tasks/` 和 `.agent/plans/`，包含模板和使用规则。
- 更新 skills 和仓库入口链接到新结构。
- 将 `.agent` 下工作流材料统一中文化。

## 非目标

- 不改 optimizer 运行时代码。
- 不改 Bazel target。
- 不重新跑数据分析。

## 需要人确认的决策

- 工作流材料放在 `kv_cache_manager/optimizer/.agent/`。
- 项目通过自然结构体现工作方式，而不是反复声明项目风格。
- 需求记录放 `.agent/tasks/`，计划或设计记录放 `.agent/plans/`。
- `.agent` 下内容使用中文。

## 校验

- 本地 Markdown 链接可解析。
- 没有旧路径残留。
- `git diff --check` 通过。
