# Optimizer 工作结构整理计划

状态：completed
日期：2026-06-10
Task：../tasks/2026-06-10-working-structure.md

## 当前状态

第一版把工作指南材料放在 `kv_cache_manager/optimizer/docs/` 下，并在若干入口中使用了显式项目风格标签。同时虽然要求功能开发前写设计文档，但没有项目级 `tasks/` 和 `plans/` 目录。

## 设计方案

采用更自然的渐进式结构：

- `AGENTS.md`：optimizer 局部工作入口和硬约束。
- `.agent/handbook/`：任务路由、模块地图、使用方式、trace 转换、配置决策、接入方式、功能开发。
- `.agent/skills/`：可执行的任务工作流。
- `.agent/tasks/`：每个用户需求一条记录。
- `.agent/plans/`：每个影响实现的需求一份计划或设计记录。

用户可见文案使用“工作手册”“工作入口”“工作流”“渐进式披露”等自然表达。

## 文件和模块

- 将工作指南文件移动到 `.agent/handbook/*`。
- 更新 `docs/README.md`、`docs/optimizer.md`、`kv_cache_manager/optimizer/README.md`、`AGENTS.md` 和 `.agent/skills/*` 中的链接。
- 新增 `.agent/tasks/README.md`、`.agent/tasks/TEMPLATE.md` 和本 task 记录。
- 新增 `.agent/plans/README.md`、`.agent/plans/TEMPLATE.md` 和本 plan 记录。
- 将 `.agent` 下材料统一中文化。

## 兼容性

这是文档和工作流结构调整，不改变 optimizer 运行行为、config 解析、trace schema 或 Bazel target。

## 校验

- 运行本地 Markdown 链接校验。
- 运行 `git diff --check`。
- 搜索旧路径和显式项目风格标签残留。

## 开源 PR 说明

这是 optimizer 仿真开发的仓库结构工作。如果希望开源仓库保持相同贡献流程，可以随 optimizer 文档改造一起提到 `https://github.com/alibaba/tair-kvcache`。
