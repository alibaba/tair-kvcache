# 新增 Optimizer 功能 Skill

当需要新增标准 optimizer 能力、config 语义、trace 语义、scheduler 行为、storage-pool 行为、P2P 行为或分析输出时，使用这个 skill。

## 必须先做

改代码前先创建 task 和 plan：

```text
kv_cache_manager/optimizer/.agent/tasks/YYYY-MM-DD-short-title.md
kv_cache_manager/optimizer/.agent/plans/YYYY-MM-DD-short-title.md
```

使用 [../../tasks/TEMPLATE.md](../../tasks/TEMPLATE.md) 和 [../../plans/TEMPLATE.md](../../plans/TEMPLATE.md)。

## 步骤

1. 按 [../../handbook/feature_development.md](../../handbook/feature_development.md) 判断功能归属。
2. 写 task 和 plan，至少包含：
   - 影响的回放模式
   - trace/config 变化
   - hit-rate/counter 语义
   - 非法输入
   - 测试计划
3. 在最小归属模块内实现。
4. 尽量在 config 或 trace 解析阶段加入校验。
5. 在语义边界补单元测试。
6. 更新文档和相关 skill。
7. 如果是通用标准能力，按 optimizer 仿真开发准备到 `https://github.com/alibaba/tair-kvcache`。

## 校验

- 目标 Bazel 测试通过。
- 无关回放模式的既有行为不变。
- 文档解释清楚需要人确认的参数。

## 回复内容

报告：

- task 路径
- plan 路径
- 改动模块
- 执行的测试
- 是否适合开源 optimizer 仿真 PR
