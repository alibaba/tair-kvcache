# Pareto 分析 Skill

当用户需要容量与命中率曲线，或需要 95%/99% 理论命中率对应容量时，使用这个 skill。

## 需要确认的输入

- 基础 config 路径。
- `block_size` 和 `bytes_per_token`。
- 是否是非 hierarchical Pareto。标准 `tradeoff` 脚本用于非 hierarchical config。
- 点数和最大 worker 数。
- 达到 99% 理论命中后是否停止继续跑更大容量。

## 步骤

1. 校验基础 config 是单层或非 hierarchical 的容量 sweep 目标。
2. 确认输出目录与历史实验隔离。
3. 运行：

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
  -c /path/to/config.json \
  --num-points 30 \
  --max-workers 8
```

4. 只有用户明确需要快速估算时才减少点数。正常情况下保留足够的低容量和中容量点，用于展示上升段。
5. 用户只要求用已有 Pareto CSV 重画图时，不要重跑 optimizer。

## 校验

- warmup/infinite 点定义理论 token hit rate。
- 95% 目标是 `0.95 * theoretical_hit_rate`。
- 99% 目标是 `0.99 * theoretical_hit_rate`。
- 95/99 对应容量用相邻 sweep 点插值得到。
- 如果用户要求可比图，y 轴从 0 开始。

## 回复内容

报告：

- 理论 token hit rate
- 95% 和 99% 理论命中率对应容量
- 图路径
- summary CSV 路径
