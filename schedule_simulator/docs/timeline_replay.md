# Timeline 回放模式

## 概述

Timeline 回放模式允许使用真实 trace 数据替代仿真器的路由策略和/或时延预测器，用于隔离各模块的误差贡献。同时 simulation_summary.json 移除了冗余的 e2e_latency 指标，新增了拓扑信息字段。

## 新增功能

### 功能 1: Timeline 回放模式

通过 `--timeline-mode` 参数启用，支持三种子模式：

| 模式 | 路由来源 | 时延来源 | 用途 |
|------|----------|----------|------|
| `route_only` | timeline 文件 | 预测器 | 隔离路由策略误差 |
| `route_and_latency` | timeline 文件 | timeline 文件 | 完全复用真实数据 |
| `latency_only` | 仿真器策略 | timeline 文件 | 隔离时延预测器误差 |

#### 输入文件格式要求

Timeline 文件为 JSONL 格式（与 `--dataset` 共用同一文件），每行需包含：

```json
{
  input_block_hash_ids: [[...], ...],
  input_length: 2048,
  timestamp: 1718000000.123,
  pods: [pod-name-xxx],
  prefill: {prefill_ms: 325.6}
}
```

- `pods`: 数组，取第一个元素作为路由目标 pod
- `prefill.prefill_ms`: 该请求的真实 prefill 耗时（毫秒）
- `input_block_hash_ids`: 用于 cache 命中率统计（所有模式下仍执行 match_prefix）

#### Pod 映射规则

仿真器自动扫描整个文件，提取所有唯一 pod 名称，**排序后**建立稳定的 `{pod_name → pod_index}` 映射。节点数由文件中实际出现的 pod 数量决定，忽略 `--num-p-instances` 参数。

### 功能 2: simulation_summary.json 字段变更

**移除的字段（6个）：**
- `mean_e2e_latency_ms`, `median_e2e_latency_ms`, `std_e2e_latency_ms`
- `p90_e2e_latency_ms`, `p95_e2e_latency_ms`, `p99_e2e_latency_ms`

**新增的字段（3个）：**
- `num_nodes`: prefill 节点数量
- `hbm_capacity_gb`: HBM 容量 (GB)
- `mem_capacity_gb`: 内存容量 (GB)

## CLI 使用示例

```bash
# route_and_latency 模式：完全使用真实路由+真实时延
python3 scripts/run_simulation.py \
    --dataset /path/to/timeline.jsonl \
    --num-prompts 1000 \
    --timeline-mode route_and_latency \
    --request-level \
    --page-size 2048 --data-block-size 256 \
    --output-dir sim_results_timeline

# route_only 模式：使用真实路由，时延由预测器计算
python3 scripts/run_simulation.py \
    --dataset /path/to/timeline.jsonl \
    --num-prompts 1000 \
    --timeline-mode route_only \
    --request-level \
    --predictor-pkl /path/to/predictor.pkl \
    --page-size 2048 --data-block-size 256 \
    --output-dir sim_results_route_only

# latency_only 模式：使用仿真器路由策略，时延来自 timeline
python3 scripts/run_simulation.py \
    --dataset /path/to/timeline.jsonl \
    --num-prompts 1000 \
    --timeline-mode latency_only \
    --request-level \
    --routing cache_aware \
    --page-size 2048 --data-block-size 256 \
    --output-dir sim_results_latency_only
```

## 代码变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `src/.../types.py` | 修改 | 新增 `TimelineMode` 枚举，`FakeRequest` 增加字段 |
| `src/.../timeline_loader.py` | **新增** | JSONL 扫描 + pod 映射 |
| `src/.../benchmark.py` | 修改 | 注入 timeline 信息到请求 |
| `src/.../run.py` | 修改 | 路由分支 + 节点数覆盖 + summary 新字段 |
| `src/.../sglang_scheduler.py` | 修改 | 时延计算分支 |
| `src/.../utils.py` | 修改 | 移除 6 个 e2e 字段 |
| `scripts/run_simulation.py` | 修改 | 新增 `--timeline-mode` 参数 |
| `tests/test_timeline_replay.py` | **新增** | 13 个测试用例 |
| `tests/assets/timeline_sample.jsonl` | **新增** | 测试数据 |
| `tests/test_metrics_and_export.py` | 修改 | 适配字段变更 |

## 设计要点

1. **单文件模式**：timeline 信息与 cache 数据在同一 JSONL 文件中，无需额外关联
2. **Cache 查询保留**：所有 timeline 模式仍执行 `match_prefix()`，确保 cache hit ratio 统计有效
3. **Fallback 机制**：当某条请求缺少 `prefill_ms` 时，自动 fallback 到预测器计算时延
4. **向后兼容**：`--timeline-mode disabled`（默认值）行为与改动前完全一致
