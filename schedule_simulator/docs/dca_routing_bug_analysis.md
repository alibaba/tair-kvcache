# DirectCacheAware 路由策略 Bug 分析与修复

## 现象

使用 DirectCacheAware 路由策略 + request 级调度模式跑 Qwen3.6 仿真时，**100% 请求被路由到 Pod #0**，其余 Pod 完全空闲。

### 实验数据（1000 请求, 5 P 节点, enable_hierarchical=True）

| 策略 | TTFT mean | TTFT p99 | Pod 分布 |
|---|---|---|---|
| RoundRobin | 52,822 ms | 105,521 ms | P0=200, P1=200, P2=200, P3=200, P4=200 |
| DirectCacheAware（修复前） | 275,642 ms | 547,176 ms | **P0=1000, P1=0, P2=0, P3=0, P4=0** |

DCA 的 TTFT 是 RR 的 **5.2 倍**，完全符合 5:1 集中的理论预期。

## 根因

### 执行时序

`time_routing_loop`（run.py:532）每个请求的处理顺序：

```
1. policy.update_load()    →  update_workload()  →  _load = total_req - completed = 0
2. policy.select_worker()  →  检查 loads 全为 0 → 走 cache 路由 → 选 Pod #0
3. _enable_scheduler()     →  _run_request_level() → 请求一步完成，进 completed_requests
```

### 核心矛盾

`update_workload`（dispatch_policy.py:97-116）在每次路由决策**前**被调用：

```python
# Worker.update_load (line 54):
self._load = max(0, self.total_req - completed_requests_len)
```

在 request 级模式中，`_run_request_level()` 在**单步内**就把请求放进 `completed_requests`。所以下一个请求路由时：

- `total_req == completed_count`（所有之前分配的请求都已完成）
- `_load = 0`（**永远**）
- **所有 Pod 负载都是 0**

### 负载追踪时序表

| 时刻 | Pod #0 total_req | Pod #0 completed | Pod #0 _load | Pod #1-4 _load |
|---|---|---|---|---|
| 第 1 个请求路由前 (update_workload) | 0 | 0 | **0** | 0 |
| 第 1 个请求路由后 (increment_load) | 1 | 0 | 1 | 0 |
| 第 1 个请求处理后 (_run_request_level) | 1 | **1** | - | 0 |
| 第 2 个请求路由前 (update_workload) | 1 | 1 | **0** | 0 |
| ... | N | N | **0** | 0 |

### 后果链

1. `select_worker` 第 346 行：`loads = [0, 0, 0, 0, 0]`
2. 第 349-352 行：`max_load - min_load = 0`，永远 ≤ `balance_abs_threshold=64` → `is_imbalanced = False`
3. 走 cache 路由（第 363-379 行）→ Pod #0 有 cache 数据 → 选 Pod #0
4. Pod #0 积累更多 cache → 正反馈循环 → **100% 集中**

### 与 iteration 级模式对比

| | Request 级 | Iteration 级 |
|---|---|---|
| 请求完成时机 | 单步内完成 | 多步 iteration 后完成 |
| 路由下一请求时前一请求状态 | 已在 completed_requests | 仍在处理中 |
| update_workload 后 _load | 永远 = 0 | > 0（反映在飞请求数） |
| is_imbalanced 能否触发 | 永远不能 | 正常触发 |

## 涉及代码

| 文件 | 行号 | 说明 |
|---|---|---|
| `dispatch_policy.py` | 51-54 | `Worker.update_load`: `_load = total_req - completed` |
| `dispatch_policy.py` | 97-116 | `BasePolicy.update_workload`: 遍历 scheduler 调用 `update_load` |
| `dispatch_policy.py` | 387-401 | `DirectCacheAwarePolicy.update_load`: 调用 `update_workload` 重置负载 |
| `dispatch_policy.py` | 346-352 | `select_worker`: 负载均衡检查（永远 False） |
| `run.py` | 532 | `time_routing_loop`: 路由前调用 `update_load` |
| `sglang_scheduler.py` | 394-435 | `_run_request_level`: 单步完成请求 |

## 修复（2026-06-10 已实施）

**方案 A（已采用）：request 级模式不调用 `update_workload`**

`dispatch_policy.py` 两处改动（CacheAwarePolicy + DirectCacheAwarePolicy）：

```python
# CacheAwarePolicy.update_load (line 294-295):
if not schedulers[0].scheduler_config.request_level_scheduling:
    await self.update_workload(schedulers, current_time)

# DirectCacheAwarePolicy.update_load (line 402-403):
if not schedulers[0].scheduler_config.request_level_scheduling:
    await self.update_workload(schedulers, current_time)
```

request 级模式下 `_load` 仅靠 `increment_load` 累积，`is_imbalanced` 可正常触发。`_load` 语义从"在飞请求数"变为"累积分配数差异"，对 request 级模式这是正确的。

## 修复验证

### 小规模验证（1000 请求, 5 pods）

| 策略 | TTFT mean | Pod 分布 |
|---|---|---|
| RoundRobin | 52,822 ms | 各 200 |
| DCA（修复后） | **52,822 ms** | **各 200** |

修复后 DCA 在 cache 冷启动时正确退化为均匀负载均衡。

### 全量验证（204 万请求, 200 pods）

| 指标 | 值 |
|---|---|
| 请求数 | 2,041,633 |
| Pod 数 | 200（e02-sg 部署） |
| TTFT mean | **577 ms** |
| TTFT p50 | 490 ms |
| TTFT p99 | 2,698 ms |
| Cache local hit | 117M tokens |
| Cache peer hit | 352M tokens |
| Cache pool hit | 23K tokens |
| **总命中率** | **86.6%** |
| 仿真耗时 | 8,667s（2h24m） |
| 仿真速度 | ~240 req/s |

结果路径：`/sgl-workspace/claude_workspace/data/qwen36_full_sim/results_full_dca_200pods/`

### 测试回归

157 个测试全通过，无回归。

## 数据处理说明

全量仿真使用的数据经过以下过滤：

1. 原始数据：`main_svc_full.jsonl`（204 万请求，285 pods，3 个部署）
2. 过滤条件：仅保留 `instance_id` 以 `e02-sg` 开头的记录
3. 过滤后：`main_svc_e02sg.jsonl`（2,041,633 请求，200 pods）
4. 排除：`ds-acb49efe`（84 pods, 1263 请求）和 `ds-458347f3`（1 pod, 1 请求）

## 复现命令

```bash
cd /sgl-workspace/claude_workspace/schedule_simulator

# 全量仿真（修复后的 DCA，200 pods）
python3 scripts/run_simulation.py \
  --dataset /sgl-workspace/claude_workspace/data/qwen36_full_sim/main_svc_e02sg.jsonl \
  --num-prompts 2041633 \
  --request-level \
  --predictor-pkl /sgl-workspace/claude_workspace/data/qwen36_predictor/qwen36_prefill_predictor.pkl \
  --kv-bytes-per-token 640 --max-num-tokens 999999999 \
  --num-p-instances 200 --routing direct_cache_aware \
  --enable-hierarchical --enable-p2p \
  --output-dir /sgl-workspace/claude_workspace/data/qwen36_full_sim/results_full_dca_200pods
```
