# Schedule Simulator — LLM 推理多实例仿真平台

## 1. 概述

基于 schedule_simulator（调度/时序建模）和 tair-kvcache Optimizer（缓存命中建模）的集成仿真平台，用于评估不同调度策略、缓存配置、部署拓扑下的 **TTFT / 吞吐量 / 缓存命中率**。

### 与 tair-kvcache 的关系

```
tair-kvcache/ (本仓库)
├── kv_cache_manager/optimizer/     C++ 缓存命中建模
│   ├── RadixTree 前缀匹配
│   ├── P2P 跨实例读取 (TierGlobalTracker)
│   ├── 容量驱逐 (LRU)
│   └── pybind/ → kvcm_py_optimizer.so
├── schedule_simulator/             Python 调度/时序仿真 ← 你在这里
│   ├── 请求生成 & 路由策略
│   ├── 时序调度仿真
│   └── 通过 HierarchicalCacheAdapter 调用 kvcm_py_optimizer.so
└── hisim/                          HiSim（独立工具，非本模块）
```

两者通过 `HierarchicalCacheAdapter` 集成，使用 `--enable-hierarchical` 一键启用。

## 2. 架构

```
输入数据 (JSONL / 随机生成)
    ↓
BenchmarkEmulator (请求生成, block_ids 透传)
    ↓
DisaggBenchmarkRunner (全局时间驱动路由)
    ├── P Policy: Random / RoundRobin / PowerOfTwo / CacheAware / DirectCacheAware
    ↓
SGLangScheduleEmulator ×N (每 P 节点一个)
    ├── iteration 级: get_batch → predict_iter → process_result (循环)
    └── request 级: predict_request_time → 一步完成
    ↓
tree_cache (可插拔)
    ├── PrefixCache (基类/no-op)
    ├── HiRadixCache (统计模式 + 带宽建模)
    ├── SimHiRadixCache (真实 token 模式)
    └── HierarchicalCacheAdapter → HierarchicalReplayManager (C++)
        ├── OptimizerManager (引擎本地 RadixTree)
        ├── TierGlobalTracker (P2P 跨实例读)
        └── HashStoragePoolManager (共享存储池)
    ↓
输出: TTFT / TPOT / 吞吐量 / 排队等待 / 三级缓存命中率 + CSV 导出
```

### 核心源码文件

| 文件 | 作用 |
|------|------|
| `src/schedule_simulator/schedule_emulator/run.py` | Runner 入口：BenchmarkRunner / DisaggBenchmarkRunner |
| `src/schedule_simulator/schedule_emulator/sglang_scheduler.py` | 调度器核心：event_loop / `_run_request_level` |
| `src/schedule_simulator/schedule_emulator/types.py` | 配置类：SchedulerConfig / PlatformConfig / RouterConfig |
| `src/schedule_simulator/schedule_emulator/dispatch/dispatch_policy.py` | 五种路由策略实现 |
| `src/schedule_simulator/schedule_emulator/hierarchical_cache_adapter.py` | Optimizer 集成桥接层 |
| `src/schedule_simulator/schedule_emulator/hierarchical_config_builder.py` | 自动生成 Optimizer JSON 配置 |
| `src/schedule_simulator/infer_time_predictor/request_level.py` | Request 级时间预测器 |
| `scripts/run_simulation.py` | CLI 入口（一键仿真） |
| `scripts/prepare_simulation_data.py` | 从 timeline 原始数据生成仿真输入 JSONL |
| `scripts/train_predictor.py` | 训练 request 级时间预测器 |

## 3. 安装与配置

### 环境要求

| 项目 | 版本 |
|------|------|
| Python | 3.12+ |
| Bazel | 6.4.0（内部定制版本） |
| OS | Linux (x86_64) |

### Step 1: Python 依赖安装

```bash
cd schedule_simulator
pip install -e .
```

### Step 2: 构建 C++ Optimizer 模块

```bash
cd /path/to/tair-kvcache  # 仓库根目录

export USE_BAZEL_VERSION=6.4.0-dev_4f82cf83afde5e0f71b76a5611cedac7ac456fef
export BAZELISK_BASE_URL=https://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/third_party_archives/bazel_binary

bazel build //kv_cache_manager/optimizer/pybind:kvcm_py_optimizer
```

产物位置：`bazel-bin/kv_cache_manager/optimizer/pybind/kvcm_py_optimizer.so`

### Step 3: 设置 PYTHONPATH

```bash
# 将 kvcm .so 路径加入 PYTHONPATH
export PYTHONPATH=/path/to/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind:$PYTHONPATH

# 验证
python3 -c "import kvcm_py_optimizer; print(OK)"
```

> **注意**：不启用 `--enable-hierarchical` 时不需要 C++ 模块，纯 Python 即可运行。

## 4. 快速开始

### 最小验证（无外部依赖）

```bash
cd schedule_simulator
python3 scripts/run_simulation.py \
  --num-prompts 100 \
  --request-level \
  --ms-per-token 0.1 \
  --num-p-instances 5 \
  --routing round_robin \
  --output-dir /tmp/quick_test
```

### 带缓存仿真（需 C++ 模块 + 数据集）

```bash
python3 scripts/run_simulation.py \
  --dataset data/enriched_input.jsonl \
  --num-prompts 5000 \
  --request-level \
  --predictor-pkl data/predictor.pkl \
  --kv-bytes-per-token 640 --max-num-tokens 999999999 \
  --num-p-instances 5 --routing direct_cache_aware \
  --enable-hierarchical --enable-p2p \
  --page-size 256 \
  --output-dir ./results_small
```

### 全量仿真（大规模评估）

```bash
python3 scripts/run_simulation.py \
  --dataset data/main_svc_e02sg.jsonl \
  --num-prompts 2041633 \
  --request-level \
  --predictor-pkl data/qwen36_prefill_predictor.pkl \
  --kv-bytes-per-token 640 --max-num-tokens 999999999 \
  --num-p-instances 200 --routing direct_cache_aware \
  --enable-hierarchical --enable-p2p \
  --page-size 256 \
  --output-dir ./results_full
```

### 概率路由仿真（推荐用于策略评估）

```bash
python3 scripts/run_simulation.py \\
  --dataset data/h21_32_256k_15min_full.jsonl \\
  --num-prompts 50000 \\
  --request-level \\
  --ms-per-token 0.1 \\
  --kv-bytes-per-token 640 --max-num-tokens 999999999 \\
  --num-p-instances 32 --routing direct_cache_aware \\
  --enable-hierarchical --enable-p2p \\
  --page-size 2048 --data-block-size 256 \\
  --topk-routing --lmax 40 --weight-prefix 30.0 --weight-load 10.0 \\
  --output-dir ./results_topk
```

> **注**：概率路由需配合 `--enable-hierarchical` 使用，且 `--routing` 必须为 `cache_aware` 或 `direct_cache_aware`。
> 关闭 `--topk-routing` 则回退到传统确定性路由，完全向后兼容。

## 5. 完整参数说明

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | None | 输入 JSONL 文件路径（不指定则随机生成） |
| `--num-prompts` | 100 | 请求数量 |
| `--request-level` | False | 启用 request 级调度（推荐，跳过 iteration 循环） |
| `--predictor-pkl` | None | 预测器 pkl 文件路径 |
| `--ms-per-token` | None | 恒定速率预测器（ms/token），与 pkl 二选一 |
| `--model` | Qwen2.5-3B | 模型名（需注册到 kunlun_commons） |
| `--device` | H20 | 硬件型号 |
| `--kv-bytes-per-token` | None | 每 token KV cache 字节数（跳过 ModelInfo 推导） |
| `--max-num-tokens` | None | L1 device cache 容量（token 数） |
| `--l2-cache-tokens` | None | L2 host cache 容量（默认 2×L1） |
| `--hbm-capacity` | None | HBM 容量 (GB)，覆盖硬件默认值 |
| `--mem-capacity` | None | DRAM 容量 (GB) |
| `--num-p-instances` | 1 | Prefill 实例数 |
| `--routing` | round_robin | 路由策略（见下表） |
| `--enable-hierarchical` | False | 启用 tair-kvcache Optimizer C++ 集成 |
| `--enable-p2p` | True | 启用 P2P 跨实例缓存读取 |
| `--page-size` | None | block 大小（tokens/block），需与数据中 block_ids 粒度一致 |
| `--write-policy` | write_through | 缓存写策略: write_through / write_back / write_through_selective |
| `--pool-capacity` | 2.0 | 共享存储池容量 (GB)，设 0 关闭 L3 |
| `--topk-routing` | False | 启用概率路由（需配合 cache_aware/direct_cache_aware） |
| `--lmax` | 40 | 满载归一化基准（Lmax） |
| `--weight-prefix` | 30.0 | 前缀命中打分权重 wp |
| `--weight-load` | 10.0 | 负载均衡打分权重 wl |
| `--output-dir` | ./sim_results | 输出目录 |
| `--seed` | 42 | 随机种子 |

### 路由策略详细说明

#### 基础策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `random` | 随机分配 | 基线对比 |
| `round_robin` | 轮询 | 均匀负载基线 |
| `power_of_two` | 随机选两个节点，取负载较低的 | 无缓存感知时的负载均衡 |

#### `cache_aware` — 近似前缀树路由

**原理**：路由层维护一棵独立的近似 RadixTree（通过 C++ Optimizer `WriteCache` 更新），每次请求到达时调用 `ChooseBestEngine` 找到最长前缀匹配的节点。

**决策流程**：
1. 调用 `ChooseBestEngine(block_ids)` 获取最优引擎 + 命中数
2. 计算 `match_rate = hit_count × page_size / input_length`
3. 若 `match_rate > cache_threshold`（默认 0.3）：检查负载均衡
   - `is_overloaded = (best_load - min_load > balance_abs_threshold) AND (best_load > min_load × balance_rel_threshold)`
   - 若过载 → 回退到最小负载节点（`load_balance_override`）
   - 若未过载 → 路由到最优缓存节点（`cache_hit`）
4. 若 `match_rate <= cache_threshold` → 回退到最小负载节点（`cache_miss_fallback`）

**关键配置**：
- `cache_threshold = 0.3`（命中率低于此值视为无缓存）
- `balance_abs_threshold = 8`（负载差绝对阈值）
- `balance_rel_threshold = 1.5`（负载差相对阈值，1.5 表示 50% 以上差距）

**限制**：近似树与引擎真实缓存状态存在延迟偏差；确定性选择导致热点集中。

**入口命令**：
```bash
python3 scripts/run_simulation.py \\
  --dataset data/input.jsonl \\
  --num-p-instances 32 --routing cache_aware \\
  --request-level --ms-per-token 0.1 \\
  --enable-hierarchical --enable-p2p \\
  --page-size 2048 --data-block-size 256
```

#### `direct_cache_aware` — 直连 Optimizer 路由（**推荐**）

**原理**：与 `cache_aware` 相同的决策逻辑，但直接查询各引擎 Optimizer 的真实 RadixTree（零延迟、零偏差），用于评估缓存路由的理想上限。

**与 `cache_aware` 的区别**：
| 维度 | cache_aware | direct_cache_aware |
|------|-------------|-------------------|
| 前缀树来源 | 路由层近似副本 | 引擎真实缓存 |
| 一致性延迟 | 有（WriteCache 更新延迟） | 无 |
| 适用场景 | 模拟线上实际行为 | 评估策略理论上限 |

**入口命令**：
```bash
python3 scripts/run_simulation.py \\
  --dataset data/input.jsonl \\
  --num-p-instances 32 --routing direct_cache_aware \\
  --request-level --ms-per-token 0.1 \\
  --enable-hierarchical --enable-p2p \\
  --page-size 2048 --data-block-size 256
```

#### `cache_aware` / `direct_cache_aware` + `--topk-routing` — 概率路由（**新增**）

**原理**：线上 Pre-Cache-Aware-Scheduler 的仿真实现。不再确定性选择单个最优节点，而是综合所有节点的“缓存命中率 + 负载”进行指数打分，再从 TopK 候选中加权随机采样。

**核心公式**：
```
p_i = hit_count_i × page_size / input_token_length   # 前缀命中比例 [0,1]
l_i = load_i / Lmax                                   # 归一化负载
l̄ = mean(l_i)                                        # 平均负载

score_i = 2^(wp × p_i + wl × (l̄ - l_i))             # 指数打分
```

**决策流程**：
1. 调用 `ChooseTopKEngines(block_ids)` 获取**所有**有命中的节点
2. 对每个候选节点计算 `score_i`
3. **TopK 截断**：`K = max(⌈√N⌉, 5)`，只保留得分最高的 K 个节点
4. **加权随机采样**：`Pr(i) = score_i / Σ(score_j ∈ topK)`

**参数说明**：
| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--topk-routing` | 关闭 | 启用概率路由 |
| `--lmax` | 40 | 满载基准（归一化分母） |
| `--weight-prefix` | 30.0 | 缓存命中权重 wp（越大越倾向高命中节点） |
| `--weight-load` | 10.0 | 负载均衡权重 wl（越大越倾向低负载节点） |

**分数量级示例**（直觉理解）：
| p_i | 负载状态 | score |
|-----|----------|-------|
| 1.0（满命中） | 空闲 | ~2³² ≈ 4.3×10⁹ |
| 1.0（满命中） | 均载 | ~2³⁰ ≈ 10⁹ |
| 0（无命中） | 空闲 | ~2² = 4 |
| 0（无命中） | 满载 | ~2⁻⁵ = 0.03 |

**优势**：
- 避免热点集中：高命中节点概率更高但不独占
- 负载均衡内嵌公式：无需事后 overload 补救
- TopK 截断：去除长尾噪音，采样更高效

**入口命令**：
```bash
python3 scripts/run_simulation.py \\
  --dataset data/input.jsonl \\
  --num-p-instances 32 --routing direct_cache_aware \\
  --request-level --ms-per-token 0.1 \\
  --enable-hierarchical --enable-p2p \\
  --page-size 2048 --data-block-size 256 \\
  --topk-routing --lmax 40 --weight-prefix 30.0 --weight-load 10.0
```

#### 策略对比总结

| 维度 | cache_aware | direct_cache_aware | + topk_routing |
|------|------------|-------------------|----------------|
| 选择方式 | 确定性：选最大命中 | 确定性：选最大命中 | 概率性：加权随机 |
| 负载均衡 | 阈值触发覆盖 | 阈值触发覆盖 | 内嵌公式 |
| 热点风险 | 高 | 高 | 低 |
| 缓存利用率 | 最高 | 最高 | 略低 |
| 尾延迟（P99） | 高（过载） | 高（过载） | 低（均匀）|
| 输出日志 | 标准字段 | 标准字段 | +概率路由指标 |

### 输入数据格式 (JSONL)

每行一个 JSON 对象：

```json
{"timestamp": 1780722000040, "input_length": 586, "output_length": 1,
 "block_ids": [435730272841023862, 3709909376369622945, ...],
 "instance_id": "pod-name"}
```

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `timestamp` | int | 是 | 请求到达时间（epoch ms） |
| `input_length` | int | 是 | 输入 token 数 |
| `output_length` | int | 是 | 输出 token 数（prefill-only 设为 1） |
| `block_ids` | list[int] | 可选 | KV cache block hash ID 列表 |
| `instance_id` | str | 可选 | 原始 pod 名称（用于路由一致性验证） |

### 输出文件

| 文件 | 内容 |
|------|------|
| `simulation_summary.json` | TTFT/TPOT 分位数、吞吐量、并发度、queue_wait、三级命中率 |
| `per_request.csv` | 逐请求: req_id, input_length, ttft_ms, queue_wait_ms, engine/peer/pool_hit |
| `per_iteration.csv` | 逐 iteration: pod, timestamp, latency_ms, batch 组成 |
| `routing_decisions.jsonl` | 逐请求路由决策日志（含概率路由指标，见下方说明） |
| `optimizer/` | Hierarchical cache 配置和分析输出 |

#### routing_decisions.jsonl 字段说明

基础字段（所有 cache_aware 策略均输出）：

| 字段 | 说明 |
|------|------|
| `req_id` | 请求 ID |
| `timestamp` | 路由时刻（仿真时间） |
| `input_length` | 输入 token 长度 |
| `best_engine_id` | 前缀匹配最优引擎 ID |
| `hit_count` | 最优引擎命中 block 数 |
| `match_rate` | 最优引擎命中率 |
| `routed_to` | 实际路由到的引擎 ID |
| `reason` | 路由原因: cache_hit / load_balance_override / cache_miss_fallback |
| `loads` | 路由时刻各节点负载快照 |
| `best_engine_load` | 最优引擎当时负载 |

概率路由额外字段（仅 `--topk-routing` 时输出，小数保留 3 位）：

| 字段 | 说明 |
|------|------|
| `routing_mode` | 固定 "probabilistic" |
| `chosen_p_i` | 被选中节点的前缀命中比例 |
| `chosen_score` | 被选中节点的打分 |
| `chosen_load` | 被选中节点的负载 |
| `l_bar` | 全局平均归一化负载 |
| `topk_size` | TopK 截断后候选数 |
| `total_candidates` | 全部候选节点数 |
| `top1_engine_id` | 最高分节点 ID |
| `top1_score` | 最高分 |
| `top1_hit_count` | 最高分节点命中 block 数 |

## 6. C++ Optimizer 接口

通过 `kvcm_py_optimizer` pybind 模块暴露的 Python API：

```python
import kvcm_py_optimizer as kvcm

# 配置加载
loader = kvcm.HierarchicalReplayConfigLoader()
loader.load("config.json")
config = loader.config()

# 管理器初始化
manager = kvcm.HierarchicalReplayManager(config)
manager.Init()

# 缓存查询（单引擎）
res = manager.GetCacheLocation(engine_id, trace_id, timestamp, block_ids, input_len)
# res.engine_hit_length  — L1 device 命中 token 数
# res.peer_hit_length    — P2P 跨实例命中 token 数
# res.storage_pool_hit_length — 共享存储池命中 token 数

# 跨引擎最优匹配（路由用）
best = manager.ChooseBestEngine(block_ids, timestamp)
# best.engine_instance_id — 最优引擎 ID
# best.hit_count          — 最长前缀匹配 block 数

# 缓存写入
manager.WriteCache(engine_id, trace_id, timestamp, block_ids)
```

### Optimizer 内部参数映射

详细的参数传入链路和容量计算见 `docs/optimizer_parameter_mapping.md`。

关键参数对应：
- `--page-size` → `OptInstanceConfig.block_size_`
- `--kv-bytes-per-token` → `OptInstanceConfig.bytes_per_token_`
- `--hbm-capacity` → `OptTierConfig("hbm").capacity_`（GB → bytes）
- `--mem-capacity` → `OptTierConfig("dram").capacity_`（GB → bytes）
- `--num-p-instances` → N 个 `OptInstanceGroupConfig`

## 7. 测试

```bash
cd schedule_simulator

# 设置 PYTHONPATH（包含 C++ 模块）
export PYTHONPATH=/path/to/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind:$PYTHONPATH

# 运行全部测试（当前 235 个）
python3 -m pytest tests/ -v

# 仅运行不依赖 C++ 模块的测试
python3 -m pytest tests/ -v --ignore=tests/test_hierarchical_pybind.py --ignore=tests/test_hierarchical_cache_adapter.py
```

测试覆盖：基础仿真、PD 分离、多实例路由、写策略、prefetch 策略、pybind 绑定、adapter、Runner 集成、配置桥接、精度验证、block_ids 透传、命中率对比、指标导出、request 级调度、DCA 路由、ChooseBestEngine 优化、硬件参数覆盖、per-pod 统计。

## 8. 开发指南

### 修改 C++ Optimizer 代码

```bash
# 1. 修改 C++ 源码
vim kv_cache_manager/optimizer/manager/hierarchical_replay_manager.cc

# 2. 重新编译
bazel build //kv_cache_manager/optimizer/pybind:kvcm_py_optimizer

# 3. 验证（无需重装，.so 自动更新）
python3 -c "import kvcm_py_optimizer; print(OK)"
```

### 新增路由策略

1. 在 `dispatch/dispatch_policy.py` 中继承 `BasePolicy`
2. 实现 `select_worker(req) -> int` 方法
3. 在 `_create_policy()` 中注册策略名
4. 注意：如果依赖 `update_workload`，需处理 request 级模式的特殊情况

### 新增时延预测器

- **Request 级**：继承 `RequestLevelTimePredictor`，实现 `predict_request_time(uncached, cached)`
- **Iteration 级**：继承 `InferTimePredictor`，实现 `predict_infer_time(batch)`

### 新增 Cache 实现

继承 `PrefixCache`，覆写 `add_to_prefetch_queue` / `match_prefix` / `on_request_complete` 等方法。

## 9. FAQ

**Q: `import kvcm_py_optimizer` 失败？**
A: 确保 PYTHONPATH 包含 `tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind/`。不使用 `--enable-hierarchical` 时不需要该模块。

**Q: `AttributeError: NoneType object has no attribute torch_dtype`？**
A: 模型名未注册到 kunlun_commons。用已注册的模型名（如 `Qwen2.5-3B`）+ `--kv-bytes-per-token` 覆盖。

**Q: DCA 路由所有请求集中到一个 Pod？**
A: 已在当前版本修复。详见 `docs/dca_routing_bug_analysis.md`。

**Q: 仿真很慢（< 50 req/s）？**
A: DCA + hierarchical 模式正常速度约 200-500 req/s（取决于实例数和数据集）。RoundRobin 更快（不做 cache 查询）。

**Q: 如何关闭 L3 共享存储池？**
A: `--pool-capacity 0`

**Q: block_ids 是什么格式？**
A: 每个 block_id 是 int64 哈希值，代表连续 `page_size` 个 token 的 KV cache block。由线上系统产生，`--page-size` 必须与数据源一致（通常为 256）。

## 10. 已知限制

1. **Request 级模式无搬运延时建模**：cache hit 的 token 搬运延时未计入 TTFT（排队建模已实现）
2. **Qwen3.6-Plus 未注册 kunlun_commons**：需用 `Qwen2.5-3B` + `--kv-bytes-per-token 640 --max-num-tokens 999999999` 覆盖
3. **per_iteration.csv 在 request 级模式下为空**
