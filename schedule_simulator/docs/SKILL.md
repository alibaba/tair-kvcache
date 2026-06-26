# Schedule Simulator — 端到端仿真操作手册

> 本文档覆盖从零开始的完整流程：代码拉取 → 编译安装 → 测试验证 → 数据准备 → 仿真执行 → 结果分析。

---

## 0. 环境要求

| 项目 | 要求 |
|------|------|
| OS | Linux (推荐 Ubuntu 22.04+) |
| Python | 3.10+ (当前使用 3.12) |
| Bazel | 6.x (编译 C++ Optimizer pybind) |
| 硬件 | 128核+ 推荐（大规模扫参并行20路） |
| 磁盘 | 仿真数据集约 2-5 GB |

---

## 1. 代码拉取

```bash
# 克隆主仓库
git clone git@github.com:alibaba/tair-kvcache.git
cd tair-kvcache

# 切到仿真器开发分支
git checkout feat/schedule-simulator-dev

# 仿真器代码位于 schedule_simulator/ 子目录
ls schedule_simulator/
#  scripts/  src/  tests/  docs/  pyproject.toml  run_sweep.sh  ...
```

### 仓库结构概览

```
tair-kvcache/                         # 主仓库
├── kv_cache_manager/optimizer/       # C++ 缓存命中建模（RadixTree + P2P + LRU驱逐）
│   └── pybind/                       # → kvcm_py_optimizer.so
├── schedule_simulator/               # Python 调度/时序仿真 ← 核心模块
│   ├── scripts/                      # 可执行脚本
│   │   ├── run_simulation.py         # 主仿真入口
│   │   ├── convert_enriched_to_sim.py # 数据格式转换
│   │   ├── analyze_pod_load.py       # Pod 负载分析
│   │   ├── run_optimizer_standalone.py # Optimizer 独立验证
│   │   ├── prepare_simulation_data.py # 数据准备
│   │   ├── train_predictor.py        # 预测器训练
│   │   └── qwen_latency_predictor.py # Qwen 时延模型
│   ├── src/schedule_simulator/       # 核心源码
│   │   ├── schedule_emulator/        # 调度器仿真（SGLang 兼容）
│   │   ├── infer_time_predictor/     # 推理时延预测
│   │   ├── data/                     # 内置数据/配置
│   │   └── dataset.py                # 数据集加载
│   ├── tests/                        # 284 个测试用例
│   ├── run_sweep.sh                  # 参数扫描脚本（96组实验）
│   └── run_sweep_supplement.sh       # 补充扫描（N=140/180）
└── hisim/                            # HiSim（独立工具，非本模块）
```

---

## 2. 编译 & 安装

### 2.1 编译 C++ Optimizer pybind（必选，用于 hierarchical cache 模式）

```bash
cd tair-kvcache

# Bazel 编译 kvcm_py_optimizer.so
bazel build //kv_cache_manager/optimizer/pybind:kvcm_py_optimizer

# 编译产物路径
ls bazel-bin/kv_cache_manager/optimizer/pybind/kvcm_py_optimizer.so

# 配置 PYTHONPATH 使 Python 可以找到 .so
export PYTHONPATH=$(pwd)/bazel-bin/kv_cache_manager/optimizer/pybind:$PYTHONPATH

# 验证
python3 -c "import kvcm_py_optimizer; print('OK')"
```

### 2.2 安装 schedule_simulator Python 包

```bash
cd schedule_simulator

# 开发模式安装（推荐，代码修改后立即生效）
pip install -e .

# 或指定 full 依赖（包含 kunlun-commons, deepestim）
pip install -e ".[full]"

# 验证安装
python3 -c "from schedule_simulator.schedule_emulator import SGLangScheduleEmulator; print('OK')"
```

### 2.3 依赖列表

| 包名 | 用途 | 是否必需 |
|------|------|---------|
| numpy | 数值计算 | 必需 |
| pandas | 数据分析/CSV导出 | 必需 |
| scipy | 统计计算 | 必需 |
| tqdm | 进度条 | 必需 |
| pytest | 测试框架 | 测试时必需 |
| joblib | 预测器 pkl 加载 | 必需 |
| zstandard | .zst 压缩文件解压 | 数据转换时必需 |
| kunlun-commons | 模型/硬件信息查询 | 可选（可用 --kv-bytes-per-token 替代） |
| deepestim | 迭代级时延预测 | 可选（可用 --ms-per-token 替代） |

---

## 3. 测试验证

```bash
cd schedule_simulator

# 全量测试（284 用例，约 9 秒）
python -m pytest tests/ -q

# 指定单个测试文件
python -m pytest tests/test_binpack_group.py -v

# 仅运行 request-level 相关测试
python -m pytest tests/test_request_level_load_balance.py tests/test_request_level_p2p.py -v
```

**期望结果**：284 passed, 1 warning, 约 9 秒完成。

> 注意：运行测试前需确保 PYTHONPATH 已包含 kvcm_py_optimizer.so 的路径。

---

## 4. 数据准备

### 4.1 数据格式（sim.jsonl）

仿真器输入为 JSONL 格式，每行一个请求：

```json
{
  "timestamp": 1780722000040,
  "input_length": 586,
  "output_length": 1,
  "block_ids": [435730272841023862, 3709909376369622945],
  "instance_id": "pod-name"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | int | epoch 毫秒时间戳（必需） |
| `input_length` | int | 输入 token 数（必需） |
| `output_length` | int | 输出 token 数，prefill-only 场景设为 1（必需） |
| `block_ids` | list[int64] | KV cache block 哈希 ID（可选，启用 cache 时必需） |
| `instance_id` | string | 原始 pod 名称（可选，timeline 模式必需） |

### 4.2 从 enriched.jsonl 转换

线上采集的 enriched.jsonl 需要经过格式转换：

```bash
python3 scripts/convert_enriched_to_sim.py \
    --input /path/to/qwen3.6-plus.enriched.jsonl \
    --output /path/to/sim.jsonl \
    --service-name qwen3.6-plus-2026-04-02-think-model-e1b8
```

转换规则：
- `timestamp`：秒级浮点 → 毫秒级整数（×1000 取整）
- `block_ids`：hex 字符串列表 → int64 列表（`int(hex, 16) % 2^63`）
- `instance_id`：从 `pods[0]` 提取
- `output_length`：固定为 1（prefill-only）

如果输入已是单服务数据无需过滤，加 `--no-filter`：

```bash
python3 scripts/convert_enriched_to_sim.py \
    --input /path/to/h21_e1b8.jsonl \
    --output /path/to/sim.jsonl \
    --no-filter
```

### 4.3 现有数据集路径（远程服务器）

| 数据集 | 路径 | 请求数 |
|--------|------|--------|
| H21 32k-256k 全量 | `/sgl-workspace/claude_workspace/data/h21_32_256k_full.jsonl` | 491,901 |
| Qwen3.6 预测器 | `/sgl-workspace/claude_workspace/data/qwen36_predictor/qwen36_prefill_predictor_qwen_lookup.pkl` | — |

---

## 5. 端到端仿真执行

### 5.1 最简示例（随机数据，无需真实数据集）

```bash
python3 scripts/run_simulation.py \
    --num-prompts 200 \
    --request-level \
    --ms-per-token 0.1 \
    --num-p-instances 5 \
    --routing round_robin \
    --output-dir ./sim_results
```

### 5.2 真实数据 + Hierarchical Cache（完整模式）

```bash
export PYTHONPATH=/path/to/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind:$PYTHONPATH

python3 scripts/run_simulation.py \
    --dataset /path/to/h21_32_256k_full.jsonl \
    --num-prompts 491901 \
    --request-level \
    --predictor-pkl /path/to/qwen36_prefill_predictor_qwen_lookup.pkl \
    --num-p-instances 16 \
    --routing bin_pack \
    --pods-per-group 4 \
    --bin-capacity 5 \
    --enable-hierarchical \
    --enable-p2p \
    --kv-bytes-per-token 46080 \
    --hbm-capacity 533 \
    --mem-capacity 340 \
    --page-size 2048 \
    --data-block-size 256 \
    --pool-capacity 0 \
    --quiet \
    --output-dir ./sim_results
```

### 5.3 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | None | JSONL 输入文件路径 |
| `--num-prompts` | 100 | 请求数量 |
| `--num-p-instances` | 1 | Prefill 实例数 |
| `--routing` | round_robin | 路由策略：random / round_robin / power_of_two / cache_aware / direct_cache_aware / bin_pack |
| `--request-level` | False | 启用 request-level 调度（推荐） |
| `--predictor-pkl` | None | 时延预测器 pkl 文件 |
| `--ms-per-token` | None | 恒定时延预测（ms/token，简易模式） |
| `--enable-hierarchical` | False | 启用 C++ Optimizer 缓存集成 |
| `--enable-p2p` / `--no-p2p` | True | P2P 跨实例缓存读取 |
| `--page-size` | 1 | 缓存 page 大小（token 数） |
| `--data-block-size` | None | 数据集中 block_ids 的粒度（如 256），与 page-size 不同时自动转换 |
| `--kv-bytes-per-token` | None | KV cache 每 token 字节数 |
| `--hbm-capacity` | None | HBM 容量（GB） |
| `--write-policy` | write_through | 缓存写策略：write_through / write_back / write_through_selective |
| `--pool-capacity` | None | L3 共享存储池容量（设 0 关闭） |
| `--pods-per-group` | None | BinPack 分组大小 |
| `--bin-capacity` | None | BinPack 每组容量上限 |
| `--output-dir` | ./sim_results | 结果输出目录 |
| `--export-routing-decisions` | False | 导出路由决策日志 |
| `--quiet` | False | 静默模式（减少日志） |
| `--pod-prefix` | None | 按 pod 前缀过滤输入数据 |
| `--timeline-mode` | disabled | Timeline 回放模式：disabled / route_only / route_and_latency / latency_only |

### 5.4 BinPack 路由专用参数

BinPack 分组路由需额外指定：
- `--pods-per-group K`：每组 K 个实例（如 4）
- `--bin-capacity C`：每组同时处理请求上限（如 5）

示例：16 实例分 4 组，每组容量 5：
```bash
--routing bin_pack --pods-per-group 4 --bin-capacity 5
```

### 5.5 Timeline 回放模式

使用线上真实路由决策进行时延仿真：
```bash
--timeline-mode route_and_latency  # 复用线上路由 + 仿真时延
--timeline-mode route_only         # 仅复用线上路由，不计算时延
--timeline-mode latency_only       # 使用仿真路由，复用线上时延
```

---

## 6. 参数扫描

大规模实验使用内置的 sweep 脚本：

```bash
# 主扫描：96 组实验（N x k x cap + cache-aware baselines），并行 20 路
bash run_sweep.sh

# 补充扫描：N=140/180 共 32 组
bash run_sweep_supplement.sh
```

扫描参数空间：
- **N（实例数）**：16, 32, 40, 60, 80, 100, 120, 168（+补充 140, 180）
- **k（pods-per-group）**：4, 8, 12, 16, 20
- **cap（bin-capacity）**：3, 5, 7

结果按 `sweep_results/bin_pack_N{n}_k{k}_cap{c}/` 和 `sweep_results/cache_aware_N{n}/` 组织。

---

## 7. 输出结果

### 7.1 输出文件

| 文件 | 内容 |
|------|------|
| `simulation_summary.json` | TTFT/TPOT/吞吐量/排队等待 + 三级缓存命中率 |
| `per_request.csv` | 每请求：ttft_ms, queue_wait_ms, engine/peer/pool_hit |
| `per_iteration.csv` | 每迭代：pod, latency_ms, batch 组成 |
| `per_pod_stats.csv` | 每 pod 统计：请求数、命中率 |
| `routing_decisions.jsonl` | 路由决策日志（需 --export-routing-decisions） |

### 7.2 关键指标

```json
{
  "avg_ttft_ms": 245.3,
  "p99_ttft_ms": 1023.5,
  "throughput_rps": 2015.8,
  "avg_queue_wait_ms": 12.4,
  "engine_hit_ratio": 0.623,
  "peer_hit_ratio": 0.089,
  "pool_hit_ratio": 0.0,
  "total_hit_ratio": 0.712
}
```

---

## 8. Optimizer 独立验证

绕过调度器，直接用 C++ Optimizer 获取理论命中率（用于对照仿真器结果是否准确）：

```bash
python3 scripts/run_optimizer_standalone.py \
    --dataset /path/to/sim.jsonl \
    --num-pods 16 \
    --hbm-capacity 533 \
    --page-size 2048 \
    --data-block-size 256 \
    --routing round_robin
```

---

## 9. Pod 负载分析

分析输入数据中各 Pod 的请求分布：

```bash
python3 scripts/analyze_pod_load.py \
    --input /path/to/sim.jsonl \
    --top 20
```

---

## 10. 常见问题

| 问题 | 解决方案 |
|------|---------|
| `import kvcm_py_optimizer` 失败 | 确保 `PYTHONPATH` 包含 `bazel-bin/.../pybind/`。不使用 `--enable-hierarchical` 时不需要该模块 |
| `NoneType has no attribute torch_dtype` | 模型名未注册到 kunlun_commons，改用 `--kv-bytes-per-token` 手动指定 |
| DCA 路由请求集中到一个 Pod | 已修复，详见 `docs/dca_routing_bug_analysis.md` |
| 仿真速度很慢 (< 50 req/s) | DCA + hierarchical 正常约 200-500 req/s，RoundRobin 更快 |
| `--page-size` 与 `--data-block-size` 的关系 | 必须同时指定。page-size=2048 + data-block-size=256 触发自动跨粒度转换 |
| 全量测试不通过 | 检查 PYTHONPATH 是否包含 .so 路径，确认 bazel build 已完成 |

---

## 11. 完整操作流程速查

```
1. git clone → checkout feat/schedule-simulator-dev
2. bazel build → 编译 kvcm_py_optimizer.so
3. export PYTHONPATH → 配置 .so 路径
4. cd schedule_simulator && pip install -e .
5. python -m pytest tests/ -q → 验证 284 用例全部通过
6. 准备 sim.jsonl（或使用 convert_enriched_to_sim.py 转换）
7. python3 scripts/run_simulation.py ... → 执行仿真
8. 查看 sim_results/simulation_summary.json → 分析结果
```
