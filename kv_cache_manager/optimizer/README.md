# KVCacheManager Optimizer

## 概述

KVCacheManager Optimizer 是一个独立的缓存优化分析模块，通过回放 trace 数据来模拟缓存读写操作，评估不同驱逐策略和配置对缓存命中率的影响，并为 KVCacheManager 主程序提供参数优化能力。

核心功能包括：
- Trace 数据的回放和模拟
- 多种驱逐策略的模拟和对比
- 缓存命中率的实时统计和分析
- Radix Tree 索引结构的可视化

## 动机

在大语言模型（LLM）推理场景中，KV Cache 的管理对系统性能至关重要。不同的驱逐策略、缓存容量配置和存储层级设置会显著影响缓存命中率和整体推理效率。Optimizer 模块的设计动机包括：

1. **策略评估**：在生产环境部署前，通过 trace 回放评估不同驱逐策略的效果，选择最优配置
2. **参数调优**：分析缓存访问模式，为驱逐策略参数（如采样率、TTL 等）提供优化建议
3. **性能预测**：预测不同容量配置下的缓存命中率，辅助资源规划
4. **问题诊断**：通过 Radix Tree 可视化和详细统计，帮助理解缓存行为和性能瓶颈

## 特性与架构

### 核心特性

- **多种驱逐策略**：支持 LRU、RandomLRU、LeafAwareLRU、TTL 等驱逐算法
- **TTL 过期机制**：当前采用 V1 语义，仅 `POLICY_TTL` 执行 TTL 过期物理清理
- **分层存储**：支持多层级存储配置，目前功能不完备
- **Trace 回放**：支持 Publisher Log、Qwen Bailian 等多种 trace 格式
- **详细统计**：提供命中率、缓存使用情况等详细统计
- **灵活配置**：通过 JSON 配置文件灵活配置实例、存储和策略
- **可视化分析**：支持 Radix Tree 可视化和命中率图表生成

标准策略配置、multi-instance replay、trace schema 和命中率口径见 [docs/strategy_config.md](docs/strategy_config.md)。标准版中 `HitRate` 统一表示整体 token hit rate，即 `HitTokens / InputTokens`；local/remote 只作为 trace `block_mask` 与 optimizer 模拟命中的诊断拆分，不作为标准结论维度。传入 optimizer config 的 Python 入口统一使用配置中的 `output_result_path`；`multi_instance_replay` 不读取完整 config，使用显式 `--output-dir`。标准 `get` trace 必须包含 `input_len`；外部只有请求级日志时可使用 `type=request`，optimizer 会按 `trace_replay.write_delay_ns` 在内部调度 delayed write；已经拆分好的 `get` / `write` trace 仍然支持。

### 架构设计

```
OptimizerManager (核心协调器)
    ├── OptEvictionManager (驱逐管理器)
    ├── OptIndexerManager (索引管理器)
    └── OptimizerRunner (Trace 执行器)
        ↓
    ├── Eviction Policies (驱逐策略)
    ├── RadixTreeIndex (索引)
    └── Trace Converter (转换器)
        ↓
    HitAnalysis (结果分析)
```


### 驱逐策略

**LRU (Least Recently Used)**
- 维护双向链表记录块的访问顺序，最近访问的块在链表头部，最久未访问的块在链表尾部

**RandomLRU**
- 结合随机采样和 LRU 策略，从缓存中随机采样一定数量的块，选择最久未访问的块进行驱逐

**LeafAwareLRU**
- 在 LRU 基础上增加了对叶子节点的感知，优先驱逐叶子节点中的块

**TTL (Time-To-Live)**
- 两阶段驱逐：先清走所有 TTL 过期的 block，若容量仍超限则按 `last_access_time` 从最旧开始兜底驱逐
- `fallback_on_pressure`（默认 true）：关闭后退化为纯 TTL，只回收过期 block，无容量兜底

#### TTL 过期机制（V1）

当前实现语义：

- 仅 `eviction_policy_type: "ttl"` 的实例会执行读写前 TTL 过期物理清理。
- 非 TTL 策略（`lru` / `random_lru` / `leaf_aware_lru`）下，TTL 视为不存在（既不做前置清理，也不做逻辑过期判定）。

| 使用模式 | 配置方式 | 驱逐行为 |
|---|---|---|
| 纯 LRU 容量管理 | `eviction_policy_type: "lru"` | LRU 按访问时间驱逐 |
| TTL 优先 + 容量兜底 | `eviction_policy_type: "ttl"` + `fallback_on_pressure: true` | 先清所有过期 block，不够则按链表尾部兜底 |
| 纯时间驱逐（无兜底） | `eviction_policy_type: "ttl"` + `fallback_on_pressure: false` | 只清过期 block，容量不足不管 |

**TTL 行为规则**：
- `default_block_ttl_seconds`：在 instance group 级别配置，`0` 表示组级禁用 TTL。
- `ttl_refresh_on_read`：控制读命中是否刷新 TTL 锚点；默认 `true`（Sliding TTL），`false` 时读不续命。
- 组级禁用 TTL 时，请求级 `ttl_seconds > 0` 不生效（写入路径会强制关闭 TTL）。
- 写入时，`ttl_anchor_time` 重置为写入时间，TTL 从该锚点开始计时。
- 读取时（`PrefixQuery`），`last_access_time` 总是刷新；仅在 `ttl_refresh_on_read=true` 时刷新 `ttl_anchor_time`。
- 读写请求执行前，会先进行一次 TTL 过期块物理清理，并由 `CleanEvictedBlocks` 统一做节点清理。
- 写入完成后，`CheckAndEvict` 负责容量驱逐；当 `fallback_on_pressure=false` 时跳过容量驱逐。
- `POLICY_TTL` 过期清理已优化为基于最小过期时间堆（min-heap）的增量回收，避免每次请求全链表扫描。

**TTL 生命周期统计注意事项（重要）**：
- TTL 模式下的 Block 生命周期统计（`birth_time_us/death_time_us/lifespan_us`）仅反映系统实际处理事件的时间点，不代表真实过期时刻。
- 由于当前采用写后惰性驱逐（Lazy Eviction），`death_time_us` 记录的是被系统清理/处理的时间，而非严格的 `last_access_time + ttl`。
- 因此 TTL 场景下的生命周期数据仅用于相对趋势分析，不建议用于精确时长评估或跨策略精确对比。

**Per-request TTL 覆盖**（通过 `WriteCache` API）：

| `ttl_seconds` 值 | 含义 |
|---|---|
| `0`（默认） | 使用 group 的 `default_block_ttl_seconds` |
| `-1` | 禁用 TTL，该 block 永不过期 |
| `>0` | 自定义 TTL（秒）；若 group 已禁用 TTL（`default_block_ttl_seconds=0`），该值会被忽略 |

### Trace 类型

```
OptimizerSchemaTrace (基类)
    ├── GetLocationSchemaTrace (读操作)
    └── WriteCacheSchemaTrace (写操作)
```

**支持的 Trace 格式**
- **Publisher Log**：KVCacheManager Event Publisher 日志，区分读写请求
- **Qwen Bailian**：百炼开源数据集格式，转换后强制区分读写请求

## 快速开始

### 步骤1: 转换trace为标准格式

```bash
cd tools/trace_converter

# 安装依赖 (首次使用)
pip install -r requirements.txt

# 转换trace
python trace_converter.py \
    -i /path/to/your_trace.jsonl \
    -o /path/to/optimizer_trace.jsonl \
    -f qwen_bailian \
    --mode optimizer
```

### 步骤2: 编译 Optimizer

```bash
bazel build \
    //kv_cache_manager/optimizer:optimizer_main \
    //kv_cache_manager/optimizer/analysis/script:optimizer_run
```

### 步骤3: 创建配置文件

创建 JSON 配置文件。下面是非分层 LRU 的最小可用配置；trace 中的 `instance_id` 必须能匹配这里的 `instances[].instance_id`。

```json
{
    "trace_file_path": "/path/to/optimizer_trace.jsonl",
    "output_result_path": "/path/to/output/result/",
    "eviction_params": {
        "eviction_mode": 1,
        "eviction_batch_size_per_instance": 100
    },
    "instance_groups": [
        {
            "group_name": "instance_group_01",
            "quota_capacity": 12000,
            "used_percentage": 1.0,
            "tier_strategy": {
                "hierarchical_eviction_enabled": false,
                "write_mode": "write_through",
                "access_propagation_enabled": true,
                "promote_enabled": true,
                "selective_write_threshold": 2
            },
            "default_block_ttl_seconds": 0,
            "ttl_refresh_on_read": true,
            "storages": [],
            "instances": [
                {
                    "instance_id": "instance-a",
                    "block_size": 16,
                    "bytes_per_token": 512,
                    "eviction_policy_type": "lru",
                    "eviction_policy_params": {
                        "sample_rate": 1.0,
                        "shard_count": 1,
                        "sample_times": 32,
                        "eviction_amplification_factor": 1.0
                    }
                }
            ]
        }
    ]
}
```


### 步骤4: 运行Optimizer

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json
```

运行完成后，会在 `output_result_path` 指定的目录下生成：

- `{instance_id}_hit_rates.csv` - 每个 instance 的命中率数据

需要画命中率时序图时加 `--draw-chart`：

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json \
    --draw-chart
```

常见入口速查：

| 需求 | 推荐入口 |
|---|---|
| 单次回放、输出 `*_hit_rates.csv` | `optimizer_run -c config.json` |
| 单次回放并画时序图 | `optimizer_run -c config.json --draw-chart` |
| 无限容量理论命中率 | config 中将 `quota_capacity` 或 tier `capacity` 设为 `-1` 后运行 `optimizer_run` |
| 非分层容量 Pareto | `tradeoff -c config.json --num-points 30` |
| 多驱逐策略 Pareto | `tradeoff -c config.json --eviction-policies lru random_lru leaf_aware_lru ttl` |
| 每个 pod/instance 独立缓存回放 | `multi_instance_replay --trace-dir ... --output-dir ...` |
| 导出生命周期 CSV | `optimizer_run -c config.json --export-lifecycle` |
| 分析 lifecycle 图表 | `analyze_lifecycle -i <lifecycle.csv 或目录>` |
| 导出/可视化 RadixTree | `export_tree -c config.json --show-hot-paths` |

### 可视化分析

Optimizer 模块提供多种可视化分析工具，用于分析缓存性能、命中率和 Radix Tree 结构。

#### 命中率随时间变化图表

运行optimizer，分析trace并绘制多实例缓存分析图表，展示所有 instance 的存储容量总和以及各自命中率随时间的变化。

**注意**: 配置文件中的 `trace_file_path` 必须是标准格式trace文件。

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json \
    --draw-chart
```

**输出**：`{output_result_path}/timeseries/multi_instance_cache_analysis.png`

#### Radix Tree 可视化

导出并可视化前缀树结构，统计并展示热节点以及所属节点的前缀路径。
具体配置见 `analysis/script/run/export_tree.py`。

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:export_tree -- \
    -c /path/to/config.json
```

**输出**：
- `{output_result_path}/radix_tree/{instance_id}_radix_tree.json` - Radix Tree 导出数据
- `{output_result_path}/radix_tree/{instance_id}_radix_tree.png` - Radix Tree 可视化图表

#### Pareto 容量-命中率曲线分析

统一入口是 `tradeoff`。它先用无限容量 warmup 获取理论命中率和最大缓存量，然后按容量点回放 optimizer，绘制容量与命中率的 Pareto 曲线。

> **适用范围**：Tradeoff 只适用于非分层模式。在分层模式下，容量扫描只修改 `quota_capacity`，不会修改各 tier 的 `storages[i].capacity`，因此不能代表真实分层容量权衡。

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/config.json \
    --num-points 30 \
    --min-capacity-ratio 1e-4 \
    --hit-rate-type total \
    --max-workers 4
```

**参数说明**：
- `-c, --config` - 配置文件路径
- `--num-points` - 每个策略最多运行的容量采样点数量（默认 30），达到 99% 理论命中率后提前停止
- `--min-capacity-ratio` - 最小容量点相对阈值（默认 `1e-4`）
- `--hit-rate-type` - 命中率类型：total/local/remote/all（默认 total）
- `--max-workers` - 并行执行的最大线程数（默认 4）
- `--plot-title` - 覆盖图标题

**输出**：`{output_result_path}/pareto/pareto_curve_{hit_rate_type}.png`

#### 多策略对比分析

通过 `--eviction-policies` 对比多个驱逐策略。每个策略独立 warmup、独立计算理论命中率，并独立在达到 99% 理论命中后停止。

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/config.json \
    --eviction-policies lru random_lru leaf_aware_lru ttl \
    --num-points 30 \
    --hit-rate-type total \
    --max-workers 4
```

**参数说明**：
- `-c, --config` - 配置文件路径
- `--eviction-policies` - 要对比的驱逐策略列表（默认 lru random_lru leaf_aware_lru ttl）
- `--num-points` - 每个策略最多运行的容量采样点数量（默认 30），达到 99% 理论命中率后提前停止
- `--hit-rate-type` - 命中率类型：total/local/remote/all（默认 total）
- `--max-workers` - 并行执行的最大线程数（默认 4）

**输出**：`{output_result_path}/pareto/multi_policy_{hit_rate_type}.png`

图形规则：X 轴为 `Capacity (GB)`，Y 轴为 `HitRate (%)`；曲线从 `(0 GB, 0%)` 开始，只画上升段；95% 和 99% 理论命中率用插值交点、虚线和标签标出。若某个容量点出现下降，会从图上剔除并在日志中打印，原始 CSV 保留。

### Python 接口示例

```python
from kv_cache_manager.optimizer import OptimizerConfigLoader, OptimizerLoader, OptimizerManager

# 加载配置
config_loader = OptimizerConfigLoader()
config = config_loader.Load("/path/to/config.json")

# 创建优化器
optimizer = OptimizerManager(config)
optimizer.Init()

# 运行
optimizer.DirectRun()

# 分析结果
optimizer.AnalyzeResults()

# 单次读写操作（需要指定instance_id）
write_res = optimizer.WriteCache("instance_id", "trace_001", timestamp, block_ids)
read_res = optimizer.GetCacheLocation("instance_id", "trace_002", timestamp, block_ids, block_mask,
                                      input_len=real_prompt_tokens)

# 写入时指定 TTL（可选）
write_res = optimizer.WriteCache("instance_id", "trace_003", timestamp, block_ids,
                                  ttl_seconds=0)     # 使用 group 默认
write_res = optimizer.WriteCache("instance_id", "trace_004", timestamp, block_ids,
                                  ttl_seconds=-1)    # 禁用 TTL，永不过期
write_res = optimizer.WriteCache("instance_id", "trace_005", timestamp, block_ids,
                                  ttl_seconds=300)   # 自定义 300 秒

# 清空缓存（保留统计）
optimizer.ClearCache("instance_id")        # 清空指定实例
optimizer.ClearAllCaches()                 # 清空所有实例

# 清空缓存并重置统计
optimizer.ClearCacheAndResetStats("instance_id")  # 清空指定实例并重置统计
optimizer.ClearAllCachesAndResetStats()           # 清空所有实例并重置统计
```

### 配置参数说明

| 参数 | 说明 |
|------|------|
| eviction_mode | 驱逐模式：1=GROUP_ROUGH, 2=INSTANCE_ROUGH, 3=INSTANCE_PRECISE |
| eviction_policy_type | 驱逐策略类型：lru、random_lru、leaf_aware_lru、ttl |
| quota_capacity | 非分层模式下的 group 总容量，单位 GB；`-1` 表示无限容量，不触发容量驱逐 |
| storages[].capacity | 分层模式下单个 tier 的容量，单位 GB；`-1` 表示该 tier 无限容量 |
| tier_strategy | 多层读写策略包，包含分层驱逐开关、写入模式、读访问传播、promote 和 selective write 阈值；顶层字段是所有相邻 tier edge 的默认策略 |
| tier_strategy.write_mode | 写入/下沉模式：`write_through`、`cascading`、`write_through_selective` |
| tier_strategy.access_propagation_enabled | 读命中上层副本时是否刷新下层副本访问时间；它不是写入模式 |
| tier_strategy.tier_flows | 相邻 tier edge 的策略覆盖；未覆盖 edge 继承 `tier_strategy` 默认策略 |
| bytes_per_token | 单 token KV 大小；Python 分析脚本用它和 `block_size` 将 block 容量换算为 GB |
| default_block_ttl_seconds | instance group 级别的默认 TTL（秒），0 = 关闭 TTL |
| ttl_refresh_on_read | instance group 级别 TTL 续命开关：true=读续命，false=固定窗口 |
| fallback_on_pressure | TTL 策略参数：过期不够时是否按 LRU 兜底（默认 true） |

## 使用案例

下面的案例都假设输入已经是标准 optimizer JSONL trace。标准结论使用 token hit rate，即 CSV 最后一行的 `AccHitRate = AccHitTokens / AccInputTokens`。示例中的 JSON 多数只展示 `instance_groups[]` 中的一项，完整 config 仍需要顶层 `trace_file_path`、`output_result_path` 和 `eviction_params`。

### 案例 1：全局池化无限容量理论命中率

用于回答“如果容量无限，当前 trace 理论上最多能命中多少”。非分层全局池化配置中把 `quota_capacity` 设为 `-1`：

```json
{
    "group_name": "global_pool",
    "quota_capacity": -1,
    "used_percentage": 1.0,
    "tier_strategy": {
        "hierarchical_eviction_enabled": false,
        "write_mode": "write_through",
        "access_propagation_enabled": true,
        "promote_enabled": true,
        "selective_write_threshold": 2
    },
    "storages": [],
    "instances": [
        {
            "instance_id": "global",
            "block_size": 256,
            "bytes_per_token": 512,
            "eviction_policy_type": "lru",
            "eviction_policy_params": {
                "sample_rate": 1.0,
                "shard_count": 1,
                "sample_times": 32,
                "eviction_amplification_factor": 1.0
            }
        }
    ]
}
```

运行：

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/global_pool_unlimited.json
```

结果看 `{output_result_path}/global_hit_rates.csv` 最后一行的 `AccHitRate`、`AccInputTokens`、`AccHitTokens`。
如果 trace 中使用真实 pod 名作为 `instance_id`，需要把上面示例里的 `instances[].instance_id` 同步改成对应名称。

### 案例 2：单层有限容量回放

用于评估给定容量下的实际 token 命中率。非分层模式只需要设置 `quota_capacity`，单位是 GB：

```json
{
    "group_name": "finite_pool",
    "quota_capacity": 742.18,
    "used_percentage": 1.0,
    "tier_strategy": {
        "hierarchical_eviction_enabled": false,
        "write_mode": "write_through",
        "access_propagation_enabled": true,
        "promote_enabled": true,
        "selective_write_threshold": 2
    },
    "storages": []
}
```

`block_size * bytes_per_token` 决定一个 block 对应的 KV 容量；Pareto 图和容量输出都依赖这个换算。

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/finite_pool.json \
    --draw-chart
```

输出：

- `{output_result_path}/*_hit_rates.csv`
- `{output_result_path}/timeseries/multi_instance_cache_analysis.png`

### 案例 3：HBM + DRAM 分层回放，write-through，读不更新下层

用于模拟线上多层缓存。`hierarchical_eviction_enabled=true` 时，每层按 `storages[].capacity` 独立驱逐；此时 `quota_capacity` 只是保留字段，不用于每层驱逐。

```json
{
    "group_name": "tiered_pool",
    "quota_capacity": 1,
    "used_percentage": 1.0,
    "tier_strategy": {
        "hierarchical_eviction_enabled": true,
        "write_mode": "write_through",
        "access_propagation_enabled": false,
        "promote_enabled": true,
        "selective_write_threshold": 2
    },
    "storages": [
        {
            "unique_name": "hbm",
            "storage_type": "hbm",
            "band_width_mbps": 20000,
            "priority": 0,
            "capacity": 1167
        },
        {
            "unique_name": "dram",
            "storage_type": "dram",
            "band_width_mbps": 20000,
            "priority": 1,
            "capacity": 1070.4
        }
    ]
}
```

运行：

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/tiered_hbm_dram.json \
    --draw-chart
```

分层结果看 `Tier<N>(name)_HitTokens`、`AccTier<N>(name)_HitRate` 和 `Tier<N>(name)_BlockNum`，图在 `{output_result_path}/timeseries/per_tier_timeseries.png`。

### 案例 4：三层 HBM + DRAM + L3，前两层 write-through，DRAM 到 L3 cascading

当每条 edge 策略不一致时，用 `tier_flows` 覆盖默认策略：

```json
{
    "tier_strategy": {
        "hierarchical_eviction_enabled": true,
        "write_mode": "write_through",
        "access_propagation_enabled": false,
        "promote_enabled": true,
        "selective_write_threshold": 2,
        "tier_flows": [
            {
                "from_tier": "hbm",
                "to_tier": "dram",
                "write_mode": "write_through",
                "access_propagation_enabled": false,
                "promote_enabled": true
            },
            {
                "from_tier": "dram",
                "to_tier": "l3",
                "write_mode": "cascading",
                "access_propagation_enabled": false,
                "promote_enabled": true
            }
        ]
    },
    "storages": [
        {"unique_name": "hbm", "storage_type": "hbm", "band_width_mbps": 20000, "priority": 0, "capacity": 1167},
        {"unique_name": "dram", "storage_type": "dram", "band_width_mbps": 20000, "priority": 1, "capacity": 960},
        {"unique_name": "l3", "storage_type": "ssd", "band_width_mbps": 20000, "priority": 2, "capacity": 2048}
    ]
}
```

`tier_strategy` 顶层字段是默认 edge 策略；`tier_flows` 只覆盖指定相邻 edge。

### 案例 5：每个 pod 独立缓存回放并聚合全局命中率

当线上每个 pod/实例独立缓存时，用 `multi_instance_replay`。输入目录下每个 JSONL 文件只能包含一个 `instance_id`。

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:multi_instance_replay -- \
    --trace-dir /path/to/pod_traces \
    --trace-glob "*.jsonl" \
    --output-dir /path/to/pod_replay_output \
    --l1-capacity 349.52 \
    --l2-capacity 614.4 \
    --block-size 1024 \
    --bytes-per-token 163840 \
    --eviction-policy lru \
    --default-tier-write-mode write_through \
    --disable-tier-access-propagation \
    --enable-promote \
    --max-workers 16 \
    --window-seconds 60
```

输出：

- `<output_dir>/<instance_id>_hit_rates.csv`：每个 pod 的回放结果
- `<output_dir>/aggregate/instance_aggregate.csv`：每个 pod 汇总
- `<output_dir>/aggregate/global_aggregate.csv`：所有 pod 聚合后的整体命中率
- `<output_dir>/aggregate/global_window_hit_rates.csv`：窗口级整体命中率

当前 `multi_instance_replay` CLI 直接支持 L1/L2 两层。需要 L3 或更复杂拓扑时，使用完整 optimizer config 或扩展脚本。

### 案例 6：容量 Pareto 图

非分层容量扫描使用 `tradeoff`。脚本先跑无限容量 warmup 获取理论命中率，再生成容量点；达到 99% 理论命中后停止。

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/global_pool_config.json \
    --num-points 30 \
    --min-capacity-ratio 1e-4 \
    --hit-rate-type total \
    --max-workers 8 \
    --plot-title "service-a Pareto"
```

多策略对比：

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/global_pool_config.json \
    --eviction-policies lru random_lru leaf_aware_lru ttl \
    --num-points 30 \
    --hit-rate-type total \
    --max-workers 8
```

输出：

- `{output_result_path}/pareto/pareto_curve_total.png`
- `{output_result_path}/pareto/multi_policy_total.png`

图上会标出 95%/99% 理论命中对应容量；下降点只从图上剔除，原始 CSV 不改。

### 案例 7：Block 生命周期分析

先在回放时导出 lifecycle CSV：

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json \
    --export-lifecycle
```

再生成生命周期统计和图：

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:analyze_lifecycle -- \
    -i /path/to/output_result_path
```

只打印统计、不画图：

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:analyze_lifecycle -- \
    -i /path/to/output_result_path \
    --stats-only
```

### 案例 8：RadixTree 热点路径排查

用于看热点前缀、叶子节点和已缓存/已驱逐 block：

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:export_tree -- \
    -c /path/to/config.json \
    --show-hot-paths \
    --hot-nodes 50 \
    --show-blocks
```

输出：

- `{output_result_path}/radix_tree/{instance_id}_radix_tree.json`
- `{output_result_path}/radix_tree/{instance_id}_hot_paths.png`

### 案例 9：结果读取口径

标准报告只看 token hit rate：

| 指标 | 说明 |
|---|---|
| `InputTokens` / `AccInputTokens` | 当前 / 累计输入 token 数 |
| `HitTokens` / `AccHitTokens` | 当前 / 累计命中 token 数 |
| `HitRate` | 当前请求 token 命中率 |
| `AccHitRate` | 累计 token 命中率，主要结论看这一列 |
| `ReadBlocks` / `HitBlocks` | block 级诊断字段，不作为最终命中率口径 |
| `Tier<N>(name)_HitTokens` | 分层命中 token 数 |
| `AccTier<N>(name)_HitRate` | 分层累计 token 命中率 |


## Trace 输入格式

### 概述

Optimizer 只接受标准格式的trace文件。使用独立的Python工具将各种trace格式转换为标准格式。

### 标准格式

Optimizer 支持两种标准 trace 类型：

- **GetLocationSchemaTrace**: 读操作 (prefill阶段)
- **WriteCacheSchemaTrace**: 写操作 (decode阶段)

**推荐**: 所有Optimizer输入统一使用Get+Write格式以保留精确的读写时序。
标准 `get` trace 必须包含 `input_len`；`keys` 只能包含完整 block key，不足一个 block 的尾部 token 不写入 `keys`，但仍计入 `InputTokens`。

---

### 转换工具

使用独立的Python工具转换各种格式 (无需bazel):

```bash
# 进入工具目录
cd tools/trace_converter

# 安装依赖 (首次使用)
pip install -r requirements.txt

# 转换trace为Optimizer格式
python trace_converter.py \
    -i your_trace.log \
    -o optimizer_trace.jsonl \
    -f <format> \
    --mode optimizer

```

**支持的格式**:
- `publisher_log`: KVCacheManager Event Publisher日志
- `qwen_bailian`: Qwen Bailian开源数据集
- `text`: 文本对话 (需要指定--tokenizer-path)

**自动发现 Converter**:

系统会自动扫描并发现所有可用的 converter：

```bash
# 使用内置 converter
python3 trace_converter.py -i input.jsonl -o output.jsonl -f qwen_bailian

# 使用自定义 converter
python3 trace_converter.py -i input.jsonl -o output.jsonl -f custom \
    --converter-module /path/to/custom_converter.py
```

详见: [Trace Converter文档](tools/trace_converter/README.md)

---

### 配置文件

**新版配置**:
```json
{
    "trace_file_path": "/path/to/optimizer_trace.jsonl",
    "output_result_path": "/path/to/output",
    "eviction_params": {
        "eviction_mode": 1,
        "eviction_batch_size_per_instance": 100
    },
    "instance_groups": [
        {
            "group_name": "instance_group_01",
            "quota_capacity": 12000,
            "used_percentage": 1.0,
            "tier_strategy": {
                "hierarchical_eviction_enabled": false,
                "write_mode": "write_through",
                "access_propagation_enabled": true,
                "promote_enabled": true,
                "selective_write_threshold": 2
            },
            "default_block_ttl_seconds": 0,
            "storages": [],
            "instances": [
                {
                    "instance_id": "instance",
                    "block_size": 16,
                    "bytes_per_token": 512,
                    "eviction_policy_type": "lru"
                }
            ]
        }
    ]
}
```

**注意**: 
- `trace_file_path` 必须是标准格式文件

---

### 使用示例

完整流程:

```bash
# 步骤1: 转换trace
cd tools/trace_converter
python trace_converter.py \
    -i /path/to/qwen_trace.jsonl \
    -o /path/to/optimizer_trace.jsonl \
    -f qwen_bailian \
    --mode optimizer

# 步骤2: 运行Optimizer
cd ../..
bazel run //kv_cache_manager/optimizer:optimizer_main -- /path/to/config.json
```

---

### 添加自定义 Trace Converter

如果需要支持新的 trace 格式,在Python工具中添加新的converter:

1. **创建Converter类**: 在任意目录创建新文件
   ```python
   from converters.base import BaseConverter
   
   class MyCustomConverter(BaseConverter):
       def convert(self, input_file: str, output_file: str) -> int:
           # 实现转换逻辑
           pass
   ```

2. **使用 - 方式1 (自动扫描目录)**:
   ```bash
   # 将converter文件放到指定目录，自动发现所有继承BaseConverter的类
   python trace_converter.py -i input.log -o output.jsonl -f my_custom \
       --converter-dir /path/to/your/converters
   ```

3. **使用 - 方式2 (显式注册文件)**:
   ```bash
   # 直接指定converter文件和类名
   python trace_converter.py -i input.log -o output.jsonl -f my_custom \
       --converter-module /path/to/my_custom_converter.py:MyCustomConverter
   ```

**无需修改 `trace_converter.py` 源码** Converter会根据类名自动推断format名称：
- `MyCustomConverter` → `my_custom`
- `QwenBailianConverter` → `qwen_bailian`

---

### 相关文档

- [Trace Converter工具文档](tools/trace_converter/README.md) - Python转换工具详细说明
