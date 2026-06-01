# KVCacheManager Optimizer 架构文档

## 目录

1. [概述](#概述)
2. [架构设计](#架构设计)
3. [核心模块](#核心模块)
4. [配置系统](#配置系统)
5. [驱逐策略](#驱逐策略)
6. [索引系统](#索引系统)
7. [Trace 处理](#trace-处理)
8. [结果分析](#结果分析)
9. [可视化分析](#可视化分析)
10. [使用说明](#使用说明)
11. [扩展指南](#扩展指南)

---

## 概述

KVCacheManager Optimizer 是一个独立的缓存优化分析模块，通过回放 trace 数据来模拟缓存读写操作，评估不同驱逐策略和配置对缓存命中率的影响，并为 KVCacheManager 主程序提供参数优化能力。

### 主要特性

- **多种驱逐策略**：支持 LRU、RandomLRU、LeafAwareLRU、TTL 等驱逐算法
- **分层存储**：支持多层级存储配置，目前功能不完备
- **Trace 回放**：支持 Publisher Log、Qwen Bailian 等多种 trace 格式
- **读写分离**：支持读写分离模式和组合模式
- **详细统计**：提供命中率、缓存使用情况等详细统计
- **灵活配置**：通过 JSON 配置文件灵活配置实例、存储和策略
- **可视化分析**：支持 Radix Tree 可视化、命中率图表和 Trade-off 曲线分析

---

## 架构设计

### 整体架构

```
main.cc (程序入口)
    ↓
OptimizerManager (核心协调器)
    ├── OptEvictionManager (驱逐管理器)
    ├── OptIndexerManager (索引管理器)
    └── OptimizerRunner (Trace 执行器)
        ↓
    ├── Eviction Policies (驱逐策略)
    │   ├── LRU
    │   ├── RandomLRU
    │   └── LeafAwareLRU
    ├── RadixTreeIndex (索引)
    └── Trace Converter (转换器)
        ↓
    HitAnalysis (结果分析)
        ↓
    Visualization Tools (可视化工具)
```

### 目录结构

```
kv_cache_manager/optimizer/
├── manager/              # 核心管理层
│   ├── optimizer_manager.h/cc       # 主协调器
│   ├── optimizer_runner.h/cc        # Trace 执行器
│   ├── eviction_manager.h/cc        # 驱逐管理器
│   ├── indexer_manager.h/cc         # 索引管理器
│   └── optimizer_loader.h/cc        # Trace 加载器
├── index/                # 索引层
│   └── radix_tree_index.h/cc        # Radix 树索引
├── eviction_policy/      # 驱逐策略层
│   ├── base.h                   # 策略基类
│   ├── common_structure.h       # 通用数据结构
│   ├── lru.h/cc                 # LRU 策略
│   ├── random_lru.h/cc          # RandomLRU 策略
│   ├── leaf_aware_lru.h/cc      # LeafAwareLRU 策略
│   └── policy_factory.h/cc      # 策略工厂
├── trace_converter/      # Trace 转换层
│   ├── optimizer_schema_trace.h  # Trace 定义
│   ├── base_converter.h          # 转换器基类
│   ├── publisher_log_converter.h/cc  # Publisher Log 转换器
│   ├── qwen_bailian_converter.h/cc    # Qwen Bailian 转换器
│   ├── converter_factory.h/cc    # 转换器工厂
│   └── trace_util.h              # Trace 工具
├── config/               # 配置层
│   ├── optimizer_config.h/cc     # 顶层配置
│   ├── instance_group_config.h/cc # 实例组配置
│   ├── instance_config.h/cc      # 实例配置
│   ├── tier_config.h/cc          # 存储层配置
│   ├── eviction_config.h         # 驱逐策略参数
│   └── types.h                   # 类型定义
├── analysis/             # 分析层
│   ├── result_structure.h        # 结果结构定义
│   ├── result_analysis.h/cc      # 命中率分析
│   └── script/                   # 分析脚本
│       ├── run/
│       │   ├── optimizer_run.py          # 单次回放 + 可选时序图
│       │   ├── tradeoff.py               # Pareto 曲线，单策略/多策略统一入口
│       │   ├── export_tree.py            # RadixTree 导出 + 可视化
│       │   ├── analyze_lifecycle.py      # Block lifecycle 统计
│       │   └── multi_instance_replay.py  # 多实例并行回放 + 聚合
│       ├── plot/
│       │   ├── hit_rate_plot.py          # 命中率时序图
│       │   ├── radix_tree_plot.py        # RadixTree 绘图
│       │   └── lifecycle_plot.py         # Lifecycle CDF/直方图
│       └── utils/
│           ├── optimizer_runner.py       # optimizer 运行封装
│           ├── csv_loader.py             # CSV 加载 + 容量点生成
│           └── plot_utils.py             # Pareto/per-tier 绘图工具
├── pybind/               # Python 绑定
│   └── py_optimizer_binding.cc   # Python 接口
├── main.cc               # 程序入口
└── optimizer_startup_config_load.json  # 配置示例
```

---

## 核心模块

### 1. OptimizerManager（优化器管理器）

**职责**：核心协调器，初始化所有子组件，管理实例组和实例配置，提供公共 API 接口。

**主要接口**：
```cpp
class OptimizerManager {
public:
    OptimizerManager(const OptimizerConfig &config);
    bool Init();
    void DirectRun();
    WriteCacheRes WriteCache(...);
    GetCacheLocationRes GetCacheLocation(...);
    void AnalyzeResults();
    std::unordered_map<std::string, RadixTreeExport> ExportRadixTrees() const;
};
```

### 2. OptimizerRunner（优化器运行器）

**职责**：执行 Trace 回放和模拟，处理两种 Trace 类型（GetLocationSchemaTrace、WriteCacheSchemaTrace），支持读写分离模式。

### 3. OptEvictionManager（驱逐管理器）

**职责**：管理跨实例的驱逐策略，支持三种驱逐模式：
- `EVICTION_MODE_GROUP_ROUGH` - 组级别粗粒度驱逐
- `EVICTION_MODE_INSTANCE_ROUGH` - 实例级别粗粒度驱逐
- `EVICTION_MODE_INSTANCE_PRECISE` - 实例级别精确驱逐

**主要接口**：
```cpp
class OptEvictionManager {
public:
    bool Init(const EvictionConfig &eviction_config);
    std::shared_ptr<EvictionPolicy> CreateAndRegisterEvictionPolicy(...);
    std::unordered_map<std::string, std::vector<BlockEntry *>> EvictByMode(...);
    size_t GetCurrentGroupUsage(...) const;
    size_t GetCurrentInstanceUsage(...) const;
};
```

### 4. OptIndexerManager（索引管理器）

**职责**：管理 RadixTreeIndex 实例，为每个实例创建索引器，支持多层存储配置。

**主要接口**：
```cpp
class OptIndexerManager {
public:
    bool CreateOptIndexer(...);
    std::shared_ptr<RadixTreeIndex> GetOptIndexer(...) const;
    void RegisterInstanceGroups(...);
    void RegisterInstances(...);
    bool CheckAndEvict(...);
    size_t GetCurrentInstanceUsage(...) const;
};
```

### 5. OptimizerLoader（Trace 加载器）

**职责**：加载和转换 trace 文件，按时间戳排序 trace，导出转换后的 trace 到文件。

---

## 配置系统

### 配置层次结构

```
OptimizerConfig (顶层配置)
    ├── trace_file_path (Trace 文件路径)
    ├── output_result_path (输出路径)
    ├── eviction_params (驱逐参数)
    │   ├── eviction_mode (驱逐模式)
    │   └── eviction_batch_size_per_instance (驱逐批量大小)
    └── instance_groups[] (实例组数组)
        ├── group_name (组名)
        ├── quota_capacity (配额容量)
        ├── used_percentage (使用百分比)
        ├── tier_strategy (多层读写策略)
        │   ├── hierarchical_eviction_enabled (分层驱逐)
        │   ├── write_mode (多层写入模式)
        │   ├── access_propagation_enabled (读访问传播)
        │   ├── promote_enabled (低层命中回填高层)
        │   └── selective_write_threshold (选择性下写阈值)
        ├── storages[] (存储层数组)
        └── instances[] (实例数组)
            ├── instance_id (实例ID)
            ├── block_size (块大小)
            ├── eviction_policy_type (驱逐策略类型)
            └── eviction_policy_params (驱逐策略参数)
```

### 配置文件示例

```json
{
    "trace_file_path": "/path/to/trace/file.jsonl",
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
            "storages": [
                {
                    "unique_name": "pace_00",
                    "storage_type": "pace",
                    "band_width_mbps": 20000,
                    "priority": 0,
                    "capacity": 100000
                }
            ],
            "instances": [
                {
                    "instance_id": "instance",
                    "block_size": 16,
                    "bytes_per_token": 512,
                    "eviction_policy_type": "random_lru",
                    "eviction_policy_params": {
                        "sample_rate": 0.1
                    }
                }
            ]
        }
    ]
}
```

### 配置参数说明

| 参数 | 说明 |
|------|------|
| trace_file_path | Trace 文件路径 |
| output_result_path | 结果输出目录；所有 config 驱动入口都使用该目录，`export_tree` 输出到其下的 `radix_tree/` |
| eviction_mode | 驱逐模式：1=GROUP_ROUGH, 2=INSTANCE_ROUGH, 3=INSTANCE_PRECISE |
| eviction_batch_size_per_instance | 粗粒度驱逐时的批量大小 |
| group_name | 实例组唯一标识 |
| quota_capacity | 组的总容量（GB）；`-1` 表示无限容量，不触发容量驱逐 |
| used_percentage | 实际使用的配额百分比 |
| instance_id | 实例唯一标识 |
| block_size | 每个 block 包含的 token 数量 |
| bytes_per_token | 单 token KV 大小；Python 分析脚本用它和 `block_size` 将 block 容量换算为 GB |
| eviction_policy_type | 驱逐策略类型：lru、random_lru、leaf_aware_lru |

---

## 驱逐策略

### 驱逐策略接口

```cpp
class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;
    virtual size_t size() const = 0;
    virtual void OnBlockWritten(BlockEntry *block) = 0;
    virtual void OnNodeWritten(std::vector<BlockEntry *> &blocks) = 0;
    virtual void OnBlockAccessed(BlockEntry *block, int64_t timestamp) = 0;
    virtual std::vector<BlockEntry *> EvictBlocks(size_t num_blocks) = 0;
    virtual std::string name() const = 0;
    virtual void set_name(const std::string &name) = 0;
};
```

### LRU 策略

**原理**：维护双向链表记录块的访问顺序，最近访问的块在链表头部，最久未访问的块在链表尾部，驱逐时从链表尾部移除块。

**时间复杂度**：
- `OnBlockAccessed()`: O(1)
- `OnBlockWritten()`: O(1)
- `EvictBlocks()`: O(n)

### RandomLRU 策略

**原理**：结合随机采样和 LRU 策略，从缓存中随机采样一定比例的块，选择最久未访问的块进行驱逐。

**时间复杂度**：
- `OnBlockAccessed()`: O(1)
- `OnBlockWritten()`: O(1)
- `EvictBlocks()`: O(m log m)，其中 m 为采样数量

### LeafAwareLRU 策略

**原理**：在 LRU 基础上增加了对叶子节点的感知，优先驱逐叶子节点中的块，提高缓存效率。

**实现特点**：
- 维护一个独立的叶子节点 LRU 链表
- 跟踪叶子节点中的所有 block
- 驱逐时优先从叶子节点链表中选择最久未访问的块

### TTL 语义分阶段说明（V1 / V2）

当前实现采用 **V1 语义**（已落地）：

- 仅 `POLICY_TTL` 实例会执行“读写前 TTL 过期物理清理”
- 非 TTL 策略（`lru` / `random_lru` / `leaf_aware_lru`）下，TTL 视为不存在
  - 不做前置 TTL 物理清理
  - 也不做 TTL 逻辑过期判定
- `POLICY_TTL` 下保证先清理过期块，再进入读写流程
- `default_block_ttl_seconds = 0` 表示组级禁用 TTL：
  - 写入路径会将 TTL 解析为“不过期”
  - 若 `fallback_on_pressure = false`，则不会发生 TTL 过期驱逐，也不会触发容量兜底驱逐
  - 若 `fallback_on_pressure = true`，则仅在容量压力时走链表尾部兜底驱逐（行为等价于 LRU）
- `POLICY_TTL` 的过期清理实现已从“每次读写前全链表扫描”优化为“最小过期时间堆（min-heap）增量回收”：
  - 写入/访问时写入过期事件
  - 清理时仅弹出到期事件，不再全量扫描
  - 使用版本号做惰性失效，避免过期事件重复处理
  - 在不改变 TTL 语义的前提下显著降低回放时延

后续 **V2 设计**（仅文档方案，暂不实现）：

- 统一语义：所有策略都支持前置过期清理
- 对过期扫描做精细化性能优化（例如 next_expire_ts / min-heap 等）
- 在不改变容量驱逐策略语义的前提下，实现跨策略一致的 TTL 生命周期

---

## 索引系统

### RadixTreeIndex 概述

**职责**：基于前缀树（Radix Tree）的数据结构，支持高效的前缀匹配查询，管理缓存的插入、查询和驱逐。

**核心操作**：
1. `InsertOnly()` - 仅插入块，不查询
2. `PrefixQuery()` - 前缀匹配查询
3. `ExportForVisualization()` - 导出前缀树用于可视化

### Radix Tree 数据结构

```cpp
struct RadixTreeNode {
    std::vector<std::unique_ptr<BlockEntry>> blocks;  // 连续的块段
    NodeStat stat;  // 节点统计信息
    RadixTreeNode *parent = nullptr;
    std::unordered_map<int64_t, std::unique_ptr<RadixTreeNode>> children;
    bool isLeaf() const { return children.empty(); }
};

struct NodeStat {
    size_t access_count = 0;
    int64_t last_access_time = 0;
    int64_t ttl = 250000;  // 默认TTL为250000微秒
};

struct BlockEntry {
    int64_t key;
    LocationStatMap location_map;
    int64_t writing_time = -1;
    int64_t last_access_time = -1;
    size_t access_count = 0;
    RadixTreeNode *owner_node = nullptr;
};
```

### Radix Tree 可视化

支持导出 Radix Tree 结构用于可视化分析，可以展示：
- 节点访问次数
- 最后访问时间
- 节点中的块数量
- 缓存的块数量
- 节点层级关系

---

## Trace 处理

### Trace 类型定义

**继承关系**：
```
OptimizerSchemaTrace (基类)
    ├── GetLocationSchemaTrace (读操作)
    ├── RequestSchemaTrace (请求级操作，内部调度读和延迟写)
    └── WriteCacheSchemaTrace (写操作)
```

### Trace 转换器

**支持的格式**：
- **Publisher Log**：转换 KVCacheManager Event Publisher 日志，区分读和写请求
- **Qwen Bailian**：转换 Qwen Bailian 数据集格式，输出读写分离 trace

**转换流程**：
1. 根据配置文件选择转换器
2. 解析日志文件并转换为标准 Trace；标准 `get` 必须带 `input_len`
3. 按时间戳排序 Trace
4. 分配唯一 Trace ID

标准 trace 接受显式 `type=get/write/request` 的 JSONL。`request` 用于外部只有请求级记录的场景，optimizer 会按 `trace_replay.write_delay_ns` 在内部调度 delayed write。`get/request.keys` 只能包含完整 block key，不足一个 block 的尾部 token 不写入 `keys`，但仍通过 `input_len` 计入 token 命中率分母。缺少 `input_len`、时间戳非法、`keys` 超过 `input_len / block_size` 等输入会在回放前失败。

---

## 结果分析

### 结果结构

```cpp
struct ReadRecord {
    int64_t timestamp_ns;
    // local = trace block_mask 带入的已有本地命中；remote = optimizer 模拟层命中
    size_t remote_read_blocks;
    size_t remote_hit_blocks;
    size_t local_read_blocks;
    size_t local_hit_blocks;
    size_t current_cache_blocks;
    size_t input_tokens;
    size_t block_size_tokens;
    std::vector<size_t> per_tier_hit_blocks;
    std::vector<std::string> tier_names;
    std::vector<size_t> per_tier_blocks;
    std::vector<size_t> blocks_per_instance;
    std::string trace_id;
};
```

### CSV 输出格式

**文件名**：`{instance_id}_hit_rates.csv`

**主要列**：
- `TimestampNs` - 时间戳（纳秒）
- `CachedBlocks` - 当前 CSV 对应 instance 的缓存 block 数
- `CachedBlocksAllInstances` - 同一 optimizer 进程内所有 instance 的总缓存 block 数
- `ReadBlocks` / `HitBlocks` - 当前请求读取和命中的 block 数
- `LocalHitBlocks` / `RemoteHitBlocks` - 诊断字段：trace `block_mask` 带入的已有本地命中 / optimizer 模拟层命中
- `InputTokens` / `HitTokens` - 当前请求的输入 token 数和命中 token 数
- `HitRate` - 当前 token 命中率，`HitTokens / InputTokens`
- `LocalHitTokens` / `RemoteHitTokens` - 诊断字段：本地 / optimizer 模拟层命中 token 数
- `AccReadBlocks` / `AccHitBlocks` - 累计读取和命中的 block 数
- `AccHitRate` - 累积 token 命中率，`AccHitTokens / AccInputTokens`
- `AccLocalHitRate` / `AccRemoteHitRate` - 诊断字段，不作为标准分析主口径
- `Tier<N>(name)_HitTokens` / `Tier<N>(name)_HitRate` / `AccTier<N>(name)_HitRate` - 分层命中 token 指标

标准分析直接按请求输入计算整体 `HitRate`，不把 local/remote 作为独立结论维度。local/remote 只用于兼容 optimizer 作为单独 L3 模拟并和 HiSim 结合、或直接分析 KVCacheManager event log 时已有的本地命中信息。

---

## 可视化分析

### 1. 命中率随时间变化图表

**脚本**：`optimizer_run.py --draw-chart`，绘图实现位于 `plot/hit_rate_plot.py`

**功能**：绘制多实例缓存分析图表，展示所有 instance 的存储容量总和以及各自命中率随时间的变化。

**运行方式**：
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json \
    --draw-chart
```

**输出**：
- `{output_result_path}/timeseries/multi_instance_cache_analysis.png`
- 分层配置额外输出 `{output_result_path}/timeseries/per_tier_timeseries.png`

**图表内容**：
- 上图：累计命中率随时间变化
- 下图：当前 trace 命中率随时间变化

### 2. Radix Tree 可视化

**脚本**：`export_tree.py`，绘图实现位于 `plot/radix_tree_plot.py`

**功能**：导出并可视化前缀树结构，统计并展示热节点以及所属节点的前缀路径。

**运行方式**：
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:export_tree -- \
    -c /path/to/config.json
```

**输出**：
- `{output_result_path}/radix_tree/{instance_id}_radix_tree.json` - Radix Tree 导出数据
- `{output_result_path}/radix_tree/{instance_id}_radix_tree.png` - Radix Tree 可视化图表
- `{output_result_path}/radix_tree/{instance_id}_hot_paths.png` - 热点路径图（传 `--show-hot-paths` 时）

**可视化内容**：
- 节点访问次数
- 最后访问时间
- 节点中的块数量
- 缓存的块数量
- 节点层级关系

### 3. Pareto 容量-命中率曲线分析

**脚本**：`tradeoff.py`

**功能**：在不同容量配置下回放 optimizer，绘制容量与 token 命中率的 Pareto 曲线。不给 `--eviction-policies` 时使用配置文件中的策略；给出多个策略时生成多策略对比图。

> **适用范围**：Trade-off 分析仅适用于非分层模式。在分层模式（`tier_strategy.hierarchical_eviction_enabled=true`）下，容量扫描仅修改 `quota_capacity`，不影响各 tier 独立的 `storages[i].capacity`，因此无法产生有意义的容量-性能权衡结果。

**运行流程**：

1. 用无限容量 warmup 跑完整 trace。实现上会将 `quota_capacity` 写为 `-1`，C++ optimizer 将负容量视作无限容量。
2. 从 warmup 读取理论命中率和最大缓存 block 数。
3. 基于最大缓存 block 数按指数分布生成最多 `--num-points` 个容量点，并用 `--min-capacity-ratio` 过滤过小点。
4. 对每个容量点运行 optimizer；达到 99% 理论命中率后停止继续扫更大容量。
5. 画图时补 `(0 GB, 0%)` 起点，只保留命中率上升段。下降点会从图上剔除，并在 stdout 打印 `Drop descending Pareto point ...`；原始 CSV 保留。

**运行方式**：
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
- `--num-points` - 每个策略最多运行的容量采样点数量（默认 30），达到 99% 理论命中后提前停止
- `--min-capacity-ratio` - 最小容量点相对阈值（默认 `1e-4`）
- `--hit-rate-type` - 命中率类型，标准分析使用 total；local/remote/all 仅作诊断拆分（默认 total）
- `--max-workers` - 并行执行的最大线程数（默认 4）
- `--plot-title` - 覆盖图标题

**输出**：`{output_result_path}/pareto/pareto_curve_{hit_rate_type}.png`

### 4. 多策略对比分析

**脚本**：`tradeoff.py --eviction-policies ...`

**功能**：对比多个驱逐策略在不同容量配置下的表现。所有 instance 使用同一个给定策略进行一轮回放；每个策略独立 warmup、独立计算理论命中率，并独立提前停止。

> **适用范围**：同单策略 Trade-off 分析，仅适用于非分层模式。

**运行方式**：
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/config.json \
    --eviction-policies lru random_lru leaf_aware_lru \
    --num-points 30 \
    --hit-rate-type total \
    --max-workers 4
```

**参数说明**：
- `-c, --config` - 配置文件路径
- `--eviction-policies` - 要对比的驱逐策略列表（默认 lru random_lru leaf_aware_lru）
- `--num-points` - 每个策略最多运行的容量采样点数量（默认 30），达到 99% 理论命中后提前停止
- `--hit-rate-type` - 命中率类型，标准分析使用 total；local/remote/all 仅作诊断拆分（默认 total）
- `--max-workers` - 并行执行的最大线程数（默认 4）

**输出**：`{output_result_path}/pareto/multi_policy_{hit_rate_type}.png`

图形规则：X 轴是 `Capacity (GB)`，Y 轴是 `HitRate (%)`；95%/99% 理论命中率会按相邻 sweep 点线性插值后标出容量和命中率。`--skip-run` 只从已有 `csv_results` 画图，不重新 warmup，因此不会生成理论 95%/99% 标注。

---

## 使用说明

### 编译

```bash
bazel build //kv_cache_manager/optimizer:optimizer_main
```

### 运行优化器

**方式：直接运行二进制文件**

```bash
bazel run //kv_cache_manager/optimizer:optimizer_main -- /path/to/config.json
```

### Python 接口

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
```

### 输出文件

运行完成后，会在 `output_result_path` 指定的目录下生成：
- `{instance_id}_hit_rates.csv` - 每个 instance 的命中率数据

---

## 扩展指南

### 添加新的驱逐策略

1. 在 `kv_cache_manager/optimizer/eviction_policy/` 创建新策略文件，继承 `EvictionPolicy` 基类
2. 在 `kv_cache_manager/optimizer/config/types.h` 添加新的策略类型枚举值
3. 在 `kv_cache_manager/optimizer/config/eviction_config.h` 添加新的参数类型
4. 在 `kv_cache_manager/optimizer/eviction_policy/policy_factory.cc` 添加新的策略创建逻辑
5. 在 BUILD 文件中添加新的源文件

### 添加新的 Trace 转换器

1. 在 `kv_cache_manager/optimizer/trace_converter/` 创建新转换器文件，继承 `BaseConverter` 基类
2. 在 `kv_cache_manager/optimizer/config/types.h` 添加新的 trace 类型枚举值
3. 在 `kv_cache_manager/optimizer/trace_converter/converter_factory.cc` 添加新的转换器创建逻辑
4. 在 BUILD 文件中添加新的源文件

### 添加新的分析指标

1. 在 `kv_cache_manager/optimizer/analysis/result_structure.h` 添加新的统计字段
2. 在 `kv_cache_manager/optimizer/analysis/result_analysis.cc` 添加新的分析逻辑
3. 实现新的导出函数来输出自定义指标

---

## 附录

### 文件索引

**核心管理器**：
- optimizer_manager.h/cc - 主协调器
- optimizer_runner.h/cc - Trace 执行器
- eviction_manager.h/cc - 驱逐管理器
- indexer_manager.h/cc - 索引管理器
- optimizer_loader.h/cc - Trace 加载器

**索引层**：
- radix_tree_index.h/cc - Radix 树索引

**驱逐策略**：
- base.h - 策略基类
- common_structure.h - 通用数据结构
- lru.h/cc - LRU 策略
- random_lru.h/cc - RandomLRU 策略
- leaf_aware_lru.h/cc - LeafAwareLRU 策略
- policy_factory.h/cc - 策略工厂

**Trace 转换**：
- optimizer_schema_trace.h - Trace 定义
- base_converter.h - 转换器基类
- publisher_log_converter.h/cc - Publisher Log 转换器
- qwen_bailian_converter.h/cc - Qwen Bailian 转换器
- converter_factory.h/cc - 转换工厂
- trace_util.h - Trace 工具

**配置**：
- optimizer_config.h/cc - 顶层配置
- instance_group_config.h/cc - 实例组配置
- instance_config.h/cc - 实例配置
- tier_config.h/cc - 存储层配置
- eviction_config.h - 驱逐策略参数
- types.h - 类型定义
- optimizer_config_loader.h/cc - 配置加载器

**分析**：
- result_structure.h - 结果结构定义
- result_analysis.h/cc - 命中率分析
- script/run/optimizer_run.py - 单次回放 + 可选时序图
- script/run/tradeoff.py - Pareto 曲线，单策略/多策略统一入口
- script/run/export_tree.py - Radix Tree 导出 + 可视化
- script/run/analyze_lifecycle.py - Block lifecycle 分析
- script/run/multi_instance_replay.py - 多实例并行回放 + 聚合
- script/plot/hit_rate_plot.py - 命中率时序图
- script/plot/radix_tree_plot.py - Radix Tree 绘图
- script/utils/optimizer_runner.py - optimizer 运行封装

**Python 绑定**：
- pybind/py_optimizer_binding.cc - Python 接口

### 术语表

| 术语 | 说明 |
|------|------|
| 驱逐策略 | 当缓存满时选择哪些块被移除的算法 |
| LRU | 最近最少使用算法 |
| RandomLRU | 结合随机采样和 LRU 的混合算法 |
| LeafAwareLRU | 叶子节点感知的 LRU 算法 |
| Radix Tree | 用于高效前缀匹配的树形数据结构 |
| Trace | 记录系统操作序列的数据 |
| Instance | 缓存系统的独立实例 |
| Instance Group | 共享资源的实例集合 |
| Block | 缓存的基本单位 |
| Hit Rate | 缓存命中次数与总访问次数的比值 |
| Prefix Match | 查找具有相同前缀的键 |
| Read/Write Separation | 将读和写操作分开处理 |
| Trade-off Curve | 容量与命中率之间的权衡曲线 |



---
