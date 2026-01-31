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

- **多种驱逐策略**：支持 LRU、RandomLRU、LeafAwareLRU 等驱逐算法
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
│       ├── optimizer_run.py      # 运行脚本
│       ├── plot_hit_rate_with_storage.py  # 命中率随时间变化图表
│       ├── export_and_visualize_tree.py   # Radix Tree 可视化
│       ├── plot_radix_tree.py            # Radix Tree 绘图
│       ├── tradeoff_curve_by_instances.py # 单策略 Trade-off 曲线
│       ├── tradeoff_curve_by_policies.py  # 多策略对比分析
│       └── optimizer_analysis_utils.py    # 分析工具函数
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

**职责**：执行 Trace 回放和模拟，处理三种 Trace 类型（GetLocationSchemaTrace、WriteCacheSchemaTrace、DialogTurnSchemaTrace），支持读写分离模式。

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
    ├── trace_type (Trace 类型)
    ├── output_result_path (输出路径)
    ├── eviction_params (驱逐参数)
    │   ├── eviction_mode (驱逐模式)
    │   └── eviction_batch_size_per_instance (驱逐批量大小)
    └── instance_groups[] (实例组数组)
        ├── group_name (组名)
        ├── quota_capacity (配额容量)
        ├── used_percentage (使用百分比)
        ├── hierarchical_eviction_enabled (分层驱逐)
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
    "trace_type": "qwen_bailian",
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
            "hierarchical_eviction_enabled": false,
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
| trace_type | Trace 类型：publisher_log 或 qwen_bailian |
| output_result_path | 结果输出目录 |
| eviction_mode | 驱逐模式：1=GROUP_ROUGH, 2=INSTANCE_ROUGH, 3=INSTANCE_PRECISE |
| eviction_batch_size_per_instance | 粗粒度驱逐时的批量大小 |
| group_name | 实例组唯一标识 |
| quota_capacity | 组的总容量（blocks） |
| used_percentage | 实际使用的配额百分比 |
| instance_id | 实例唯一标识 |
| block_size | 每个 block 包含的 token 数量 |
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

---

## 索引系统

### RadixTreeIndex 概述

**职责**：基于前缀树（Radix Tree）的数据结构，支持高效的前缀匹配查询，管理缓存的插入、查询和驱逐。

**核心操作**：
1. `InsertOnly()` - 仅插入块，不查询
2. `PrefixQuery()` - 前缀匹配查询
3. `InsertWithQuery()` - 插入并查询（组合操作）
4. `ExportForVisualization()` - 导出前缀树用于可视化

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
    std::vector<int64_t> token_ids;
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
    │   └── DialogTurnSchemaTrace (对话轮次)
    └── WriteCacheSchemaTrace (写操作)
```

### Trace 转换器

**支持的格式**：
- **Publisher Log**：转换 KVCacheManager Event Publisher 日志，区分读和写请求
- **Qwen Bailian**：转换 Qwen Bailian 数据集格式，直接创建 DialogTurnSchemaTrace

**转换流程**：
1. 根据配置文件选择转换器
2. 解析日志文件并转换为标准 Trace
3. 按时间戳排序 Trace
4. 分配唯一 Trace ID

---

## 结果分析

### 结果结构

```cpp
struct ResultCounters {
    uint64_t total_blocks = 0;
    uint64_t total_write_blocks = 0;
    uint64_t total_read_blocks = 0;
    uint64_t total_hit_blocks = 0;
    uint64_t total_read_requests = 0;
    uint64_t total_requests = 0;
};

struct ReadRecord {
    int64_t timestamp_us;
    size_t external_read_blocks;
    size_t external_hit_blocks;
    size_t internal_read_blocks;
    size_t internal_hit_blocks;
    size_t current_cache_blocks;
    std::vector<size_t> blocks_per_instance;
};

struct Result {
    ResultCounters counters;
    std::vector<ReadRecord> read_results;
    std::vector<WriteRecord> write_results;
};
```

### CSV 输出格式

**文件名**：`{instance_id}_hit_rates.csv`

**主要列**：
- `TimestampUs` - 时间戳（微秒）
- `CachedBlocksCurrentInstance` - 当前实例的缓存块数
- `CachedBlocksAllInstance` - 所有实例的总缓存块数
- `HitRate` - 当前命中率
- `AccHitRate` - 累积总命中率
- `InternalHitRate` / `ExternalHitRate` - 内部/外部命中率

---

## 可视化分析

### 1. 命中率随时间变化图表

**脚本**：`plot_hit_rate_with_storage.py`

**功能**：绘制多实例缓存分析图表，展示所有 instance 的存储容量总和以及各自命中率随时间的变化。

**运行方式**：
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json \
    --draw-chart
```

**输出**：
- `{output_result_path}/multi_instance_cache_analysis.png`

**图表内容**：
- 上图：累计命中率随时间变化
- 下图：当前 trace 命中率随时间变化

### 2. Radix Tree 可视化

**脚本**：`export_and_visualize_tree.py`

**功能**：导出并可视化前缀树结构，统计并展示热节点以及所属节点的前缀路径。

**运行方式**：
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:visualize_tree -- \
    -c /path/to/config.json
```

**输出**：
- `{output_result_path}/radix_tree_{instance_id}.json` - Radix Tree 导出数据
- `{output_result_path}/radix_tree_{instance_id}.png` - Radix Tree 可视化图表

**可视化内容**：
- 节点访问次数
- 最后访问时间
- 节点中的块数量
- 缓存的块数量
- 节点层级关系

### 3. 单策略 Trade-off 曲线分析

**脚本**：`tradeoff_curve_by_instances.py`

**功能**：生成在不同容量配置下的多个 instance 命中率曲线，用于评估容量与命中率的权衡关系，仅使用配置中的驱逐策略。

**运行方式**：
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff_analysis_run_by_instances -- \
    -c /path/to/config.json \
    --warmup-capacity 30000000 \
    --num-points 40 \
    --hit-rate-type total \
    --max-workers 4
```

**参数说明**：
- `-c, --config` - 配置文件路径
- `--warmup-capacity` - Warmup 阶段使用的大容量（默认 30000000）
- `--num-points` - 容量采样点数量（默认 40）
- `--hit-rate-type` - 命中率类型：total/internal/external/all（默认 total）
- `--max-workers` - 并行执行的最大线程数（默认 4）

**输出**：`{output_result_path}/pareto_curve_{hit_rate_type}.png`

### 4. 多策略对比分析

**脚本**：`tradeoff_curve_by_policies.py`

**功能**：对比每个 instance 多个驱逐策略在不同容量配置下的性能表现，所有 instance 统一用一种类型的驱逐策略。

**运行方式**：
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff_analysis_run_by_policies -- \
    -c /path/to/config.json \
    --warmup-capacity 30000000 \
    --eviction-policies lru random_lru leaf_aware_lru \
    --num-points 40 \
    --hit-rate-type total \
    --max-workers 4
```

**参数说明**：
- `-c, --config` - 配置文件路径
- `--warmup-capacity` - Warmup 阶段使用的大容量（默认 30000000）
- `--eviction-policies` - 要对比的驱逐策略列表（默认 lru random_lru leaf_aware_lru）
- `--num-points` - 容量采样点数量（默认 40）
- `--hit-rate-type` - 命中率类型：total/internal/external/all（默认 total）
- `--max-workers` - 并行执行的最大线程数（默认 4）

**输出**：`{output_result_path}/multi_policy_{hit_rate_type}.png`

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
- script/optimizer_run.py - 运行脚本
- script/plot_hit_rate_with_storage.py - 命中率图表
- script/export_and_visualize_tree.py - Radix Tree 可视化
- script/plot_radix_tree.py - Radix Tree 绘图
- script/tradeoff_curve_by_instances.py - 单策略 Trade-off 曲线
- script/tradeoff_curve_by_policies.py - 多策略对比分析
- script/optimizer_analysis_utils.py - 分析工具函数

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
