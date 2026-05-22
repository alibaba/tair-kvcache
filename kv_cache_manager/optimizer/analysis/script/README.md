# Optimizer 分析脚本

KVCacheManager optimizer 的分析与可视化工具集。

Bazel target 前缀：

```
//kv_cache_manager/optimizer/analysis/script
```

标准 Python 入口的输出规则：凡是传入 optimizer config 的入口都只使用 config 中的 `output_result_path`；`multi_instance_replay` 不读取完整 config，必须通过 `--output-dir` 指定输出目录。

标准报告直接按请求输入计算整体 `HitRate = HitTokens / InputTokens`。CSV 中的 local/remote 字段只是诊断拆分：local 来自 trace `block_mask` 带入的已有本地命中 block，remote 来自 optimizer 模拟层命中；标准分析结论不按 local/remote 分组。

---

## 1. 单次运行 — `optimizer_run`

运行一次 optimizer 仿真，输出命中率 CSV，可选生成时序图。

```bash
# 基本运行
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- -c config.json

# 运行 + 生成命中率时序图
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- -c config.json --draw-chart

# 运行 + 导出 lifecycle CSV（用于后续 lifecycle 分析）
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- -c config.json --export-lifecycle
```

### 参数

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `-c, --config` | ✅ | — | optimizer 配置文件路径（JSON） |
| `--draw-chart` | — | false | 生成命中率时序图 |
| `--export-lifecycle` | — | false | 导出 lifecycle CSV（内存消耗大） |
| `--enable-template-analysis` | — | false | 启用模板前缀分析；会拖慢回放速度，开启后才会生成模板前缀相关 CSV |

### 输出

```
<output_result_path>/
├── *_hit_rates.csv                       # 每个 instance 的命中率时序数据
├── *_template_prefix_traces.csv          # per-trace 模板归属明细（需 --enable-template-analysis）
├── *_template_prefix_summary.csv         # 模板级汇总（需 --enable-template-analysis）
├── *_lifecycle.csv                       # block 生命周期数据（需 --export-lifecycle）
└── timeseries/
    └── multi_instance_cache_analysis.png # 命中率时序图（需 --draw-chart）
```

`optimizer_run --draw-chart` 在分层配置下还会生成 `timeseries/per_tier_timeseries.png`；非分层配置会跳过该图。

---

## 2. 多实例回放 — `multi_instance_replay`

按 instance trace 并行运行 optimizer，并聚合 token hit rate。每个输入 JSONL 必须是标准 optimizer schema，且一个文件只能包含一个 `instance_id`。

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:multi_instance_replay -- \
    --trace-dir /path/to/instance_traces \
    --trace-glob "*.jsonl" \
    --output-dir /path/to/output \
    --l1-capacity 50 \
    --l2-capacity 128 \
    --block-size 16 \
    --bytes-per-token 512 \
    --eviction-policy lru \
    --default-tier-write-mode write_through \
    --max-workers 32
```

### 输入与输出参数

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `--trace-dir` | 条件必需 | — | instance trace 目录；与 `--trace-files` 互斥。非 `--aggregate-only` 模式下二选一 |
| `--trace-files` | 条件必需 | — | 显式传入多个 per-instance JSONL trace 文件；与 `--trace-dir` 互斥 |
| `--trace-glob` | — | `*.jsonl` | 仅配合 `--trace-dir` 使用，控制扫描哪些文件 |
| `--recursive` | — | false | 仅配合 `--trace-dir` 使用，递归扫描子目录 |
| `--output-dir` | ✅ | — | 输出目录；回放结果、生成 config、聚合 CSV 都写到这里 |
| `--bucket-name` | — | 见下 | 写入聚合 CSV 的 `Bucket` 列；只用于标记实验来源，不影响回放和 hit rate 计算 |
| `--skip-existing` | — | false | 如果 `<output-dir>/<instance_id>_hit_rates.csv` 已存在，则跳过该 instance 的 replay，并在本轮聚合中使用该 CSV |
| `--max-workers` | — | `min(cpu_count, instance_count)` | 并发 optimizer 进程数；`0` 表示使用默认值 |
| `--log-level` | — | 4 | 子进程 KVCM logger 等级 |

`--bucket-name` 默认值：使用 `--trace-dir` 时为 trace 目录名；`--aggregate-only` 时为 output 目录名；使用 `--trace-files` 时默认为空。
非 `--aggregate-only` 模式只聚合本轮 `--trace-dir` / `--trace-files` 选中的 instance CSV，不会扫描并混入 `--output-dir` 下的其他历史 CSV；未使用 `--skip-existing` 时，脚本会先删除本轮 instance 的旧 CSV，避免 worker 成功检查误读旧结果。

### 回放配置参数

这些参数会写入脚本为每个 instance 生成的 optimizer config。

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `--l1-capacity` | — | 50.0 | tier 0 容量，单位 GB |
| `--l2-capacity` | — | 128.0 | tier 1 容量，单位 GB；`0` 表示关闭 L2，只保留 L1 |
| `--block-size` | — | 16 | 每个 block 的 token 数 |
| `--bytes-per-token` | — | 512 | 单 token KV 大小；用于生成 config 和容量换算 |
| `--eviction-policy` | — | `lru` | 每个 instance 的驱逐策略；可选 `lru` / `random_lru` / `leaf_aware_lru` / `ttl` |
| `--eviction-policy-params` | — | 策略默认值 | JSON object，覆盖策略参数，例如 `'{"sample_rate": 0.5}'` |
| `--eviction-mode` | — | 3 | optimizer eviction mode：`1=group rough`，`2=instance rough`，`3=instance precise` |
| `--eviction-batch-size` | — | 100 | 每个 instance 单次驱逐批大小 |
| `--default-tier-write-mode` | — | `write_through` | 写入 `tier_strategy.write_mode`，作为所有相邻 tier edge 的默认写入策略；可选 `write_through` / `cascading` / `write_through_selective` |
| `--tier-flow-config` | — | 空 | JSON array 或 JSON 文件路径，写入 `tier_strategy.tier_flows`，用于覆盖相邻 tier edge 的策略 |
| `--enable-tier-access-propagation` | — | true | 写入 `tier_strategy.access_propagation_enabled=true`；命中上层副本时，同时刷新后续持有副本 tier 的访问时间；与 `--disable-tier-access-propagation` 互斥 |
| `--disable-tier-access-propagation` | — | false | 写入 `tier_strategy.access_propagation_enabled=false`；命中上层副本时，只刷新命中 tier，不刷新下层冷热；与 `--enable-tier-access-propagation` 互斥 |
| `--selective-write-threshold` | — | 2 | 写入 `tier_strategy.selective_write_threshold`；`write_through_selective` 下，命中层访问次数达到该阈值后复制到下一层；必须为正整数 |
| `--enable-promote` | — | true | 写入 `tier_strategy.promote_enabled=true`；开启从低层级向高层级 promote；与 `--disable-promote` 互斥 |
| `--disable-promote` | — | false | 写入 `tier_strategy.promote_enabled=false`；关闭 promote；与 `--enable-promote` 互斥 |
| `--disable-hierarchical-eviction` | — | false | 写入 `tier_strategy.hierarchical_eviction_enabled=false`；默认生成分层配置 |
| `--used-percentage` | — | 1.0 | 写入 config 的 group `used_percentage` |
| `--default-block-ttl-seconds` | — | 0 | 默认 block TTL 秒数；主要用于 `--eviction-policy ttl` |
| `--ttl-refresh-on-read` | — | true | TTL 策略下读命中刷新 last access time；与 `--no-ttl-refresh-on-read` 互斥 |
| `--no-ttl-refresh-on-read` | — | false | TTL 策略下读命中不刷新 last access time；与 `--ttl-refresh-on-read` 互斥 |

未显式传 `--eviction-policy-params` 时，脚本按策略生成默认参数：`random_lru` 使用 `{"sample_rate": 1.0}`；`ttl` 使用 `{"fallback_on_pressure": true}`；其他策略使用 leaf-aware/random-lru 兼容参数。

### 聚合参数

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `--aggregate-only` | — | false | 跳过 replay，只聚合 `--output-dir` 下已有的 `*_hit_rates.csv`；此时不需要 `--trace-dir` / `--trace-files` |
| `--start-ns` | — | — | 聚合时间范围起点，纳秒；区间语义为 `[start_ns, end_ns)` |
| `--end-ns` | — | — | 聚合时间范围终点，纳秒；区间语义为 `[start_ns, end_ns)` |
| `--window-ns` | — | — | 窗口大小，纳秒；与 `--window-seconds` 互斥，设置后生成窗口级聚合 |
| `--window-seconds` | — | — | 窗口大小，秒；与 `--window-ns` 互斥，设置后生成窗口级聚合 |
| `--include-instance-windows` | — | false | 生成 per-instance 窗口聚合；只有设置窗口大小时才有输出 |
| `--aggregation-chunksize` | — | 1000000 | 聚合读取 hit-rate CSV 的 pandas chunk 行数，调大可提速但占更多内存 |

### 输出

```
<output_dir>/
├── configs/
│   └── <instance_id>.json                 # 每个 instance 的生成配置
├── <instance_id>_hit_rates.csv            # 每个 instance 的 token hit-rate 时序
├── multi_instance_replay_tasks.csv         # 任务清单：InstanceId / TraceFile / ConfigPath / CsvPath
├── multi_instance_replay_summary.csv       # 子进程执行结果：成功状态、耗时、错误信息
└── aggregate/
    ├── instance_aggregate.csv
    ├── global_aggregate.csv
    ├── global_window_hit_rates.csv        # 需 --window-ns 或 --window-seconds
    └── instance_window_hit_rates.csv      # 需窗口参数 + --include-instance-windows
```

---

## 3. Pareto 曲线分析 — `tradeoff`

在多个容量点上运行 optimizer，绘制容量-命中率权衡曲线。自动判断单策略/多策略模式。

> **适用范围**：Tradeoff 分析仅适用于非分层模式。在分层模式（`tier_strategy.hierarchical_eviction_enabled=true`）下，容量扫描仅修改 `quota_capacity`，而驱逐决策依据各 tier 独立的 `storages[i].capacity`，因此扫描结果无法反映真实的容量-性能权衡关系。

### 单策略模式

不指定 `--eviction-policies`，使用配置文件中的默认策略。每个 instance 一条曲线。

```bash
# 默认 40 个容量点
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- -c config.json

# 自定义采样点数
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- -c config.json --num-points 30

# 保存 CSV + 生成每个容量点的时序图
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- -c config.json --save-csv --plot-timeseries

# 从 <output_result_path>/csv_results 加载（跳过实验）
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- -c config.json --skip-run

# 自定义坐标轴范围
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- -c config.json --x-min 1000000 --y-min 0.5
```

### 多策略对比模式

指定 `--eviction-policies`，每个 instance 一个子图，每个策略一条曲线。

```bash
# 对比三种策略
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c config.json --eviction-policies lru leaf_aware_lru random_lru

# 多策略 + skip-run
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c config.json --eviction-policies lru leaf_aware_lru --skip-run

# 只为特定容量点生成时序图
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c config.json --eviction-policies lru leaf_aware_lru \
    --save-csv --plot-timeseries --plot-capacity 5000000 10000000
```

### 参数

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `-c, --config` | ✅ | — | 配置文件路径 |
| `--eviction-policies` | — | 配置默认 | 驱逐策略列表（空格分隔）；不传时使用 config 中的策略 |
| `--warmup-capacity` | — | 10000.0 | warmup 阶段容量上限，单位 GB；应足够容纳全量 trace 数据 |
| `--num-points` | — | 40 | 容量采样点数；实际容量点按指数分布生成，并过滤过小容量 |
| `--hit-rate-type` | — | `total` | 绘制命中率类型；可选 `total` / `local` / `remote` / `all` |
| `--max-workers` | — | 4 | 并行实验线程数 |
| `--save-csv` | — | false | 保留每次运行的 CSV 文件 |
| `--skip-run` | — | false | 跳过实验，从 `<output_result_path>/csv_results` 加载已有 `cap_<capacity>_<policy>/` 结果 |
| `--plot-timeseries` | — | false | 为容量点生成时序图；必须配合 `--save-csv` 或 `--skip-run` 才有 CSV 可画 |
| `--plot-capacity` | — | 全部 | 只为指定容量点生成时序图；传入的是 `cap_<capacity>_<policy>` 里的 block 容量整数 |
| `--x-min` | — | 自动 | Pareto 图 X 轴最小值 |
| `--x-max` | — | 自动 | Pareto 图 X 轴最大值 |
| `--y-min` | — | 自动 | Pareto 图 Y 轴最小值 |
| `--y-max` | — | 自动 | Pareto 图 Y 轴最大值 |

`--hit-rate-type all` 会分别输出 total/local/remote 三类图。标准结论看 total token hit rate；local/remote 仅作为诊断拆分。

### 输出

```
<output_result_path>/
├── pareto/
│   ├── pareto_curve_<type>.png           # 单策略 Pareto 散点图
│   ├── multi_policy_<type>.png           # 多策略对比子图
├── timeseries/
│   └── multi_instance_cache_analysis.png # 时序图（需 --plot-timeseries）
└── csv_results/                          # 需 --save-csv
    └── cap_<capacity>_<policy>/
        ├── *_hit_rates.csv
        ├── *_template_prefix_traces.csv
        └── *_template_prefix_summary.csv
```

---

## 4. 前缀树可视化 — `export_tree`

运行 optimizer 后导出 RadixTree 结构为 JSON，并生成可视化图。

```bash
# 热点路径可视化（推荐）
bazel run //kv_cache_manager/optimizer/analysis/script:export_tree -- \
    -c config.json --show-hot-paths --hot-nodes 20

# 完整树可视化（小树适用）
bazel run //kv_cache_manager/optimizer/analysis/script:export_tree -- -c config.json

# 只看统计信息
bazel run //kv_cache_manager/optimizer/analysis/script:export_tree -- -c config.json --stats-only

# 打印热点路径的 block 序列详情
bazel run //kv_cache_manager/optimizer/analysis/script:export_tree -- \
    -c config.json --show-hot-paths --show-blocks
```

### 参数

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `-c, --config` | ✅ | — | 配置文件路径 |
| `--hot-nodes` | — | 10 | Top K 热点节点数 |
| `--show-hot-paths` | — | false | 只可视化热点路径（推荐大树使用） |
| `--show-blocks` | — | false | 打印热点路径的 block 序列 |
| `--max-blocks` | — | 100 | 每个节点最多显示的 block 数 |
| `--stats-only` | — | false | 只打印统计，不生成图片 |
| `--layout` | — | auto | 布局算法：auto / graphviz / custom |
| `--node-size` | — | 2000 | 节点基础大小 |

### 从已有 JSON 直接可视化

如果已有导出的 JSON 文件，直接用 `plot/radix_tree_plot.py` 画图，无需重跑 optimizer：

```bash
# 热点路径可视化
python kv_cache_manager/optimizer/analysis/script/plot/radix_tree_plot.py \
    -i tree_export.json -o output.png --show-hot-paths --hot-nodes 15

# 完整树可视化
python kv_cache_manager/optimizer/analysis/script/plot/radix_tree_plot.py \
    -i tree_export.json -o tree_full.png

# 只看统计
python kv_cache_manager/optimizer/analysis/script/plot/radix_tree_plot.py \
    -i tree_export.json --stats
```

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `-i, --input` | ✅ | — | 导出的 JSON 文件路径 |
| `-o, --output` | — | 交互显示 | 输出图片路径 |
| `--hot-nodes` | — | 10 | Top K 热点节点数 |
| `--show-hot-paths` | — | false | 只可视化热点路径 |
| `--node-size` | — | 2000 | 节点基础大小 |
| `--no-labels` | — | false | 不显示节点标签 |
| `--stats` | — | false | 只打印统计 |
| `--max-nodes` | — | 500 | 完整树节点数警告阈值 |
| `--layout` | — | auto | 布局算法 |
| `--show-blocks` | — | false | 打印 block 序列 |
| `--max-blocks` | — | 50 | 每节点最多显示 block 数 |

### 输出

```
<output_result_path>/
└── radix_tree/
    ├── <instance>_radix_tree.json        # 前缀树结构数据
    ├── <instance>_radix_tree.png         # 完整树可视化
    └── <instance>_hot_paths.png          # 热点路径可视化
```

---

## 5. Block Lifecycle 分析 — `analyze_lifecycle`

分析 block 的生命周期数据，产出统计报告 + CDF 图 + Access Count 直方图。

```bash
# 分析单个文件
bazel run //kv_cache_manager/optimizer/analysis/script:analyze_lifecycle -- \
    -i instance_lifecycle.csv

# 批量分析目录下所有 *_lifecycle.csv
bazel run //kv_cache_manager/optimizer/analysis/script:analyze_lifecycle -- \
    -i output_dir/

# 只看统计信息（不生成图表）
bazel run //kv_cache_manager/optimizer/analysis/script:analyze_lifecycle -- \
    -i instance_lifecycle.csv --stats-only
```

### 参数

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `-i, --input` | ✅ | — | lifecycle CSV 文件或包含 `*_lifecycle.csv` 的目录 |
| `--stats-only` | — | false | 只打印统计信息，不生成图表 |

### 统计报告内容

| 类别 | 指标 |
|------|------|
| 总体概览 | 总 block 数、唯一 BlockKey 数、复活率、最多复活次数、即逐占比（lifespan=0）、零访问占比、缓存利用率 |
| Physical Lifespan | 均值、中位数、标准差、min/max、P25/P75/P90/P95/P99 |
| Active Lifespan | 均值、中位数、标准差、min/max、P25/P75/P90/P95/P99 |
| Access Count | 均值、中位数、标准差、min/max、P25/P75/P90/P95/P99 |

### 输出

```
<input_dir>/
└── lifecycle/
    ├── <instance>_physical_lifespan_cdf.png  # Physical Lifespan CDF（全量 + Evicted）
    ├── <instance>_active_lifespan_cdf.png    # Active Lifespan CDF
    └── <instance>_access_count.png           # Access Count 直方图（全量 + 去零两张子图）
```

控制台同步输出统计报告。

---

## 6. 多层存储 Per-Tier 输出

启用多层存储（`tier_strategy.hierarchical_eviction_enabled=true` 且至少 2 个 `storages`）时，标准 `*_hit_rates.csv` 会包含：

- `Tier<N>(<tier_name>)_HitTokens`
- `Tier<N>(<tier_name>)_HitRate`
- `AccTier<N>(<tier_name>)_HitRate`
- `Tier<N>(<tier_name>)_BlockNum`

`optimizer_run --draw-chart` 仅在存在 `tier_strategy.hierarchical_eviction_enabled=true` 且至少 2 个 `storages` 的 group 时生成 `timeseries/per_tier_timeseries.png`。所有 `HitRate` / `AccHitRate` / `AccTier*_HitRate` 均为 token hit rate。

---

## 库模块说明

以下模块被 `run/` 入口脚本调用，不直接运行（`radix_tree_plot.py` 除外，见上文）。

### analysis/

| 模块 | 说明 |
|------|------|
| `lifecycle_analysis.py` | Lifecycle CSV 读取、统计计算（lifespan/access/revival 分位数）、绘图数据提取。被 `analyze_lifecycle` 调用 |

### plot/

| 模块 | 说明 |
|------|------|
| `hit_rate_plot.py` | 命中率时序图绘制：读取 `*_hit_rates.csv`，绘制多 instance 双子图（命中率 + 缓存块数 ZOH 对齐）；per-tier 时序图。被 `optimizer_run` 和 `tradeoff` 调用 |
| `radix_tree_plot.py` | RadixTree 可视化：完整树 / 热点路径绘图、热点节点统计。**可独立运行**（见第 4 节） |
| `lifecycle_plot.py` | Physical/Active Lifespan CDF + Access Count 直方图（全量 + 去零两张子图）。被 `analyze_lifecycle` 调用 |

### utils/

| 模块 | 说明 |
|------|------|
| `optimizer_runner.py` | optimizer 运行封装：配置加载、warmup pass、并行实验框架（ThreadPoolExecutor）。被 `optimizer_run`、`tradeoff`、`export_tree` 调用 |
| `csv_loader.py` | CSV 结果加载、容量列表生成（指数分布采样）、`--skip-run` 模式的数据加载。被 `tradeoff` 调用 |
| `plot_utils.py` | 统一绘图风格（`setup_plot_style`）、Pareto 曲线绘图、多策略子图绘图、per-tier 对比曲线。被 `tradeoff` 调用 |
| `window_aggregator.py` | 多实例结果聚合：全局/单实例汇总、时间窗口聚合。被 `multi_instance_replay` 调用 |

---

## 目录结构

```
script/
├── BUILD                         # Bazel 构建定义
├── run/                          # 入口层
│   ├── optimizer_run.py          # 单次运行
│   ├── tradeoff.py               # Pareto 曲线 + per-tier 分析
│   ├── export_tree.py            # 前缀树导出
│   ├── analyze_lifecycle.py      # Lifecycle 分析
│   └── multi_instance_replay.py  # 多实例并行回放 + 聚合
│
├── analysis/                     # 分析层
│   └── lifecycle_analysis.py     # 读取 + 统计 + 数据提取
│
├── plot/                         # 可视化层
│   ├── hit_rate_plot.py          # 命中率时序图 + per-tier 时序图
│   ├── radix_tree_plot.py        # 前缀树可视化
│   └── lifecycle_plot.py         # CDF + 直方图
│
└── utils/                        # 工具层
    ├── optimizer_runner.py       # optimizer 运行封装
    ├── csv_loader.py             # CSV 加载 + 容量列表
    ├── plot_utils.py             # 绘图风格 + Pareto 绘图 + per-tier 曲线
    └── window_aggregator.py      # 多实例聚合
```

---

## 输出目录总览

标准 Python 入口共享输出规则：config 驱动入口使用 `<output_result_path>`；`multi_instance_replay` 使用 `--output-dir`；`analyze_lifecycle` 不读取 config，图表固定输出到输入 lifecycle CSV 所在目录或输入目录下的 `lifecycle/`。

```
<output_result_path 或 output_dir>/
│
│  # ── C++ optimizer 原始数据输出 ──────────────────────────────
├── *_hit_rates.csv                           # 命中率时序（每条 trace 上报）
├── *_template_prefix_traces.csv              # per-trace 模板归属明细（需模板分析）
├── *_template_prefix_summary.csv             # 模板级汇总（需模板分析）
├── *_lifecycle.csv                           # block 生命周期（需 --export-lifecycle）
│
│  # ── Python 图表输出 ────────────────────────────────────────
├── pareto/                                   # tradeoff
│   ├── pareto_curve_<type>.png
│   ├── multi_policy_<type>.png
│
├── timeseries/                               # optimizer_run --draw-chart
│   ├── multi_instance_cache_analysis.png     # tradeoff --plot-timeseries
│   └── per_tier_timeseries.png               # per-tier 时序图（仅分层配置）
│
├── lifecycle/                                # analyze_lifecycle
│   ├── *_physical_lifespan_cdf.png
│   ├── *_active_lifespan_cdf.png
│   └── *_access_count.png
│
├── radix_tree/                               # export_tree
│   ├── *_radix_tree.json
│   ├── *_radix_tree.png
│   └── *_hot_paths.png
│
├── aggregate/                                # multi_instance_replay
│   ├── instance_aggregate.csv
│   ├── global_aggregate.csv
│   ├── global_window_hit_rates.csv           # 需窗口参数
│   └── instance_window_hit_rates.csv         # 需窗口参数 + --include-instance-windows
│
│  # ── tradeoff --save-csv 实验中间数据 ─────────────────────────
└── csv_results/
    └── cap_<N>_<policy>/
        ├── *_hit_rates.csv
        ├── *_template_prefix_traces.csv
        └── *_template_prefix_summary.csv
```
