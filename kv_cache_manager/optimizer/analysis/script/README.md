# Optimizer 分析脚本

KVCacheManager optimizer 的分析与可视化工具集。

Bazel target 前缀：

```
//kv_cache_manager/optimizer/analysis/script
```

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

### 输出

- `<output_result_path>/*_hit_rates.csv` — 每个 instance 的命中率时序数据
- `<output_result_path>/multi_instance_cache_analysis.png` — 命中率时序图（需 `--draw-chart`）
- `<output_result_path>/*_lifecycle.csv` — block 生命周期数据（需 `--export-lifecycle`）

---

## 2. Pareto 曲线分析 — `tradeoff`

在多个容量点上运行 optimizer，绘制容量-命中率权衡曲线。自动判断单策略/多策略模式。

### 单策略模式

不指定 `--eviction-policies`，使用配置文件中的默认策略。每个 instance 一条曲线。

```bash
# 默认 40 个容量点
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- -c config.json

# 自定义采样点数
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- -c config.json --num-points 30

# 保存 CSV + 生成每个容量点的时序图
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- -c config.json --save-csv --plot-timeseries

# 从已有 CSV 加载（跳过实验）
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
| `--eviction-policies` | — | 配置默认 | 驱逐策略列表（空格分隔） |
| `--warmup-capacity` | — | 30000000 | warmup 阶段容量 |
| `--num-points` | — | 40 | 容量采样点数（指数分布） |
| `--hit-rate-type` | — | total | 命中率类型：total / internal / external / all |
| `--max-workers` | — | 4 | 并行实验线程数 |
| `--save-csv` | — | false | 保留每次运行的 CSV 文件 |
| `--csv-output-dir` | — | `<output>/csv_results` | CSV 保存目录 |
| `--export-lifecycle` | — | false | 导出 lifecycle CSV |
| `--skip-run` | — | false | 跳过实验，从已有 CSV 加载 |
| `--plot-timeseries` | — | false | 为容量点生成时序图（需 `--save-csv` 或 `--skip-run`） |
| `--plot-capacity` | — | 全部 | 只为指定容量点生成时序图（空格分隔） |
| `--x-min/--x-max` | — | 自动 | X 轴（容量）范围 |
| `--y-min/--y-max` | — | 0~1 | Y 轴（命中率）范围 |

### 输出

- `pareto_curve_<type>.png` — 单策略 Pareto 散点图
- `multi_policy_<type>.png` — 多策略对比子图
- `csv_results/cap_<capacity>_<policy>/` — 每次运行的 CSV（需 `--save-csv`）

---

## 3. 前缀树可视化 — `export_tree`

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
| `-o, --output-dir` | — | `<output_result_path>` | 输出目录 |
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

- `<instance>_radix_tree.json` — 前缀树结构数据
- `<instance>_radix_tree.png` — 完整树可视化
- `<instance>_hot_paths.png` — 热点路径可视化

---

## 4. Block Lifecycle 分析 — `analyze_lifecycle`

分析 block 的生命周期数据，产出统计报告 + CDF 图 + Access Count 直方图。

```bash
# 分析单个文件
bazel run //kv_cache_manager/optimizer/analysis/script:analyze_lifecycle -- \
    -i instance_lifecycle.csv

# 分析单个文件，输出到指定目录
bazel run //kv_cache_manager/optimizer/analysis/script:analyze_lifecycle -- \
    -i instance_lifecycle.csv -o results/

# 批量分析目录下所有 *_lifecycle.csv
bazel run //kv_cache_manager/optimizer/analysis/script:analyze_lifecycle -- \
    -i output_dir/ -o results/

# 只看统计信息（不生成图表）
bazel run //kv_cache_manager/optimizer/analysis/script:analyze_lifecycle -- \
    -i instance_lifecycle.csv --stats-only
```

### 参数

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `-i, --input` | ✅ | — | lifecycle CSV 文件或包含 `*_lifecycle.csv` 的目录 |
| `-o, --output-dir` | — | 输入文件所在目录 | 图表输出目录 |
| `--stats-only` | — | false | 只打印统计信息，不生成图表 |

### 统计报告内容

| 类别 | 指标 |
|------|------|
| 总体概览 | 总 block 数、唯一 BlockKey 数、复活率、最多复活次数、即逐占比（lifespan=0）、零访问占比、缓存利用率 |
| Physical Lifespan | 均值、中位数、标准差、min/max、P25/P75/P90/P95/P99 |
| Active Lifespan | 均值、中位数、标准差、min/max、P25/P75/P90/P95/P99 |
| Access Count | 均值、中位数、标准差、min/max、P25/P75/P90/P95/P99 |

### 输出

- 控制台统计报告
- `<instance>_physical_lifespan_cdf.png` — Physical Lifespan CDF（全量 + Evicted）
- `<instance>_active_lifespan_cdf.png` — Active Lifespan CDF
- `<instance>_access_count.png` — Access Count 直方图（全量 + 去零两张子图）

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
| `hit_rate_plot.py` | 命中率时序图绘制：读取 `*_hit_rates.csv`，绘制多 instance 双子图（命中率 + 缓存块数 ZOH 对齐）。被 `optimizer_run` 和 `tradeoff` 调用 |
| `radix_tree_plot.py` | RadixTree 可视化：完整树 / 热点路径绘图、热点节点统计。**可独立运行**（见第 3 节） |
| `lifecycle_plot.py` | Physical/Active Lifespan CDF + Access Count 直方图（全量 + 去零两张子图）。被 `analyze_lifecycle` 调用 |

### utils/

| 模块 | 说明 |
|------|------|
| `optimizer_runner.py` | optimizer 运行封装：配置加载、warmup pass、并行实验框架（ThreadPoolExecutor）。被 `optimizer_run`、`tradeoff`、`export_tree` 调用 |
| `csv_loader.py` | CSV 结果加载、容量列表生成（指数分布采样）、`--skip-run` 模式的数据加载。被 `tradeoff` 调用 |
| `plot_utils.py` | 统一绘图风格（`setup_plot_style`）、Pareto 曲线绘图、多策略子图绘图。被 `tradeoff` 调用 |

---

## 目录结构

```
script/
├── BUILD                         # Bazel 构建定义
├── run/                          # 入口层
│   ├── optimizer_run.py          # 单次运行
│   ├── tradeoff.py               # Pareto 曲线
│   ├── export_tree.py            # 前缀树导出
│   └── analyze_lifecycle.py      # Lifecycle 分析
│
├── analysis/                     # 分析层
│   └── lifecycle_analysis.py     # 读取 + 统计 + 数据提取
│
├── plot/                         # 可视化层
│   ├── hit_rate_plot.py          # 命中率时序图
│   ├── radix_tree_plot.py        # 前缀树可视化
│   └── lifecycle_plot.py         # CDF + 直方图
│
└── utils/                        # 工具层
    ├── optimizer_runner.py       # optimizer 运行封装
    ├── csv_loader.py             # CSV 加载 + 容量列表
    └── plot_utils.py             # 绘图风格 + Pareto 绘图
```