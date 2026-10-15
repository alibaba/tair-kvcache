# KVCacheManager Optimizer

> [中文](README_zh.md) | English

## Overview

KVCacheManager Optimizer is a standalone cache optimization analysis module. It replays trace data to simulate cache read/write operations, evaluates the impact of different eviction policies and configurations on cache hit rate, and provides parameter optimization capabilities for the KVCacheManager main program.

Core capabilities include:
- Trace data replay and simulation
- Simulation and comparison of multiple eviction policies
- Real-time statistics and analysis of cache hit rate
- Visualization of the Radix Tree index structure
- LiteHit: capacity-independent hit-rate analysis for full-attention scenarios (a single replay produces facts, arbitrary LRU capacities are projected after the fact, with re-blocking and multi block-size fanout supported). See [liteHit/README.md](liteHit/README.md)

## Motivation

In large language model (LLM) inference scenarios, KV Cache management is critical to system performance. Different eviction policies, cache capacity configurations, and storage tier settings significantly affect cache hit rate and overall inference efficiency. The design motivations of the Optimizer module include:

1. **Policy evaluation**: Before deploying to production, evaluate the effectiveness of different eviction policies through trace replay and select the optimal configuration
2. **Parameter tuning**: Analyze cache access patterns and provide optimization suggestions for eviction policy parameters (such as sample rate, TTL, etc.)
3. **Performance prediction**: Predict cache hit rate under different capacity configurations to assist resource planning
4. **Problem diagnosis**: Help understand cache behavior and performance bottlenecks through Radix Tree visualization and detailed statistics

## Features and Architecture

### Core Features

- **Multiple eviction policies**: Supports eviction algorithms such as LRU, RandomLRU, LeafAwareLRU, and TTL
- **TTL expiration mechanism**: Currently adopts V1 semantics; only `POLICY_TTL` performs physical cleanup of TTL-expired entries
- **Tiered storage**: Supports multi-tier storage configuration; currently the feature is not fully complete
- **Trace replay**: Supports multiple trace formats such as Publisher Log and Qwen Bailian
- **Detailed statistics**: Provides detailed statistics such as hit rate and cache usage
- **Flexible configuration**: Flexibly configure instances, storage, and policies through JSON configuration files
- **Visual analysis**: Supports Radix Tree visualization and hit-rate chart generation

Standard strategy configuration, multi-instance replay, trace schema, and hit-rate metrics are described in [docs/strategy_config.md](docs/strategy_config.md). In the standard edition, `HitRate` uniformly denotes the overall token hit rate, i.e. `HitTokens / InputTokens`; local/remote serve only as a diagnostic split between the trace `block_mask` and the optimizer's simulated hits, and are not used as a standard conclusion dimension. Python entry points that pass an optimizer config uniformly use `output_result_path` from the config; `multi_instance_replay` does not read the full config and uses an explicit `--output-dir`. A standard `get` trace must contain `input_len`; when only request-level logs are available externally, `type=request` can be used, and the optimizer will internally schedule delayed writes according to `trace_replay.write_delay_ns`; already-split `get` / `write` traces are still supported (replay path only; LiteHit facts replay recognizes and ignores `write` events, and a `get` submission is treated as write-back complete).

### Architecture Design

```
OptimizerManager (core coordinator)
    ├── OptEvictionManager (eviction manager)
    ├── OptIndexerManager (indexer manager)
    └── OptimizerRunner (trace executor)
        ↓
    ├── Eviction Policies
    ├── RadixTreeIndex (index)
    └── Trace Converter
        ↓
    HitAnalysis (result analysis)
```

### 在线订阅 KVCM 事件

`online_optimizer_server_main` 可以主动发现 KVCM，并通过 KVCM 现有的 Meta gRPC 端口订阅缓存读取事件。KVCM 不需要知道 Optimizer 地址，也不新增事件端口；Optimizer 是 gRPC 客户端，调用 `OptimizerEventStreamService.SubscribeEvents`，KVCM 通过 response stream 写入 `TraceQueryRequest`。

KVCM 侧先在原有 server 配置中启用 optimizer publisher：

```properties
kvcm.event.event_publishers_configs={"log":{"enable":true,"queue_size":10000},"optimizer":{"enable":true,"queue_size":100000,"max_subscribers":4,"subscriber_queue_size":10000}}
```

Optimizer 侧在现有 JSON 配置中加入订阅配置，不使用额外配置文件：

```json
{
    "kvcm_event_subscriptions": [
        {
            "service_discovery_url": "static://127.0.0.1:6381",
            "consumer_id": "online-optimizer",
            "discovery_refresh_interval_ms": 5000
        },
        {
            "service_discovery_url": "vipserver://another-kvcm",
            "consumer_id": "online-optimizer",
            "discovery_refresh_interval_ms": 5000
        }
    ]
}
```

空数组表示不订阅 KVCM。每个数组元素对应一个独立 KVCM 部署；`service_discovery_url` 指向该 KVCM 的 `kvcm.service.rpc_port`，支持通用服务发现 URL（如 `static://`、`vipserver://`、`spectrum://`），同一个 URL 不能重复配置。每个 KVCM 的发现结果只作为 seed：Optimizer 周期调用任一健康 seed 的 `MetaService.GetClusterInfo` 获取当前 Leader，并且只向该 Leader 维持一条 `SubscribeEvents` stream。某个 KVCM 切主后，它自己的下一次刷新会先同步新 Leader 的配置，再关闭旧 stream、连接新 Leader；断流和切主不影响其他 KVCM 的订阅。

部署时也可以通过 `kvcm_optimizer.kvcm_event_subscriptions`（或下划线形式 `kvcm_optimizer_kvcm_event_subscriptions`）传入完整 JSON 数组，不需要为每个 KVCM 增加独立配置文件。

`OptimizerEventStreamService.GetConfiguration` 与事件流共用 KVCM 的 Meta gRPC 端口。Optimizer 启动时及每次服务发现刷新时拉取一次 Instance Group / Instance 快照，先创建缺失的 Group，再注册缺失的 Instance；收到未知 `instance_id` 时还会立即唤醒一次配置刷新。因此 KVCM 新增实例后不需要再提前调用 Optimizer 的注册 API。当前同步只添加新配置，不删除或热更新已经存在的 Optimizer 配置。

自动注册采用能从 KVCM 配置直接确定的口径：Instance Group 的 quota byte 数转换成一个 GiB 容量点，使用当前 Optimizer 支持的 LRU，并开启 prefix hash；Instance 按 full-only 注册。对于包含多个 location spec group 的 KVCM，可在该订阅中显式设置 `"full_location_spec_group_name": "full_cache"`：仅所选组计入 Full block 字节数，不注册 linear 状态，也不计入 Mamba 容量；所选组不存在时配置同步报错。不设置时保留单组自动推断行为，多组实例跳过并记录 unsupported。当前在线 indexer 不支持 shared group quota，因此同组各 Instance 分别按完整 Group quota 模拟。这里不猜测 linear/mamba 周期。

每个 KVCM 订阅器固定使用两个线程：一个 supervisor 线程负责服务发现、Leader 查询和配置同步，一个 stream 线程负责读取当前 Leader 的事件；收到事件后直接调用 `OnlineOptimizerManager`，不增加额外事件队列。未知 Instance 的首条事件会记录并丢弃，并且只唤醒对应 KVCM 的配置刷新；配置刷新完成后的后续事件正常进入统计。事件时间戳用于 LiteHit 和线性 indexer 的 TTL 判定，旧客户端未设置时间戳时仍回退到 Optimizer 本机墙钟。

在线 full-attention 实例还会输出 `mrc` gauge（Prometheus 名称默认为 `kvcm_optimizer_mrc`，标签为 `instance_group`、`instance_id` 和 `target_hit_rate_percent`，单位 byte）。`target_hit_rate_percent` 是相对于本上报窗口理论最大可命中量的比例，不是绝对请求命中率：目标命中量等于窗口理论无限容量最大可命中 block 数乘以该比例，`mrc` 则表示保留这些目标命中所需的最小 LRU 容量。当前固定输出 60%、80%、90%、95%、99%、99.5% 六个相对目标。例如理论最大命中率为 68.6% 时，95% 相对目标对应约 65.17% 的绝对命中率，而不是 95%。每次上报会原子取走并清空仅供 MRC 使用的 hit curve，不影响查询数、命中率等累计指标。该值直接聚合 LiteHit 产生的容量无关 hit curve，不依赖预先配置的离散容量点；周期内尚无理论可命中 block 时值为 0。

在线 LiteHit full-attention 实例统计 block reuse interval（同一 block 两次有效访问之间的 producer 时间差）。每个监控上报周期输出 `trace_query_reuse_interval_min_seconds`、`trace_query_reuse_interval_max_seconds`、`trace_query_reuse_interval_avg_seconds`、`trace_query_reuse_interval_p95_seconds`、`trace_query_reuse_interval_p99_seconds`、`trace_query_reuse_interval_p995_seconds`（p99.5）和 `trace_query_reuse_interval_p999_seconds`（p99.9），单位为秒，标签为 `instance_group`、`instance_id`。每次上报后清空本周期样本，但保留各 block 的上次访问时间；均值按本周期所有 block 复用样本加权，不是请求均值的平均。空周期的 Prometheus 时间值为 NaN，KMonitor 不发送该周期的时间值；真实的零间隔仍正常输出 0。只在内部保留本周期样本数以计算平均值，不维护或上报累计 reuse count。统计容量无关，包含前缀 miss 后仍存活的 block；单请求内重复 key 最多计一次，首次访问及 TTL 已过期 block 不产生样本。时间戳未设置时使用 Optimizer 本机到达时间。p95/p99/p99.5/p99.9 采用 nearest-rank（ceil(N×p)）口径，使用固定 58 KiB/Instance 的内部对数直方图近似，以桶上界估计并限制不超过本周期 max；非零时间值的向上相对误差小于 1/128（约 0.78125%），零间隔精确。直方图随周期重置，不保存全部样本，不对外输出分桶占比；各分位数不能跨周期或跨 Instance 直接平均成总体分位数。

当前仅 LiteHit full-attention 路径提供原生 reuse interval 统计；旧 linear indexer 不上报这些指标，也不再使用旧 hit-age 分桶作为兼容数据源。

### P99 retrospective capacity estimate

Full-attention instances with theoretical analysis enabled keep a bounded,
in-memory history: 288 fixed five-minute buckets (approximately 24 hours).
History uses MRC99 and reuse-interval P99 from the same **non-overlapping five-minute
window**, aligned to wall-clock multiples of five minutes. Each window is finalized
once on the first subsequent request or report; no intermediate 30-second history
estimate is created. Ordinary MRC and reuse gauges still drain their own reporting
windows (normally 30 seconds), without clearing the history inputs. One additional
58 KiB reuse histogram and one sparse MRC accumulator hold the current window;
there are no rolling slices or per-request sample queues. Percentiles are computed
from sample counts, not averages of per-report percentiles. Local receipt time
determines membership; producer timestamps still determine reuse durations.
Startup contributes only available samples to its first partial window. Empty
windows create no observations. The completed window is anchored at its end minus
one millisecond (`t`), even if the report is delayed, and the MRC capacity is
backfilled across `[t - reuse_p99, t]`;
overlapping observations take the **maximum**, never the sum. Both interval
endpoints round down to five-minute bucket starts, so short/zero reuse still
updates the current bucket. This is a window-level heuristic, not an exact
per-key reconstruction or a guarantee of the true minimum cache capacity.

Two Gauge snapshots are exported with `instance_group`, `instance_id`, and
fixed `slot` (0–287) labels: `trace_query_capacity_history_p99_timestamp_ms`
and `trace_query_capacity_history_p99_bytes`. A timestamp of 0 means unobserved;
zero bytes with a positive timestamp is a valid observation. Slots expire even
in idle reporting windows. Restart, instance recreation, and ResetStats clear
the history. Clock regression does not resurrect expired history. New requests
can revise retained older buckets, but cannot revise history already discarded.
There is no per-request history backfill and no external queue/storage dependency.
Requests accumulate the separate five-minute input distributions and may finalize
the previous window at a boundary. Reporters export the retained history every
30 seconds; they do not store extra history points or recalculate it at that rate.
The core ring uses about 4.5 KiB/Instance; metric registry/tag overhead is extra,
and these metrics add 576 series per eligible Instance.

The fixed-slot metrics above remain for compatibility. The live revisable panel
uses immutable-timestamp v2 metrics and Range queries that include later reports,
so selecting a past display interval does not freeze its revisions at the selected
end. See [the v2 history query contract](../../docs/design/optimizer_capacity_history_v2.md).
Previously unavailable observations are not recovered on deployment.

### 所选时间范围的 token 加权命中率

full-attention 路径周期上报以下累计分析状态，Prometheus 默认加 `kvcm_optimizer_` 前缀。
这些指标沿用 `trace_query_blocks_total` 的 **Gauge 快照类型**，数值具有累计计数语义：
正常运行只增不减，`ResetStats`、实例重建或进程重启后归零；并非每个上报周期的增量。
KMonitor 同样按累计 Gauge 上报，不使用会转换成每秒速率的 QPS 类型。

| 指标后缀 | 标签 | 累计值 |
| --- | --- | --- |
| `trace_query_input_tokens_total` | `instance_group`, `instance_id` | 成功进入分析的规范化输入 token 总数，包含不完整尾块对应的 token |
| `trace_query_max_hit_tokens_total` | 同上 | 理论最大命中 block 总数 × 该 Instance 的 block size；未启用理论分析时不输出 |
| `trace_query_hit_tokens_total` | 同上，加 `capacity_gb` | 指定模拟容量命中 block 总数 × block size |

分子、分母取自同一个加锁的 InstanceSummary，不混用 service 输入计数，也不从命中率反推计数。
首次访问和 TTL 过期后的访问仍计入输入，命中按分析器结果计入；TTL 驱逐不扣减历史命中。
空闲时累计值保持不变。旧 linear 路径不输出这些 token 指标，避免把 block 命中率伪装成 token 命中率。

对于一个指定 exporter/Instance，所选时间范围 `W` 的理论加权值为
`increase(kvcm_optimizer_trace_query_max_hit_tokens_total[W]) / increase(kvcm_optimizer_trace_query_input_tokens_total[W])`。
这里对具有 counter 语义的累计浮点快照使用 `increase`，不能使用会忽略归零的普通差值，
也不能用 `avg_over_time(interval_query_*hit_rate[W])` 替代 token 加权。
指定容量的分子换成 `trace_query_hit_tokens_total`，分母通过 `ignoring(capacity_gb) group_left` 匹配。

Grafana Table/Stat 使用 **Instant** 查询，评估时刻为所选范围结束时间，`W` 使用 `$__range`。
有多个 exporter 时必须对每条原始序列先 `increase` 再按业务标签求和，之后相除；
保留 workspace/application/deployment/instance group/instance 的隔离，容量只保留在有限容量分子上。
不要先把累计值求和再 `increase`，也不要跨容量累加同一份输入；重复采集同一 exporter 的副本应先由采集链路去重。
面板单位用 `percentunit`，分母增量为 0 时显示无样本，不补成 0% 或 100%。

注意：这是同口径累计计数的时间范围统计，不是请求级审计。
`increase` 会校正可见的计数归零并对窗口边界外推（见 [Prometheus 文档](https://prometheus.io/docs/prometheus/latest/querying/functions/#increase)）。
新版本上线前没有历史 token 计数，不能回填；窗口内至少需要两个有效样本。
采集空洞、不可见的重置及边界处只有部分上报周期时仍存在监控采样误差，短于上报周期的窗口不能精确还原。

### Eviction Policies

**LRU (Least Recently Used)**
- Maintains a doubly linked list recording the access order of blocks; the most recently accessed block is at the head of the list, and the least recently accessed block is at the tail

**RandomLRU**
- Combines random sampling with the LRU policy: randomly samples a number of blocks from the cache and evicts the least recently accessed one among them

**LeafAwareLRU**
- Adds leaf-node awareness on top of LRU, preferentially evicting blocks within leaf nodes

**TTL (Time-To-Live)**
- Two-phase eviction: first clears all TTL-expired blocks; if capacity still exceeds the limit, falls back to evicting from the oldest by `last_access_time`
- `fallback_on_pressure` (default true): when disabled, degrades to pure TTL, only reclaiming expired blocks with no capacity fallback

#### TTL Expiration Mechanism (V1)

Current implementation semantics:

- Only instances with `eviction_policy_type: "ttl"` perform physical TTL expiration cleanup before reads/writes.
- Under non-TTL policies (`lru` / `random_lru` / `leaf_aware_lru`), TTL is treated as nonexistent (neither pre-cleanup nor logical expiration judgment is performed).

| Usage mode | Configuration | Eviction behavior |
|---|---|---|
| Pure LRU capacity management | `eviction_policy_type: "lru"` | LRU evicts by access time |
| TTL priority + capacity fallback | `eviction_policy_type: "ttl"` + `fallback_on_pressure: true` | First clears all expired blocks; if insufficient, falls back by list tail |
| Pure time eviction (no fallback) | `eviction_policy_type: "ttl"` + `fallback_on_pressure: false` | Only clears expired blocks; ignores capacity shortage |

**TTL behavior rules**:
- `default_block_ttl_seconds`: configured at the instance group level; `0` means TTL is disabled at the group level.
- `ttl_refresh_on_read`: controls whether a read hit refreshes the TTL anchor; default `true` (Sliding TTL); when `false`, reads do not extend lifetime.
- When TTL is disabled at the group level, request-level `ttl_seconds > 0` does not take effect (the write path forcibly disables TTL).
- On write, `ttl_anchor_time` is reset to the write time, and TTL starts counting from that anchor.
- On read (`PrefixQuery`), `last_access_time` is always refreshed; `ttl_anchor_time` is refreshed only when `ttl_refresh_on_read=true`.
- Before a read/write request is executed, a physical cleanup of TTL-expired blocks is performed first, and `CleanEvictedBlocks` uniformly handles node cleanup.
- After a write completes, `CheckAndEvict` handles capacity eviction; when `fallback_on_pressure=false`, capacity eviction is skipped.
- `POLICY_TTL` expiration cleanup has been optimized to incremental reclamation based on a min-heap of expiration times, avoiding a full list scan on every request.

**TTL lifecycle statistics notes (important)**:
- Block lifecycle statistics in TTL mode (`birth_time_us/death_time_us/lifespan_us`) only reflect the time points at which the system actually processes events, not the real expiration moments.
- Because the current implementation uses lazy eviction after write, `death_time_us` records the time when the block is cleaned up/processed by the system, not strictly `last_access_time + ttl`.
- Therefore, lifecycle data in TTL scenarios should only be used for relative trend analysis, and is not recommended for precise duration estimation or precise cross-policy comparison.

**Per-request TTL override** (via the `WriteCache` API):

| `ttl_seconds` value | Meaning |
|---|---|
| `0` (default) | Use the group's `default_block_ttl_seconds` |
| `-1` | Disable TTL; the block never expires |
| `>0` | Custom TTL (seconds); if the group has disabled TTL (`default_block_ttl_seconds=0`), this value is ignored |

### Trace Types

```
OptimizerSchemaTrace (base class)
    ├── GetLocationSchemaTrace (read operation)
    └── WriteCacheSchemaTrace (write operation)
```

**Supported Trace Formats**
- **Publisher Log**: KVCacheManager Event Publisher log, distinguishing read and write requests
- **Qwen Bailian**: Qwen Bailian open-source dataset format; forcibly distinguishes read and write requests after conversion

## Quick Start

### Step 1: Convert trace to standard format

```bash
cd tools/trace_converter

# Install dependencies (first time only)
pip install -r requirements.txt

# Convert trace
python trace_converter.py \
    -i /path/to/your_trace.jsonl \
    -o /path/to/optimizer_trace.jsonl \
    -f qwen_bailian \
    --mode optimizer
```

### Step 2: Build the Optimizer

```bash
bazel build \
    //kv_cache_manager/optimizer:optimizer_main \
    //kv_cache_manager/optimizer/analysis/script:optimizer_run
```

### Step 3: Create a configuration file

Create a JSON configuration file. Below is a minimal usable configuration for non-tiered LRU; the `instance_id` in the trace must match the `instances[].instance_id` here.

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


### Step 4: Run the Optimizer

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json
```

After running, the following is generated in the directory specified by `output_result_path`:

- `{instance_id}_hit_rates.csv` - hit-rate data for each instance

To draw a hit-rate time-series chart, add `--draw-chart`:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json \
    --draw-chart
```

Quick reference for common entry points:

| Need | Recommended entry point |
|---|---|
| Single replay, output `*_hit_rates.csv` | `optimizer_run -c config.json` |
| Single replay with time-series chart | `optimizer_run -c config.json --draw-chart` |
| Infinite-capacity theoretical hit rate | Set `quota_capacity` or tier `capacity` to `-1` in config, then run `optimizer_run` |
| Non-tiered capacity Pareto | `tradeoff -c config.json --num-points 30` |
| Multi eviction-policy Pareto | `tradeoff -c config.json --eviction-policies lru random_lru leaf_aware_lru ttl` |
| Per-pod/instance independent cache replay | `multi_instance_replay --trace-dir ... --output-dir ...` |
| Export lifecycle CSV | `optimizer_run -c config.json --export-lifecycle` |
| Analyze lifecycle charts | `analyze_lifecycle -i <lifecycle.csv or dir>` |
| Export/visualize RadixTree | `export_tree -c config.json --show-hot-paths` |

### Visual Analysis

The Optimizer module provides a variety of visual analysis tools for analyzing cache performance, hit rate, and Radix Tree structure.

#### Hit Rate Over Time Chart

Run the optimizer to analyze the trace and draw multi-instance cache analysis charts, showing the sum of storage capacity across all instances and their respective hit rates over time.

**Note**: The `trace_file_path` in the configuration file must be a standard-format trace file.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json \
    --draw-chart
```

**Output**: `{output_result_path}/timeseries/multi_instance_cache_analysis.png`

#### Radix Tree Visualization

Export and visualize the prefix tree structure, counting and displaying hot nodes and the prefix paths of their associated nodes.
See `analysis/script/run/export_tree.py` for specific configuration.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:export_tree -- \
    -c /path/to/config.json
```

**Output**:
- `{output_result_path}/radix_tree/{instance_id}_radix_tree.json` - Radix Tree export data
- `{output_result_path}/radix_tree/{instance_id}_radix_tree.png` - Radix Tree visualization chart

#### Pareto Capacity-Hit-Rate Curve Analysis

The unified entry point is `tradeoff`. It first uses infinite-capacity warmup to obtain the theoretical hit rate and maximum cache size, then replays the optimizer per capacity point and draws the Pareto curve of capacity versus hit rate.

> **Scope of applicability**: Tradeoff only applies to non-tiered mode. In tiered mode, the capacity scan only modifies `quota_capacity` and does not modify each tier's `storages[i].capacity`, so it cannot represent the real tiered capacity tradeoff.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/config.json \
    --num-points 30 \
    --min-capacity-ratio 1e-4 \
    --hit-rate-type total \
    --max-workers 4
```

**Parameter description**:
- `-c, --config` - configuration file path
- `--num-points` - maximum number of capacity sampling points to run per policy (default 30); stops early after reaching 99% of the theoretical hit rate
- `--min-capacity-ratio` - minimum capacity point relative threshold (default `1e-4`)
- `--hit-rate-type` - hit-rate type: total/local/remote/all (default total)
- `--max-workers` - maximum number of parallel execution threads (default 4)
- `--plot-title` - override the chart title

**Output**: `{output_result_path}/pareto/pareto_curve_{hit_rate_type}.png`

#### Multi-Policy Comparison Analysis

Compare multiple eviction policies via `--eviction-policies`. Each policy warms up independently, computes its theoretical hit rate independently, and independently stops after reaching 99% of the theoretical hit rate.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/config.json \
    --eviction-policies lru random_lru leaf_aware_lru ttl \
    --num-points 30 \
    --hit-rate-type total \
    --max-workers 4
```

**Parameter description**:
- `-c, --config` - configuration file path
- `--eviction-policies` - list of eviction policies to compare (default lru random_lru leaf_aware_lru ttl)
- `--num-points` - maximum number of capacity sampling points to run per policy (default 30); stops early after reaching 99% of the theoretical hit rate
- `--hit-rate-type` - hit-rate type: total/local/remote/all (default total)
- `--max-workers` - maximum number of parallel execution threads (default 4)

**Output**: `{output_result_path}/pareto/multi_policy_{hit_rate_type}.png`

Chart rules: the X axis is `Capacity (GB)`, the Y axis is `HitRate (%)`; the curve starts from `(0 GB, 0%)` and only the rising segment is drawn; the 95% and 99% theoretical hit rates are marked with interpolation intersections, dashed lines, and labels. If a capacity point shows a decrease, it is removed from the chart and printed in the log, while the original CSV is retained.

### Python Interface Example

```python
from kv_cache_manager.optimizer import OptimizerConfigLoader, OptimizerLoader, OptimizerManager

# Load configuration
config_loader = OptimizerConfigLoader()
config = config_loader.Load("/path/to/config.json")

# Create the optimizer
optimizer = OptimizerManager(config)
optimizer.Init()

# Run
optimizer.DirectRun()

# Analyze results
optimizer.AnalyzeResults()

# Single read/write operations (instance_id must be specified)
write_res = optimizer.WriteCache("instance_id", "trace_001", timestamp, block_ids)
read_res = optimizer.GetCacheLocation("instance_id", "trace_002", timestamp, block_ids, block_mask,
                                      input_len=real_prompt_tokens)

# Specify TTL on write (optional)
write_res = optimizer.WriteCache("instance_id", "trace_003", timestamp, block_ids,
                                  ttl_seconds=0)     # Use group default
write_res = optimizer.WriteCache("instance_id", "trace_004", timestamp, block_ids,
                                  ttl_seconds=-1)    # Disable TTL, never expires
write_res = optimizer.WriteCache("instance_id", "trace_005", timestamp, block_ids,
                                  ttl_seconds=300)   # Custom 300 seconds

# Clear cache (keep statistics)
optimizer.ClearCache("instance_id")        # Clear the specified instance
optimizer.ClearAllCaches()                 # Clear all instances

# Clear cache and reset statistics
optimizer.ClearCacheAndResetStats("instance_id")  # Clear the specified instance and reset statistics
optimizer.ClearAllCachesAndResetStats()           # Clear all instances and reset statistics
```

### Configuration Parameter Description

| Parameter | Description |
|------|------|
| eviction_mode | Eviction mode: 1=GROUP_ROUGH, 2=INSTANCE_ROUGH, 3=INSTANCE_PRECISE |
| eviction_policy_type | Eviction policy type: lru, random_lru, leaf_aware_lru, ttl |
| quota_capacity | Total group capacity in non-tiered mode, in GB; `-1` means infinite capacity, no capacity eviction triggered |
| storages[].capacity | Capacity of a single tier in tiered mode, in GB; `-1` means infinite capacity for that tier |
| tier_strategy | Multi-tier read/write policy package, including hierarchical eviction switch, write mode, read access propagation, promote and selective write thresholds; top-level fields are the default policy for all adjacent tier edges |
| tier_strategy.write_mode | Write/sink mode: `write_through`, `cascading`, `write_through_selective` |
| tier_strategy.access_propagation_enabled | Whether to refresh the access time of lower-tier copies when a read hits an upper-tier copy; it is not a write mode |
| tier_strategy.tier_flows | Policy overrides for adjacent tier edges; edges not overridden inherit the `tier_strategy` default policy |
| bytes_per_token | KV size per token; Python analysis scripts use it together with `block_size` to convert block capacity to GB |
| default_block_ttl_seconds | Default TTL (seconds) at the instance group level; 0 = disable TTL |
| ttl_refresh_on_read | TTL refresh switch at the instance group level: true=read refreshes, false=fixed window |
| fallback_on_pressure | TTL policy parameter: whether to fall back to LRU when expiration is insufficient (default true) |

## Use Cases

The following cases all assume the input is already a standard optimizer JSONL trace. Standard conclusions use token hit rate, i.e. `AccHitRate = AccHitTokens / AccInputTokens` in the last row of the CSV. The JSON in the examples mostly shows only one item in `instance_groups[]`; a complete config still requires the top-level `trace_file_path`, `output_result_path`, and `eviction_params`.

### Case 1: Global Pooled Infinite-Capacity Theoretical Hit Rate

Used to answer "if capacity were infinite, how much could the current trace theoretically hit at most". In a non-tiered global pooled configuration, set `quota_capacity` to `-1`:

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

Run:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/global_pool_unlimited.json
```

For results, look at `AccHitRate`, `AccInputTokens`, and `AccHitTokens` in the last row of `{output_result_path}/global_hit_rates.csv`.
If the trace uses real pod names as `instance_id`, you need to change `instances[].instance_id` in the example above to the corresponding names.

### Case 2: Single-Tier Finite-Capacity Replay

Used to evaluate the actual token hit rate under a given capacity. Non-tiered mode only requires setting `quota_capacity`, in GB:

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

`block_size * bytes_per_token` determines the KV capacity corresponding to one block; Pareto charts and capacity output all depend on this conversion.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/finite_pool.json \
    --draw-chart
```

Output:

- `{output_result_path}/*_hit_rates.csv`
- `{output_result_path}/timeseries/multi_instance_cache_analysis.png`

### Case 3: HBM + DRAM Tiered Replay, write-through, reads do not update lower tier

Used to simulate online multi-tier caching. When `hierarchical_eviction_enabled=true`, each tier evicts independently according to `storages[].capacity`; in this case `quota_capacity` is only a reserved field and is not used for per-tier eviction.

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

Run:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/tiered_hbm_dram.json \
    --draw-chart
```

For tiered results, look at `Tier<N>(name)_HitTokens`, `AccTier<N>(name)_HitRate`, and `Tier<N>(name)_BlockNum`; the chart is at `{output_result_path}/timeseries/per_tier_timeseries.png`.

### Case 4: Three-Tier HBM + DRAM + L3, first two tiers write-through, DRAM to L3 cascading

When the policy differs per edge, use `tier_flows` to override the default policy:

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

The top-level fields of `tier_strategy` are the default edge policy; `tier_flows` only overrides the specified adjacent edges.

### Case 5: Per-Pod Independent Cache Replay with Global Hit-Rate Aggregation

When each online pod/instance caches independently, use `multi_instance_replay`. Each JSONL file in the input directory can only contain one `instance_id`.

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

Output:

- `<output_dir>/<instance_id>_hit_rates.csv`: replay result for each pod
- `<output_dir>/aggregate/instance_aggregate.csv`: summary per pod
- `<output_dir>/aggregate/global_aggregate.csv`: overall hit rate after aggregating all pods
- `<output_dir>/aggregate/global_window_hit_rates.csv`: window-level overall hit rate

The current `multi_instance_replay` CLI directly supports two tiers, L1/L2. When L3 or more complex topologies are needed, use a full optimizer config or extend the script.

### Case 6: Capacity Pareto Chart

Non-tiered capacity scanning uses `tradeoff`. The script first runs an infinite-capacity warmup to obtain the theoretical hit rate, then generates capacity points; it stops after reaching 99% of the theoretical hit rate.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/global_pool_config.json \
    --num-points 30 \
    --min-capacity-ratio 1e-4 \
    --hit-rate-type total \
    --max-workers 8 \
    --plot-title "service-a Pareto"
```

Multi-policy comparison:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/global_pool_config.json \
    --eviction-policies lru random_lru leaf_aware_lru ttl \
    --num-points 30 \
    --hit-rate-type total \
    --max-workers 8
```

Output:

- `{output_result_path}/pareto/pareto_curve_total.png`
- `{output_result_path}/pareto/multi_policy_total.png`

The chart marks the capacities corresponding to 95%/99% theoretical hit rate; decreasing points are only removed from the chart, and the original CSV is not modified.

### Case 7: Block Lifecycle Analysis

First export the lifecycle CSV during replay:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json \
    --export-lifecycle
```

Then generate lifecycle statistics and charts:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:analyze_lifecycle -- \
    -i /path/to/output_result_path
```

To print statistics only, without drawing charts:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:analyze_lifecycle -- \
    -i /path/to/output_result_path \
    --stats-only
```

### Case 8: RadixTree Hot Path Troubleshooting

Used to inspect hot prefixes, leaf nodes, and cached/evicted blocks:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:export_tree -- \
    -c /path/to/config.json \
    --show-hot-paths \
    --hot-nodes 50 \
    --show-blocks
```

Output:

- `{output_result_path}/radix_tree/{instance_id}_radix_tree.json`
- `{output_result_path}/radix_tree/{instance_id}_hot_paths.png`

### Case 9: Result Reading Metrics

Standard reports only look at token hit rate:

| Metric | Description |
|---|---|
| `InputTokens` / `AccInputTokens` | Current / cumulative input token count |
| `HitTokens` / `AccHitTokens` | Current / cumulative hit token count |
| `HitRate` | Current request token hit rate |
| `AccHitRate` | Cumulative token hit rate; the main conclusion looks at this column |
| `ReadBlocks` / `HitBlocks` | Block-level diagnostic fields, not used as the final hit-rate metric |
| `Tier<N>(name)_HitTokens` | Tiered hit token count |
| `AccTier<N>(name)_HitRate` | Tiered cumulative token hit rate |


## Trace Input Format

### Overview

The Optimizer only accepts standard-format trace files. Use a standalone Python tool to convert various trace formats to the standard format.

### Standard Format

The Optimizer supports two standard trace types:

- **GetLocationSchemaTrace**: read operation (prefill phase)
- **WriteCacheSchemaTrace**: write operation (decode phase)

**Recommendation**: Unify all Optimizer inputs to the Get+Write format to preserve precise read/write timing.
A standard `get` trace must contain `input_len`; `keys` can only contain complete block keys — trailing tokens that do not fill a full block are not written into `keys`, but are still counted into `InputTokens`.

---

### Conversion Tool

Use a standalone Python tool to convert various formats (no bazel required):

```bash
# Enter the tool directory
cd tools/trace_converter

# Install dependencies (first time only)
pip install -r requirements.txt

# Convert trace to Optimizer format
python trace_converter.py \
    -i your_trace.log \
    -o optimizer_trace.jsonl \
    -f <format> \
    --mode optimizer

```

**Supported formats**:
- `publisher_log`: KVCacheManager Event Publisher log
- `qwen_bailian`: Qwen Bailian open-source dataset
- `text`: text conversation (requires specifying --tokenizer-path)

**Auto-discovery of Converters**:

The system automatically scans and discovers all available converters:

```bash
# Use a built-in converter
python3 trace_converter.py -i input.jsonl -o output.jsonl -f qwen_bailian

# Use a custom converter
python3 trace_converter.py -i input.jsonl -o output.jsonl -f custom \
    --converter-module /path/to/custom_converter.py
```

See: [Trace Converter documentation](tools/trace_converter/README.md)

---

### Configuration File

**New configuration**:
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

**Note**:
- `trace_file_path` must be a standard-format file

---

### Usage Example

Complete workflow:

```bash
# Step 1: Convert trace
cd tools/trace_converter
python trace_converter.py \
    -i /path/to/qwen_trace.jsonl \
    -o /path/to/optimizer_trace.jsonl \
    -f qwen_bailian \
    --mode optimizer

# Step 2: Run the Optimizer
cd ../..
bazel run //kv_cache_manager/optimizer:optimizer_main -- /path/to/config.json
```

---

### Adding a Custom Trace Converter

If you need to support a new trace format, add a new converter in the Python tool:

1. **Create a Converter class**: create a new file in any directory
   ```python
   from converters.base import BaseConverter
   
   class MyCustomConverter(BaseConverter):
       def convert(self, input_file: str, output_file: str) -> int:
           # Implement conversion logic
           pass
   ```

2. **Usage - Option 1 (auto-scan directory)**:
   ```bash
   # Place the converter file in the specified directory; automatically discovers all classes inheriting BaseConverter
   python trace_converter.py -i input.log -o output.jsonl -f my_custom \
       --converter-dir /path/to/your/converters
   ```

3. **Usage - Option 2 (explicitly register file)**:
   ```bash
   # Directly specify the converter file and class name
   python trace_converter.py -i input.log -o output.jsonl -f my_custom \
       --converter-module /path/to/my_custom_converter.py:MyCustomConverter
   ```

**No need to modify the `trace_converter.py` source code** — the Converter automatically infers the format name from the class name:
- `MyCustomConverter` → `my_custom`
- `QwenBailianConverter` → `qwen_bailian`

---

### Related Documentation

- [Trace Converter tool documentation](tools/trace_converter/README.md) - detailed description of the Python conversion tool
