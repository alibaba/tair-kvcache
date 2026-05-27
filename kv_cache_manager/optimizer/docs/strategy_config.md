# Optimizer 标准策略配置说明

本文档定义标准版 optimizer 的策略配置、multi-infer replay 入口和命中率口径。后续新增实验脚本应复用这里的字段语义，避免在脚本里引入另一套配置命名。

## 标准回放模式

optimizer 按仿真问题分为三个标准入口：

| 模式 | 入口 | 适用场景 | instance 语义 | local / remote 语义 |
|---|---|---|---|---|
| KVCM/L3-only | `optimizer_main` / `optimizer_run` | 只模拟 KVCM/L3，或直接回放 KVCM 日志 | KVCM instance | `local` = trace `block_mask` 中 engine 已命中的 block；`remote` = KVCM/L3 模拟命中 |
| engine-local-only | `multi_infer_replay` | 只模拟一个或多个推理实例本地 L1/L2，trace 已确定请求归属 | 推理实例 | `local` = engine 本地 L1/L2 命中；`remote` 通常为 0 |
| engine-local + L3 pool | `hierarchical_replay_main` | 需要完整模拟推理实例本地 L1/L2 与 KVCM/L3 池化；trace 可指定推理实例，也可由 scheduler 分配 | `engine_instance` = 推理实例；`pool_instance` = KVCM/L3 instance | `local` = engine L1/L2 命中；`remote` = L3 pool 命中 |

`hierarchical_replay_main` 是否使用调度器由 `infer_scheduling_strategy` 决定：trace 已指定推理实例时使用 `preserve_trace`；需要模拟调度时使用 `round_robin` 或 `prefix_hit`。
hierarchical 模式使用 `infer_clusters + pool_config` 描述拓扑：同一个 L3 pool 下的同构推理实例只写一份 model、tier 列表和层间 flow，再列出推理实例 ids；L3 pool 直接复用普通 optimizer config，可表达单层或多层 L3。多层只由 `storages` 数量决定：单层不写 `tier_flows`，多层必须写满所有相邻 edge。开启 `enable_lifecycle_tracking=true` 后，推理侧输出到 `output_result_path/infer`，pool 输出到 `pool_config.output_result_path`。

## 指标口径

标准版只对外暴露一种命中率口径：

```text
HitRate = HitTokens / InputTokens
AccHitRate = AccHitTokens / AccInputTokens
```

相关约定：

- `HitRate`、`AccHitRate`、`LocalHitRate`、`RemoteHitRate` 和 `Tier*_HitRate` 都是 token hit rate。
- 标准命中率只按 token 计算，但 CSV 保留 block 读/命中计数，便于核对读放大、命中规模和容量行为。
- `HitRate` 是最终整体结论；`LocalHit*` / `RemoteHit*` 是组成拆分。`local` 永远表示 engine 本地命中，`remote` 永远表示 KVCM/L3 命中；在 KVCM/L3-only 模式下，engine 本地命中来自 trace `block_mask`。
- 标准 `get` trace 必须包含 `input_len`，`InputTokens` 直接使用 `input_len`。
- 不再兼容缺失 `input_len` 的旧 `get` trace。其他来源的日志需要先转换成 optimizer schema。
- `keys` 只包含完整 block key；不足一个 block 的尾部 token 不写入 `keys`，但仍计入 `InputTokens`。例如 `block_size=16`、`input_len=33` 时，`keys` 最多包含 2 个完整 block，尾部 1 个 token 只进入分母。

标准 `*_hit_rates.csv` 的核心列：

| 列 | 说明 |
|---|---|
| `TimestampNs` | trace 时间戳，单位 ns |
| `CachedBlocks` | 当前 CSV 对应 instance 的缓存 block 数 |
| `CachedBlocksAllInstances` | 同一 optimizer 进程内所有 instance 的总缓存 block 数 |
| `ReadBlocks` / `HitBlocks` | 当前请求读取 / 命中的 block 数 |
| `LocalHitBlocks` / `RemoteHitBlocks` | engine 本地命中 / KVCM 或 L3 pool 命中 block 数 |
| `InputTokens` / `HitTokens` | 当前请求的输入 token 数 / 命中 token 数 |
| `LocalHitTokens` / `RemoteHitTokens` | engine 本地命中 / KVCM 或 L3 pool 命中 token 数 |
| `HitRate` | 当前请求整体 token hit rate |
| `LocalHitRate` / `RemoteHitRate` | 当前请求 local / remote token hit rate |
| `AccReadBlocks` / `AccHitBlocks` | 累计读取 / 命中的 block 数 |
| `AccInputTokens` / `AccHitTokens` | 累计输入 token 数 / 累计命中 token 数 |
| `AccLocalHitTokens` / `AccRemoteHitTokens` | 累计 local / remote 命中 token 数 |
| `AccHitRate` | 累计整体 token hit rate |
| `AccLocalHitRate` / `AccRemoteHitRate` | 累计 local / remote token hit rate |
| `AccWriteBlocks` | 截至当前时间的累计写入 block 数 |
| `Tier<N>(name)_HitTokens` | 当前请求在某个 tier 的命中 token 数 |
| `Tier<N>(name)_HitRate` / `AccTier<N>(name)_HitRate` | 当前 / 累计 tier token hit rate |
| `Tier<N>(name)_BlockNum` | 当前 tier 的缓存 block 数 |

## 顶层配置

```json
{
  "trace_file_path": "/path/to/optimizer_trace.jsonl",
  "output_result_path": "/path/to/output",
  "eviction_params": {
    "eviction_mode": 1,
    "eviction_batch_size_per_instance": 100
  },
  "trace_replay": {
    "write_delay_ns": 1
  },
  "instance_groups": []
}
```

`output_result_path` 是所有 config 驱动入口的结果输出目录，包括 `optimizer_main`、`optimizer_run`、`tradeoff` 和 `export_tree`。`export_tree` 会写到该目录下的 `radix_tree/`。`multi_infer_replay` 不读取完整 optimizer config，需要通过 `--output-dir` 显式指定输出目录。

### eviction_params

| 字段 | 类型 | 默认 | 说明 |
|---|---:|---:|---|
| `eviction_mode` | int | 必填 | `1`=`GROUP_ROUGH`，`2`=`INSTANCE_ROUGH`，`3`=`INSTANCE_PRECISE` |
| `eviction_batch_size_per_instance` | int | 必填 | 每轮每实例最多驱逐的 block 数。rough 模式必须大于 0 |

标准示例使用 `eviction_mode=1`、`eviction_batch_size_per_instance=100`。

### trace_replay

| 字段 | 类型 | 默认 | 说明 |
|---|---:|---:|---|
| `write_delay_ns` | int64 | `1` | `type=request` trace 的内部写入延迟。回放时先在 `timestamp_ns` 执行读，再在 `timestamp_ns + write_delay_ns` 调度写入。必须大于 0 |

## 标准 trace schema

optimizer 回放输入只接受 JSONL，每行一条标准 trace。字段不完整时直接失败，不做旧格式推断。

`timestamp_ns`、`get.input_len`、`block_mask` offset 和 `ttl_us` 必须落在 `int64_t` 范围内：

```text
[-9223372036854775808, 9223372036854775807]
```

`keys` 支持 JSON signed/unsigned 整数。超过 `INT64_MAX=9223372036854775807` 的 unsigned 64-bit 值会按补码稳定映射到内部 `int64_t`，例如 `9223372036854775808 -> -9223372036854775808`、`18446744073709551615 -> -1`。

Get trace：

```json
{
  "type": "get",
  "instance_id": "instance-a",
  "trace_id": "trace_instance-a_1000",
  "timestamp_ns": 1000,
  "keys": [101, 102, 103],
  "input_len": 33,
  "query_type": "prefix_match",
  "block_mask": [],
  "sw_size": 0,
  "location_spec_names": []
}
```

Request trace：

当外部 trace 只有请求级记录、没有显式拆成读写两行时，使用 `request`。它等价于：

- 在 `timestamp_ns` 执行一次 `get`
- 在 `timestamp_ns + trace_replay.write_delay_ns` 执行一次 `write`

```json
{
  "type": "request",
  "instance_id": "instance-a",
  "trace_id": "trace_instance-a_1000",
  "timestamp_ns": 1000,
  "keys": [101, 102, 103],
  "input_len": 33,
  "query_type": "prefix_match",
  "block_mask": [],
  "sw_size": 0,
  "location_spec_names": [],
  "ttl_us": 0
}
```

Write trace：

```json
{
  "type": "write",
  "instance_id": "instance-a",
  "trace_id": "trace_instance-a_1001",
  "timestamp_ns": 1001,
  "keys": [101, 102, 103],
  "ttl_us": 0
}
```

必填字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | string | 只能是 `get`、`write` 或 `request` |
| `instance_id` | string | 非空，必须匹配配置中的 instance |
| `timestamp_ns` | int64 | ns 时间戳，必须为正整数；不再接受 `timestamp_us` |
| `keys` | int64/uint64 array | block key 列表，可为空 |

可选公共字段：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `trace_id` | string | 空字符串 | 请求标识，用于调试和模板分析 |

`get` 专用字段：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `input_len` | int64 | 必填 | 输入 token 数，必须为正整数；`InputTokens` 直接使用该值 |
| `query_type` | string | `prefix_match` | 当前只支持 `prefix_match`；其他值会被跳过 |
| `block_mask` | bool array 或非负 int64 | 空数组 | trace 已经知道的本地命中 block。数组形式逐 block 标记；整数形式表示从前缀开始的本地命中 block 数 |
| `sw_size` | int32 | 0 | 滑窗参数，当前标准前缀匹配通常为 0 |
| `location_spec_names` | string array | 空数组 | 兼容字段，标准分析通常为空 |

`write` 专用字段：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `ttl_us` | int64 | 0 | 请求级 TTL，单位微秒；`0` 使用 group 默认 TTL，`-1` 表示禁用 TTL |

`request` 字段：

`request` 复用 `get` 的所有字段，并额外支持 `write.ttl_us`。`request` 内部生成的写入会使用同一组 `keys` 和该 `ttl_us`。

`write` 只读取 `type`、`instance_id`、`trace_id`、`timestamp_ns`、`keys` 和 `ttl_us`；其他字段包括 `input_len` 和 `block_mask` 都忽略。`block_mask` 只用于 `get/request`，表示请求进入 KVCM/L3 前 engine 本地已命中的 block。直接分析请求输入时通常可传空数组，此时整体 `HitRate` 仍按 `HitTokens / InputTokens` 计算。

旧格式或不合法输入会失败，包括：

- 缺少 `type`、`instance_id`、`timestamp_ns`、`keys`，或 `get` / `request` trace 缺少 `input_len`。
- `get/request.input_len <= 0`、`timestamp_ns <= 0`，或 `keys` 不是数组。
- `get/request.keys.size() > input_len / block_size`。标准 trace 的 `keys` 只能包含完整 block，不允许把不足一个 block 的尾部 key 写入 trace。
- 使用 `timestamp_us` 但没有 `timestamp_ns`。
- `keys` 中存在非整数。
- `block_mask` 数组中存在非 bool 值，或 offset 为负数 / 超过 `INT64_MAX`。
- legacy dialog-style trace 只有 `query_type` / `block_mask` / decode metadata 但没有显式 `type=get/write`。

## instance group 配置

```json
{
  "group_name": "instance-a",
  "quota_capacity": 178,
  "used_percentage": 1.0,
  "tier_flows": [
    {
      "from_tier": "hbm",
      "to_tier": "dram",
      "write_mode": "write_through",
      "access_propagation_enabled": false,
      "write_propagation_enabled": false,
      "promote_enabled": true,
      "selective_write_threshold": 2
    }
  ],
  "ttl_config": {
    "default_block_ttl_seconds": 0,
    "refresh_on_read": true
  },
  "storages": [
    {"unique_name": "hbm", "storage_type": "pace", "band_width_mbps": 20000, "priority": 0, "capacity": 50},
    {"unique_name": "dram", "storage_type": "hf3fs", "band_width_mbps": 20000, "priority": 1, "capacity": 128}
  ],
  "instances": []
}
```

| 字段 | 类型 | 默认 | 标准语义 |
|---|---:|---:|---|
| `group_name` | string | 必填 | 实例组名称。multi-infer replay 中通常等于推理 instance 的 `instance_id` |
| `quota_capacity` | number | 必填 | group 总容量，单位 GB。单层模式按该字段驱逐；多层模式下主要用于记录总容量；`-1` 表示无限容量 |
| `used_percentage` | number | 必填 | 容量水位比例，实际阈值为 capacity × used_percentage |
| `tier_flows` | array | 单层为空 | 相邻 tier edge 的完整流动策略。`storages` 超过一层时必须正好配置 `storages.size()-1` 条 |
| `ttl_config` | object | 必填 | TTL 配置，见下表 |
| `storages` | array | 必填 | tier 列表，按 `priority` 从小到大排序 |
| `instances` | array | 必填 | 该 group 下的 optimizer instance 列表 |

多层规则固定如下：

- `storages.size()==1`：单层模式，不允许配置 `tier_flows`。
- `storages.size()>1`：多层模式，必须为每条相邻 edge 配置一条 `tier_flows`，不允许缺省、不允许跨层、不允许重复。
- 旧字段 `tier_strategy`、`hierarchical_eviction_enabled`、group 级 `write_mode/access_propagation_enabled/write_propagation_enabled/promote_enabled/selective_write_threshold` 都不再支持。

### ttl_config

| 字段 | 类型 | 标准语义 |
|---|---:|---|
| `default_block_ttl_seconds` | int | group 默认 TTL 秒数，`0` 表示组级禁用 TTL |
| `refresh_on_read` | bool | TTL 策略下读命中是否刷新 TTL 锚点 |

### write_mode

| 值 | 写入行为 | 驱逐行为 | 适用场景 |
|---|---|---|---|
| `write_through` | 写入时同时落所有 tier | 各 tier 独立按自身容量驱逐 | 基线、全量副本、多层独立命中率分析 |
| `cascading` | 写入时只落 tier 0 | tier i 驱逐出的 block 降级到 tier i+1，最后一层驱逐后丢弃 | HBM→DRAM→SSD 逐级下沉 |
| `write_through_selective` | 初始只落 `from_tier` | 命中层访问次数达到该 edge 的 `selective_write_threshold` 后复制到 `to_tier` | 控制低层写放大，只让热块下沉 |

`write_mode` 只接受上表三个值，其他值会导致 config 解析失败。

### tier_flows

`tier_flows` 是唯一的多层读写流动配置。每个 flow 必须引用 `storages` 中相邻的两个 `unique_name`，不支持跨层跳配，也不允许重复配置同一条 edge。
配置加载时会按 `storages[*].priority` 排序后校验 edge。未知 tier、非相邻 edge、重复 edge、重复 `unique_name` 或重复 `priority` 都会直接报错并拒绝加载。

```json
{
  "tier_flows": [
    {
      "from_tier": "hbm",
      "to_tier": "dram",
      "write_mode": "write_through",
      "access_propagation_enabled": false,
      "write_propagation_enabled": false,
      "promote_enabled": true,
      "selective_write_threshold": 2
    },
    {
      "from_tier": "dram",
      "to_tier": "ssd",
      "write_mode": "cascading",
      "access_propagation_enabled": false,
      "write_propagation_enabled": false,
      "promote_enabled": false,
      "selective_write_threshold": 2
    }
  ]
}
```

单条 flow 的字段：

| 字段 | 类型 | 默认 | 标准语义 |
|---|---:|---:|---|
| `from_tier` | string | 必填 | edge 的高优先级 tier 名称，必须等于某个 `storages[i].unique_name` |
| `to_tier` | string | 必填 | edge 的低优先级 tier 名称，必须等于相邻的 `storages[i+1].unique_name` |
| `write_mode` | string | 必填 | 这条 edge 的写入/驱逐下沉策略 |
| `access_propagation_enabled` | bool | 必填 | 访问命中高层后，是否跨这条 edge 刷新下层访问时间 |
| `write_propagation_enabled` | bool | 必填 | 写 touch 命中高层已有副本后，是否跨这条 edge 刷新下层已有副本访问时间 |
| `promote_enabled` | bool | 必填 | 低层命中后，是否允许跨这条 edge 回填到高层 |
| `selective_write_threshold` | int | 必填 | 这条 edge 使用 `write_through_selective` 时的下写阈值，必须为正整数 |

### access_propagation_enabled

这个开关与 `write_mode` 正交：

- `true`：一个 block 在多个 tier 有副本时，读命中最高优先级 tier 后，也刷新后续副本 tier 的访问时间。这是默认行为。
- `false`：只刷新实际命中的最高优先级 tier，不刷新下层副本的访问时间。适用于评估 write-through 或 cascading/promote 后多副本场景下的下层独立冷热衰减。

`promote_enabled=true` 时，低优先级 tier 命中会触发向更高优先级 tier 逐层复制。比如 L3 命中会补齐 L2 和 L1，L2 命中只补 L1，不会额外写入更低层。复制动作会走容量检查，可能立刻触发对应 tier 的驱逐。

## storage 配置

```json
{
  "unique_name": "hbm",
  "storage_type": "pace",
  "band_width_mbps": 20000,
  "priority": 0,
  "capacity": 50
}
```

| 字段 | 类型 | 说明 |
|---|---:|---|
| `unique_name` | string | tier 名称，会进入 CSV 的 `Tier<N>(name)_*` 列 |
| `storage_type` | string | 存储类型标签，当前主要用于配置记录 |
| `band_width_mbps` | number | 带宽标签，当前主要用于分析记录 |
| `priority` | int | tier 优先级，越小越靠近计算侧 |
| `capacity` | number | tier 容量，单位 GB；`-1` 表示该 tier 无限容量，不触发该 tier 的容量驱逐 |

## instance 配置

```json
{
  "instance_id": "instance-a",
  "block_size": 16,
  "bytes_per_token": 512,
  "eviction_policy_type": "lru",
  "eviction_policy_params": {}
}
```

| 字段 | 类型 | 说明 |
|---|---:|---|
| `instance_id` | string | trace 中的实例 ID，必须与 trace 行内 `instance_id` 匹配 |
| `block_size` | int | 每个 block 的 token 数。token hit rate 会用它把命中 block 转为命中 token |
| `bytes_per_token` | int | 单 token KV 大小。`bytes_per_block = block_size * bytes_per_token` |
| `eviction_policy_type` | string | `lru`、`random_lru`、`leaf_aware_lru`、`ttl` |
| `eviction_policy_params` | object | 策略参数，见下文 |

## eviction_policy_params

### lru / leaf_aware_lru

```json
{
  "sample_rate": 1.0,
  "shard_count": 1,
  "sample_times": 32,
  "eviction_amplification_factor": 1.0
}
```

| 字段 | 说明 |
|---|---|
| `sample_rate` | 采样比例。`1.0` 表示完整 LRU |
| `shard_count` | LRU 分片数 |
| `sample_times` | 每次采样次数 |
| `eviction_amplification_factor` | 驱逐放大系数 |

### random_lru

```json
{
  "sample_rate": 1.0
}
```

`random_lru` 只要求 `sample_rate`，用于控制采样范围。

### ttl

```json
{
  "fallback_on_pressure": true
}
```

| 字段 | 说明 |
|---|---|
| `fallback_on_pressure=true` | 先清理 TTL 过期 block；容量仍超限时按 LRU 兜底 |
| `fallback_on_pressure=false` | 纯 TTL，只清理过期 block；容量压力不会触发 LRU 兜底 |

TTL 只在 `eviction_policy_type="ttl"` 时执行。非 TTL 策略会忽略 `ttl_config` 的过期清理语义。

## 标准多推理实例回放

标准版保留 `multi_infer_replay` 作为 engine-local-only 入口，不再把 multi-machine scheduler 作为默认回放入口；需要模拟调度或 L3 pool 时使用 `hierarchical_replay_main`。
脚本完整参数以 [analysis/script/README.md](../analysis/script/README.md) 为准；这里给出标准回放配置示例和输出约定。
`multi_instance_replay` 不读取完整 optimizer config；它根据 CLI 参数为每个 pod/instance 生成单实例 config，然后并行运行 optimizer。当前 CLI 直接支持 L1/L2 两层容量；需要 L3 或更复杂 tier 拓扑时，需要使用完整 optimizer config 跑单次回放，或扩展该脚本的 config 生成逻辑。

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:multi_infer_replay -- \
  --trace-dir /path/to/instance_traces \
  --trace-glob "*.jsonl" \
  --output-dir /path/to/output \
  --bucket-name bucket-a \
  --l1-capacity 50 \
  --l2-capacity 128 \
  --block-size 16 \
  --bytes-per-token 512 \
  --eviction-policy lru \
  --default-tier-write-mode write_through \
  --max-workers 32
```

输出：

- 输出根目录为 `--output-dir`。
- `configs/<instance_id>.json`：每个 instance 的生成配置。
- `<instance_id>_hit_rates.csv`：每个 instance 的标准 token hit-rate 时序。
- `aggregate/instance_aggregate.csv`：每个 instance 的聚合结果。
- `aggregate/global_aggregate.csv`：所有 instance 汇总后的全局结果。
- `aggregate/global_window_hit_rates.csv`：指定 `--window-ns` 或 `--window-seconds` 时生成的窗口结果。

multi-infer replay 聚合后的 `HitRate` 仍然是 token hit rate，计算方式为所有推理 instance 的 `HitTokens` 总和除以 `InputTokens` 总和。
`--bucket-name` 只写入聚合 CSV 的 `Bucket` 列，用于标记实验来源；`--trace-glob` 和 `--recursive` 只在使用 `--trace-dir` 扫描输入文件时生效。
`--default-tier-write-mode`、`--tier-flow-config`、`--enable/disable-tier-access-propagation`、`--enable/disable-promote` 和 `--selective-write-threshold` 会写入生成 config 的 `tier_strategy`，语义与上文一致。
