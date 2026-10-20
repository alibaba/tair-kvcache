# Prometheus Metrics 端点

KVCacheManager 在 Admin HTTP 服务端口（默认 6492）上提供了兼容 Prometheus
的指标抓取端点 `GET /metrics`。

## 快速开始

默认配置下该端点自动启用。将 Prometheus 实例指向 Admin HTTP 端口即可：

```yaml
# prometheus.yml
scrape_configs:
  - job_name: kvcache_manager
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ["<host>:6492"]
```

然后使用 PromQL 查询指标：

```promql
# 服务 QPS（1 分钟速率）
rate(kvcm_service_query_counter[1m])

# 搜索缓存命中率
kvcm_meta_indexer_search_cache_hit_ratio

# GetCacheLocation Block 级命中率（5 分钟窗口）
rate(kvcm_manager_get_cache_location_hit_block_counter[5m])
  / rate(kvcm_manager_get_cache_location_query_block_counter[5m])

# 每个后端的存储使用率
kvcm_data_storage_storage_usage_ratio
```

## 配置

以下两个配置项控制 Prometheus 端点：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `kvcm.metrics.enable_prometheus` | `true` | 启用/禁用 `GET /metrics` 端点 |
| `kvcm.metrics.prometheus_prefix` | `kvcm` | 所有指标名称的前缀 |

可通过配置文件、`--env` 启动参数或环境变量进行设置：

```bash
# 禁用端点
kv_cache_manager_bin -e 'kvcm.metrics.enable_prometheus=false'

# 修改指标名称前缀
kv_cache_manager_bin -e 'kvcm.metrics.prometheus_prefix=myapp'
```

### 公共环境变量 tag

公共配置 `kvcm.metrics.env_tag_mappings` 支持将环境变量解析为进程级 tag：

```bash
export APP_WORKSPACE=workspace_a
export APP_DEPLOYMENT=deployment_a
export kvcm_metrics_env_tag_mappings='workspace=APP_WORKSPACE,deployment_name=APP_DEPLOYMENT'
```

配置格式是 `tag名=源环境变量名`，多个映射以逗号分隔。启动时读取源变量，
变量不存在或值为空时跳过；非空值（包括首尾空白）保持原样。修改后需重启。
空配置不添加 tag，重复 tag 名、
空名或错误的映射语法会使配置解析失败。配置遵循 KVCM 的配置文件、`-e`
参数及系统环境变量入口，独立于 reporter 类型。

Prometheus 在导出阶段合并这些 tag，包括原本没有 tag 的指标：

```text
kvcm_service_query_counter{deployment_name="deployment_a",workspace="workspace_a"} 12345
```

label 名沿用下述字符转换规则；转换后与指标自身 label 重名时，指标自身的值优先。
公共 tag 应使用 workspace、deployment_name 等部署维度，转换后的名称不要与已有业务
label 重名。即使单条输出保留了业务 label，也可能让原本不同的序列变得相同；
服务不检查跨序列冲突。

多个公共 tag 转换为同一个 label 名时，启动解析会输出 ERROR，
打印重名的 label、被覆盖的值和最终生效的值，最终只保留一个值，服务继续启动。
例如 `host.name=ENV_A,host-name=ENV_B` 都转换为 `host_name`；配置者应避免这种重名，
不依赖冲突时的覆盖顺序。名称转换和公共 tag 冲突处理只在启动时执行。
公共 tag 中的保留 label 名 `__*`、`le` 和 `quantile` 被忽略，以保护指标名和
histogram/summary 的维度。启动时对每个被忽略的 tag 输出一行 WARN，说明原始 tag 名和
转换后的 label 名；指标自身的 `le` 等 label 正常保留。值按 Prometheus
文本格式转义。公共 tag 不写入 MetricsRegistry，因此 `/api/getMetrics` 返回原始 tag。

其他 metrics 后端可通过公共初始化参数获取相同的 tag key/value，并按各自规则处理；
当前已适配的输出后端是 Prometheus。

## 输出格式

端点输出 Prometheus 文本展示格式
（`text/plain; version=0.0.4; charset=utf-8`）。

内部指标名称（点分隔，如 `service.query_counter`）会被转换为
Prometheus 兼容的名称——将点替换为下划线，并添加配置的前缀：

```
service.query_counter  ->  kvcm_service_query_counter
meta_indexer.total_key_count  ->  kvcm_meta_indexer_total_key_count
```

`CounterValue` 类型的指标以 `# TYPE ... counter` 输出。
`GaugeValue` 类型的指标以 `# TYPE ... gauge` 输出。

`MetricsTags`（键值对）以 Prometheus 标签形式输出：

```
kvcm_data_storage_storage_usage_ratio{type="hf3fs",unique_name="nfs_01"} 0.75
kvcm_cache_manager_group_usage_ratio{instance_group="default"} 0.42
```

### Label 规范

为了让 `data_storage.*` 系列指标能在 PromQL 中互相 `join` /
聚合，所有按存储实例维度的 `data_storage.*` 序列统一使用两个
label：

- `type`：后端类型，例如 `hf3fs`、`nfs`、`pace`、`tair_mempool`。
- `unique_name`：后端实例的 `global_unique_name`。

```
kvcm_data_storage_create_counter{type="nfs",unique_name="nfs_01"} 100
kvcm_data_storage_create_keys_counter{type="nfs",unique_name="nfs_01"} 12800
kvcm_data_storage_healthy_status{type="nfs",unique_name="nfs_01"} 1
kvcm_data_storage_storage_usage_ratio{type="nfs",unique_name="nfs_01"} 0.6
```

## 输出示例

```
# HELP kvcm_service_query_counter service.query_counter
# TYPE kvcm_service_query_counter counter
kvcm_service_query_counter 12345
# HELP kvcm_service_query_rt_us service.query_rt_us
# TYPE kvcm_service_query_rt_us gauge
kvcm_service_query_rt_us 523.5
# HELP kvcm_meta_indexer_total_key_count meta_indexer.total_key_count
# TYPE kvcm_meta_indexer_total_key_count gauge
kvcm_meta_indexer_total_key_count 42000
# HELP kvcm_data_storage_storage_usage_ratio data_storage.storage_usage_ratio
# TYPE kvcm_data_storage_storage_usage_ratio gauge
kvcm_data_storage_storage_usage_ratio{type="hf3fs",unique_name="store_01"} 0.6
kvcm_data_storage_storage_usage_ratio{type="nfs",unique_name="store_02"} 0.3
```

## 可用指标

该端点导出 `MetricsRegistry` 中注册的所有指标，包括逐请求指标（累积
计数器和最新值仪表）以及周期性指标（每 `kvcm.metrics.report_interval_ms`
刷新一次，默认 20 秒）。

### 逐请求指标

| 指标 | 类型 | 说明 |
|---|---|---|
| `service.query_counter` | counter | 总查询次数 |
| `service.error_counter` | counter | 总错误次数 |
| `service.query_rt_us` | gauge | 最近一次查询响应时间（微秒） |
| `service.request_queue_size` | gauge | 请求队列大小 |
| `manager.request_key_count` | gauge | 每次请求的 key 数量 |
| `manager.prefix_match_len` | gauge | 前缀匹配长度 |
| `manager.get_cache_location_query_block_counter` | counter | GetCacheLocation 查询的 Block 总数（累计） |
| `manager.get_cache_location_hit_block_counter` | counter | GetCacheLocation 命中的 Block 总数（累计） |
| `manager.prefix_match_time_us` | gauge | GetHostCacheState 等前缀匹配的外层总延迟（微秒） |
| `meta_searcher.indexer_get_time_us` | gauge | MetaSearcher 调用 MetaIndexer 读取 metadata 的墙钟时间（微秒） |
| `meta_indexer.get_io_time_us` | gauge | metadata backend 调用区间的墙钟并集；local 模式包含 LRU/锁/复制，不包含并行 projection，也不表示 Redis I/O |
| `meta_searcher.host_projection_time_us` | gauge | GetHostCacheState location 可见性及 host/spec projection 回调区间的墙钟并集（微秒） |
| `meta_searcher.host_prefix_reduce_time_us` | gauge | GetHostCacheState 普通/Mamba host 前缀归约时间（微秒） |
| `meta_indexer.search_cache_hit_ratio` | gauge | 搜索缓存命中率 |
| `data_storage.create_keys_counter` | counter | 已创建 key 总数 |

### 周期性指标

| 指标 | 类型 | 说明 |
|---|---|---|
| `meta_indexer.total_key_count` | gauge | 所有索引器的 key 总数 |
| `meta_indexer.total_cache_usage` | gauge | 缓存使用总量（字节） |
| `data_storage.healthy_status` | gauge | 存储后端读取可用状态（1/0） |
| `data_storage.writable_status` | gauge | 存储后端写入可用状态（1/0）；只读降级时为 0 |
| `data_storage.storage_usage_ratio` | gauge | 存储使用率 |
| `cache_manager.write_location_expire_size` | gauge | 已过期的写入位置数量 |
| `cache_manager_group.usage_ratio` | gauge | 实例组 KVCM 配额容量使用率（不含 EventReport） |
| `cache_manager_instance.key_count` | gauge | 单实例 key 数量 |
| `cache_manager_instance.byte_size` | gauge | 单实例 KVCM 配额字节大小（不含 EventReport） |

对于 TairMempool，`data_storage.healthy_status` 为 1 表示后端已打开且健康探测状态正常，否则为 0。状态由后台探测更新，单次请求超时不代表指标立即变为 0。超时参数见[配置说明](configuration.md#tairmempool)。

完整指标列表取决于当前使用的 `MetricsReporter` 类型。`kmonitor`
类型的 reporter 会填充最完整的指标集。

上述 GetHostCacheState 分段指标是嵌套关系：`meta_indexer.get_io_time_us` 位于
`meta_searcher.indexer_get_time_us` 内，后者与 projection/reduce 又位于
`manager.prefix_match_time_us` 内。渐进 local 查询会流水化 backend read 与 projection，两者可能重叠，排障时
不能把它们相加。`get_io_time_us` 是历史命名；当实例使用 `storage_type=local` 时没有 Redis 网络调用。

## 与 KMonitor 指标对照

KVCacheManager 同时通过 KMonitor 与 Prometheus `/metrics` 端点导出
指标。两条管线写入不同的存储，因此部分 KMonitor 指标名在
Prometheus 侧并不以**同名**输出 —— 最常见的是 `*.qps` 一族，由
KMonitor agent 在上报时计算。Prometheus 侧请用 PromQL
`rate(<counter>[Xm])` 等价获取。

### QPS 类指标

| KMonitor 指标 | Prometheus 等价 |
|---|---|
| `service.qps` | `rate(kvcm_service_query_counter[1m])` |
| `service.error_qps` | `rate(kvcm_service_error_counter[1m])` |
| `event_report.qps` | `rate(kvcm_event_report_request_counter[1m])` |
| `event_report.error_qps` | `rate(kvcm_event_report_error_counter[1m])` |
| `data_storage.create_qps` | `rate(kvcm_data_storage_create_counter[1m])` |
| `data_storage.create_keys_qps` | `rate(kvcm_data_storage_create_keys_counter[1m])` |

Prometheus 侧存储的是底层 *counter*（单调递增），KMonitor 的
`*.qps` 值由 agent 在上报时计算。两种视图反映的是同一事件流，
查询 Prometheus 时直接用 counter + `rate` 即可。

`event_report.*` 按固定标签 `instance_group`、`instance_id`、`type` 和
`event_type` 拆分。其中 `type` 为 `event_report_l1p5` 或
`event_report_l2`，`event_type` 为协议定义的六种事件类型之一（未知值
统一记为 `unknown`）。一个批量请求包含某事件类型时，该类型 QPS 计一次；
同一请求内同类型的多个 event 不重复计数。`event_report.request_rt_us`
记录整次 `ReportEvent` 请求的端到端耗时；请求失败时
`event_report.error_qps` 按该请求包含的事件类型分别计一次。

注：`data_storage.create_keys_qps` 在 Prom 侧也会以 gauge 形式
导出"最近一次批次大小"（不是每秒速率）。每秒速率请使用
`rate(kvcm_data_storage_create_keys_counter[1m])`，gauge 仅作
"最近一次批次大小"诊断用。

## 架构

Prometheus 端点通过一个轻量级序列化器（`PrometheusExporter`）实现，
在抓取时读取已有的 `MetricsRegistry`。不依赖任何外部 Prometheus
客户端库。

```
Prometheus  --GET /metrics-->  AdminServiceHttp
                                  |
                                  v
                           PrometheusExporter::Expose()
                                  |
                                  v
                           MetricsRegistry::GetAllMetrics()
                                  |
                                  v
                           text/plain 响应
```

该端点与 `MetricsReporter` 管线正交。任何 reporter 类型（`kmonitor`、
`local`、`logging`、`dummy`）都向同一个 `MetricsRegistry` 写入数据，
Prometheus 端点从中读取。
