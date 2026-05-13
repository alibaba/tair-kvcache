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

# 缓存命中率
kvcm_meta_indexer_search_cache_hit_ratio

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
| `manager.prefix_match_time_us` | gauge | 前缀匹配延迟（微秒） |
| `meta_indexer.search_cache_hit_ratio` | gauge | 搜索缓存命中率 |
| `data_storage.create_keys_counter` | counter | 已创建 key 总数 |

### 周期性指标

| 指标 | 类型 | 说明 |
|---|---|---|
| `meta_indexer.total_key_count` | gauge | 所有索引器的 key 总数 |
| `meta_indexer.total_cache_usage` | gauge | 缓存使用总量（字节） |
| `data_storage.healthy_status` | gauge | 存储后端健康状态（1/0） |
| `data_storage.storage_usage_ratio` | gauge | 存储使用率 |
| `cache_manager.write_location_expire_size` | gauge | 已过期的写入位置数量 |
| `cache_manager_group.usage_ratio` | gauge | 实例组容量使用率 |
| `cache_manager_instance.key_count` | gauge | 单实例 key 数量 |
| `cache_manager_instance.byte_size` | gauge | 单实例字节大小 |

完整指标列表取决于当前使用的 `MetricsReporter` 类型。`kmonitor`
类型的 reporter 会填充最完整的指标集。

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
