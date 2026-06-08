# Prometheus Metrics Endpoint

KVCacheManager exposes a Prometheus-compatible metrics scrape endpoint
at `GET /metrics` on the Admin HTTP service port (default 6492).

## Quick Start

With default settings, the endpoint is enabled automatically. Point
your Prometheus instance at the Admin HTTP port:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: kvcache_manager
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ["<host>:6492"]
```

Then query metrics with PromQL:

```promql
# Service QPS (rate over 1 minute)
rate(kvcm_service_query_counter[1m])

# Search cache hit ratio
kvcm_meta_indexer_search_cache_hit_ratio

# GetCacheLocation block-level hit rate (5-minute window)
rate(kvcm_manager_get_cache_location_hit_block_counter[5m])
  / rate(kvcm_manager_get_cache_location_query_block_counter[5m])

# Storage usage per backend
kvcm_data_storage_storage_usage_ratio
```

## Configuration

Two config keys control the Prometheus endpoint:

| Key | Default | Description |
|---|---|---|
| `kvcm.metrics.enable_prometheus` | `true` | Enable/disable the `GET /metrics` endpoint |
| `kvcm.metrics.prometheus_prefix` | `kvcm` | Prefix prepended to all metric names |

Set via config file, `--env` flag, or environment variable:

```bash
# Disable the endpoint
kv_cache_manager_bin -e 'kvcm.metrics.enable_prometheus=false'

# Change the metric name prefix
kv_cache_manager_bin -e 'kvcm.metrics.prometheus_prefix=myapp'
```

## Output Format

The endpoint outputs the Prometheus text exposition format
(`text/plain; version=0.0.4; charset=utf-8`).

Internal metric names (dotted, e.g. `service.query_counter`) are
translated to Prometheus-compatible names by replacing dots with
underscores and prepending the configured prefix:

```
service.query_counter  ->  kvcm_service_query_counter
meta_indexer.total_key_count  ->  kvcm_meta_indexer_total_key_count
```

`CounterValue` metrics are emitted as `# TYPE ... counter`.
`GaugeValue` metrics are emitted as `# TYPE ... gauge`.

`MetricsTags` (key-value pairs) are emitted as Prometheus labels:

```
kvcm_data_storage_storage_usage_ratio{type="hf3fs",unique_name="nfs_01"} 0.75
kvcm_cache_manager_group_usage_ratio{instance_group="default"} 0.42
```

### Label Conventions

To allow PromQL `join` / aggregation across the `data_storage.*`
metric family, every per-instance `data_storage.*` series uses the
same two labels:

- `type`: backend type, e.g. `hf3fs`, `nfs`, `pace`, `tair_mempool`.
- `unique_name`: the backend instance's `global_unique_name`.

```
kvcm_data_storage_create_counter{type="nfs",unique_name="nfs_01"} 100
kvcm_data_storage_create_keys_counter{type="nfs",unique_name="nfs_01"} 12800
kvcm_data_storage_healthy_status{type="nfs",unique_name="nfs_01"} 1
kvcm_data_storage_storage_usage_ratio{type="nfs",unique_name="nfs_01"} 0.6
```

## Example Output

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

## Available Metrics

The endpoint exposes all metrics registered in the shared
`MetricsRegistry`. These include both per-query metrics (accumulated
counters and latest-value gauges) and interval metrics (refreshed
every `kvcm.metrics.report_interval_ms`, default 20s).

### Per-Query Metrics

| Metric | Type | Description |
|---|---|---|
| `service.query_counter` | counter | Total query count |
| `service.error_counter` | counter | Total error count |
| `service.query_rt_us` | gauge | Last query response time (us) |
| `service.request_queue_size` | gauge | Request queue size |
| `manager.request_key_count` | gauge | Keys per request |
| `manager.prefix_match_len` | gauge | Prefix match length |
| `manager.get_cache_location_query_block_counter` | counter | Total blocks queried via GetCacheLocation (cumulative) |
| `manager.get_cache_location_hit_block_counter` | counter | Total blocks hit via GetCacheLocation (cumulative) |
| `manager.prefix_match_time_us` | gauge | Prefix match latency (us) |
| `meta_indexer.search_cache_hit_ratio` | gauge | Search cache hit ratio |
| `data_storage.create_keys_counter` | counter | Total created keys |

### Interval Metrics

| Metric | Type | Description |
|---|---|---|
| `meta_indexer.total_key_count` | gauge | Total keys across all indexers |
| `meta_indexer.total_cache_usage` | gauge | Total cache usage bytes |
| `data_storage.healthy_status` | gauge | Storage backend health (1/0) |
| `data_storage.storage_usage_ratio` | gauge | Storage usage ratio |
| `cache_manager.write_location_expire_size` | gauge | Expired write locations |
| `cache_manager_group.usage_ratio` | gauge | Group capacity usage ratio |
| `cache_manager_instance.key_count` | gauge | Per-instance key count |
| `cache_manager_instance.byte_size` | gauge | Per-instance byte size |

The full list depends on the active `MetricsReporter` type. The
`kmonitor` reporter populates the most complete set of metrics.

## Mapping to KMonitor Metrics

KVCacheManager simultaneously exports metrics via KMonitor and via
the Prometheus `/metrics` endpoint. The two pipelines write to
different metric stores, so a few KMonitor metric names are not
emitted by Prometheus *under the same name* — most commonly the
`*.qps` family, which is materialized server-side by KMonitor's QPS
reducer. Use `rate(<counter>[Xm])` in PromQL instead.

### QPS-style metrics

| KMonitor metric | Prometheus equivalent |
|---|---|
| `service.qps` | `rate(kvcm_service_query_counter[1m])` |
| `service.error_qps` | `rate(kvcm_service_error_counter[1m])` |
| `data_storage.create_qps` | `rate(kvcm_data_storage_create_counter[1m])` |
| `data_storage.create_keys_qps` | `rate(kvcm_data_storage_create_keys_counter[1m])` |

The Prometheus side stores the underlying *counter* (monotonically
increasing). KMonitor's `*.qps` value is computed by the agent at
report time. Both views describe the same event stream — pick `rate`
on the counter when querying Prometheus.

Note that `data_storage.create_keys_qps` is also exported as a Prom
gauge with the latest *batch size* (not a per-second rate). Prefer
`rate(kvcm_data_storage_create_keys_counter[1m])` for the per-second
view, and the gauge only for "the most recent batch size" diagnostics.

## Architecture

The Prometheus endpoint is implemented as a lightweight serializer
(`PrometheusExporter`) that reads the existing `MetricsRegistry` at
scrape time. It does not use any external Prometheus client library.

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
                           text/plain response
```

The endpoint is orthogonal to the `MetricsReporter` pipeline. Any
reporter type (`kmonitor`, `local`, `logging`, `dummy`) populates
the same `MetricsRegistry`, and the Prometheus endpoint reads from it.
