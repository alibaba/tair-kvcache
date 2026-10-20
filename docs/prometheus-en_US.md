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

### Common tags from environment variables

The backend-independent setting `kvcm.metrics.env_tag_mappings` resolves
environment variables into process-level tags:

```bash
export APP_WORKSPACE=workspace_a
export APP_DEPLOYMENT=deployment_a
export kvcm_metrics_env_tag_mappings='workspace=APP_WORKSPACE,deployment_name=APP_DEPLOYMENT'
```

Use comma-separated `tag_name=ENV_NAME` mappings. Values are read at startup;
missing or empty variables are skipped. Nonempty values, including leading
and trailing whitespace, are preserved. Restart the process after changing
the mapping or source values. An empty setting adds no tags. Duplicate tag
names, empty names, or malformed mappings fail configuration parsing. The
setting supports the usual config-file, `-e`, and environment-variable
inputs and works independently of the reporter type.

Prometheus merges these tags at export time, including metrics without
existing labels:

```text
kvcm_service_query_counter{deployment_name="deployment_a",workspace="workspace_a"} 12345
```

Label names use the character conversion rules below. Metric labels take
precedence over common tags after conversion. Use deployment dimensions
such as workspace and deployment_name for common tags, and avoid names that
overlap existing business labels after conversion. Preserving a business
label on an individual sample can still collapse distinct series; the
service does not check for cross-series collisions.

When multiple common tags convert to the same label name, startup parsing
prints an ERROR with the duplicate label name, the overwritten value, and
the effective value. Only one value is retained, and startup continues.
For example, `host.name=ENV_A,host-name=ENV_B` both convert to `host_name`.
Avoid such collisions and do not rely on overwrite order. Common-tag name
conversion and collision handling run once at startup.
Common tags named `__*`, `le`, or `quantile` after conversion
are omitted to protect metric names and histogram/summary dimensions.
Startup prints one WARN per omitted tag with its original and normalized name;
metric-specific labels such as `le` are preserved. Values are escaped for
the Prometheus text format. Common tags are not stored in MetricsRegistry,
so `/api/getMetrics` continues to return the original metric tags.

Other metrics backends receive the same resolved key/value pairs through
common initialization parameters and can apply their own rules. Currently,
Prometheus implements common-tag output.

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
| `manager.prefix_match_time_us` | gauge | Outer total latency for GetHostCacheState-style prefix matching (us) |
| `meta_searcher.indexer_get_time_us` | gauge | Wall time spent reading metadata through MetaIndexer (us) |
| `meta_indexer.get_io_time_us` | gauge | Wall-time union of metadata-backend call intervals; local mode includes LRU/locks/copies, excludes parallel projection, and does not imply Redis I/O |
| `meta_searcher.host_projection_time_us` | gauge | Wall-time union of GetHostCacheState visibility and host/spec projection callback intervals (us) |
| `meta_searcher.host_prefix_reduce_time_us` | gauge | GetHostCacheState normal/Mamba host-prefix reduction time (us) |
| `meta_indexer.search_cache_hit_ratio` | gauge | Search cache hit ratio |
| `data_storage.create_keys_counter` | counter | Total created keys |

### Interval Metrics

| Metric | Type | Description |
|---|---|---|
| `meta_indexer.total_key_count` | gauge | Total keys across all indexers |
| `meta_indexer.total_cache_usage` | gauge | Total cache usage bytes |
| `data_storage.healthy_status` | gauge | Storage backend read availability (1/0) |
| `data_storage.writable_status` | gauge | Storage backend write availability (1/0); 0 in read-only mode |
| `data_storage.storage_usage_ratio` | gauge | Storage usage ratio |
| `cache_manager.write_location_expire_size` | gauge | Expired write locations |
| `cache_manager_group.usage_ratio` | gauge | Group KVCM quota usage ratio (excluding EventReport) |
| `cache_manager_instance.key_count` | gauge | Per-instance key count |
| `cache_manager_instance.byte_size` | gauge | Per-instance KVCM quota bytes (excluding EventReport) |

The full list depends on the active `MetricsReporter` type. The
`kmonitor` reporter populates the most complete set of metrics.

The GetHostCacheState phase metrics above are nested:
`meta_indexer.get_io_time_us` is inside `meta_searcher.indexer_get_time_us`, and
the indexer/projection/reduction phases are inside `manager.prefix_match_time_us`.
Progressive local queries pipeline backend reads with projection, so these
intervals can overlap and must not be added together. `get_io_time_us` is a
historical name; with `storage_type=local` it contains no Redis network operation.

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
