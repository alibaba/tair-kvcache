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
Request-scoped gauges are emitted only once after each new sample; a later
scrape omits them until another request updates them. State gauges such as
`service.request_queue_size` remain visible on every scrape.

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
| `http.request_counter` | counter | Parsed ReportEvent HTTP requests observed by the instrumented handler |
| `http.service_query_counter` | counter | ReportEvent HTTP requests with a completed ServiceCallGuard sample |
| `http.request_parse_time_us_sum` | counter | Cumulative JSON-to-protobuf parsing time for ReportEvent HTTP requests (us) |
| `http.service_callback_time_us_sum` | counter | Cumulative ReportEvent HTTP callback time, including request-context setup and teardown (us) |
| `http.response_serialize_time_us_sum` | counter | Cumulative protobuf-to-JSON response serialization time (us) |
| `http.handler_time_us_sum` | counter | Cumulative ReportEvent arena-handler time, including arena cleanup (us) |
| `http.service_query_rt_us_sum` | counter | Cumulative ServiceCallGuard query time for ReportEvent HTTP requests (us) |
| `http.request_context_rt_us_sum` | counter | Cumulative request-context time captured before ServiceCallGuard reporting and access logging (us) |
| `http.service_finalize_time_us_sum` | counter | Cumulative ServiceCallGuard response-debug, metrics-reporting, and access-log time (us) |
| `http.request_receive_wait_time_us_sum` | counter | Cumulative time cinatra waits for and assembles complete HTTP requests (us) |
| `http.io_event_loop_lag_us_sum` | counter | Cumulative maximum I/O-loop timer lag observed since the previous instrumented request on the same I/O thread (us) |
| `http.response_build_time_us_sum` | counter | Cumulative cinatra HTTP response header/buffer construction time (us) |
| `http.socket_write_time_us_sum` | counter | Cumulative time spent awaiting cinatra's full-response socket `async_write` (us) |
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
| `data_storage.healthy_status` | gauge | Storage backend health (1/0) |
| `data_storage.storage_usage_ratio` | gauge | Storage usage ratio |
| `cache_manager.write_location_expire_size` | gauge | Expired write locations |
| `cache_manager_group.usage_ratio` | gauge | Group capacity usage ratio |
| `cache_manager_instance.key_count` | gauge | Per-instance key count |
| `cache_manager_instance.byte_size` | gauge | Per-instance byte size |

The full list depends on the active `MetricsReporter` type. The
`kmonitor` reporter populates the most complete set of metrics.

The `http.*` metrics are currently emitted only for `/api/reportEvent` and
carry the same `api_name`, `instance_group`, `instance_id`, and optional
`type` labels as its service collector. Calculate request-weighted stage
averages from counter deltas, for example:

```promql
rate(kvcm_http_request_parse_time_us_sum[1m])
  / rate(kvcm_http_request_counter[1m])

rate(kvcm_http_service_query_rt_us_sum[1m])
  / rate(kvcm_http_service_query_counter[1m])

rate(kvcm_http_socket_write_time_us_sum[1m])
  / rate(kvcm_http_request_counter[1m])
```

`request_parse`, `service_callback`, and `response_serialize` partition the
instrumented handler; `handler_time` additionally includes arena cleanup.
`service_finalize` is a nested finalization phase inside `service_callback` and
must not be added to it.
`request_receive_wait` starts when the connection coroutine begins waiting
for the HTTP header and ends when the body is assembled, so it also contains
keep-alive idle time and is not pure network-transfer time. `io_event_loop_lag`
uses a 10-ms timer per I/O thread and attributes the maximum timer delay seen
between two instrumented requests to the next one. It diagnoses event-loop
head-of-line blocking from synchronous handlers rather than an exact
per-request queue duration; it can overlap `request_receive_wait` and is not
an additive latency stage. `response_build` and `socket_write` occur after
the handler returns; socket write ends when the server-side `async_write`
completes and excludes client response reading. Client send/read/JSON parsing
still requires client-side instrumentation.

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
