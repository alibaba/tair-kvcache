# KVCacheManager MetaService API curl Examples

## Register Instance
```bash
curl -g -vvv -X POST http://localhost:6382/api/registerInstance \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_123",
    "instance_group": "test_group",
    "instance_id": "test_instance_id",
    "model_deployment": {
        "model_name": "test",
        "dtype": "fp16",
        "use_mla": false,
        "tp_size": 1,
        "dp_size": 1,
        "lora_name": "custom_lora",
        "pp_size": 1,
        "extra": "extra",
        "user_data": "custom_user_data"
    },
    "block_size": 8,
    "default_query_type": "QT_PREFIX_MATCH",
    "location_spec_infos": [
        {"name": "tp0", "size": 4096000}
    ]
}'
```
`default_query_type` is optional. When `GetHostCacheState` does not set request-level `query_type`, the service uses this registered value.

## Get Instance Info
```bash
curl -g -vvv -X POST http://localhost:6382/api/getInstanceInfo \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_124",
    "instance_id": "test_instance"
}'
```

## Get Cache Location
```bash
curl -g -vvv -X POST http://localhost:6382/api/getCacheLocation \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_125",
    "instance_id": "test_instance",
    "block_keys": [123],
    "block_mask": {
        "offset": 0
    }
}'
```

## Start Write Cache
```bash
curl -g -vvv -X POST http://localhost:6382/api/startWriteCache \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_126",
    "instance_id": "test_instance_id_2",
    "block_keys": [1234, 4567, 1234],
    "token_ids": [],
    "write_timeout_seconds": 10
}'
```

## Finish Write Cache
```bash
curl -g -vvv -X POST http://localhost:6382/api/finishWriteCache \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_127",
    "instance_id": "test_instance",
    "write_session_id": "session_id_from_start_write",
    "success_blocks": {
        "bool_masks": {
          "values": [true]
        }
    }
}'
```

Note: To use the Finish Write Cache API, you need to replace "session_id_from_start_write" with the actual write_session_id returned by the Start Write Cache API.

## Remove Cache
```bash
curl -g -vvv -X POST http://localhost:6382/api/removeCache \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_128",
    "instance_id": "test_instance",
    "block_keys": [123],
    "block_mask": {
        "offset": 0
    }
}'
```

## Report Event

完整的调用方接口契约、错误处理、查询可见性和自动化测试矩阵见
[ReportEvent 与查询接口行为说明](report_event.md)。

`reportEvent` is the cache-subscriber ingestion API. Subscribers use ordered
incremental events for steady-state traffic and may use a complete snapshot to
repair the baseline:

- Realtime-only subscriber: REGISTER and send local deltas directly; no initial
  snapshot is required.
- RTP-LLM subscriber: poll full cache status when authoritative reconciliation
  is needed, then continue sending local deltas.
- vLLM subscriber: map KV events to ordered `EVENT_BLOCK_ADD`,
  `EVENT_BLOCK_DELETE`, and snapshots after restart, full-clear, or event gaps.

`EVENT_BLOCK_SNAPSHOT` is authoritative for all GPU, CPU, and Disk cache owned
by one reporter (`instance_id + storage_type + host_ip_port`). `medium` is
specified by each block. The request must contain the complete block set and
every block's complete spec set; it cannot be paginated or mixed with
ADD/DELETE in the same request. An empty snapshot clears all media owned by
that reporter.

KVCM serializes a reporter's full snapshot and incremental mutations. A
snapshot first closes the reporter's delta write gate, waits for already
admitted deltas to finish, and then performs the full update. ADD/DELETE calls
arriving meanwhile wait until snapshot commit or abort; different reporters
remain independent. A second concurrent snapshot receives
`SNAPSHOT_IN_PROGRESS`.

KVCM keeps the location id stable and appends only its reserved `s_version`
parameter to each event-report URI. After all metadata writes and `Sync`
succeed, KVCM publishes the in-memory committed token and returns it as
`committed_snapshot_version`. The token is a reconciliation and cleanup
generation, not a query visibility fence. While a reporter is registered and
available, queries accept well-formed legacy, old, current and in-flight
generations. The existing reclaimer asynchronously removes older KVCM metadata
after a successful snapshot and never sends a physical DELETE to the external
reporter URI.

After a KVCM restart, REGISTER makes well-formed historical cache metadata
readable again. The first ADD/DELETE creates a new process-local generation and
continues normally. `snapshot_required=true` with an empty committed version is
an advisory, not a delta admission requirement; realtime-only reporters may
ignore it. A snapshot-capable reporter may reconcile later.

Event-report metadata is a soft cache index. Stale candidates and partial
snapshot writes may be returned; a failed physical cache read must be treated
as a normal cache miss. Reporter registration and liveness remain hard
visibility gates, and malformed URI metadata is rejected.

Snapshots should be rare: event gap, explicit repair, or an optional
very-low-frequency fallback. The 30-second server-side minimum interval is a
safety limit, not a recommended reporting period.
Callers must not set `s_version`. `EVENT_HOST_DOWN` is terminal and must be sent
as the only event in its request.

```bash
curl -g -vvv -X POST http://localhost:6382/api/reportEvent \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_131",
    "instance_id": "test_instance",
    "host_ip_port": "192.168.2.1:8080",
    "storage_type": "ST_EVENT_REPORT_L2",
    "events": [
      {
        "event_type": "EVENT_BLOCK_SNAPSHOT",
        "block_snapshot": {
          "blocks": [
            {
              "block_key": "123",
              "medium": "gpu",
              "specs": [
                {
                  "name": "full_attention:group=0:tp=0",
                  "uri": "event_report://physical-storage:9600/gpu/123?size=4096"
                }
              ]
            }
          ]
        }
      }
    ]
}'
```

## Trim Cache
```bash
curl -g -vvv -X POST http://localhost:6382/api/trimCache \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_129",
    "instance_id": "test_instance",
    "strategy": "TS_REMOVE_ALL_CACHE",
    "begin_timestamp": 0,
    "end_timestamp": 0
}'
```

## Get Cache Meta
```bash
curl -g -vvv -X POST http://localhost:6382/api/getCacheMeta \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_130",
    "instance_id": "test_instance",
    "block_keys": [123],
    "block_mask": {
        "offset": 0
    },
    "detail_level": 1
}'
```
