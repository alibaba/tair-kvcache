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

For `QT_PREFIX_MATCH_WITH_MAMBA`, register both F and L location spec groups (classified by the first character of the group name). `GetHostCacheState` evaluates only these groups: it requires a complete Full prefix and a complete Linear state at the resume point. Other registered groups, including W, are ignored without rejecting the request. This mode does not validate sliding-window completeness or require `sliding_window_tokens`.

For `QT_PREFIX_MATCH_WITH_SWA`, register F/W location spec groups and a positive `model_deployment.sliding_window_tokens` (token count). `GetHostCacheState` rounds the window up to blocks using the instance `block_size`. It returns the longest Full prefix whose trailing SWA window is complete; near the prompt start, the available shorter history is sufficient. This query type applies to `GetHostCacheState`.

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
    "include_checksums": true,
    "block_mask": {
        "offset": 0
    }
}'
```

`include_checksums` 默认为 `false`；只有显式设为 `true`，服务端才把已保存的 checksum 放入响应。
响应中每个 `locations[].location_specs[]` 都可能包含独立的 `checksum`。它是不透明字符串，
非空表示返回了校验值；空字符串或字段省略表示未返回校验值。
`GetCacheLocationsByBackend` 使用同名开关和相同语义。

## Start Write Cache
```bash
curl -g -vvv -X POST http://localhost:6382/api/startWriteCache \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_126",
    "instance_id": "test_instance_id_2",
    "block_keys": [1234],
    "token_ids": [],
    "write_timeout_seconds": 10
}'
```

若希望在指定存储类型保留副本，可在 Start Write Cache 请求中增加：

```json
{
  "storage_replica_requirement": {
    "storage_type": "ST_TAIRMEMPOOL",
    "min_location_count": 1
  }
}
```

也可排除 EventReport，使其余存储类型（包括 DRAM 和 SSD）合计满足数量门槛：

```json
{
  "storage_replica_requirement": {
    "excluded_storage_types": ["ST_EVENT_REPORT_L1P5", "ST_EVENT_REPORT_L2"],
    "min_location_count": 1
  }
}
```

顶层 `min_replica_count` 必须不传或为 0。新字段不传或为 `null` 时保持旧行为。指定类型与非空排除列表必须二选一；排除模式的 `storage_type` 不传或为 `ST_UNSPECIFIED`。空对象、两种模式同时设置、未知枚举数值、排除列表中的 `ST_UNSPECIFIED`、指定 EventReport 写入以及非正数量均返回 `INVALID_ARGUMENT`。重复排除项不改变含义。HTTP 沿用忽略未知枚举名称的解析规则，混合列表仅保留可识别项，须使用正确枚举名或数值；不提供对 JSON 原文的严格校验。

指定模式仅统计目标类型的有效 Location，不足时每个 key 本次最多分配一份；客户端仍须写入返回的 URI 并调用 Finish Write Cache。数量是补写门槛，不保证并发上限、不同后端分布或可读副本数下限。显式类型覆盖 Group 的 PREFER/ALWAYS 偏好和本次 StartWrite 的 Mark 引导，但仍受 `storage_candidates`、可写性及配额限制。

类型约束按后端注册配置识别，`ST_VCNS_3FS` 与普通 `ST_3FS` 分别处理。VCNS 返回的 Location 仍可使用原有 `ST_3FS` 协议类型；服务端根据其 URI 对应的后端识别配置类型并计数，不要求响应的 `type` 与请求类型字面相等。

排除模式只统计未被排除类型的 Location；普通补写从过滤后的候选执行原 Group 策略，不写入被排除类型。范围内可执行且尚未覆盖的 Mark 优先引导到具体后端，即使数量已达标也可补写；Mark 沿用迁移准入路径，可以指向普通候选名单外的后端。目标类型被排除时忽略本次 Mark 引导并保留 Mark，暂时不可用时沿用原退路。Finish 清标规则不变。已有 SSD 满足数量要求时，不会仅因 DRAM 副本被回收而补回 DRAM。

在前置检查通过且需要普通分配时，指定类型无可写候选、类型配额已满或排除后无符合 Group 策略的候选，沿用既有映射返回 `INSTANCE_NOT_EXIST`；Group 总配额已满或分配失败返回 `INTERNAL_ERROR`。若全部候选不可用，可能在计数前就返回 `INTERNAL_ERROR`。指定模式不跨类型回退，排除模式不回退到排除类型，均不把分配失败当作跳过成功。详细计数、Mark 和失败语义见[设计说明](../design/start_write_storage_replica.md)。

## Finish Write Cache
```bash
curl -g -vvv -X POST http://localhost:6382/api/finishWriteCache \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_127",
    "instance_id": "test_instance_id_2",
    "write_session_id": "session_id_from_start_write",
    "success_blocks": {
        "bool_masks": {
          "values": [true]
        }
    },
    "checksum_batches": [
        {
            "location_spec_name": "tp0",
            "checksums": ["crc64:0000deadbeef"]
        }
    ]
}'
```

Note: To use the Finish Write Cache API, you need to replace "session_id_from_start_write" with the actual write_session_id returned by the Start Write Cache API.

`checksum_batches` 可省略；每组通过 `location_spec_name` 标识独立 payload，其 `checksums` 必须与 Start
Write Cache **返回并捕获的紧凑 session batch** 等长，而不是原始请求或成功子集。失败位置保留占位但
不会使用。空、重复、StartWrite 未分配的 spec name、长度错误、非法 success mask 或与 StartWrite 不同的
`instance_id` 都会被拒绝，且不会消费 session，可以修正后重试。checksum 由调用方计算并编码为
opaque `string`，Manager 原样保存非空值。空字符串表示未提供，不更新已有 checksum，也不新增 Redis 字段；
成功位置也可使用空字符串占位。某个 block 未分配该 spec 时，对应的占位值会被忽略。
字段号 5 的 `locations` 只为滚动升级保留并已废弃，client 仍按旧行为发送，但当前 server 不用它做
checksum 对齐；新调用方使用字段号 6 的 `checksum_batches`。详细调用链
见[Checksum 元数据存储](../design/checksum_metadata.md)。

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
arriving meanwhile wait until snapshot commit or abort; both wait directions
are bounded by `snapshot_delta_drain_timeout_ms` (10 seconds by default).
Timeout returns retryable `SNAPSHOT_IN_PROGRESS`; a timed-out delta does not
abort the snapshot or acquire a mutation lease. Different reporters remain
independent. A second concurrent snapshot also receives `SNAPSHOT_IN_PROGRESS`.

KVCM keeps the location id stable and appends only its reserved `s_v`
parameter to each event-report URI. After all metadata writes and `Sync`
succeed, KVCM publishes the in-memory committed token and returns it as
`committed_snapshot_version`. A successful complete snapshot makes that token
the reporter's strict query fence: locations composed entirely of older or
legacy generations become invisible immediately, while the existing reclaimer
removes their KVCM metadata asynchronously and never sends a physical DELETE to
the external reporter URI. During later snapshot I/O, strict queries continue
to recognize only the last committed generation; the candidate becomes visible
when it commits. After an admitted snapshot failure, or before the first
successful snapshot after process recovery, queries temporarily accept all
well-formed historical generations to avoid false negatives from in-place
metadata replacement.

After a KVCM restart, REGISTER makes well-formed historical cache metadata
readable again. The first ADD/DELETE creates a new process-local generation and
continues normally. `snapshot_required=true` with an empty committed version is
an advisory, not a delta admission requirement; realtime-only reporters may
ignore it. A snapshot-capable reporter may reconcile later.

Event-report metadata remains a cache index rather than proof of physical
existence. Soft recovery mode, failed snapshot writes, and mixed-generation
locations may still return stale candidates; a failed physical cache read must
be treated as a normal cache miss. Reporter registration and liveness remain
hard visibility gates, and malformed URI metadata is rejected.

Snapshots should be rare: event gap, explicit repair, or an optional
very-low-frequency fallback. The 30-second server-side minimum interval is a
safety limit, not a recommended reporting period.
Callers must not set `s_v`. `EVENT_HOST_DOWN` is terminal and must be sent
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
                  "uri": "event_report://physical-storage:9600/gpu/123?size=4096",
                  "checksum": "crc64:0000deadbeef"
                }
              ]
            }
          ]
        }
      }
    ]
}'
```

`BLOCK_ADD` 和 `BLOCK_SNAPSHOT` 的每个 spec 都可携带调用方计算的 `checksum`，类型为字符串。
非空字符串表示提供了值；空字符串等同未提供。
Manager 将其作为 opaque 值保存，查询时仍需设置 `include_checksums=true`。完整生命周期语义见
[Checksum 元数据存储](../design/checksum_metadata.md)。

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
    "detail_level": 1,
    "include_checksums": true
}'
```

Get Cache Meta 与 Get Cache Location 一样，仅在 `include_checksums=true` 时为每个返回
`location_specs[]` 携带非空的 `checksum`。
