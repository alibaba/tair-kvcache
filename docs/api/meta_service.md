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
    "include_checksums": true,
    "block_mask": {
        "offset": 0
    }
}'
```

`include_checksums` 默认为 `false`；只有显式设为 `true`，服务端才把已保存的 checksum 放入响应。
响应中每个 `locations[].location_specs[]` 都可能包含独立的 `checksum` 和 `checksum_present`。只有
`checksum_present=true` 时该 spec 的数值有效；`checksum=0` 是合法值，不能用数值本身判断是否存在。
老数据没有 checksum 时返回 `checksum_present=false`。`GetCacheLocationsByBackend` 使用同名开关和
相同语义。

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
            "checksums": ["0"]
        }
    ]
}'
```

Note: To use the Finish Write Cache API, you need to replace "session_id_from_start_write" with the actual write_session_id returned by the Start Write Cache API.

`checksum_batches` 可省略；每组通过 `location_spec_name` 标识独立 payload，其 `checksums` 必须与 Start
Write Cache **返回并捕获的紧凑 session batch** 等长，而不是原始请求或成功子集。失败位置保留占位但
不会使用。空、重复、StartWrite 未分配的 spec name、长度错误、非法 success mask 或与 StartWrite 不同的
`instance_id` 都会被拒绝，且不会消费 session，可以修正后重试。数值由调用方或 KVCM TransferClient
计算，Manager 将其作为 opaque `int64` 保存，`0` 同样有效。

`uri_batches` 同样可省略，用于在下述 safe-replacement 策略允许时提交 backend 实际写入 URI；当前只有
`pace` 的动态 allocation 可以合法地产生与 StartWrite 不同的 URI。每组 `uris` / `uri_present` 都必须与同一个
紧凑 session 等长；`uri_present=false` 要配空字符串占位，
`true` 要配非空、可解析的 URI。URI 与 checksum batch 可以覆盖不同 spec 集合，但所有非空 batch 的长度必须
一致，且 spec 必须属于该 StartWrite session。checksum/URI batch 各最多 4096 组、每类 value 总数最多
`2^20`，spec name 最多 1024 bytes；单 URI 上限为 16 KiB，请求内 URI 总量上限为 48 MiB。

`uri_batches` 不是任意重定向接口。结构/预算校验在 session 消费前完成；随后 Manager 在单 location RMW 中
把 actual URI 与 StartWrite URI 做 safe-replacement 校验。非 `pace` scheme 只接受 canonical URI 完全相同；
`pace` 只允许 allocation 输出的 path/offset、`node_id` 和 `range_id` 改变，authority、path 是否为空、`size`、
`media_type` 及其他 query 参数必须保持，且不得注入新参数。因此示例式的 `file:///actual/...` 改写会被拒绝。
可解析但违反该语义策略的 URI 是 session 消费后的发布失败，不能修正后重放同一 session；调用方必须隔离、
查询确认并按 backend allocation identity 补偿。

### FinishWrite wire/API 版本边界

- proto tag 5 `locations` 只为 legacy/滚动升级保留并已废弃。当前 server 不用它做 checksum 对齐，也不把它
  作为 actual URI 更新来源；
- tag 6 是 `checksum_batches`，tag 7 是 `uri_batches`。C++ client API v3 的
  `FinishWrite(..., FinishWriteOptions)` 保留既有 ABI，会继续发送 legacy tag 5，并可发送 tag 6；
- C++ client API v4 使用独立的 `FinishWriteIntegrityOptions` / `FinishWriteWithIntegrity`，发送 tag 6/7，且故意
  不发送 tag 5。加载动态 client 的进程必须先检查 `KVCMStagedChecksumRuntimeApiVersion() >= 4`；旧 subclass
  只有在 URI batch 为空时才可回退到 v3，带 URI 时返回 `ER_CHECKSUM_UNAVAILABLE`；
- 老 server 会忽略未知 tag 6/7，所以 RPC 成功不等于 checksum/actual URI 已持久化。升级必须 server-first。
  checksum 提交通过 `include_checksums=true` 查询确认每个目标 spec 的 presence/value/exact URI；URI-only 提交
  则确认完整 location/spec/exact URI，不应要求未提交的 checksum presence。

### 发布、部分失败与调用方责任

对单个 location，本次请求**实际携带的** actual URI、checksum 和 `CLS_SERVING` 状态在同一次 metadata RMW
中发布；字段省略是 patch 语义，URI-only 更新会保留已有 checksum，因此可以形成“新 URI + 旧 checksum”的
最终状态，只是不会暴露 RMW 的中间状态。调用方若覆盖 payload，必须同时提交新 checksum，或明确运行在没有
checksum baseline 的模式。不同 key/location 之间不是事务。主发布任一槽失败时，Manager
会把原始失败位置和**本次所有 intended-success location**（包括可能已经成功发布的槽）纳入 authoritative
cleanup：从持久存储重新读取，写入 `CLS_DELETING` 并执行 post-write `Sync`，同时保留已知 actual URI。
只有已经 durable fence 且 exact serialized value 仍匹配的 `DELETING` location 才进入物理删除；物理删除失败
会保留 metadata，在 backend 没有 lease/conditional-delete 能力时不会盲目重试，以免 URI 被复用后删除新对象。

如果调用方在写失败后已经拿到部分 actual URI，应以全失败 success mask、空 legacy `locations` 和完整 session
长度的稀疏 `uri_batches` 调用 v4 Finish，让 Manager 回收真实分配。v4 abort 只有在本地明确为 `NOT_SENT` 时
才可降级为 legacy zero-mask Finish；结果为 `SENT_OR_UNCERTAIN` 时绝不能重试同一 session。legacy 降级可以
消费 session，但不能携带 dynamic URI，也不能证明新分配对象已回收。

若 Finish 请求可能已经发送但回包丢失或报错，结果属于 uncertain commit：checksum 路径按 key/spec 精确查询
presence/value/URI，URI-only 路径查询 location/spec/URI；只有观察到本次写入的 exact metadata 才可接受，
否则隔离这些 key，并执行 `RemoveCache` 或等价补偿。终态 cleanup 失败的内部指标为
`cache_cleanup.permanent_failure_location_count{stage}`，Prometheus 名为
`kvcm_cache_cleanup_permanent_failure_location_count{stage}`；它不替代调用方查询与隔离。

上述 cleanup 只能回收调用方实际拿到并提交的 known actual URI。如果单次 SDK Put 在报错/超时前已经产生
部分动态分配，却没有返回这些 URI，Manager 无法从 StartWrite URI 推导真实 allocation identity；这类隐藏副作用
必须由 backend 的原子性、lease/失败回滚或人工对账闭环处理。

详细 checksum 计算、校验阶段和清理状态机见[数据完整性设计](../design/data_integrity.md)。

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

KVCM keeps the location id stable and appends only its reserved `s_version`
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
                  "uri": "event_report://physical-storage:9600/gpu/123?size=4096",
                  "checksum": "0",
                  "checksum_present": true
                }
              ]
            }
          ]
        }
      }
    ]
}'
```

`BLOCK_ADD` 和 `BLOCK_SNAPSHOT` 的每个 spec 都可携带调用方计算的 `checksum`；HTTP 中 `int64`
按 protobuf JSON 规则推荐使用十进制字符串，`checksum_present=true` 用于区分合法值 `0` 与未提供。
Manager 将其作为 opaque 值保存，查询时仍需设置 `include_checksums=true`。完整生命周期语义见
[数据完整性设计](../design/data_integrity.md)。

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
`location_specs[]` 携带有效的 `checksum` / `checksum_present`。
