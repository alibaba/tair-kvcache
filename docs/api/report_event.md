# ReportEvent 与查询接口行为说明

本文面向推理引擎、Subscriber 和其他 KVCM 调用方，说明 `ReportEvent` 上报之后
`GetCacheLocation`、`GetCacheLocationsByBackend` 和 `GetHostCacheState` 会看到什么。

本文描述的是调用方可以依赖的接口行为。内部实现和设计取舍见
[ReportEvent 增量上报与可选全量对账设计](../design/report_event_snapshot_uri_version.md)。

## 1. 调用方先记住这 10 条

1. `EVENT_NODE_REGISTER` 推荐在 reporter 启动时发送，但不是数据上报的硬前置；第一条合法
   HEARTBEAT、ADD、DELETE 或 SNAPSHOT 可以懒初始化 reporter。
2. 只支持实时事件的引擎可以一直使用 `ADD/DELETE`，永远不发 snapshot。
3. 支持全量的引擎应把 snapshot 当作低频纠偏手段，不能分页，也不应高频发送。
4. 一次 snapshot 是一个 reporter 在一个 storage type 下、跨全部 medium 的完整状态。
5. 客户端不得在 URI 中填写 `s_version`；KVCM 会自动追加。
6. `committed_snapshot_version` 是 32 位十六进制 opaque generation，不能按大小比较；成功完整
   snapshot 后它会成为该 reporter 的严格查询栅栏。
7. `snapshot_required=true` 表示请求到达时当前进程还没有该 reporter generation；创建 generation
   的首条 ADD/DELETE 自身仍返回 `true`，下一条事件才返回 `false`。该提示不会阻止合法增量。
8. 当前进程尚未收到该 reporter 的任何合法事件，或节点 unavailable 时，查询不会返回该
   节点的数据；节点存活状态是硬门槛。
9. snapshot 失败、进程重启或从未成功 snapshot 的 soft 模式可能返回 stale cache candidate；
   成功 snapshot 会立即隐藏完全属于旧 generation 的 location。真实 cache 读取失败仍必须按 miss 处理。
10. `ReportEvent` 不是整批事务。出现 `item_results` 时，应按事件下标逐项处理；成功项可能已经生效。

## 2. Reporter 身份与数据粒度

### 2.1 Reporter 身份

一个独立 reporter 由以下三个字段确定：

```text
instance_id + storage_type + host_ip_port
```

因此：

- 同一 host 的 `ST_EVENT_REPORT_L1P5` 和 `ST_EVENT_REPORT_L2` 是两个独立 reporter；
- 同一 host 在两个 instance 中互不影响；
- 同一 instance 的两个 host 互不影响；
- snapshot 栅栏、generation、限流和节点生命周期都按上述身份隔离。

`RegisterInstance` 与 `EVENT_NODE_REGISTER` 不是同一层注册：

- `RegisterInstance` 持久化 instance 的静态配置，并在 KVCM 重启时恢复；
- `EVENT_NODE_REGISTER` 只管理某个 reporter 的进程内 liveness/medium 状态，是可选的启动信号；
- KVCM 重启不会要求客户端重新 `RegisterInstance`，也不会要求重新发送
  `EVENT_NODE_REGISTER`；下一条合法 ReportEvent 会补齐运行时 reporter 状态。

### 2.2 Block、medium 与 spec

KVCM 更新的最小逻辑身份是：

```text
block_key + medium + spec.name
```

- `medium` 是 block 级属性，例如 `gpu`、`hbm`、`memory`、`disk`；
- `spec.name` 区分一个 block 内的多个物理组成部分，例如不同 TP、full attention 或 mamba state；
- 同一 `block_key` 可以同时存在于多个 medium；
- 同一 `block_key + medium` 可以包含多个不同的 spec name；
- 同一 `block_key + medium + spec.name` 是集合语义，不是引用计数。

如果引擎内部有多个完全相同的物理 block，它们最终映射到相同
`block_key + medium + spec.name`，KVCM 无法区分“有 1 份”还是“有 N 份”。调用方应在上报前
去重；一次 DELETE 会删除这个逻辑 spec。

如果同一个 block 有多个不同 spec，snapshot 时应把它们合并到同一个
`BlockSnapshotItem.specs` 中，不能把同一 `block_key + medium` 拆成多个 snapshot item。

## 3. 推荐调用时序

### 3.1 纯实时模式

适用于只有增量事件、没有全量能力的推理引擎：

```text
[可选 REGISTER] -> ADD / DELETE / HEARTBEAT
                         |
                         +--> 持续 ADD / DELETE / HEARTBEAT
```

第一条通过参数校验并获得 delta lease 的 ADD 或 DELETE 会创建当前进程的 generation。
REGISTER、HEARTBEAT，以及在获得 lease 前就被拒绝的增量不会创建 generation。获得 lease 后
如果 metadata 写入失败，generation 可能已经建立；它只是内部对账标签，不代表该事件已经写入。

REGISTER 的作用是提前建立节点存活状态、声明 medium，并在同一 KVCM 进程中从 HOST_DOWN/
grace cleanup 后显式重新启用 reporter。正常数据面不应因为 KVCM 重启而停下来等待再次 REGISTER。

### 3.2 实时 + 低频全量模式

```text
[可选 REGISTER]
   |
   +--> 正常发送 ADD / DELETE / HEARTBEAT
   |
   +--> 发现事件断档、显式修复或低频校准
            |
            +--> 发送一个完整 SNAPSHOT
            |
            +--> 继续发送 ADD / DELETE
```

Snapshot 不是心跳，不应按秒周期发送。默认最小成功 snapshot 间隔为 30 秒，实际值由
EventReport storage 的 `snapshot_min_interval_ms` 配置。snapshot 关闭写门后等待已准入 delta
排空，以及新 delta 等待 in-flight snapshot 完成，共用 10 秒的默认上限，由同一 storage 的
`snapshot_delta_drain_timeout_ms` 配置。

### 3.3 进程启动时合并请求

Fresh reporter 可以在一个请求中发送：

```text
REGISTER -> ADD/DELETE -> HEARTBEAT
```

也可以发送：

```text
REGISTER -> SNAPSHOT -> HEARTBEAT
```

对于当前 KVCM 进程内没有 tombstone 的 fresh reporter，也可以把
ADD/DELETE/SNAPSHOT 放在 REGISTER 前面：合法 mutation 会先懒初始化 reporter，后面的
REGISTER 再补充 medium 并刷新 liveness。为了让启动日志和节点观测更清晰，仍推荐启动阶段
先单独 REGISTER。

这个“顺序无关”只适用于 fresh reporter。同一 KVCM 进程内，如果该 reporter 已经被
HOST_DOWN 或 grace cleanup 留下 tombstone，请求按事件顺序处理，REGISTER 必须位于
ADD/DELETE/SNAPSHOT 之前。若 mutation 排在 REGISTER 前面，前面的 mutation 返回
`NODE_NOT_REGISTERED`，后面的 REGISTER 仍可成功；调用方随后按原顺序重试失败的 mutation。
最稳妥的方式是单独发送 REGISTER，成功后再发送 mutation。

### 3.4 KVCM 重启

Instance/InstanceGroup 和 cache metadata 会持久化；reporter 节点表、liveness 和 generation
是进程内状态。KVCM 重启后的行为是：

1. 收到该 reporter 的第一条合法事件前，Redis 中残留的历史 event-report metadata 不可见；
2. 第一条 HEARTBEAT、ADD、DELETE 或 SNAPSHOT 会自动重建 reporter 节点状态，不要求再次 REGISTER；
3. 如果第一条只是 HEARTBEAT，响应为 `snapshot_required=true`、
   `committed_snapshot_version=""`，格式合法的历史 metadata 可以重新成为 cache candidate；
4. 如果第一条是 ADD/DELETE，它正常成功并建立新 generation；该次响应仍为
   `snapshot_required=true`，后续事件复用 generation 时变为 `false`；
5. 该增量只更新明确涉及的 spec，不删除其他历史 block/spec；
6. snapshot-capable 调用方可以稍后补一次完整 snapshot；
7. realtime-only 调用方可以继续只发增量。

这保证 Subscriber 不需要感知 KVCM 重启，正常心跳或数据上报会自动恢复链路，但可能重新暴露
少量历史 cache candidate。

## 4. ReportEvent 公共字段

HTTP 接口为 `POST /api/reportEvent`，gRPC 方法为 `ReportEvent`。

```json
{
  "trace_id": "subscriber-20260727-001",
  "instance_id": "model-a",
  "host_ip_port": "10.0.0.8:8080",
  "storage_type": "ST_EVENT_REPORT_L1P5",
  "events": []
}
```

字段要求：

| 字段 | 要求 |
| --- | --- |
| `instance_id` | 必填，必须已经通过 `RegisterInstance` 创建 |
| `host_ip_port` | 必填，必须稳定且与查询侧识别的 worker 地址一致；不能包含 location id 分隔符 `#` |
| `storage_type` | 非空请求必填，只支持 `ST_EVENT_REPORT_L1P5`、`ST_EVENT_REPORT_L2` |
| `events` | 有序事件列表；空列表是 no-op success |
| `trace_id` | 建议每次请求唯一，便于排查 |

每个 `EventItem.event_type` 必须与同一个 item 中实际出现的 `event_params` oneof 字段一致。
即使 `node_register`、`heartbeat`、`host_down` 的消息内容为空，对应字段本身也必须出现。
event type 与 payload 缺失或错配时，该 item 返回 `INVALID_ARGUMENT`，不会把错误 payload
当成目标事件执行。

InstanceGroup 必须把对应 EventReport storage 配置在
`event_report_storage_candidates` 中，否则返回 `INSTANCE_NOT_EXIST`。

## 5. 各事件行为

### 5.1 EVENT_NODE_REGISTER

```json
{
  "event_type": "EVENT_NODE_REGISTER",
  "node_register": {
    "mediums": ["gpu", "memory", "disk"]
  }
}
```

行为：

- 建立或刷新该 reporter 的节点状态；
- 重复 REGISTER 可安全重试，medium 列表会合并；但每次成功 REGISTER 都是新的 reporter
  lifecycle 栅栏，会使更早生命周期中尚未落盘的 mutation/cleanup 失效，因此不能把
  REGISTER 当作 HEARTBEAT 高频发送，也不应与普通数据请求无序并发；
- REGISTER 本身不创建 `committed_snapshot_version`；
- REGISTER 后可以直接发 ADD/DELETE；
- KVCM 重启后不必重新 REGISTER；
- 同一 KVCM 进程内，HOST_DOWN 或超过 grace 被清理会留下 tombstone；此时必须显式
  REGISTER 才能重新启用同一 reporter identity，避免迟到事件意外复活已下线节点；
- tombstone 后如果在同一个请求里携带 REGISTER 和 mutation，REGISTER 必须排在 mutation
  前面；建议先单独 REGISTER，成功后再重试 mutation。

### 5.2 EVENT_HEARTBEAT

```json
{
  "event_type": "EVENT_HEARTBEAT",
  "heartbeat": {
    "system_status": {
      "engine_version": "v1.2.3",
      "load": "0.42"
    }
  }
}
```

行为：

- 只刷新节点存活时间；
- `system_status` 是观测信息，不参与 cache 正确性判断；
- fresh reporter 或 KVCM 重启后的第一条 HEARTBEAT 会懒初始化节点，返回 `OK`；
- 同一进程内已被 HOST_DOWN/grace cleanup tombstone 的 reporter 返回 `NODE_NOT_REGISTERED`；
- unavailable 但仍处于 grace period 的 reporter 可通过 HEARTBEAT 恢复；
- HEARTBEAT 不创建或改变 generation。

### 5.3 EVENT_BLOCK_ADD

```json
{
  "event_type": "EVENT_BLOCK_ADD",
  "block_add": {
    "block_key": "123",
    "medium": "gpu",
    "specs": [
      {
        "name": "full_attention:group=0:tp=0",
        "uri": "event_report://10.0.0.8:9600/gpu/123?size=4096"
      },
      {
        "name": "mamba_state:group=0:tp=0",
        "uri": "event_report://10.0.0.8:9600/gpu/123?size=1024"
      }
    ]
  }
}
```

要求：

- `block_key` 必须是可解析的十进制整数文本；
- `medium` 非空，且不能包含 location id 分隔符 `#`；
- `specs` 非空；
- 每个 `spec.name` 非空，且同一事件内不能重复；
- 每个 URI 必须合法；
- 客户端 URI 不能带 `s_version`；
- 旧字段 `block_add.uri` 已废弃，只有 `specs` 生效。

行为：

- 按 spec name merge；
- 本次未涉及的其他 spec 会保留；
- 同一逻辑 spec 再次 ADD 会覆盖旧 URI；
- 首条合法 ADD 可以创建 generation；
- KVCM 会给每个 URI 追加一个 `s_version`。

### 5.4 EVENT_BLOCK_DELETE

```json
{
  "event_type": "EVENT_BLOCK_DELETE",
  "block_delete": {
    "block_key": "123",
    "medium": "gpu",
    "spec_names": [
      "full_attention:group=0:tp=0"
    ]
  }
}
```

要求：

- `block_key`、`medium` 非空且合法；`medium` 不能包含 location id 分隔符 `#`；
- `spec_names` 非空；
- spec name 非空且同一事件内不能重复。

行为：

- 只删除指定 `block_key + medium + spec_names`；
- 同 location 中未指定的 spec 保留；
- block、medium/location 或 spec 不存在都视为幂等成功，响应仍为 `OK`；
- 同一请求存在缺失的 `block_key + medium` 时，KVCM 只打一条聚合 debug 日志，
  记录缺失 target 数量，不逐 block 打日志；
- 首条合法 DELETE 即使没有实际删到数据，也可以创建 generation。

### 5.5 EVENT_BLOCK_SNAPSHOT

```json
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
            "uri": "event_report://10.0.0.8:9600/gpu/123?size=4096"
          },
          {
            "name": "mamba_state:group=0:tp=0",
            "uri": "event_report://10.0.0.8:9600/gpu/123?size=1024"
          }
        ]
      },
      {
        "block_key": "456",
        "medium": "memory",
        "specs": [
          {
            "name": "full_attention:group=0:tp=0",
            "uri": "event_report://10.0.0.8:9600/memory/456"
          }
        ]
      }
    ]
  }
}
```

Snapshot 的完整性规则：

- `blocks` 是该 reporter 在当前 storage type 下、跨全部 medium 的完整 block 集合；
- 不能拆成多个 ReportEvent 请求分页；
- 每个 item 的 specs 是该 `block_key + medium` 的完整 spec 集合；
- 同一 `block_key + medium` 只能出现一次；
- 相同 block key 可以出现在不同 medium；
- `medium` 不能包含 location id 分隔符 `#`；
- 每个 block 的 specs 必须非空，且 spec name 不能重复；
- `blocks=[]` 表示该 reporter 当前没有任何 cache，会异步清理其全部旧 location。

Snapshot 的更新语义：

- snapshot 中出现的 `block_key + medium` 会原地替换完整 spec 集合；
- 已存在 block 中被本次省略的 spec 会被替换掉；
- 整个 snapshot 中被省略的旧 block 由异步 cleanup 最终删除；
- 成功后返回新的 generation；
- 失败时不回滚已经完成的部分 metadata 写入，应完整重试。

### 5.6 EVENT_HOST_DOWN

```json
{
  "event_type": "EVENT_HOST_DOWN",
  "host_down": {}
}
```

行为：

- 必须是请求中的唯一事件；
- 立即把 reporter 标记为不可用并从节点表注销；
- 查询立即隐藏该 reporter，metadata 异步清理；
- 重复 HOST_DOWN 幂等；
- 同一 KVCM 进程内后续重新使用该 reporter identity 时必须先 REGISTER；KVCM 重启会清空
  进程内 tombstone，下一条合法汇报可以再次懒初始化。

## 6. 请求组合与批处理

| 请求内容 | 行为 |
| --- | --- |
| fresh reporter 的 `REGISTER + ADD/DELETE + HEARTBEAT` | 允许；REGISTER 位于 mutation 前后均可，仍推荐先 REGISTER |
| fresh reporter 的 `REGISTER + SNAPSHOT + HEARTBEAT` | 允许；REGISTER 位于 SNAPSHOT 前后均可，仍推荐先 REGISTER |
| tombstone reporter 的 `REGISTER + mutation` | 允许，但按事件顺序处理；REGISTER 必须位于 ADD/DELETE/SNAPSHOT 之前 |
| 多个 ADD/DELETE | 允许，按请求顺序解释 |
| SNAPSHOT + ADD/DELETE | 整个请求在写入前返回 `INVALID_ARGUMENT` |
| 两个 SNAPSHOT | 整个请求在写入前返回 `INVALID_ARGUMENT` |
| HOST_DOWN + 任意其他事件 | 整个请求在写入前返回 `INVALID_ARGUMENT` |
| 空 events | no-op success；不查 backend，也不刷新节点或 generation，响应状态字段保持默认值 |

同一请求内，如果相同 `block_key + medium + spec.name` 被多次修改，最后一次操作获胜。例如：

```text
ADD A -> DELETE A -> ADD A(new URI)
```

最终保存最后一次 ADD 的 URI。被折叠的事件共享最终 metadata 写入结果。

不同请求之间没有为相同逻辑 spec 提供全局事件序号。调用方必须保证同一 reporter 的相同
`block_key + medium + spec.name` 按正确顺序送达；并发更新不同 spec name 可以安全 merge。

## 7. Snapshot 与增量并发

同一 reporter 的执行顺序是：

```text
已经开始的 delta 完成
        |
snapshot 关闭新的 delta 入口
        |
snapshot replace + commit/abort
        |
等待中的 delta 使用 commit 后的 generation 继续
```

调用方可依赖：

- snapshot 不会越过已经获得写 lease 的 delta；
- snapshot 开始后到达的 ADD/DELETE 会在服务端等待，不返回 `DELTA_IN_PROGRESS`；
- ADD/DELETE 等待超过统一写门超时后返回可重试的 `SNAPSHOT_IN_PROGRESS`，且不会获得
  delta lease 或取消 snapshot；
- snapshot commit 后，等待中的增量不会再被刚完成的 snapshot 覆盖；

当 meta storage 使用 `cached + persistent_type=async_redis` 时，snapshot 与增量具有相同的
Redis 返回点：mutation 成功进入异步队列并更新 local cache 后即可 commit/返回，不等待 Redis
consumer flush。`committed_snapshot_version` 是当前 KVCM 进程的查询栅栏，不是 Redis flush
checkpoint；响应后的 pipeline 失败不会改写已经返回的 ReportEvent 状态。
- snapshot abort 后，等待中的增量继续执行；
- 第二个并发 snapshot 返回 `SNAPSHOT_IN_PROGRESS`；
- 不同 host、instance 或 storage type 不共享该写门。

snapshot 关闭写门后最多等待 EventReport storage 的
`snapshot_delta_drain_timeout_ms`（默认 10 秒）让已经获得 lease 的 delta 排空。如果等待
超时，服务端会 abort 尚未开始 metadata 写入的 candidate、重新打开写门并返回可重试的
`SNAPSHOT_IN_PROGRESS`；committed generation 不变。调用方应退避后重发完整 snapshot。

新 delta 等待 in-flight snapshot 使用同一个超时。超时后 snapshot 继续执行，失败的
ADD/DELETE 返回 `SNAPSHOT_IN_PROGRESS`，调用方应退避并幂等重试对应失败项。调用方
HTTP/gRPC timeout 应大于服务端配置值，并保留传输层超时后的幂等重试。

## 8. Snapshot 限流

限流维度为：

```text
instance_id + storage_type + host_ip_port
```

- 只在成功 snapshot commit 后开始计时；
- 限流期内返回 `SNAPSHOT_RATE_LIMITED`；
- `retry_after_ms` 给出当前请求至少还要等待的时间；
- 失败 snapshot 不启动冷却，可立即完整重试；
- ADD、DELETE、REGISTER、HEARTBEAT 不受 snapshot 限流影响。

推荐处理：

```text
SNAPSHOT_RATE_LIMITED -> 等待 retry_after_ms + 少量 jitter -> 重发完整 snapshot
SNAPSHOT_IN_PROGRESS  -> SNAPSHOT 失败时重发完整 snapshot
                      -> ADD/DELETE 失败时幂等重试对应失败项
```

## 9. 响应字段

```json
{
  "header": {
    "status": {
      "code": "OK",
      "message": ""
    },
    "request_id": "..."
  },
  "item_results": [],
  "committed_snapshot_version": "0123456789abcdef0123456789abcdef",
  "retry_after_ms": "0",
  "snapshot_required": false,
  "extra_info": ""
}
```

### 9.1 committed_snapshot_version

- 当前 KVCM 进程中，该 reporter 已发布的 generation；
- 32 个十六进制字符；
- opaque、随机、不可排序；
- 第一条合法 ADD/DELETE 或成功 SNAPSHOT 可创建/更新；
- 普通增量沿用当前值；
- REGISTER、HEARTBEAT，以及获得 delta lease 前失败的 mutation 不创建新值；
- 增量获得 lease 后若 metadata 写入失败，响应仍可能带新建的 generation；是否成功以
  `header.status`/`item_results` 为准；
- KVCM 重启或 reporter 注销后为空；
- 空 events 是不解析 reporter 状态的纯 no-op，因此该响应字段为空；不能用空 events 查询 generation；
- 成功完整 snapshot 后作为严格查询 fence；普通 delta 创建的 generation 不会单独把 soft
  reporter 切到 strict。

客户端不能把“数值更大”解释为“更新”，只能做字符串相等判断。

### 9.2 snapshot_required

`snapshot_required` 表示“本次请求到达时当前进程还没有该 reporter generation”。对于首条合法
ADD/DELETE，KVCM 会在请求内创建 generation 并返回到 `committed_snapshot_version`，但本次响应
仍保留 `snapshot_required=true`；下一条事件复用该 generation 时才返回 `false`：

| 场景 | 值 |
| --- | --- |
| fresh REGISTER 后 | `true` |
| KVCM 重启后的第一条 HEARTBEAT | `true` |
| 只有 REGISTER/HEARTBEAT、还没有合法 mutation | `true` |
| 创建 generation 的第一条合法 ADD/DELETE | `true` |
| 后续复用已有 generation 的 ADD/DELETE | `false` |
| 成功 SNAPSHOT 后 | `false` |
| invalid-only 首批增量 | 仍为 `true` |
| realtime-only reporter | 可忽略该提示，继续发合法增量 |

空 events 不读取 reporter 状态，`snapshot_required` 保持 proto 默认值 `false`。需要确认当前
generation 时应查看正常 REGISTER/HEARTBEAT/数据事件的响应，不能发送空 events 探测。

### 9.3 item_results

- 全部事件成功时为空或不出现在 JSON 中；
- 任一事件失败时，长度与 `events` 完全相同，顺序一一对应；
- `header.status.code` 是第一个失败项的错误码；
- `ReportEvent partially failed` 不代表整批回滚；
- 调用方只重试失败项时，仍要保持原有事件顺序。

例如 fresh reporter 发送 `[ADD, REGISTER, HEARTBEAT]`，三个事件都成功，`item_results` 为空，
ADD 创建 generation；显式 REGISTER 的先后顺序不会成为数据面的失败原因。若其中某个 item
参数非法，只有该 item 返回 `INVALID_ARGUMENT`，同批其他合法 item 仍可生效。

对于已有 tombstone 的 reporter，`[ADD, REGISTER]` 按顺序返回
`[NODE_NOT_REGISTERED, OK]`；后面的 REGISTER 不会追溯性地让前面的 ADD 成功。调用方应在
REGISTER 成功后重试该 ADD。

### 9.4 retry_after_ms 与 extra_info

- `retry_after_ms` 只在 `SNAPSHOT_RATE_LIMITED` 时有业务含义；
- proto JSON 的 `uint64` 可能编码成字符串，调用方应同时兼容字符串和数字；
- `extra_info` 是 InstanceGroup 透传的 opaque JSON，不能用于核心流程判断。

## 10. 错误码与调用方动作

| 错误码 | 含义 | 建议动作 |
| --- | --- | --- |
| `OK` | 全部事件成功 | 正常继续 |
| `INVALID_ARGUMENT` | 请求形状或某事件参数非法 | 不盲重试；修正失败项 |
| `INSTANCE_NOT_EXIST` | instance、MetaSearcher 或指定 storage type backend 不存在 | 检查 Instance/InstanceGroup/storage 配置 |
| `NODE_NOT_REGISTERED` | 同一进程内 reporter 已被 HOST_DOWN/grace cleanup tombstone | 确认节点确实重新上线后先单独 REGISTER，成功后再按原顺序重试失败的 mutation |
| `SNAPSHOT_IN_PROGRESS` | 同 reporter 已有 snapshot、snapshot 未能及时排空已准入 delta，或 delta 未能及时等到 snapshot 完成 | 按失败事件类型退避重试：SNAPSHOT 重发完整全量，ADD/DELETE 幂等重试失败项 |
| `SNAPSHOT_RATE_LIMITED` | 距上次成功 snapshot 太近 | 按 `retry_after_ms` 重试完整 snapshot |
| `SNAPSHOT_REQUIRED` | 等待期间 reporter 被注销/关闭等状态变化 | 重新 REGISTER；realtime-only 可直接重试增量 |
| `INTERNAL_ERROR` | metadata replace、commit 等内部失败 | 保留实时链路；完整重试 snapshot 或幂等重试增量 |

## 11. 查询行为

### 11.1 所有查询共享的可见性规则

Event-report location 只有同时满足以下条件才可参与业务查询：

1. location id 能解析出正确的 storage type、medium 和 reporter host；
2. 当前 KVCM 进程已收到该 reporter 的合法 REGISTER、HEARTBEAT 或数据事件；
3. reporter 当前 available；
4. location 至少包含一个 spec；
5. 每个 URI 语法合法；
6. URI 如果带 `s_version`，只能出现一次且必须是 32 位十六进制文本。

version 可见性分为两种状态：

- **strict**：一次完整 snapshot 成功后启用。location 必须至少包含一个当前
  `committed_snapshot_version`；完全由正在写入的 snapshot candidate、旧 generation
  或无 `s_version` 的 legacy spec 组成的 location 不可见；
- **soft**：KVCM 重启后、尚未成功完成完整 snapshot 时，或一次已经准入的 snapshot
  写入失败后启用。此时接受全部格式合法的历史 generation 和 legacy URI，允许少量陈旧
  candidate，由后续增量或完整 snapshot 自愈。

同一 location 可以混合多个合法 generation。strict 模式下只要至少一个 spec 属于当前
committed generation，location 仍可见，以免按 spec 增量刷新时误隐藏有效数据；candidate
只有成功 commit 后才成为新的 committed。若 snapshot 失败则回到 soft，格式合法的 candidate
和历史 generation 可重新参与查询。一个 location 中只要有任一 malformed spec，整个
location fail closed。查询不比较 `s_version` 的大小。

`EventReportBackend::MightExist` 是底层无 instance/location-id 上下文的保守接口，只能对它
能够反查 owner 的当前 generation 做判断。上述三个公共查询入口使用带 instance 和完整
location 的查询检查器，不以 `MightExist` 的 token 规则代替本节业务可见性规则。

### 11.2 GetCacheLocation

HTTP 接口为 `POST /api/getCacheLocation`：

```json
{
  "trace_id": "query-001",
  "instance_id": "model-a",
  "query_type": "QT_BATCH_GET",
  "block_keys": [123, 456],
  "block_mask": {
    "offset": 0
  }
}
```

返回的 EventReport URI 会包含 KVCM 追加的 `s_version`。调用方应把 URI 交给相应 connector
读取；真实读取失败按普通 cache miss 处理，不能把 metadata 命中当作数据一定存在。

当多个 medium 或多个 backend 返回相同 spec name 时，普通查询可能按选择策略合并/去重。
需要明确指定 backend 时使用 `GetCacheLocationsByBackend`；需要同时表达多个 cache 组成部分时，
应使用不同且稳定的 spec name。

### 11.3 GetCacheLocationsByBackend

该接口当前支持 `QT_BATCH_GET`，可通过 `backend_selectors` 明确选择
`ST_EVENT_REPORT_L1P5`、`ST_EVENT_REPORT_L2` 等 backend，适合验证两种 EventReport storage 的
隔离状态。

`location_spec_names` 不只是返回结果的投影条件，也是 backend/peer 选择前的候选条件：

- 为空时，location 中任意合法 spec 都可使该 location 成为候选；
- 非空时，location 至少包含一个请求的 spec name 才能成为候选；
- 同一 EventReport location 由 `storage_type + medium + host_ip_port` 标识，其中各 spec 必须属于
  同一个 reporter endpoint；location 命中过滤后，selector 从第一个合法 spec URI 提取 peer；
- 多个 peer 的 prefix/coverage 相同时按 endpoint 字典序选择，保证调用方按 spec 分批查询时
  各批次不会因为容器遍历顺序选择不同 peer；
- peer 选择完成后，响应仍只保留 `location_spec_names` 指定的 specs。

因此 spec name 是 reporter 与查询方之间的稳定协议字段，不能用 object size 代替：不同 cache
group 即使 byte size 相同，也必须使用不同且稳定的 spec name。调用方如果每个 key 需要的 spec
不同，应先按 spec name 分组发起查询，再按原 object key 合并结果；`location_spec_names` 是一次
请求级过滤条件，不是 per-key 数组。确定性 tie-break 只消除无序遍历造成的抖动；若各组候选
peer 集合不同，分组请求无法保证得到全局最优公共 peer。该能力需要后续扩展 per-key spec filter
或等价的联合选择接口。

### 11.4 GetHostCacheState

HTTP 接口为 `POST /api/getHostCacheState`：

```json
{
  "trace_id": "host-state-001",
  "instance_id": "model-a",
  "query_type": "QT_PREFIX_MATCH",
  "block_cache_keys": [100, 101, 102, 103],
  "medium": ["gpu", "memory"]
}
```

返回：

```json
{
  "header": {
    "status": {
      "code": "OK"
    }
  },
  "hosts": [
    {
      "host_ip_port": "10.0.0.8:8080",
      "local": "3",
      "p2p_1_fetch": "0",
      "p2p_1_total_match": "3"
    }
  ]
}
```

规则：

- 输入 block key 的顺序有意义；
- 每个 host 从第一个 key 开始连续计数，遇到第一个 miss 即停止；
- 第一个 key 就 miss 的 host 不返回；
- `medium` 为空表示考虑所有 medium；
- `medium` 非空时只使用指定 medium；
- `QT_UNSPECIFIED` 使用 RegisterInstance 时配置的 `default_query_type`；
- 支持 `QT_PREFIX_MATCH` 和 `QT_PREFIX_MATCH_WITH_MAMBA`，其他类型返回参数错误；
- 同一个 host 在多个 backend 的有效 cache 会按 host 汇总参与匹配；
- `local` 包含同一 host 的 subscriber 与 Vineyard 上报；
- 非混合注意力对 full local-miss 使用 Prefix 选择远端 Vineyard；混合注意力先对
  FullAttention group 使用 Prefix，再对 Mamba local-miss spec 使用 Coverage；
- `p2p_1_fetch` 表示各 P2P 阶段实际选中并拉取的 spec 所属的去重 block key 数；
- `p2p_1_total_match` 表示本地 cache 与实际选中的远端 spec 合并后的最终前缀；
- 远端 P2P 候选只使用 `ST_EVENT_REPORT_L2`，且不会让 `local` 为 0 的 host 出现在响应中；
- reporter unavailable 时，该 host 对应的 event-report location 不参与匹配。

成功完整 snapshot 后，`GetHostCacheState` 会立即忽略完全属于旧 generation 的 location。
snapshot 失败、KVCM 重启恢复或 realtime-only reporter 仍使用 soft metadata，因此
`local` 在这些模式下仍可能是 false positive。

## 12. 节点生命周期与查询

| Reporter 状态 | ReportEvent | 查询 |
| --- | --- | --- |
| 当前进程从未见过 | 第一条合法 REGISTER/HEARTBEAT/ADD/DELETE/SNAPSHOT 懒初始化 | 第一条合法事件前不可见，之后按 liveness 可见 |
| registered + available | 所有合法事件按规则执行 | 可见 |
| heartbeat timeout、仍在 grace | ADD/DELETE/SNAPSHOT 仍可能被接受；HEARTBEAT 可恢复 | 不可见 |
| grace 内 HEARTBEAT 恢复 | 保留原 generation 和已写 metadata | 重新可见 |
| 超过 grace 被 unregister | tombstone 后 mutation 返回 `NODE_NOT_REGISTERED` | 不可见，metadata 异步清理 |
| HOST_DOWN | 立即 unregister，异步清理 | 立即不可见 |
| tombstone 后重新 REGISTER | 可继续增量；generation 初始为空 | 未清干净的合法历史 metadata 可能重新可见 |
| KVCM 重启 | 下一条合法事件自动恢复，不要求 REGISTER | 首条事件前隐藏；之后历史 metadata 可能重新可见 |

调用方不要用 ADD/DELETE 代替 HEARTBEAT：mutation 不刷新 heartbeat。暂时 unavailable 时虽然
metadata 写入可能成功，但恢复 HEARTBEAT 前查询仍不可见。

## 13. Snapshot 非原子窗口与重试

Snapshot 使用稳定 location 原地更新，不提供原子查询视图：

| 阶段/失败点 | 响应 generation | 查询可能看到什么 | 调用方动作 |
| --- | --- | --- | --- |
| 参数校验失败 | 旧值 | 原数据 | 修正请求 |
| 限流/并发拒绝 | 旧值 | 原数据 | 按错误码退避 |
| snapshot 写入中 | 旧值 | 新旧合法 candidate | 不做版本大小判断 |
| 部分 replace 失败 | 旧值 | 已写入的新数据 + 未覆盖的旧数据 | 完整重试 |
| commit 失败 | 旧值 | 已写入的新数据可能可见 | 完整重试 |
| commit 成功、cleanup 未完成 | 新值 | 完全属于旧 version 的遗漏 block 已不可见 | cleanup 仅回收空间 |
| cleanup 与新写重叠 | 新值 | 新写必须保留 | 无需特殊处理 |
| KVCM 重启 | 空 | 第一条合法汇报后历史数据可能可见 | 正常继续心跳/增量；可选低频全量 |

Snapshot-capable 调用方的安全重试单位始终是“整个完整 snapshot”，而不是只重试上次失败的
block 子集。

Snapshot cleanup 只删除 KVCM 中的 event-report metadata，不会向 reporter URI 对应的外部
cache 发起物理 DELETE。清理以稳定 location 为粒度：如果一次增量在新 generation 下只刷新
了其中一个 spec，location 内可能暂时混合当前、旧或无 version 的 legacy spec；只要存在
当前或 in-flight spec，清理就保留整个 location，避免误删刚成功的增量。旧 sibling spec
由后续完整 snapshot 替换或回收。

清理扫描耗时通过
`event_report.snapshot_cleanup_scan_latency_ms{instance_id,host,type}` 暴露。典型场景下，
1 个 instance、10 个 reporter、每台 5000 个 block，一次单 reporter snapshot 约产生 5000
次 metadata replace，并以 1000 key 为批次扫描该 instance 约 5 万个 key。10 台同时完整对账
约有 5 万次 replace、累计约 50 万次 key 检查；具体 Redis 命令数受 backend batching 影响。
如果扫描延迟或 cleanup backlog 持续升高，应考虑按 reporter 建反向索引，而不是继续扩大
全 instance 扫描频率。

## 14. 调用方上线检查清单

- [ ] InstanceGroup 配置了正确的 `event_report_storage_candidates`
- [ ] reporter 自身启动时建议发送 REGISTER；不依赖 KVCM 重启后再次 REGISTER
- [ ] 正确区分“进程重启后的懒恢复”和“HOST_DOWN/grace tombstone 后显式 REGISTER”
- [ ] 实时-only 模式不会因 `snapshot_required=true` 停止发送增量
- [ ] 客户端 URI 不包含 `s_version`
- [ ] `block_key + medium + spec.name` 在发送前已去重
- [ ] snapshot 合并同一 block 的全部 specs，且包含全部 medium
- [ ] snapshot 不分页、不与 ADD/DELETE/HOST_DOWN 混发
- [ ] 按 `item_results` 下标处理部分失败
- [ ] 正确处理 `SNAPSHOT_IN_PROGRESS`、`SNAPSHOT_RATE_LIMITED` 和 `retry_after_ms`
- [ ] HTTP/gRPC timeout 能覆盖 snapshot 栅栏等待，并支持幂等重试
- [ ] 持续发送 HEARTBEAT，不用数据事件代替 HEARTBEAT
- [ ] HOST_DOWN 单独发送
- [ ] GetHostCacheState 使用有序 block keys，并正确处理 prefix 含义
- [ ] metadata 命中后的物理 cache 读取失败按 miss 处理

## 15. 自动化测试覆盖矩阵

以下矩阵是本文接口契约与自动化测试的对应关系。测试文件：

- Backend UT：`kv_cache_manager/data_storage/test/event_report_backend_test.cc`
- Manager UT：`kv_cache_manager/manager/test/cache_manager_test.cc`
- Meta UT：`kv_cache_manager/manager/test/meta_searcher_test.cc`
- PR HTTP 集成：`integration_test/meta_service/http_interface_test.py`
- 基础集成：`integration_test/meta_service/test_report_event.py`
- Snapshot 集成：`integration_test/meta_service/test_report_event_snapshot.py`
- 重启集成：`integration_test/meta_service/test_report_event_restart.py`

PR HTTP 集成由 `//integration_test/meta_service:http_interface_test` 启动真实 KVCM 进程，属于默认
`//integration_test/...` CI。基础 ReportEvent 脚本当前没有 Bazel target；Snapshot target 带
`manual` 标签，重启脚本也需显式执行，因此三者不属于默认 GitHub CI。覆盖结论必须以显式执行
结果为准；同样，普通 GitHub check 通过不代表已经执行 ASAN。

Snapshot target 是外部服务测试，单独执行 `bazel test` 不会替它启动 KVCM。先按脚本头部说明
启动 meta/admin HTTP 服务，再显式传入端口。功能与容量入口分别为：

```bash
bazel run //integration_test/meta_service:test_report_event_snapshot -- \
  --host localhost --http_port 56020 --admin_http_port 56040 \
  --instance_id event_report_functional --skip-bench

bazel run //integration_test/meta_service:test_report_event_snapshot -- \
  --host localhost --http_port 56020 --admin_http_port 56040 \
  --instance_id event_report_bench --only-bench

# 仅运行小 block / 大批量单请求基准；默认使用进程内 local metadata backend
bazel run //integration_test/meta_service:test_report_event_snapshot -- \
  --host localhost --http_port 56020 --admin_http_port 56040 \
  --instance_id event_report_large_delta \
  --bench-test test_20_large_single_request_delta_scaling
```

同一 KVCM 进程上重复执行时，fixture 会通过 `listStorage` 校验已有 storage 的 type 和 EventReport
时序配置；配置不一致应立即失败，不能把任意 `addStorage` 错误当作“可能已存在”后继续测试。
heartbeat/grace 短时序测试使用独立 storage/instance group，不得缩短功能与容量用例的主 storage。

| ID | 用户行为 | 自动化覆盖 |
| --- | --- | --- |
| I-01 | instance、host、storage type 隔离 | `ScopesAreIsolatedByInstanceAndReporterHost`；`test_15/16/17_dual_type_*` |
| I-02 | 同 host 的 L1.5/L2 独立增删和 HOST_DOWN | `TestReportEventL1P5L2BlockAddAreIsolated`；基础集成 `test_15~17` |
| R-01 | REGISTER 幂等并合并 medium | `RegisterNodeWithMediums`；snapshot 集成 `test_01/02` |
| R-02 | fresh REGISTER+ADD+HEARTBEAT 同请求 | `TestReportEventRegisterThenFirstDeltaInSameRequest`；snapshot 集成 `test_11_mixed_batch` |
| R-03 | fresh reporter 的 ADD 排在显式 REGISTER 前仍成功并懒初始化 reporter | `TestReportEventDeltaBeforeExplicitRegisterSucceedsInSameRequest`；snapshot 集成 `test_30_*` |
| R-04 | REGISTER+SNAPSHOT+HEARTBEAT 同请求 | snapshot 集成 `test_29_*` |
| R-05 | 空 events 是纯 no-op，响应 generation/snapshot_required/retry 使用默认值 | snapshot 集成 `test_12_empty_batch` |
| R-06 | 成功、部分失败和 no-op 的响应字段，以及 InstanceGroup extra_info 透传 | snapshot 集成 `test_01/02/03/10/12/13/33` |
| D-01 | HOST_DOWN/grace tombstone 后 ADD/DELETE 被拒绝；REGISTER 必须先成功，再重试 mutation | snapshot 集成 `test_16b/26_*` |
| D-02 | 无 REGISTER、无初始 snapshot 的第一条 ADD 懒初始化、创建 generation 并可查询 | `TestReportEventLazilyRestoresReporterWithoutRegisterOrSnapshot`；snapshot 集成 `test_30_*` |
| D-03 | 无初始 snapshot 的第一条 DELETE 创建 generation且幂等 | `TestReportEventFirstDeleteWithoutSnapshotCreatesReusableVersion`；snapshot 集成 `test_31_*` |
| D-04 | invalid-only 首批增量不创建 generation | `TestReportEventInvalidFirstDeltaDoesNotCreateVersionButPartialBatchDoes`；snapshot 集成 `test_26_*` |
| D-05 | 部分失败批次保留合法项进度和 item_results 顺序 | 同上；snapshot 集成 `test_23_*` |
| D-06 | 未预注册 reporter 的多线程首批增量只懒建一个节点并发布一个 generation | `ConcurrentFirstDeltasPublishExactlyOneReusableVersion`；snapshot 集成 `test_28_*` |
| D-07 | ADD 按 spec name merge、保留未触碰 spec | `TestReportEventBlockAddMergesLocationSpecs`；snapshot 集成 `test_05b`、`test_22_*` |
| D-08 | DELETE 精确删除 spec、missing delete 批量幂等 OK | `TestReportEventBlockDeleteRemovesLocationSpecs`、`TestReportEventMissingBlockDeletesRemainSuccessfulAsOneBatch`；snapshot 集成 `test_06/07` |
| D-09 | 同请求 ADD/DELETE 最后操作获胜 | `TestReportEventSameRequestDeltaOrderUsesLastOperationPerSpec`；snapshot 集成 `test_25_*` |
| D-10 | 并发写不同 spec 不互相丢失 | snapshot 集成 `test_24_*` |
| D-11 | 重复物理逻辑身份是 set 而非 refcount | `TestReportEventRepeatedPhysicalDeltaUsesSetNotReferenceCountSemantics` |
| D-12 | KVCM 只追加一个合法 s_version，不改变客户端 URI 其他部分 | 基础集成 `_assert_profile_specs_in_locations`；snapshot 集成 `_assert_reporter_scope` |
| D-13 | 第一条 delta 已创建 generation、metadata 写失败时准确报错，重试复用 generation | `TestReportEventFirstDeltaMetadataFailureReportsFailureAndReusesGeneration` |
| D-14 | 同一 spec 的折叠事件共享最终 metadata 写入失败结果 | `TestReportEventFoldedDeltaEventsShareFinalWriteFailure` |
| S-01 | snapshot 跨 medium 完整上报、响应返回 generation | `TestReportEventSnapshotReplacesCompleteSpecSetPerBlock`；snapshot 集成 `test_17/22` |
| S-02 | 同 block 跨 medium 合法，同 block+medium 重复非法 | `TestReportEventRejectsCanonicalDuplicateSnapshotKeysButAllowsDifferentMedia`；snapshot 集成 `test_27_*` |
| S-03 | snapshot block 内重复 spec name 被拒绝 | `TestReportEventRejectsDuplicateSpecNamesWithinSnapshotBlock` |
| S-04 | snapshot 参数错误无写入副作用 | `TestReportEventRejectsDuplicatePhysicalSnapshotItemsWithoutStateChange`；snapshot 集成 `test_19/27` |
| S-05 | 空 snapshot 清空 reporter | `TestReportEventEmptySnapshotCommitsAndReclaimsPreviousBlocks`；snapshot 集成 `test_17` |
| S-06 | snapshot 遗漏 block 后 cleanup 最终删除 | `TestReportEventSnapshotCommitReclaimsOnlyStaleReporterLocations`；snapshot 集成 `test_22` |
| S-07 | snapshot 和 delta 两种先后顺序均收敛 | `TestReportEventDeltaAlreadyAdmittedThenSnapshotWins`、`TestReportEventSnapshotGateThenDeltaInheritsNewTokenAndWins`；snapshot 集成 `test_24` |
| S-08 | 第二个 snapshot busy，成功后 rate limit 带 retry_after | `ConcurrentSnapshotsHaveExactlyOneWinner`、`SnapshotRateLimitReturnsRetryDelay`；snapshot 集成 `test_18/24` |
| S-09 | snapshot 部分 replace 失败后仍可读并可完整重试；成功路径不等待 persistent Sync | `TestReportEventPartialSnapshotFailureKeepsCacheReadableAndRetryConverges`、`TestReportEventSnapshotCommitsWithoutWaitingForPersistentSync` |
| S-10 | cleanup 与下一轮写入 CAS/attempt epoch 竞争不删除新值，并在 epoch 变化后按批次提前退出 | `TestSnapshotCleanupPreservesInFlightStableLocationUntilAbort`、`TestSnapshotCleanupCASPreservesLocationRefreshedAfterScan`、`TestOldSnapshotCleanupDoesNotDeleteLaterAbortedAttemptWrites` |
| S-11 | snapshot commit 后增量刷新 mixed/legacy location，旧 cleanup 不删除新写 | `TestSnapshotCleanupPreservesPostCommitDeltaOnMixedGenerationLocation`、`TestSnapshotCleanupPreservesCurrentDeltaBesideLegacySpec`；snapshot 集成 `test_32_*` |
| S-12 | snapshot cleanup 只删 metadata，不调用外部 URI backend | `TestMetadataOnlyLocationDeleteSkipsPhysicalBackend`、`TestCleanupLocationsByPredicateSubmitsExactObservedValue` |
| S-13 | unavailable reporter 可完成 snapshot，但 HEARTBEAT 恢复前查询保持隐藏 | `TestReportEventSnapshotWhileUnavailableCommitsButStaysHiddenUntilHeartbeat` |
| Q-01 | soft 接受合法历史/legacy；strict 只接受至少含 committed spec 的 location，in-flight candidate 在 commit 前不单独可见 | `TestGetCheckLocDataExistFuncFencesVersionsAfterSuccessfulSnapshot`、`TestGetCheckLocDataExistFuncEventReportUriValidationMatrix` |
| Q-02 | 空 spec、坏 URI、重复/畸形 s_version 整条 location fail closed | `TestGetCheckLocDataExistFuncEventReportUriValidationMatrix` |
| Q-03 | malformed location id、错误 storage type、未知 host 不可见 | 同上；`TestGetCheckLocDataExistFunc_MissingEventReportBackendFailsClosed` |
| Q-04 | 节点 available/unavailable/unregistered 控制真实查询入口 | `TestGetCacheLocationEnforcesReporterLifecycleAndBatchOrdering`；snapshot 集成 `test_16a/16b` |
| Q-05 | GetHostCacheState 的 prefix、medium、默认 query type | `TestGetHostCacheState`、`TestGetHostCacheStatePrefixMatchWithMamba`；snapshot 集成 `test_16` |
| Q-06 | 纯增量模式同时通过 GetCacheLocation 和 GetHostCacheState 校验 | snapshot 集成 `test_11/26/28` |
| Q-07 | GetCacheLocationsByBackend 同样执行 reporter liveness 过滤 | `TestGetCacheLocationsByBackendWithBackendSelectors`；snapshot 集成 `test_16a` |
| Q-08 | 三个查询入口在 timeout 隐藏和 HEARTBEAT 恢复时结果一致 | snapshot 集成 `test_16a_heartbeat_timeout_then_recovery` |
| Q-09 | snapshot 成功后即使异步 cleanup 尚未运行，遗漏的旧 version block 也立即不可见 | `TestSuccessfulSnapshotImmediatelyFencesOmittedOldVersionBeforeCleanup`、`TestGetCacheLocationEnforcesReporterLifecycleAndBatchOrdering` |
| Q-10 | backend 查询在 peer 选择前按 spec 过滤，并对同覆盖率 peer 确定性择优 | `TestGetCacheLocationsByBackendWithBackendSelectors`；`EventReportPrefixFiltersRequestedSpecBeforePeerSelection`；`EventReportCoverageFiltersRequestedSpecBeforePeerSelection`；`EventReportPrefixTieBreaksByPeerAddress`；`EventReportCoverageTieBreaksByPeerAddress`；HTTP 集成 `test_event_report_requested_spec_filters_before_peer_selection` |
| Q-11 | requested spec 不存在时 Prefix/Coverage 都返回与输入等长的空结果，不回退到其他 spec | `EventReportUnknownRequestedSpecReturnsNoCandidate`；HTTP 集成 `test_event_report_requested_spec_filters_before_peer_selection` |
| Q-12 | requested spec 按 any-of 语义匹配，重复 name 不改变结果；匹配同一 location 的非首个 spec 时仍使用该 reporter endpoint | `EventReportRequestedSpecMatchesAnyNameIncludingNonFirstSpec` |
| Q-13 | requested-spec gap 会终止 Prefix，但 Coverage 可跳过 gap 继续返回后续命中；响应投影后 `spec_size` 始终等于实际 specs 数量 | `EventReportRequestedSpecGapStopsPrefixButNotCoverage`；`TestGetCacheLocationsByBackendWithBackendSelectors` |
| L-01 | 自动 heartbeat timeout 隐藏、grace 内恢复原 generation | `MightExistFollowsAutomaticLivenessAndFullReporterLifecycle`；snapshot 集成 `test_16a` |
| L-02 | unavailable 期间增量可写但保持不可见，HEARTBEAT 后恢复 | `TestReportEventLazilyRestoresReporterWithoutRegisterOrSnapshot`；snapshot 集成 `test_16a` |
| L-03 | 超过 grace 后按 generation 原子 unregister，最终 metadata 删除持有 generation lease，旧 cleanup 不伤重新注册数据 | `HeartbeatRecoveryFencesCleanupAlreadySelectedByLivenessLoop`、`ConditionalUnregisterCannotRemoveNewGeneration`、`CleanupLeaseFencesReregisterThroughFinalDeleteStage`；snapshot 集成 `test_16b` |
| L-04 | HOST_DOWN 原子捕获 generation 并注销，立即隐藏、异步清理、重复调用幂等 | `HostDownAtomicallyCapturesGenerationAndLeavesReregisteredNodeIntact`、`TestHostDownMakesAlreadyAdmittedDeltaInvisibleWithoutDeadlock`；snapshot 集成 `test_08/09/20` |
| L-05 | 同 instance 另一 host、同 host 另一 instance 不受 liveness 影响 | `TwoInstancesSameHostIsolated`；snapshot 集成 `test_16a` |
| L-06 | ADD/DELETE/SNAPSHOT 不刷新 heartbeat，数据事件不能替代 HEARTBEAT | `DataMutationsDoNotRefreshHeartbeat`；snapshot 集成 `test_16b` |
| P-01 | KVCM 重启前旧数据持久化，首条新汇报前隐藏 | 重启集成 `test_report_event_restart.py --phase prepare/verify` |
| P-02 | 重启后首条 HEARTBEAT 无需 REGISTER 即恢复历史 candidate，响应提示 snapshot_required | 同上 |
| P-03 | 重启后首条增量成功并保留未触碰历史数据 | `TestReportEventRestartKeepsHistoricalCacheAndAcceptsDeltaWithoutSnapshot`；重启集成 |
| P-04 | 重启后可选 snapshot 最终重新收敛 | 重启集成 |
| C-01 | Close/Unregister/HOST_DOWN 唤醒 snapshot/delta waiter | `CloseUnblocksSnapshotAndDeltaWaiters`、`UnregisterCancelsSnapshotWaitingForActiveDelta`、`TestHostDownCancelsSnapshotAlreadyWritingMetadata` |
| C-02 | 不同 reporter 的 snapshot 不互相阻塞 | `OtherReporterIsNotBlockedBySnapshot`；snapshot 集成 `test_18` |
| C-03 | 自动 liveness cleanup 在 snapshot 等待 active delta 时注销并唤醒 waiter | `AutomaticLivenessCleanupCancelsSnapshotWaitingForActiveDelta` |
| C-04 | 已进入 metadata I/O 的旧 lifecycle 请求不能在 HOST_DOWN、REGISTER、新 snapshot 后恢复写入 | `TestOldDeltaCannotCrossReporterLifecycleAfterReregisterAndSnapshot` |
| C-05 | snapshot 等待 active delta 超时后 abort candidate、保留 committed generation 并重新打开 delta 写门 | `SnapshotDrainTimeoutAbortsCandidateAndReopensWriteGate` |
| C-06 | delta 等待 in-flight snapshot 超时后返回可重试错误，且不会中止 snapshot；节点校验不会在超时栅栏前无限等待 | `DeltaWaitTimeoutReturnsSnapshotInProgressWithoutAbortingSnapshot`、`TestReportEventDeltaGateTimeoutReturnsSnapshotInProgressAndCanRetry` |
| C-07 | mutation lease 每个 RMW 阶段只获取一次，但阻塞在 metadata read 的旧请求仍可被 HOST_DOWN/新 lifecycle 抢占 | `TestBatchMutationWriteLeaseIsAcquiredOncePerRmwPhase`、`TestBatchMutationWriteLeaseFailurePreventsAllWrites`、`TestHostDownCancelsSnapshotAlreadyWritingMetadata`、`TestHostDownMakesAlreadyAdmittedDeltaInvisibleWithoutDeadlock`、`TestOldDeltaCannotCrossReporterLifecycleAfterReregisterAndSnapshot` |
| V-01 | reporter host、ADD/DELETE/SNAPSHOT medium 含 `#` 时拒绝且无写入副作用 | `RegisterNodeWithMediums`、`TestReportEventRejectsInvalidRequestsAndMapsItemErrors`；snapshot 集成 `test_19_*` |
| V-02 | instance/host/storage type/instance backend 的公共字段校验 | `TestReportEventRejectsInvalidRequestsAndMapsItemErrors`；snapshot 集成 `test_14_*` |
| V-03 | event_type 与 oneof payload 缺失/错配时该 item fail closed；同批其他合法 mutation 可独立懒初始化并生效 | `TestReportEventRejectsMismatchedPayloadsWithoutSideEffects`；snapshot 集成 `test_13/33` |
| V-04 | ADD 的 key、medium、spec 数量/name/URI/s_version 完整校验且无副作用 | `TestReportEventMutationValidationMatrixHasNoSideEffects`；snapshot 集成 `test_19/33` |
| V-05 | DELETE 的 key、medium、spec_names 数量/空值/重复完整校验且无副作用 | `TestReportEventMutationValidationMatrixHasNoSideEffects`；snapshot 集成 `test_19/33` |
| V-06 | SNAPSHOT 的 key、medium、spec、URI、s_version、重复 block 完整校验且无副作用 | `TestReportEventMutationValidationMatrixHasNoSideEffects`；snapshot 集成 `test_19/27/33` |
| V-07 | snapshot+delta、双 snapshot、HOST_DOWN 混发均在任何副作用前整批拒绝 | `TestReportEventRejectsRequestShapeBeforeAnySideEffect`；snapshot 集成 `test_17/33` |
| CFG-01 | snapshot interval/drain timeout 默认值、正数校验、负数拒绝、JSON/proto round trip，并由 backend 加载 | `TestEventReportStorageSpecSnapshotSettingsDefaultAndValidation`、`TestEventReportStorageSpecJsonRoundTripIncludesSnapshotSettings`、`EventReportStorageSpecProtoRoundTripPreservesSnapshotSettings`、`BasicAccessors` |
| PERF-01 | 100 线程共 1 万 ADD、50 线程共 5000 混合批次均无错误 | `EventReportBenchTest.test_17/18` |
| PERF-02 | 10 reporter × 每台 5000 blocks 完整 snapshot，并查询每台首/中/末 block | `EventReportBenchTest.test_19_ten_reporters_full_snapshot_capacity` |
| PERF-03 | 单请求 512 个跨重复 medium 的增量正确落盘；手工容量测试记录 100/1000/5000/20000 个 ADD 的总 RT 与单 event 开销 | `TestReportEventLargeDeltaBatchAcrossRepeatedMediums`、`EventReportBenchTest.test_20_large_single_request_delta_scaling` |

上表中的参数解析、故障注入、CAS 竞态等内部边界使用 UT 验证；跨 HTTP 的正常流程、并发流程、
节点生命周期和 Redis/KVCM 重启使用集成测试验证。
