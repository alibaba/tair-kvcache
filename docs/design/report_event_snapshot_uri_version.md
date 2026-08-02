# ReportEvent 增量汇报与可选全量对账设计

> 本文描述 ReportEvent 的目标语义。代码、协议和测试应以本文为准。

## 1. 要解决的问题

推理引擎通过 `ReportEvent` 把本地 KV cache 状态汇报给 KVCM。接入方有两类：

- 只发送实时 `ADD/DELETE`，不具备全量 snapshot 能力；
- 平时发送实时事件，在事件断档或低频校准时额外发送完整 snapshot。

因此必须满足：

1. 即使没有显式 REGISTER、也从未发送过 snapshot，合法 ADD/DELETE 仍能正常工作；
2. KVCM 重启后不因进程内 version 丢失而隐藏全部历史 cache metadata；
3. snapshot-capable reporter 仍可用完整全量纠偏并回收旧 metadata；
4. 当前进程尚未收到 reporter 的任何合法事件，或节点失活时，其 metadata 不能被查询返回。

Event-report metadata 是 cache 索引，不是数据唯一副本。少量 stale candidate 可以接受，实际
读取 cache 失败时必须按普通 cache miss 处理。

## 2. 核心结论

### 2.1 增量是主链路

第一条合法 HEARTBEAT、ADD、DELETE 或 SNAPSHOT 可以懒初始化 reporter 节点状态；第一条
ADD 或 DELETE 直接建立一个进程内 version，然后执行写入。不要求客户端先 REGISTER，也
不要求先发送空 snapshot。

只支持实时事件的推理引擎可以一直不发送 snapshot。

显式 REGISTER 仍用于 reporter 自身启动时提前声明 medium 和建立 liveness；同一 KVCM
进程内发生 HOST_DOWN 或 grace cleanup 后会留下 tombstone，必须由 REGISTER 明确清除，
防止迟到数据事件复活已下线节点。KVCM 重启会清空该进程内 tombstone。

每次成功 REGISTER 同时开启一个新的 lifecycle generation：重复请求会合并 medium 并返回
成功，但会取消更早 lifecycle 中尚未进入最终 metadata 写入阶段的 mutation/cleanup。因此
REGISTER 是启动/重建边界，不是 HEARTBEAT 的替代品；调用方不应高频发送或与普通数据请求
无序并发。

这里的 REGISTER 是 `ReportEvent` 内的 `EVENT_NODE_REGISTER`，不是持久化 instance 静态配置
的 `RegisterInstance`。后者由 RegistryManager 恢复；前者只是 reporter 运行时状态，因此
可以从下一条 ReportEvent 的 `instance_id + storage_type + host_ip_port` 懒重建。

### 2.2 Snapshot 是可选的完整对账

一个 reporter 的身份是：

```text
instance_id + storage_type + host_ip_port
```

代码中的 `ReporterSnapshotKey` 只保存 `instance_id + host_ip_port`，因为每个
`EventReportBackend` 已按 `storage_type` 隔离。

一次 snapshot 是该 reporter 在当前 storage type 下、跨全部 medium 的完整 cache 状态。
`medium` 是 block 级属性；空 `blocks` 表示该 reporter 当前没有任何 cache。

### 2.3 s_version 同时用于查询栅栏与清理

KVCM 在 reporter 提供的 URI 上只追加：

```text
s_version=<opaque-version>
```

Version 的用途：

- 标记本进程内一段增量或一次 snapshot generation；
- 成功完整 snapshot 后立即隐藏完全属于旧 generation 的 location；
- snapshot 成功后帮助 reclaimer 找出旧 generation；
- 通过响应、日志和监控对齐一次对账。

查询有两种简单模式：

```text
strict：location 至少包含一个 committed version
soft：接受全部格式合法的历史 version 和 legacy URI
```

只有成功完整 snapshot 才进入 strict。KVCM 重启、纯增量尚未成功 snapshot，或一次已经准入的
snapshot 失败时使用 soft；下一次完整 snapshot 成功后立即恢复 strict。这样常规成功路径能
立即隐藏 snapshot 遗漏的旧 block，而少见的失败原地写入不会造成 cache false negative。

### 2.4 节点生命周期仍是硬门槛

满足当前 strict/soft version 规则的 metadata 可以作为 cache 候选，但 reporter 还必须：

- 当前 KVCM 进程已收到合法 REGISTER、HEARTBEAT 或数据事件并建立运行时节点；
- 当前 available；
- location id 能解析到该 reporter；
- storage type 与 backend 一致。

heartbeat timeout、HOST_DOWN、UNREGISTER 后仍按现有生命周期逻辑隐藏或清理该 reporter 的
metadata。节点清理扫描记录 location 的精确观察值，删除时做 compare-and-delete；旧生命周期
的 cleanup 即使与重新 REGISTER 后的写入重叠，也不能删除新写入。

## 3. 为什么不再持久化 version

Version 只保存在 `EventReportBackend` 内存，不再写入 MetaIndexer，也不做重启扫描恢复。

原因：

- cache metadata 不是唯一副本；
- KVCM 重启后保留大部分历史候选，比为了恢复严格 version 而引入 marker、扫描和复杂状态机
  更符合缓存场景；
- 第一条新增量可以建立新的进程内 generation；
- 后续成功 snapshot 仍能把多 generation 状态重新收敛。

代价是 KVCM 重启后可能同时存在旧、新 generation，少量已过期 cache 可能被返回。调用方
必须把真实 cache 读取失败处理成 miss。

## 4. 接口

### 4.1 ADD

```json
{
  "instance_id": "model-a",
  "host_ip_port": "10.0.0.8:8080",
  "storage_type": "ST_EVENT_REPORT_L1P5",
  "events": [{
    "event_type": "EVENT_BLOCK_ADD",
    "block_add": {
      "block_key": "123",
      "medium": "gpu",
      "specs": [{
        "name": "tp0",
        "uri": "rtp-llm://10.0.0.8:9600/hbm/123"
      }]
    }
  }]
}
```

客户端不得携带 `s_version`，由 KVCM 追加。

### 4.2 DELETE

DELETE 按 `block_key + medium + spec_names` 删除指定 specs，不存在的目标视为幂等成功。

### 4.3 SNAPSHOT

```json
{
  "instance_id": "model-a",
  "host_ip_port": "10.0.0.8:8080",
  "storage_type": "ST_EVENT_REPORT_L1P5",
  "events": [{
    "event_type": "EVENT_BLOCK_SNAPSHOT",
    "block_snapshot": {
      "blocks": [
        {
          "block_key": "123",
          "medium": "gpu",
          "specs": [{
            "name": "tp0",
            "uri": "rtp-llm://10.0.0.8:9600/hbm/123"
          }]
        },
        {
          "block_key": "456",
          "medium": "memory",
          "specs": [{
            "name": "tp0",
            "uri": "rtp-llm://10.0.0.8:9600/memory/456"
          }]
        }
      ]
    }
  }]
}
```

约束：

- 一个请求最多一个 snapshot；
- snapshot 不能与 ADD、DELETE、HOST_DOWN 混合；
- 每个 `block_key + medium` 的 specs 必须完整；
- 同一 medium 不允许重复 block key；
- 相同 block key 可以出现在不同 medium；
- snapshot 不能拆成多个普通请求分页。

### 4.4 响应

```proto
CommonResponseHeader header = 1;
repeated ErrorCode item_results = 2;
string committed_snapshot_version = 3;
uint64 retry_after_ms = 4;
bool snapshot_required = 5;
string extra_info = 6;
```

`committed_snapshot_version` 是当前进程内 reconciliation generation：

- 第一条 ADD/DELETE 可以创建；
- snapshot 成功后更新；
- KVCM 刚重启且还没有新 mutation 时为空。

`snapshot_required` 的响应语义：

- 当前进程尚无 generation 时为 true；
- 第一条通过校验并获得写 lease 的增量，或成功 snapshot，建立 generation 后为 false；
- 它不是增量准入条件；
- realtime-only 客户端可以忽略；
- snapshot-capable 客户端可把它当作“方便时补一次全量”的提示。

## 5. 增量处理

ADD/DELETE：

1. 校验请求、block、medium、spec name 和 URI；
2. 若当前进程尚无 reporter 节点则懒初始化；若命中 HOST_DOWN/grace tombstone 则拒绝；
3. 获取 delta lease；若 snapshot 正在执行，则最多等待
   `snapshot_delta_drain_timeout_ms` 后返回可重试的 `SNAPSHOT_IN_PROGRESS`；
4. 如果尚无 committed generation，在锁内生成一个不可复用的 opaque version；
5. ADD URI 追加该 `s_version`；
6. ADD 按 spec name 合并，DELETE 按 spec name 删除；
7. 请求完成后释放 lease。

`host_ip_port` 和 `medium` 会进入以 `#` 分隔的稳定 location id，因此必须非空且不能包含
`#`。这项校验在创建节点、解析 ReportEvent 和构造 location id 三层执行，避免写入一个后续
无法归属到 reporter 的 location。

Generation 在获得 delta lease 时建立。参数校验、tombstone 等准入失败不会创建 generation；
获得 lease 后如果 metadata 写入失败，generation 可以保留。这不会伪装事件成功：
`header.status`/`item_results` 仍返回真实写入结果，generation 只承担后续 URI 标记和对账。

同一请求中的增量按请求顺序解释。相同
`(block_key, medium, spec_name)` 多次出现时，最终操作覆盖前序操作。

当 KVCM 重启后第一条增量使用新 generation 时，同一 location 中未被本次增量触碰的旧 specs
必须保留，只覆盖本次明确上报的 spec name。否则一次局部 ADD 会误删同 block 的其他 cache
信息。

## 6. Snapshot 处理

1. 在关闭写门前完成全部校验；
2. 若当前进程尚无 reporter 节点则懒初始化；若命中 tombstone 则拒绝；
3. 检查 per-reporter 最小 snapshot 间隔；
4. `BeginSnapshot` 生成 candidate version，关闭新 delta 入口；
5. 最多等待 `snapshot_delta_drain_timeout_ms`（默认 10 秒）让已获得 lease 的 delta 完成；
   超时则 abort candidate、重新打开写门并返回可重试的 `SNAPSHOT_IN_PROGRESS`；
6. 为本次全部 URI 追加 candidate `s_version`；
7. 使用稳定 location id 原地替换完整 specs；
8. 更新内存 committed generation并打开写门；
9. 返回新的 `committed_snapshot_version`；
10. 异步扫描并回收该 reporter 的旧 generation。

当 meta storage 使用 `cached + persistent_type=async_redis` 时，第 7 步与增量使用相同的写入
语义：先把 Redis mutation 放入异步队列，再同步更新 local cache。Snapshot 不等待 Redis
consumer flush；成功响应表示本次全部 mutation 已成功 enqueue 且 local cache 已更新，不表示
Redis 已经执行完成。响应后发生的 pipeline 失败通过异步 Redis 日志和指标暴露，不回溯修改
已经返回的 ReportEvent 结果。

不同 reporter、不同 storage type 可以并行。第二个并发 snapshot 返回
`SNAPSHOT_IN_PROGRESS`。

`BeginSnapshot` 在持有节点状态锁时再次确认 reporter 运行时节点仍存在。这样即使请求层检查后并发
发生 HOST_DOWN/UNREGISTER，也不能重新创建一个脱离节点生命周期的 snapshot state。

`EventReportStorageSpec.snapshot_delta_drain_timeout_ms` 是 reporter 写门两个等待方向共用的
上限，默认 10 秒。`BeginSnapshot` 超时发生在任何 snapshot metadata 写入之前，因此只需清空
in-flight candidate、保留 committed generation 并唤醒等待中的 delta；调用方应退避并完整
重试 snapshot。新 delta 等待 in-flight snapshot 超时时不会取消 snapshot，也不会获得 delta
lease；调用方收到 `SNAPSHOT_IN_PROGRESS` 后应退避并幂等重试失败的 ADD/DELETE。

Snapshot 期间到达的 ADD/DELETE 会在配置的上限内等待。Snapshot 在上限内完成后，它们基于
新的 generation 继续写入，因此不会被刚完成的全量覆盖。

## 7. Snapshot 失败

Snapshot 使用稳定 location id 原地覆盖，不做 copy-on-write。

如果部分 replace 或 commit 失败：

- candidate 不成为 committed generation；
- snapshot abort，等待中的 delta 继续；
- 已经写入的部分 metadata 不回滚；
- 已写入的新候选和未覆盖的旧候选都可能被查询；
- reporter 查询暂时切回 soft；
- 不为失败 snapshot 单独启动 cleanup；
- 客户端可完整重试，后续成功 snapshot 恢复 strict 并立即收敛查询视图。

如果在 metadata 写入前等待 active delta 超时，则没有 snapshot 部分写入；candidate 被直接
abort，committed generation 和已有 metadata 都不变。

这是有意的缓存可用性取舍。

| 失败点 | committed generation | 查询表现 | 后续 |
| --- | --- | --- | --- |
| 参数校验 | 不变 | 原数据不变 | 修正请求 |
| 频率限制 | 不变 | 原数据不变 | 按 `retry_after_ms` 重试 |
| 部分 replace | 不变 | 新旧候选可能共存 | 完整重试 |
| commit | 不变 | 已写候选仍可能可见 | 完整重试 |
| cleanup 中断 | 已更新 | 完全属于旧 version 的 location 已被 strict 隐藏 | 下次 snapshot 再清理空间 |

## 8. 查询规则

业务查询 event-report location 时：

1. 按 instance 和 storage type 找到 `EventReportBackend`；
2. 从稳定 location id 解析 `medium + host_ip_port`；
3. reporter 必须已注册且 available；
4. location 至少包含一个 spec；
5. 每个 URI 必须合法；
6. URI 如果有 `s_version`，该值必须唯一且格式合法；
7. strict 模式下，location 至少包含一个 committed version。

soft 模式接受：

- 无 `s_version` 的历史 URI；
- 旧 generation；
- 当前 generation；
- snapshot in-flight generation；
- 同一 location 内混合的多个合法 generation。

strict 模式只接受至少含一个当前 committed spec 的 location。增量按 spec name merge，因此
mixed-generation location 中的 committed spec 会保护整个稳定 location；完全由 snapshot
candidate、旧 generation 或 legacy spec 组成的 location 在 candidate commit 前不可见。
若 snapshot 失败则立即回到 soft，这些格式合法的候选和历史 metadata 可重新成为 cache
candidate。

任一 spec malformed 时整个 location fail closed。

公共查询路径通过 `CacheManager` 的 location-aware checker 执行上述规则，它同时拥有
instance、storage type 和稳定 location id 上下文。`EventReportBackend::MightExist` 是底层
无这些上下文的保守接口，只能验证能够由当前 token 反查 owner 的 URI，不能用它替代公共查询
的 generation 兼容规则。

## 9. KVCM 重启

进程重启后：

1. 持久化的 Instance/InstanceGroup 和 cache metadata 恢复，进程内节点表和 committed
   generation 为空；
2. 收到 reporter 的第一条合法事件前，历史 metadata 因缺少 liveness 状态而不可见；
3. 第一条 HEARTBEAT、ADD、DELETE 或 SNAPSHOT 自动重建 reporter，不要求客户端再次 REGISTER；
4. 如果第一条是 HEARTBEAT，响应 `snapshot_required=true`、
   `committed_snapshot_version=""`，格式合法的历史 metadata 可以重新查询；
5. 如果第一条是 ADD/DELETE，它正常成功并创建新 generation；
6. 该增量只覆盖明确触碰的 spec，不清理其他历史 block；
7. 后续实时增量继续刷新热点数据；
8. 若客户端支持 snapshot，可在方便时发送一次完整全量，立即建立 strict 查询栅栏并最终
   清理旧 generation；
9. 若客户端只支持实时事件，也可一直运行，允许少量未触碰历史 metadata 保留。

第一条新汇报可能让 Redis 中尚未清理的旧 metadata 再次可见。因此新推理进程复用完全相同
的 reporter identity 时，建议旧进程发送 HOST_DOWN；同一 KVCM 进程内的 tombstone 会要求
新进程显式 REGISTER。若 KVCM 也已重启、tombstone 丢失，则依赖 cache miss 容错，并建议
新进程在具备能力时做一次 snapshot。

## 10. 并发

同一 reporter 共用写门：

```text
已有 delta 获得 lease
        |
snapshot 关闭新 delta 入口
        |
等待已有 delta 完成
        |
snapshot replace + commit/abort
        |
打开入口，等待中的 delta 继续
```

保证：

- snapshot 不越过已开始的 delta；
- snapshot 开始后到达的 delta 不会被本轮 snapshot 覆盖；
- commit 后等待 delta 使用新 generation；
- abort 后等待 delta 使用旧 generation；若此前没有旧 generation，则创建一个新 generation；
- Close、Unregister、HOST_DOWN 唤醒 waiter；等待 active delta 另有可配置超时，不会无界阻塞；
- metadata read-modify-write 在最终写阶段持有 lifecycle generation lease；旧请求不能在
  HOST_DOWN、重新 REGISTER 和新 snapshot 后恢复写入；
- mutation 在已经持有 metadata 锁时只做非阻塞的 per-reporter lifecycle lease 获取；若同一
  reporter 的 HOST_DOWN/REGISTER lifecycle writer 已经开始等待，则旧 mutation 立即失败，
  避免与 cleanup 的 `lifecycle -> metadata` 顺序形成锁序反转；不同 reporter 使用独立
  fence，不会因其他 host 的 HEARTBEAT/REGISTER 产生假失败；
- liveness unregister 的 generation 比较与节点删除在同一把锁内完成；
- 显式 HOST_DOWN 的 generation 捕获与节点删除同样在同一把锁内完成，Heartbeat/REGISTER
  只能在线性化的 HOST_DOWN 之前或之后生效，不能在中间恢复后又被旧请求删除；
- 旧节点 cleanup 在最终条件删除阶段持有 generation lease，只能删除扫描时看到的原值，
  不能越过重新 REGISTER 或删除新 lifecycle 刷新的稳定 location；
- 不同 reporter 不共享写门。

## 11. 限流与清理

`EventReportStorageSpec.snapshot_min_interval_ms` 提供 per-reporter 最小 snapshot 间隔，默认
30 秒。完整维度是：

```text
instance_id + storage_type + host_ip_port
```

只对成功 snapshot 开始计时；失败 snapshot 可立即重试；ADD、DELETE、REGISTER、HEARTBEAT
不受限流影响。

`EventReportStorageSpec.snapshot_delta_drain_timeout_ms` 提供 reporter 写门的统一等待上限，
默认 10 秒，同时约束 snapshot admission 等待已准入 delta 排空，以及新 delta 等待 in-flight
snapshot 完成。该配置在 backend 级统一设置，对其管理的每个 reporter 独立生效。snapshot
admission 超时会 abort 当前 candidate；delta 等待超时只拒绝该 delta，不会 abort snapshot，
两者都不影响其他 reporter。

成功 snapshot 后复用现有任务执行器扫描 instance metadata：

- 只处理目标 storage type 和 reporter；
- event-report URI 描述外部 cache，reclaimer 只删除 KVCM metadata，不调用外部 URI
  backend 的物理 DELETE；
- location 内只要包含任一当前 committed generation 就保留；
- 下一轮 snapshot 已开始时，location 内只要包含任一 in-flight generation 也保留；
- 完全由旧 generation 或无 version legacy spec 组成的 location 作为 stale；
- malformed version 的 location 作为 stale；
- 使用观察值条件删除，避免旧 cleanup 删除刚刷新的新值。
- cleanup 同时携带 snapshot attempt epoch：epoch 变化后在下一批扫描前退出；对已经扫描的
  location 仍逐条校验 epoch 和 URI generation，批次取消不能替代最终删除条件。

之所以按“任一 spec 匹配就保留”，是因为 delta 按 spec name merge，而 cleanup 只能按稳定
location 删除。Snapshot commit 后、cleanup 扫描前到达的 delta 可能只把一个 spec 刷到新
generation，其他 sibling 仍带旧 generation 或无 version 的 legacy URI；此时删除整个
location 会造成成功增量的 false negative。保留 mixed-generation location 允许少量 stale
sibling 暂存，后续完整 snapshot 会替换或回收。

成功 snapshot 进入 strict 后，cleanup 只负责最终空间回收，不是查询正确性的前提。扫描耗时记录为
`event_report.snapshot_cleanup_scan_latency_ms{instance_id,host,type}`。

容量估算：典型的 1 个 instance、10 个 reporter、每台 5000 个 block 场景中，一次单 reporter
完整 snapshot 约有 5000 次 metadata replace，并以 1000 key 为批次扫描该 instance 约 5 万
个 key；10 台同时全量约有 5 万次 replace、累计约 50 万次 key 检查。实际 Redis 命令数取决于
MetaIndexer backend 的 batching。若清理扫描延迟或 backlog 长期超过运行阈值，应引入
reporter -> location 反向索引，而不是继续提高全量频率。

## 12. 测试要求

### 12.1 单元测试

- fresh reporter 未显式 REGISTER 时，第一条 HEARTBEAT/ADD/DELETE/SNAPSHOT 可懒初始化；
- 第一条 ADD、第一条 DELETE 都成功并创建 version；
- HOST_DOWN/grace tombstone 后的迟到事件被拒绝，显式 REGISTER 后恢复；
- 第一条 delta 后不发送 snapshot，查询可见；
- 重注册后第一条 delta 创建新 version；
- 新 generation 的局部 ADD 保留旧 generation 的其他 specs；
- 健康 reporter 在 soft 模式接受合法 old/unversioned URI，成功 snapshot 进入 strict 后只接受
  至少含 committed spec 的 location；
- malformed version、空 location fail closed；
- unavailable/unregistered reporter 不可见，heartbeat 恢复后可见；
- 旧 lifecycle cleanup 与重新 REGISTER 后的 delta 竞争时，compare-and-delete 保留新值；
- snapshot 与 delta 两种先后顺序、commit 和 abort 均不死锁；
- snapshot 等待 active delta、delta 等待 in-flight snapshot 两个方向均有统一可配置超时，
  且超时后可重试；
- snapshot 部分失败后，已写候选仍可见且完整重试收敛；
- cleanup 与下一轮 snapshot 的 CAS 竞态不删除新值；
- snapshot commit 后 delta 刷新 mixed-generation location 时，旧 cleanup 不删除新写；
- snapshot cleanup 只删除 metadata，不调用外部 URI backend；
- 成功 snapshot 最终清理旧数据，失败 snapshot 不触发专用清理；
- snapshot 限流、storage type 隔离和 liveness 竞态。

### 12.2 集成测试

- 纯实时模式：不要求 REGISTER，直接 ADD/DELETE/GET，全程无 snapshot；
- 实时增量与低频 snapshot 混合；
- 一个 snapshot 同时包含多个 medium；
- snapshot 遗漏 block 后 reclaimer 最终删除；
- KVCM 重启后首条新汇报前旧数据不可见；
- 首条 HEARTBEAT 无需 REGISTER 即恢复旧 metadata；
- 重启后第一条 delta 成功且不删除未触碰历史 block；
- realtime-only reporter 不补 snapshot 也能持续 ADD/DELETE；
- heartbeat timeout、恢复、超过 grace cleanup；
- 多 reporter、多 instance、多 storage type 隔离；
- snapshot partial failure、完整重试；
- snapshot commit 后立即到达的 delta 在异步 cleanup 后仍可查询；
- reporter host 或 medium 含 `#` 时 fail closed 且无写入副作用；
- ASAN/TSAN 或等价并发检测（仅记录实际执行结果，不能由普通 CI 结果推断）。

Snapshot 与重启 HTTP 测试 target 带 `manual` 标签，不属于默认 GitHub CI；需要显式执行并单独记录结果。

### 12.3 Vineyard 跨仓集成镜像

Vineyard 的 ReportEvent 集成不能继续使用不含 `event_report` protobuf 的旧 KVCM 镜像；否则
`addStorage` 会返回 `missing or invalid fields: {StorageConfig: {storage_spec}}`，后续用例全部是
同一前置失败的连锁结果。

需要验证 KVCM 分支时，手动运行 `.github/workflows/build-dev-image.yml` 并选择
`flavor=integration`。该 flavor 只构建 CI 所需的 `linux/amd64` 生产镜像，并发布唯一的
`integration-<UTC time>-<short sha>` tag；把该精确 tag 写入 Vineyard 的
`.aoneci/v6d-pytest-integration.yaml`，禁止使用 `latest`。合并前至少确认 Vineyard 的
group-aware 三节点用例、green、KVCM fault 和 PACE fault 都实际执行，且从失败制品中的
pytest 汇总判断结果，不能根据 Aone 对自由脚本显示的 `NOT_RUN` 状态推断。

该 workflow 在 dev 容器内完成 Bazel 构建后，也必须在容器仍存活时把 server tar 复制到
Docker build context，并把文件 owner 改回 runner 用户。`bazel-bin` 可能指向容器内的
`/root/.cache/bazel`；容器退出后再从宿主 runner 读取该 symlink 会得到 `Permission denied`
或断链，不能据此误判为编译失败。

## 13. 接受的取舍

本方案优先 cache availability，不提供原子 snapshot 查询视图：

- snapshot 写入期间可能查到新旧 generation；
- failed snapshot 的部分写入可能被返回；
- 成功 snapshot 后，完全属于旧 generation 的遗漏 block 立即不可见；
- mixed-generation location 中若含 snapshot 后成功 delta 写入的当前 spec，整个 location
  仍保持可见；
- KVCM 重启后的第一条合法汇报可能让历史 metadata 重新可见；
- realtime-only reporter 永远不做 snapshot 时，未触碰的旧数据可能长期存在。

soft/失败恢复和 mixed-generation 场景仍可能产生 cache false positive。实际 cache 读取失败必须按
miss 处理。

本方案不提供 version 持久化、重启扫描恢复、历史 snapshot 查询、跨 reporter 原子快照、
snapshot 分页或 exactly-once event delivery。

## 14. 总结

```text
[可选 REGISTER] + realtime ADD/DELETE
        |
        +--> 不依赖首次 snapshot
        |
        +--> 可选 snapshot 做完整对账
                  |
                  +--> 成功：切换 generation + 异步清理
                  +--> 失败：保留已写候选 + 实时链路继续
```

Version 用于标记与回收，node liveness 控制硬可见性，snapshot 只负责可选的最终收敛。
