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

### 2.3 s_version 只用于标记与清理

KVCM 在 reporter 提供的 URI 上只追加：

```text
s_version=<opaque-version>
```

Version 的用途：

- 标记本进程内一段增量或一次 snapshot generation；
- snapshot 成功后帮助 reclaimer 找出旧 generation；
- 通过响应、日志和监控对齐一次对账。

Version 不再是查询可见性的硬屏障。查询不会要求：

```text
uri.s_version == committed_snapshot_version
```

### 2.4 节点生命周期仍是硬门槛

旧 version、新 version、无 version 的合法 metadata 都可以作为 cache 候选，但 reporter 必须：

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
string committed_snapshot_version = 3;
uint64 retry_after_ms = 4;
bool snapshot_required = 5;
string extra_info = 6;
```

`committed_snapshot_version` 是当前进程内 reconciliation generation：

- 第一条 ADD/DELETE 可以创建；
- snapshot 成功后更新；
- KVCM 刚重启且还没有新 mutation 时为空。

`snapshot_required` 兼容现有响应字段：

- 当前进程尚无 generation 时为 true；
- 第一条通过校验并获得写 lease 的增量，或成功 snapshot，建立 generation 后为 false；
- 它不是增量准入条件；
- realtime-only 客户端可以忽略；
- snapshot-capable 客户端可把它当作“方便时补一次全量”的提示。

## 5. 增量处理

ADD/DELETE：

1. 校验请求、block、medium、spec name 和 URI；
2. 若当前进程尚无 reporter 节点则懒初始化；若命中 HOST_DOWN/grace tombstone 则拒绝；
3. 获取 delta lease；若 snapshot 正在执行则等待其 commit/abort；
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
5. 等待已获得 lease 的 delta 完成；
6. 为本次全部 URI 追加 candidate `s_version`；
7. 使用稳定 location id 原地替换完整 specs；
8. 对本次 block keys 执行 `Sync`；
9. 更新内存 committed generation并打开写门；
10. 返回新的 `committed_snapshot_version`；
11. 异步扫描并回收该 reporter 的旧 generation。

不同 reporter、不同 storage type 可以并行。第二个并发 snapshot 返回
`SNAPSHOT_IN_PROGRESS`。

`BeginSnapshot` 在持有节点状态锁时再次确认 reporter 运行时节点仍存在。这样即使请求层检查后并发
发生 HOST_DOWN/UNREGISTER，也不能重新创建一个脱离节点生命周期的 snapshot state。

Snapshot 期间到达的 ADD/DELETE 会等待。Snapshot 完成后，它们基于新的 generation 继续
写入，因此不会被刚完成的全量覆盖。

## 7. Snapshot 失败

Snapshot 使用稳定 location id 原地覆盖，不做 copy-on-write。

如果部分 replace、Sync 或 commit 失败：

- candidate 不成为 committed generation；
- snapshot abort，等待中的 delta 继续；
- 已经写入的部分 metadata 不回滚；
- 已写入的新候选和未覆盖的旧候选都可能被查询；
- 不为失败 snapshot 单独启动 cleanup；
- 客户端可完整重试，后续成功 snapshot 最终收敛。

这是有意的缓存可用性取舍。

| 失败点 | committed generation | 查询表现 | 后续 |
| --- | --- | --- | --- |
| 参数校验 | 不变 | 原数据不变 | 修正请求 |
| 频率限制 | 不变 | 原数据不变 | 按 `retry_after_ms` 重试 |
| 部分 replace | 不变 | 新旧候选可能共存 | 完整重试 |
| Sync/commit | 不变 | 已写候选仍可能可见 | 完整重试 |
| cleanup 中断 | 已更新 | 旧候选继续可见 | 下次 snapshot 再清理 |

## 8. 查询规则

业务查询 event-report location 时：

1. 按 instance 和 storage type 找到 `EventReportBackend`；
2. 从稳定 location id 解析 `medium + host_ip_port`；
3. reporter 必须已注册且 available；
4. location 至少包含一个 spec；
5. 每个 URI 必须合法；
6. URI 如果有 `s_version`，该值必须唯一且格式合法。

查询接受：

- 无 `s_version` 的历史 URI；
- 旧 generation；
- 当前 generation；
- snapshot in-flight generation；
- 同一 location 内混合的多个合法 generation。

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
8. 若客户端支持 snapshot，可在方便时发送一次完整全量，最终清理旧 generation；
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
snapshot replace + Sync + commit/abort
        |
打开入口，等待中的 delta 继续
```

保证：

- snapshot 不越过已开始的 delta；
- snapshot 开始后到达的 delta 不会被本轮 snapshot 覆盖；
- commit 后等待 delta 使用新 generation；
- abort 后等待 delta 使用旧 generation；若此前没有旧 generation，则创建一个新 generation；
- Close、Unregister、HOST_DOWN 唤醒 waiter，不死锁；
- 旧节点 cleanup 只能删除扫描时看到的原值，不能删除重新 REGISTER 后刷新的稳定 location；
- 不同 reporter 不共享写门。

## 11. 限流与清理

`EventReportStorageSpec.snapshot_min_interval_ms` 提供 per-reporter 最小 snapshot 间隔，默认
30 秒。完整维度是：

```text
instance_id + storage_type + host_ip_port
```

只对成功 snapshot 开始计时；失败 snapshot 可立即重试；ADD、DELETE、REGISTER、HEARTBEAT
不受限流影响。

成功 snapshot 后复用现有任务执行器扫描 instance metadata：

- 只处理目标 storage type 和 reporter；
- event-report URI 描述外部 cache，reclaimer 只删除 KVCM metadata，不调用外部 URI
  backend 的物理 DELETE；
- location 内只要包含任一当前 committed generation 就保留；
- 下一轮 snapshot 已开始时，location 内只要包含任一 in-flight generation 也保留；
- 完全由旧 generation 或无 version legacy spec 组成的 location 作为 stale；
- malformed version 的 location 作为 stale；
- 使用观察值条件删除，避免旧 cleanup 删除刚刷新的新值。

之所以按“任一 spec 匹配就保留”，是因为 delta 按 spec name merge，而 cleanup 只能按稳定
location 删除。Snapshot commit 后、cleanup 扫描前到达的 delta 可能只把一个 spec 刷到新
generation，其他 sibling 仍带旧 generation 或无 version 的 legacy URI；此时删除整个
location 会造成成功增量的 false negative。保留 mixed-generation location 允许少量 stale
sibling 暂存，后续完整 snapshot 会替换或回收。

Cleanup 只负责最终空间回收，不是查询正确性的前提。扫描耗时记录为
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
- 健康 reporter 的 old/current/in-flight/unversioned URI 可见；
- malformed version、空 location fail closed；
- unavailable/unregistered reporter 不可见，heartbeat 恢复后可见；
- 旧 lifecycle cleanup 与重新 REGISTER 后的 delta 竞争时，compare-and-delete 保留新值；
- snapshot 与 delta 两种先后顺序、commit 和 abort 均不死锁；
- snapshot 部分失败或 Sync 失败后，已写候选仍可见且完整重试收敛；
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
- snapshot partial/Sync failure、完整重试；
- snapshot commit 后立即到达的 delta 在异步 cleanup 后仍可查询；
- reporter host 或 medium 含 `#` 时 fail closed 且无写入副作用；
- ASAN/TSAN 或等价并发检测。

## 13. 接受的取舍

本方案优先 cache availability，不提供原子 snapshot 查询视图：

- snapshot 写入期间可能查到新旧 generation；
- failed snapshot 的部分写入可能被返回；
- cleanup 完成前，snapshot 遗漏的旧 block 可能短暂返回；
- KVCM 重启后的第一条合法汇报可能让历史 metadata 重新可见；
- realtime-only reporter 永远不做 snapshot 时，未触碰的旧数据可能长期存在。

这些都是 cache false positive。实际 cache 读取失败必须按 miss 处理。

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
