# KVMeta EMB Cache 设计

## 目标与边界

KVMeta 是为 EPD 场景提供的变长 embedding cache。每个 key 对应一个 opaque byte object，KVCM 只管理 key、实际字节数、存储位置和生命周期，数据仍由客户端直接读写 PACE。

V1 只支持 TairMempool/PACE（DRAM 或 SSD），不支持 NFS、Mooncake、3FS，也不修改 TairMempool provider。功能默认关闭；开启后与原 MetaService 共用 `kvcm.service.rpc_port`，通过不同的 protobuf service 全名路由。

这是 cache，不是 source of truth。调用方必须能在 miss、容量不足或数据面失败时重新计算 embedding。

## 组件

调用链如下：

```text
RTP-LLM
  -> KvMetaObjectClient
       -> KvMetaClient -> KVMeta gRPC -> KvMetaManager -> MetaIndexer
       -> KvMetaTransferClient -> PACE SDK
```

- `KvMetaManager`：实例注册、exact-key 元数据、写会话、实际用量和 LRU 回收。
- `KvMetaObjectClient`：组合元数据事务和 PACE 数据搬运。
- `KvMetaTransferClient`：按每批对象的真实长度配置自己私有的 PACE SDK client，并串行执行该 client 的 I/O；不改变普通定长 `TransferClient` 或 TairMempool provider。
- `KvMetaReclaimer`：只处理 KVMeta instance，按 group 的真实用量回收。

KVMeta 使用带完整 schema marker 的内部 instance。原 KVCache GC/Reclaimer 跳过这些 instance，防止两套回收逻辑重复处理；普通 KVCache 主链路保持不变。

## 数据模型

公共 `instance_id` 被编码成独立的内部 instance id。业务 key 映射为 64-bit 索引 key，完整业务 key 编码在 location id 中，因此哈希冲突仍可区分。

每个对象只有一个名为 `value` 的 location spec：

```text
key -> CacheLocation {
  status: WRITING | SERVING
  type: TAIR_MEMPOOL | TAIR_MEMPOOL_SSD
  uri: pace://<backend>/<offset>?node_id=...&media_type=...&range_id=...&size=<bytes>
}
```

URI 中的 `size` 是对象实际字节数。服务端和客户端都会校验 backend 名、PACE 地址字段、media type 和 size，防止把错误位置交给数据面。

## 写入流程

一次写入由 `PutStart` 和 `PutFinish` 组成：

1. `PutStart` 查询所有 key。
2. 已存在且 size 相同的 `SERVING` 对象返回 hit；size 不同返回 `SIZE_MISMATCH`；`WRITING` 返回 `WRITE_IN_PROGRESS`。
3. 对 miss，选择当前可写的 PACE backend，逐对象按真实 size 分配空间。
4. 每次分配成功后立即以条件写创建对应的 `WRITING` metadata；全部完成后在内存中保存有期限的 write session。启动阶段回滚失败也进入同一个 cleanup worker 重试。
5. 客户端把 bytes 写入 PACE。
6. 全部成功后 `PutFinish` 条件地把本 session 的 generation 改为 `SERVING`；任一失败则删除本 session 的全部 metadata，并尽力释放 PACE allocation。metadata 清理遇到暂时错误时会保留内部清理任务重试。

`WRITING` 对 `Get` 不可见。session 到期会走与失败相同的清理。多 key session 是整批成功/失败语义，但不承诺多个 key 在同一时刻线性化可见。

## 容量统计

容量按已经提交的 `SERVING` 对象实际字节统计，计数器复用每个 `MetaIndexer` 的 storage usage：

- `PutStart` 不增加 usage，也不预留容量。
- `PutFinish` 成功把对象从 `WRITING` 改为 `SERVING` 后，增加该对象的真实 size。
- `Remove` 或 GC 条件删除成功后，减少同样的 size。
- 物理删除失败不把 metadata 和 usage 加回；cache metadata 已不可见，但会留下需由 PACE 运维观测的孤儿 allocation。
- Leader 恢复时扫描 KVMeta metadata，删除遗留 `WRITING`，并重新汇总所有 `SERVING` size，覆盖内存计数器。

更新 `SERVING` 状态和 usage 时只使用按 instance 分片的 metadata mutation lock，保证 metadata 与计数器不会交叉更新。`PutStart` 不进入该锁。

## 容量准入

`PutStart` 复用现有 storage selector 做一次快照检查：当前已提交 usage 尚未达到 group/type capacity 才允许继续。它不把本次请求 size 加入判断，也没有容量 reservation 或全局锁。

因此并发写允许短暂超调。例如 capacity 为 100 GiB，当前 usage 为 99 GiB 时，一个 4 GiB 对象仍可分配并提交到 103 GiB；后续分配可能由 PACE 拒绝，或由异步 GC 回收到目标水位。

`used_percentage=0.8` 是 GC 的回收目标，不是 `PutStart` 的硬拒绝线。PACE allocator 是物理分配能否成功的最终判断者，KVCM 不接管它的失败或 fallback 策略。

## GC

KVMeta Reclaimer 周期处理已注册的 KVMeta group：

1. 列出 group 内 KVMeta instance，并汇总各自 `MetaIndexer::GetStorageUsage()`。
2. 计算 `target = floor(group_capacity * used_percentage)`。
3. usage 不高于 target 时不处理。
4. 从每个 indexer 的 persistent keyspace 采样 candidate，并用命中的 hot-cache access time 覆盖持久层时间；合并后按 `last_access_time` 全局排序。这样 metadata hot cache 淘汰 key 后 GC 仍能继续推进。
5. 选择至多一个 batching size，或预计删除字节达到 `usage - target` 为止。
6. 按 generation 条件删除 metadata；只有删除成功的 `SERVING` 对象才扣减 usage。
7. metadata 删除后等待现有 `delay_before_delete_ms` grace period，再尽力释放对应 PACE allocation；显式 `Remove` 使用相同的 reader grace period。
8. 后续轮次继续采样，直到 usage 不高于 target。

generation 由 allocation 创建时间和完整 PACE URI 共同标识，`WRITING -> SERVING` 不改变它。条件删除因此不会误删已经替换的新 allocation。回收是异步、渐进的；采样不足或 provider 删除失败不会阻塞读写主链路。

## 恢复与 HA

升主后，原 KVCache 先按既有流程恢复并放流。随后启动 KVMeta 自己的 maintenance worker；worker 在首轮 GC 前从 persistent metadata source 扫描全部对象并重建 usage，不使用双层 metadata 模式下有界的 hot-cache 视图。恢复期间 KVMeta 请求返回 `SERVICE_NOT_READY`，不阻塞原 MetaService，也不在 Server 中增加第二套恢复线程生命周期。

降主或停服时先关闭 KVMeta 新请求，再停止 session expiry 和 maintenance worker。

## 配置约束

- `kvcm.kv_meta.enabled=true` 才注册服务。
- KVMeta group 应独立配置，不与普通 KVCache instance 混用。
- group 必须配置 LRU reclaim policy 和 `[0, 1)` 范围的 `used_percentage`，为异步回收保留空间。
- `storage_candidates` 非空且全部是同一 PACE storage type；非 fallback 的 `CPS_ALWAYS_*` 必须匹配该 DRAM 或 SSD tier。V1 的一个 group 只管理一个 tier，GC target 取 group capacity 与该 tier quota 的较小值。
- RTP wrapper 使用现有环境变量，并把普通 `RECO_INSTANCE_GROUP` / instance id 加 `kve_` 前缀形成 KVMeta identity；该前缀规则属于 RTP client，不是服务端协议要求。

## 失败语义

- `Get` miss：返回未命中，由调用方重算。
- PACE allocation 或 I/O 失败：本次 cache write 失败，不影响推理结果。
- 数据面 SDK 超时：client 会先等待已提交给 PACE 的任务退出，再返回超时并关闭该 data-plane client，防止旧 I/O 在 allocation 释放后继续访问。
- `PutFinish` transport 失败：提交结果可能未知，不能盲目重试；先 `Get` 对账。
- `Remove` 是幂等 metadata 删除，但遇到同 key 的 `WRITING` 会返回 `WRITE_IN_PROGRESS`；transport 失败后也不能自动重试删除，因为期间可能出现新 generation。
- 同一 object client 的 data-plane 操作在 metadata 调用前串行，`save` 的排队时间不会消耗服务端 write lease；不同 client 仍由服务端 generation 条件写协调。
- `Trim` 仅保留旧 proto 定义，不注册 V1 handler；按 key 删除使用 `Remove`，容量治理使用自动 GC。

## 非目标

V1 不提供强一致容量 reservation、跨 key 原子可见性、持久化 session、独立端口、多后端 object storage 或业务内容校验。以上能力不应通过修改普通 KVCache、Admin 或 TairMempool provider 主链路来隐式实现。
