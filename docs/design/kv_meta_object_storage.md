# KVMeta 通用对象存储设计

## 目标和边界

EPD 分离场景中，KV cache 继续使用现有 `MetaService + CacheManager` 定长 block 链路；embedding 等
opaque value 使用新增的 KVMeta exact-key 链路。服务端管理元数据、allocation URI 和写事务；client 侧新增
独立的 exact-size 数据面，并由 `KvMetaObjectClient` 将两者组合成完整对象读写。该数据面不放宽现有
`TransferClient` 的定长校验。

这条链路遵守以下隔离条件：

- `kvcm.kv_meta.rpc_port=0` 时完全禁用，不创建 manager、会话线程或 gRPC server；
- 启用后使用独立端口和 gRPC server，不向现有 MetaService 增加 RPC；
- `KvMetaManager` 不调用 `StartWriteCache` / `FinishWriteCache`，也不修改 `DataStorageBackend::Create` 的
  定长 batch 接口；
- KVMeta instance 使用保留的内部 namespace 和完整 schema marker，恢复逻辑不会仅凭字符串前缀处理实例；
- 普通 CacheReclaimer、Migration 和 Cache GC 跳过 KVMeta instance，embedding 数量不会消耗 KVCache
  后台扫描预算；
- 主服务完成原有升主恢复并放流后，KVMeta 才在独立可取消线程中恢复。KVMeta 恢复失败不会阻止主服务可用。
- 降主会先关闭 KVMeta 请求门并取消全量 Trim；Trim 每 1000 个 key 检查取消、每次最多删除 256 个对象，
  因而不会因 namespace 总量无限延迟主服务清理。现有 KVCache 请求和后台任务不获取这些 KVMeta 锁。

KVMeta 必须使用专用 Instance Group。服务端会拒绝把 KVMeta instance 注册到已经含普通 instance 的 group。
如果之后误用普通 RegisterInstance 向该 group 加入 KVCache instance，后续 KVMeta allocation 会 fail closed，
避免继续扩大混用；部署侧仍应禁止这种配置。

## 不同 value size 的处理

现有 storage `Create(keys, object_size)` 的一次调用只接受一个 size。KVMeta 不修改这个公共接口，而是对每个
实际缺失的 key 发起一次 singleton `Create`：

```text
PutStart keys       key-a       key-b       key-c
value_sizes          1536        4096         768
                         \          |          /
singleton Create   Create(1536) Create(4096) Create(768)
```

这样既支持同一 batch 内不同长度，也避免 NFS/HF3FS/Dummy 等可将 batch 打包进同一物理文件的 backend
在按 key 删除时误删相邻对象。allocation URI 的 `size` 必须与请求值完全一致，否则服务端释放 allocation 并
返回 `SIZE_MISMATCH`。响应还显式携带 `ValueLocation.value_size`；C++ client 会再次逐项对照请求长度，发现
不一致时自动以失败 mask 回滚会话，不向业务暴露错误 location。

同一个 key 已存在（包含 active 写入）时，也必须与本次 `value_sizes` 完全一致才返回 `key_mask=true`；尺寸
不同直接返回 `SIZE_MISMATCH`，且发生在新 allocation 之前。跨进程并发 PutStart 的条件插入失败后，服务端会
重新读取赢家 metadata，再次校验 exact key、状态、storage URI/backend 和尺寸，不能把不同尺寸的对象当作命中。

当前默认限制为每请求 64 个 key、单 value 1 GiB、单 batch 4 GiB、写会话最长 1800 秒、每个 KVCM
进程最多 4096 个 active KVMeta 写会话；0 byte value 不允许写入。会话上限会在 allocation 前预检，并在
最终登记时再次原子校验，避免异常流量形成无界进程内状态。

## exact key 与碰撞处理

MetaIndexer 的一级 key 是 `int64`，KVMeta 将业务 string key 哈希成一级 key，同时把完整业务 key 编码进稳定
location id。读取、更新和删除始终同时指定一级 key 与完整 location id，因此不同 string 即使哈希碰撞也不会
互相命中。一个请求中哈希相同的项会拆成不同 metadata batch，满足底层 batch 对一级 key 唯一性的要求。
恢复和 Trim 还会反解 location id，并核对原 key 的完整编码和哈希；只有精确归属于该一级 key 的 location 才会
被统计或删除。URI scheme、hostname 对应的已注册 backend type 和 URI `size` 也必须一致，异常 metadata
一律 fail closed。

业务 instance id 同样会编码进保留的内部 instance id；对外响应只返回原始 id。

## 写入状态机和失败语义

```text
PutStart
  -> singleton allocation
  -> exact conditional metadata insert（active，不可读）
  -> metadata Sync
  -> 返回 session + locations

调用方写数据
  -> PutFinish(all true)
  -> exact conditional commit + Sync
  -> value 可读

任一 false / session 超时
  -> exact conditional metadata delete
  -> 删除本代 allocation

降主
  -> 关闭 KVMeta 请求门，非阻塞请求停止 session expiry worker
  -> 先按原顺序排空主服务请求、停止主 GC/Migration
  -> 仅清空进程内 session，不在主服务 cleanup 上逐个执行存储 IO
  -> 下一任 leader 的独立 KVMeta recovery 清理残留 active metadata/allocation
```

一个 Put 会话采用整批失败语义：`success_keys` 与 `PutStartResponse.locations` 一一对应，只要包含一个 `false`
就回滚全部新对象。空 mask、长度不匹配或过期 session 都不会被当作成功；长度不匹配也不会消费有效 session，
调用方可以用正确 mask 重试。V1 的 commit/rollback 使用逐 key 条件更新与失败补偿，不承诺多 key 在并发 `Get`
观察下具备线性化的同一瞬间可见性。

`Remove` 不会抢占 active write session。批次中任一 key 仍在写时返回 `WRITE_IN_PROGRESS`，整批不产生删除
副作用；该 session 只能由对应的 `PutFinish`、session timeout 或 leader 恢复流程收敛。这样独立的修复/删除
请求不会使 writer 仍持有的 URI 提前失效。

通用对象在 metadata 中保持 `CLS_NEW`，避免进入按定长 block 设计的 reclaimer/migration。正
`create_time` 是 active marker，负值是 KVMeta 私有 committed marker。Get 只返回 committed 对象。启动恢复
只删除 schema marker 正确且仍为 active 的对象，并使用完整序列化值做条件删除；无法确认归属时宁可保留数据，
不会冒险删除可能已被新一代 metadata 引用的 URI。

降主先用非阻塞信号关闭新 session 准入、取消 Trim 并唤醒 expiry worker，然后按原有顺序排空主服务请求并停止
主 GC/Migration；只有在这些主链路步骤完成后才 join KVMeta worker。降主不逐个同步回滚任意数量的 session，
避免 embedding 规模反向阻塞主 KVCM 生命周期。active 对象始终不可读，其 metadata 和 allocation 会在新
leader 放开 KVMeta 服务之前完成清理，因此这一延迟不改变可见性语义。

metadata reservation、commit 和 delete 均经过 `Sync` 持久化屏障；删除只有在 metadata delete 已持久化后才
释放物理 allocation。恢复按有界 batch 清理 stale active 写入，并在一个无删除、无错误的稳定扫描完成后，按
每个 committed URI 的真实 `size` 重建 KVMeta 专用 instance 的动态 byte usage。该恢复不改普通 instance 的
计量值。

## API 与 RTP 接入

协议位于 `kv_meta_service.proto`，核心流程为：

1. 在专用 Instance Group 上调用 `RegisterInstance`；
2. `Get(keys)` 返回严格 request-aligned 的 `hit_mask` 和 `locations`；
3. miss 调用 `PutStart(keys, value_sizes)`，只写 `key_mask=false` 对应的紧凑 `locations`；
4. 每项按自己的 `value_size` 完成数据面写入；
5. 无论成功或失败都调用 `PutFinish`。成功 mask 按紧凑 `locations` 对齐，而不是按原始 keys 对齐。

RTP C++ 侧可链接包含本功能的 RPM 中的 `kv_cache_manager_client.so` 并包含
`kv_meta_object_client.h`。`KvMetaObjectClient` 在创建时调用 `RegisterInstance` 获取服务端权威 storage
配置，并组合 `KvMetaClient` 与 `KvMetaTransferClient`；`SaveObjects` 完成 PutStart、仅写 miss、校验实际
URI 和 PutFinish，任一步失败都会以失败 mask 回滚已开始的 session。`LoadObjects` 在任何数据 IO 前验证全部
key 命中、value size、URI 和 buffer。每个变长对象以 singleton SDK 调用搬运，但同一批次仍共享 wrapper
级总超时预算；deadline 后排队任务不会再发起 IO，已经运行的任务则在返回前安全 drain，避免调用方释放
buffer 后 backend 继续访问。这样不会触发固定 block backend 对同批对象等长、同 allocation 的历史假设。

底层 `KvMetaClient` 支持
多 KVMeta 地址：读请求和同配置注册遇到 transport 错误会尝试下一地址；所有请求收到明确的 not-leader 或
not-ready 响应时都会故障转移并记住成功 endpoint。所有数据变更 RPC 的 transport 错误都无法证明服务端
是否已经执行，因此客户端不会自动重试；特别是重放 `Remove`/`Trim` 可能误删第一次调用后创建的新一代对象。
调用方应把结果视为不确定，并按具体 mutation 查询或审计；key 级结果可通过 `Get` 确认。未提交的 active
allocation 由 session timeout 或下一任 leader 的恢复流程清理。`PutStart` 后 `Get` 仍 miss 可能表示 session
尚处于 active 状态，调用方要等原 write timeout 过去再发起新写；C++ 接口以
`ER_INVALID_GRPCSTATUS` 返回这类 transport 结果。

`KvMetaClient` 本身仍只负责元数据和 allocation；需要自行编排事务的调用方也可直接组合它与
`KvMetaTransferClient`。现有 `TransferClient` 继续按普通 instance 的固定 `location_spec_infos` 校验 buffer
size；KVMeta 的 marker spec 固定为 1，不能用它搬运任意长度 embedding。变长策略只由
`KvMetaTransferClient::Create` 的显式初始化开启，并受 `max_object_bytes` 上限约束。C++ 和 Python pybind
均导出推荐的 `KvMetaObjectClient`。

Python 调用方可依赖 Bazel target
`//kv_cache_manager/protocol/protobuf:kv_meta_service_py_proto` 使用生成的 gRPC stub。

## V1 运维限制

KVMeta V1 没有自动 LRU/TTL 逐出，这是有意的主链路隔离边界。专用 group 达到 byte capacity 或进程级
active session 上限后，
`PutStart` 会失败；业务或运维需要按 key 调用 `Remove`，或通过 `Trim(TS_REMOVE_ALL_CACHE)` 清空 metadata 和
数据。`TS_REMOVE_ALL_META` 只删 metadata、保留物理数据，仅用于明确的数据所有权/修复场景；时间范围 Trim
尚不支持。`Remove`/`TS_REMOVE_ALL_CACHE` 会调用所选 storage backend 的 Delete，但最终物理回收能力遵循
该 backend 的现有实现；例如当前开源 NFS backend 的 Delete 是幂等 no-op，本次不会为了 KVMeta 改动这一
主链路共用行为。metadata 删除会先持久化，再释放物理 allocation，以保证任何时刻都不会把仍可读的 URI
提前删除。若后端随后返回物理删除失败，接口会报错，但该 URI 已不再被 metadata 引用；随机 generation URI
保证后续同 key 写入不会复用它，运维仍需依赖后端的 orphan 清理能力回收这类空间。
