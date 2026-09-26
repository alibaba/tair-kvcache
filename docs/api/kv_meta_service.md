# KVMeta 通用对象 API

## 1. API 定位

KVMeta 是面向 embedding 等变长 opaque value 的 Cache 元数据与写事务 API。协议定义在
`kv_cache_manager/protocol/protobuf/kv_meta_service.proto`，完整 gRPC service 名为
`kv_cache_manager.proto.kv_meta.MetaService`。

KVMeta 只管理对象 key、真实字节数、storage location 和生命周期，不传输对象 bytes，也不保存 tensor shape、
dtype、内容 checksum 等业务信息。它不是 source of truth；推荐调用方使用 `KvMetaObjectClient` 组合本文 RPC 和
exact-size 数据面，并为所有 miss/容量/读取故障保留重算路径。

总体架构、隔离和 HA 设计见 [KVMeta EMB Cache 系统设计](../design/kv_meta_object_storage.md)。
本文描述当前已实现的 V1 wire contract；设计文档第 13 节的 capability、generation、cleanup ledger 和 object-set
均为后续演进方案，现有 client/server 不得假设这些字段或 RPC 已存在。

## 2. 调用前提

1. 服务端必须配置 `kvcm.kv_meta.enabled=true`；KVMeta 与既有 MetaService 共用
   `kvcm.service.rpc_port`，按不同的 protobuf service 全名路由；
2. Instance Group 必须只用于 KVMeta，不能混入普通 KV cache instance；group 必须配置 `POLICY_LRU`、
   `used_percentage in [0,1]`、合法的 `delay_before_delete_ms`，且进程级 reclaim sampling/batching 均非零；
   metadata backend 必须是会在读取时刷新热度的 `local` 或 `cached`（测试环境可用 `dummy`）；`cached` 必须提供
   非空、合法的 URI，且 hot `cache_type` 为默认/显式 `local`；直连
   `redis`/`async_redis` 当前不会刷新 `BP#lru_time`，因此 KVMeta 注册会 fail closed，普通 KVCache 不受影响。
   其中 `local` 是纯进程内状态，不具备 crash/failover 恢复能力；共享 TairMempool/PACE 的生产部署必须使用
   `cached`，其 persistent 层使用 Redis/async Redis，并同时使用持久化 Registry；
   `storage_candidates` 还必须唯一、已注册并具有 exact-object ownership（EventReport 不满足）；文件型 backend
   配置必须能生成词法规范的绝对 KVMeta namespace（NFS 拼接型 `root_path` 必须带目录分隔符），storage spec 的
   动态类型必须与 backend type 一致，最长 hash/size/blkid 组合后的 URI 也必须处于配置上限内；否则注册或新对象
   allocation 返回 `SERVICE_NOT_READY`；
3. 调用方先执行 `RegisterInstance`，并使用响应中的权威 `storage_configs` 初始化数据面；
4. 每次 RPC 都通过 `CommonResponseHeader.status` 判断业务结果，不能只看 gRPC transport status；
5. 公共 `instance_id` 只能通过本 service 使用。编码后的 `__kv_meta_v1__...` 前缀属于服务端保留 namespace；旧
   Meta/Admin API 会返回 `INVALID_ARGUMENT`，Admin 列表也不会暴露这些内部实例；
6. 同一 `(instance_id, key)` 必须永久表示相同内容。key 至少纳入 tenant、模型/预处理 revision、输入 digest、
   tensor schema/version；服务端把“同 key、同 size”视为 hit，但不会比较 value bytes；
7. miss、`RESOURCE_EXHAUSTED`、`WRITE_IN_PROGRESS`、服务不可用或数据面 Load 失败都必须允许调用方重算。Cache 写回失败
   不应使本次推理失败。

使用内部 TairMempool/PACE 数据面时，variable-size 路径复用现有 PACE 配置和端点。服务端先分配
exact-size GA，client 只对该地址执行 I/O；不允许 client 自行补分配，也不走 fixed-block gather/scatter。
PACE SDK 必须在 Get/Put 返回前通过 `pace_synchronize` 排空已提交 I/O，否则不满足当前 V1 的地址释放契约。

### 2.1 与原 `kv_meta_service.proto` 的关系

实现沿用仓库原有的 `kv_meta_service.proto`、package、service 名和 RPC 名，没有再建立第二套协议。变长对象能力以
protobuf 向后兼容的方式占用原消息中的新 field number：`ValueLocation.value_size=4`、
`GetResponse.hit_mask=3`、`PutStartRequest.value_sizes=5` 和 `PutFinishRequest.success_keys=5`，并补齐明确错误码；
既有字段编号没有改写。

这里的“wire format 可演进”不等于旧 client/server 可以混用新语义：缺少 `value_sizes` 或 `success_keys` 的旧请求
会被新版服务端拒绝；缺少 `hit_mask` 或 `value_size` 的旧响应也会被新版官方 client fail closed。部署 KVMeta V1
时，client library、生成的 proto stub 和 server 必须使用包含上述字段的匹配版本。既有 KV cache MetaService
协议不受影响。

## 3. 总体调用流程

### 3.1 写入

```text
RegisterInstance（每个 client 初始化时，幂等）
        |
        v
PutStart(keys, value_sizes, write_timeout_seconds)
        |
        +-- key_mask=true  -> 同 key、同尺寸 committed 对象，不写数据（不校验内容）
        |
        +-- key_mask=false -> 按 compact locations 写入 exact bytes
                                  |
                                  v
                    PutFinish(session, success_keys)
```

`PutStart` 成功并返回非空 session 后，无论数据写成功还是失败，调用方都必须执行 `PutFinish`。客户端崩溃或
transport 结果不确定时，服务端 session timeout/leader recovery 负责清理不可读的 active 对象。

### 3.2 读取和删除

```text
Get(keys) -> 校验全部 hit/location/size -> 数据面 Load
Remove(keys) -> durable tombstone -> backend-specific Delete policy -> metadata finalization + Sync
Trim(instance) -> 按策略清理整个 KVMeta instance
```

## 4. 通用契约

### 4.1 字符串与批次边界

| 项目 | 约束 |
|---|---:|
| 每次 keys 数量 | 1..64，且不能重复 |
| key | 1..512 bytes |
| instance id / group / session id | 1..512 bytes |
| `user_data` | 0..64 KiB |
| 单 location URI | 不超过 64 KiB，query 参数不超过 64 个 |
| 单 value | 1 byte..1 GiB |
| 单批 value 总量 | 不超过 4 GiB |
| `write_timeout_seconds` | 1..1800 |

服务端和官方 client 都执行边界校验。超限请求在 allocation 或数据 I/O 前失败。

### 4.2 响应头

所有响应包含 `CommonResponseHeader`：

- `status.code`：KVMeta 业务错误码；
- `status.message`：简要结果和服务端错误上下文；
- `request_id`：服务端请求标识；
- `tracer_result`：仅在请求开启 span tracer 时返回。

非 `OK` 响应中的业务 payload 不应被使用。

## 5. RPC 契约

### 5.1 `RegisterInstance`

输入 `instance_group`、`instance_id` 和可选 `user_data`：

- group 必须已存在并且只包含 KVMeta instance；
- group 类型由已持久化的成员派生；KVMeta 与普通 KVCache 的注册在同一控制面临界区内双向互斥，任一方先注册后，
  另一类 instance 再加入同 group 都会在 registry mutation 前被拒绝；
- group 必须有当前 KVMeta Reclaimer 可执行的 LRU 配置，且进程级 sampling/batching 非零；无配置、非 LRU、
  非法 watermark/read grace 或关闭采样/批量回收均返回 `SERVICE_NOT_READY`，且不会创建 instance；
- group 的 metadata backend 必须是 `local`、使用 local hot layer 的合法 `cached`，或测试用 `dummy`；直连 Redis 不刷新读热度，不能在
  `POLICY_LRU` 名义下静默退化为采样/并列顺序淘汰。这个校验只证明回收算法有可信的读热度，不证明 metadata
  能跨进程恢复：`local`/`dummy` 的 `Sync` 只是进程内 barrier。生产共享 Cache 使用以 Redis/async Redis 为
  persistent 层的 `cached`，以及持久化 Registry；
- `storage_candidates` 必须唯一并全部指向已注册的 exact-object backend；配置身份必须与 registry name 一致，文件型
  配置必须能安全生成 `kvmeta/<instance-hash>/<key-hash>/<32-byte nonce>`；EventReport 只表示外部 block 观测，
  不授予 KVCM 创建/删除所有权，因此不能作为 EMB value storage；
- 相同 instance/group、KVMeta schema 和 `user_data` 的重复注册幂等；
- group、schema 或既有 instance 配置不一致时失败；
- 成功响应的 `storage_configs` 是后续 transfer client 的权威 backend 配置，并且只包含该 group 已校验的
  `storage_candidates`；普通 KVCache 的 migration source/target 不属于 KVMeta 数据面，不会混入响应导致整个
  exact-object client 初始化失败。

`Sync` 的强度由 metadata backend 决定。对 `cached` 的持久层，它是后续物理 Delete 前的持久化 ownership
barrier；对纯内存 `local`/`dummy`，它只能排序当前进程中的读写，进程退出后 location、usage 和待回收 owner 都会
丢失。KVCM 不会扫描 TairMempool 或文件 namespace 来反向重建这些记录。因此 `local` 只适合 UT、单进程临时环境，
或已经具备独立 namespace TTL/sweeper 且明确接受重启后 cache 全失效和 orphan 的场景，不能作为共享 PACE Cache
的生产 HA 配置。

### 5.2 `GetInstanceInfo`

按公共 `instance_id` 查询注册信息。响应返回公共 instance id，不暴露 KVCM 内部 namespace 编码。

### 5.3 `Get`

- `query_type` 只接受 `QT_UNSPECIFIED` 或 `QT_BATCH_GET`；
- `metas` 在 V1 未实现，非空时返回 `UNSUPPORTED`；
- `locations` 和 `hit_mask.values` 始终与请求 keys 严格等长、同下标；
- miss 的 `hit_mask=false`，对应 `locations[i]` 为空；
- active 对象不可读，表现为 miss；
- hit location 必须包含恰好一个名为 `value` 的 URI，并携带真实 `value_size`；URI authority 只能是不超过 512 bytes、
  由字母、数字、`.`、`_`、`-` 组成的非空 backend hostname，不能带 userinfo/port；Mooncake URI `key` 必须是完整
  canonical KVMeta object key；
  TairMempool/PACE URI path 必须是完整的
  `/<uint64 offset>`，可选的 `node_id`/`media_type`/`range_id` 若出现必须是完整 `uint16`；文件型 backend path
  必须是词法规范的绝对、非根路径（拒绝空 segment、`.`/`..` segment 和尾随 `/`），可打包文件型 backend 的
  `blkid` 必须缺失或等于 `0`；每个 query item 都必须显式写成 `key=value`，不接受会被 parser 规范化为相同文本的
  裸 `key`。

V1 `Get` 不创建 server-side read lease，返回 location 后不会 pin 物理 allocation。调用方必须确保对应数据面 Load
结束前没有其他 client 执行同对象的 Remove/Trim/GC；数据面读取失败应把整组当作 miss 并重算。需要读删并发保护
的后续设计见总体文档第 13 节。

### 5.4 `PutStart`

`keys` 与 `value_sizes` 必须严格等长。成功响应有两种数组形状：

| 字段 | 对齐方式 |
|---|---|
| `key_mask.values` | 与请求 keys 等长；`true` 表示同 key、同尺寸对象已 committed，不表示服务端比较过内容 |
| `locations` | 只包含 `key_mask=false` 的 miss，按请求中的相对顺序紧凑排列 |

其他语义：

- 全部命中时 `write_session_id` 和 `locations` 均为空；
- 调用方必须保证同一 key 永远对应相同 bytes；否则同尺寸错误内容会成为无法检测的 false hit；
- 有任一 miss 时返回非空 session，且每个 location 的 `value_size`、URI `size` 与请求值完全相等；
- 每个 miss 必须取得独立的 backend allocation；重复判断使用 backend 的物理 Delete identity，而非包含 `size`
  等 metadata query 的完整 URI，异常复用会在发布 reservation 前被拒绝；
- 同 key 的 committed 对象尺寸不同时，整批返回 `SIZE_MISMATCH`，不创建 allocation；
- 任一 key 仍 active 时，无论本次请求尺寸是否相同，整批都返回可重试的 `WRITE_IN_PROGRESS`，不把未提交的 size
  当成永久冲突，也不把 active 对象误报为命中；
- 容量、storage type quota 或 active-session 数量不足时，不会返回可用 session；已产生的候选 allocation 或
  reservation 会在返回前进入补偿清理。`Availability()` 与最终 session 登记之间被其他 group 抢占容量时，也执行
  同一条补偿路径；
- backend 分配失败直接返回给调用方，不触发 KVCM 驱逐。KVCM Reclaimer 只根据自身 group/type
  逻辑配额、metadata key 上限和水位回收，不猜测共享 Provider 的可用物理空间；
- 新 reservation 尚未被 session 接管时，只有补偿删除完成持久化才能返回普通超时、`RESOURCE_EXHAUSTED`
  或 not-leader；若补偿结果无法证明，
  返回 `OUTCOME_UNKNOWN` 并关闭 KVMeta admission/maintenance，直到 leader recovery 完成，不影响普通 KVCache。
- group reclaim 配置被热更新为非法值时，全部命中的请求仍可幂等返回；包含任一 miss 的请求在 backend allocation
  前返回 `SERVICE_NOT_READY`。Remove/Trim 仍可用于安全排空已有对象。

storage Create 抛出的标准或未知异常不会穿透服务线程。因为异常调用可能已经分配对象却没有返回 identity，服务端
返回 `OUTCOME_UNKNOWN` 并关闭 KVMeta admission/maintenance，避免每次重试都形成一个不可追踪 orphan；异常前已明确拿到的
singleton allocation 做一次补偿删除；异常调用本身若在 provider 端已经分配但没有返回 URI，则只能由 backend
orphan 清理发现，服务端不会猜测或重放该 Create。

非 `OK` Create 结果必须携带空 URI；若同时返回 URI，服务端无法证明它是本次新 allocation、诊断信息还是已有对象，
因此不使用该 URI 做补偿删除，返回 `OUTCOME_UNKNOWN` 并关闭 KVMeta admission。`OK` 的文件型 URI 除了精确匹配
本次随机 object key，还必须位于已注册 backend 配置的真实 root/mount/root_dir；仅有相同 `kvmeta/...` 后缀不足以
获得 Delete 权限。

Create 返回数量与单 key 请求不等、URI 不满足 backend 的 singleton ownership shape，或者返回的 file/Mooncake
identity 不对应本次请求生成的 `kvmeta/<instance-hash>/<key-hash>/<nonce>` 对象及注册 backend root 时，这个响应中的 URI
**不构成物理 Delete 授权**：它可能是其他对象或共享 allocation。
服务端只会清理更早的、由独立且形状完整的 singleton Create 响应证明归属的候选对象，随后关闭
KVMeta admission/maintenance 并等待 leader recovery。无法定位的 allocation 留给 backend orphan 机制，不会为了回收空间而
冒险删除不属于本次请求的数据。该类 provider 契约损坏返回 `INTERNAL_ERROR`，不是业务对象的
`SIZE_MISMATCH`。若文件型/Mooncake 响应恰好只有一个，且随机 path/object key 已唯一证明本次 allocation 的归属，
但 URI `size` 与请求值不同，服务端可以安全地只补偿删除该 allocation 一次，随后同样以 `INTERNAL_ERROR`
fail closed。PACE GA 没有 generation token；任一必需字段、exact size 或介质兼容性校验失败时，返回中的 GA 不构成
Delete authority，adapter 不发送补偿 Delete，而是返回 `OUTCOME_UNKNOWN` 并关闭 KVMeta gate。

PACE 的 KVMeta 专用 adapter 复用现有 `POST /v1/api/gas/batch`，但固定使用 `count=1` 和该对象的 exact size。
成功 URI 仍是 legacy GA 加 `node_id/media_type/range_id/size`；服务端要求字段完整、canonical、`node_id` 非零、请求
size 一致，并拒绝同一批中重复的物理 GA。当注册介质为 `0`（不指定）时，根据 Provider 模式接受 PACE 返回的
`0/2/5`；显式 DRAM/SSD 仍必须精确匹配。当前协议没有 generation token，所以这些字段只能证明本次 singleton
Create 的返回形状，不能让未知结果的 Delete 变成可安全重试操作。

官方 client 对成功 `PutStart` 的响应做完整形状校验。响应畸形时，只有使用可寻址 session 和服务端确认的 item
count 成功执行 `PutFinish(false)`，才会返回原始的 `INTERNAL_ERROR`/`SIZE_MISMATCH`；session id 缺失或超限、无法
推导 cardinality、abort transport 失败、`OUTCOME_UNKNOWN` 或任何明确 abort 拒绝，都返回 mutation
`OUTCOME_UNKNOWN`（transport 失败保留对应 ambiguous code）。因此 Python wrapper 会设置 `unknown_outcome=True`，
调用方不能把一次可能仍持有 reservation 的 PutStart 当作干净拒绝直接重试。

### 5.5 `PutFinish`

- `write_session_id` 必须非空且属于相同 `instance_id`；
- `success_keys.values` 必须存在、非空，并与 `PutStartResponse.locations` 等长，而不是与原始 keys 等长；
- 全部为 `true` 时提交本 session 的全部新对象；
- 任一为 `false` 时回滚本 session 的全部新对象；
- 空 mask、长度错误或过期 session 不能被当作成功；
- 长度错误不会消费仍有效的 session，调用方可用正确 mask 重试；
- `locations` 字段为未来客户端分配模式保留；当前服务端分配模式以 session 保存的位置为准。

回滚和 session timeout 会先把 exact active owner 条件改成读不可见、仍计费的 `CLS_DELETING` tombstone 并完成
`Sync`，再按 backend 能力清理。NFS 使用不可复用的随机路径：删除失败时保留 tombstone/usage，recovery 可重试。
TairMempool/PACE 路径要求数据面在返回前通过 `pace_synchronize` 排空已提交 I/O，因此失败 Finish 和
timeout 可立即进入清理。legacy GA Delete 只发送一次；成功则正常完成 metadata，结果不确定时
逻辑完成 metadata、关闭 KVMeta admission/maintenance，并把底层地址计为可能 orphan，绝不重放该 GA。
进程重启/换主若看到 PACE retired tombstone，只做 metadata finalization，因为无法判断上一任是否已经发出 Delete。

若 active metadata 的删除无法证明已经持久化，session 已被消费且不能再充当 owner，服务端返回
`OUTCOME_UNKNOWN`（已过期调用仍保留其 timeout 契约），并关闭 KVMeta admission/maintenance 直到 recovery。
多 key commit 只成功了一部分时也先持久化当前视图再判定每个 allocation 的 owner；该 barrier、reload 或补偿删除
不确定时同样 fail closed。gate 关闭后拒绝新的 PutStart、Remove 和 Trim，但允许已发布 session 完成 finalization。
仅仅 URI 相同不构成删除 metadata 的授权，遇到同 allocation 的意外 owner 会保留它并交由 recovery。
同理，exact owner 到 tombstone 的条件更新返回 `NOT_FOUND`/`MISMATCH` 时，不会对旧 URI 发物理 Delete：其他 actor
可能已经完成或替换 ownership。服务端返回 `OUTCOME_UNKNOWN` 并关闭 KVMeta gate，而不是用 metadata absence
猜测物理所有权；仍存在的持久化 record 由 recovery 对账。

多 key commit/rollback 使用逐 key 条件更新和失败补偿，不承诺并发 `Get` 观察到同一瞬间的全有或全无。

### 5.6 `Remove`

- 精确删除给定 committed keys；不存在的 key 幂等成功；
- 任一 key 仍有 active write 时，整批返回 `WRITE_IN_PROGRESS`，不删除任何 key；
- 服务端先把 exact owner 条件转换并持久化为 tombstone，再执行 backend-specific Delete，最后删除 tombstone 并 `Sync`；
- 显式 Remove 不等待自动 Reclaimer 的 `delay_before_delete_ms`，调用方必须先排空 consumer；
- tombstone barrier 或最终 metadata barrier 失败时返回 `OUTCOME_UNKNOWN` 并保留可恢复状态。NFS Delete 失败同样
  保留 tombstone/usage 供 recovery 重试；PACE Delete 结果不确定时逻辑完成 metadata、释放逻辑 quota、关闭 KVMeta
  并留下可能物理 orphan。调用方不能盲目重放 mutation；若 stable location id 已被 replacement owner 占用，旧 URI
  不进入物理 Delete。

### 5.7 `Trim`

| 策略 | V1 行为 |
|---|---|
| `TS_REMOVE_ALL_CACHE` | 删除 instance 的 KVMeta metadata 和可归属的物理对象 |
| `TS_REMOVE_ALL_META` | 只删除 metadata，保留物理对象；仅用于明确的修复场景 |
| `TS_TIMESTAMP` | 返回 `UNSUPPORTED` |
| `TS_UNSPECIFIED` | 返回 `INVALID_ARGUMENT` |

存在 active session 或正在提交/回滚的 session 时，Trim 整体返回 `WRITE_IN_PROGRESS`，不产生删除副作用。
`TS_REMOVE_ALL_CACHE` 与 Remove 使用相同的 `tombstone -> backend-specific Delete -> metadata finalization` 语义。
Trim 跨 batch 记录是否已经改变 metadata；只要任一早期 batch 已进入退休状态，后续的物理删除失败、scan/校验错误或
降主取消都返回 `OUTCOME_UNKNOWN`，不会返回可整体自动重试的 not-leader/普通 I/O 错误。NFS 未完成对象继续由
durable tombstone 持有并计费；PACE 已尝试的 legacy GA 不会由 recovery 重发。`TS_REMOVE_ALL_META` 则按定义只删除 metadata，调用方必须提前确认
对应物理对象将由 backend/namespace 清理机制回收。任一 batch 的 metadata 删除已进入内存但无法证明持久化时，
per-instance Trim fence 即使随调用返回而销毁，全局 KVMeta gate 仍保持关闭，直到 recovery。scan 后条件删除发现
replacement owner 时也遵循相同的 fail-closed 规则，并禁止用旧 URI 发物理 Delete。

容量观测有两张不能混用的账：MetaIndexer usage 是 active/committed/retired metadata 的逻辑归属字节，用于
KVMeta group/type quota、准入和 Reclaimer；PACE/provider usage 是包含共享 workload、碎片和 orphan 的底层物理
占用，用于 allocator 硬保护与运维告警。NFS 在物理终态与最终 metadata barrier 都成功前不降低逻辑账；PACE
unknown Delete 会逻辑完成并关闭 KVMeta，因此逻辑账可能下降而物理账不降。任何时候都不能仅凭 metadata counter
推导 provider 剩余物理容量，底层 allocator 仍须独立执行硬容量保护和告警。

### 5.8 自动 Reclaimer

自动回收在 RPC 主链路之外异步运行。容量不足的 `PutStart` 返回 `RESOURCE_EXHAUSTED` 并唤醒按需回收，调用方按延迟
预算选择有界重试或直接重算。Reclaimer 先把 committed metadata 持久化为读不可见、仍计入 quota 的
`CLS_DELETING` tombstone；本批全部 reader fence 持久化后才设置统一的有限 grace deadline。grace 到期后再次确认
tombstone durable，再按 backend 策略删除。NFS 随机路径的错误/异常按 100ms 到 30s 指数退避重试；PACE legacy
GA 只调用一次，失败、短结果或异常会增加 attempted/uncertain 对象与字节指标，关闭 KVMeta 侧路并保留 durable
tombstone；recovery 只做 metadata finalization，绝不重发 GA。reader fence barrier 失败或状态转换结果不确定时也会
关闭 KVMeta；有限 deadline 的 `Sync` 暂时失败时保留 batch，并在物理删除前重试 barrier。一次 reclaim round
同时选中可重试后端和 PACE 时会按 delete policy 拆成独立 pending batch，任何一类的错误都不会污染另一类的重试语义。

最终 metadata 删除的 `EC_NOENT` 只有在同一 pending batch 已记录本次 compare-and-delete 已应用时，才可作为
`Sync` 重试成功；意外缺失或 replacement owner 会关闭 KVMeta maintenance。进程退出时 tombstone 是持久化 cleanup
WAL；下一任 leader 重放 NFS 随机路径 Delete，但对 PACE retired record 只做 metadata finalization，避免旧 GA 已复用
后被误删。普通 KVCache 请求不经过这条链路。

进程内 pending 队列最多容纳 1024 个 batch、20000 个对象和 4 TiB。候选选择同时受剩余 object/byte budget 约束，
超出剩余额度的候选会被跳过或裁剪，而不是让一个过大的采样结果永久阻塞后续回收。

## 6. 错误码和调用方动作

| 错误码 | 含义 | 建议动作 |
|---|---|---|
| `OK` | 请求成功 | 使用响应 payload |
| `INVALID_ARGUMENT` | 字段、边界或数组形状非法 | 修正请求，不重试原请求 |
| `UNSUPPORTED` | V1 不支持该模式 | 改用受支持模式 |
| `DUPLICATE_ENTITY` | instance 已存在但注册配置不一致 | 对照既有 instance 配置，不覆盖重试 |
| `INSTANCE_NOT_EXIST` | 未注册或内部 schema 不匹配 | 检查注册和部署配置 |
| `SERVER_NOT_LEADER` / `SERVICE_NOT_READY` | endpoint 当前不能服务，或 KVMeta group 没有有效 LRU/读热度回收配置 | endpoint 问题可切换地址；配置问题先修复 group，避免无界重试 |
| `RESOURCE_EXHAUSTED` / `REACH_MAX_ENTITY_CAPACITY` | byte quota、session 或实体容量到限 | 释放对象或扩容后再试 |
| `WRITE_IN_PROGRESS` | 相同 key 或 instance 正在写/finalize；active size 仍是临时值 | 等原 session 收敛，不并发覆盖 |
| `SESSION_NOT_FOUND` | session 过期、不存在或 instance 不匹配 | 查询最终状态，不把它当成功 |
| `SIZE_MISMATCH` | 已 committed 对象或 location 尺寸不一致 | 使用新 key，或先确认并删除旧对象 |
| `NOT_FOUND` | 对象/instance 元数据不存在 | 按业务 miss 处理 |
| `IO_ERROR` | metadata/storage 操作失败或超时 | 根据操作幂等性判断，避免盲目重放 mutation |
| `OUTCOME_UNKNOWN` | mutation 后的回滚/对账无法证明唯一最终状态 | 查询最终状态；不得盲目重试 mutation |
| `INTERNAL_ERROR` / `UNKNOWN_ERROR` | 服务端不变量或未知错误 | 记录 trace/request id 并排查 |

## 7. Failover 与 transport error

多地址 client 在一个总 `call_timeout_ms` 预算内工作：

- 可安全重试的调用会在尚未尝试的 endpoint 间分配剩余预算，避免黑洞首地址独占总 deadline；
- `Get`、`GetInstanceInfo` 和同配置 `RegisterInstance` 遇到 transport error 可以尝试下一 endpoint；
- 所有 RPC 收到明确的 not-leader/not-ready 业务响应时可以 failover；
- `PutStart`、`PutFinish`、`Remove` 和 `Trim` 的 transport error 具有不确定结果，官方 client 不自动重放；
- C++ client 将这类结果映射为 `ER_INVALID_GRPCSTATUS`。

例如 `PutStart` transport error 后 `Get` 仍 miss，可能只是原 session 仍 active，并不代表可以立即用同 key
重写。调用方应等待原 write timeout 或通过审计确认结果。

## 8. 推荐客户端

### 8.1 `KvMetaObjectClient`

推荐业务使用它完成：

- 初始化时注册 instance 并创建数据面；
- 写入前校验整个 batch；
- `PutStart` 后只写 miss；
- 校验服务端 location、真实 URI 和 actual URI；
- 任何写入错误调用失败 mask 回滚；
- 读取时要求全部命中并在 I/O 前完成 location/size 校验。

exact-size transfer client 会依据注册响应中的权威 storage config 重建文件型对象的完整路径，并校验 canonical
instance hash、key hash 和 32-byte nonce；仅处于相同 `kvmeta/` 前缀、但跨 root、缺少 segment 或追加 segment 的
URI 都不会进入 SDK I/O。这项限制只作用于 KVMeta 数据面，不改变普通固定 block `TransferClient`。

数据面保护 caller-owned buffer，可能在名义 timeout 后等待不可取消 I/O 结束。调用方必须在同步方法返回前保持
buffer 和其 owner 存活。

C++ 调用方可显式调用幂等的 `KvMetaObjectClient::Close()`；它会拒绝新请求、等待本 client 已准入的同步操作结束，
再释放 metadata/data-plane 资源。之后的合法操作返回 `ER_CLIENT_NOT_EXISTS`。Python `close()` 和 context manager
会调用同一 native 清理入口。

该 drain 只保护本地 buffer 生命周期，不会续约服务端 client commit deadline。当前 V1 只接受在同步
Get/Put 返回前已经终止并排空本次 I/O 的 backend。无法提供该终态保证的配置不能用于 EMB cache。

KVMeta 的内部 TairMempool adapter 直接复用现有 PACE MetaService：Create 使用
`POST /v1/api/gas/batch` 且 `count=1`，Delete 使用 `DELETE /v1/api/gas?ga=...`。这两者都走 storage candidate 已有的
PACE HTTP endpoint；KVCM 对外的 KVMeta gRPC service 则与旧 MetaService 共用 `kvcm.service.rpc_port`，两种“同端口”
不要混淆。固定块 KVCache 继续调用原有 Create/Delete 路径，variable-size 校验和 at-most-once
删除策略只在 KVMeta side capability 开启。

当前 PACE URI 没有 generation token，GA 又可复用。因此 Delete 失败、短结果或异常一律视为 outcome unknown：
KVCM 关闭 KVMeta、记录可能 orphan，并且不再发送同一 GA。生产部署必须监控 Provider 物理用量和 orphan 告警；若
不能接受这种受控空间泄漏，需要先实现设计文档第 13 节的 generation-bound 条件删除，不能靠重试 legacy API 规避。

Python wheel 的推荐入口是：

```python
from kv_cache_manager.client import KvMetaObjectClient, KvMetaObjectClientConfig

client = KvMetaObjectClient(
    KvMetaObjectClientConfig(
        addresses=("127.0.0.1:6381",),
        instance_id="rtp-emb",
        instance_group="epd-emb-only",
        transfer_client_config=transfer_json,
    )
)
client.save(keys, contiguous_tensors)
client.load(keys, destination_tensors)
client.remove(keys)
client.close()
```

高层 Python client 在 native I/O 前校验整个逻辑调用，并自动按 64 objects / 4 GiB 拆成服务大小的 batch。
`save`/`remove` 不自动重试，`save` 也不自动删除已完成的前序 batch：通用 key 可能已经存在，而 transport error
可能具有不确定结果。异常 `KvMetaObjectClientError` 会给出 batch 位置、已确认完成数量及
`unknown_outcome`，由拥有 key 生命周期的上层决定审计或清理策略。

wheel 的 package 与 native extension 同时导出 `KV_META_OBJECT_API_VERSION=2`，extension 的值来自所链接 client
shared library 的版本查询。高层 client 在 native client 初始化前校验版本、必需类型、枚举成员和工厂方法；版本缺失
或不匹配直接失败，避免 Python wrapper、extension 与 client library 混装。
native mutation 若返回未知/畸形 code，也按 `unknown_outcome=True` 失败关闭，不能把它解释成可安全重试的明确拒绝。

### 8.2 底层客户端

需要自行编排事务时可以分别使用：

- `KvMetaClient`：只处理本文 metadata RPC；
- `KvMetaTransferClient`：只按 URI、`value_size` 和 caller buffer 搬运 exact bytes。

现有固定 block `TransferClient` 不接受 KVMeta 任意长度对象，不能互换。

Python 调用方可以依赖 Bazel target
`//kv_cache_manager/protocol/protobuf:kv_meta_service_py_proto` 使用生成的 gRPC stub，或使用
`//kv_cache_manager/client/pybind:kvcm_py_client_lib_wheel` 构建同时包含上述高层 client 与 native object API 的
wheel。
`kv-cache-manager-client` RPM 只发布 C++ headers 和 `kv_cache_manager_client.so`，不包含 Python wheel。
