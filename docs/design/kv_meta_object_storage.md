# KVMeta EMB Cache 设计与实现

## 1. 总体设计

KVMeta 为 EPD/多模态场景提供变长 embedding cache：

- KVCM 管理 key、对象长度、PACE 地址、状态、用量和 GC。
- embedding bytes 由客户端直接读写 TairMempool/PACE，不经过 KVCM Server。
- KVMeta 与原 MetaService 共用 RPC 端口，但使用独立 protobuf service、instance schema 和 Reclaimer。
- V1 仅支持 TairMempool DRAM/SSD，不修改普通 KVCache 和 TairMempool provider。
- KVMeta 是旁路 cache；miss、容量不足或 I/O 失败时，RTP 必须能重新计算 embedding。

```mermaid
flowchart LR
    RTP["RTP-LLM"] --> OBJ["KvMetaObjectClient"]
    OBJ -->|"metadata RPC"| KVCM["KVCM 同一 RPC 端口"]
    KVCM --> MGR["KvMetaManager"]
    MGR --> META["MetaIndexer"]
    OBJ -->|"直接搬运 bytes"| PACE["TairMempool / PACE"]
    GC["KvMetaReclaimer"] --> META
    GC --> PACE
```

核心取舍：

1. 对象不可变：相同 key、相同 size 视为命中，不覆盖旧值。
2. 只有 `SERVING` 对象可读、计入 usage。
3. `PutStart` 不加容量锁、不预留空间，允许短暂超调。
4. GC 根据已提交字节异步回收到目标水位。
5. 删除按 generation 条件执行，不能误删同 key 的新对象。

## 2. 数据模型

### 2.1 Instance 隔离

公共 instance id 被编码为独立内部 ID：

```text
__kv_meta_v1__<hex(public_instance_id)>
```

KVMeta instance 使用完整 schema marker：

```text
block_size       = 1
location_spec    = { "value": 1 }
model_name       = "__kv_meta_object__"
dtype            = "opaque_bytes"
deployment.extra = "kv_meta_v1"
```

`1` 只是注册协议的 schema marker，不是对象大小。KVMeta group 必须专用；普通 GC/Reclaimer 跳过 KVMeta instance，KVMeta Reclaimer 也不会处理混有普通 instance 的 group。

### 2.2 Key 和 Location

```text
internal_key = Hash64(business_key)
location_id  = "kvmeta:v1:" + hex(business_key)

CacheLocation {
  status: WRITING | SERVING
  type: TAIR_MEMPOOL | TAIR_MEMPOOL_SSD
  create_time: <generation time>
  spec: { name: "value", uri: "pace://...?size=<actual-bytes>" }
}
```

完整业务 key 保存在 location id 中，用于区分 64 位 hash 冲突。服务端和客户端都校验 storage type、backend、URI 和真实 size。

Generation 由 location id、type、create time 和完整 URI 等字段共同确定。`WRITING -> SERVING` 不改变 generation。

## 3. 状态机和请求流程

```mermaid
stateDiagram-v2
    [*] --> ABSENT
    ABSENT --> WRITING: PutStart 分配 PACE 并发布 metadata
    WRITING --> SERVING: PutFinish 全部成功
    WRITING --> ABSENT: 失败 / 超时 / 恢复清理
    SERVING --> ABSENT: Remove
    SERVING --> DELETING: GC 条件标记，不再可读
    DELETING --> ABSENT: grace 后物理删除返回并条件清理 metadata
```

### 3.1 写入

1. 客户端校验 key、对象真实大小和 buffer。
2. `PutStart` 查询 exact key：
   - `SERVING` 且 size 相同：返回 hit。
   - `SERVING` 但 size 不同：返回 `SIZE_MISMATCH`。
   - `WRITING`：返回 `WRITE_IN_PROGRESS`。
3. 对 miss 按真实 size 分组，复用 PACE 批量 Create；返回 location 仍保持请求顺序。分配或后续 metadata 发布失败时，释放尚未公布的私有分配，并对已公布项做 generation 保护的清理。
4. 条件创建 `WRITING` metadata，并记录有期限的内存 write session。
5. 客户端直接向 PACE 写 bytes。
6. 客户端确认全部数据面写入已结束后调用 `PutFinish`；全部成功时，条件执行 `WRITING -> SERVING` 并增加 usage。
7. 已确认全部远端 I/O 停止后的失败，可清理本 session 的全部 generation，并尽力释放 allocation；状态未知时不发送 `PutFinish`，等待 session 过期只清理 metadata。

多 key session 提供整批最终成功/失败语义，但不保证多个 key 同时线性化可见。

超时不能证明后台 I/O 已停止。未收到客户端结束确认的 `WRITING` 只清理 metadata，保留 allocation，避免旧写入污染复用地址。

### 3.2 读取

1. `Get` 只返回 `SERVING` 对象的 URI 和 `value_size`，返回前在 shard fence 下复核 generation；并发删除或替换使原 generation 失效时返回 miss。
2. object client 要求整批命中，并校验目标 buffer 字节数。
3. PACE SDK 直接写入调用方 CPU/CUDA buffer。

读取失败后必须丢弃全部目标 buffer，因为报错前可能已发生部分写入。

当前 C++ client 在 PACE Load 后不做第二次 metadata Get。读删除并发依赖 `delay_before_delete_ms`：它必须覆盖最长读取时间及调度抖动。

### 3.3 删除

1. 读取当前 exact generation。
2. 对 `WRITING` 返回 `WRITE_IN_PROGRESS`。
3. 按 generation 条件删除 metadata。
4. 删除成功才扣 usage。
5. 确认 metadata 删除已持久化，否则保留 allocation 并返回错误。
6. 等待 `delay_before_delete_ms` 后，best-effort 删除 PACE allocation；等待被降主/退出打断时保留 allocation。

不存在的 key 视为删除成功。transport error 时结果可能未知，不能盲目重试。

## 4. 变长对象实现

对象真实长度同时出现在：

```text
PutStartRequest.value_sizes[i]
PACE URI 的 size
本地 tensor/buffer 的总字节数
```

三者必须一致。`KvMetaTransferClient` 持有私有 PACE SDK client，每批 I/O 前按本批对象长度更新 `spec_byte_sizes_per_block`。该配置不影响普通定长 TransferClient，也不需要修改 TairMempool provider。

同一个 object client 的 Save/Load 串行执行，并在 `PutStart` 前取得 data-plane mutex，避免本地排队消耗 write lease。不同 client 仍可并发，由服务端条件写协调。

SDK 超时或抛异常后会销毁私有 wrapper 并等待后台任务退出，防止旧任务继续访问 buffer 或已经释放的 allocation；该 data-plane client 随后不再可用。

## 5. 容量统计和准入

### 5.1 Usage

```text
instance_usage[type] = Σ 已提交且 metadata 尚未删除的对象 value_size（含等待回收的 DELETING）
group_usage          = Σ group 内全部 KVMeta instance_usage
```

每个 `MetaIndexer` 使用按 storage type 划分的原子计数器：

| 操作 | Usage |
|---|---|
| 创建/清理 `WRITING` | 不变 |
| `WRITING -> SERVING` 成功 | 增加真实 size |
| Remove 条件删除已提交对象 / GC 清理对应 `DELETING` metadata | 减少真实 size |
| 条件删除失败 | 不变 |
| PACE 物理删除失败 | 不恢复 |

metadata 发布、提交和删除按 instance 与 key 共同选择 mutation lock，批量操作按固定顺序获取锁。同 key 的 usage 更新、删除确认和新 generation 保持受保护，不同 key 可以并行；PACE 分配不持有该锁。写会话使用到期时间索引，避免每次新写入唤醒后扫描全部会话；默认最多 65536 个活动会话，并上报活动数和拒绝数。

usage 是逻辑已提交字节，不等于 PACE 物理占用，不包含：

- `WRITING` allocation。
- allocator 对齐和管理开销。
- metadata 已删除但物理删除失败、取消或无法确认写入停止而保留的孤儿 allocation。
- PACE Create 后、metadata 发布前崩溃留下的 allocation。

### 5.2 快照准入

`PutStart` 只检查当前已提交 usage 是否达到 group/type capacity，不计算：

```text
current_usage + requested_bytes
```

也不加容量锁或创建 reservation。例如 capacity 为 100 GiB、usage 为 99 GiB 时，4 GiB 写入仍可继续并提交到 103 GiB。PACE allocator 决定物理分配是否成功，后台 GC 负责异步收敛。

`used_percentage=0.8` 是 GC 目标，不是写入拒绝线。

## 6. GC

有效容量和目标水位：

```text
capacity = min(group capacity, matching PACE tier quota)
target   = floor(capacity * used_percentage)
excess   = group_usage - target
```

当扣除仍有效的在途回收额度后 `group_usage > target` 时，KvMetaReclaimer 执行：

1. 复用普通 Cache 的采样入口；热缓存回填完成时在本地采样，恢复中从持久层采样。
2. 排除已在途 key；复用维护性 touch 让这些 key 让出冷分片采样位置，避免等待宽限期的 key 占满下一轮候选。
3. 合并候选并按访问时间排序，按需扩大批次读取 metadata，只保留合法的 SERVING generation。
4. 复用普通 Cache 的单批大小、每轮请求数、pending 对象数/字节数和超时配置，向 SchedulePlanExecutor 提交有界异步删除任务。
5. executor 校验预期 generation，将 metadata 标记为 DELETING，并按现有删除流程确认持久化。
6. executor 延迟调度完整 reader grace，不占 worker 睡眠；之后按 storage 分组执行物理删除。
7. 物理删除返回后条件清理 metadata、扣减 usage。删除失败不会恢复可读状态，也不会自动重试物理释放。
8. 超时任务停止抵扣预测回收额度，但保留 pending 去重和限额占用，避免重复释放；后续完成时再移除 pending。

这是异步采样 LRU，不是每轮全量扫描。普通 maintenance 读取不刷新业务访问时间；对已在途候选的维护性 touch 只调整本地采样顺序，不记录业务命中或写持久层。本地或 hot-cache 业务命中复用读取时已更新的访问时间，不重复 upsert；纯 Redis 命中仍更新持久化 LRU。

回收指标分别记录提交对象/字节、实际成功删除 metadata 的对象/字节和物理删除成功的对象/字节；任务完成批数不能当作对象回收 QPS。pending 最老时长、额度超时及限额跳过可用于定位停滞。当前没有长期 DELETING 的自动修复或物理失败持久化重试队列，失败或取消仍可能留下孤儿 allocation。读取安全要求 grace 覆盖完整读取时间及调度抖动。

关键观测项（`kv_meta.` 为 registry/KMonitor 名称前缀）：

- `reclaim.deleted_objects_total` / `reclaim.deleted_bytes_total`：实际 metadata 删除累计量。
- `reclaim.physical_deleted_objects_total` / `reclaim.physical_deleted_bytes_total`：物理删除成功累计量；与 metadata 删除量分开判断。
- `reclaim.pending_objects` / `reclaim.pending_bytes` / `reclaim.pending_requests`：全部未完成任务占用；`reclaim.pending_credit_bytes` 仅表示仍可抵扣的预测回收额度。
- `reclaim.pending_oldest_ms`、`reclaim.round_time_us` 和 `reclaim.stage_count`：排查慢任务、慢轮次、限额跳过与超时。
- `active_write_sessions`、`write_session_capacity`、`write_session_rejected_total`：会话容量和拒绝。
- `mutation_lock_wait_us` / `mutation_lock_hold_us` / `mutation_lock_count`：累计等待/持有微秒与次数，窗口增量相除得到平均耗时。

累计计数在 KMonitor 中按累计 Gauge 上报，应在查询侧计算窗口增量/时间；Prometheus 使用 counter 的 rate。例如对象回收速度为 `rate(kvcm_kv_meta_reclaim_deleted_objects_total[1m])`。旧 `reclaim.deleted_count` / `reclaim.deleted_bytes` 是最后完成批次的 Gauge，不代表每秒速度。

## 7. 恢复和服务隔离

升主后启动 KVMeta maintenance worker，等待 Registry 完整恢复及相关 MetaIndexer 创建完成，在首轮 GC 前扫描 persistent metadata：

- 删除遗留 `WRITING` metadata，保留可能仍被旧客户端写入的 allocation。
- 汇总全部合法 `SERVING` 和等待回收的 `DELETING` size。
- 重建每个 MetaIndexer 的 usage。

同一实例完整扫描按 internal key 去重，每个 key 下仍计量所有合法 location。单 key 在读取前消失时继续扫描；扫描本身失败时不发布该实例的 usage，也不标记 ready。

恢复完成前 KVMeta 返回 `SERVICE_NOT_READY`，不阻塞原 MetaService。KVMeta 不信任可能滞后的持久化 usage，而以对象 metadata 扫描结果重建。

降主或停服时先关闭 KVMeta 新请求，再停止 session/reclaimer worker并等待在途请求完成。write session 不持久化，遗留 `WRITING` 由下一任 Leader 清理。

## 8. RTP-LLM 接入

RTP facade 复用现有 `RECO_*` endpoint、VIPServer、PACE SDK 和 timeout，不增加 EMB 专用环境变量。

```text
RECO_INSTANCE_GROUP=pace_group_m3

派生：
instance_group = kve_pace_group_m3
instance_id    = kve_pace_group_m3
```

若配置 `RECO_INSTANCE_ID_SALT=model-a`，instance id 为 `kve_model-a`。运维需要提前创建派生后的专用 group；client 自动注册 instance，但不创建 group。

```python
import torch

from kv_cache_manager.client import KvMetaObjectClientError
from rtp_llm.multimodal.kvcm import RtpKvMetaObjectClient


emb_cache = RtpKvMetaObjectClient()


def get_embedding(key, shape, encode, request_id):
    output = torch.empty(shape, dtype=torch.float32)
    try:
        if emb_cache.try_load_one(key, output, trace_id=request_id):
            return output
    except KvMetaObjectClientError:
        pass

    embedding = encode()
    try:
        emb_cache.save_one(key, embedding, trace_id=request_id)
    except KvMetaObjectClientError:
        pass
    return embedding
```

业务 key 必须包含 tenant、模型/预处理版本、内容 digest、shape、dtype、layout 和 cache schema。稳定共享 key 不应在每次读取后 Remove，正常淘汰交给 Reclaimer。

RTP 仍负责 singleflight、回源、内容校验、poisoned entry 修复、指标和 EPD 调度；facade 只提供 exact-object cache primitive。

## 9. 失败语义与限制

| 场景 | 处理 |
|---|---|
| Get miss | 重算 embedding |
| PACE allocation/I/O 失败 | cache 失败，推理继续 |
| PutFinish/Remove transport error | 结果可能未知，不盲目重试 |
| session 超时或恢复遇到 `WRITING` | 条件清理 metadata，保留 allocation |
| PACE Delete 失败 | metadata 已 miss，记录孤儿风险 |
| Leader 恢复未完成 | KVMeta 暂不可用，普通 MetaService 不受影响 |

V1 不提供强一致容量 reservation、跨 key 原子可见、持久化 session、read lease、内容 checksum、多 tier 混合存储和 `Trim` handler。

PACE 接口尚无写入隔离机制。KVMeta 无法安全自动回收未获客户端结束确认的 allocation，因此这些孤儿仍占物理容量。

主要限制：

| 项目 | 上限 |
|---|---:|
| native batch | 64 objects |
| key / instance id / group | 512 bytes |
| 单对象 | 1 GiB |
| 单批总大小 | 4 GiB |
| 活跃 write session | 4096 |
| write timeout | 1800 秒 |

## 10. 实现细节

### 10.1 Session 和回滚

- session 使用 `steady_clock` deadline，避免系统时间回拨。
- session map 满时，在创建新 PACE allocation 前拒绝请求。
- 后台超时只清理 metadata；显式 `PutFinish` 确认全部写入停止后，才可释放 allocation。metadata 清理失败转为 cleanup-only session，每秒重试。
- PACE Create 成功但 metadata 尚未发布时失败，直接 best-effort Delete。
- 若进程恰在 Create 后、metadata 发布前崩溃，Leader recovery 无法从 metadata 发现该 allocation。

### 10.2 部分提交

底层 metadata batch 可能部分成功。实现记录已经 `SERVING` 的项及其 usage；整批未全部成功时，释放 mutation lock 后清理本 session 的所有 generation，已提交项在清理时扣回 usage。

### 10.3 Hash 冲突

索引使用 64 位 hash，完整 key 保存在 location id。批量操作将相同 hash 拆成不同 unique-key layer，因此冲突不会导致两个业务 key 相互覆盖。

### 10.4 删除顺序

```text
条件删除 metadata
    -> 扣减 SERVING usage
    -> 等待 delay_before_delete_ms
    -> best-effort PACE Delete
```

这是主动 Remove 的清理顺序。后台 GC 使用第 6 节的异步 `DELETING -> reader grace -> 物理删除 -> metadata 清理` 流程；每个异步任务以最后一次状态标记完成时间为起点等待完整 grace，等待期间降主或退出则保留 allocation。

该顺序优先保证缓存立即 miss 和主链路不被物理删除故障阻塞，代价是物理 usage 可能与逻辑 usage 偏离。

### 10.5 测试覆盖

- [`kv_meta_manager_test.cc`](../../kv_cache_manager/manager/test/kv_meta_manager_test.cc)：状态机、session、usage、软容量、恢复和 GC。
- [`kv_meta_service_impl_test.cc`](../../kv_cache_manager/service/test/kv_meta_service_impl_test.cc)：RPC shape、错误码和 leader gate。
- [`kv_meta_object_client_test.cc`](../../kv_cache_manager/client/test/kv_meta_object_client_test.cc)：metadata/数据面编排、回滚和生命周期。
- [`kv_meta_transfer_client_test.cc`](../../kv_cache_manager/client/test/kv_meta_transfer_client_test.cc)：exact-size I/O 和 timeout。
- [`kv_meta_object_client_py_test.py`](../../kv_cache_manager/client/test/kv_meta_object_client_py_test.py)：Python batching、异常和 Close。
- RTP：`mm_kvcm_emb_config_test.py`、`mm_kvcm_emb_client_test.py`、`mm_kvcm_emb_client_integration_test.py`。

### 10.6 代码索引

- [`kv_meta_service.proto`](../../kv_cache_manager/protocol/protobuf/kv_meta_service.proto)
- [`server.cc`](../../kv_cache_manager/service/server.cc)
- [`kv_meta_manager.cc`](../../kv_cache_manager/manager/kv_meta_manager.cc)
- [`kv_meta_instance.h`](../../kv_cache_manager/manager/kv_meta_instance.h)
- [`meta_storage_backend_manager.cc`](../../kv_cache_manager/meta/meta_storage_backend_manager.cc)
- [`storage_usage_data.cc`](../../kv_cache_manager/meta/storage_usage_data.cc)
- [`kv_meta_object_client_impl.cc`](../../kv_cache_manager/client/src/kv_meta_object_client_impl.cc)
- [`kv_meta_transfer_client_impl.cc`](../../kv_cache_manager/client/src/kv_meta_transfer_client_impl.cc)
- [`kv_meta_object_client.py`](../../kv_cache_manager/client/kv_meta_object_client.py)

## 11. Review 重点

1. `delay_before_delete_ms` 是否覆盖线上最大读取耗时；否则需要增加 Load 后 generation 校验。
2. 是否具备 PACE 物理用量、Delete 失败和孤儿 allocation 告警。
3. 生产 metadata/Registry 是否持久化，保证重启后能重建 usage 和 allocation ownership。
4. RTP key 是否完整包含内容和版本信息，并实现热点 miss singleflight。
5. 是否接受无容量 reservation 带来的短暂超调和 PACE 分配失败回退。
6. RTP 文档不得宣称当前支持 NFS 或 post-load generation fence；GC 的 `DELETING` 不代表长期 tombstone 自动修复能力。
