# Meta 内存主存储与 Redis 异步备份设计

状态：**已实现并完成单元回归**。2026-09-16 修订，基于提交 `aac9d4dc` 及当前工作区实现。本文描述当前目标行为；真实 Redis 故障压测、完整 ASan/TSan 与生产吞吐验证仍按 10.2 执行，不能由单元测试结果替代。

## 1. 结论与适用边界

保留 `MetaStorageBackendConfig.memory_primary`，默认 `false`，保持原模式。开启后，普通写按恢复状态选择顺序：**Recover 保持原有 persistent-first，Running 才切换为 local-first**：

```text
启动/Leader 接管：Open Redis + 恢复辅助计数 → Init 完成、可服务；后台 Recover 继续回填
Recover 阶段：保留缺失 key 回源、写前补齐与 tombstone；普通写 Redis → 条件写 local
Running 阶段：普通读只读 local；普通写 local → 条件备份 Redis
Running 写结果：local 成功、Redis 失败 → 返回成功，不回退 local；local 失败 → 不提交该项备份
物理删除准入：shard lock 内 local CAS/提交备份 → 锁外 Sync → 调度物理删除
写入回滚：复用原 Reconcile 分类、删除与确认；不增加 memory_primary 专属缺失项补偿
```

“取消反压”只针对 Running 阶段由条件写入口提交的普通 Redis 备份：业务线程不等待队列腾出容量、不等待 Redis 写入；队列满时丢弃该次备份，不撤销内存更新。Recover 阶段 Redis 仍是主写，使用 `WaitAndReserve` 和原 `async_enqueue_timeout_ms`。**不取消启动/Recover 时读取 Redis 的依赖，物理删除继续复用原 executor 锁外 Sync。** Recover 对未恢复 key 的读取、写前补齐和普通写仍可能受 Redis 影响。

V1 仅支持 `cached + local + async_redis`。复用现有双 backend、MetaIndexer 分片锁、MPSC 队列、Redis consumer 和恢复链路，不新增 backend 类型、manager 子类、线程池、WAL 或自动全量同步服务。同步 `redis` 无法提供 Running 阶段的非等待备份，因此要求显式配置 `persistent_type=async_redis`。

边界如下：

- Redis Open 或辅助计数恢复失败，Init 仍失败；Init 完成后无需等待全量回填即可读写。Recover 未完成前保留原有回源和条件双写，不将尚未回填的空内存视为完整索引；并发读允许读到旧值，写入最终按原恢复协议收敛。
- local 容量必须按全量索引配置，不将容量不足设计成可持续运行的降级模式。容量不足按原接口报错并由运维扩容，不新增自动扩容、切回 Redis-first 或复杂容错状态机；未完成回填不能伪装成 Running。
- 运行时普通写成功表示 local 已提交，不表示 Redis 已保存或落盘。备份可能缺失，重启恢复的是 Redis 实际保存的状态，不保证恢复全部已确认写入。
- 进入 Running 后，Redis 故障时既有内存数据可读，普通写入不受备份队列反压；Recover 阶段 Redis 故障导致写入失败或受原队列反压是预期行为。物理回收和需要锁外 `Sync` 的回滚可以暂停/失败。
- 保留备份恢复不等于零数据丢失：普通新增可能丢失、逻辑删除可能回退；外部存储自行回收等行为也不受 KVCM 的物理删除屏障控制，须保留既有有效性/版本校验。
- 本功能不保证进程 OOM、数据存储故障、注册表/选主故障下可用。若控制面与元数据备份共用 Redis，需单独评估故障域。

## 2. 现有链路与关联约束

主要代码入口：

- [MetaStorageBackendManager](../../kv_cache_manager/meta/meta_storage_backend_manager.cc)：双写、Recover、回填、maintenance、Sync 和辅助元数据。
- [MetaStorageBackend](../../kv_cache_manager/meta/meta_storage_backend.h)、[MetaCacheBaseBackend](../../kv_cache_manager/meta/meta_cache_base_backend.h)：将通用条件写提升至公共抽象；PutIfAbsent、内存统计等 cache 专属能力留在原层。
- [MetaAsyncRedisBackend](../../kv_cache_manager/meta/meta_async_redis_backend.cc)、[MpscWriteQueue](../../kv_cache_manager/meta/mpsc_write_queue.cc)：入队、容量检查、批量写和 barrier。
- [MetaIndexer](../../kv_cache_manager/meta/meta_indexer.cc)、[MetaIndexerManager](../../kv_cache_manager/meta/meta_indexer_manager.cc)：分片锁、RMW、计数和 Indexer 生命周期。
- [SchedulePlanExecutor](../../kv_cache_manager/manager/schedule_plan_executor.cc)、[MetaSearcher](../../kv_cache_manager/manager/meta_searcher.cc)：删除前权威读取、CAS、锁外 `Sync` 及写入回滚。

原模式普通写入为 persistent-first，只有前一阶段成功的项才进入 local。async Redis 的成功通常只是入队成功；队列满时的容量等待发生在 MetaIndexer 分片锁内，会延长同分片其他请求的等待。memory-primary 在 Recover 复用该行为，进入 Running 后才通过公共条件写接口把 Redis 作为非等待 Secondary。

本次不能遗漏的关联点：

| 位置 | 修订后的处理 |
|---|---|
| Redis Open、Recover、GetMetaData | 保留 Redis 依赖及原恢复来源，不改为 local no-op 或懒建连 |
| Recover 期间的 EnsureKeyInCache、回填、tombstone | 完整保留原机制和 persistent-first 普通写，不增加前台读过滤 |
| Running 普通 Get/Exists/List/采样 | 复用现有 cache-only 分支，不重复添加模式判断 |
| maintenance 权威读、回填与双层比较 | 新模式以 local 为准，不以落后的 Redis 值覆盖 local |
| maintenance 读前 Sync | 新模式不需要为了读取 local 而等待 Redis |
| 物理删除、回滚前 Sync | 对齐原提交，在 shard lock 外调用真实 Sync，依赖现有 RedisClient 重试 |
| 运行时 PutMetaData | 改走已有 consumer 的有界异步提交，避免辅助计数持久化继续阻塞热路径 |

## 3. 配置与兼容性

### 3.1 配置位置

`memory_primary` 放在 `MetaStorageBackendConfig` 内，与 `storage_type`、`storage_uri` 同级，不放进锁配置，也不在 URI 中重复设置同名开关。

```json
{
  "meta_storage_backend_config": {
    "storage_type": "cached",
    "storage_uri": "redis://redis-backup:6379/?db=3&persistent_type=async_redis&cache_type=local&capacity=4096&async_max_size=102400",
    "memory_primary": true
  }
}
```

这是 `MetaIndexerConfig` 内的配置片段。容量值仅用于展示；`capacity` 沿用 local 的 MiB 单位，`async_max_size` 按每队列 key 操作数计量。不增加 payload 字节配置、外部保留槽位或独立重试队列。

Running 阶段的 Redis 条件备份不使用 `async_enqueue_timeout_ms` 等待容量；Recover 阶段的主写和原模式继续使用原等待参数与策略。队列中的辅助消息容量计量见第 6 节。

### 3.2 配置链路

| 位置 | 改动 |
|---|---|
| [meta_storage_backend_config.h](../../kv_cache_manager/config/meta_storage_backend_config.h) / [.cc](../../kv_cache_manager/config/meta_storage_backend_config.cc) | 默认 false 的成员、getter/setter、JSON 双向读写；旧 JSON 缺省关闭 |
| [admin_service.proto](../../kv_cache_manager/protocol/protobuf/admin_service.proto) | `MetaStorageBackendConfig` 增加 `bool memory_primary = 3;`，不变更原字段编号 |
| [manager_message_proto_util.cc](../../kv_cache_manager/service/util/manager_message_proto_util.cc) | `CacheConfigToProto` / `CacheConfigFromProto` 双向透传 |
| backend manager Init | 校验支持的组合；创建 persistent 子配置时传递开关，不只复制 URI |
| [kvcm_ops/instance_group/util.py](../../package/kvcm_ops/kvcm/instance_group/util.py) | 模型、JSON 解析/序列化、CLI 配置入口和相应校验适配 |
| [create_instance_group.py](../../package/kvcm_ops/kvcm/instance_group/create_instance_group.py)、[update_instance_group.py](../../package/kvcm_ops/kvcm/instance_group/update_instance_group.py) | 示例和 GET→编辑→PUT 保留新字段，避免修改其他参数时清回 false |
| 协议生成、配置文档、测试 | 使用既有生成链路，不手改生成产物；补默认值、双向转换及更新兼容测试 |

组合合法性只在 manager Init 校验；async backend 只校验自身资源参数。不在每次请求重复解析 URI、校验组合或动态转换 backend 类型。

模式在 Indexer 初始化时固定，不提供热切换。更新注册配置不等于现有 Indexer 自动重建。所有可能接管的服务及读改写配置的工具升级后才可开启，避免旧程序忽略字段、按旧语义接管。

## 4. WriteRoute 统一方案

### 4.1 备选方案比较

| 方案 | 优点 | 结合当前代码的主要问题 | 结论 |
|---|---|---|---|
| 每个函数分别判断 bool | 局部改动直观 | 若所有 Get、生命周期和维护函数都复制两套流程，会散落大量分支 | 只在策略确有差异的位置判断，不全面铺开 |
| 两个 owning backend 直接改名为 primary/secondary 并互换 | 普通双写的先后顺序容易表达 | 物理能力与逻辑角色并不对称；恢复和 Sync 仍必须找到 Redis；旧模式读也不是 primary-first | 不做全局互换，借用“运行时权威来源”的逻辑概念 |
| 派生 MetaStorageBackendManager 子类 | 可以隔离不同写入实现 | 当前 manager 非多态：方法和析构非 virtual，状态/helper 私有；需改构造链路与可见性，且 Recover/GC/Sync 仍有交叉差异 | V1 不采用，避免扩展继承层级和复制流程 |
| 保留物理 backend + 单个 WriteRoute 工厂 + 通用条件写 | 保留所有权、类型能力和原恢复链路，复用同一普通双写流程 | 需提升条件写接口，并让 async Redis 在原入队循环中筛选 | **采用** |

选择最后一种。Primary/Secondary 只表达普通写的第一、第二阶段，不改 backend 所有权，不派生 manager，不新增包装类、运行时分发表或缓存指针。

### 4.2 为什么不能只交换 primary/secondary

当前成员类型不同：

```cpp
std::unique_ptr<MetaStorageBackend> persistent_backend_;
std::unique_ptr<MetaCacheBaseBackend> cache_backend_;
```

当前 `MetaCacheBaseBackend` 独有按前序错误码过滤写入、`PutIfAbsent`、内存占用等能力。只交换指针不足以让 Redis 接收前序结果；将所有 cache 能力上移或让 Redis 继承 cache 抽象也不合适。本方案只提升通用条件写，保留两个成员的现有类型。

更重要的是，两种模式不只是写入顺序不同：

| 职责 | 原 cached 模式 | 新模式 Recover | 新模式 Running |
|---|---|---|---|
| 普通写入顺序 | Redis → 条件写 local | Redis → 条件写 local | local → 尽力备份 Redis |
| 返回值 | 保留 persistent 失败，也反映 local 写入失败 | 与原模式相同 | 返回 local 结果，不合并备份错误 |
| 普通查询首选 | local | local 缺失时回源 Redis | local |
| maintenance 权威来源 | persistent，保留现有双层保护 | 沿用恢复期延期和读取语义 | local |
| 启动恢复方向 | Redis → local | Redis → local | 已完成 |
| 物理删除前 Sync | Redis | Redis | Redis |
| 内存统计与 LRU | local | local | local |

因此，不能把所有 `persistent_backend_` 调用机械替换为 `primary_backend_`；否则新模式的 Recover、GetMetaData、Sync 会错误地转向 local。也不能统一为“返回 primary 结果”，这会改变原模式对 local 失败的处理。

### 4.3 访问器与角色选择

保留两个 owning 成员和不可变配置 `memory_primary_`。调用点自行决定是否 local-first，再通过唯一的 `GetWriteRoute(bool)` 形成该次操作不可变的角色选择：

```cpp
WriteRoute GetWriteRoute(bool local_primary) noexcept {
    return {
        local_primary ? *cache_backend_ : *persistent_backend_,
        local_primary ? persistent_backend_.get() : cache_backend_.get(),
        local_primary,
    };
}
```

普通写调用点传入 `memory_primary_ && recover_state_ == RecoverState::kRunning`；maintenance 根据自身前置条件传入 `memory_primary_`。即使恢复状态在调用中途切换，同一次操作也按已经捕获的 route 完成。

单 backend 的 Secondary 为 nullptr。初始化已保证新模式有 local + async Redis，访问器不重新校验 URI、组合或指针关系，也不 dynamic_cast。RecoverState 继续用于回源、tombstone 生命周期及原 maintenance 延期规则，不能机械删除全部阶段判断。

Open、Recover、GetMetaData 和 Sync 仍直接访问 `persistent_backend_`；内存/LRU 统计仍直接访问 `cache_backend_`，不临时改成员指针或将主次角色反复切换。

### 4.4 将已有条件写提升到公共接口

将以下带 `previous_error_codes` 的重载从 `MetaCacheBaseBackend` 提升至 `MetaStorageBackend`，保留原参数顺序、批量接口和逐 key 返回形式：

| 重载 | local | async Redis |
|---|---|---|
| Put / Upsert | 保留现有实现，只执行前序 EC_OK 项 | 仅将前序 EC_OK 项纳入原入队分组 |
| Delete / DeleteLocations | 保留原普通条件删除行为，不扩大旧模式授权条件 | EC_OK / EC_NOENT 可提交幂等删除，其他错误跳过 |
| DeleteLocationsForMaintenance | 保留现有 no-touch override，接受 EC_OK / EC_NOENT | 基类适配到带前序结果的 DeleteLocations，无需复制实现 |

PutIfAbsent（含条件重载）、全量 no-touch cache 读取、GetMemUsage、GetOldestAccessTime 等仍留在 cache 抽象。普通与条件重载须正确保留 `using` / `override`，避免 C++ 同名重载被隐藏。

普通 Delete 的旧 local 实现只接受 EC_OK，而新 Redis 备份需要接受 EC_NOENT；这是两种第二阶段的既定幂等语义，不通过额外 `allow_noent` 公共参数统一。接口注明 NOENT 不属于必须传播的硬错误，具体是否执行幂等删除由对应重载约定。maintenance 的原结果归并、整 key 删除 gate 不顺带修改。

为避免让同步 Redis、dummy 及所有现有测试替身都被迫实现未使用的能力，四个通用条件写在基类提供返回 EC_UNIMPLEMENTED 的默认实现；实际可被选为 Secondary 的 local/async Redis 均 override。组合可达性由现有 Init 限制保证，不增加请求时 capability 判断，也不提供逐 key 同步调用或通用压缩批次的默认实现。

### 4.5 普通双写只保留一条流程

以 Put 为例，以下为流程示意；原入参整理、结果契约检查及 local 耗时统计仍在相应位置执行：

```cpp
const auto route = GetWriteRoute(
    memory_primary_ && recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning);
auto primary_results = route.primary.Put(ctx, keys, locations, properties);
// 在传递给条件写前，检查 primary_results 与 keys 的位置契约。
if (!route.secondary) {
    return primary_results;
}
auto secondary_results = route.secondary->Put(ctx, keys, locations, properties, primary_results);
if (route.local_primary) {
    return primary_results;
}
// 原模式消费并检查 secondary_results，保留 local 失败的返回语义。
return secondary_results;
```

Upsert、整 key Delete 和部分 Delete 采用同样结构。普通 Upsert 保留必要的写前补齐；maintenance 继续复用候选分组、no-touch 删除和空 key 计数，不复制整个函数。

删除 manager 中的 `BackupSuccessful`、各调用点的备份 lambda 和压缩列 tuple。变量/日志中的 `persistent_results`、`cache results` 按实际角色更名为 primary/secondary，避免 local 第一阶段失败仍报成 Redis 失败。`cache_backend_*_time_us` 仍测量物理 local 调用：新模式在第一阶段计时，旧模式在第二阶段；不能仅因 Secondary 改为 Redis 就把备份入队时间计入 cache 指标。

新模式不消费第二阶段结果来决定业务成功：备份返回错误或畸形长度均不能回退 local、扣回计数或触发业务失败。异步 backend 在自己的提交边界统计被丢弃的备份，manager 不再重复统计或遍历备份错误；第一阶段返回形状仍必须在条件写前验证。其他已由上层校验的 key/location/property 维度不在条件写和分组循环中重复检查。

### 4.6 筛选合入 Redis 原有分组循环

复用 `EnqueueWriteOp`，为其增加可选的前序结果引用（实现可用仅在调用期间有效的指针）；无条件入口不构造全 EC_OK 数组。是否为非等待备份由调用角色自然区分：Recover 主写使用无条件入口，Running Secondary 使用带前序结果的条件入口。两类入口共用原 WriteOp 构造、key hash 分组、Enqueue 和 consumer：

1. 初始化逐 key 结果：有前序结果则保留其错误码；无前序结果沿用 EC_OK。
2. 在现有 `queue_to_indices` 循环中，按操作类型跳过不被授权的项；仅入组项参与队列容量、失败和业务备份统计。
3. 复用原 sub_op 构建循环，按原索引取 keys、locations、properties 或 location IDs，保持列对齐及同 key 顺序；Recover 主写使用 `WaitAndReserve`，Running 条件备份使用 `TryReserve`。成功入队项返回 EC_OK，拒绝项返回原入队错误。
4. 前序全部失败时不创建队列消息、不调用 Enqueue。local 失败本就不应持久化，不能算作“丢备份”。
5. Redis 返回错误、容量不足和 Sync 超时继续通过原错误码处理；内存分配、序列化等非预期 C++ 异常遵循 fail-fast，由 `noexcept` 写入口或 consumer 线程触发进程终止，不在 async backend 内吞掉异常后继续运行。

前序结果不进入 QueueItem，也不由 consumer 引用。V1 保留现有 owning WriteOp，不引入 borrowed-view 对象或新的生命周期协议；只把被授权项复制到按队列持有的 WriteOp，逐 key 返回数组仍需独立构造。收益是移除 manager 的额外压缩批次、重复遍历和分散备份分支，不是把 `BackupSuccessful` 原样下移到基类。

## 5. 普通写入与返回语义

### 5.1 写入顺序

Recover 阶段保持原链路：

```text
MetaIndexer 原有分批和 shard lock
  → Redis 主写按原策略等待容量并入队
  → local 条件写仅处理 Redis 接受的项
  → 返回 local 条件写结果
```

Running 阶段切换为：

```text
MetaIndexer 原有分批和 shard lock
  → local 提交，得到逐 key 结果
  → route.secondary 条件写接收原批次与 local 结果
  → async Redis 在原 key hash 分组时筛选 → queue.TryReserve → 构造并发布
      ├─ 成功：原 consumer 后续执行
      └─ 失败：记录丢弃数量，不改 local 结果
  → 原有结果处理及计数更新
```

| 操作 | 新模式备份条件 |
|---|---|
| Put / Upsert，包括 location/property 更新 | 仅 local EC_OK 项；分别保持原全量替换/局部合并语义 |
| 整 key Delete | EC_OK 或 EC_NOENT 均可提交幂等 DEL，返回值仍取 local |
| 指定 location 删除 | local EC_OK / EC_NOENT 可提交幂等 HDEL；空 key 回收复用原流程提交 DEL |
| maintenance 删除 | local 决定整 key/部分删除，使用 no-touch 接口；不复制 GC 算法 |

manager 将原批次及第一阶段结果交给 Secondary 条件写，不再压缩副本。Recover 的 Secondary 是 local，完全复用原条件写；Running 的 Secondary 是 async Redis，在原分组循环中筛选。Running 下 local 容量不足等失败项不能写入 Redis，备份失败不触发 local 回滚、业务写失败或重复计数。删除中的 EC_NOENT 表示幂等收敛，不属于容量/执行失败，不能与新增失败混为一谈。

local 的 `strict_capacity_limit`、`no_evict_on_insert`、上层配额不变；不能为了写入继续成功而静默淘汰权威索引。

### 5.2 顺序与生命周期

- 分片锁继续覆盖 local 与 Secondary 入队，避免同 key 操作反序；物理删除的 Sync 保持在锁外。
- 复用同 key 固定队列、单 consumer FIFO，不新增全局双写锁或 per-key mutex。
- 队列持有独立容器和足够生命周期的不可变 location 引用，不保留请求栈、RequestContext 或 local entry 裸指针；保持 consumer 侧 Redis 编码。
- 只丢弃尚未入队的新备份，不从队列中间移除、合并操作；失败旧批次不能重排到新操作后面。
- 保持现有 CAS、expected location value、EventReport version/lifecycle 保护及 Instance 隔离，不将新模式冒充为纯 local 以绕过备份快路径。

## 6. 复用队列 key 容量

### 6.1 构造 payload 前预留容量

改造前的 `WaitForCapacity` 只是检查，`Push` 不检查容量，两者之间没有占位；并发生产者可以同时通过检查。原实现还允许空队列接收超限 item。因此，直接设等待时间为 0 只能保留软阈值，不能保证严格不超限。

统一使用 `MpscWriteQueue::TryReserve` / `WaitAndReserve`，在复制 payload 和构造队列节点前完成 key 容量预留：

1. 复用原分组结果得到每组 key 数，用 CAS 原子占用 key 容量；容量不足时不复制 properties、locations 或 location IDs。
2. 预留成功后继续复用原 owning `WriteOp`、节点发布和 FIFO consumer；非预期分配异常保持 fail-fast，不转换成业务错误。
3. Running 条件备份容量不足立即返回；Recover/原模式使用同一原子预留并等待原 `async_enqueue_timeout_ms`，继续允许空队列接收 key 数超限的单个 item。
4. key 容量在 consumer 出队时释放。Sync barrier 对齐原实现使用无条件 `Push`，不参与普通备份容量；该控制消息的 item 数上限属于独立优化。

普通 WriteOp 按 key 操作数计量。新模式下的辅助 metadata 写占一个 key 容量单位并使用同一套容量预留；业务 key 指标与辅助消息数量分开核算，不把 metadata 或 barrier 计为成功备份的业务 key。

Running 阶段任一容量满了只停止接收备份，不停止 local 写入。预留失败不构造完整备份，也不转存到另一个无界容器；Recover 阶段等待失败按原主写错误返回，local 不提交对应项。

### 6.2 key 容量不等于 RSS 硬上限

`async_max_size` 只限制每队列 key 操作数，同一个 key 的 property/location payload 大小可能不同，consumer Pop 后的 in-flight 批次也不再计入队列容量。

本方案不增加字节估算、in-flight 字节计数或新的 URI 参数，避免每次写入额外遍历 payload，并保持队列准入和释放逻辑简单。生产内存仍通过 `async_max_size`、`async_max_batch`、实际 value 上限和 Redis 故障压测共同约束；不能把 key 有界队列宣传成进程 RSS 硬上限。

### 6.3 建连、失败和关闭

保留 Redis Open、读池初始化、网络超时及现有有限重试。运行时故障由原 consumer 处理，不新增重连线程或无限重试。重新连通后继续后续操作，但已丢失操作不会自动补回。

关闭沿用停止接入、排空在途调用、停止 consumer、drain、join、资源清理的生命周期。队列内容不依赖 local entry 存活，从而兼容 manager 先关 cache、后关 persistent 的顺序。保留 async_drain_ms；它不包含任意长的额外等待，也不是绝对 Close 上限，已开始的 Redis 调用仍受网络超时/有限重试约束。drain 丢弃必须可观测。

## 7. 启动恢复、运行时查询和 GC

### 7.1 保留恢复，但区分 Open 成功和恢复完成

现有 Open 启动 `AsyncRecoverTask` 后返回；`CacheManager::DoRecover` 也可能在部分失败后启动重试并返回成功。不能把这两个返回值视为所有 Instance 的内存已完整恢复。

新旧模式均保留 Open 启动 `AsyncRecoverTask` 后返回的原流程。Indexer 完成 Open 和辅助计数恢复后即可发布，不必等待 SCAN/Get/Backfill 全量结束。恢复失败沿用已有有限重试、停留 kRecover 和请求回源行为，不新增就绪门控、故障接管或重新恢复机制。

`MetaIndexerManager` 和 `MetaSearcherManager` 恢复原创建/查找/清理流程，不增加 initializing 表、条件变量或 cleanup epoch。原 Init 内短期 Redis Open/GetMetaData 仍处于既有锁范围；全量恢复在线程中执行，不在全局表锁内等待扫描完成。锁外初始化可作为独立优化，不与本配置绑定。

新模式 Recover 的普通写完整沿用 persistent-first：Redis 主写未接受时不修改 local，成功项才条件写 local。Upsert/部分删除仍复用 `EnsureKeyInCache` 补齐未恢复 key 的旧字段；后台 SCAN/Get 继续通过 `PutIfAbsent` 回填；整 key 删除继续用既有 `deleted_keys_` 防止迟到回填复活。普通读沿用 Recover 回源，不增加 `BlockedRecoverFallbacks` 或逐 API 过滤；并发读可能短暂看到旧数据，符合既定的最终一致性要求。

恢复线程完成最后一批回填后，以 release store 发布 Running；每次普通写只捕获一次阶段，因此跨越切换点的在途 Recover 写仍按 persistent-first 完成，之后的新写才使用 local-first。这样无需为前台回源新增 tombstone 判定、dirty-key 表或阶段切换协议。

不更改原恢复协议，不给 Recover 添加第二种降级状态，不保证 Redis 故障时未恢复 key 可写。Recover 中 Redis 已接受而 local 因容量不足失败属于 local 容量配置错误，沿原错误和 Reconcile 链路处理；需要扩容，不能把该状态当作可持续降级运行。未完整恢复时不得将 RecoverState 标成 Running。

辅助计数恢复失败时释放 backend，防止失败对象析构时 PersistMetaData 覆盖恢复基线。其余生命周期沿原实现，不增加新的发布协议。

### 7.2 普通读取与 maintenance

Running 后普通 Get/Exists/List/采样沿用现有 cache-only 路径，local 缺失不回源；两种模式 Recover 期间均保留原回源、EnsureKeyInCache 和后台 tombstone。Recover 的枚举仍不是一致性快照，不据此推断已恢复完整索引。

maintenance 的差异集中在 manager：

- `GetLocationsFromPrimary` 在新模式始终读 local；原模式仍读 persistent。Recover 时尚未回填的 key 可以暂不纳入 maintenance 候选，不为了维护读取落后备份来覆盖 local。
- 新模式全量 location 读取必须 no-touch；可在 cache 层补窄接口，复用 local 已有 `GetForOneKeyForMaintenance`，不为所有 backend 新建一套主备 API。
- 保留原 `refresh_cache_from_persistent` 参数和 `RefreshCacheFromPersistent` 命名；RMW 在 shard lock 内对新模式整体跳过覆盖式刷新，不只跳过 Running，不产生无意义的成功结果数组。旧模式继续原刷新流程；普通部分写需要的缺失 key 补齐见 7.1，不能用覆盖式 refresh 替代。
- 新模式 `GetLocationsForMaintenance` 只依据 local，跳过为读取 persistent 权威视图而设置的读前 Sync；`DeleteLocationsForMaintenance` 继续保留原 Recover 删除延期，Running 共享候选分组、精确值 CAS、no-touch 删除及空 key 回收，不比较迟到备份。该延期不阻断普通写删。
- Running 的 metadata-only 回收可继续执行；涉及物理数据删除的任务仍须通过第 8 节的锁外 `Sync`，不能把两者统称为“Redis 故障时 GC 全部可用”。

## 8. 对齐原物理删除 Sync

### 8.1 锁外 Sync

物理删除沿用当前分支的锁外 Sync，不增加通用 `confirm_persistence`、`CanSync` 和队列级粘性失败状态；但 `DELETING` 写入必须满足一次非阻塞的 Secondary 准入：

```text
shard lock 内读取/CAS local DELETING，并通过现有条件 Upsert 尝试 Redis 备份入队
→ 队列未准入的项返回失败，不进入物理删除任务
→ 释放 shard lock
→ SchedulePlanExecutor 仅对已准入项调用真实 Sync(keys)
→ Sync 成功才调度物理删除
```

MetaIndexer 在构造 Upsert 批次的既有 location 遍历中标记 `DELETING` key，BackendManager 直接复用该稀疏标记和原返回码，不重复遍历 location，也不向 MetaSearcher/RMW 调用链新增布尔参数。准入仍使用现有 `TryReserve`，不在 MetaIndexer shard mutex 内等待容量；普通写继续以 local 结果为准。Redis 请求失败继续依赖 RedisClient 的有限重试，Sync 超时或失败仍阻止本次物理删除。

该选择不扩展异步队列一致性协议。现有 barrier 只等待它之前已入队的消息被 consumer 处理，不提供逐 WriteOp 回执；如果某个写在 barrier 入队前已被单独消费并最终失败，后续 barrier 不能精确关联该失败。V1 先关闭队列满导致 `DELETING` 备份根本未入队却被后续空 barrier 误判成功的确定性漏洞；后续若实际故障测试证明需要更强确认，再单独设计 per-write completion/sequence，不引入队列级永久熔断。

同样对齐原实现：`Sync` 最终失败只阻止本次物理删除，不回滚已经提交的 local `DELETING`；相同删除请求会跳过该状态，本次不额外增加自动重试/补偿状态机。RedisClient 的有限重试用于降低该情况的发生概率，但不改变这个失败终态。上线故障压测若确认需要自动恢复，应在 executor 层单独设计可重试任务，而不是把网络等待重新放回 shard lock。

### 8.2 哪些屏障保留，哪些可省略

| 调用目的 | 新模式处理 |
|---|---|
| maintenance 读前等待旧写，以便读取 persistent 权威视图 | 读 local 已受 shard lock 保护，可跳过 Redis 等待 |
| maintenance 删除后的 local 可见性 | 沿用 RequiresMaintenancePostDeleteSync 的已有 cached 判断，无需另加相同判断 |
| executor 允许物理删除 | 新旧模式均保留 executor 原位置的锁外真实 Sync |
| 写入回滚后允许释放 URI | 沿用原 Reconcile：实际删除引用的项须 Sync；本次失败新增的缺失项不额外补偿 |

不再增加原稿的通用 `SyncForServing` 并替换全部调用。读前策略收敛为 manager 的 `SyncBeforeMaintenanceRead(keys)` 薄入口：旧模式委托原 Sync，新模式因 maintenance 读 local 而跳过；它不暴露给物理删除流程。

### 8.3 Reconcile 恢复原链路，不为容量配置错误增加补偿

此前 memory_primary 专属修改针对“Recover Redis 写成功、local 因容量不足失败，跨到 Running 才回滚”。本方案明确 Recover 保持原 persistent-first；该时序与原 cached 模式相同，local 容量不足属于配置错误，复用原错误和 Reconcile 语义，不为 memory-primary 再增加一套补偿。

因此恢复 Reconcile 原来的 BatchDeleteLocations 调用、成功删除项 Sync 与 EC_NOENT 分支，撤销以下新增逻辑：

- `memory_primary` 捕获和将读取 EC_NOENT 改成 EC_OK 的专属 modifier；
- 为本次失败新增的缺失 location 强行发送 HDEL；
- 将上述 EC_NOENT 项纳入 Sync，以及对其释放 URI 额外要求 Sync 成功。

不新增 `delete_missing` 参数、写入阶段/来源字段、按容量错误分类的补偿分支。不单独保留为该改造引入的私有删除实现，优先恢复原函数链路。

撤销补偿不等于撤销所有回滚：生成 ID 前就失败的项仍按原方式直接清理；失败/未知项仍由原 Reconcile 查询并处理，不能仅看到新增错误码就跳过它。实际删除到引用的项仍需 Sync 才释放 URI；批次整体失败时，已经成功的成员仍走原 DELETING → Sync → 物理删除流水线。单纯 Redis 备份失败不会让普通新增返回失败，因此不会触发该业务回滚。

该结论只针对 Reconcile 已筛选出的本次失败新增及其新生成 ID，不推广为“任意 local NOENT 都证明 Redis 无引用”。普通显式删除的幂等 EC_NOENT 备份语义仍保留。Recover 直接复用原 persistent-first 和回滚链路，Running 的 local-first 失败项不会形成 Redis 新引用，两阶段均无需专属补偿。

新模式删除准入 Sync 失败时不执行物理删除；回滚 Sync 失败时保留 URI。不能把失败当成回收成功或扣减实际用量。pending credit 继续按既有成功/确定失败/结果未知的终态规则处理：确定未开始物理删除的失败可以释放任务占位，但不代表释放了物理容量。Redis 故障时允许物理回收暂不可用，不能承诺无限持续写入。

## 9. 辅助元数据、备份观测与恢复边界

### 9.1 辅助元数据仍从 Redis 恢复

保留 GetMetaData/RecoverMetaData，不在新模式中委托给 local no-op 或将旧计数强制归零。

运行时 PersistMetaData 会被容量选择等热路径调用，故新模式 PutMetaData 不能同步操作 Redis。保留当前工作区已有的 metadata 写类型、consumer 和 Redis Set 编码；使用真实 metadata_key，不伪装成某个普通 block key。同一 metadata_key 固定一个队列，按原调用顺序有界入队，不新增线程或单独队列。原模式 PutMetaData 保持原行为。

辅助 metadata 消息按第 6 节占一个容量单位；入队/执行失败计入备份缺失。消费端区分其统计与普通业务 key，不能让零 key 写被 EnqueueWriteOp 的空批次分支直接忽略。

这仍是近似的周期性计数备份，不是跨队列一致性快照。普通数据和计数可能因丢备份出现偏差，恢复后配额/回收会受影响；不可宣称已实现精确计数恢复。需要精确基线时执行受控校准，不顺带增加跨队列事务协议。

### 9.2 最小可观测性

复用 AsyncWriteStats 和现有队列/pipeline 指标，补充 `async_dropped_key_count`、`async_dropped_metadata_count`，按采集周期读取并清零。不增加 LockMetrics、逐 key 日志、dirty key 集合或新的全局统计锁。`Sync` 超时或失败通过调用结果反馈，不作为业务丢写或 metadata 丢写计数。

业务成功 QPS、Redis 实际完成量、备份缺失和物理回收失败必须区分。未观察到本周期失败也不证明历史备份完整。

### 9.3 重启与备份校准

保留自动 Open/Recover，故障重启允许恢复较旧的 Redis 状态。普通新增丢失可能造成 miss，逻辑删除丢失可能恢复旧元数据；外部回收、EventReport 生命周期等仍依赖原有效性保护。不能承诺仅靠物理删除屏障保证所有逻辑删除不回退。

物理删除前的 `Sync` 保护的是：本次 barrier 失败时 KVCM 不主动删除物理数据。它不等价于逐写入确认、Redis 的磁盘/副本持久化保证，也不替代外部存储的存在性校验；其既有边界见 8.1。

Redis 重新连通不会自动补齐丢写。需要恢复精确备份或关闭开关时，采用受控校准：

1. local 仍存活时，停止该 Instance 的请求修改和 GC/reclaimer 等后台修改；排空或终止旧 consumer，确保旧任务不会覆盖校准结果。
2. 使用稳定的 local 全量快照，按原完整 Put 语义替换 Redis 对应 key；删除该 Instance 前缀中 local 不再存在的多余 key，不能仅 Upsert，也不能清理其他 Instance。
3. 从同一快照重建辅助计数，逐批检查写入结果并校验数据；连接恢复、空队列或单次 Sync 成功不能代替这一步。
4. 保持停写直到旧 Indexer 完全退出、新配置 Indexer 完成 Redis Recover，再开放流量。部署沿用现有进程/Leader 生命周期；当前 DeleteMetaIndexer 未实现，不能假定已有在线单实例切换 API。

当前没有自动完成上述全量校准的完整工具。本次不建设在线 resync/WAL；没有受控校准能力时，必须接受从旧备份恢复。local 已丢失时无法补回未备份更新，可从业务来源重建，或接受 Redis 中的旧状态。

## 10. 测试、发布与改动控制

### 10.1 必测范围

优先扩展现有测试夹具，不建设平行 mock backend 框架：

| 测试层 | 重点用例 |
|---|---|
| config / service / kvcm_ops | 默认关闭、JSON/proto 双向透传、GET→PUT 不丢字段、合法/非法组合、persistent 子配置收到开关 |
| 公共条件写接口 | 通过 MetaStorageBackend 指针调度到 local/async Redis；普通重载不被隐藏；未支持 backend 的默认条件写不产生写入；cache 专属能力不迁移 |
| 普通写入 | Recover 覆盖 persistent-first 及 Redis 失败不提交 local；Running 覆盖四类 local-first 写及 property 更新、混合 EC_OK/EC_NOSPC、全失败和空批次；local 失败项不备份，非 `DELETING` 备份失败不改 local 结果/计数 |
| Secondary 筛选与返回 | 非连续成功索引跨多个队列、locations/properties/IDs 对齐；容量按入组项计量；删除 NOENT 授权与旧 local gate 不变；新模式普通项返回 primary、`DELETING` 项合并 Secondary 准入错误，旧模式返回 secondary，单 backend 不调用第二阶段；cache 指标仍只统计 local |
| queue / async Redis | Recover 无条件主写使用 WaitAndReserve；Running 条件备份的 TryReserve 不超限、超大 item 拒绝、满队列立即失败；key 预留在 payload 构造前完成并在出队时释放；metadata 使用同一 key 容量准入，Sync barrier 使用零容量 `PushBarrier`；非预期异常 fail-fast |
| 顺序与生命周期 | 同 key Put→Upsert→Delete→重建顺序；无额外全局锁；Close 不出现悬空引用/泄漏；drain 有界且丢弃可见 |
| Open / Recover | Open/辅助计数失败保持 Init 失败；后台回填期间可读写；Recover Redis 主写失败时 local 不提交并保留原反压；切入 Running 后新调用改为 local-first；恢复失败不进入 Running；原 Manager 创建回归 |
| Recover 并发 | 暂停 Redis 主写并在回调中切换 Running，验证在途调用仍完成 persistent-first；保留 PutIfAbsent、EnsureKeyInCache 和后台 tombstone；不增加前台读过滤 |
| maintenance | local no-touch 权威读和精确 CAS、空 key 回收；旧 Redis 值不覆盖 local；metadata-only 操作不因 Redis 等待阻塞 |
| 物理删除 Sync | CAS 在 shard lock 内完成；`DELETING` 备份队列准入失败的项不进入物理删除；真实 Sync 在 executor 锁外执行且失败时不调度物理删除；验证 barrier 准入和超时行为，不把它声明为逐写入确认 |
| 回滚 | Recover 复用原 persistent-first 和 Reconcile；Running 的 local 失败新增不产生 Put 备份；Reconcile 不为缺失新 ID 增加专属 HDEL/Sync 分支；实际删到引用仍需锁外 Sync |
| 辅助元数据与恢复 | 启动读 Redis 计数；运行时 PutMetaData 不同步访问 Redis；容量限制、序列化、失败可见；测试普通更新/逻辑删除回退和计数偏差的已声明边界 |
| 隔离与发布 | 不同 Instance 数据与恢复互不串用；校准只改目标前缀；停止旧 writer 后再接管；旧程序/工具不能在开启状态下接管或改写配置 |
| 原模式回归 | 默认/false 时 persistent-first、条件 cache 写、Recover、Sync、辅助 metadata、pure local 快路径和原队列策略不变 |

使用 fake client/latch 控制竞态，不靠长 sleep 判断。真实 Redis 测试覆盖停服、限速、断连和恢复，持续故障超过队列填满时间；并发/Lifetime 用例在可用环境补跑 ASan/TSan。

### 10.2 性能验证

同一基线、同资源负载比较原模式、新模式、纯 local 参考组；另对比本次统一前的 BackupSuccessful 实现与统一后的 Secondary 条件写。覆盖健康/慢速/故障 Redis、小大批次、多 location、热点 key，以及 BlockAdd/BlockDelete、普通写删、metadata-only GC 和物理回收并发；专门增加 local 全成功、部分成功和全失败批次，观察过滤/复制/分配成本。

记录业务成功 QPS、P50/P99、CPU、RSS、local/queue 用量、Redis 完成/丢弃量、物理回收失败与容量变化。锁等待/持有时间通过压测 profiler 采样，不新增生产 LockMetrics。

memory-primary 的主要收益来自消除普通备份入队在 shard lock 内的容量等待；容量已满时又可在复制完整 payload 前拒绝，降低故障态的分配和拷贝开销。健康无积压时原 async 模式本就不等 Redis 网络写，全成功批次可能性能相近，不能预先承诺吞吐提高多少。故障时验证已恢复数据的普通路径不等待备份；Recover 的未恢复 key 仍可受读取 Redis 影响，锁外 `Sync` 失败及容量不足是已声明边界。

### 10.3 实施与发布顺序

本次在既有队列、consumer 和锁外 `Sync` 实现上增量收敛，不重做这些链路：

| 文件/层 | 本次修改边界 |
|---|---|
| meta_storage_backend.h / meta_cache_base_backend.h | 提升四个通用条件写与 maintenance 条件删除接口，保留 cache 专属接口；不让 Redis 继承 cache |
| meta_async_redis_backend.h / .cc | 条件写重载及原入队分组筛选；构造 payload 前预留 key 容量；复用原 consumer/统计 |
| meta_local_backend.h / .cc | 尽量只调整继承/重载可见性；原条件写、no-touch 和容量逻辑不变 |
| meta_storage_backend_manager.h / .cc | 用唯一的 GetWriteRoute(bool) 统一主次 Backend 选择；普通写只在 Running 选择 local primary，maintenance 显式传入自身策略；删除 BackupSuccessful；仅 `DELETING` Upsert 返回 Secondary 准入结果；保留原恢复链路及准确的 local 计时 |
| meta_indexer.cc / manager 相关调用点 | 恢复原 `MetaIndexer::Sync` 和 executor 锁外调用；保留原分片锁、计数和 Manager 生命周期 |
| meta_searcher.cc / 相关测试 | 撤销 memory-primary 专属 Reconcile 补偿及通用持久化确认参数，复用原调用链；保留批次整体失败的回滚与物理删除 Sync |

顺序为“接口上移及 async 条件写测试 → Recover 保持原顺序、Running 启用 local-first → maintenance 数据源/刷新条件适配 → 撤销 Reconcile 专属补偿 → 全链路回归与故障压测”。全部闭环前不得启用生产开关。所有可能接管的服务及运维工具升级后再开启，模式切换必须结束旧 writer。

不增加配置字段、backend 类型、manager 子类、新线程、外部保留槽位或业务锁；不复制恢复/GC 算法，不顺带重构旧模式。默认 false、pure local 快路径、原锁外 `Sync` 及 Instance 隔离均需回归。复用已有测试夹具与 mock，不因抽象上移建立另一套测试框架。

实现不改变模块依赖方向，复用原架构图；同步更新 [模块架构与关联关系](module_architecture.md) 的元数据运行路径和 [配置指南](../configuration.md)。

### 10.4 验证记录与本次验收状态

本次已实现 `WriteRoute` 条件双写、Recover persistent-first / Running local-first 阶段切换、一次性路由快照，以及构造前 key 容量预留；并撤销前台恢复读过滤和 Reconcile 的 memory-primary 专属补偿。真实故障压测与系统级动态检查仍是上线前事项。

- 相关 22 个 Bazel 目标全部通过，覆盖 meta queue/async Redis/backend manager/local/redis/dummy/indexer/indexer manager，manager searcher/executor/GC/reclaimer/migration/cache manager/metrics recorder，配置、proto codec 与 metrics reporter。
- 定向用例覆盖 Recover 主写失败不提交 local、Running local-first、跨阶段调用保持已捕获顺序、Primary 保留反压、Secondary 满队列立即失败、executor 锁外 Sync 及原 Reconcile 路径。
- BackendManager、async Redis、MetaSearcher、SchedulePlanExecutor 的 MemoryPrimary 用例各重复 20 轮通过。
- kvcm_ops 实例组测试 39 个用例通过，含配置默认值、CLI 及 GET→编辑→PUT 保留开关。验证命令：`/usr/bin/python3 -B -m unittest discover -s package/kvcm_ops/test -p 'instance_group*test.py'`。
- `git diff --check` 通过且无未解决冲突。
- 尚未进行真实 Redis 停服/限速故障压测、完整系统 ASan/TSan 或生产吞吐对比；单元测试中的满队列非等待断言不能代替生产 QPS/P99 结论。上线前仍按 10.2 灰度验证，默认不开启配置。
