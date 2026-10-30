# Recover 期间允许 EventReport GC 清理

关联：[EventReport 后台 GC](event_report_background_gc.md)、[Meta 内存主存储与异步备份](meta_memory_primary_async_backup.md)。

## 1. 目标与方案

取消 cached metadata 在 Recover 期间对 EventReport maintenance 删除的整体拒绝，让已经进入内存的失效 metadata 可以及时清理。

**cached EventReport GC 只在原 shard 锁内校验实际内存目标，不读取 Redis 做双层比较，也不执行读前 Sync。** 内存 key 或目标缺失就跳过，不回源；整 key 与部分删除均根据内存 Location ID 决定。

**复用现有 `deleted_keys_`：整 key 删除登记 tombstone；部分 Location 删除保留内存 key。** 在线写入仍可重建 key；内存缺失且有删除标记时跳过 Redis 旧值补齐，直接以本次业务数据创建。不为等待 GC 的 Redis 删除增加在线 Sync，不新增删除集合、版本号、线程或重试队列。

**普通 GC 不再用 Redis 全值覆盖已有内存 key，也不因 Recover 整体暂停。** 复用锁内目标 Location 的 expected-value 校验；Recover 缺 key 时按删除标记过滤后完整补齐，Running 缺 key 或目标 Location 时直接跳过，不回源、不新增 Sync，详见 4.4。

本次只处理 GC 删除与本轮 recover 回填、在线回源补齐、普通 GC 强刷的竞争。两类 GC 的内存校验依赖正常业务成功更新 hot、恢复与补齐以完整 key 建立 hot、各元数据修改共用原 shard 锁，并要求 4.4 的普通 GC 不再覆盖已有 hot K。沿用 Reporter recovery grace、cleanup token/lease、完整 expected-value 校验和周期扫描；保留写入失败的返回及成功门控，不扩展为双写失败后的跨层补偿、Redis-only 数据修复、冷 key 扫描或跨重启删除持久性保证。模块职责和依赖关系不变。

## 2. 当前问题

改动前，[`DeleteLocationsForMaintenance()`](../../kv_cache_manager/meta/meta_storage_backend_manager.cc) 在 cached Recover 期间返回 `EC_OUT_OF_LIMIT`。扫描仍可继续，但删除要等恢复完成。

若直接移除这个 guard，会出现：

```text
前提：K 已通过在线写入、写前补齐等路径进入内存
recover：从 Redis 读到 K 的旧值，尚未回填
GC：     删除最后一个 Location，回收整个 K
recover：PutIfAbsent(K, 旧值) 成功，旧 metadata 复活
```

普通整 key `Delete()` 已登记 `deleted_keys_`，`BackfillKeysToCache()` 已在 `deleted_keys_mutex_` 下完成“检查标记 + PutIfAbsent”。但 maintenance 的整 key 分支直接调用底层 backend，没有登记标记。

因此需要接入已有机制，而不是新建另一套 tombstone。

此外，普通 GC 的覆盖式刷新也可能把 Redis 尚未执行删除的旧 Location 带回内存，这个交错不需要发生双写失败。普通 GC 使用 4.4 的目标条件校验，仅在 Recover 缺 key 时完整补齐，Running 缺失时直接跳过。

## 3. EventReport GC 删除规则

本节针对 cached EventReport 的 maintenance 删除；普通 GC 的规则见 4.4，single-backend 的保留行为见 4.1。这里的两层分别是 Redis（persistent）和内存（hot）：删除仍更新两层，但删除前校验只读 hot。一个 key K 对应一个 block 的元数据，下面可以有多个 Location，分别描述该 block 的缓存位置。

| 操作 | 以 `K = {L1, L2}` 为例 |
|---|---|
| 删除一个 Location | 删除 L1 后，K 仍存在，保留 L2 和其他属性 |
| 删除全部 Location，但保留 key | K 仍存在，只是 Location 列表为空 |
| 回收整个 key | K 的条目、所有 Location 和其他属性一起消失 |

对 EventReport GC，这些操作都只清理元数据，不直接删除 Reporter 上的实际 KVCache 数据。`PutIfAbsent` 判断的是 **K 是否存在**，所以空 Location 列表和整个 key 不存在，在恢复竞争中有不同含义。上表用于区分状态；正常清理最后一个 Location 时按 3.1 直接回收 K，不先留下空 key 再等待其他清理。

沿用 MetaIndexer 的 per-key shard lock 和原有 RMW。先完成 token 和 hot expected-value 校验；至少有一个目标通过校验，才用 hot Location ID 判断整 key 或部分删除，无有效目标则直接跳过。**hot 中仍有本次通过校验的删除目标之外的 Location，就保留 K，只删除已通过校验的目标；否则直接回收整个 K。** 同一个 key 下的其他 Location 称为“非目标 sibling”，包括本次校验未通过、必须保留的目标。

不读取 Redis Location ID 参与分组。例如此前删除 L2 已入队，hot 为 `{L1}`、Redis 暂时仍为 `{L1,L2}`；本次 L1 通过校验后可以直接删除整个 K。Redis 按同 key 顺序执行此前的删除与本次 DEL，不会保留 L2。该判断依赖正常写入成功反映到 hot 的前提，不用于修复双写失败造成的 Redis-only sibling。

| 场景 | 恢复期处理 |
|---|---|
| hot K 存在，且只剩本次通过校验的删除目标 | 删除 Redis 中的 K，登记 K 的 tombstone，再删除内存 K（见 3.1） |
| hot K 仍有非目标 Location | 删除两层已通过校验的目标，保留内存 K（见 3.2） |
| hot K 或目标 Location 缺失 | 读取阶段按 `EC_NOENT` 跳过，不查询 Redis、不修改任一层；同批其他目标继续（见 3.2） |
| 读取失败、值不匹配或 token 失效 | 对应目标沿用原错误/mismatch 处理（见 3.3） |

### 3.1 整 key 删除

例如 hot K 只有 L1，而 L1 已通过本次删除校验，就可以回收整个 K，无须等待 Redis 中此前的写入执行完成。回收后内存没有 K，单靠 `PutIfAbsent` 挡不住旧值重新插入，因此需要额外记住“K 已被主动删除”。

这个记号就是 key tombstone：执行 `deleted_keys_.insert(K)`，把 K 记录到现有集合，不保存旧 Location 内容。“接入 tombstone”就是让 maintenance 的整 key 删除也执行这一步；后台 recover 看到标记后跳过 K 的回填，在线写前读取和补齐也跳过已删除 K 的 Redis 旧内容，正常业务仍可重新创建 K（见 4.3）。

在现有 `whole_key_indexes` 分支中接入登记，保持原删除结果和计数语义。以下是 Recover 的 persistent-first 顺序；Running 按 4.1 保持原配置的写入顺序：

```text
持有原有 per-key shard lock
  → persistent Delete(K)
  → 对结果为 EC_OK / EC_NOENT 的 K：
      持有 deleted_keys_mutex_
      若当前仍为 Recover，执行 deleted_keys_.insert(K)
      释放 deleted_keys_mutex_
  → 按原成功门控删除 hot K（persistent NOENT 也要收敛 hot copy）
  → 按原结果更新计数并返回
```

`deleted_keys_mutex_` 只保护登记，不跨 Redis I/O 持锁。persistent 删除失败的 key 不登记、不删除 hot copy；结果数量异常沿用当前失败处理。若 persistent 删除已接受但 hot 删除失败，保留标记并返回原错误，由后续扫描重试。

为什么这个顺序足够：

- 回填先拿到锁：它先完成插入，随后 GC 登记并删除 hot copy。
- GC 先登记：回填在同一把锁下看到标记，跳过旧值。

不会出现“回填检查到无标记，GC 已经删完，回填才插入”的窗口，因为回填的检查与插入本来就在同一个临界区。

### 3.2 部分 Location 删除

本节始终只删除目标 Location，不回收整个 K，也不向 `deleted_keys_` 登记 K。保护来自内存 key 仍然存在。

**内存 K 存在时，可以删除 L1：**

```text
删除前：Redis K = {L1, L2}，内存 K = {L1, L2}
GC：    只删除两层的 L1，保留 K = {L2}
recover 即使持有旧的 {L1, L2}，PutIfAbsent 也因内存 K 已存在而跳过
```

通过 no-touch 读取 hot 的 Location ID，以返回码判断 key 是否存在：`EC_OK` 表示存在，即使 Location ID 列表为空；`EC_NOENT` 才表示缺失。检查和删除都在原有 shard lock 内，阻止在线写入/删除在两者之间改变 K；当前 local 配置 `no_evict_on_insert=true`，不会因其他 key 插入而自动淘汰 K。只有 hot 中仍有非目标 Location 时才进入部分删除；最后一个 Location 的删除走 3.1。底层 Location 擦除不自行回收 key，本方案也不单独扫描已有空 key。

**执行时内存 K 缺失，就暂时跳过这次 Location 删除。** 原因仍是旧值回填，而不是这次要删除整个 key：

```text
假如在内存 K 缺失时直接删除 L1：
① recover 已读到 Redis 的 K = {L1, L2}，尚未回填
② GC 只删除 Redis 中的 L1，Redis 变为 K = {L2}，内存仍没有 K
③ recover 执行 PutIfAbsent(K, {L1, L2}) 成功，L1 重新出现
```

GC 扫描后会异步提交删除任务。扫描只说明 K 当时在内存，任务排队期间其他删除操作可能已移除 K，所以执行时必须重新检查。cached 路径只读取实际 hot，K 或目标缺失时按 `EC_NOENT` 跳过，不再用 Redis 中仍存在的目标授权删除。该规则同时适用于 Recover 和 Running，不为 EventReport 清理主动回源补齐。

删除入口仍保留防御性准入：若绕过正常读取路径，最终检查发现 Recover 的 hot K 缺失，则在修改任一层前返回 `EC_OUT_OF_LIMIT`，不按 Redis 状态继续删除。正常 RMW 已在读取阶段跳过缺失目标，不以该防御分支作为常规路径。

正常 `EC_NOENT` 跳过不逐 key 报错，也不因此将 GC 任务记为失败，无需额外日志限频。正常 RMW 从 hot 读取到删除始终持有同一把 shard 锁，当前 local 配置不会自动淘汰 K，因此不会触发上述缺失防御分支。若生产路径意外返回 `EC_OUT_OF_LIMIT`，沿用普通错误日志，便于发现调用或锁约束被破坏的问题。

**后续如何处理：** 本次任务结束，不原地等待或保存“待删 L1”任务。K 经后台 recover、在线写入或写前补齐进入内存后，后续 GC 轮次重新判断 L1 是否仍符合清理条件，再按当前 hot 状态选择部分或整 key 删除。不必等待整个 recover 完成，也不在 K 刚进入内存时立即触发删除。

如果 K 一直没有进入内存，GC 不会为这次旧候选主动读取 Redis；K 也可能已经被其他操作整体删除。本方案允许其他满足条件的 key 继续清理，不承诺恢复期间清完所有 Redis metadata。

### 3.3 删除前的现有校验

GC 扫描得到的是候选，执行时还要重新确认当前能否删除。这些是已有保护，放开 recover guard 后继续沿用：

| 情况 | 具体含义与处理 |
|---|---|
| 读取失败 | 执行时目标读取失败或返回结果异常，无法确认当前状态；返回错误，不凭扫描时的旧结果直接删除 |
| expected-value 不匹配 | 同一个 Location ID 对应的元数据已经变化；返回 `EC_MISMATCH`，跳过该目标，避免误删被更新的记录 |
| cleanup token 失效 | 清理依据对应的 Reporter 状态已变化，例如扫描时已下线并满足清理期限，执行前重新注册上线；旧 token 不能继续授权删除，即使 Location 内容还未变化 |

cached 路径的 expected-value 比较读取实际 hot，不回源、不与 Redis 做第二次比较；single-backend 校验唯一后端。比较的是 **Location 的完整序列化值**，不是新增一个 Location 版本号：

```text
执行时 Location.ToJsonString() == 扫描时保存的 expected_location_value
```

比较字段包括 `id`、`status`、`type`、`spec_size`、`create_time` 和 `location_specs`（含 URI）。EventReport 的一个具体例子如下，V1、V2 是已有 generation 标记的示意名称：

```text
Reporter 已提交 V2，GC 扫描到 L1 的 URI 仍含 s_version=V1
执行前，增量 ADD 将同一个 L1 更新为 s_version=V2
Reporter 的 committed version 仍是 V2，cleanup token 可以仍有效
但 L1 的完整 JSON 已变化，expected-value 比较会拒绝这次旧删除
```

`s_version` 是已有的快照/增量 generation 标记，仅是整值比较的一部分；其他序列化字段变化也会导致不匹配。这里没有新增 Location 版本号。相关实现见 [`CacheLocation`](../../kv_cache_manager/meta/cache_location.h) 和 [`BatchDeleteLocations`](../../kv_cache_manager/manager/meta_searcher.cc)。

检查未通过的目标不据此执行删除，也不据此登记 tombstone。只有通过校验并进入 3.1 的整 key 删除、persistent 返回成功或已不存在时，才按恢复状态登记 K。本次跳过不代表永远不删除，后续扫描会按最新状态重新判断。

## 4. 放开 guard 必须同步处理的边界

### 4.1 恢复期路由与 Sync

**cached EventReport maintenance 在 Recover 和 Running 都只读取 hot，不执行读前 Sync，也不做双层比较。** 目标读取和整 key 判定均采用 no-touch hot 接口；hot K 或目标缺失则跳过，不回源。读取依据与写入顺序分开决定，不能因为读取 hot 就把 Recover 的写入切成 local-first。

Recover 未完成表示可能还有 key 尚未进入内存，不意味着已存在 K 只恢复了一部分 Location。完整回填与 `EnsureKeyInCache()` 建立 K，`PutIfAbsent` 不覆盖已有 K；正常业务写入与 GC 在同一 shard 锁内更新 hot。业务先更新目标时，GC 能从 hot 的完整 expected value 比较发现变化，即使 Redis 更新仍在排队，也无须等待它执行完。

每个 RMW batch 在取得 shard lock 后固定一次**写路由**，复用现有 `WriteRoute`：

```text
write_local_primary = memory_primary_ && 本批次开始时状态为 Running
```

| cached 阶段/模式 | 删除前校验 | 写入顺序 |
|---|---|---|
| Recover，包括 memory_primary=true | 只检查 hot，无读前 Sync | persistent-first：Redis 接受删除后登记整 key tombstone，再按成功门控删除 hot |
| Running，memory_primary=false | 只检查 hot，无读前 Sync | 沿用 persistent-first，不登记恢复期标记 |
| Running，memory_primary=true | 只检查 hot，无读前 Sync | 沿用 local-primary 和原备份语义 |

批次中途恢复完成，当前批次仍按已选写路由结束，下一批再切换；tombstone 登记时仍按 4.2 重读当前状态。hot 读取不依赖 `write_local_primary` 的取值，不把读取来源和写路由绑定为同一个开关，不引入新的路由框架或跨批次屏障。

同 key 写入保持既有队列顺序。此前某个 Location 的删除尚未落 Redis 时，hot 已反映这次删除；后续 GC 根据 hot 决定部分删除或 DEL K，不能因为 Redis 的旧 sibling 而把本可整 key 删除的操作降成部分删除、留下空 key。整 key tombstone 与保留部分删除的 hot K 分别阻止旧回填，4.4 禁止普通 GC 用 Redis 全值覆盖已有 hot，两者必须同时满足。

无 hot 副本的 single-backend 保持原行为：读取和判断唯一后端；single async Redis 仍在 shard 锁内执行读前及删后 Sync，同步后端沿用其原 Sync 实现。普通 GC 标记 `DELETING` 后、物理删除前已有的锁外 Sync 也保留。cached EventReport 本身不执行物理删除，不增加删后 Sync。

这里取消的是为了读取 Redis 做校验而等待队列的步骤，不取消 Redis 写入；同步 Redis 的删除仍可能等待网络返回，异步入队仍可能产生原有背压。原有写失败返回、成功门控和计数语义保持不变，不承诺消除所有 Redis 等待或修复双写失败留下的跨层差异。

### 4.2 tombstone 的登记与释放

新增登记逻辑在拿到 `deleted_keys_mutex_` 后重新读取当前恢复状态，不能仅依赖批次开始时的 Recover 判断。这样可以复用现有恢复结束顺序：“完成所有后台回填 → 发布 Running → 在同一把锁下清空集合”，避免清空后又按旧状态登记标记。

恢复失败仍停留在 Recover，沿用现有保留 tombstone 的行为。正常业务可以重新创建 K，标记阻止旧 Redis 内容参与恢复回填和在线补齐，不拒绝业务新写；新写不清除标记，已有内存新值始终正常使用。集合属于单个 Instance 的 backend manager，不跨 Instance 共享。

新增空间为本轮恢复期间 GC 整 key 删除的唯一 key 数量；不保存 Location 内容。长时间停留 Recover 时集合会积累，不能通过 TTL 或容量淘汰提前移除标记。本次沿用现有生命周期，不另建回收协议。

### 4.3 在线重建：跳过旧数据，不等待删除完成

只保护后台 recover 不够。`async_redis` 的删除成功可能仅表示 DEL 已入队：GC 删除内存 K 后，在线 Upsert 的 `EnsureKeyInCache()` 仍可能读到 Redis 的旧 `{L1}`，补入内存再加入 L2，形成错误的 `{L1,L2}`。这个交错不需要发生存储错误，属于本次必须处理的边界。

在线读取和写前补齐复用现有标记，遵守以下规则：

| 状态 | 处理 |
|---|---|
| 内存 K 存在 | 使用内存值；即使所查询的 Location 不存在，也不回源；新写正常合并 |
| Recover，内存 K 缺失，有删除标记 | 视为逻辑不存在，不读 Redis 旧内容；业务可以直接创建新 K |
| Recover，内存 K 缺失，无删除标记 | 沿用原有 Redis 回源/补齐，保留尚未恢复的有效数据 |
| 已进入 Running | 沿用 cache-only 行为，不回源 |

```text
GC：     DEL K 入队 → 登记标记 → 删除内存 K
业务：   新增 L2；内存缺失且有标记，跳过 Redis 补齐
         正常提交 Upsert，创建内存 K={L2}
Redis：  按同 key 顺序执行 DEL、Upsert，最终为 K={L2}
recover：旧 {L1} 被标记挡住，不能再次插入
```

判断落在第一次可能回源的读取处，不能只改 `EnsureKeyInCache()`：在线 RMW 会先读取 Location/Location ID/key status，再生成写入内容。相应读取返回 key/Location `EC_NOENT`，让上层按新建处理并正确增加 key count；真正的读取错误仍返回错误，不伪装成不存在。普通 Get、Properties、Exists 的恢复期回源使用同一规则，避免各入口对已删除 key 得出相反结论。

复用小型回源过滤 helper，在已有 `deleted_keys_mutex_` 内一起判读当前恢复状态与删除集合。若恢复已完成并清空集合，仍因 Running 而禁止回源；不依赖进入函数时读到的旧 Recover 状态。在线 RMW 继续持有原 per-key 锁，名单锁只用于短暂检查，不跨 Redis I/O。

本改动不增加业务侧 Sync，不因删除标记拒绝新写。普通写原有的 Redis-first 入队背压、后端失败语义保持不变，不承诺消除存储故障或跨重启恢复已丢弃的异步写。

### 4.4 普通 GC：目标校验与按需补齐

原有的覆盖式刷新会用 Redis 全值覆盖内存，Redis NOENT 时还会删除内存 key。普通 GC 清理 L2 前执行这种刷新，可能把 EventReport 已删除的 L1 带回来：

| 步骤 | 内存中的 K | Redis 中的 K |
|---|---|---|
| 初始状态 | `{L1, L2}` | `{L1, L2}` |
| EventReport GC 删除 L1，Redis 删除尚在队列中 | `{L2}` | `{L1, L2}` |
| 普通 GC 清理 L2 前全量覆盖 K | `{L1, L2}`，L1 被带回 | `{L1, L2}` |

**普通 GC 通过锁内目标校验确认待删的 L2 没变。** `EnsureCacheForGc()` 只在 Recover 缺 key 时按需完整补齐，再执行目标条件修改；已有 K 不被全量覆盖。不新增强刷保护集合。

以下规则仅针对 cached 模式，在原有 per-key shard lock 内执行：

| 执行时的状态 | 普通 GC 处理 |
|---|---|
| 内存有 K，也有目标 L2 | 使用内存中的 L2，与扫描时保存的完整序列化值及预期状态比较；一致才按原流程标记 `DELETING`，不一致则跳过 L2 |
| 内存有 K，但没有目标 L2 | 跳过 L2，不从 Redis 补回；空 Location 列表也属于 K 存在 |
| Recover，内存没有 K，且有 `deleted_keys_` 标记 | 跳过该 key，不读 Redis 旧值、不新建 K |
| Recover，内存没有 K，且没有删除标记 | 复用 `EnsureKeyInCache()` 的完整读取与 `PutIfAbsent`，再读取实际内存值进行条件校验；读取或补齐失败沿用错误处理，补齐后目标仍缺失则跳过 |
| Running，内存没有 K | 将该候选视为失效并跳过，不回源、不新增读前 Sync |

扫描只产生候选；锁内读取、比较和标记删除状态属于同一个临界区，业务不能插到比较与修改之间。业务若先更新了 L2，比较就会不匹配。该机制沿用原有 expected-value RMW；只把缺失或不匹配的目标排除出后续删除任务，同批其他目标继续，不能把跳过当作成功标记后继续物理删除。

Recover 的补齐必须是完整 key 的 `PutIfAbsent`，不能只把 Redis 中的一个 Location 填入空 K。补齐前复用 4.3 的当前恢复状态与删除标记过滤；若此时已进入 Running，就不再回源。补齐后始终重新读取内存，不直接拿 Redis 返回值授权删除。已存在的 K 不被覆盖，普通 GC 的最终 Location 条件删除和空 key 回收继续沿用原链路；EventReport 按第 3 节只依据 hot 选择部分或整 key 删除，缺失时直接跳过。

Recover 完成只表示后台回填结束，不代表 Redis 异步队列已经清空。因此上述“已有 K 不覆盖、Running 缺 K 不回源”的规则在恢复结束后仍然有效，不恢复无条件全量覆盖。以表中的例子，普通 GC 只校验内存中的 L2，再按原流程清理，L1 就不会被强刷带回。

Redis 操作与 Sync 的范围保持明确：

- 普通 GC 已有的锁外前置读取继续用于筛选候选，不拿该结果覆盖内存；最终以锁内目标校验决定是否进入删除任务。
- 条件修改和删除仍按原路由更新 Redis 与内存；`async_redis` 成功可能仅表示入队，并非取消 Redis 写入。
- 不为普通 GC 新增每个候选的锁内 Redis 重读或读前 Sync。Recover 缺 key 时的完整补齐仍可能读 Redis、持有原 shard 锁；不能把整个清理过程理解成纯内存操作。
- 普通 GC 物理删除前已有的锁外 Sync、在线写入流程和 single-backend 行为均保留；cached EventReport 采用 4.1 的 hot-only 校验，不执行读前或删后 Sync。

普通 GC 在 Running 阶段只处理内存中的目标，不主动补齐 Redis-only key。GC 只扫描内存；执行时目标已经缺失，可以结束本次处理，后续进入内存的有效候选仍可被重新扫描。不为部分写失败造成的 Redis-only 残留增加修复流程，也不承诺清完所有 Redis metadata。
