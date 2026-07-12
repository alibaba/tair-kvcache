# ReportEvent 元数据一致性与 Location 索引设计

## 1. 背景

`ReportEvent` 用于接收外部 subscriber 上报的 KVCache 事件。KVCM 不主动连接所有推理 Worker，而是由 RTP-LLM、vLLM、V6D 等侧的 subscriber 将本地 cache 变化归一化成 block 级事件，上报给 KVCM。

其中 `EVENT_BLOCK_SNAPSHOT` 是一个按 location 做全量对账的事件：subscriber 上报某个 `host_ip_port + medium` 当前完整拥有的 block 集合，KVCM 需要和自己已有的元数据做 diff，补齐新增 block，并删除已经不在该 location 上的 block。

如果每次 snapshot diff 都从 Redis 全量扫描所有 CacheMeta，再筛选目标 location，会带来几个问题：

- 请求路径延迟不可控，block 数量增长后会直接影响 subscriber 上报。
- `SCAN` 不是一致性快照，并发写入/删除时可能重复或遗漏。
- 多 KVCM 进程各自扫描重建内存索引，浪费 Redis 和 KVCM 资源。
- 依赖 meta backend 支持全量 scan，不利于抽象和后续替换 backend。

因此 snapshot diff 需要一份可持久化、跨 KVCM 共享的 `location_id -> block_key` 二级索引。同时，二级索引不能成为新的事实来源，读链路必须显式处理索引与 CacheMeta 不一致的情况。

## 2. 目标与非目标

### 目标

- `ReportEvent` 写入的 block cache 信息仍然存入普通 CacheMeta，和现有查询链路复用同一份元数据。
- 为 `EVENT_BLOCK_SNAPSHOT` 提供持久化的 location 反向索引，避免请求路径全库扫描。
- 读链路容忍索引和 CacheMeta 的短暂不一致，不因索引脏数据误删或误命中。
- 写链路通过顺序设计，让新写入尽量只产生 over-index，不产生新的 under-index。
- 支持 RTP-LLM full snapshot 和 vLLM KV events 两类 subscriber。

### 非目标

- 不让 KVCM 主动拉取所有 Worker 的 cache 状态。
- 不把 location 反向索引暴露成用户查询接口。
- 当前不实现 mamba-state 的精细匹配。接口接收相关 `LocationSpec.name`，但存储和匹配仍只处理 full-attention。
- 不要求所有 meta backend 一开始都支持原子多 key 事务；Redis 后端优先实现强一些的一致性语义，其他 backend 可以先实现同接口的最佳努力版本。

## 3. 核心概念

### CacheMeta

CacheMeta 是唯一事实来源，结构仍然是：

```text
instance_id + block_key -> CacheLocationMap
```

每个 `CacheLocation` 包含：

- `location_id`
- `storage_type`
- `status`
- `LocationSpec[]`

`GetCacheMeta` / `GetCacheLocation` 等查询接口只信任 CacheMeta，不直接使用 location 反向索引。

### Location

`ReportEvent` 的 diff scope 是一个 location。location 由 event backend 根据 `host_ip_port + medium` 构造，例如 V6D 可构造为：

```text
kvs#v6d#{medium}#{host_ip_port}
```

`storage_type` 表示大的存储系统，比如 Vineyard/V6D；`medium` 表示该 host 下可独立 diff 的位置，比如 `gpu`、`mem`、`disk`。更细粒度的 full-attention group、TP、mamba-state 等信息放在 `LocationSpec.name` 中。

### Location 反向索引

二级索引用于加速 snapshot diff：

```text
instance_id + location_id -> set(block_key)
```

它是 CacheMeta 的派生索引，不是事实来源。索引允许短暂 over-index：索引里有某个 block_key，但 CacheMeta 中没有对应 location。读链路会过滤这类脏数据，后续由延迟/离线 repair 清理索引。

索引不应产生新的 under-index：CacheMeta 中已有某个 location，但索引缺少该 block_key。under-index 会导致 snapshot diff 找不到应删除的旧 block，读路径无法在不扫描全量 CacheMeta 的情况下完全发现它。因此写路径要通过顺序和失败处理避免新 under-index。

## 4. 持久化模型

location 反向索引属于 meta 层能力，调用入口挂在 `MetaStorageBackend` / `MetaIndexer` / `MetaSearcher` 这一层，避免把索引逻辑散落在 `CacheManager`。`CacheManager` 只表达业务语义：上报 add/delete/snapshot；`MetaSearcher` 负责 CacheMeta 与二级索引的一致性维护。

抽象接口等价于：

```cpp
class LocationKeyIndexBackend {
public:
    virtual ErrorCode AddKeys(RequestContext*, const std::string& location_id,
                              const KeyVector& keys) = 0;
    virtual ErrorCode RemoveKeys(RequestContext*, const std::string& location_id,
                                 const KeyVector& keys) = 0;
    virtual ErrorCode GetKeys(RequestContext*, const std::string& location_id,
                              KeyVector& out_keys) = 0;
};
```

Redis / AsyncRedis 后端使用 set 存储并与 CacheMeta 共用 instance namespace：

```text
{meta_prefix}:locidx:{hash(location_id)}:{shard_id} -> SET(block_key)
```

当前实现先使用单 set，接口保留了后续分 shard 的空间，避免单 location block 数过大时改调用层。Local 后端的 location index 是内存态；Dummy 后端持久化 CacheMeta，并在重新 `Open()` 时从 CacheMeta 重建 location index，因此它可以覆盖本地持久化重启场景，但不是独立持久索引。Redis 后端的 set 才是多 KVCM 共享场景下的持久索引来源。

## 5. 写路径设计

### 5.1 Add / Upsert

Add 或 upsert 某个 `block_key -> location_id` 时，顺序应为：

1. 先把 `block_key` 加入 location 反向索引。
2. 再写入或更新 CacheMeta。
3. 如果索引写失败，则不写 CacheMeta，并返回该 item 失败。
4. 如果索引写成功但 CacheMeta 写失败，允许留下 over-index。

这样最坏情况是索引里多了一个 key。后续 snapshot 读索引时会 BatchGet CacheMeta 过滤，发现该 key 并没有对应 location 后记录为 stale，交给延迟/离线 repair 清理。更重要的是，成功写入 CacheMeta 的 location 基本不会缺少反向索引。

### 5.2 Delete

Delete 某个 `block_key -> location_id` 时，顺序应为：

1. 先从 CacheMeta 删除该 location。
2. 再从 location 反向索引删除该 `block_key`。
3. CacheMeta 删除失败则返回失败，不删除索引。
4. CacheMeta 确认删除成功后，再删除索引；如果索引删除失败，允许留下 over-index。
5. 如果 CacheMeta 返回 `EC_NOENT`，请求热路径不急于删除索引，避免和并发 add-before-meta 形成竞态。此时留下的 over-index 由读路径过滤，后续离线 repair 处理。

这个顺序同样保证失败后最多是 over-index，而不是 CacheMeta 仍有 location 但索引已经丢失。

### 5.3 RemoveCache / HostDown / Reclaimer

所有会删除 CacheMeta 的路径都必须维护 location 反向索引，包括：

- `ReportEvent(EVENT_BLOCK_DELETE)`
- `ReportEvent(EVENT_BLOCK_SNAPSHOT)` diff 出来的删除
- `HOST_DOWN` 触发的 host location 清理
- `RemoveCache`
- reclaimer 删除

这些路径如果能在删除前读到旧 `CacheLocationMap`，应先记录要删除的 `location_id` 列表，再按“先删 CacheMeta，后删索引”的顺序执行。

### 5.4 Snapshot

`EVENT_BLOCK_SNAPSHOT` 的单 location 对账流程：

1. 读取该 location 的反向索引，得到 candidate keys。
2. 对 candidate keys 做 `BatchGetLocation`。
3. 只保留 CacheMeta 中仍然包含该 `location_id` 且状态可服务的 keys，得到 `existing_keys`。
4. 对于索引有但 CacheMeta 不再包含该 location 的 keys，记录为 stale index；热路径只过滤，不立即删除索引。
5. 从上报 payload 得到 `reported_keys`。
6. `to_add = reported_keys - existing_keys`。
7. `to_delete = existing_keys - reported_keys`。
8. 对 `to_add` 走 Add / Upsert 顺序。
9. 对 `to_delete` 走 Delete 顺序。

重要约束：snapshot 只能删除 `existing_keys - reported_keys`，不能删除“索引中存在但 CacheMeta 未确认存在”的 key，也不能根据索引直接判断某个 key 有效。

## 6. 读路径不一致处理

### 6.1 Snapshot Diff 读索引

索引读取结果必须经过 CacheMeta 二次确认。伪代码：

```text
candidate_keys = LocationIndex.GetKeys(location_id)
location_maps = BatchGetLocation(candidate_keys)

existing_keys = {}
stale_index_keys = {}

for key, location_map in zip(candidate_keys, location_maps):
    if location_map contains location_id and location is serving/full-attention:
        existing_keys.add(key)
    else:
        stale_index_keys.add(key)

record stale_index_keys for metrics / delayed repair
```

这样 over-index 不会造成误删，也不会让不存在的 cache 被当成命中。不要在这个同步读路径里立刻删除 stale index，因为 add 写路径是先写 index 再写 CacheMeta；如果此时把刚写入的 index 当作 stale 删除，后续 CacheMeta 写成功后反而会制造 under-index。

### 6.2 正常 Cache 查询

`GetCacheMeta` / `GetCacheLocation` 不读 location 反向索引，仍从 CacheMeta 读取 block 对应的 `CacheLocationMap`，并按现有规则过滤：

- instance 隔离
- `CacheLocationStatus`
- `DataStorageType`
- `LocationSpec.name`
- full-attention / mamba-state 兼容策略

因此索引 over-index 不影响正常 cache 命中。

### 6.3 Under-index 的处理

under-index 是需要重点避免的状态。因为如果 CacheMeta 中存在 `location_id`，但 location 索引缺少该 key，则 snapshot diff 无法发现这个旧 key，自然也无法在该 key 未上报时删除它。

设计上应把 under-index 作为异常状态处理：

- 新写路径通过 add-before-meta、delete-after-meta 避免产生新的 under-index。
- Redis 后端可以用 Lua 或 pipeline + 幂等重试进一步收敛失败窗口。
- 启动时不在请求路径全量扫描重建索引。
- 提供低优先级 repair job 或运维工具，对历史数据和异常状态做离线校验：扫描 CacheMeta，重建/补齐 location 索引，并输出修复指标。
- 对 snapshot diff 暴露指标：candidate 数、stale index 数、reported 数、add 数、delete 数；离线 repair 暴露独立修复量和失败数。如果 stale index 或 repair 失败持续升高，应告警。

换句话说：读路径要容忍 over-index；under-index 不能靠普通读路径完全修复，只能靠写路径不制造、离线 repair 兜底。

## 7. RTP-LLM 与 vLLM 接入

### RTP-LLM

RTP-LLM subscriber 可以继续使用现有本地 cache 状态采集方式，但对 KVCM 上报时应转换为：

- `EVENT_NODE_REGISTER`：注册 host 和可上报 medium。
- `EVENT_HEARTBEAT`：汇报活性和观测指标。
- `EVENT_BLOCK_SNAPSHOT`：按 `host_ip_port + medium` 汇报该 location 的完整 block 集合。

RTP-LLM 的 full snapshot 是 authoritative input，但 KVCM 仍只删除 CacheMeta 二次确认后存在于该 location 的 keys。

### vLLM

vLLM KV events 可以转换为：

- block 创建或变为可服务：`EVENT_BLOCK_ADD`
- block 删除：`EVENT_BLOCK_DELETE`
- worker 清空或重建本地 cache：空或完整 `EVENT_BLOCK_SNAPSHOT`

vLLM 上报的 group id、attention 类型、TP 等信息放入 `LocationSpec.name`。当前 KVCM 存储和匹配 full-attention；mamba-state spec 可以上报，但会被过滤掉，后续再扩展精细匹配。

### V6D / 本地显存混合位置

`storage_type` 仍表示大的存储系统，例如 Vineyard/V6D；`medium` 表示该系统下的 diff location，例如 `mem` 或 `disk`；`LocationSpec.uri` 表示具体地址。推理引擎内部显存和外部 V6D 内存/磁盘可以作为不同 `storage_type + medium + uri` 组合表达，不需要在 request 级别再引入更细的存储层级。

## 8. 多 KVCM 与失败语义

多 KVCM 进程共享 Redis 后端时，location 反向索引也必须共享同一 namespace，与 CacheMeta 使用同样的 instance 隔离和 prefix 规则。

所有索引操作都应幂等：

- 重复 Add：结果不变。
- 重复 Delete：结果不变。
- 重复 Snapshot：最终收敛到上报集合。

失败时的语义：

- Add 索引失败：不写 CacheMeta，item 返回失败。
- Add CacheMeta 失败：item 返回失败，可能留下 over-index。
- Delete CacheMeta 返回 `EC_NOENT`：业务删除可视为幂等成功，但热路径不删除索引，避免误清理并发 add-before-meta 的 index。
- Delete 索引失败：item 可以返回成功或部分成功，取决于调用方是否要求强一致；但必须打指标，因为它会留下可修复的 over-index。
- Snapshot 中发现 stale index：不影响 snapshot 主流程，打指标并交给延迟 repair。

## 9. 测试要求

单元测试需要覆盖：

- Add 成功后 CacheMeta 与 location index 都可见。
- Add 索引失败时不写 CacheMeta。
- Add CacheMeta 失败时留下 over-index，snapshot 读路径能过滤。
- Delete 成功后 CacheMeta 和 index 都删除。
- Delete 索引失败留下 over-index，snapshot 读路径能过滤。
- Snapshot 只删除 CacheMeta 二次确认存在的 keys。
- Snapshot 对 stale index 不误删。
- mamba-state specs 被接收但不进入当前 full-attention 存储和匹配。
- RTP-LLM full snapshot 与 vLLM add/delete/snapshot 混合事件顺序保持。

集成测试需要覆盖：

- Redis meta backend 下 `ReportEvent` 写入后重启 KVCM，location index 仍可用于 snapshot diff。
- 多 KVCM 进程共享同一 Redis index 时，A 写入、B snapshot 能正确 diff。
- 构造 over-index 后，snapshot 不误删；stale index 清理由延迟/离线 repair 覆盖。
- 历史无 index 数据通过 repair job 补齐后，snapshot 不需要请求路径 scan。

## 10. 演进步骤

1. 在 meta 层增加 location index backend 抽象和 Redis 实现。
2. 将 `MetaSearcher` 的内存 `location_key_index_` 改为持久化 index 读写，并保留小范围内存缓存仅做性能优化，不能作为事实来源。
3. 调整 `BatchUpsertLocations` 和所有删除 location 的路径，按 add-before-meta、delete-after-meta 顺序维护 index。
4. 重写 `EVENT_BLOCK_SNAPSHOT` diff：读持久化 index，BatchGet CacheMeta 二次确认，请求热路径只过滤 stale index。
5. 增加指标和日志，观察 over-index 修复量、索引操作失败量、snapshot diff 数量。
6. 增加离线 repair 工具，用于老数据迁移和异常状态修复；请求路径不做全量 scan。
