# 分层驱逐与多层缓存流程

本文档描述 optimizer 模块中多层(tiered)缓存的完整数据流，涵盖初始化、读写、驱逐、用量统计和 CSV 输出。

---

## 1. 核心数据结构

### 1.1 BlockEntry

```cpp
struct BlockEntry {
    int64_t key;
    LocationStatMap location_map;   // map<tier_name, TierStat>
    // ...
};
```

`location_map` 是写穿和分层驱逐的核心：
- 写入时，block 同时在所有 tier 注册 → `{"tier0": stat, "tier1": stat}`
- tier0 驱逐后 → `{"tier1": stat}`（block 仍存活）
- 所有 tier 都驱逐后 → `{}`（block 彻底死亡，等待从 radix tree 清理）

### 1.2 TieredPolicyGroup

```cpp
struct TieredPolicyGroup {
    vector<shared_ptr<EvictionPolicy>> policies;
    vector<OptTierConfig> tier_configs;

    const shared_ptr<EvictionPolicy>& shared_policy() const;  // = policies.front()
};
```

- **非分层模式**：`policies` 只有一个 name="shared" 的策略，通过 `shared_policy()` 访问。
- **分层模式**：每个 tier 各一个独立策略，通过 `policies[tier_idx]` 访问。

### 1.3 QueryHit

```cpp
struct QueryHit {
    size_t local_hit_blocks = 0;
    size_t remote_hit_blocks = 0;
    vector<size_t> per_tier_hit_blocks;  // 按 tier priority 索引
};
```

### 1.4 ReadRecord

```cpp
struct ReadRecord {
    int64_t timestamp_us;
    size_t remote_read_blocks;          // 本次请求中来自远端的 block 数
    size_t remote_hit_blocks;           // 远端请求中命中缓存的 block 数
    size_t local_read_blocks;           // 本次请求中来自本地的 block 数
    size_t local_hit_blocks;            // 本地请求中命中缓存的 block 数
    size_t current_cache_blocks;        // 当前 instance 总物理占用（所有 tier 之和）
    vector<size_t> per_tier_hit_blocks; // 各层命中 block 数
    vector<string>  tier_names;         // 各层名称（用于 CSV 列头）
    vector<size_t> per_tier_blocks;     // 各层当前 block 数
    vector<size_t> blocks_per_instance; // 所有 instance 各自的总占用
};
```

---

## 2. 初始化阶段

```
IndexerManager::CreateOptIndexer(instance_config, storage_configs, hierarchical_eviction_enabled)
│
├─ EvictionManager::CreateAndRegisterEvictionPolicy(...)
│   │
│   ├─ hierarchical = true:
│   │    for each tier in storage_configs:
│   │      创建独立 EvictionPolicy(name = tier.unique_name())
│   │    → policies = [policy_tier0, policy_tier1, ...]
│   │
│   └─ hierarchical = false:
│        创建单一 EvictionPolicy(name = "shared")
│        → policies = [shared_policy]
│
└─ 创建 RadixTreeIndex(instance_id, policies)
   RadixTreeIndex 持有 tier_policies_ = policies
   tier_names_ = [policy.name() for each policy]
```

`IndexerManager` 还会 `RegisterInstanceGroups` / `RegisterInstances`，将 group 和 instance 配置存入内部 map，供后续驱逐查找。

---

## 3. Trace 回放主循环

```
OptimizerRunner::Run(config)
  └─ RunTraces(traces)
       └─ for each trace: RunTrace(trace)
            │
            ├─ DialogTurnSchemaTrace  → HandleDialogTurn()    // 读+写
            ├─ GetLocationSchemaTrace → HandleGetLocation()   // 纯读
            └─ WriteCacheSchemaTrace  → HandleWriteCache()    // 纯写
```

每条 trace 处理完后均调用 `stats_collector_->UpdateTimestamp(instance_id, timestamp_us)` 推进时间线。

---

## 4. 三种 Handler

### 4.1 HandleGetLocation（纯读查询）

```
GetIndexer(instance_id)              → 获取 RadixTreeIndex
indexer->PrefixQuery(keys, mask, ts, &query_hit)   → 前缀匹配，填充 QueryHit
计算 local_read_blocks / remote_read_blocks（从 block_mask 解析）
SubmitReadRecord(...)                → 构建 ReadRecord 并提交
```

**不触发驱逐。**

### 4.2 HandleWriteCache（纯写入）

```
GetIndexer(instance_id)
indexer->InsertOnly(keys, ts)        → 插入 block（写穿到所有 tier）
indexer_manager_->CheckAndEvict(instance_id, ts)   → ★触发驱逐检查
StatsCollector::OnWriteComplete(WriteRecord)
```

### 4.3 HandleDialogTurn（读+写）

```
GetIndexer(instance_id)
indexer->InsertWithQuery(total_keys, ts, &query_hit)   → 边查询边插入
indexer_manager_->CheckAndEvict(instance_id, ts)       → ★触发驱逐检查
SubmitReadRecord(..., local_read_blocks=0, remote_read_blocks=keys.size())
StatsCollector::OnWriteComplete(WriteRecord{decode_blocks})
```

`InsertWithQuery` 合并了 PrefixQuery + Insert 的逻辑，避免两次遍历 radix tree。
DialogTurn 的读请求全部视为 remote read（整个 prefill 来自远端）。

---

## 5. 读写在 RadixTreeIndex 中的具体行为

### 5.1 写入：写穿（Write-Through）

`AppendNewBlocks` / `AppendEvictBlocks` 中：

```cpp
// 1. 为所有 tier 写入 location
for (const auto &name : tier_names_) {
    AppendBlockLocation(entry_ptr, name, timestamp);
}
// → block.location_map = {"tier0": stat, "tier1": stat, ...}

// 2. 注册到所有 tier 的驱逐队列
for (auto &policy : tier_policies_) {
    policy->OnNodeWritten(inserted_blocks);
}
```

**结论：一个 block 写入后，同时存在于所有层。**

### 5.2 读取：任意一层有就命中

`PrefixQuery` / `InsertQuery` 中：

```cpp
if (IsBlockEvict(block)) break;  // IsBlockEvict = location_map.empty()
```

只要 `location_map` 非空（任意一层还持有 block），就算命中。

### 5.3 Per-Tier 命中归属

`RecordTieredHit` 中：

```cpp
// 按 priority 从高到低检查，命中归到最高优先级层
for (size_t i = 0; i < tier_names_.size(); ++i) {
    if (block->location_map.count(tier_names_[i])) {
        query_hit->per_tier_hit_blocks[i]++;
        break;   // 只记一次
    }
}
```

**结论：命中归属于 block 仍在的最高优先级层。**

### 5.4 访问更新

`OnBlockAccessed` 中，block 级别统计只更新一次，但遍历所有 tier 更新各层的策略和 per-tier 统计：

```cpp
block->access_count += 1;          // block 全局统计
block->last_access_time = timestamp;

for each tier:
    if block in tier's location_map:
        policy[tier].OnBlockAccessed(block, ts)   // 更新该层 LRU 位置
        tier_stat.access_count += 1               // 更新该层访问统计
```

---

## 6. 驱逐流程

### 6.1 入口：CheckAndEvict

```
IndexerManager::CheckAndEvict(instance_id, timestamp)
│
├─ 查找 instance_config → group_name → group_config
│
├─ EvictionManager::EvictByMode(instance_id, group_config)
│   → 返回 map<instance_id, vector<BlockEntry*>>
│
└─ 清理阶段（关键！）:
     for each 被驱逐的 block:
       if block->location_map.empty():     // ★所有 tier 都驱逐了
         truly_evicted.push_back(block)
     indexer->CleanEmptyBlocks(truly_evicted, timestamp)
```

**只有 `location_map` 完全为空的 block 才从 radix tree 中真正删除。**
tier0 驱逐后如果 tier1 仍持有，block 保留在 tree 中。

### 6.2 EvictByMode：构建任务 + 分发

```
EvictByMode(instance_id, group_config)
│
├─ 构建任务列表 tasks: vector<(optional<tier_idx>, excess)>
│   │
│   ├─ 分层模式:
│   │    for tier_idx in 0..num_tiers:
│   │      excess = GetExcessUsage(group_config, tier_idx)
│   │      if excess > 0 → tasks.add(tier_idx, excess)
│   │
│   └─ 非分层模式:
│        excess = GetExcessUsage(group_config, nullopt)
│        if excess > 0 → tasks.add(nullopt, excess)
│
└─ for (tier_idx, excess) in tasks:
     DispatchEviction(instance_id, group_config, tier_idx, excess)
     合并各 tier 的驱逐结果
```

### 6.3 用量计算

#### GetExcessUsage(group_config, tier_idx)

```
capacity = tier_idx有值 ? storages[tier_idx].capacity()   // 该层独立容量
                       : group_config.quota_capacity()     // group 整体配额

current_used = GetCurrentGroupUsage(group_config, tier_idx)

quota = capacity * used_percentage

return max(current_used - quota, 0)
```

#### GetCurrentGroupUsage(group_config, tier_idx)

```
for each instance in group:
    tier_idx有值 → total += policies[tier_idx]->size()
    nullopt     → total += shared_policy()->size()
return total
```

### 6.4 DispatchEviction：按模式分发

```
DispatchEviction(instance_id, group_config, tier_idx, excess)
│
├─ GROUP_ROUGH      → EvictByGroupRough(...)
├─ INSTANCE_ROUGH   → EvictByInstance(..., precise=false)
└─ INSTANCE_PRECISE → EvictByInstance(..., precise=true)
```

### 6.5 三种驱逐模式

| 模式 | 驱逐范围 | Batch 行为 |
|------|----------|-----------|
| GROUP_ROUGH | group 内所有 instance 轮询 | 每轮固定 batch_size，可能超驱逐 |
| INSTANCE_ROUGH | 仅触发写入的 instance | 每轮固定 batch_size，可能超驱逐 |
| INSTANCE_PRECISE | 仅触发写入的 instance | 最后一轮 cap 到剩余量，不超驱逐 |

所有模式都使用统一的策略选择：

```cpp
auto &policy = tier_idx.has_value()
    ? policies[tier_idx.value()]   // 分层: 该 tier 的独立策略
    : shared_policy();              // 非分层: shared 策略
```

`policy->EvictBlocks(count)` 从策略中移除 block，并清除 block 的 `location_map` 中对应 tier 的条目。

---

## 7. 统计记录与 CSV 输出

### 7.1 SubmitReadRecord

```
SubmitReadRecord(instance_id, ts, query_hit, indexer, local_read, remote_read)
│
├─ record.current_cache_blocks
│    = GetCurrentInstanceUsage(instance_id)
│    = Σ policy->size() for ALL tiers      // instance 物理存储总占用
│
├─ record.blocks_per_instance
│    = 遍历所有 instance，各自的 GetCurrentInstanceUsage
│
├─ record.per_tier_hit_blocks = query_hit.per_tier_hit_blocks
├─ record.tier_names          = indexer->GetTierNames()
├─ record.per_tier_blocks
│    = GetCurrentInstanceUsagePerTier(instance_id)
│    = [policy_0->size(), policy_1->size(), ...]
│
└─ stats_collector_->OnReadComplete(instance_id, record)
```

### 7.2 CSV 列布局

```
Timestamp |
RemoteReadBlockNum | RemoteHitBlockNum | LocalReadBlockNum | LocalHitBlockNum |
AccHitRate |
AccTier0(name)_HitRate | AccTier1(name)_HitRate | ... |
Tier0(name)_HitBlockNum  | Tier1(name)_HitBlockNum  | ... |
Tier0(name)_BlockNum     | Tier1(name)_BlockNum     | ... |     ← per-tier 当前用量
CacheBlockNum |                                                ← 总用量(所有 tier 之和)
Instance0_BlockNum | Instance1_BlockNum | ...                    ← 各 instance 用量
```

---

## 8. 用量语义总结

| 函数 | 语义 | 用途 |
|------|------|------|
| `GetCurrentInstanceUsage(id)` | Σ 所有 tier 的 policy->size() | instance 总物理占用 |
| `GetCurrentInstanceUsagePerTier(id)` | 各 tier 的 policy->size() 数组 | CSV per-tier 列 |
| `GetCurrentGroupUsage(config, tier_idx)` | group 内所有 instance 在指定 tier 的用量之和 | 驱逐判断 |
| `GetExcessUsage(config, tier_idx)` | max(group_usage - quota, 0) | 需驱逐的 block 数 |

---

## 9. 端到端示例

以 2 tier (tier0=GPU, tier1=NFS)、分层驱逐、INSTANCE_PRECISE 模式为例：

```
1. 初始化
   CreateOptIndexer → policies = [policy_gpu, policy_nfs]

2. HandleWriteCache (写入 100 blocks)
   InsertOnly:
     每个 block.location_map = {"gpu": stat, "nfs": stat}
     policy_gpu.OnNodeWritten(100 blocks)
     policy_nfs.OnNodeWritten(100 blocks)
   → policy_gpu.size() = 100, policy_nfs.size() = 100

3. CheckAndEvict
   EvictByMode:
     tier0(gpu): excess = GetExcessUsage(config, 0)
       capacity = gpu.capacity() = 80
       current  = group 内所有 instance 的 policy_gpu.size() 之和 = 100
       quota    = 80 * 0.9 = 72
       excess   = 100 - 72 = 28
     tier1(nfs): excess = GetExcessUsage(config, 1)
       capacity = nfs.capacity() = 200
       excess   = 0 (未超额)
   
   DispatchEviction(tier_idx=0, excess=28):
     EvictByInstance(precise=true):
       从 policy_gpu 驱逐 28 blocks
       每个被驱逐 block: location_map.erase("gpu")
       → 这些 block 的 location_map = {"nfs": stat}（仍存活）
   
   清理阶段:
     28 blocks 的 location_map 非空 → 不清理，保留在 radix tree

4. HandleGetLocation (读查询)
   PrefixQuery:
     block.location_map = {"nfs": stat} → 非空 → 命中
     RecordTieredHit: 归到 tier1(nfs)
   
5. 后续再 tier1 也超额时:
   驱逐 tier1 → block.location_map = {} → 彻底死亡
   CleanEmptyBlocks → 从 radix tree 删除
```
