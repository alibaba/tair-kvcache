# UsageSnapshot 统一用量封装 — 设计文档

## 背景与动机

当前用量统计散布在多层调用中，`SubmitReadRecord` 每次手动拼装 `ReadRecord` 的各字段：

```cpp
// optimizer_runner.cc — SubmitReadRecord
record.current_cache_blocks = eviction_manager_->GetCurrentInstanceUsage(instance_id);
// ... 遍历 indexer_map 填充 blocks_per_instance ...
record.per_tier_blocks = eviction_manager_->GetCurrentInstanceUsagePerTier(instance_id);
```

问题：
1. **拼装分散** — 每新增用量维度，必须修改 `SubmitReadRecord`，容易遗漏
2. **语义耦合** — `ReadRecord` 既承载命中率数据，又承载用量快照，职责不清
3. **重复查询** — `GetCurrentInstanceUsage` 遍历所有 tier 做 sum，`GetCurrentInstanceUsagePerTier` 再遍历一次

## 设计方案

### 核心数据结构

```cpp
// 文件: kv_cache_manager/optimizer/config/types.h（或独立 usage_snapshot.h）

struct UsageSnapshot {
    // ---- Instance 维度 ----
    std::string instance_id;
    size_t total_blocks = 0;                   // 所有 tier 物理副本总数
    std::vector<size_t> per_tier_blocks;       // per-tier 用量明细
    std::vector<std::string> tier_names;       // per-tier 名称（与 per_tier_blocks 对齐）

    // ---- Group 维度（可选，按需填充）----
    std::vector<size_t> blocks_per_instance;   // group 内各 instance 的 total_blocks
    std::vector<std::string> instance_ids;     // 与 blocks_per_instance 对齐，保证稳定顺序
    size_t group_total_blocks = 0;             // group 级汇总
};
```

### 提供方: OptEvictionManager

在 `OptEvictionManager` 新增一个方法，一次性构建完整快照：

```cpp
// eviction_manager.h
UsageSnapshot TakeInstanceSnapshot(const std::string &instance_id,
                                    const OptInstanceGroupConfig &group_config) const;
```

实现要点：
- 遍历 `instance_tiered_policy_map_` 中 `instance_id` 的 policies，**一次遍历**同时产出 `per_tier_blocks` 和 `total_blocks`
- 遍历 group 内所有 instance 产出 `blocks_per_instance`，使用 `group_config.instances()` 的顺序（**稳定有序**，消除当前 unordered_map 遍历的顺序问题）
- `tier_names` 从 `TieredPolicyGroup::tier_configs` 或 `RadixTreeIndex::GetTierNames()` 获取
- `instance_ids` 从 `group_config.instances()` 的顺序提取

### 消费方: OptimizerRunner::SubmitReadRecord

重构前：

```cpp
void OptimizerRunner::SubmitReadRecord(...) {
    ReadRecord record{};
    record.timestamp_us = timestamp_us;
    record.current_cache_blocks = eviction_manager_->GetCurrentInstanceUsage(instance_id);
    // 手动遍历 indexer_map 填充 blocks_per_instance（顺序不稳定）
    auto indexer_map = indexer_manager_->GetAllOptIndexers();
    record.blocks_per_instance.resize(indexer_map.size(), 0);
    size_t idx = 0;
    for (const auto &pair : indexer_map) {
        record.blocks_per_instance[idx] = eviction_manager_->GetCurrentInstanceUsage(pair.first);
        idx++;
    }
    record.per_tier_blocks = eviction_manager_->GetCurrentInstanceUsagePerTier(instance_id);
    record.tier_names = indexer->GetTierNames();
    // ...
}
```

重构后：

```cpp
void OptimizerRunner::SubmitReadRecord(...) {
    ReadRecord record{};
    record.timestamp_us = timestamp_us;

    auto snapshot = eviction_manager_->TakeInstanceSnapshot(instance_id, group_config);
    record.current_cache_blocks = snapshot.total_blocks;
    record.blocks_per_instance = std::move(snapshot.blocks_per_instance);
    record.per_tier_blocks = std::move(snapshot.per_tier_blocks);
    record.tier_names = std::move(snapshot.tier_names);
    // ...
}
```

### ReadRecord 的演化

短期内 `ReadRecord` 保持现有字段不变，`SubmitReadRecord` 从 `UsageSnapshot` 搬运到 `ReadRecord`。

中期可考虑 `ReadRecord` 直接持有 `UsageSnapshot`，消除搬运：

```cpp
struct ReadRecord {
    int64_t timestamp_us;
    // 命中率相关
    size_t remote_read_blocks;
    size_t remote_hit_blocks;
    size_t local_read_blocks;
    size_t local_hit_blocks;
    std::vector<size_t> per_tier_hit_blocks;
    // 用量快照（替代原先的散装字段）
    UsageSnapshot usage;
    // ...
};
```

这需要同步修改 `HitRateTracker::ExportHitRates` 中对 `ReadRecord` 字段的引用。

## 依赖关系

```
SubmitReadRecord
    └── TakeInstanceSnapshot (new)
            ├── TieredPolicyGroup::policies[i]->size()
            └── group_config.instances() (稳定顺序)

HitRateTracker::ExportHitRates
    └── ReadRecord.usage.per_tier_blocks (or ReadRecord.per_tier_blocks)
```

## 实施步骤

| 步骤 | 内容 | 影响文件 |
|------|------|----------|
| 1 | 在 `types.h` 定义 `UsageSnapshot` | `config/types.h` |
| 2 | 在 `OptEvictionManager` 实现 `TakeInstanceSnapshot` | `manager/eviction_manager.h/.cc` |
| 3 | `SubmitReadRecord` 改用 `TakeInstanceSnapshot` | `manager/optimizer_runner.cc` |
| 4 | `SubmitReadRecord` 不再需要 `indexer_manager_->GetAllOptIndexers()` | `manager/optimizer_runner.h` (移除依赖) |
| 5 | （可选）`ReadRecord` 持有 `UsageSnapshot`，同步更新 `HitRateTracker` | `analysis/stats_record.h`, `tracker/hit_rate_tracker.cc` |

## 附带收益

1. **顺序稳定** — `blocks_per_instance` 使用 `group_config.instances()` 顺序，CSV 列不再漂移
2. **单次遍历** — `TakeInstanceSnapshot` 内部一次 loop 同时产出 total 和 per-tier
3. **职责清晰** — 用量采集集中在 `OptEvictionManager`，`OptimizerRunner` 只消费快照
4. **扩展友好** — 新增用量维度只需在 `UsageSnapshot` 加字段 + `TakeInstanceSnapshot` 填充
