#pragma once
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/replay_instance_config.h"
#include "kv_cache_manager/optimizer/config/replay_instance_group_config.h"
#include "kv_cache_manager/optimizer/config/tier_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/base.h"
#include "kv_cache_manager/optimizer/tier_flow/tier_flow_recorder.h"
namespace kv_cache_manager {

struct TieredPolicyGroup {
    // 驱逐策略列表，按 tier 顺序从高到低排序。
    // - 非分层模式: 仅包含一个 name="shared" 的策略，通过 shared_policy() 访问。
    // - 分层模式:   每个 tier 各一个策略，通过 policies[tier_idx] 访问。
    std::vector<std::shared_ptr<EvictionPolicy>> policies;
    std::vector<OptTierConfig> tier_configs; // 对应的 tier 配置

    size_t tier_count() const { return policies.size(); }

    std::shared_ptr<EvictionPolicy> GetPolicyByIndex(size_t idx) const {
        return idx < policies.size() ? policies[idx] : nullptr;
    }

    // 非分层模式下的唯一策略。仅在 policies 包含单个 "shared" 策略时使用。
    const std::shared_ptr<EvictionPolicy> &shared_policy() const { return policies.front(); }

    // 测试及外部便捷入口：委托给所有层策略
    void OnBlockWritten(BlockEntry *block) {
        for (auto &p : policies) {
            p->OnBlockWritten(block);
        }
    }
    void OnBlockAccessedWithOptions(BlockEntry *block, int64_t timestamp, bool refresh_ttl_on_read) {
        for (auto &p : policies) {
            p->OnBlockAccessedWithOptions(block, timestamp, refresh_ttl_on_read);
        }
    }
};

class OptEvictionManager {
public:
    using EvictedBlocks = std::unordered_map<std::string, std::vector<BlockEntry *>>;
    struct EvictionResult {
        EvictedBlocks evicted_blocks;
        TierFlowRecorder tier_flow;
    };

    OptEvictionManager() = default;
    ~OptEvictionManager() = default;
    bool Init(const EvictionConfig &eviction_config);

    TieredPolicyGroup *CreateAndRegisterEvictionPolicy(const OptimizerReplayInstanceConfig &instance_config,
                                                       const std::vector<OptTierConfig> &storage_configs,
                                                       bool hierarchical_eviction_enabled = false);

    // 统一驱逐入口，内部根据 hierarchical_eviction_enabled 与 tier edge write_mode 处理分层/非分层逻辑
    // eviction_timestamp 仅在 cascading 降级时用来写入新 tier 的 TierStat，其他分支不使用
    EvictionResult EvictByMode(const std::string &instance_id,
                               const OptimizerReplayInstanceGroupConfig &instance_group_config,
                               int64_t eviction_timestamp);

    // 显式过期驱逐：遍历所有实例调用 EvictExpired()
    std::unordered_map<std::string, std::vector<BlockEntry *>>
    ActiveEvictExpired(const OptimizerReplayInstanceGroupConfig &instance_group_config, int64_t current_timestamp);

    // ---- 用量查询接口 ----

    // 统一超额计算(bytes)：tier_idx 有值表示分层(对标 storages[tier_idx].capacity)，nullopt 表示非分层(对标
    // quota_capacity)
    size_t GetExcessUsage(const OptimizerReplayInstanceGroupConfig &instance_group_config,
                          std::optional<size_t> tier_idx) const;

    // Group 用量(bytes)：tier_idx 有值表示指定 tier 的用量，nullopt 表示 shared_policy 用量
    size_t GetCurrentGroupUsageBytes(const OptimizerReplayInstanceGroupConfig &instance_group_config,
                                     std::optional<size_t> tier_idx = std::nullopt) const;

    // Instance 用量：物理存储总占用（累加所有 tier）
    size_t GetCurrentInstanceUsage(const std::string &instance_id) const;

    // Instance per-tier 用量明细
    std::vector<size_t> GetCurrentInstanceUsagePerTier(const std::string &instance_id) const;

    // 级联降级：将 blocks 复制到 tier_{next_idx} 的 location_map + LRU 队列
    void DemoteToNextTier(const std::string &instance_id,
                          size_t next_tier_idx,
                          const std::vector<BlockEntry *> &blocks,
                          int64_t timestamp,
                          const std::vector<TierFlowStrategy> &tier_flow_strategies,
                          TierFlowRecorder *tier_flow = nullptr);

private:
    // 驱逐模式分发：根据 eviction_mode 调用对应的驱逐实现
    EvictedBlocks DispatchEviction(const std::string &instance_id,
                                   const OptimizerReplayInstanceGroupConfig &instance_group_config,
                                   std::optional<size_t> tier_idx,
                                   size_t excess);

    // 驱逐核心实现
    // tier_idx: nullopt 表示非分层模式(使用 shared_policy)，有值表示分层模式(使用 policies[tier_idx])
    // excess: 需要驱逐的 bytes 数量
    EvictedBlocks EvictByGroupRough(const OptimizerReplayInstanceGroupConfig &instance_group_config,
                                    std::optional<size_t> tier_idx,
                                    size_t excess);
    // precise=false: 每轮固定 batch size; precise=true: 每轮 cap 到剩余所需数量
    EvictedBlocks EvictByInstance(const std::string &instance_id,
                                  const OptimizerReplayInstanceGroupConfig &instance_group_config,
                                  std::optional<size_t> tier_idx,
                                  size_t excess,
                                  bool precise);

private:
    EvictionConfig eviction_config_;
    std::unordered_map<std::string, TieredPolicyGroup> instance_tiered_policy_map_;
};

} // namespace kv_cache_manager
