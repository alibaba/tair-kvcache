#include "kv_cache_manager/optimizer/manager/eviction_manager.h"

#include <algorithm>
#include <unordered_set>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/eviction_policy/policy_factory.h"
#include "kv_cache_manager/optimizer/index/radix_tree_index.h"
namespace kv_cache_manager {
namespace {
bool IsCascadingEdge(const std::vector<TierFlowStrategy> &tier_flow_strategies, size_t edge_idx) {
    return edge_idx < tier_flow_strategies.size() &&
           tier_flow_strategies[edge_idx].write_mode == TierWriteMode::CASCADING;
}
} // namespace

bool OptEvictionManager::Init(const EvictionConfig &eviction_config) {
    eviction_config_ = eviction_config;
    if (eviction_config_.eviction_mode() == EvictionMode::EVICTION_MODE_UNSPECIFIED) {
        KVCM_LOG_ERROR("Eviction mode is unspecified.");
        return false;
    }
    if (eviction_config_.eviction_mode() == EvictionMode::EVICTION_MODE_GROUP_ROUGH ||
        eviction_config_.eviction_mode() == EvictionMode::EVICTION_MODE_INSTANCE_ROUGH) {
        if (eviction_config_.eviction_batch_size_per_instance() <= 0) {
            KVCM_LOG_ERROR("Eviction batch size per instance must be valid for rough eviction modes.");
            return false;
        }
    }
    return true;
}

TieredPolicyGroup *
OptEvictionManager::CreateAndRegisterEvictionPolicy(const OptInstanceConfig &instance_config,
                                                    const std::vector<OptTierConfig> &storage_configs,
                                                    bool hierarchical_eviction_enabled) {
    auto it = instance_tiered_policy_map_.find(instance_config.instance_id());
    if (it != instance_tiered_policy_map_.end()) {
        KVCM_LOG_WARN("Eviction policy already exists for instance_id: %s", instance_config.instance_id().c_str());
        return &it->second;
    }

    TieredPolicyGroup group;

    if (hierarchical_eviction_enabled) {
        // 分层：为每个 tier 各创建独立驱逐策略
        size_t num_tiers = storage_configs.size();
        for (size_t i = 0; i < num_tiers; ++i) {
            auto policy = EvictionPolicyFactory::CreatePolicy(instance_config.eviction_policy_type(),
                                                              storage_configs[i].unique_name(),
                                                              eviction_config_.eviction_batch_size_per_instance(),
                                                              instance_config.eviction_policy_param());
            if (!policy) {
                KVCM_LOG_ERROR("Failed to create eviction policy for tier: %s",
                               storage_configs[i].unique_name().c_str());
                return nullptr;
            }
            group.policies.push_back(std::move(policy));
        }
    } else {
        // 非分层: 单 "shared" 策略
        auto policy = EvictionPolicyFactory::CreatePolicy(instance_config.eviction_policy_type(),
                                                          "shared",
                                                          eviction_config_.eviction_batch_size_per_instance(),
                                                          instance_config.eviction_policy_param());
        if (!policy) {
            KVCM_LOG_ERROR("Failed to create eviction policy for instance_id: %s",
                           instance_config.instance_id().c_str());
            return nullptr;
        }
        group.policies.push_back(std::move(policy));
    }

    auto [inserted_it, success] = instance_tiered_policy_map_.emplace(instance_config.instance_id(), std::move(group));
    return &inserted_it->second;
}

std::unordered_map<std::string, std::vector<BlockEntry *>> OptEvictionManager::EvictByMode(
    const std::string &instance_id, const OptInstanceGroupConfig &instance_group_config, int64_t eviction_timestamp) {
    std::unordered_map<std::string, std::vector<BlockEntry *>> all_evicted;

    if (eviction_config_.eviction_mode() == EvictionMode::EVICTION_MODE_UNSPECIFIED) {
        KVCM_LOG_WARN("Eviction mode is unspecified, no eviction performed for instance: %s", instance_id.c_str());
        return all_evicted;
    }

    const bool hierarchical = instance_group_config.hierarchical_eviction_enabled();
    const auto tier_flow_strategies = instance_group_config.tier_flow_strategies();

    if (hierarchical) {
        // 分层模式：tier 0 → tier_{N-1} 串行。只有 edge write_mode=cascading 时，
        // tier i 驱逐出的 block 才会写入 tier i+1；write-through/selective edge 不做驱逐下沉。
        const size_t num_tiers = instance_group_config.storages().size();
        for (size_t tier_idx = 0; tier_idx < num_tiers; ++tier_idx) {
            size_t excess = GetExcessUsage(instance_group_config, tier_idx);
            if (excess == 0) {
                continue;
            }
            KVCM_LOG_DEBUG("Hierarchical eviction: tier %zu excess: %zu bytes", tier_idx, excess);
            auto tier_evicted = DispatchEviction(instance_id, instance_group_config, tier_idx, excess);

            const bool demote_to_next_tier =
                tier_idx + 1 < num_tiers && IsCascadingEdge(tier_flow_strategies, tier_idx);
            for (auto &[inst_id, blocks] : tier_evicted) {
                if (demote_to_next_tier && !blocks.empty()) {
                    DemoteBlocksToTierChain(inst_id, tier_idx + 1, blocks, eviction_timestamp, tier_flow_strategies);
                }
                auto &vec = all_evicted[inst_id];
                vec.insert(vec.end(), blocks.begin(), blocks.end());
            }
        }
        return all_evicted;
    }

    // 非分层分支：shared 策略按 group quota 驱逐
    std::vector<std::pair<std::optional<size_t>, size_t>> tasks;
    size_t excess = GetExcessUsage(instance_group_config, std::nullopt);
    if (excess > 0) {
        KVCM_LOG_DEBUG("Non-hierarchical eviction: excess: %zu bytes", excess);
        tasks.emplace_back(std::nullopt, excess);
    }

    for (const auto &[tier_idx, excess] : tasks) {
        auto tier_evicted = DispatchEviction(instance_id, instance_group_config, tier_idx, excess);
        for (auto &[inst_id, blocks] : tier_evicted) {
            auto &vec = all_evicted[inst_id];
            vec.insert(vec.end(), blocks.begin(), blocks.end());
        }
    }

    return all_evicted;
}

void OptEvictionManager::DemoteBlocksToTierChain(const std::string &instance_id,
                                                 size_t start_tier_idx,
                                                 const std::vector<BlockEntry *> &blocks,
                                                 int64_t timestamp,
                                                 const std::vector<TierFlowStrategy> &tier_flow_strategies) {
    auto it = instance_tiered_policy_map_.find(instance_id);
    if (it == instance_tiered_policy_map_.end()) {
        KVCM_LOG_WARN("DemoteBlocksToTierChain: eviction policy not found for instance: %s", instance_id.c_str());
        return;
    }
    if (start_tier_idx >= it->second.policies.size()) {
        // 没有下一层 → 彻底丢弃，Demote 无操作
        return;
    }
    for (BlockEntry *block : blocks) {
        AppendBlockToTierChain(block, start_tier_idx, it->second.policies, tier_flow_strategies, timestamp);
    }
}

std::unordered_map<std::string, std::vector<BlockEntry *>>
OptEvictionManager::DispatchEviction(const std::string &instance_id,
                                     const OptInstanceGroupConfig &instance_group_config,
                                     std::optional<size_t> tier_idx,
                                     size_t excess) {
    switch (eviction_config_.eviction_mode()) {
    case EvictionMode::EVICTION_MODE_GROUP_ROUGH:
        return EvictByGroupRough(instance_group_config, tier_idx, excess);
    case EvictionMode::EVICTION_MODE_INSTANCE_ROUGH:
        return EvictByInstance(instance_id, instance_group_config, tier_idx, excess, false);
    case EvictionMode::EVICTION_MODE_INSTANCE_PRECISE:
        return EvictByInstance(instance_id, instance_group_config, tier_idx, excess, true);
    default:
        return {};
    }
}

std::unordered_map<std::string, std::vector<BlockEntry *>> OptEvictionManager::EvictByGroupRough(
    const OptInstanceGroupConfig &instance_group_config, std::optional<size_t> tier_idx, size_t excess) {
    std::unordered_map<std::string, std::vector<BlockEntry *>> evict_blocks;
    auto group_name = instance_group_config.group_name();

    if (tier_idx.has_value()) {
        KVCM_LOG_DEBUG("GroupRough eviction: tier %zu, excess: %zu bytes", tier_idx.value(), excess);
    } else {
        KVCM_LOG_DEBUG("GroupRough eviction: group %s, excess: %zu bytes", group_name.c_str(), excess);
    }

    // 循环驱逐直到达到 excess 字节数，轮询所有实例
    size_t total_evicted_bytes = 0;
    size_t round = 0;
    while (total_evicted_bytes < excess) {
        round++;
        bool any_evicted_this_round = false;
        for (const auto &instance_config : instance_group_config.instances()) {
            auto instance_id_in_group = instance_config.instance_id();
            auto it = instance_tiered_policy_map_.find(instance_id_in_group);
            if (it == instance_tiered_policy_map_.end()) {
                KVCM_LOG_WARN("Eviction policy not found for instance: %s", instance_id_in_group.c_str());
                continue;
            }

            // 根据 tier_idx 选择策略：有值用分层策略，无值用 shared_policy
            if (tier_idx.has_value() && tier_idx.value() >= it->second.policies.size()) {
                continue; // 该实例没有这个 tier 的策略
            }
            auto &eviction_policy =
                tier_idx.has_value() ? it->second.policies[tier_idx.value()] : it->second.shared_policy();
            if (!eviction_policy->NeedCapacityEviction()) {
                continue;
            }

            // 每轮驱逐 eviction_batch_size_per_instance_ 个块
            auto instance_evicted_blocks =
                eviction_policy->EvictBlocks(eviction_config_.eviction_batch_size_per_instance());
            if (!instance_evicted_blocks.empty()) {
                any_evicted_this_round = true;
                const size_t evicted_bytes =
                    instance_evicted_blocks.size() * static_cast<size_t>(instance_config.bytes_per_block());
                total_evicted_bytes += evicted_bytes;
                evict_blocks[instance_id_in_group].insert(evict_blocks[instance_id_in_group].end(),
                                                          instance_evicted_blocks.begin(),
                                                          instance_evicted_blocks.end());
                KVCM_LOG_DEBUG("Round %zu: Evicted %zu blocks (%zu bytes) from instance: %s (total: %zu/%zu bytes)",
                               round,
                               instance_evicted_blocks.size(),
                               evicted_bytes,
                               instance_id_in_group.c_str(),
                               total_evicted_bytes,
                               excess);
            }
        }
        // 如果这一轮没有任何实例驱逐到块，说明已经无可驱逐的块了，退出循环
        if (!any_evicted_this_round) {
            KVCM_LOG_WARN("No more blocks can be evicted from any instance in group: %s (evicted: %zu bytes, required: "
                          "%zu bytes)",
                          group_name.c_str(),
                          total_evicted_bytes,
                          excess);
            break;
        }
    }
    KVCM_LOG_DEBUG("Eviction completed for group: %s, total evicted: %zu bytes, required: %zu bytes, rounds: %zu",
                   group_name.c_str(),
                   total_evicted_bytes,
                   excess,
                   round);
    return evict_blocks;
}

std::unordered_map<std::string, std::vector<BlockEntry *>>
OptEvictionManager::EvictByInstance(const std::string &instance_id,
                                    const OptInstanceGroupConfig &instance_group_config,
                                    std::optional<size_t> tier_idx,
                                    size_t excess,
                                    bool precise) {
    std::unordered_map<std::string, std::vector<BlockEntry *>> evict_blocks;

    if (tier_idx.has_value()) {
        KVCM_LOG_DEBUG("Instance%s eviction: instance %s, tier %zu, excess: %zu bytes",
                       precise ? "Precise" : "Rough",
                       instance_id.c_str(),
                       tier_idx.value(),
                       excess);
    } else {
        KVCM_LOG_DEBUG("Instance%s eviction: instance %s, excess: %zu bytes",
                       precise ? "Precise" : "Rough",
                       instance_id.c_str(),
                       excess);
    }

    auto it = instance_tiered_policy_map_.find(instance_id);
    if (it == instance_tiered_policy_map_.end()) {
        KVCM_LOG_ERROR("Eviction policy not found for instance: %s", instance_id.c_str());
        return evict_blocks;
    }
    if (tier_idx.has_value() && tier_idx.value() >= it->second.policies.size()) {
        KVCM_LOG_ERROR("Tier index %zu out of range for instance: %s", tier_idx.value(), instance_id.c_str());
        return evict_blocks;
    }

    // 根据 tier_idx 选择策略：有值用分层策略，无值用 shared_policy
    auto &eviction_policy = tier_idx.has_value() ? it->second.policies[tier_idx.value()] : it->second.shared_policy();
    if (!eviction_policy->NeedCapacityEviction()) {
        return evict_blocks;
    }

    // Find bytes_per_block for the target instance
    int64_t bpb = 1;
    for (const auto &ic : instance_group_config.instances()) {
        if (ic.instance_id() == instance_id) {
            bpb = ic.bytes_per_block();
            break;
        }
    }

    size_t total_evicted_bytes = 0;
    size_t round = 0;
    while (total_evicted_bytes < excess) {
        round++;
        int32_t evict_count = eviction_config_.eviction_batch_size_per_instance();
        if (precise) {
            const size_t remaining_bytes = excess - total_evicted_bytes;
            const size_t remaining_blocks = (remaining_bytes + static_cast<size_t>(bpb) - 1) / static_cast<size_t>(bpb);
            evict_count = static_cast<int32_t>(
                std::min(static_cast<size_t>(eviction_config_.eviction_batch_size_per_instance()), remaining_blocks));
        }
        auto round_evicted_blocks = eviction_policy->EvictBlocks(evict_count);
        if (round_evicted_blocks.empty()) {
            KVCM_LOG_WARN("No more blocks can be evicted from instance: %s (evicted: %zu bytes, required: %zu bytes)",
                          instance_id.c_str(),
                          total_evicted_bytes,
                          excess);
            break;
        }
        evict_blocks[instance_id].insert(
            evict_blocks[instance_id].end(), round_evicted_blocks.begin(), round_evicted_blocks.end());
        total_evicted_bytes += round_evicted_blocks.size() * static_cast<size_t>(bpb);
        KVCM_LOG_DEBUG("Round %zu: Evicted %zu blocks from instance: %s (total: %zu/%zu bytes)",
                       round,
                       round_evicted_blocks.size(),
                       instance_id.c_str(),
                       total_evicted_bytes,
                       excess);
    }
    return evict_blocks;
}

std::unordered_map<std::string, std::vector<BlockEntry *>>
OptEvictionManager::ActiveEvictExpired(const OptInstanceGroupConfig &instance_group_config, int64_t current_timestamp) {
    std::unordered_map<std::string, std::vector<BlockEntry *>> result;

    // TTL 是 block 级属性，过期时从所有 tier 中清除。
    // 遍历所有 tier policies 调用 EvictExpired()（各自从内部结构移除），去重后统一 clear location_map。
    for (const auto &instance_config : instance_group_config.instances()) {
        auto instance_id = instance_config.instance_id();
        auto it = instance_tiered_policy_map_.find(instance_id);
        if (it == instance_tiered_policy_map_.end()) {
            KVCM_LOG_WARN("[ActiveEvictExpired] Eviction policy not found for instance: %s", instance_id.c_str());
            continue;
        }
        auto &policies = it->second.policies;

        // 从所有 tier 收集过期 block（去重）
        std::unordered_set<BlockEntry *> expired_set;
        for (auto &policy : policies) {
            policy->AdvanceClock(current_timestamp);
            if (policy->size() == 0) {
                continue;
            }
            auto evicted = policy->EvictExpired();
            for (auto *block : evicted) {
                expired_set.insert(block);
            }
        }

        if (expired_set.empty()) {
            continue;
        }

        // block 级 TTL 过期：清除所有 tier 的 location
        for (auto *block : expired_set) {
            block->location_map.clear();
        }

        result[instance_id] = std::vector<BlockEntry *>(expired_set.begin(), expired_set.end());
        KVCM_LOG_DEBUG(
            "Actively evicted %zu expired blocks from instance: %s", expired_set.size(), instance_id.c_str());
    }

    return result;
}

size_t OptEvictionManager::GetCurrentGroupUsageBytes(const OptInstanceGroupConfig &instance_group_config,
                                                     std::optional<size_t> tier_idx) const {
    size_t total_bytes = 0;
    for (const auto &instance_config : instance_group_config.instances()) {
        auto it = instance_tiered_policy_map_.find(instance_config.instance_id());
        if (it == instance_tiered_policy_map_.end())
            continue;
        if (tier_idx.has_value()) {
            // 分层模式：累加指定 tier 的用量(bytes)
            if (tier_idx.value() < it->second.policies.size()) {
                total_bytes += it->second.policies[tier_idx.value()]->size() *
                               static_cast<size_t>(instance_config.bytes_per_block());
            }
        } else {
            // 非分层模式：累加 shared_policy 的用量(bytes)
            total_bytes += it->second.shared_policy()->size() * static_cast<size_t>(instance_config.bytes_per_block());
        }
    }
    return total_bytes;
}

size_t OptEvictionManager::GetExcessUsage(const OptInstanceGroupConfig &instance_group_config,
                                          std::optional<size_t> tier_idx) const {
    // group_capacity and tier capacity are stored in bytes
    int64_t capacity = 0;
    if (tier_idx.has_value()) {
        // 分层模式：该 tier 的独立容量
        if (tier_idx.value() >= instance_group_config.storages().size()) {
            return 0;
        }
        capacity = instance_group_config.storages()[tier_idx.value()].capacity();
    } else {
        // 非分层模式：group 整体配额
        capacity = instance_group_config.quota_capacity();
    }
    if (capacity < 0) {
        return 0;
    }
    size_t current_used_bytes = GetCurrentGroupUsageBytes(instance_group_config, tier_idx);
    size_t quota_bytes = static_cast<size_t>(capacity * instance_group_config.used_percentage());
    return current_used_bytes > quota_bytes ? current_used_bytes - quota_bytes : 0;
}

size_t OptEvictionManager::GetCurrentInstanceUsage(const std::string &instance_id) const {
    auto it = instance_tiered_policy_map_.find(instance_id);
    if (it == instance_tiered_policy_map_.end()) {
        KVCM_LOG_ERROR("Instance eviction policy not found for instance_id: %s", instance_id.c_str());
        return 0;
    }
    // 物理存储总占用：累加所有 tier 的 block 数
    // 非分层模式下只有一个 shared policy，效果等同于原实现
    size_t total = 0;
    for (const auto &policy : it->second.policies) {
        total += policy->size();
    }
    return total;
}

std::vector<size_t> OptEvictionManager::GetCurrentInstanceUsagePerTier(const std::string &instance_id) const {
    auto it = instance_tiered_policy_map_.find(instance_id);
    if (it == instance_tiered_policy_map_.end())
        return {};
    std::vector<size_t> result;
    result.reserve(it->second.policies.size());
    for (const auto &policy : it->second.policies) {
        result.push_back(policy->size());
    }
    return result;
}

} // namespace kv_cache_manager
