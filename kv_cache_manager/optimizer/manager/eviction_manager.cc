#include "kv_cache_manager/optimizer/manager/eviction_manager.h"

#include <algorithm>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/eviction_policy/policy_factory.h"
namespace kv_cache_manager {
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
        // Write-through: 为每个 tier 各创建独立驱逐策略
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
            group.tier_configs.push_back(storage_configs[i]);
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

std::unordered_map<std::string, std::vector<BlockEntry *>>
OptEvictionManager::EvictByMode(const std::string &instance_id, const OptInstanceGroupConfig &instance_group_config) {
    std::unordered_map<std::string, std::vector<BlockEntry *>> all_evicted;

    if (eviction_config_.eviction_mode() == EvictionMode::EVICTION_MODE_UNSPECIFIED) {
        KVCM_LOG_WARN("Eviction mode is unspecified, no eviction performed for instance: %s", instance_id.c_str());
        return all_evicted;
    }

    // 构建驱逐任务列表：(tier_idx, excess)
    std::vector<std::pair<std::optional<size_t>, size_t>> tasks;
    if (instance_group_config.hierarchical_eviction_enabled()) {
        size_t num_tiers = instance_group_config.storages().size();
        for (size_t tier_idx = 0; tier_idx < num_tiers; ++tier_idx) {
            size_t excess = GetExcessUsage(instance_group_config, tier_idx);
            if (excess > 0) {
                KVCM_LOG_DEBUG("Hierarchical eviction: tier %zu excess: %zu", tier_idx, excess);
                tasks.emplace_back(tier_idx, excess);
            }
        }
    } else {
        size_t excess = GetExcessUsage(instance_group_config, std::nullopt);
        if (excess > 0) {
            KVCM_LOG_DEBUG("Non-hierarchical eviction: excess: %zu", excess);
            tasks.emplace_back(std::nullopt, excess);
        }
    }

    // 执行驱逐并合并结果
    for (const auto &[tier_idx, excess] : tasks) {
        auto tier_evicted = DispatchEviction(instance_id, instance_group_config, tier_idx, excess);
        for (auto &[inst_id, blocks] : tier_evicted) {
            auto &vec = all_evicted[inst_id];
            vec.insert(vec.end(), blocks.begin(), blocks.end());
        }
    }

    return all_evicted;
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
        KVCM_LOG_DEBUG("GroupRough eviction: tier %zu, excess: %zu", tier_idx.value(), excess);
    } else {
        KVCM_LOG_DEBUG("GroupRough eviction: group %s, excess: %zu", group_name.c_str(), excess);
    }

    // 循环驱逐直到达到 excess 数量，轮询所有实例
    size_t total_evicted = 0;
    size_t round = 0;
    while (total_evicted < excess) {
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

            // 每轮驱逐 eviction_batch_size_per_instance_ 个块
            auto instance_evicted_blocks =
                eviction_policy->EvictBlocks(eviction_config_.eviction_batch_size_per_instance());
            if (!instance_evicted_blocks.empty()) {
                any_evicted_this_round = true;
                total_evicted += instance_evicted_blocks.size();
                evict_blocks[instance_id_in_group].insert(evict_blocks[instance_id_in_group].end(),
                                                          instance_evicted_blocks.begin(),
                                                          instance_evicted_blocks.end());
                KVCM_LOG_DEBUG("Round %zu: Evicted %zu blocks from instance: %s (total: %zu/%zu)",
                               round,
                               instance_evicted_blocks.size(),
                               instance_id_in_group.c_str(),
                               total_evicted,
                               excess);
            }
        }
        // 如果这一轮没有任何实例驱逐到块，说明已经无可驱逐的块了，退出循环
        if (!any_evicted_this_round) {
            KVCM_LOG_WARN("No more blocks can be evicted from any instance in group: %s (evicted: %zu, required: %zu)",
                          group_name.c_str(),
                          total_evicted,
                          excess);
            break;
        }
    }
    KVCM_LOG_DEBUG("Eviction completed for group: %s, total evicted: %zu, required: %zu, rounds: %zu",
                   group_name.c_str(),
                   total_evicted,
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
        KVCM_LOG_DEBUG("Instance%s eviction: instance %s, tier %zu, excess: %zu",
                       precise ? "Precise" : "Rough",
                       instance_id.c_str(),
                       tier_idx.value(),
                       excess);
    } else {
        KVCM_LOG_DEBUG("Instance%s eviction: instance %s, excess: %zu",
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

    size_t total_evicted = 0;
    size_t round = 0;
    while (total_evicted < excess) {
        round++;
        int32_t evict_count = eviction_config_.eviction_batch_size_per_instance();
        if (precise) {
            evict_count = std::min(evict_count, static_cast<int32_t>(excess - total_evicted));
        }
        auto round_evicted_blocks = eviction_policy->EvictBlocks(evict_count);
        if (round_evicted_blocks.empty()) {
            KVCM_LOG_WARN("No more blocks can be evicted from instance: %s (evicted: %zu, required: %zu)",
                          instance_id.c_str(),
                          total_evicted,
                          excess);
            break;
        }
        evict_blocks[instance_id].insert(
            evict_blocks[instance_id].end(), round_evicted_blocks.begin(), round_evicted_blocks.end());
        total_evicted += round_evicted_blocks.size();
        KVCM_LOG_DEBUG("Round %zu: Evicted %zu blocks from instance: %s (total: %zu/%zu)",
                       round,
                       round_evicted_blocks.size(),
                       instance_id.c_str(),
                       total_evicted,
                       excess);
    }
    return evict_blocks;
}

size_t OptEvictionManager::GetCurrentGroupUsage(const OptInstanceGroupConfig &instance_group_config,
                                                std::optional<size_t> tier_idx) const {
    size_t total = 0;
    for (const auto &instance_config : instance_group_config.instances()) {
        auto it = instance_tiered_policy_map_.find(instance_config.instance_id());
        if (it == instance_tiered_policy_map_.end())
            continue;
        if (tier_idx.has_value()) {
            // 分层模式：累加指定 tier 的用量
            if (tier_idx.value() < it->second.policies.size()) {
                total += it->second.policies[tier_idx.value()]->size();
            }
        } else {
            // 非分层模式：累加 shared_policy 的用量
            total += it->second.shared_policy()->size();
        }
    }
    return total;
}

size_t OptEvictionManager::GetExcessUsage(const OptInstanceGroupConfig &instance_group_config,
                                          std::optional<size_t> tier_idx) const {
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
    size_t current_used = GetCurrentGroupUsage(instance_group_config, tier_idx);
    size_t quota = static_cast<size_t>(capacity * instance_group_config.used_percentage());
    return current_used > quota ? current_used - quota : 0;
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
