#include "kv_cache_manager/optimizer/manager/eviction_manager.h"

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

std::shared_ptr<EvictionPolicy>
OptEvictionManager::CreateAndRegisterEvictionPolicy(const OptInstanceConfig &instance_config,
                                                    const std::vector<OptTierConfig> &storage_configs,
                                                    bool hierarchical_eviction_enabled) {
    // 提供开启多层的配置：hierarchical_eviction_enabled
    // 开启时：目前只对第一层创建驱逐策略实例，后续层级不逐出
    // 关闭时：所有层级共享同一个驱逐队列，即只创建一个驱逐策略实例
    if (instance_eviction_policy_map_.find(instance_config.instance_id()) != instance_eviction_policy_map_.end()) {
        KVCM_LOG_WARN("Eviction policy already exists for instance_id: %s", instance_config.instance_id().c_str());
        return instance_eviction_policy_map_[instance_config.instance_id()];
    }
    std::shared_ptr<EvictionPolicy> eviction_policy;
    if (hierarchical_eviction_enabled) {
        // 分层驱逐，每层单独创建驱逐策略
        // 目前仅创建第一层的驱逐策略，后续层级不逐出
        eviction_policy = EvictionPolicyFactory::CreatePolicy(instance_config.eviction_policy_type(),
                                                              storage_configs.front().unique_name(),
                                                              eviction_config_.eviction_batch_size_per_instance(),
                                                              instance_config.eviction_policy_param());
    } else {
        // 非分层驱逐，所有层级共享同一个驱逐队列
        // 只创建一个驱逐策略实例
        eviction_policy = EvictionPolicyFactory::CreatePolicy(instance_config.eviction_policy_type(),
                                                              "shared",
                                                              eviction_config_.eviction_batch_size_per_instance(),
                                                              instance_config.eviction_policy_param());
    }
    if (!eviction_policy) {
        KVCM_LOG_ERROR("Failed to create eviction policy for instance_id: %s", instance_config.instance_id().c_str());
        return nullptr;
    }
    instance_eviction_policy_map_[instance_config.instance_id()] = eviction_policy;
    return eviction_policy;
}

std::unordered_map<std::string, std::vector<BlockEntry *>>
OptEvictionManager::EvictByMode(const std::string &instance_id, const OptInstanceGroupConfig &instance_group_config) {
    std::unordered_map<std::string, std::vector<BlockEntry *>> evict_blocks;

    switch (eviction_config_.eviction_mode()) {
    case EvictionMode::EVICTION_MODE_UNSPECIFIED: {
        KVCM_LOG_WARN("Eviction mode is unspecified, no eviction performed for instance: %s", instance_id.c_str());
        break;
    }
    case EvictionMode::EVICTION_MODE_GROUP_ROUGH: {
        // 基于实例组的粗略驱逐
        evict_blocks = EvictByGroupRough(instance_id, instance_group_config);
        break;
    }
    case EvictionMode::EVICTION_MODE_INSTANCE_ROUGH: {
        // 基于实例的粗略驱逐
        evict_blocks = EvictByInstanceRough(instance_id, instance_group_config);
        break;
    }
    case EvictionMode::EVICTION_MODE_INSTANCE_PRECISE: {
        // 基于实例的精确驱逐
        evict_blocks = EvictByInstancePrecise(instance_id, instance_group_config);
        break;
    }
    }
    return evict_blocks;
}

std::unordered_map<std::string, std::vector<BlockEntry *>>
OptEvictionManager::EvictByGroupRough(const std::string &instance_id,
                                      const OptInstanceGroupConfig &instance_group_config) {
    std::unordered_map<std::string, std::vector<BlockEntry *>> evict_blocks;
    auto group_name = instance_group_config.group_name();
    size_t excess_bytes = GetExcessUsageForInstanceInGroup(instance_group_config);
    if (excess_bytes == 0) {
        // 不需要驱逐
        return evict_blocks;
    }
    KVCM_LOG_DEBUG("Evicting blocks for group: %s, excess: %zu bytes", group_name.c_str(), excess_bytes);
    // 循环驱逐直到达到 excess 数量
    size_t total_evicted_bytes = 0;
    size_t round = 0;
    while (total_evicted_bytes < excess_bytes) {
        round++;
        bool any_evicted_this_round = false;
        for (const auto &instance_config : instance_group_config.instances()) {

            auto instance_id_in_group = instance_config.instance_id();
            auto it = instance_eviction_policy_map_.find(instance_id_in_group);
            if (it == instance_eviction_policy_map_.end()) {
                KVCM_LOG_WARN("Eviction policy not found for instance: %s", instance_id_in_group.c_str());
                continue;
            }
            auto eviction_policy = it->second;

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
                               excess_bytes);
            }
        }
        // 如果这一轮没有任何实例驱逐到块，说明已经无可驱逐的块了，退出循环
        if (!any_evicted_this_round) {
            KVCM_LOG_WARN("No more blocks can be evicted from any instance in group: %s (evicted: %zu bytes, required: "
                          "%zu bytes)",
                          group_name.c_str(),
                          total_evicted_bytes,
                          excess_bytes);
            break;
        }
    }
    KVCM_LOG_DEBUG("Eviction completed for group: %s, total evicted: %zu bytes, required: %zu bytes, rounds: %zu",
                   group_name.c_str(),
                   total_evicted_bytes,
                   excess_bytes,
                   round);
    return evict_blocks;
}

std::unordered_map<std::string, std::vector<BlockEntry *>>
OptEvictionManager::EvictByInstanceRough(const std::string &instance_id,
                                         const OptInstanceGroupConfig &instance_group_config) {
    std::unordered_map<std::string, std::vector<BlockEntry *>> evict_blocks;
    size_t excess_bytes = GetExcessUsageForInstanceInGroup(instance_group_config);
    if (excess_bytes == 0) {
        // 不需要驱逐
        return evict_blocks;
    }
    KVCM_LOG_DEBUG("Evicting blocks for instance: %s, excess: %zu bytes", instance_id.c_str(), excess_bytes);
    // Find bytes_per_block for the target instance
    int64_t bpb = 1;
    for (const auto &ic : instance_group_config.instances()) {
        if (ic.instance_id() == instance_id) {
            bpb = ic.bytes_per_block();
            break;
        }
    }
    // 循环驱逐直到达到 excess 数量
    size_t total_evicted_bytes = 0;
    size_t round = 0;
    auto it = instance_eviction_policy_map_.find(instance_id);
    if (it == instance_eviction_policy_map_.end()) {
        KVCM_LOG_ERROR("Eviction policy not found for instance: %s", instance_id.c_str());
        return evict_blocks;
    }
    auto eviction_policy = it->second;
    while (total_evicted_bytes < excess_bytes) {
        round++;
        // 每轮驱逐 eviction_batch_size_per_instance_ 个块
        auto round_evicted_blocks = eviction_policy->EvictBlocks(eviction_config_.eviction_batch_size_per_instance());
        if (round_evicted_blocks.empty()) {
            // 无法驱逐更多块了
            KVCM_LOG_WARN("No more blocks can be evicted from instance: %s (evicted: %zu bytes, required: %zu bytes)",
                          instance_id.c_str(),
                          total_evicted_bytes,
                          excess_bytes);
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
                       excess_bytes);
    }
    return evict_blocks;
}

std::unordered_map<std::string, std::vector<BlockEntry *>>
OptEvictionManager::EvictByInstancePrecise(const std::string &instance_id,
                                           const OptInstanceGroupConfig &instance_group_config) {
    std::unordered_map<std::string, std::vector<BlockEntry *>> evict_blocks;
    size_t excess_bytes = GetExcessUsageForInstanceInGroup(instance_group_config);
    if (excess_bytes == 0) {
        // 不需要驱逐
        return evict_blocks;
    }
    KVCM_LOG_DEBUG("Evicting blocks for instance: %s, excess: %zu bytes", instance_id.c_str(), excess_bytes);
    // Find bytes_per_block for the target instance
    int64_t bpb = 1;
    for (const auto &ic : instance_group_config.instances()) {
        if (ic.instance_id() == instance_id) {
            bpb = ic.bytes_per_block();
            break;
        }
    }
    auto it = instance_eviction_policy_map_.find(instance_id);
    if (it == instance_eviction_policy_map_.end()) {
        KVCM_LOG_ERROR("Eviction policy not found for instance: %s", instance_id.c_str());
        return evict_blocks;
    }
    size_t total_evicted_bytes = 0;
    size_t round = 0;
    auto eviction_policy = it->second;
    while (total_evicted_bytes < excess_bytes) {
        round++;
        const size_t remaining_bytes = excess_bytes - total_evicted_bytes;
        const size_t remaining_blocks = (remaining_bytes + static_cast<size_t>(bpb) - 1) / static_cast<size_t>(bpb);
        auto evict_count = static_cast<int32_t>(
            std::min(static_cast<size_t>(eviction_config_.eviction_batch_size_per_instance()), remaining_blocks));
        // 每轮驱逐 eviction_batch_size_per_instance_ 个块
        auto round_evicted_blocks = eviction_policy->EvictBlocks(evict_count);
        if (round_evicted_blocks.empty()) {
            // 无法驱逐更多块了
            KVCM_LOG_WARN("No more blocks can be evicted from instance: %s (evicted: %zu bytes, required: %zu bytes)",
                          instance_id.c_str(),
                          total_evicted_bytes,
                          excess_bytes);
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
                       excess_bytes);
    }
    return evict_blocks;
}

size_t OptEvictionManager::GetCurrentGroupUsageBytes(const OptInstanceGroupConfig &instance_group_config) const {
    size_t current_group_used_bytes = 0;

    for (const auto &instance_config : instance_group_config.instances()) {
        current_group_used_bytes += instance_eviction_policy_map_.at(instance_config.instance_id())->size() *
                                    static_cast<size_t>(instance_config.bytes_per_block());
    }
    return current_group_used_bytes;
}

size_t OptEvictionManager::GetExcessUsageForInstanceInGroup(const OptInstanceGroupConfig &instance_group_config) const {
    // group_capacity and tier capacity are stored in bytes
    int64_t group_capacity_bytes = 0;
    // TODO 多层容量计算的简单实现，后续优化
    if (instance_group_config.hierarchical_eviction_enabled()) {
        group_capacity_bytes = instance_group_config.storages().front().capacity();
    } else {
        group_capacity_bytes = instance_group_config.quota_capacity();
    }
    auto used_percentage = instance_group_config.used_percentage();
    size_t current_group_used_bytes = GetCurrentGroupUsageBytes(instance_group_config);
    size_t quota_bytes = static_cast<size_t>(group_capacity_bytes * used_percentage);
    if (current_group_used_bytes > quota_bytes) {
        return current_group_used_bytes - quota_bytes;
    }
    return 0;
}

size_t OptEvictionManager::GetCurrentInstanceUsage(const std::string &instance_id) const {
    size_t current_instance_used = 0;
    auto instance_it = instance_eviction_policy_map_.find(instance_id);
    if (instance_it == instance_eviction_policy_map_.end()) {
        KVCM_LOG_ERROR("Instance eviction policy not found for instance_id: %s", instance_id.c_str());
        return current_instance_used;
    }
    current_instance_used = instance_it->second->size();
    return current_instance_used;
}
} // namespace kv_cache_manager