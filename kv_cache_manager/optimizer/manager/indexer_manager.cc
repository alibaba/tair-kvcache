#include "kv_cache_manager/optimizer/manager/indexer_manager.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/logger.h"
namespace kv_cache_manager {
OptIndexerManager::OptIndexerManager(const std::shared_ptr<OptEvictionManager> &eviction_manager)
    : eviction_manager_(eviction_manager) {}

bool OptIndexerManager::CreateOptIndexer(const OptimizerReplayInstanceConfig &instance_config,
                                         const std::vector<OptTierConfig> &storage_configs,
                                         bool hierarchical_eviction_enabled,
                                         TierWriteMode tier_write_mode,
                                         int64_t default_ttl_ns,
                                         size_t selective_write_threshold,
                                         bool tier_access_propagation_enabled,
                                         std::vector<TierFlowStrategy> tier_flow_strategies) {

    std::string instance_id = instance_config.instance_id();
    auto indexer = GetOptIndexer(instance_id);
    if (indexer) {
        KVCM_LOG_WARN("Optimizer indexer already exists, instance_id: %s", instance_id.c_str());
        return false;
    }
    // 每个index实例对应一个instance_config，以及包含多层
    auto *policy_group = eviction_manager_->CreateAndRegisterEvictionPolicy(
        instance_config, storage_configs, hierarchical_eviction_enabled);
    if (!policy_group) {
        KVCM_LOG_ERROR("Failed to create eviction policy for instance_id: %s", instance_id.c_str());
        return false;
    }

    // 传递策略列表与层间流动策略给 RadixTreeIndex。
    // 非分层模式下 tier_write_mode 被忽略（tier_policies_ 中仅含单个 "shared" 策略，全层写等同单层写）
    const TierWriteMode effective_mode = hierarchical_eviction_enabled ? tier_write_mode : TierWriteMode::WRITE_THROUGH;
    indexer = std::make_shared<RadixTreeIndex>(instance_id,
                                               policy_group->policies,
                                               effective_mode,
                                               default_ttl_ns,
                                               selective_write_threshold,
                                               tier_access_propagation_enabled,
                                               std::move(tier_flow_strategies));

    opt_indexer_map_[instance_id] = indexer;
    KVCM_LOG_INFO("Create optimizer indexer success, instance_id: %s", instance_id.c_str());
    return true;
}
std::shared_ptr<RadixTreeIndex> OptIndexerManager::GetOptIndexer(const std::string &instance_id) const {
    auto iter = opt_indexer_map_.find(instance_id);
    if (iter != opt_indexer_map_.end()) {
        return iter->second;
    }
    return nullptr;
}

std::unordered_map<std::string, std::shared_ptr<RadixTreeIndex>> OptIndexerManager::GetAllOptIndexers() const {
    return opt_indexer_map_;
}

size_t OptIndexerManager::GetOptIndexerSize() const { return opt_indexer_map_.size(); }

size_t OptIndexerManager::GetInstanceBlockSize(const std::string &instance_id) const {
    auto it = instance_configs_.find(instance_id);
    if (it == instance_configs_.end() || it->second.block_size() <= 0) {
        return 0;
    }
    return static_cast<size_t>(it->second.block_size());
}

void OptIndexerManager::RegisterInstanceGroups(
    const std::unordered_map<std::string, OptimizerReplayInstanceGroupConfig> &instance_groups) {
    instance_group_configs_ = instance_groups;
}
void OptIndexerManager::RegisterInstances(
    const std::unordered_map<std::string, OptimizerReplayInstanceConfig> &instances) {
    instance_configs_ = instances;
}

const OptimizerReplayInstanceGroupConfig *
OptIndexerManager::FindInstanceGroupConfig(const std::string &instance_id) const {
    auto instance_it = instance_configs_.find(instance_id);
    if (instance_it == instance_configs_.end()) {
        KVCM_LOG_ERROR("Instance config not found for instance_id: %s", instance_id.c_str());
        return nullptr;
    }
    const auto &instance_config = instance_it->second;

    auto group_it = instance_group_configs_.find(instance_config.instance_group_name());
    if (group_it == instance_group_configs_.end()) {
        KVCM_LOG_ERROR("Instance group config not found for group_name: %s",
                       instance_config.instance_group_name().c_str());
        return nullptr;
    }
    return &group_it->second;
}

OptIndexerManager::EvictedBlocks OptIndexerManager::EvictExpiredBeforeAccess(const std::string &instance_id,
                                                                             int64_t current_timestamp) {
    EvictedBlocks empty_result;
    const auto *group_config = FindInstanceGroupConfig(instance_id);
    if (!group_config) {
        return empty_result;
    }

    return eviction_manager_->ActiveEvictExpired(*group_config, current_timestamp);
}

OptIndexerManager::EvictedBlocks OptIndexerManager::CheckAndEvict(const std::string &instance_id,
                                                                  int64_t eviction_timestamp) {
    EvictedBlocks empty_result;
    const auto *group_config = FindInstanceGroupConfig(instance_id);
    if (!group_config) {
        return empty_result;
    }

    // 统一驱逐入口：内部根据 hierarchical_eviction_enabled 与 tier edge write_mode 决定分支
    return eviction_manager_->EvictByMode(instance_id, *group_config, eviction_timestamp);
}

void OptIndexerManager::CleanEvictedBlocks(const EvictedBlocks &evicted_blocks,
                                           int64_t eviction_timestamp,
                                           bool use_logical_expire_time) {
    if (evicted_blocks.empty()) {
        return;
    }
    for (const auto &[inst_id, blocks] : evicted_blocks) {
        auto indexer = GetOptIndexer(inst_id);
        if (!indexer) {
            continue;
        }
        std::vector<BlockEntry *> truly_evicted;
        std::unordered_set<BlockEntry *> seen;
        for (auto *block : blocks) {
            if (block != nullptr && block->location_map.empty() && seen.insert(block).second) {
                truly_evicted.push_back(block);
            }
        }
        if (!truly_evicted.empty()) {
            // 传入 eviction_timestamp + use_logical_expire_time 标志，
            // 由 CleanEmptyBlocks 内部 per-block 计算各自的 logical expire time，
            indexer->CleanEmptyBlocks(truly_evicted, eviction_timestamp, use_logical_expire_time);
        }
    }
}

size_t OptIndexerManager::GetCurrentInstanceUsage(const std::string &instance_id) const {
    return eviction_manager_->GetCurrentInstanceUsage(instance_id);
}

bool OptIndexerManager::ClearCache(const std::string &instance_id) {
    auto indexer = GetOptIndexer(instance_id);
    if (!indexer) {
        KVCM_LOG_ERROR("Optimizer indexer not found for instance_id: %s", instance_id.c_str());
        return false;
    }

    indexer->Clear();
    KVCM_LOG_INFO("Cleared cache for instance_id: %s", instance_id.c_str());
    return true;
}

void OptIndexerManager::ClearAllCaches() {
    for (const auto &[instance_id, indexer] : opt_indexer_map_) {
        if (indexer) {
            indexer->Clear();
            KVCM_LOG_INFO("Cleared cache for instance_id: %s", instance_id.c_str());
        }
    }
    KVCM_LOG_INFO("Cleared all caches for %zu instances", opt_indexer_map_.size());
}

} // namespace kv_cache_manager
