#include "kv_cache_manager/optimizer/manager/indexer_manager.h"

#include <algorithm>

#include "kv_cache_manager/common/logger.h"
namespace kv_cache_manager {
OptIndexerManager::OptIndexerManager(const std::shared_ptr<OptEvictionManager> &eviction_manager)
    : eviction_manager_(eviction_manager) {}

bool OptIndexerManager::CreateOptIndexer(const OptInstanceConfig &instance_config,
                                         const std::vector<OptTierConfig> &storage_configs,
                                         bool hierarchical_eviction_enabled) {

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

    // 传递策略列表给 RadixTreeIndex
    indexer = std::make_shared<RadixTreeIndex>(instance_id, policy_group->policies);

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

void OptIndexerManager::RegisterInstanceGroups(
    const std::unordered_map<std::string, OptInstanceGroupConfig> &instance_groups) {
    instance_group_configs_ = instance_groups;
}
void OptIndexerManager::RegisterInstances(const std::unordered_map<std::string, OptInstanceConfig> &instances) {
    instance_configs_ = instances;
}

bool OptIndexerManager::CheckAndEvict(const std::string &instance_id, int64_t eviction_timestamp) {

    auto instance_it = instance_configs_.find(instance_id);
    if (instance_it == instance_configs_.end()) {
        KVCM_LOG_ERROR("Instance config not found for instance_id: %s", instance_id.c_str());
        return false;
    }
    const auto &instance_config = instance_it->second;
    auto group_name = instance_config.instance_group_name();
    // 获取 group 配置
    auto group_it = instance_group_configs_.find(group_name);

    if (group_it == instance_group_configs_.end()) {
        KVCM_LOG_ERROR("Instance group config not found for group_name: %s",
                       instance_config.instance_group_name().c_str());
        return false;
    }
    const auto &group_config = group_it->second;

    // 统一驱逐入口：内部根据 hierarchical_eviction_enabled 决定分层/非分层模式
    auto evicted_blocks = eviction_manager_->EvictByMode(instance_id, group_config);

    // 清理 location_map 为空的 block (彻底驱逐)
    if (!evicted_blocks.empty()) {
        for (auto &[inst_id, blocks] : evicted_blocks) {
            auto indexer = GetOptIndexer(inst_id);
            KVCM_LOG_DEBUG("Evicted %zu blocks from instance_id: %s by CheckAndEvict", blocks.size(), inst_id.c_str());
            if (indexer) {
                // 只对 location_map 完全为空的 block 执行 CleanEmptyBlocks
                // (tier0 驱逐后 block 仍在 tier1，不应清理)
                std::vector<BlockEntry *> truly_evicted;
                for (auto *block : blocks) {
                    if (block->location_map.empty()) {
                        truly_evicted.push_back(block);
                    }
                }
                if (!truly_evicted.empty()) {
                    indexer->CleanEmptyBlocks(truly_evicted, eviction_timestamp);
                }
            }
        }
    }
    return !evicted_blocks.empty();
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
