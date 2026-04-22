#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/instance_config.h"
#include "kv_cache_manager/optimizer/config/instance_group_config.h"
#include "kv_cache_manager/optimizer/config/tier_config.h"
#include "kv_cache_manager/optimizer/index/radix_tree_index.h"
#include "kv_cache_manager/optimizer/manager/eviction_manager.h"
namespace kv_cache_manager {
class OptIndexerManager {
public:
    using EvictedBlocks = std::unordered_map<std::string, std::vector<BlockEntry *>>;

    OptIndexerManager(const std::shared_ptr<OptEvictionManager> &eviction_manager);
    ~OptIndexerManager() = default;
    bool CreateOptIndexer(const OptInstanceConfig &instance_config,
                          const std::vector<OptTierConfig> &storage_configs,
                          bool hierarchical_eviction_enabled = false,
                          int64_t default_ttl_us = 0);

    std::shared_ptr<RadixTreeIndex> GetOptIndexer(const std::string &instance_id) const;
    std::unordered_map<std::string, std::shared_ptr<RadixTreeIndex>> GetAllOptIndexers() const;

    size_t GetOptIndexerSize() const;

public:
    void RegisterInstanceGroups(const std::unordered_map<std::string, OptInstanceGroupConfig> &instance_groups);
    void RegisterInstances(const std::unordered_map<std::string, OptInstanceConfig> &instances);

    // 仅做过期清理，不做容量驱逐；返回待清理的 block 列表
    EvictedBlocks EvictExpiredBeforeAccess(const std::string &instance_id, int64_t current_timestamp);

    // 仅做容量驱逐；返回待清理的 block 列表
    EvictedBlocks CheckAndEvict(const std::string &instance_id);

    // 统一清理驱逐后的 block，并触发节点清理
    void CleanEvictedBlocks(const EvictedBlocks &evicted_blocks,
                            int64_t eviction_timestamp,
                            bool use_logical_expire_time = false);

    // 获取容量使用情况
    size_t GetCurrentInstanceUsage(const std::string &instance_id) const;

    // 清空指定实例的缓存
    bool ClearCache(const std::string &instance_id);

    // 清空所有实例的缓存
    void ClearAllCaches();

private:
    std::unordered_map<std::string, std::shared_ptr<RadixTreeIndex>> opt_indexer_map_;
    std::shared_ptr<OptEvictionManager> eviction_manager_;

    std::unordered_map<std::string, OptInstanceGroupConfig> instance_group_configs_;
    std::unordered_map<std::string, OptInstanceConfig> instance_configs_;
};

} // namespace kv_cache_manager