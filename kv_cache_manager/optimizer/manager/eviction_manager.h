#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/instance_config.h"
#include "kv_cache_manager/optimizer/config/instance_group_config.h"
#include "kv_cache_manager/optimizer/config/tier_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/base.h"
namespace kv_cache_manager {
class OptEvictionManager {
public:
    OptEvictionManager() = default;
    ~OptEvictionManager() = default;
    bool Init(const EvictionConfig &eviction_config);

    std::shared_ptr<EvictionPolicy> CreateAndRegisterEvictionPolicy(const OptInstanceConfig &instance_config,
                                                                    const std::vector<OptTierConfig> &storage_configs,
                                                                    bool hierarchical_eviction_enabled = false);
    std::unordered_map<std::string, std::vector<BlockEntry *>>
    EvictByMode(const std::string &instance_id, const OptInstanceGroupConfig &instance_group_config);

    size_t GetCurrentGroupUsage(const OptInstanceGroupConfig &instance_group_config) const;
    size_t GetCurrentInstanceUsage(const std::string &instance_id) const;
    size_t GetExcessUsageForInstanceInGroup(const OptInstanceGroupConfig &instance_group_config) const;

private:
    std::unordered_map<std::string, std::vector<BlockEntry *>>
    EvictByGroupRough(const std::string &instance_id, const OptInstanceGroupConfig &instance_group_config);
    std::unordered_map<std::string, std::vector<BlockEntry *>>
    EvictByInstanceRough(const std::string &instance_id, const OptInstanceGroupConfig &instance_group_config);
    std::unordered_map<std::string, std::vector<BlockEntry *>>
    EvictByInstancePrecise(const std::string &instance_id, const OptInstanceGroupConfig &instance_group_config);

private:
    EvictionConfig eviction_config_;
    std::unordered_map<std::string, std::shared_ptr<EvictionPolicy>> instance_eviction_policy_map_;
};

} // namespace kv_cache_manager