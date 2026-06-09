#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/online_optimizer/config/optimizer_instance_group.h"
#include "kv_cache_manager/online_optimizer/config/optimizer_instance_info.h"
#include "kv_cache_manager/online_optimizer/indexer/cache_indexer.h"

namespace kv_cache_manager {

class OptimizerRegistryManager;

struct InstanceState {
    std::shared_ptr<const OptimizerInstanceInfo> instance_info;
    std::shared_ptr<const OptimizerInstanceGroup> instance_group;

    std::unique_ptr<CacheIndexer> indexer;

    int64_t size_full_only = 0;
    int64_t size_full_linear = 0;
    int32_t linear_step = 1;
    int64_t avg_bytes_per_block = 0;

    std::vector<int64_t> capacity_blocks;
    int64_t max_capacity_blocks = 0;
    int32_t primary_capacity_index = 0;

    std::mutex mutex;

    int64_t total_queries = 0;
    int64_t total_blocks_queried = 0;
    std::vector<int64_t> total_hits_per_capacity;
};

struct TraceQueryResult {
    int64_t cache_hit_count = 0;
    int64_t total_blocks = 0;
    std::vector<int64_t> hit_count_per_capacity;
    std::vector<double> capacity_gb;
    int64_t current_unique_keys = 0;
};

struct RegisterInstanceResult {
    std::vector<int64_t> capacity_blocks;
    int64_t avg_bytes_per_block = 0;
    int64_t size_full_only = 0;
    int64_t size_full_linear = 0;
};

struct InstanceSummary {
    std::string instance_id;
    std::string instance_group;
    int32_t block_size = 0;
    int64_t total_queries = 0;
    int64_t total_blocks_queried = 0;
    int64_t total_hits = 0;
    double hit_rate = 0.0;
    int64_t unique_keys = 0;
    int64_t avg_bytes_per_block = 0;
    int32_t linear_step = 0;
    int64_t peak_unique_keys = 0;
    int64_t eviction_count = 0;
    int64_t memory_usage_bytes = 0;
};

class OnlineOptimizerManager {
public:
    explicit OnlineOptimizerManager(std::shared_ptr<OptimizerRegistryManager> registry_manager);
    ~OnlineOptimizerManager() = default;

    OnlineOptimizerManager(const OnlineOptimizerManager &) = delete;
    OnlineOptimizerManager &operator=(const OnlineOptimizerManager &) = delete;

    ErrorCode RegisterInstance(const OptimizerInstanceInfo &instance_info,
                               const OptimizerInstanceGroup &instance_group,
                               RegisterInstanceResult &result);

    ErrorCode RemoveInstance(const std::string &instance_id);

    ErrorCode TraceQuery(const std::string &instance_id,
                         const std::vector<int64_t> &block_keys,
                         TraceQueryResult &result);

    ErrorCode ListInstances(const std::string &instance_group_filter,
                            std::vector<InstanceSummary> &summaries) const;

    ErrorCode ResetStats(const std::string &instance_id);

    ErrorCode GetInstanceState(const std::string &instance_id,
                               std::function<void(const InstanceState &)> visitor) const;

    // Recovery: reload persisted instances from registry
    ErrorCode Recover();

    std::shared_ptr<OptimizerRegistryManager> registry_manager() const { return registry_manager_; }

private:
    ErrorCode RegisterInstanceInternal(const OptimizerInstanceInfo &instance_info,
                                        const OptimizerInstanceGroup &instance_group,
                                        RegisterInstanceResult &result);

    static int64_t ComputeSizeForGroup(const std::vector<LocationSpecInfo> &specs,
                                       const std::vector<LocationSpecGroup> &groups,
                                       const std::string &group_name);

    static int64_t ComputeTotalSize(const std::vector<LocationSpecInfo> &specs);

    std::shared_ptr<OptimizerRegistryManager> registry_manager_;
    mutable std::shared_mutex instances_mutex_;
    std::unordered_map<std::string, std::shared_ptr<InstanceState>> instances_;
};

} // namespace kv_cache_manager
