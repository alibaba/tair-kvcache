#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "kv_cache_manager/mrc/online_mrc_config.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_group.h"
#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"
#include "kv_cache_manager/optimizer/liteHit/lite_hit.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"

namespace kv_cache_manager {

class MetricsRegistry;
class OnlineOptimizerManager;
struct RegisterInstanceResult;

// Optimizer-owned online MRC facts. Groups and instances are registered in
// the formal OnlineOptimizerManager/OptimizerRegistryManager. Request facts
// stay process-memory-only and are retained without a count or time limit.
class OnlineMrcFactRegistry {
public:
    OnlineMrcFactRegistry(const OnlineMrcConfig &config,
                          const std::vector<OptimizerInstanceGroup> &instance_groups,
                          std::shared_ptr<MetricsRegistry> metrics_registry,
                          std::shared_ptr<OnlineOptimizerManager> manager);

    // Creates every configured formal group before the stream starts.
    bool Init();

    bool Observe(const proto::optimizer::CacheEventBatch &batch);
    void ReportMetrics();

    size_t InstanceCount() const;
    size_t GroupCount() const;
    size_t GroupInstanceCount(const std::string &instance_group) const;
    size_t FactCount(const std::string &instance_id) const;
    uint64_t MetaGeneration(const std::string &instance_id) const;

    // Changes only the reporting projection grid. Existing formal groups,
    // LiteHit state, and all request facts remain unchanged.
    bool UpdateCapacityGrid(const std::vector<double> &capacity_gb_grid);
    std::vector<double> CapacityGrid() const;
    uint64_t ProjectionGeneration() const;

private:
    struct StoredFact {
        uint64_t input_token_len = 0;
        RequestFact fact;
    };

    struct InstanceContext {
        mutable std::mutex mutex;
        std::string instance_id;
        std::string instance_group;
        std::string serialized_instance_info;
        uint64_t meta_generation = 1;
        int64_t last_timestamp_ns = 0;
        uint64_t out_of_order_events = 0;
        int32_t block_size = 0;
        int64_t block_bytes = 0;
        bool enable_prefix_hash = true;
        LiteHit lite_hit;
        std::deque<StoredFact> facts;
    };

    struct GroupContext {
        std::set<std::string> instance_ids;
    };

    std::shared_ptr<InstanceContext> RegisterOrGetInstance(const proto::optimizer::CacheEventBatch &batch);
    bool RegisterFormalInstance(const proto::optimizer::CacheEventBatch &batch,
                                std::string &serialized_instance_info,
                                RegisterInstanceResult &result);
    void MoveInstanceToGroup(const std::string &instance_id,
                             const std::string &old_group,
                             const std::string &new_group);
    static bool ValidateCapacityGrid(const std::vector<double> &capacity_gb_grid);
    static uint64_t FactMemoryBytes(const std::deque<StoredFact> &facts);

    OnlineMrcConfig config_;
    std::vector<OptimizerInstanceGroup> instance_groups_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<OnlineOptimizerManager> manager_;
    mutable std::mutex contexts_mutex_;
    std::map<std::string, std::shared_ptr<InstanceContext>> contexts_;
    std::map<std::string, GroupContext> groups_;
    bool initialized_ = false;

    mutable std::mutex projection_mutex_;
    std::vector<double> capacity_gb_grid_;
    uint64_t projection_generation_ = 1;
};

} // namespace kv_cache_manager
