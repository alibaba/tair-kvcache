#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/analysis/stats_tracker.h"

namespace kv_cache_manager {

// ============================================================================
// Block 生命周期追踪器
// ============================================================================
class BlockLifecycleTracker : public StatsTracker {
public:
    BlockLifecycleTracker();

    void OnBlockBirth(const std::string &instance_id, BlockEntry *block, int64_t timestamp) override;
    void OnBlockEviction(const std::string &instance_id, BlockEntry *block, int64_t timestamp) override;
    void Finalize(const std::string &instance_id, int64_t final_timestamp) override;
    void Export(const std::string &instance_id, const OptimizerConfig &config) override;
    void Reset(const std::string &instance_id) override;

private:
    struct InstanceData {
        std::vector<std::shared_ptr<BlockLifecycleRecord>> records;
        std::unordered_map<int64_t, std::shared_ptr<BlockLifecycleRecord>> alive_blocks;
    };

    std::unordered_map<std::string, InstanceData> instance_data_;
};

} // namespace kv_cache_manager
