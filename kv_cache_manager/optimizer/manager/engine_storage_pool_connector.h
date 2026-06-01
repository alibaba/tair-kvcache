#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kv_cache_manager/optimizer/config/hierarchical_replay_config.h"
#include "kv_cache_manager/optimizer/config/insight_simulator_types.h"
#include "kv_cache_manager/optimizer/config/types.h"

namespace kv_cache_manager {

class OptimizerManager;

struct EngineStoragePoolReadResult {
    size_t storage_pool_hit_blocks = 0;
};

class EngineStoragePoolConnector {
public:
    EngineStoragePoolConnector(OptimizerManager *engine_manager, OptimizerManager *storage_pool_manager);

    EngineStoragePoolReadResult ApplyReadFlow(const std::string &engine_instance_id,
                                              const std::string &storage_pool_instance_id,
                                              const std::string &trace_id,
                                              int64_t timestamp,
                                              const std::vector<int64_t> &block_ids,
                                              size_t engine_hit_blocks,
                                              int64_t input_len,
                                              const StoragePoolFlowConfig &flow);

    void ApplyWriteFlow(const std::string &engine_instance_id,
                        const std::string &storage_pool_instance_id,
                        const std::string &trace_id,
                        int64_t timestamp,
                        const std::vector<int64_t> &block_ids,
                        int64_t ttl_us,
                        const StoragePoolFlowConfig &flow,
                        const WriteCacheRes &engine_write_res);

    void ApplyCascadingEvictions(const std::string &storage_pool_instance_id,
                                 const std::string &trace_id,
                                 int64_t timestamp,
                                 int64_t ttl_us,
                                 const StoragePoolFlowConfig &flow,
                                 const WriteCacheRes &engine_write_res);

private:
    void WriteMaterializedSequence(const std::string &storage_pool_instance_id,
                                   const std::string &trace_id,
                                   int64_t timestamp,
                                   const std::vector<int64_t> &block_ids,
                                   const std::vector<size_t> &materialized_indices,
                                   int64_t ttl_us,
                                   bool touch_existing);

    void WriteMaterializedSequences(const std::string &storage_pool_instance_id,
                                    const std::string &trace_id,
                                    int64_t timestamp,
                                    const std::vector<MaterializedKeySequence> &sequences,
                                    int64_t ttl_us,
                                    bool touch_existing);

    OptimizerManager *engine_manager_ = nullptr;
    OptimizerManager *storage_pool_manager_ = nullptr;
};

} // namespace kv_cache_manager
