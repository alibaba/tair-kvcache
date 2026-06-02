#include "kv_cache_manager/optimizer/manager/engine_storage_pool_connector.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/optimizer/manager/optimizer_manager.h"

namespace kv_cache_manager {

EngineStoragePoolConnector::EngineStoragePoolConnector(OptimizerManager *engine_manager,
                                                       OptimizerManager *storage_pool_manager)
    : engine_manager_(engine_manager), storage_pool_manager_(storage_pool_manager) {
    if (engine_manager_ == nullptr || storage_pool_manager_ == nullptr) {
        throw std::invalid_argument("EngineStoragePoolConnector requires initialized managers");
    }
}

EngineStoragePoolReadResult EngineStoragePoolConnector::ApplyReadFlow(const std::string &engine_instance_id,
                                                                      const std::string &storage_pool_instance_id,
                                                                      const std::string &trace_id,
                                                                      int64_t timestamp,
                                                                      const std::vector<int64_t> &block_ids,
                                                                      size_t engine_hit_blocks,
                                                                      int64_t input_len,
                                                                      const StoragePoolFlowConfig &flow) {
    EngineStoragePoolReadResult result;
    if (block_ids.empty()) {
        return result;
    }

    engine_hit_blocks = std::min(engine_hit_blocks, block_ids.size());
    BlockMaskVector storage_pool_mask(block_ids.size(), false);
    std::fill(storage_pool_mask.begin(), storage_pool_mask.begin() + engine_hit_blocks, true);
    const auto storage_pool_res = storage_pool_manager_->GetCacheLocation(storage_pool_instance_id,
                                                                          trace_id,
                                                                          timestamp,
                                                                          block_ids,
                                                                          storage_pool_mask,
                                                                          input_len,
                                                                          flow.local_read_touch_enabled(),
                                                                          false);
    result.storage_pool_hit_blocks =
        std::min(static_cast<size_t>(std::max<int64_t>(storage_pool_res.kvcm_hit_length, 0)),
                 block_ids.size() - engine_hit_blocks);

    if (result.storage_pool_hit_blocks > 0 && flow.promote_enabled()) {
        const size_t promoted_prefix_len = engine_hit_blocks + result.storage_pool_hit_blocks;
        std::vector<int64_t> promoted_prefix(block_ids.begin(), block_ids.begin() + promoted_prefix_len);
        auto promote_res = engine_manager_->WriteCacheWithTtlUs(
            engine_instance_id, trace_id, timestamp, promoted_prefix, 0, false, false);
        ApplyCascadingEvictions(storage_pool_instance_id, trace_id, timestamp, 0, flow, promote_res);
    }
    return result;
}

void EngineStoragePoolConnector::ApplyWriteFlow(const std::string &engine_instance_id,
                                                const std::string &storage_pool_instance_id,
                                                const std::string &trace_id,
                                                int64_t timestamp,
                                                const std::vector<int64_t> &block_ids,
                                                int64_t ttl_us,
                                                const StoragePoolFlowConfig &flow,
                                                const WriteCacheRes &engine_write_res) {
    (void)block_ids;
    if (flow.write_mode() == TierWriteMode::WRITE_THROUGH) {
        WriteMaterializedSequences(storage_pool_instance_id,
                                   trace_id,
                                   timestamp,
                                   engine_write_res.pool_source_write_sequences,
                                   ttl_us,
                                   flow.shadow_write_touch_enabled());
    } else if (flow.write_mode() == TierWriteMode::CASCADING) {
        ApplyCascadingEvictions(storage_pool_instance_id, trace_id, timestamp, ttl_us, flow, engine_write_res);
    } else if (flow.write_mode() == TierWriteMode::WRITE_THROUGH_SELECTIVE) {
        for (const auto &sequence : engine_write_res.pool_source_write_sequences) {
            const auto threshold_indices = engine_manager_->PoolSourceWriteTouchIndicesAtLeast(
                engine_instance_id, sequence.keys, flow.selective_write_threshold(), timestamp);
            const std::unordered_set<size_t> threshold_index_set(threshold_indices.begin(), threshold_indices.end());
            std::vector<size_t> selected_indices;
            for (const size_t idx : sequence.materialized_indices) {
                if (threshold_index_set.count(idx) > 0) {
                    selected_indices.push_back(idx);
                }
            }
            if (!selected_indices.empty()) {
                WriteMaterializedSequence(storage_pool_instance_id,
                                          trace_id,
                                          timestamp,
                                          sequence.keys,
                                          selected_indices,
                                          ttl_us,
                                          flow.shadow_write_touch_enabled());
            }
        }
    }
}

void EngineStoragePoolConnector::ApplyCascadingEvictions(const std::string &storage_pool_instance_id,
                                                         const std::string &trace_id,
                                                         int64_t timestamp,
                                                         int64_t ttl_us,
                                                         const StoragePoolFlowConfig &flow,
                                                         const WriteCacheRes &engine_write_res) {
    if (flow.write_mode() != TierWriteMode::CASCADING) {
        return;
    }
    WriteMaterializedSequences(
        storage_pool_instance_id, trace_id, timestamp, engine_write_res.evicted_materialized_sequences, ttl_us, false);
}

void EngineStoragePoolConnector::WriteMaterializedSequence(const std::string &storage_pool_instance_id,
                                                           const std::string &trace_id,
                                                           int64_t timestamp,
                                                           const std::vector<int64_t> &block_ids,
                                                           const std::vector<size_t> &materialized_indices,
                                                           int64_t ttl_us,
                                                           bool touch_existing) {
    if (!block_ids.empty() && !materialized_indices.empty()) {
        storage_pool_manager_->WriteCacheWithMaterializedIndices(
            storage_pool_instance_id, trace_id, timestamp, block_ids, materialized_indices, ttl_us, touch_existing);
    }
}

void EngineStoragePoolConnector::WriteMaterializedSequences(const std::string &storage_pool_instance_id,
                                                            const std::string &trace_id,
                                                            int64_t timestamp,
                                                            const std::vector<MaterializedKeySequence> &sequences,
                                                            int64_t ttl_us,
                                                            bool touch_existing) {
    for (const auto &sequence : sequences) {
        WriteMaterializedSequence(storage_pool_instance_id,
                                  trace_id,
                                  timestamp,
                                  sequence.keys,
                                  sequence.materialized_indices,
                                  ttl_us,
                                  touch_existing);
    }
}

} // namespace kv_cache_manager
