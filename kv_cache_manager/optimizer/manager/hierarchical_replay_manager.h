#pragma once

#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/config/hierarchical_replay_config.h"
#include "kv_cache_manager/optimizer/config/insight_simulator_types.h"
#include "kv_cache_manager/optimizer/manager/engine_storage_pool_connector.h"
#include "kv_cache_manager/optimizer/manager/optimizer_manager.h"
#include "kv_cache_manager/optimizer/trace_loader/optimizer_schema_trace.h"

namespace kv_cache_manager {

struct HierarchicalGetCacheLocationRes {
    std::string trace_id;
    int64_t engine_hit_length = 0;
    int64_t storage_pool_hit_length = 0;
    int64_t total_hit_length = 0;
};

class HierarchicalReplayManager {
public:
    explicit HierarchicalReplayManager(const HierarchicalReplayConfig &config);
    ~HierarchicalReplayManager() = default;

    bool Init();
    void DirectRun();
    void RunTraces(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces);
    void RunTrace(const std::shared_ptr<OptimizerSchemaTrace> &trace);
    void AnalyzeResults();

    HierarchicalGetCacheLocationRes GetCacheLocation(const std::string &engine_instance_id,
                                                     const std::string &trace_id,
                                                     int64_t timestamp,
                                                     const std::vector<int64_t> &block_ids,
                                                     int64_t input_len);
    WriteCacheRes WriteCache(const std::string &engine_instance_id,
                             const std::string &trace_id,
                             int64_t timestamp,
                             const std::vector<int64_t> &block_ids,
                             int64_t ttl_seconds = 0);
    WriteCacheRes WriteCacheWithTtlUs(const std::string &engine_instance_id,
                                      const std::string &trace_id,
                                      int64_t timestamp,
                                      const std::vector<int64_t> &block_ids,
                                      int64_t ttl_us);

private:
    struct CombinedReadRecord {
        std::string trace_id;
        std::string engine_instance_id;
        std::string storage_pool_instance_id;
        int64_t timestamp_ns = 0;
        size_t read_blocks = 0;
        size_t engine_hit_blocks = 0;
        size_t storage_pool_hit_blocks = 0;
        size_t input_tokens = 0;
        size_t block_size_tokens = 0;
    };

    struct CombinedWriteRecord {
        int64_t timestamp_ns = 0;
        size_t write_blocks = 0;
    };

    struct PendingWrite {
        int64_t timestamp_ns = 0;
        uint64_t sequence = 0;
        WriteCacheSchemaTrace trace;
    };

    struct PendingWriteCompare {
        bool operator()(const PendingWrite &lhs, const PendingWrite &rhs) const {
            if (lhs.timestamp_ns != rhs.timestamp_ns) {
                return lhs.timestamp_ns > rhs.timestamp_ns;
            }
            return lhs.sequence > rhs.sequence;
        }
    };

    bool ValidateAndBuildMappings();
    void ScheduleTraces(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) const;
    void RunTracesWithPrefixHitScheduling(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces);
    std::string
    ChoosePrefixHitEngineInstance(const std::vector<int64_t> &block_ids, int64_t timestamp, size_t request_idx) const;
    void HandleRequest(const RequestSchemaTrace &trace);
    void ScheduleRequestWrite(const RequestSchemaTrace &trace);
    void FlushPendingWritesThrough(int64_t timestamp_ns);
    void FlushAllPendingWrites();
    void RunPendingWrite(const WriteCacheSchemaTrace &trace);
    void ExportCombinedHitRates() const;
    const std::string &StoragePoolInstanceForEngine(const std::string &engine_instance_id) const;
    const StoragePoolFlowConfig &StoragePoolFlowForEngine(const std::string &engine_instance_id) const;

    HierarchicalReplayConfig config_;
    std::unique_ptr<OptimizerManager> engine_manager_;
    std::unique_ptr<OptimizerManager> storage_pool_manager_;
    std::unique_ptr<EngineStoragePoolConnector> engine_storage_pool_connector_;
    std::unordered_map<std::string, std::string> engine_to_storage_pool_;
    std::unordered_map<std::string, StoragePoolFlowConfig> engine_storage_pool_flow_;
    std::unordered_map<std::string, size_t> engine_block_size_;
    std::vector<std::string> sorted_engine_instance_ids_;
    std::vector<CombinedReadRecord> combined_read_records_;
    std::vector<CombinedWriteRecord> combined_write_records_;
    int64_t write_delay_ns_ = 1;
    uint64_t next_pending_write_sequence_ = 0;
    std::priority_queue<PendingWrite, std::vector<PendingWrite>, PendingWriteCompare> pending_writes_;
};

} // namespace kv_cache_manager
