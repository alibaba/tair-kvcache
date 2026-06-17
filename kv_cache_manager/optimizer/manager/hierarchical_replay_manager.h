#pragma once

#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kv_cache_manager/optimizer/config/hierarchical_replay_config.h"
#include "kv_cache_manager/optimizer/config/insight_simulator_types.h"
#include "kv_cache_manager/optimizer/manager/optimizer_manager.h"
#include "kv_cache_manager/optimizer/p2p/tier_global_tracker.h"
#include "kv_cache_manager/optimizer/scheduler/infer_engine_scheduler.h"
#include "kv_cache_manager/optimizer/storage_pool/hash_storage_pool_manager.h"
#include "kv_cache_manager/optimizer/trace_loader/optimizer_schema_trace.h"

namespace kv_cache_manager {

struct HierarchicalGetCacheLocationRes {
    std::string trace_id;
    int64_t engine_hit_length = 0;
    int64_t peer_hit_length = 0;
    int64_t storage_pool_hit_length = 0;
    int64_t total_hit_length = 0;
};


struct ChooseBestEngineRes {
    std::string engine_instance_id;
    int64_t hit_count = 0;
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
                                                     int64_t input_len,
                                                     const std::string &query_type = "prefix_match");
    ChooseBestEngineRes ChooseBestEngine(const std::vector<int64_t> &block_ids,
                                         int64_t timestamp);

    std::vector<ChooseBestEngineRes> ChooseTopKEngines(const std::vector<int64_t> &block_ids,
                                                       int64_t timestamp,
                                                       int64_t top_k = 0);

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
        std::string storage_pool_id;
        int64_t timestamp_ns = 0;
        size_t read_blocks = 0;
        size_t engine_hit_blocks = 0;
        size_t peer_hit_blocks = 0;
        size_t storage_pool_hit_blocks = 0;
        size_t input_tokens = 0;
        size_t block_size_tokens = 0;
        std::string peer_source_infer_id;
    };

    struct CombinedWriteRecord {
        int64_t timestamp_ns = 0;
        size_t write_blocks = 0;
    };

    struct PoolWriteIoRecord {
        std::string trace_id;
        std::string engine_instance_id;
        std::string storage_pool_id;
        std::string reason;
        int64_t timestamp_ns = 0;
        size_t inserted_blocks = 0;
        size_t existing_blocks = 0;
        size_t block_size_tokens = 0;
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
    void RunTracesWithPrefixHitScheduling(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces);
    void HandleRequest(const RequestSchemaTrace &trace);
    void ScheduleRequestWrite(const RequestSchemaTrace &trace);
    void FlushPendingWritesThrough(int64_t timestamp_ns);
    void FlushAllPendingWrites();
    void RunPendingWrite(const WriteCacheSchemaTrace &trace);
    void ExportCombinedHitRates() const;
    void ExportReadIo() const;
    void ExportPoolWriteIo() const;
    const std::string &StoragePoolForEngine(const std::string &engine_instance_id) const;
    const std::string &EngineReadQueryTypeForEngine(const std::string &engine_instance_id) const;
    const StoragePoolFlowConfig &StoragePoolFlowForEngine(const std::string &engine_instance_id) const;
    const std::string &ClusterForEngine(const std::string &engine_instance_id) const;
    const std::vector<std::string> &InferIdsForCluster(const std::string &cluster_id) const;
    const std::vector<P2PReadFlowConfig> &P2PReadFlowsForCluster(const std::string &cluster_id) const;
    void ApplyEngineTierEvents(const std::vector<TierFlowKeyEvent> &events);
    void FillEngineFromHitIndices(const std::string &engine_instance_id,
                                  const std::string &storage_pool_id,
                                  const std::string &trace_id,
                                  int64_t timestamp,
                                  const std::vector<int64_t> &block_ids,
                                  const std::vector<size_t> &hit_indices,
                                  const StoragePoolFlowConfig &flow);
    TierGlobalPeerSelection ApplyP2PReadFlow(const std::string &engine_instance_id,
                                             const std::string &storage_pool_id,
                                             const std::string &trace_id,
                                             int64_t timestamp,
                                             const std::vector<int64_t> &block_ids,
                                             const P2PReadFlowConfig &flow,
                                             const StoragePoolFlowConfig &storage_pool_flow,
                                             std::vector<bool> *satisfied_mask);
    HashStoragePoolReadResult ReadStoragePool(const std::string &engine_instance_id,
                                              const std::string &storage_pool_id,
                                              const std::string &trace_id,
                                              int64_t timestamp,
                                              const std::vector<int64_t> &block_ids,
                                              const std::vector<size_t> &engine_hit_indices,
                                              int64_t input_len,
                                              const std::string &query_type,
                                              const StoragePoolFlowConfig &flow);
    WriteCacheRes WriteStoragePoolKeys(const std::string &engine_instance_id,
                                       const std::string &storage_pool_id,
                                       const std::string &reason,
                                       const std::string &trace_id,
                                       int64_t timestamp,
                                       const std::vector<int64_t> &keys,
                                       int64_t ttl_us,
                                       bool touch_existing);
    void ApplyStoragePoolWriteFlow(const std::string &engine_instance_id,
                                   const std::string &storage_pool_id,
                                   const std::string &trace_id,
                                   int64_t timestamp,
                                   int64_t ttl_us,
                                   const StoragePoolFlowConfig &flow,
                                   const WriteCacheRes &engine_write_res);
    void ApplyStoragePoolCascadingEvictions(const std::string &storage_pool_id,
                                            const std::string &engine_instance_id,
                                            const std::string &reason,
                                            const std::string &trace_id,
                                            int64_t timestamp,
                                            int64_t ttl_us,
                                            const StoragePoolFlowConfig &flow,
                                            const std::vector<int64_t> &evicted_keys);

    HierarchicalReplayConfig config_;
    std::unique_ptr<OptimizerManager> engine_manager_;
    std::unique_ptr<HashStoragePoolManager> storage_pool_manager_;
    std::unordered_map<std::string, std::string> engine_to_storage_pool_;
    std::unordered_map<std::string, std::string> engine_to_cluster_;
    std::unordered_map<std::string, std::vector<std::string>> cluster_infer_ids_;
    std::unordered_map<std::string, std::vector<P2PReadFlowConfig>> cluster_p2p_read_flows_;
    std::unordered_map<std::string, std::string> engine_read_query_type_;
    std::unordered_map<std::string, StoragePoolFlowConfig> engine_storage_pool_flow_;
    std::unordered_map<std::string, size_t> engine_block_size_;
    InferEngineScheduler infer_engine_scheduler_;
    TierGlobalTracker p2p_tracker_;
    std::vector<CombinedReadRecord> combined_read_records_;
    std::vector<CombinedWriteRecord> combined_write_records_;
    std::vector<PoolWriteIoRecord> pool_write_io_records_;
    int64_t write_delay_ns_ = 1;
    uint64_t next_pending_write_sequence_ = 0;
    std::priority_queue<PendingWrite, std::vector<PendingWrite>, PendingWriteCompare> pending_writes_;
};

} // namespace kv_cache_manager
