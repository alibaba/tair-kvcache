#pragma once
#include <memory>
#include <string>
#include <unordered_map>

#include "kv_cache_manager/optimizer/analysis/stats_collector.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/manager/eviction_manager.h"
#include "kv_cache_manager/optimizer/manager/indexer_manager.h"
#include "kv_cache_manager/optimizer/trace_loader/optimizer_schema_trace.h"

namespace kv_cache_manager {
class OptimizerRunner {
public:
    explicit OptimizerRunner(const std::shared_ptr<OptIndexerManager> &indexer_manager,
                             const std::shared_ptr<OptEvictionManager> &eviction_manager,
                             const std::shared_ptr<StatsCollector> &stats_collector,
                             const std::unordered_map<std::string, bool> &instance_group_ttl_disabled,
                             const std::unordered_map<std::string, bool> &instance_ttl_refresh_on_read)
        : indexer_manager_(indexer_manager)
        , eviction_manager_(eviction_manager)
        , stats_collector_(stats_collector)
        , instance_group_ttl_disabled_(instance_group_ttl_disabled)
        , instance_ttl_refresh_on_read_(instance_ttl_refresh_on_read){};
    ~OptimizerRunner() = default;
    void Run(OptimizerConfig &config);
    void RunTraces(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces);
    void RunTrace(std::shared_ptr<OptimizerSchemaTrace> trace);

public:
    void HandleGetLocation(const GetLocationSchemaTrace &trace);
    void HandleWriteCache(const WriteCacheSchemaTrace &trace);

private:
    std::shared_ptr<RadixTreeIndex> GetIndexer(const std::string &instance_id);
    void SubmitReadRecord(const std::string &instance_id,
                          const std::string &trace_id,
                          const std::vector<int64_t> &keys,
                          int64_t timestamp_ns,
                          const QueryHit &query_hit,
                          const std::shared_ptr<RadixTreeIndex> &indexer,
                          size_t local_read_block_num,
                          size_t remote_read_block_num);

    std::shared_ptr<OptIndexerManager> indexer_manager_;
    std::shared_ptr<OptEvictionManager> eviction_manager_;
    std::shared_ptr<StatsCollector> stats_collector_;
    std::unordered_map<std::string, bool> instance_group_ttl_disabled_;
    std::unordered_map<std::string, bool> instance_ttl_refresh_on_read_;
};
} // namespace kv_cache_manager
