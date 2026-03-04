#pragma once
#include <memory>
#include <string>

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
                             const std::shared_ptr<StatsCollector> &stats_collector)
        : indexer_manager_(indexer_manager), eviction_manager_(eviction_manager), stats_collector_(stats_collector){};
    ~OptimizerRunner() = default;
    void Run(OptimizerConfig &config);
    void RunTraces(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces);
    void RunTrace(std::shared_ptr<OptimizerSchemaTrace> trace);

public:
    void HandleGetLocation(const GetLocationSchemaTrace &trace);
    void HandleWriteCache(const WriteCacheSchemaTrace &trace);
    void HandleDialogTurn(const DialogTurnSchemaTrace &trace);

private:
    ReadRecord BuildReadRecord(const std::string &instance_id, int64_t timestamp_us);

    std::shared_ptr<OptIndexerManager> indexer_manager_;
    std::shared_ptr<OptEvictionManager> eviction_manager_;
    std::shared_ptr<StatsCollector> stats_collector_;
};
} // namespace kv_cache_manager
