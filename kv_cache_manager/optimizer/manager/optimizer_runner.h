#pragma once
#include <memory>
#include <string>

#include "kv_cache_manager/optimizer/analysis/result_structure.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/manager/eviction_manager.h"
#include "kv_cache_manager/optimizer/manager/indexer_manager.h"
#include "kv_cache_manager/optimizer/trace_converter/optimizer_schema_trace.h"

namespace kv_cache_manager {
class OptimizerRunner {
public:
    explicit OptimizerRunner(const std::shared_ptr<OptIndexerManager> &indexer_manager,
                             const std::shared_ptr<OptEvictionManager> &eviction_manager,
                             const std::unordered_map<std::string, std::shared_ptr<Result>> &result_map)
        : indexer_manager_(indexer_manager), eviction_manager_(eviction_manager), result_map_(result_map){};
    ~OptimizerRunner() = default;
    void Run(OptimizerConfig &config);
    void RunTraces(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces, bool rw_separation);
    void RunTrace(std::shared_ptr<OptimizerSchemaTrace> trace, bool rw_separation);

public:
    void HandleGetLocation(const GetLocationSchemaTrace &trace);
    void HandleWriteCache(const WriteCacheSchemaTrace &trace);
    void HandleDialogTurn(const DialogTurnSchemaTrace &trace);

private:
    std::shared_ptr<OptIndexerManager> indexer_manager_;
    std::shared_ptr<OptEvictionManager> eviction_manager_;
    std::unordered_map<std::string, std::shared_ptr<Result>> result_map_;
};
} // namespace kv_cache_manager