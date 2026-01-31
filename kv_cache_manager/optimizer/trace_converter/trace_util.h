#pragma once
#include <algorithm>
#include <memory>
#include <vector>

#include "kv_cache_manager/optimizer/trace_converter/optimizer_schema_trace.h"

namespace kv_cache_manager {
class TraceTimeSorter {
public:
    TraceTimeSorter() = default;
    ~TraceTimeSorter() = default;
    static void SortTracesByTimestamp(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces);
    static std::pair<int64_t, int64_t>
    GetTraceTimeRange(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces);
    static std::vector<std::shared_ptr<OptimizerSchemaTrace>> FilterTracesByTimeRange(
        const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces, int64_t start_time_us, int64_t end_time_us);

private:
    static bool CompareByTimestamp(const std::shared_ptr<OptimizerSchemaTrace> &a,
                                   const std::shared_ptr<OptimizerSchemaTrace> &b);
};

std::vector<int64_t> ApplyPrefixHash(const std::vector<int64_t> &hash_ids);
void AddTraceId(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces);
} // namespace kv_cache_manager