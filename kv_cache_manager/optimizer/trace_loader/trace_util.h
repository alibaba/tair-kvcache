#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/optimizer/trace_loader/optimizer_schema_trace.h"

namespace kv_cache_manager {
class TraceTimeSorter {
public:
    TraceTimeSorter() = default;
    ~TraceTimeSorter() = default;
    static void SortTracesByTimestamp(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces);
    static std::pair<int64_t, int64_t>
    GetTraceTimeRange(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces);
    static std::vector<std::shared_ptr<OptimizerSchemaTrace>> FilterTracesByTimeRange(
        const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces, int64_t start_time_ns, int64_t end_time_ns);

private:
    static bool CompareByTimestamp(const std::shared_ptr<OptimizerSchemaTrace> &a,
                                   const std::shared_ptr<OptimizerSchemaTrace> &b);
};

std::string DefaultTraceId(const OptimizerSchemaTrace &trace);
void EnsureTraceId(const std::shared_ptr<OptimizerSchemaTrace> &trace);
void AddTraceId(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces);
} // namespace kv_cache_manager
