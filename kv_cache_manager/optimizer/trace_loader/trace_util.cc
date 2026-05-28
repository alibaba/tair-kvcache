#include "kv_cache_manager/optimizer/trace_loader/trace_util.h"

namespace kv_cache_manager {

void TraceTimeSorter::SortTracesByTimestamp(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) {
    std::sort(traces.begin(), traces.end(), CompareByTimestamp);
}

std::pair<int64_t, int64_t>
TraceTimeSorter::GetTraceTimeRange(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) {
    if (traces.empty()) {
        return {0, 0};
    }
    int64_t min_time = traces.front()->timestamp_ns();
    int64_t max_time = traces.front()->timestamp_ns();
    for (const auto &trace : traces) {
        int64_t timestamp = trace->timestamp_ns();
        if (timestamp < min_time) {
            min_time = timestamp;
        }
        if (timestamp > max_time) {
            max_time = timestamp;
        }
    }
    return {min_time, max_time};
}

std::vector<std::shared_ptr<OptimizerSchemaTrace>> TraceTimeSorter::FilterTracesByTimeRange(
    const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces, int64_t start_time_ns, int64_t end_time_ns) {
    std::vector<std::shared_ptr<OptimizerSchemaTrace>> filtered_traces;
    for (const auto &trace : traces) {
        int64_t timestamp = trace->timestamp_ns();
        if (timestamp >= start_time_ns && timestamp <= end_time_ns) {
            filtered_traces.push_back(trace);
        }
    }
    SortTracesByTimestamp(filtered_traces);
    return filtered_traces;
}

bool TraceTimeSorter::CompareByTimestamp(const std::shared_ptr<OptimizerSchemaTrace> &a,
                                         const std::shared_ptr<OptimizerSchemaTrace> &b) {
    return a->timestamp_ns() < b->timestamp_ns();
}

void AddTraceId(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) {
    for (const auto &trace : traces) {
        trace->set_trace_id("trace_" + trace->instance_id() + "_" + std::to_string(trace->timestamp_ns()));
    }
}
} // namespace kv_cache_manager