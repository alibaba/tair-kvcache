#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/optimizer/scheduler/infer_active_window.h"
#include "kv_cache_manager/optimizer/trace_loader/optimizer_schema_trace.h"

namespace kv_cache_manager {

class InferEngineScheduler {
public:
    void SetEngineInstanceIds(std::vector<std::string> engine_instance_ids);
    void SetActiveWindows(const std::vector<InferEngineActiveWindow> &active_windows);

    void ScheduleTraces(const std::string &strategy, std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) const;

    void BuildTraceActiveWindows(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces,
                                 int64_t write_delay_ns,
                                 bool require_known_infer_id);

    [[nodiscard]] std::vector<std::string> ActiveInferIds(const std::vector<std::string> &infer_ids,
                                                          int64_t timestamp_ns) const;
    [[nodiscard]] bool IsInferActiveAt(const std::string &infer_id, int64_t timestamp_ns) const;

    [[nodiscard]] std::string ChoosePrefixHitEngineInstance(
        const std::vector<int64_t> &block_ids,
        int64_t timestamp_ns,
        size_t request_idx,
        const std::function<size_t(const std::string &, const std::vector<int64_t> &, int64_t)> &prefix_match_count)
        const;

    [[nodiscard]] const std::vector<std::string> &engine_instance_ids() const { return engine_instance_ids_; }
    [[nodiscard]] bool has_active_windows() const { return !active_windows_.empty(); }

private:
    using TraceSchedulingHandler =
        void (InferEngineScheduler::*)(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &) const;

    [[nodiscard]] static const std::unordered_map<std::string, TraceSchedulingHandler> &TraceSchedulingHandlers();

    void ScheduleRoundRobin(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) const;
    [[nodiscard]] std::vector<std::string> ActiveEngineInstanceIds(int64_t timestamp_ns) const;

    std::vector<std::string> engine_instance_ids_;
    std::unordered_set<std::string> engine_instance_id_set_;
    InferActiveWindowSet active_windows_;
};

} // namespace kv_cache_manager
