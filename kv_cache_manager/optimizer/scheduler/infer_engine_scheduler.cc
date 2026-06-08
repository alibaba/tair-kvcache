#include "kv_cache_manager/optimizer/scheduler/infer_engine_scheduler.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "kv_cache_manager/optimizer/scheduler/infer_scheduling_strategy.h"

namespace kv_cache_manager {

void InferEngineScheduler::SetEngineInstanceIds(std::vector<std::string> engine_instance_ids) {
    std::sort(engine_instance_ids.begin(), engine_instance_ids.end());
    engine_instance_ids.erase(std::unique(engine_instance_ids.begin(), engine_instance_ids.end()),
                              engine_instance_ids.end());

    engine_instance_id_set_.clear();
    engine_instance_id_set_.reserve(engine_instance_ids.size());
    for (const auto &engine_instance_id : engine_instance_ids) {
        engine_instance_id_set_.insert(engine_instance_id);
    }
    engine_instance_ids_ = std::move(engine_instance_ids);
    active_windows_.Clear();
}

void InferEngineScheduler::SetActiveWindows(const std::vector<InferEngineActiveWindow> &active_windows) {
    active_windows_.SetConfiguredWindows(active_windows, engine_instance_id_set_);
}

void InferEngineScheduler::ScheduleTraces(const std::string &strategy,
                                          std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) const {
    if (UsesTraceInferAssignment(strategy)) {
        return;
    }
    const auto &handlers = TraceSchedulingHandlers();
    auto it = handlers.find(strategy);
    if (it != handlers.end()) {
        (this->*(it->second))(traces);
        return;
    }
    throw std::runtime_error("Unknown infer_scheduling_strategy: " + strategy);
}

const std::unordered_map<std::string, InferEngineScheduler::TraceSchedulingHandler> &
InferEngineScheduler::TraceSchedulingHandlers() {
    static const std::unordered_map<std::string, TraceSchedulingHandler> handlers = {
        {"round_robin", &InferEngineScheduler::ScheduleRoundRobin},
    };
    return handlers;
}

void InferEngineScheduler::ScheduleRoundRobin(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) const {
    if (engine_instance_ids_.empty()) {
        throw std::runtime_error("round_robin scheduling requires at least one engine instance");
    }
    size_t request_idx = 0;
    std::string current_engine_instance_id = engine_instance_ids_.front();
    for (auto &trace : traces) {
        if (!trace) {
            continue;
        }
        if (std::dynamic_pointer_cast<GetLocationSchemaTrace>(trace)) {
            const auto active_engine_ids = ActiveEngineInstanceIds(trace->timestamp_ns());
            if (active_engine_ids.empty()) {
                throw std::runtime_error("round_robin scheduling has no active engine instance at timestamp " +
                                         std::to_string(trace->timestamp_ns()));
            }
            current_engine_instance_id = active_engine_ids[request_idx % active_engine_ids.size()];
            request_idx++;
        } else if (request_idx == 0) {
            const auto active_engine_ids = ActiveEngineInstanceIds(trace->timestamp_ns());
            if (active_engine_ids.empty()) {
                throw std::runtime_error("round_robin scheduling has no active engine instance at timestamp " +
                                         std::to_string(trace->timestamp_ns()));
            }
            current_engine_instance_id = active_engine_ids.front();
        } else if (!IsInferActiveAt(current_engine_instance_id, trace->timestamp_ns())) {
            throw std::runtime_error("round_robin scheduled write targets inactive engine instance: " +
                                     current_engine_instance_id);
        }
        trace->set_instance_id(current_engine_instance_id);
    }
}

void InferEngineScheduler::BuildTraceActiveWindows(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces,
                                                   int64_t write_delay_ns,
                                                   bool require_known_infer_id) {
    active_windows_.BuildFromTrace(traces, write_delay_ns, require_known_infer_id, engine_instance_id_set_);
}

std::vector<std::string> InferEngineScheduler::ActiveInferIds(const std::vector<std::string> &infer_ids,
                                                              int64_t timestamp_ns) const {
    return active_windows_.FilterActive(infer_ids, timestamp_ns);
}

bool InferEngineScheduler::IsInferActiveAt(const std::string &infer_id, int64_t timestamp_ns) const {
    return active_windows_.IsActiveAt(infer_id, timestamp_ns);
}

std::vector<std::string> InferEngineScheduler::ActiveEngineInstanceIds(int64_t timestamp_ns) const {
    return active_windows_.FilterActive(engine_instance_ids_, timestamp_ns);
}

std::string InferEngineScheduler::ChoosePrefixHitEngineInstance(
    const std::vector<int64_t> &block_ids,
    int64_t timestamp_ns,
    size_t request_idx,
    const std::function<size_t(const std::string &, const std::vector<int64_t> &, int64_t)> &prefix_match_count) const {
    size_t best_match = 0;
    std::vector<std::string> candidates;
    for (const auto &engine_instance_id : ActiveEngineInstanceIds(timestamp_ns)) {
        const size_t match = prefix_match_count(engine_instance_id, block_ids, timestamp_ns);
        if (match > best_match) {
            best_match = match;
            candidates.clear();
            candidates.push_back(engine_instance_id);
        } else if (match == best_match) {
            candidates.push_back(engine_instance_id);
        }
    }
    if (candidates.empty()) {
        throw std::runtime_error("prefix_hit scheduling has no engine candidates");
    }
    if (candidates.size() == 1) {
        return candidates.front();
    }
    return candidates[request_idx % candidates.size()];
}

} // namespace kv_cache_manager
