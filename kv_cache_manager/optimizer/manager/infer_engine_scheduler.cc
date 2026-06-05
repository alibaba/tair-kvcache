#include "kv_cache_manager/optimizer/manager/infer_engine_scheduler.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

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
    active_windows_.clear();
}

void InferEngineScheduler::SetActiveWindows(const std::vector<InferEngineActiveWindow> &active_windows) {
    active_windows_.clear();
    for (const auto &window : active_windows) {
        if (window.start_ns > window.end_ns) {
            throw std::runtime_error("infer active window start_ns must be <= end_ns: " + window.infer_id);
        }
        if (engine_instance_id_set_.find(window.infer_id) == engine_instance_id_set_.end()) {
            throw std::runtime_error("infer active window references unknown infer_id: " + window.infer_id);
        }
        active_windows_[window.infer_id].push_back(ActiveWindow{window.start_ns, window.end_ns});
    }
    for (auto &entry : active_windows_) {
        std::sort(entry.second.begin(), entry.second.end(), [](const ActiveWindow &lhs, const ActiveWindow &rhs) {
            if (lhs.first_timestamp_ns != rhs.first_timestamp_ns) {
                return lhs.first_timestamp_ns < rhs.first_timestamp_ns;
            }
            return lhs.last_timestamp_ns < rhs.last_timestamp_ns;
        });
    }
}

void InferEngineScheduler::ScheduleTraces(const std::string &strategy,
                                          std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) const {
    if (strategy == "preserve_trace" || strategy == "prefix_hit") {
        return;
    }
    if (strategy != "round_robin") {
        throw std::runtime_error("Unknown infer_scheduling_strategy: " + strategy);
    }
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
        } else if (!IsActiveAt(current_engine_instance_id, trace->timestamp_ns())) {
            throw std::runtime_error("round_robin scheduled write targets inactive engine instance: " +
                                     current_engine_instance_id);
        }
        trace->set_instance_id(current_engine_instance_id);
    }
}

void InferEngineScheduler::BuildTraceActiveWindows(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces,
                                                   int64_t write_delay_ns,
                                                   bool require_known_infer_id) {
    active_windows_.clear();
    for (const auto &trace : traces) {
        if (!trace) {
            continue;
        }
        if (require_known_infer_id &&
            engine_instance_id_set_.find(trace->instance_id()) == engine_instance_id_set_.end()) {
            throw std::runtime_error("trace active window references unknown infer_id: " + trace->instance_id());
        }
        RecordActivity(trace->instance_id(), trace->timestamp_ns());
        if (std::dynamic_pointer_cast<RequestSchemaTrace>(trace)) {
            if (trace->timestamp_ns() > std::numeric_limits<int64_t>::max() - write_delay_ns) {
                throw std::runtime_error("request write timestamp overflows int64 while building active windows: " +
                                         trace->trace_id());
            }
            RecordActivity(trace->instance_id(), trace->timestamp_ns() + write_delay_ns);
        }
    }
    if (require_known_infer_id && active_windows_.empty()) {
        throw std::runtime_error("trace active window source produced no infer activity");
    }
}

std::vector<std::string> InferEngineScheduler::ActiveInferIds(const std::vector<std::string> &infer_ids,
                                                              int64_t timestamp_ns) const {
    if (active_windows_.empty()) {
        return infer_ids;
    }

    std::vector<std::string> active_ids;
    active_ids.reserve(infer_ids.size());
    for (const auto &infer_id : infer_ids) {
        if (IsActiveAt(infer_id, timestamp_ns)) {
            active_ids.push_back(infer_id);
        }
    }
    return active_ids;
}

bool InferEngineScheduler::IsInferActiveAt(const std::string &infer_id, int64_t timestamp_ns) const {
    return IsActiveAt(infer_id, timestamp_ns);
}

std::vector<std::string> InferEngineScheduler::ActiveEngineInstanceIds(int64_t timestamp_ns) const {
    if (active_windows_.empty()) {
        return engine_instance_ids_;
    }

    std::vector<std::string> active_ids;
    active_ids.reserve(engine_instance_ids_.size());
    for (const auto &engine_instance_id : engine_instance_ids_) {
        if (IsActiveAt(engine_instance_id, timestamp_ns)) {
            active_ids.push_back(engine_instance_id);
        }
    }
    return active_ids;
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

void InferEngineScheduler::RecordActivity(const std::string &engine_instance_id, int64_t timestamp_ns) {
    if (engine_instance_id_set_.find(engine_instance_id) == engine_instance_id_set_.end()) {
        return;
    }
    auto it = active_windows_.find(engine_instance_id);
    if (it == active_windows_.end()) {
        active_windows_.emplace(engine_instance_id, std::vector<ActiveWindow>{{timestamp_ns, timestamp_ns}});
        return;
    }
    auto &window = it->second.front();
    window.first_timestamp_ns = std::min(window.first_timestamp_ns, timestamp_ns);
    window.last_timestamp_ns = std::max(window.last_timestamp_ns, timestamp_ns);
}

bool InferEngineScheduler::IsActiveAt(const std::string &engine_instance_id, int64_t timestamp_ns) const {
    if (active_windows_.empty()) {
        return true;
    }
    auto it = active_windows_.find(engine_instance_id);
    if (it == active_windows_.end()) {
        return false;
    }
    return std::any_of(it->second.begin(), it->second.end(), [timestamp_ns](const ActiveWindow &window) {
        return timestamp_ns >= window.first_timestamp_ns && timestamp_ns <= window.last_timestamp_ns;
    });
}

} // namespace kv_cache_manager
