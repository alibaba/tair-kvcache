#include "kv_cache_manager/optimizer/scheduler/infer_active_window.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace kv_cache_manager {

void InferActiveWindowSet::Clear() { active_windows_.clear(); }

void InferActiveWindowSet::SetConfiguredWindows(const std::vector<InferEngineActiveWindow> &active_windows,
                                                const std::unordered_set<std::string> &known_infer_ids) {
    active_windows_.clear();
    for (const auto &window : active_windows) {
        if (window.start_ns > window.end_ns) {
            throw std::runtime_error("infer active window start_ns must be <= end_ns: " + window.infer_id);
        }
        ValidateKnownInferId(
            window.infer_id, true, known_infer_ids, "infer active window references unknown infer_id: ");
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

void InferActiveWindowSet::BuildFromTrace(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces,
                                          int64_t write_delay_ns,
                                          bool require_known_infer_id,
                                          const std::unordered_set<std::string> &known_infer_ids) {
    active_windows_.clear();
    for (const auto &trace : traces) {
        if (!trace) {
            continue;
        }
        ValidateKnownInferId(trace->instance_id(),
                             require_known_infer_id,
                             known_infer_ids,
                             "trace active window references unknown infer_id: ");
        RecordActivity(trace->instance_id(), trace->timestamp_ns(), known_infer_ids);
        if (std::dynamic_pointer_cast<RequestSchemaTrace>(trace)) {
            if (trace->timestamp_ns() > std::numeric_limits<int64_t>::max() - write_delay_ns) {
                throw std::runtime_error("request write timestamp overflows int64 while building active windows: " +
                                         trace->trace_id());
            }
            RecordActivity(trace->instance_id(), trace->timestamp_ns() + write_delay_ns, known_infer_ids);
        }
    }
    if (require_known_infer_id && active_windows_.empty()) {
        throw std::runtime_error("trace active window source produced no infer activity");
    }
}

std::vector<std::string> InferActiveWindowSet::FilterActive(const std::vector<std::string> &infer_ids,
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

bool InferActiveWindowSet::IsActiveAt(const std::string &infer_id, int64_t timestamp_ns) const {
    if (active_windows_.empty()) {
        return true;
    }
    auto it = active_windows_.find(infer_id);
    if (it == active_windows_.end()) {
        return false;
    }
    return std::any_of(it->second.begin(), it->second.end(), [timestamp_ns](const ActiveWindow &window) {
        return timestamp_ns >= window.first_timestamp_ns && timestamp_ns <= window.last_timestamp_ns;
    });
}

void InferActiveWindowSet::ValidateKnownInferId(const std::string &infer_id,
                                                bool require_known_infer_id,
                                                const std::unordered_set<std::string> &known_infer_ids,
                                                const std::string &error_prefix) {
    if (require_known_infer_id && known_infer_ids.find(infer_id) == known_infer_ids.end()) {
        throw std::runtime_error(error_prefix + infer_id);
    }
}

void InferActiveWindowSet::RecordActivity(const std::string &infer_id,
                                          int64_t timestamp_ns,
                                          const std::unordered_set<std::string> &known_infer_ids) {
    if (known_infer_ids.find(infer_id) == known_infer_ids.end()) {
        return;
    }
    auto it = active_windows_.find(infer_id);
    if (it == active_windows_.end()) {
        active_windows_.emplace(infer_id, std::vector<ActiveWindow>{{timestamp_ns, timestamp_ns}});
        return;
    }
    auto &window = it->second.front();
    window.first_timestamp_ns = std::min(window.first_timestamp_ns, timestamp_ns);
    window.last_timestamp_ns = std::max(window.last_timestamp_ns, timestamp_ns);
}

} // namespace kv_cache_manager
