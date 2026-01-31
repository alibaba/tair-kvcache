#include "kv_cache_manager/service/util/fault_injector.h"

namespace kv_cache_manager {
std::string ToString(const FaultTriggerStrategy &type) {
    switch (type) {
    case FaultTriggerStrategy::ALWAYS:
        return "always";
    case FaultTriggerStrategy::ONCE:
        return "once";
    default:
        return "unrecognized";
    }
}

std::string ToString(const FaultType &type) {
    switch (type) {
    case FaultType::INTERNAL_ERROR:
        return "internal_error";
    default:
        return "unrecognized";
    }
}

FaultInjector &FaultInjector::GetInstance() {
    static FaultInjector instance;
    return instance;
}

void FaultInjector::InjectFault(const std::string &method, const MethodFaultConfig &config) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    configs_[method] = config;
}

std::optional<MethodFaultConfig> FaultInjector::GetFault(const std::string &method, const std::string &trace_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = configs_.find(method);
    if (it == configs_.end()) {
        return std::nullopt;
    }
    const auto &config = it->second;
    if (config.fault_trigger_strategy == FaultTriggerStrategy::ALWAYS) {
        return config;
    }
    if (config.fault_trigger_strategy == FaultTriggerStrategy::ONCE) {
        auto pos = trace_id.rfind('_');
        if (pos == std::string::npos || pos + 1 >= trace_id.size()) {
            KVCM_LOG_WARN(
                "Method %s failed to parse the call count from trace_id: %s", method.c_str(), trace_id.c_str());
            return std::nullopt;
        }
        if (config.trigger_at_call == std::stoi(trace_id.substr(pos + 1))) {
            return config;
        }
    }
    return std::nullopt;
}

bool FaultInjector::RemoveFault(const std::string &method) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = configs_.find(method);
    if (it == configs_.end()) {
        KVCM_LOG_WARN("Method %s not found", method.c_str());
        return false;
    }
    configs_.erase(it);
    return true;
}

void FaultInjector::ClearFaults() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    configs_.clear();
}

} // namespace kv_cache_manager