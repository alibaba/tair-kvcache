#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
enum class FaultType : int { INTERNAL_ERROR = 0, };

enum class FaultTriggerStrategy : int {
    ALWAYS = 0,
    ONCE = 1,
};

struct MethodFaultConfig {
    FaultType fault_type{FaultType::INTERNAL_ERROR};
    FaultTriggerStrategy fault_trigger_strategy{FaultTriggerStrategy::ALWAYS};
    int32_t trigger_at_call{-1};
};

std::string ToString(const FaultType &type);

std::string ToString(const FaultTriggerStrategy &type);

class FaultInjector {
public:
    static FaultInjector &GetInstance();

    void InjectFault(const std::string &method, const MethodFaultConfig &config);

    std::optional<MethodFaultConfig> GetFault(const std::string &method, const std::string &trace_id) const;

    bool MatchFaultTrigger(const MethodFaultConfig &config, const std::string &trace_id) const;

    bool RemoveFault(const std::string &method);

    void ClearFaults();

private:
    FaultInjector() = default;
    ~FaultInjector() = default;

    FaultInjector(const FaultInjector &) = delete;
    FaultInjector &operator=(const FaultInjector &) = delete;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, MethodFaultConfig> configs_;
};
} // namespace kv_cache_manager