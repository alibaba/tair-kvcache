#pragma once

#include <memory>

namespace kv_cache_manager {

class MetricsRegistry;
enum class RoleState;

// Observes lease ownership and lifecycle phase; never changes request admission.
class NodeStatusMetrics {
public:
    NodeStatusMetrics() = delete;
    static constexpr const char *kIsLeaderName = "is_leader";
    static constexpr const char *kRoleStateName = "role_state";

    static void Publish(const std::shared_ptr<MetricsRegistry> &metrics_registry, bool is_leader);
    // Refresh all four fixed labels on every sample, clearing the previous phase.
    static void PublishRoleState(const std::shared_ptr<MetricsRegistry> &metrics_registry, RoleState state);
};

} // namespace kv_cache_manager
