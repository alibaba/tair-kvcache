#include "kv_cache_manager/metrics/node_status_metrics.h"

#include <array>
#include <utility>

#include "kv_cache_manager/config/leader_elector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

void NodeStatusMetrics::Publish(const std::shared_ptr<MetricsRegistry> &metrics_registry, bool is_leader) {
    if (metrics_registry) {
        metrics_registry->GetGauge(kIsLeaderName) = is_leader ? 1.0 : 0.0;
    }
}

void NodeStatusMetrics::PublishRoleState(const std::shared_ptr<MetricsRegistry> &metrics_registry, RoleState state) {
    if (!metrics_registry) {
        return;
    }
    static constexpr std::array<std::pair<RoleState, const char *>, 4> states{{
        {RoleState::FOLLOWER, "follower"},
        {RoleState::PROMOTING, "promoting"},
        {RoleState::LEADER, "leader"},
        {RoleState::DEMOTING, "demoting"},
    }};
    for (const auto &[role, label] : states) {
        metrics_registry->GetGauge(kRoleStateName, {{"state", label}}) = state == role ? 1.0 : 0.0;
    }
}

} // namespace kv_cache_manager
