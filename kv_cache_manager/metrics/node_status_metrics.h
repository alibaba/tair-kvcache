#pragma once

#include <memory>

namespace kv_cache_manager {

class MetricsRegistry;

// Observes leader lease ownership only; does not affect request admission.
class NodeStatusMetrics {
public:
    NodeStatusMetrics() = delete;
    static constexpr const char *kIsLeaderName = "is_leader";

    static void Publish(const std::shared_ptr<MetricsRegistry> &metrics_registry, bool is_leader);
};

} // namespace kv_cache_manager
