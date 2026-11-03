#include "kv_cache_manager/metrics/node_status_metrics.h"

#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

void NodeStatusMetrics::Publish(const std::shared_ptr<MetricsRegistry> &metrics_registry, bool is_leader) {
    if (metrics_registry) {
        metrics_registry->GetGauge(kIsLeaderName) = is_leader ? 1.0 : 0.0;
    }
}

} // namespace kv_cache_manager
