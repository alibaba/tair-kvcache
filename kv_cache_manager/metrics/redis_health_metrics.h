#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {
class RegistryManager;
class MetaIndexerManager;
struct MetricsLifecycle;

// Runs on its own reporting thread, never on a business request or /metrics scrape.
class RedisHealthMetrics {
public:
    using Targets = std::set<std::pair<std::string, std::string>>; // (instance_group, private Redis URI)
    // nullopt means a busy snapshot, not deletion or loss of leadership.
    using TargetProvider = std::function<std::optional<Targets>()>;
    using Probe = std::function<bool(const std::string &)>;
    static constexpr const char *kMetricName = "meta_storage.redis_healthy_status";

    RedisHealthMetrics(
        std::shared_ptr<MetricsRegistry> registry,
        std::shared_ptr<MetricsLifecycle> lifecycle,
        TargetProvider targets,
        Probe probe = [](const std::string &uri) { return ProbeRedis(uri, 1000); },
        std::function<bool()> active = []() { return true; });

    // An empty target set (including followers) removes previously published series.
    void Report();
    void RequestStop() { stopped_.store(true, std::memory_order_relaxed); }
    static std::optional<Targets> CollectTargets(const RegistryManager &registry, const MetaIndexerManager &indexers);
    static std::optional<std::string> RedisUri(const std::string &storage_type, const std::string &storage_uri);
    // A fresh connection verifies PING, authentication and DB selection. No writes.
    // The timeout applies to connect and each socket operation; there are no retries.
    static bool ProbeRedis(const std::string &uri, int timeout_ms = 1000);

private:
    using Series = std::map<MetricsTags, std::set<std::string>>;
    static Series GroupByLabels(const Targets &targets);
    void RemoveStale(const Series &targets);
    std::shared_ptr<MetricsRegistry> registry_;
    std::shared_ptr<MetricsLifecycle> lifecycle_;
    TargetProvider targets_;
    Probe probe_;
    std::function<bool()> active_;
    Series published_;
    std::atomic<bool> stopped_{false};
};
} // namespace kv_cache_manager
