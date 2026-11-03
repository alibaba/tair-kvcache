#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace kv_cache_manager {
class MetricsRegistry;
struct MetricsLifecycle;

// Runs on its own reporting thread, never on a business request or /metrics scrape.
class RedisHealthMetrics {
public:
    using Targets = std::map<std::string, std::string>; // instance_group -> private Redis URI
    using TargetProvider = std::function<Targets()>;
    using Probe = std::function<bool(const std::string &)>;
    static constexpr const char *kMetricName = "meta_storage.redis_healthy_status";

    RedisHealthMetrics(std::shared_ptr<MetricsRegistry> registry,
                       std::shared_ptr<MetricsLifecycle> lifecycle,
                       TargetProvider targets,
                       Probe probe = [](const std::string &uri) { return ProbeRedis(uri, 1000); });

    // An empty target set (including followers) removes previously published series.
    void Report();
    static std::optional<std::string> RedisUri(const std::string &storage_type, const std::string &storage_uri);
    // A fresh connection verifies PING, authentication and DB selection. No writes.
    // The timeout applies to connect and each socket operation; there are no retries.
    static bool ProbeRedis(const std::string &uri, int timeout_ms = 1000);

private:
    void RemoveStale(const Targets &targets);
    std::shared_ptr<MetricsRegistry> registry_;
    std::shared_ptr<MetricsLifecycle> lifecycle_;
    TargetProvider targets_;
    Probe probe_;
    Targets published_;
};
} // namespace kv_cache_manager
