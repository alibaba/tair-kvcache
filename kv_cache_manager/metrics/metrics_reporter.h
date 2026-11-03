#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

class CacheManager;
class MetricsCollector;
class MetricsRegistry;

class MetricsReporter {
public:
    virtual ~MetricsReporter() = default;
    virtual bool Init(std::shared_ptr<CacheManager> cache_manager,
                      std::shared_ptr<MetricsRegistry> metrics_registry,
                      const std::string &config,
                      const MetricsTags &common_tags = {}) = 0;
    virtual void ReportPerQuery(MetricsCollector *collector, std::optional<bool> request_failed = std::nullopt) = 0;
    virtual void ReportInterval() = 0;
    virtual std::shared_ptr<MetricsRegistry> GetMetricsRegistry() = 0;
    const MetricsTags &GetCommonTags() const { return common_tags_; }

    // Answers "does this node currently hold the leader lease". The server
    // installs it once at startup; reporters that publish the node-level
    // is_leader gauge stays silent while it is unset, which
    // keeps the optimizer and existing unit tests free of extra series.
    using LeaderStateProvider = std::function<bool()>;
    void SetLeaderStateProvider(LeaderStateProvider provider) { leader_state_provider_ = std::move(provider); }

protected:
    const LeaderStateProvider &GetLeaderStateProvider() const { return leader_state_provider_; }

    // Process-level tags resolved at startup. Each backend decides how to
    // export them; they never participate in MetricsRegistry series identity.
    MetricsTags common_tags_;
    LeaderStateProvider leader_state_provider_;
};

} // namespace kv_cache_manager
