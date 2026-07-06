#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/metrics/metrics_collector.h"

namespace kv_cache_manager {

struct PerCapacityHitInfo {
    double capacity_gb;
    int64_t hit_count;
};

// Re-define macros that were #undef'd at the bottom of metrics_collector.h.
// The underlying helper macros (DECLARE_METRICS_NAME_, DEFINE_SET_METRICS_GAUGE_, etc.)
// are still in scope from metrics_registry.h.
#define KVCM_GAUGE_METRICS(group, name)                                                                                \
public:                                                                                                                \
    DECLARE_METRICS_NAME_(group, name);                                                                                \
    DEFINE_COPY_METRICS_GAUGE_(group, name)                                                                            \
    DEFINE_SET_METRICS_GAUGE_(group, name)                                                                             \
    DEFINE_GET_METRICS_GAUGE_(group, name)                                                                             \
    DEFINE_STEAL_METRICS_GAUGE_(group, name)                                                                           \
                                                                                                                       \
private:                                                                                                               \
    DECLARE_METRICS_GAUGE_(group, name);

#define KVCM_CHRONO_METRICS(group, name, method)                                                                       \
    KVCM_GAUGE_METRICS(group, name)                                                                                    \
public:                                                                                                                \
    ChronoScopeGuard Make##method##Scope(bool auto_finish = true) {                                                    \
        return ChronoScopeGuard(&METRICS_(group, name), auto_finish);                                                  \
    }

#ifndef KVCM_OPTIMIZER_METRICS_COLLECTOR_PTR
#define KVCM_OPTIMIZER_METRICS_COLLECTOR_PTR(name)                                                                     \
    std::dynamic_pointer_cast<OptimizerServiceMetricsCollector>(KVCM_METRICS_COLLECTOR_(name))
#endif

class OptimizerServiceMetricsCollector final : public MetricsCollector {
    KVCM_COUNTER_METRICS(service, query_counter)
    KVCM_CHRONO_METRICS(service, query_rt_us, ServiceQuery)
    KVCM_GAUGE_METRICS(service, error_code)
    KVCM_COUNTER_METRICS(service, error_counter)

public:
    OptimizerServiceMetricsCollector() = delete;
    explicit OptimizerServiceMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) noexcept;
    OptimizerServiceMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry,
                                     MetricsTags metrics_tags) noexcept;
    ~OptimizerServiceMetricsCollector() override = default;

    bool Init() override;

    void set_instance_id(const std::string &id) { instance_id_ = id; }
    const std::string &instance_id() const { return instance_id_; }

    void set_total_blocks(int64_t v) { total_blocks_ = v; }
    int64_t total_blocks() const { return total_blocks_; }

    void set_cache_hit_count(int64_t v) { cache_hit_count_ = v; }
    int64_t cache_hit_count() const { return cache_hit_count_; }

    void set_per_capacity_hits(std::vector<PerCapacityHitInfo> v) { per_capacity_hits_ = std::move(v); }
    const std::vector<PerCapacityHitInfo> &per_capacity_hits() const { return per_capacity_hits_; }

    void set_max_hit_count(int64_t v) { max_hit_count_ = v; }
    int64_t max_hit_count() const { return max_hit_count_; }

    void set_max_hit_rate(double v) { max_hit_rate_ = v; }
    double max_hit_rate() const { return max_hit_rate_; }

    void set_client_ip(const std::string &v) { client_ip_ = v; }
    const std::string &client_ip() const { return client_ip_; }

private:
    std::string instance_id_;
    std::string client_ip_;
    int64_t total_blocks_ = 0;
    int64_t cache_hit_count_ = 0;
    std::vector<PerCapacityHitInfo> per_capacity_hits_;
    int64_t max_hit_count_ = -1;
    double max_hit_rate_ = 0.0;
};

#undef KVCM_CHRONO_METRICS
#undef KVCM_GAUGE_METRICS

} // namespace kv_cache_manager
