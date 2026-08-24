#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_reporter.h"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/common_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_kmonitor_metrics_reporter.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_collector.h"

namespace kv_cache_manager {

namespace {

std::string FormatTargetHitRatePercent(uint32_t target_basis_points) {
    const uint32_t whole = target_basis_points / 100;
    const uint32_t fraction = target_basis_points % 100;
    if (fraction == 0) {
        return std::to_string(whole);
    }
    if (fraction % 10 == 0) {
        return std::to_string(whole) + "." + std::to_string(fraction / 10);
    }
    return std::to_string(whole) + "." + (fraction < 10 ? "0" : "") + std::to_string(fraction);
}

std::string ResolveInstanceGroup(const std::shared_ptr<OnlineOptimizerManager> &manager,
                                 const std::string &instance_id) {
    if (!manager || instance_id.empty()) {
        return {};
    }
    auto registry = manager->registry_manager();
    if (!registry) {
        return {};
    }
    auto instance_info = registry->GetInstanceInfo(instance_id);
    return instance_info ? instance_info->instance_group_name() : std::string{};
}

} // namespace

OptimizerMetricsReporter::OptimizerMetricsReporter(std::shared_ptr<OnlineOptimizerManager> manager,
                                                   std::shared_ptr<MetricsRegistry> metrics_registry,
                                                   std::shared_ptr<OptimizerKmonitorMetricsReporter> kmonitor_reporter)
    : manager_(std::move(manager))
    , metrics_registry_(std::move(metrics_registry))
    , kmonitor_reporter_(std::move(kmonitor_reporter)) {}

OptimizerMetricsReporter::~OptimizerMetricsReporter() = default;

void OptimizerMetricsReporter::ReportInterval() {
    std::vector<InstanceSummary> summaries;
    manager_->ListInstances("", summaries);
    std::vector<IntervalMetricInfo> interval_metrics;
    manager_->TakeIntervalMetrics(interval_metrics);
    std::vector<MrcMetricInfo> mrc_metrics;
    manager_->TakeMrcMetrics(mrc_metrics);

    for (const auto &summary : summaries) {
        MetricsTags instance_tags = {{"instance_group", summary.instance_group}, {"instance_id", summary.instance_id}};

        metrics_registry_->GetGauge("trace_query_total", instance_tags) = static_cast<double>(summary.total_queries);
        metrics_registry_->GetGauge("trace_query_blocks_total", instance_tags) =
            static_cast<double>(summary.total_blocks_queried);
        if (summary.max_hit_rate >= 0) {
            metrics_registry_->GetGauge("trace_query_max_hit_rate", instance_tags) = summary.max_hit_rate;
        }
        metrics_registry_->GetGauge("trace_query_unique_keys", instance_tags) =
            static_cast<double>(summary.unique_keys);
        metrics_registry_->GetGauge("trace_query_bytes_per_block", instance_tags) =
            static_cast<double>(summary.bytes_per_block);
        metrics_registry_->GetGauge("trace_query_linear_step", instance_tags) =
            static_cast<double>(summary.linear_step);
        metrics_registry_->GetGauge("trace_query_eviction_count", instance_tags) =
            static_cast<double>(summary.eviction_count);
        metrics_registry_->GetGauge("trace_query_memory_usage_bytes", instance_tags) =
            static_cast<double>(summary.memory_usage_bytes);
        metrics_registry_->GetGauge("trace_query_kv_cache_usage_bytes", instance_tags) =
            static_cast<double>(summary.kv_cache_usage_bytes);
        metrics_registry_->GetGauge("trace_query_ttl_eviction_count", instance_tags) =
            static_cast<double>(summary.ttl_eviction_count);

        for (const auto &capacity : summary.per_capacity_hit_rates) {
            MetricsTags capacity_tags = instance_tags;
            capacity_tags["capacity_gb"] = std::to_string(capacity.capacity_gb);
            metrics_registry_->GetGauge("trace_query_hit_rate", capacity_tags) = capacity.hit_rate;
            metrics_registry_->GetGauge("trace_query_capacity_efficiency", capacity_tags) =
                summary.max_hit_rate > 0 ? capacity.hit_rate / summary.max_hit_rate
                                         : std::numeric_limits<double>::quiet_NaN();
        }

        for (const auto &bucket : summary.hit_age_bucket_ratios) {
            const std::string bucket_label =
                bucket.threshold_seconds > 0 ? std::to_string(bucket.threshold_seconds) + "s" : "inf";
            MetricsTags bucket_tags = instance_tags;
            bucket_tags["age_bucket"] = bucket_label;
            metrics_registry_->GetGauge("trace_query_hit_age_bucket_ratio", bucket_tags) = bucket.ratio;
        }
    }

    for (const auto &metric : interval_metrics) {
        MetricsTags interval_tags = {{"instance_group", metric.instance_group}, {"instance_id", metric.instance_id}};
        if (metric.has_theoretical_max_hit_rate) {
            metrics_registry_->GetGauge("interval.query_max_hit_rate", interval_tags) = metric.max_hit_rate;
        }
        for (const auto &capacity : metric.per_capacity_hit_rates) {
            MetricsTags capacity_tags = interval_tags;
            capacity_tags["capacity_gb"] = std::to_string(capacity.capacity_gb);
            metrics_registry_->GetGauge("interval.query_hit_rate", capacity_tags) = capacity.hit_rate;
            metrics_registry_->GetGauge("interval.query_capacity_efficiency", capacity_tags) =
                metric.has_theoretical_max_hit_rate && metric.max_hit_rate > 0
                    ? capacity.hit_rate / metric.max_hit_rate
                    : std::numeric_limits<double>::quiet_NaN();
        }
    }

    for (const auto &metric : mrc_metrics) {
        MetricsTags tags = {{"instance_group", metric.instance_group},
                            {"instance_id", metric.instance_id},
                            {"target_hit_rate_percent", FormatTargetHitRatePercent(metric.target_basis_points)}};
        metrics_registry_->GetGauge("mrc", tags) = static_cast<double>(metric.capacity_bytes);
    }

    if (kmonitor_reporter_) {
        kmonitor_reporter_->ReportInterval(summaries, mrc_metrics);
    }
}

void OptimizerMetricsReporter::ReportPerQuery(MetricsCollector *collector) {
    auto *p = dynamic_cast<OptimizerServiceMetricsCollector *>(collector);
    if (!p) {
        return;
    }

    Counter query_counter;
    COPY_METRICS_(p, service, query_counter, query_counter);
    ++query_counter;

    if (p->input_token_len() > 0) {
        Counter input_tokens_total;
        COPY_METRICS_(p, service, input_tokens_total, input_tokens_total);
        input_tokens_total += static_cast<uint64_t>(p->input_token_len());
    }

    double query_rt_us;
    GET_METRICS_(p, service, query_rt_us, query_rt_us);
    Counter query_rt_total;
    COPY_METRICS_(p, service, query_rt_us_total, query_rt_total);
    if (!std::isnan(query_rt_us)) {
        query_rt_total += static_cast<uint64_t>(query_rt_us);
    }

    double error_code;
    GET_METRICS_(p, service, error_code, error_code);
    if (!std::isnan(error_code) && !CommonUtil::IsZeroDouble(error_code)) {
        Counter error_counter;
        COPY_METRICS_(p, service, error_counter, error_counter);
        ++error_counter;
    }

    const std::string instance_group = ResolveInstanceGroup(manager_, p->instance_id());
    MetricsTags service_tags = p->GetMetricsTags();
    MetricsTags query_tags = {
        {"instance_group", instance_group}, {"instance_id", p->instance_id()}, {"client_ip", p->client_ip()}};
    if (metrics_registry_ && !p->instance_id().empty()) {
        service_tags = {{"instance_group", instance_group}, {"instance_id", p->instance_id()}};

        ++metrics_registry_->GetCounter("service.query_counter", service_tags);
        if (p->input_token_len() > 0) {
            metrics_registry_->GetCounter("service.input_tokens_total", service_tags) +=
                static_cast<uint64_t>(p->input_token_len());
        }
        metrics_registry_->GetGauge("service.query_rt_us", service_tags) = query_rt_us;
        if (!std::isnan(query_rt_us)) {
            metrics_registry_->GetCounter("service.query_rt_us_total", service_tags) +=
                static_cast<uint64_t>(query_rt_us);
        }
        metrics_registry_->GetGauge("service.error_code", service_tags) = error_code;
        if (!std::isnan(error_code) && !CommonUtil::IsZeroDouble(error_code)) {
            ++metrics_registry_->GetCounter("service.error_counter", service_tags);
        }

        if (p->total_blocks() > 0) {
            for (const auto &capacity : p->per_capacity_hits()) {
                const double hit_rate = capacity.hit_rate >= 0.0 ? capacity.hit_rate
                                                                 : static_cast<double>(capacity.hit_count) /
                                                                       static_cast<double>(p->total_blocks());
                MetricsTags capacity_tags = query_tags;
                capacity_tags["capacity_gb"] = std::to_string(capacity.capacity_gb);
                metrics_registry_->GetGauge("query_hit_rate", capacity_tags) = hit_rate;
                metrics_registry_->GetGauge("query_hit_count", capacity_tags) = static_cast<double>(capacity.hit_count);
                metrics_registry_->GetGauge("query_capacity_efficiency", capacity_tags) =
                    p->max_hit_rate() > 0 ? hit_rate / p->max_hit_rate() : std::numeric_limits<double>::quiet_NaN();
            }

            metrics_registry_->GetGauge("query_total_blocks", query_tags) = static_cast<double>(p->total_blocks());
            if (p->max_hit_count() >= 0) {
                metrics_registry_->GetGauge("query_max_hit_count", query_tags) =
                    static_cast<double>(p->max_hit_count());
                metrics_registry_->GetGauge("query_max_hit_rate", query_tags) = p->max_hit_rate();
            }
        }
    }

    if (kmonitor_reporter_) {
        kmonitor_reporter_->ReportPerQuery(p, service_tags, query_tags);
    }
}

void OptimizerMetricsReporter::RemoveInstanceMetrics(const std::string &instance_id) {
    if (!metrics_registry_) {
        return;
    }
    MetricsTags filter = {{"instance_id", instance_id}};
    const auto removed = metrics_registry_->RemoveByTagFilter(filter);
    if (removed > 0) {
        KVCM_LOG_INFO("OptimizerMetricsReporter: removed %zu metrics for instance[%s]", removed, instance_id.c_str());
    }
}

} // namespace kv_cache_manager
