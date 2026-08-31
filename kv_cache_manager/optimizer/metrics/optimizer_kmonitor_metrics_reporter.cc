#include "kv_cache_manager/optimizer/metrics/optimizer_kmonitor_metrics_reporter.h"

#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/MetricsReporter.h"
#include "kv_cache_manager/common/common_util.h"
#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/metrics/kmon_param.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"
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

} // namespace

#define DECLARE_METRICS(group, name) std::unique_ptr<kmonitor::MutableMetric> group##_##name##_metrics;

struct OptimizerKmonitorMetricsReporter::KmonContext {
    kmonitor::KMonitor *kmonitor = nullptr;

    DECLARE_METRICS(service, qps);
    DECLARE_METRICS(service, query_rt_us);
    DECLARE_METRICS(service, error_qps);

    DECLARE_METRICS(query, hit_rate);
    DECLARE_METRICS(query, hit_count);
    DECLARE_METRICS(query, total_blocks);
    DECLARE_METRICS(query, max_hit_count);
    DECLARE_METRICS(query, max_hit_rate);
    DECLARE_METRICS(query, capacity_efficiency);

    DECLARE_METRICS(trace, query_total);
    DECLARE_METRICS(trace, query_blocks_total);
    DECLARE_METRICS(trace, query_max_hit_rate);
    DECLARE_METRICS(trace, query_unique_keys);
    DECLARE_METRICS(trace, query_bytes_per_block);
    DECLARE_METRICS(trace, query_linear_step);
    DECLARE_METRICS(trace, query_eviction_count);
    DECLARE_METRICS(trace, query_memory_usage_bytes);
    DECLARE_METRICS(trace, query_kv_cache_usage_bytes);
    DECLARE_METRICS(trace, query_ttl_eviction_count);
    DECLARE_METRICS(trace, query_hit_rate);
    DECLARE_METRICS(trace, query_capacity_efficiency);
    DECLARE_METRICS(trace, query_hit_age_bucket_ratio);

    std::unique_ptr<kmonitor::MutableMetric> mrc_metrics;

    struct MapHashFunc {
        size_t operator()(const std::map<std::string, std::string> &m) const noexcept {
            size_t hash = 0;
            for (const auto &pair : m) {
                hash ^= (std::hash<std::string>()(pair.first) ^ (std::hash<std::string>()(pair.second) << 1));
            }
            return hash;
        }
    };

    mutable std::shared_mutex mutex_;
    std::unordered_map<MetricsTags, kmonitor::MetricsTags, MapHashFunc> tag_cache_;

    kmonitor::MetricsTags GetKmonitorTags(const MetricsTags &base_tags) {
        {
            std::shared_lock read_guard(mutex_);
            auto iter = tag_cache_.find(base_tags);
            if (iter != tag_cache_.end()) {
                return iter->second;
            }
        }
        std::unique_lock write_guard(mutex_);
        auto iter = tag_cache_.find(base_tags);
        if (iter != tag_cache_.end()) {
            return iter->second;
        }
        kmonitor::MetricsTags tags(base_tags);
        tag_cache_[base_tags] = tags;
        return tags;
    }
};

#undef DECLARE_METRICS

OptimizerKmonitorMetricsReporter::OptimizerKmonitorMetricsReporter(std::string prefix) : prefix_(std::move(prefix)) {}

OptimizerKmonitorMetricsReporter::~OptimizerKmonitorMetricsReporter() { Shutdown(); }

bool OptimizerKmonitorMetricsReporter::Init() {
    kmon_ctx_ = std::make_unique<KmonContext>();

    KmonParam param;
    param.Init();

    if (!param.kmonitor_metrics_reporter_cache_limit.empty()) {
        size_t limit = std::atoll(param.kmonitor_metrics_reporter_cache_limit.c_str());
        if (limit > 0) {
            kmonitor::MetricsReporter::setMetricsReporterCacheLimit(limit);
            KVCM_LOG_INFO("OptimizerKmonitorMetricsReporter: set metrics reporter cache limit [%lu].", limit);
        }
    }

    if (param.kmonitor_normal_sample_period > 0) {
        KVCM_LOG_INFO("OptimizerKmonitorMetricsReporter: set normal sample period [%d] seconds.",
                      param.kmonitor_normal_sample_period);
        kmonitor::MetricLevelConfig level_config;
        level_config.period[kmonitor::FATAL] = static_cast<unsigned int>(param.kmonitor_normal_sample_period);
        kmonitor::MetricLevelManager::SetGlobalLevelConfig(level_config);
    }

    kmonitor::MetricsConfig config;
    config.set_tenant_name(param.kmonitor_tenant);
    config.set_service_name(param.kmonitor_service_name);

    std::string sink_address = param.kmonitor_sink_address;
    if (!param.kmonitor_port.empty()) {
        sink_address += ":" + param.kmonitor_port;
    }
    config.set_sink_address(sink_address.c_str());
    config.set_enable_log_file_sink(param.kmonitor_enable_log_file_sink);
    config.set_manually_mode(param.kmonitor_manually_mode);
    config.set_inited(true);
    config.AddGlobalTag("hippo_slave_ip", param.hippo_slave_ip);

    for (const auto &pair : param.kmonitor_tags) {
        config.AddGlobalTag(pair.first, pair.second);
    }

    if (std::getenv("HIPPO_ROLE")) {
        auto host_ip = EnvUtil::GetEnv("HIPPO_SLAVE_IP", "");
        config.AddGlobalTag("host_ip", host_ip);
        config.AddGlobalTag("container_ip", EnvUtil::GetEnv("RequestedIP", host_ip));
        config.AddGlobalTag("hippo_role", EnvUtil::GetEnv("HIPPO_ROLE", ""));
        config.AddGlobalTag("hippo_app", EnvUtil::GetEnv("HIPPO_APP", ""));
        config.AddGlobalTag("hippo_group", EnvUtil::GetEnv("HIPPO_SERVICE_NAME", ""));
    }

    if (!kmonitor::KMonitorFactory::Init(config)) {
        KVCM_LOG_ERROR("OptimizerKmonitorMetricsReporter: KMonitorFactory::Init failed");
        kmon_ctx_.reset();
        return false;
    }

    kmonitor::KMonitorFactory::registerBuildInMetrics(nullptr, param.kmonitor_metrics_prefix);
    kmonitor::KMonitorFactory::Start();
    if (!InitMetrics()) {
        kmonitor::KMonitorFactory::Shutdown();
        kmon_ctx_.reset();
        return false;
    }
    return true;
}

#define REGISTER_QPS_METRIC(group, name)                                                                               \
    do {                                                                                                               \
        std::string metric_name = #group "." #name;                                                                    \
        kmon_ctx_->group##_##name##_metrics.reset(                                                                     \
            reporter->RegisterMetric(metric_name, kmonitor::QPS, kmonitor::FATAL));                                    \
        if (!kmon_ctx_->group##_##name##_metrics) {                                                                    \
            KVCM_LOG_ERROR("failed to register metric:[%s]", metric_name.c_str());                                     \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#define REGISTER_GAUGE_METRIC(group, name)                                                                             \
    do {                                                                                                               \
        std::string metric_name = #group "." #name;                                                                    \
        kmon_ctx_->group##_##name##_metrics.reset(                                                                     \
            reporter->RegisterMetric(metric_name, kmonitor::GAUGE, kmonitor::FATAL));                                  \
        if (!kmon_ctx_->group##_##name##_metrics) {                                                                    \
            KVCM_LOG_ERROR("failed to register metric:[%s]", metric_name.c_str());                                     \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

bool OptimizerKmonitorMetricsReporter::InitMetrics() {
    kmon_ctx_->kmonitor = kmonitor::KMonitorFactory::GetKMonitor(prefix_);
    if (!kmon_ctx_->kmonitor) {
        KVCM_LOG_ERROR("OptimizerKmonitorMetricsReporter: GetKMonitor failed for prefix[%s]", prefix_.c_str());
        return false;
    }

    auto *reporter = kmon_ctx_->kmonitor;
    REGISTER_QPS_METRIC(service, qps);
    REGISTER_GAUGE_METRIC(service, query_rt_us);
    REGISTER_QPS_METRIC(service, error_qps);

    REGISTER_GAUGE_METRIC(query, hit_rate);
    REGISTER_GAUGE_METRIC(query, hit_count);
    REGISTER_GAUGE_METRIC(query, total_blocks);
    REGISTER_GAUGE_METRIC(query, max_hit_count);
    REGISTER_GAUGE_METRIC(query, max_hit_rate);
    REGISTER_GAUGE_METRIC(query, capacity_efficiency);

    REGISTER_GAUGE_METRIC(trace, query_total);
    REGISTER_GAUGE_METRIC(trace, query_blocks_total);
    REGISTER_GAUGE_METRIC(trace, query_max_hit_rate);
    REGISTER_GAUGE_METRIC(trace, query_unique_keys);
    REGISTER_GAUGE_METRIC(trace, query_bytes_per_block);
    REGISTER_GAUGE_METRIC(trace, query_linear_step);
    REGISTER_GAUGE_METRIC(trace, query_eviction_count);
    REGISTER_GAUGE_METRIC(trace, query_memory_usage_bytes);
    REGISTER_GAUGE_METRIC(trace, query_kv_cache_usage_bytes);
    REGISTER_GAUGE_METRIC(trace, query_ttl_eviction_count);
    REGISTER_GAUGE_METRIC(trace, query_hit_rate);
    REGISTER_GAUGE_METRIC(trace, query_capacity_efficiency);
    REGISTER_GAUGE_METRIC(trace, query_hit_age_bucket_ratio);

    kmon_ctx_->mrc_metrics.reset(reporter->RegisterMetric("mrc", kmonitor::GAUGE, kmonitor::FATAL));
    if (!kmon_ctx_->mrc_metrics) {
        KVCM_LOG_ERROR("failed to register metric:[mrc]");
        return false;
    }

    KVCM_LOG_INFO("OptimizerKmonitorMetricsReporter initialized, prefix[%s]", prefix_.c_str());
    return true;
}

#undef REGISTER_QPS_METRIC
#undef REGISTER_GAUGE_METRIC

void OptimizerKmonitorMetricsReporter::ReportPerQuery(OptimizerServiceMetricsCollector *collector,
                                                      const MetricsTags &service_tags,
                                                      const MetricsTags &query_tags) {
    if (!collector || !kmon_ctx_ || !kmon_ctx_->kmonitor) {
        return;
    }

    const auto kmonitor_service_tags = kmon_ctx_->GetKmonitorTags(service_tags);
    kmon_ctx_->service_qps_metrics->Report(&kmonitor_service_tags, 1.0);

    double query_rt_us;
    GET_METRICS_(collector, service, query_rt_us, query_rt_us);
    if (!std::isnan(query_rt_us)) {
        kmon_ctx_->service_query_rt_us_metrics->Report(&kmonitor_service_tags, query_rt_us);
    }

    double error_code;
    GET_METRICS_(collector, service, error_code, error_code);
    if (!std::isnan(error_code) && !CommonUtil::IsZeroDouble(error_code)) {
        kmon_ctx_->service_error_qps_metrics->Report(&kmonitor_service_tags, 1.0);
    }

    if (collector->total_blocks() <= 0) {
        return;
    }

    for (const auto &info : collector->per_capacity_hits()) {
        const double hit_rate =
            info.hit_rate >= 0.0 ? info.hit_rate
                                 : static_cast<double>(info.hit_count) / static_cast<double>(collector->total_blocks());
        MetricsTags capacity_tags = query_tags;
        capacity_tags["capacity_gb"] = std::to_string(info.capacity_gb);
        const auto tags = kmon_ctx_->GetKmonitorTags(capacity_tags);
        kmon_ctx_->query_hit_rate_metrics->Report(&tags, hit_rate);
        kmon_ctx_->query_hit_count_metrics->Report(&tags, static_cast<double>(info.hit_count));
        if (collector->max_hit_rate() > 0) {
            kmon_ctx_->query_capacity_efficiency_metrics->Report(&tags, hit_rate / collector->max_hit_rate());
        }
    }

    const auto tags = kmon_ctx_->GetKmonitorTags(query_tags);
    kmon_ctx_->query_total_blocks_metrics->Report(&tags, static_cast<double>(collector->total_blocks()));
    if (collector->max_hit_count() >= 0) {
        kmon_ctx_->query_max_hit_count_metrics->Report(&tags, static_cast<double>(collector->max_hit_count()));
        kmon_ctx_->query_max_hit_rate_metrics->Report(&tags, collector->max_hit_rate());
    }
}

void OptimizerKmonitorMetricsReporter::ReportInterval(const std::vector<InstanceSummary> &summaries,
                                                      const std::vector<MrcMetricInfo> &mrc_metrics) {
    if (!kmon_ctx_ || !kmon_ctx_->kmonitor) {
        return;
    }

    for (const auto &summary : summaries) {
        MetricsTags base_tags = {{"instance_group", summary.instance_group}, {"instance_id", summary.instance_id}};
        const auto tags = kmon_ctx_->GetKmonitorTags(base_tags);
        kmon_ctx_->trace_query_total_metrics->Report(&tags, static_cast<double>(summary.total_queries));
        kmon_ctx_->trace_query_blocks_total_metrics->Report(&tags, static_cast<double>(summary.total_blocks_queried));
        if (summary.max_hit_rate >= 0) {
            kmon_ctx_->trace_query_max_hit_rate_metrics->Report(&tags, summary.max_hit_rate);
        }
        kmon_ctx_->trace_query_unique_keys_metrics->Report(&tags, static_cast<double>(summary.unique_keys));
        kmon_ctx_->trace_query_bytes_per_block_metrics->Report(&tags, static_cast<double>(summary.bytes_per_block));
        kmon_ctx_->trace_query_linear_step_metrics->Report(&tags, static_cast<double>(summary.linear_step));
        kmon_ctx_->trace_query_eviction_count_metrics->Report(&tags, static_cast<double>(summary.eviction_count));
        kmon_ctx_->trace_query_memory_usage_bytes_metrics->Report(&tags,
                                                                  static_cast<double>(summary.memory_usage_bytes));
        kmon_ctx_->trace_query_kv_cache_usage_bytes_metrics->Report(&tags,
                                                                    static_cast<double>(summary.kv_cache_usage_bytes));
        kmon_ctx_->trace_query_ttl_eviction_count_metrics->Report(&tags,
                                                                  static_cast<double>(summary.ttl_eviction_count));

        for (const auto &capacity : summary.per_capacity_hit_rates) {
            MetricsTags capacity_base_tags = base_tags;
            capacity_base_tags["capacity_gb"] = std::to_string(capacity.capacity_gb);
            const auto capacity_tags = kmon_ctx_->GetKmonitorTags(capacity_base_tags);
            kmon_ctx_->trace_query_hit_rate_metrics->Report(&capacity_tags, capacity.hit_rate);
            if (summary.max_hit_rate > 0) {
                kmon_ctx_->trace_query_capacity_efficiency_metrics->Report(&capacity_tags,
                                                                           capacity.hit_rate / summary.max_hit_rate);
            }
        }

        for (const auto &bucket : summary.hit_age_bucket_ratios) {
            const std::string bucket_label =
                bucket.threshold_seconds > 0 ? std::to_string(bucket.threshold_seconds) + "s" : "inf";
            MetricsTags bucket_base_tags = base_tags;
            bucket_base_tags["age_bucket"] = bucket_label;
            const auto bucket_tags = kmon_ctx_->GetKmonitorTags(bucket_base_tags);
            kmon_ctx_->trace_query_hit_age_bucket_ratio_metrics->Report(&bucket_tags, bucket.ratio);
        }
    }

    for (const auto &metric : mrc_metrics) {
        MetricsTags base_tags = {{"instance_group", metric.instance_group},
                                 {"instance_id", metric.instance_id},
                                 {"target_hit_rate_percent", FormatTargetHitRatePercent(metric.target_basis_points)}};
        const auto tags = kmon_ctx_->GetKmonitorTags(base_tags);
        kmon_ctx_->mrc_metrics->Report(&tags, static_cast<double>(metric.capacity_bytes));
    }
}

void OptimizerKmonitorMetricsReporter::Shutdown() {
    if (!kmon_ctx_) {
        return;
    }
    kmonitor::KMonitorFactory::Shutdown();
    kmon_ctx_.reset();
    KVCM_LOG_INFO("OptimizerKmonitorMetricsReporter shutdown complete");
}

} // namespace kv_cache_manager
