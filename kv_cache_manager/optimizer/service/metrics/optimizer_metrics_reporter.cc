#include "kv_cache_manager/optimizer/service/metrics/optimizer_metrics_reporter.h"

#include <cmath>
#include <shared_mutex>
#include <unordered_map>

#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/MetricsReporter.h"
#include "kv_cache_manager/common/common_util.h"
#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/metrics/kmon_param.h"
#include "kv_cache_manager/optimizer/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/service/metrics/optimizer_metrics_collector.h"

namespace kv_cache_manager {

#define DECLARE_METRICS(group, name) std::unique_ptr<kmonitor::MutableMetric> group##_##name##_metrics;

struct OptimizerMetricsReporter::KmonContext {
    kmonitor::KMonitor *kmonitor = nullptr;

    // per-query service metrics
    DECLARE_METRICS(service, qps);
    DECLARE_METRICS(service, query_rt_us);
    DECLARE_METRICS(service, error_qps);

    // per-query optimizer business metrics
    DECLARE_METRICS(query, hit_rate);
    DECLARE_METRICS(query, hit_count);
    DECLARE_METRICS(query, total_blocks);

    // intervallic metrics
    DECLARE_METRICS(trace, query_total);
    DECLARE_METRICS(trace, query_blocks_total);
    DECLARE_METRICS(trace, query_max_hit_rate);
    DECLARE_METRICS(trace, query_unique_keys);
    DECLARE_METRICS(trace, query_avg_bytes_per_block);
    DECLARE_METRICS(trace, query_linear_step);
    DECLARE_METRICS(trace, query_eviction_count);
    DECLARE_METRICS(trace, query_memory_usage_bytes);
    DECLARE_METRICS(trace, query_kv_cache_usage_bytes);
    DECLARE_METRICS(trace, query_ttl_eviction_count);
    DECLARE_METRICS(trace, query_hit_rate);

    DECLARE_METRICS(query, max_hit_count);
    DECLARE_METRICS(query, max_hit_rate);
    DECLARE_METRICS(query, capacity_efficiency);

    DECLARE_METRICS(trace, query_capacity_efficiency);
    DECLARE_METRICS(trace, query_hit_age_bucket_ratio);

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
        {
            std::unique_lock write_guard(mutex_);
            auto iter = tag_cache_.find(base_tags);
            if (iter != tag_cache_.end()) {
                return iter->second;
            }
            kmonitor::MetricsTags tags(base_tags);
            tag_cache_[base_tags] = tags;
            return tags;
        }
    }
};

#undef DECLARE_METRICS

OptimizerMetricsReporter::OptimizerMetricsReporter(std::shared_ptr<OnlineOptimizerManager> manager,
                                                   std::shared_ptr<MetricsRegistry> metrics_registry,
                                                   const std::string &prefix)
    : manager_(std::move(manager)), metrics_registry_(std::move(metrics_registry)), prefix_(prefix) {}

OptimizerMetricsReporter::~OptimizerMetricsReporter() = default;

bool OptimizerMetricsReporter::InitKmonitor() {
    kmon_ctx_ = std::make_unique<KmonContext>();

    KmonParam param;
    param.Init();

    if (!param.kmonitor_metrics_reporter_cache_limit.empty()) {
        size_t limit = std::atoll(param.kmonitor_metrics_reporter_cache_limit.c_str());
        if (limit > 0) {
            kmonitor::MetricsReporter::setMetricsReporterCacheLimit(limit);
            KVCM_LOG_INFO("OptimizerMetricsReporter: set metrics reporter cache limit [%lu].", limit);
        }
    }

    if (param.kmonitor_normal_sample_period > 0) {
        KVCM_LOG_INFO("OptimizerMetricsReporter: set kmonitor normal sample period [%d] seconds.",
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
        KVCM_LOG_ERROR("OptimizerMetricsReporter: KMonitorFactory::Init failed");
        kmon_ctx_.reset();
        return false;
    }

    kmonitor::KMonitorFactory::registerBuildInMetrics(nullptr, param.kmonitor_metrics_prefix);
    KVCM_LOG_INFO("OptimizerMetricsReporter: registerBuildInMetrics finished");

    kmonitor::KMonitorFactory::Start();

    return InitMetrics();
}

#define REGISTER_QPS_METRIC(group, name)                                                                               \
    do {                                                                                                               \
        std::string metric_name = #group "." #name;                                                                    \
        kmon_ctx_->group##_##name##_metrics.reset(                                                                     \
            reporter->RegisterMetric(metric_name, kmonitor::QPS, kmonitor::FATAL));                                    \
        if (nullptr == kmon_ctx_->group##_##name##_metrics) {                                                          \
            KVCM_LOG_ERROR("failed to register metric:[%s]", metric_name.c_str());                                     \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#define REGISTER_GAUGE_METRIC(group, name)                                                                             \
    do {                                                                                                               \
        std::string metric_name = #group "." #name;                                                                    \
        kmon_ctx_->group##_##name##_metrics.reset(                                                                     \
            reporter->RegisterMetric(metric_name, kmonitor::GAUGE, kmonitor::FATAL));                                  \
        if (nullptr == kmon_ctx_->group##_##name##_metrics) {                                                          \
            KVCM_LOG_ERROR("failed to register metric:[%s]", metric_name.c_str());                                     \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

bool OptimizerMetricsReporter::InitMetrics() {
    kmon_ctx_->kmonitor = kmonitor::KMonitorFactory::GetKMonitor(prefix_);
    if (!kmon_ctx_->kmonitor) {
        KVCM_LOG_ERROR("OptimizerMetricsReporter: GetKMonitor failed for prefix[%s]", prefix_.c_str());
        kmon_ctx_.reset();
        return false;
    }

    auto *reporter = kmon_ctx_->kmonitor;

    // service metrics
    REGISTER_QPS_METRIC(service, qps);
    REGISTER_GAUGE_METRIC(service, query_rt_us);
    REGISTER_QPS_METRIC(service, error_qps);

    // optimizer business metrics
    REGISTER_GAUGE_METRIC(query, hit_rate);
    REGISTER_GAUGE_METRIC(query, hit_count);
    REGISTER_GAUGE_METRIC(query, total_blocks);

    // intervallic metrics
    REGISTER_GAUGE_METRIC(trace, query_total);
    REGISTER_GAUGE_METRIC(trace, query_blocks_total);
    REGISTER_GAUGE_METRIC(trace, query_max_hit_rate);
    REGISTER_GAUGE_METRIC(trace, query_unique_keys);
    REGISTER_GAUGE_METRIC(trace, query_avg_bytes_per_block);
    REGISTER_GAUGE_METRIC(trace, query_linear_step);
    REGISTER_GAUGE_METRIC(trace, query_eviction_count);
    REGISTER_GAUGE_METRIC(trace, query_memory_usage_bytes);
    REGISTER_GAUGE_METRIC(trace, query_kv_cache_usage_bytes);
    REGISTER_GAUGE_METRIC(trace, query_ttl_eviction_count);
    REGISTER_GAUGE_METRIC(trace, query_hit_rate);
    REGISTER_GAUGE_METRIC(trace, query_capacity_efficiency);

    REGISTER_GAUGE_METRIC(query, max_hit_count);
    REGISTER_GAUGE_METRIC(query, max_hit_rate);
    REGISTER_GAUGE_METRIC(query, capacity_efficiency);

    REGISTER_GAUGE_METRIC(trace, query_hit_age_bucket_ratio);

    KVCM_LOG_INFO("OptimizerMetricsReporter: kmonitor initialized, prefix[%s]", prefix_.c_str());
    return true;
}

#undef REGISTER_QPS_METRIC
#undef REGISTER_GAUGE_METRIC

#define REPORT_METRICS(group, name, value)                                                                             \
    do {                                                                                                               \
        kmon_ctx_->group##_##name##_metrics->Report(&tags, value);                                                     \
    } while (0)

#define REPORT_METRICS_WHEN(group, name, value, pred)                                                                  \
    do {                                                                                                               \
        if (pred) {                                                                                                    \
            REPORT_METRICS(group, name, value);                                                                        \
        }                                                                                                              \
    } while (0)

#define REPORT_COLLECTED_METRICS(group, name)                                                                          \
    do {                                                                                                               \
        double v;                                                                                                      \
        GET_METRICS_(p, group, name, v);                                                                               \
        REPORT_METRICS(group, name, v);                                                                                \
    } while (0)

#define REPORT_STEAL_METRICS(group, name)                                                                              \
    do {                                                                                                               \
        Gauge gauge;                                                                                                   \
        COPY_METRICS_(p, group, name, gauge);                                                                          \
        const auto raw_metrics_value = gauge.GetRaw();                                                                 \
        if (raw_metrics_value == nullptr || !raw_metrics_value->touched.load(std::memory_order_relaxed)) {             \
            break;                                                                                                     \
        }                                                                                                              \
        double v;                                                                                                      \
        STEAL_METRICS_(p, group, name, v);                                                                             \
        if (!(std::isnan(v))) {                                                                                        \
            REPORT_METRICS(group, name, v);                                                                            \
        }                                                                                                              \
    } while (0)

void OptimizerMetricsReporter::ReportInterval() {
    std::vector<InstanceSummary> summaries;
    manager_->ListInstances("", summaries);

    for (const auto &s : summaries) {
        MetricsTags prom_tags = {{"instance_id", s.instance_id}};

        Gauge query_total = metrics_registry_->GetGauge("trace_query_total", prom_tags);
        query_total = static_cast<double>(s.total_queries);

        Gauge blocks_total = metrics_registry_->GetGauge("trace_query_blocks_total", prom_tags);
        blocks_total = static_cast<double>(s.total_blocks_queried);

        Gauge max_hit_rate = metrics_registry_->GetGauge("trace_query_max_hit_rate", prom_tags);
        max_hit_rate = s.max_hit_rate;

        Gauge unique_keys = metrics_registry_->GetGauge("trace_query_unique_keys", prom_tags);
        unique_keys = static_cast<double>(s.unique_keys);

        Gauge avg_bytes = metrics_registry_->GetGauge("trace_query_avg_bytes_per_block", prom_tags);
        avg_bytes = static_cast<double>(s.avg_bytes_per_block);

        Gauge linear_step = metrics_registry_->GetGauge("trace_query_linear_step", prom_tags);
        linear_step = static_cast<double>(s.linear_step);

        Gauge eviction_count = metrics_registry_->GetGauge("trace_query_eviction_count", prom_tags);
        eviction_count = static_cast<double>(s.eviction_count);

        Gauge memory_usage = metrics_registry_->GetGauge("trace_query_memory_usage_bytes", prom_tags);
        memory_usage = static_cast<double>(s.memory_usage_bytes);

        Gauge kv_cache_usage = metrics_registry_->GetGauge("trace_query_kv_cache_usage_bytes", prom_tags);
        kv_cache_usage = static_cast<double>(s.kv_cache_usage_bytes);

        Gauge ttl_eviction = metrics_registry_->GetGauge("trace_query_ttl_eviction_count", prom_tags);
        ttl_eviction = static_cast<double>(s.ttl_eviction_count);

        for (const auto &cap_info : s.per_capacity_hit_rates) {
            std::string cap_str = std::to_string(cap_info.capacity_gb);
            MetricsTags cap_tags = {{"instance_id", s.instance_id}, {"capacity_gb", cap_str}};
            Gauge rate = metrics_registry_->GetGauge("trace_query_hit_rate", cap_tags);
            rate = cap_info.hit_rate;

            if (s.max_hit_rate > 0) {
                Gauge achievement = metrics_registry_->GetGauge("trace_query_capacity_efficiency", cap_tags);
                achievement = cap_info.hit_rate / s.max_hit_rate;
            }
        }

        for (const auto &bucket : s.hit_age_bucket_ratios) {
            std::string bucket_label =
                bucket.threshold_seconds > 0 ? std::to_string(bucket.threshold_seconds) + "s" : "inf";
            MetricsTags bucket_tags = {{"instance_id", s.instance_id}, {"age_bucket", bucket_label}};
            Gauge bucket_ratio = metrics_registry_->GetGauge("trace_query_hit_age_bucket_ratio", bucket_tags);
            bucket_ratio = bucket.ratio;
        }
    }

    // --- Kmonitor ---
    if (!kmon_ctx_ || !kmon_ctx_->kmonitor) {
        return;
    }

    for (const auto &s : summaries) {
        MetricsTags base_tags = {{"instance_id", s.instance_id}};
        kmonitor::MetricsTags tags = kmon_ctx_->GetKmonitorTags(base_tags);

        REPORT_METRICS(trace, query_total, static_cast<double>(s.total_queries));
        REPORT_METRICS(trace, query_blocks_total, static_cast<double>(s.total_blocks_queried));
        REPORT_METRICS(trace, query_max_hit_rate, s.max_hit_rate);
        REPORT_METRICS(trace, query_unique_keys, static_cast<double>(s.unique_keys));
        REPORT_METRICS(trace, query_avg_bytes_per_block, static_cast<double>(s.avg_bytes_per_block));
        REPORT_METRICS(trace, query_linear_step, static_cast<double>(s.linear_step));
        REPORT_METRICS(trace, query_eviction_count, static_cast<double>(s.eviction_count));
        REPORT_METRICS(trace, query_memory_usage_bytes, static_cast<double>(s.memory_usage_bytes));
        REPORT_METRICS(trace, query_kv_cache_usage_bytes, static_cast<double>(s.kv_cache_usage_bytes));
        REPORT_METRICS(trace, query_ttl_eviction_count, static_cast<double>(s.ttl_eviction_count));

        for (const auto &cap_info : s.per_capacity_hit_rates) {
            std::string cap_str = std::to_string(cap_info.capacity_gb);
            MetricsTags cap_tags = {{"instance_id", s.instance_id}, {"capacity_gb", cap_str}};
            kmonitor::MetricsTags ktags = kmon_ctx_->GetKmonitorTags(cap_tags);
            kmon_ctx_->trace_query_hit_rate_metrics->Report(&ktags, cap_info.hit_rate);

            if (s.max_hit_rate > 0) {
                kmon_ctx_->trace_query_capacity_efficiency_metrics->Report(&ktags, cap_info.hit_rate / s.max_hit_rate);
            }
        }

        for (const auto &bucket : s.hit_age_bucket_ratios) {
            std::string bucket_label =
                bucket.threshold_seconds > 0 ? std::to_string(bucket.threshold_seconds) + "s" : "inf";
            MetricsTags bucket_tags = {{"instance_id", s.instance_id}, {"age_bucket", bucket_label}};
            kmonitor::MetricsTags ktags = kmon_ctx_->GetKmonitorTags(bucket_tags);
            kmon_ctx_->trace_query_hit_age_bucket_ratio_metrics->Report(&ktags, bucket.ratio);
        }
    }
}

void OptimizerMetricsReporter::ReportPerQuery(MetricsCollector *collector) {
    if (!collector) {
        return;
    }

    auto *p = dynamic_cast<OptimizerServiceMetricsCollector *>(collector);
    if (!p) {
        return;
    }

    const auto &client_ip = p->client_ip();

    // --- Prometheus: increment service counters ---
    {
        Counter query_counter;
        COPY_METRICS_(p, service, query_counter, query_counter);
        ++query_counter;

        double error_code_v;
        GET_METRICS_(p, service, error_code, error_code_v);
        if (!std::isnan(error_code_v) && !CommonUtil::IsZeroDouble(error_code_v)) {
            Counter error_counter;
            COPY_METRICS_(p, service, error_counter, error_counter);
            ++error_counter;
        }
    }

    // --- Prometheus: per-capacity hit metrics ---
    if (metrics_registry_ && p->total_blocks() > 0) {
        const auto &per_cap = p->per_capacity_hits();
        const auto &instance_id = p->instance_id();

        for (const auto &info : per_cap) {
            double hit_rate = static_cast<double>(info.hit_count) / static_cast<double>(p->total_blocks());
            std::string cap_str = std::to_string(info.capacity_gb);

            MetricsTags prom_tags = {{"instance_id", instance_id}, {"client_ip", client_ip}, {"capacity_gb", cap_str}};

            Gauge rate_gauge = metrics_registry_->GetGauge("query_hit_rate", prom_tags);
            rate_gauge = hit_rate;

            Gauge hit_gauge = metrics_registry_->GetGauge("query_hit_count", prom_tags);
            hit_gauge = static_cast<double>(info.hit_count);

            if (p->max_hit_rate() > 0) {
                Gauge eff_gauge = metrics_registry_->GetGauge("query_capacity_efficiency", prom_tags);
                eff_gauge = hit_rate / p->max_hit_rate();
            }
        }

        MetricsTags base_tags = {{"instance_id", instance_id}, {"client_ip", client_ip}};
        Gauge blocks_gauge = metrics_registry_->GetGauge("query_total_blocks", base_tags);
        blocks_gauge = static_cast<double>(p->total_blocks());

        if (p->max_hit_count() >= 0) {
            Gauge max_hit_gauge = metrics_registry_->GetGauge("query_max_hit_count", base_tags);
            max_hit_gauge = static_cast<double>(p->max_hit_count());
            Gauge max_rate_gauge = metrics_registry_->GetGauge("query_max_hit_rate", base_tags);
            max_rate_gauge = p->max_hit_rate();
        }
    }

    // --- Kmonitor ---
    if (!kmon_ctx_ || !kmon_ctx_->kmonitor) {
        return;
    }

    const kmonitor::MetricsTags tags = kmon_ctx_->GetKmonitorTags(p->GetMetricsTags());

    // service metrics
    REPORT_METRICS(service, qps, 1.0);
    REPORT_STEAL_METRICS(service, query_rt_us);
    double service_error_code_v;
    STEAL_METRICS_(p, service, error_code, service_error_code_v);
    REPORT_METRICS_WHEN(
        service, error_qps, 1.0, !std::isnan(service_error_code_v) && !CommonUtil::IsZeroDouble(service_error_code_v));

    // optimizer business metrics
    if (p->total_blocks() > 0) {
        const auto &per_cap = p->per_capacity_hits();
        const auto &instance_id = p->instance_id();

        for (const auto &info : per_cap) {
            double hit_rate = static_cast<double>(info.hit_count) / static_cast<double>(p->total_blocks());
            std::string cap_str = std::to_string(info.capacity_gb);

            MetricsTags base_tags = {{"instance_id", instance_id}, {"client_ip", client_ip}, {"capacity_gb", cap_str}};
            kmonitor::MetricsTags tags = kmon_ctx_->GetKmonitorTags(base_tags);

            REPORT_METRICS(query, hit_rate, hit_rate);
            REPORT_METRICS(query, hit_count, static_cast<double>(info.hit_count));
            REPORT_METRICS_WHEN(query, capacity_efficiency, hit_rate / p->max_hit_rate(), p->max_hit_rate() > 0);
        }

        MetricsTags base_tags = {{"instance_id", instance_id}, {"client_ip", client_ip}};
        kmonitor::MetricsTags tags = kmon_ctx_->GetKmonitorTags(base_tags);
        REPORT_METRICS(query, total_blocks, static_cast<double>(p->total_blocks()));

        if (p->max_hit_count() >= 0) {
            REPORT_METRICS(query, max_hit_count, static_cast<double>(p->max_hit_count()));
            REPORT_METRICS(query, max_hit_rate, p->max_hit_rate());
        }
    }
}

#undef REPORT_METRICS
#undef REPORT_METRICS_WHEN
#undef REPORT_COLLECTED_METRICS
#undef REPORT_STEAL_METRICS

void OptimizerMetricsReporter::ShutdownKmonitor() {
    if (kmon_ctx_) {
        kmonitor::KMonitorFactory::Shutdown();
        kmon_ctx_.reset();
        KVCM_LOG_INFO("OptimizerMetricsReporter: KMonitor shutdown complete");
    }
}

void OptimizerMetricsReporter::RemoveInstanceMetrics(const std::string &instance_id) {
    if (!metrics_registry_) {
        return;
    }
    MetricsTags filter = {{"instance_id", instance_id}};
    auto removed = metrics_registry_->RemoveByTagFilter(filter);
    if (removed > 0) {
        KVCM_LOG_INFO("OptimizerMetricsReporter: removed %zu metrics for instance[%s]", removed, instance_id.c_str());
    }
}

} // namespace kv_cache_manager
