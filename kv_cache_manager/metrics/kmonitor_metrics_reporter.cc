#include "kv_cache_manager/metrics/kmonitor_metrics_reporter.h"

#include <limits>
#include <unordered_map>

#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/MetricsReporter.h"
#include "kv_cache_manager/common/common_util.h"
#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/metrics/kmon_param.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {

// void stopKmonitorFactory() { kmonitor::KMonitorFactory::Shutdown(); }

void setHippoTags(kmonitor::MetricsConfig &config) {
    if (std::getenv("HIPPO_ROLE")) {
        auto host_ip = EnvUtil::GetEnv("HIPPO_SLAVE_IP", "");
        config.AddGlobalTag("host_ip", host_ip);
        config.AddGlobalTag("container_ip", EnvUtil::GetEnv("RequestedIP", host_ip));
        config.AddGlobalTag("hippo_role", EnvUtil::GetEnv("HIPPO_ROLE", ""));
        config.AddGlobalTag("hippo_app", EnvUtil::GetEnv("HIPPO_APP", ""));
        config.AddGlobalTag("hippo_group", EnvUtil::GetEnv("HIPPO_SERVICE_NAME", ""));
    }
}

} // namespace

#define DECLARE_METRICS(group, name) std::unique_ptr<kmonitor::MutableMetric> group##_##name##_metrics;

struct KmonitorMetricsReporter::Context {
    kmonitor::KMonitor *kmonitor = nullptr;

    /* ===================== per query metrics ====================== */

    // service metrics
    DECLARE_METRICS(service, qps);
    DECLARE_METRICS(service, query_rt_us);
    DECLARE_METRICS(service, error_qps);
    DECLARE_METRICS(service, request_queue_size);

    // manager metrics metrics
    DECLARE_METRICS(manager, request_key_count);
    DECLARE_METRICS(manager, prefix_match_len);
    DECLARE_METRICS(manager, prefix_match_time_us);
    DECLARE_METRICS(manager, lock_write_location_retry_times);
    DECLARE_METRICS(manager, write_cache_io_cost_us);
    DECLARE_METRICS(manager, filter_write_cache_time_us);
    DECLARE_METRICS(manager, gen_write_location_us);
    DECLARE_METRICS(manager, put_write_location_manager_us);
    DECLARE_METRICS(manager, batch_get_location_time_us);
    DECLARE_METRICS(manager, batch_add_location_time_us);
    DECLARE_METRICS(manager, batch_update_location_time_us);

    // meta searcher metrics
    DECLARE_METRICS(meta_searcher, indexer_get_time_us);
    DECLARE_METRICS(meta_searcher, indexer_read_modify_write_time_us);
    DECLARE_METRICS(meta_searcher, index_serialize_time_us);
    DECLARE_METRICS(meta_searcher, index_deserialize_time_us);
    DECLARE_METRICS(meta_searcher, indexer_query_times);

    // meta indexer metrics
    DECLARE_METRICS(meta_indexer, query_key_count);
    DECLARE_METRICS(meta_indexer, get_not_exist_key_count);
    DECLARE_METRICS(meta_indexer, query_batch_num);
    DECLARE_METRICS(meta_indexer, search_cache_hit_count);
    DECLARE_METRICS(meta_indexer, search_cache_miss_count);
    DECLARE_METRICS(meta_indexer, search_cache_hit_ratio);
    DECLARE_METRICS(meta_indexer, io_data_size);
    DECLARE_METRICS(meta_indexer, put_io_time_us);
    DECLARE_METRICS(meta_indexer, update_io_time_us);
    DECLARE_METRICS(meta_indexer, upsert_io_time_us);
    DECLARE_METRICS(meta_indexer, delete_io_time_us);
    DECLARE_METRICS(meta_indexer, get_io_time_us);
    DECLARE_METRICS(meta_indexer, read_modify_write_put_key_count);
    DECLARE_METRICS(meta_indexer, read_modify_write_update_key_count);
    DECLARE_METRICS(meta_indexer, read_modify_write_skip_key_count);
    DECLARE_METRICS(meta_indexer, read_modify_write_delete_key_count);

    // data storage metrics
    DECLARE_METRICS(data_storage, create_qps);
    DECLARE_METRICS(data_storage, create_keys_qps);
    DECLARE_METRICS(data_storage, create_time_us);

    /* ==================== intervallic metrics ===================== */

    // meta indexer metrics
    DECLARE_METRICS(meta_indexer, total_key_count);
    DECLARE_METRICS(meta_indexer, total_cache_usage);
    DECLARE_METRICS(meta_indexer, total_data_size);

    // data storage metrics
    DECLARE_METRICS(data_storage, healthy_status);
    DECLARE_METRICS(data_storage, storage_usage_ratio);

    // schedule plan executor metrics
    DECLARE_METRICS(scheduler_plan_executor, waiting_task_count);
    DECLARE_METRICS(scheduler_plan_executor, executing_task_count);

    // cache reclaimer metrics
    DECLARE_METRICS(cache_reclaimer, block_submit_count);
    DECLARE_METRICS(cache_reclaimer, location_submit_count);
    DECLARE_METRICS(cache_reclaimer, block_del_count);
    DECLARE_METRICS(cache_reclaimer, location_del_count);

    // cache manager
    DECLARE_METRICS(cache_manager, write_location_expire_size);
    DECLARE_METRICS(cache_manager_group, usage_ratio);
    DECLARE_METRICS(cache_manager_instance, key_count);
    DECLARE_METRICS(cache_manager_instance, byte_size);

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
            if (LIKELY(iter != tag_cache_.end())) {
                return iter->second;
            }
        }
        {
            std::unique_lock write_guard(mutex_);
            // double check
            auto iter = tag_cache_.find(base_tags);
            if (LIKELY(iter != tag_cache_.end())) {
                return iter->second;
            }
            kmonitor::MetricsTags tags(base_tags);
            tag_cache_[base_tags] = tags;
            return tags;
        }
    }
};

#undef DECLARE_METRICS

KmonitorMetricsReporter::KmonitorMetricsReporter() : ctx_(new Context{}) {}

KmonitorMetricsReporter::~KmonitorMetricsReporter() {}

bool KmonitorMetricsReporter::Init(std::shared_ptr<CacheManager> cache_manager,
                                   std::shared_ptr<MetricsRegistry> metrics_registry,
                                   const std::string &config) {
    if (!LocalMetricsReporter::Init(std::move(cache_manager), std::move(metrics_registry), config)) {
        return false;
    }

    KmonParam param;
    param.Init();

    if (!param.kmonitor_metrics_reporter_cache_limit.empty()) {
        size_t limit = std::atoll(param.kmonitor_metrics_reporter_cache_limit.c_str());
        if (limit > 0) {
            kmonitor::MetricsReporter::setMetricsReporterCacheLimit(limit);
            KVCM_LOG_INFO("set metrics reporter cache limit [%lu].", limit);
        }
    }

    if (param.kmonitor_normal_sample_period > 0) {
        KVCM_LOG_INFO("set kmonitor normal sample period [%d] seconds.", param.kmonitor_normal_sample_period);

        kmonitor::MetricLevelConfig config;
        config.period[kmonitor::FATAL] = static_cast<unsigned int>(param.kmonitor_normal_sample_period);
        kmonitor::MetricLevelManager::SetGlobalLevelConfig(config);
    }

    kmonitor::MetricsConfig metrics_config;
    metrics_config.set_tenant_name(param.kmonitor_tenant);
    metrics_config.set_service_name(param.kmonitor_service_name);

    std::string sink_address = param.kmonitor_sink_address;
    if (!param.kmonitor_port.empty()) {
        sink_address += ":" + param.kmonitor_port;
    }
    metrics_config.set_sink_address(sink_address.c_str());

    metrics_config.set_enable_log_file_sink(param.kmonitor_enable_log_file_sink);
    // metrics_config.set_enable_prometheus_sink(param.kmonitor_enable_prometheus_sink);
    metrics_config.set_manually_mode(param.kmonitor_manually_mode);
    metrics_config.set_inited(true);

    metrics_config.AddGlobalTag("hippo_slave_ip", param.hippo_slave_ip);

    for (const auto &pair : param.kmonitor_tags) {
        metrics_config.AddGlobalTag(pair.first, pair.second);
    }

    setHippoTags(metrics_config);

    if (!kmonitor::KMonitorFactory::Init(metrics_config)) {
        KVCM_LOG_ERROR("Init kmonitor factory failed");
        return false;
    }

    kmonitor::KMonitorFactory::registerBuildInMetrics(nullptr, param.kmonitor_metrics_prefix);
    KVCM_LOG_INFO("KMonitorFactory::registerBuildInMetrics() finished");

    kmonitor::KMonitorFactory::Start();
    KVCM_LOG_INFO("KMonitorFactory::Start() finished");

    return InitMetrics();
}

#define REGISTER_QPS_METRIC(group, name)                                                                               \
    do {                                                                                                               \
        std::string metric_name = #group "." #name;                                                                    \
        ctx_->group##_##name##_metrics.reset(reporter->RegisterMetric(metric_name, kmonitor::QPS, kmonitor::FATAL));   \
        if (nullptr == ctx_->group##_##name##_metrics) {                                                               \
            KVCM_LOG_ERROR("failed to register metric:[%s]", metric_name.c_str());                                     \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#define REGISTER_GAUGE_METRIC(group, name)                                                                             \
    do {                                                                                                               \
        std::string metric_name = #group "." #name;                                                                    \
        ctx_->group##_##name##_metrics.reset(reporter->RegisterMetric(metric_name, kmonitor::GAUGE, kmonitor::FATAL)); \
        if (nullptr == ctx_->group##_##name##_metrics) {                                                               \
            KVCM_LOG_ERROR("failed to register metric:[%s]", metric_name.c_str());                                     \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

bool KmonitorMetricsReporter::InitMetrics() {
    ctx_->kmonitor = kmonitor::KMonitorFactory::GetKMonitor("kvcm_default");
    auto reporter = ctx_->kmonitor;

    /* ===================== per query metrics ====================== */

    // service metrics
    REGISTER_QPS_METRIC(service, qps);
    REGISTER_GAUGE_METRIC(service, query_rt_us);
    REGISTER_QPS_METRIC(service, error_qps);
    REGISTER_GAUGE_METRIC(service, request_queue_size);

    // manager metrics
    REGISTER_GAUGE_METRIC(manager, request_key_count);
    REGISTER_GAUGE_METRIC(manager, prefix_match_len);
    REGISTER_GAUGE_METRIC(manager, prefix_match_time_us);
    REGISTER_GAUGE_METRIC(manager, lock_write_location_retry_times);
    REGISTER_GAUGE_METRIC(manager, write_cache_io_cost_us);
    REGISTER_GAUGE_METRIC(manager, filter_write_cache_time_us);
    REGISTER_GAUGE_METRIC(manager, gen_write_location_us);
    REGISTER_GAUGE_METRIC(manager, put_write_location_manager_us);
    REGISTER_GAUGE_METRIC(manager, batch_get_location_time_us);
    REGISTER_GAUGE_METRIC(manager, batch_add_location_time_us);
    REGISTER_GAUGE_METRIC(manager, batch_update_location_time_us);

    // meta searcher metrics
    REGISTER_GAUGE_METRIC(meta_searcher, indexer_get_time_us);
    REGISTER_GAUGE_METRIC(meta_searcher, indexer_read_modify_write_time_us);
    REGISTER_GAUGE_METRIC(meta_searcher, index_serialize_time_us);
    REGISTER_GAUGE_METRIC(meta_searcher, index_deserialize_time_us);
    REGISTER_GAUGE_METRIC(meta_searcher, indexer_query_times);

    // meta indexer metrics
    REGISTER_GAUGE_METRIC(meta_indexer, query_key_count);
    REGISTER_GAUGE_METRIC(meta_indexer, get_not_exist_key_count);
    REGISTER_GAUGE_METRIC(meta_indexer, query_batch_num);
    REGISTER_GAUGE_METRIC(meta_indexer, search_cache_hit_count);
    REGISTER_GAUGE_METRIC(meta_indexer, search_cache_miss_count);
    REGISTER_GAUGE_METRIC(meta_indexer, search_cache_hit_ratio);
    REGISTER_GAUGE_METRIC(meta_indexer, io_data_size);
    REGISTER_GAUGE_METRIC(meta_indexer, put_io_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, update_io_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, upsert_io_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, delete_io_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, get_io_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, read_modify_write_put_key_count);
    REGISTER_GAUGE_METRIC(meta_indexer, read_modify_write_update_key_count);
    REGISTER_GAUGE_METRIC(meta_indexer, read_modify_write_skip_key_count);
    REGISTER_GAUGE_METRIC(meta_indexer, read_modify_write_delete_key_count);

    // data storage metrics
    REGISTER_QPS_METRIC(data_storage, create_qps);
    REGISTER_QPS_METRIC(data_storage, create_keys_qps);
    REGISTER_GAUGE_METRIC(data_storage, create_time_us);

    /* ==================== intervallic metrics ===================== */

    // meta indexer metrics
    REGISTER_GAUGE_METRIC(meta_indexer, total_key_count);
    REGISTER_GAUGE_METRIC(meta_indexer, total_cache_usage);
    REGISTER_GAUGE_METRIC(meta_indexer, total_data_size);

    // data storage metrics
    REGISTER_GAUGE_METRIC(data_storage, healthy_status);
    REGISTER_GAUGE_METRIC(data_storage, storage_usage_ratio);

    // schedule plan executor metrics
    REGISTER_GAUGE_METRIC(scheduler_plan_executor, waiting_task_count);
    REGISTER_GAUGE_METRIC(scheduler_plan_executor, executing_task_count);

    // cache reclaimer metrics
    REGISTER_GAUGE_METRIC(cache_reclaimer, block_submit_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, location_submit_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, block_del_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, location_del_count);

    // cache manager
    REGISTER_GAUGE_METRIC(cache_manager, write_location_expire_size);
    REGISTER_GAUGE_METRIC(cache_manager_group, usage_ratio);
    REGISTER_GAUGE_METRIC(cache_manager_instance, key_count);
    REGISTER_GAUGE_METRIC(cache_manager_instance, byte_size);

    return true;
}

#undef REGISTER_QPS_METRIC
#undef REGISTER_GAUGE_METRIC

#define REPORT_METRICS(group, name, value)                                                                             \
    do {                                                                                                               \
        ctx_->group##_##name##_metrics->Report(&tags, value);                                                          \
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
        double v;                                                                                                      \
        STEAL_METRICS_(p, group, name, v);                                                                             \
        if (!(std::isnan(v))) {                                                                                        \
            REPORT_METRICS(group, name, v);                                                                            \
        }                                                                                                              \
    } while (0)

void KmonitorMetricsReporter::ReportPerQuery(MetricsCollector *collector) {
    LocalMetricsReporter::ReportPerQuery(collector);

    if (!ctx_->kmonitor) {
        KVCM_LOG_WARN("kmonitor is null, ReportPerQuery ignored.");
        return;
    }

    if (dynamic_cast<ServiceMetricsCollector *>(collector)) {
        auto *p = dynamic_cast<ServiceMetricsCollector *>(collector);

        // TODO 所有Query维度指标使用相同大tag，kmon扛得住吗？
        const kmonitor::MetricsTags tags = ctx_->GetKmonitorTags(p->GetMetricsTags());

        // service metrics
        REPORT_METRICS(service, qps, 1.0);
        REPORT_COLLECTED_METRICS(service, query_rt_us);
        REPORT_COLLECTED_METRICS(service, request_queue_size);

        do {
            // TODO(rui): better handling double error_code
            double service_error_code_v;
            GET_METRICS_(p, service, error_code, service_error_code_v);
            REPORT_METRICS_WHEN(service, error_qps, 1.0, !CommonUtil::IsZeroDouble(service_error_code_v));
        } while (false);

        // manager metrics
        REPORT_COLLECTED_METRICS(manager, request_key_count);
        REPORT_COLLECTED_METRICS(manager, prefix_match_len);
        REPORT_COLLECTED_METRICS(manager, prefix_match_time_us);
        REPORT_COLLECTED_METRICS(manager, lock_write_location_retry_times);
        REPORT_COLLECTED_METRICS(manager, write_cache_io_cost_us);
        REPORT_COLLECTED_METRICS(manager, filter_write_cache_time_us);
        REPORT_COLLECTED_METRICS(manager, gen_write_location_us);
        REPORT_COLLECTED_METRICS(manager, put_write_location_manager_us);
        REPORT_COLLECTED_METRICS(manager, batch_get_location_time_us);
        REPORT_STEAL_METRICS(manager, batch_add_location_time_us);
        REPORT_COLLECTED_METRICS(manager, batch_update_location_time_us);

        // meta searcher metrics
        REPORT_COLLECTED_METRICS(meta_searcher, indexer_get_time_us);
        REPORT_STEAL_METRICS(meta_searcher, indexer_read_modify_write_time_us);
        REPORT_STEAL_METRICS(meta_searcher, index_serialize_time_us);
        REPORT_COLLECTED_METRICS(meta_searcher, index_deserialize_time_us);
        REPORT_COLLECTED_METRICS(meta_searcher, indexer_query_times);

        // meta indexer metrics
        REPORT_COLLECTED_METRICS(meta_indexer, query_key_count);
        REPORT_COLLECTED_METRICS(meta_indexer, get_not_exist_key_count);
        REPORT_STEAL_METRICS(meta_indexer, query_batch_num);
        REPORT_COLLECTED_METRICS(meta_indexer, search_cache_hit_count);
        REPORT_COLLECTED_METRICS(meta_indexer, search_cache_miss_count);
        REPORT_COLLECTED_METRICS(meta_indexer, search_cache_hit_ratio);
        REPORT_COLLECTED_METRICS(meta_indexer, io_data_size);
        REPORT_COLLECTED_METRICS(meta_indexer, put_io_time_us);
        REPORT_COLLECTED_METRICS(meta_indexer, update_io_time_us);
        REPORT_STEAL_METRICS(meta_indexer, upsert_io_time_us);
        REPORT_COLLECTED_METRICS(meta_indexer, delete_io_time_us);
        REPORT_COLLECTED_METRICS(meta_indexer, get_io_time_us);
        REPORT_STEAL_METRICS(meta_indexer, read_modify_write_put_key_count);
        REPORT_COLLECTED_METRICS(meta_indexer, read_modify_write_update_key_count);
        REPORT_COLLECTED_METRICS(meta_indexer, read_modify_write_skip_key_count);
        REPORT_COLLECTED_METRICS(meta_indexer, read_modify_write_delete_key_count);
    } else if (dynamic_cast<DataStorageMetricsCollector *>(collector)) {
        const auto *p = dynamic_cast<DataStorageMetricsCollector *>(collector);
        const kmonitor::MetricsTags tags = ctx_->GetKmonitorTags(p->GetMetricsTags());
        REPORT_METRICS(data_storage, create_qps, 1.0);
        REPORT_COLLECTED_METRICS(data_storage, create_keys_qps);
        REPORT_COLLECTED_METRICS(data_storage, create_time_us);
    }
}

void KmonitorMetricsReporter::ReportInterval() {
    LocalMetricsReporter::ReportInterval();

    if (!cache_manager_) {
        return;
    }

    if (!ctx_->kmonitor) {
        KVCM_LOG_WARN("kmonitor is null, ReportInterval ignored.");
        return;
    }

    do {
        // for meta indexer accumulative metrics
        // these metrics are already updated by local metrics reporter
        // so report them directly to kmonitor
        const auto p = KVCM_META_INDEXER_ACC_METRICS_COLLECTOR_PTR(MetaIndexerAccumulative);
        if (!p) {
            break;
        }

        double total_key_count_v;
        double total_cache_usage_v;
        GET_METRICS_(p, meta_indexer, total_key_count, total_key_count_v);
        GET_METRICS_(p, meta_indexer, total_cache_usage, total_cache_usage_v);

        const kmonitor::MetricsTags tags;
        REPORT_METRICS(meta_indexer, total_key_count, total_key_count_v);
        REPORT_METRICS(meta_indexer, total_cache_usage, total_cache_usage_v);
    } while (false);

    do {
        // for data storage metrics
        // these metrics are already updated by local metrics reporter
        // so report them directly to kmonitor
        const auto &vec = data_storage_interval_metrics_collectors_.GetMetricsCollectors();
        for (const auto &mc : vec) {
            if (const auto p = std::dynamic_pointer_cast<DataStorageIntervalMetricsCollector>(mc); p != nullptr) {
                const kmonitor::MetricsTags tags = ctx_->GetKmonitorTags(p->GetMetricsTags());
                double healthy_status_v;
                double storage_usage_ratio_v;
                GET_METRICS_(p, data_storage, healthy_status, healthy_status_v);
                REPORT_METRICS(data_storage, healthy_status, healthy_status_v);
                GET_METRICS_(p, data_storage, storage_usage_ratio, storage_usage_ratio_v);
                REPORT_METRICS(data_storage, storage_usage_ratio, storage_usage_ratio_v);
            }
        }
    } while (false);

    do {
        // schedule plan executor
        // note these metrics are reported to the local metrics registry
        // in real-time, but for kmonitor they are reported with time interval
        const auto spe = cache_manager_->schedule_plan_executor();
        if (!spe) {
            break;
        }

        std::uint64_t w_task_count_v;
        std::uint64_t e_task_count_v;

        GET_METRICS_(spe, schedule_plan_executor, waiting_task_count, w_task_count_v);
        GET_METRICS_(spe, schedule_plan_executor, executing_task_count, e_task_count_v);

        const kmonitor::MetricsTags tags;
        REPORT_METRICS(scheduler_plan_executor, waiting_task_count, static_cast<double>(w_task_count_v));
        REPORT_METRICS(scheduler_plan_executor, executing_task_count, static_cast<double>(e_task_count_v));
    } while (false);

    do {
        // cache reclaimer
        // note these metrics are reported to the local metrics registry
        // in real-time, but for kmonitor they are reported with time interval
        const auto cr = cache_manager_->cache_reclaimer();
        if (!cr) {
            break;
        }

        std::uint64_t blk_submit_count_v;
        std::uint64_t loc_submit_count_v;
        std::uint64_t blk_del_count_v;
        std::uint64_t loc_del_count_v;

        GET_METRICS_(cr, cache_reclaimer, block_submit_count, blk_submit_count_v);
        GET_METRICS_(cr, cache_reclaimer, location_submit_count, loc_submit_count_v);
        GET_METRICS_(cr, cache_reclaimer, block_del_count, blk_del_count_v);
        GET_METRICS_(cr, cache_reclaimer, location_del_count, loc_del_count_v);

        const kmonitor::MetricsTags tags;
        REPORT_METRICS(cache_reclaimer, block_submit_count, static_cast<double>(blk_submit_count_v));
        REPORT_METRICS(cache_reclaimer, location_submit_count, static_cast<double>(loc_submit_count_v));
        REPORT_METRICS(cache_reclaimer, block_del_count, static_cast<double>(blk_del_count_v));
        REPORT_METRICS(cache_reclaimer, location_del_count, static_cast<double>(loc_del_count_v));
    } while (false);

    do {
        {
            const auto p = KVCM_CACHE_MANAGER_METRICS_COLLECTOR_PTR(CacheManager);
            if (!p) {
                break;
            }

            double write_location_expire_size_v;
            GET_METRICS_(p, cache_manager, write_location_expire_size, write_location_expire_size_v);
            const kmonitor::MetricsTags tags;
            REPORT_METRICS(cache_manager, write_location_expire_size, write_location_expire_size_v);
        }

        {
            const auto &vec = cache_manager_group_interval_metrics_collectors_.GetMetricsCollectors();
            for (const auto &mc : vec) {
                if (const auto p = std::dynamic_pointer_cast<CacheManagerGroupMetricsCollector>(mc); p != nullptr) {
                    const kmonitor::MetricsTags tags = ctx_->GetKmonitorTags(p->GetMetricsTags());
                    double usage_ratio_v;
                    GET_METRICS_(p, cache_manager_group, usage_ratio, usage_ratio_v);
                    REPORT_METRICS(cache_manager_group, usage_ratio, usage_ratio_v);
                }
            }
        }

        {
            const auto &vec = cache_manager_instance_interval_metrics_collectors_.GetMetricsCollectors();
            for (const auto &mc : vec) {
                if (const auto p = std::dynamic_pointer_cast<CacheManagerInstanceMetricsCollector>(mc); p != nullptr) {
                    const kmonitor::MetricsTags tags = ctx_->GetKmonitorTags(p->GetMetricsTags());
                    double key_count_v, byte_size_v;
                    GET_METRICS_(p, cache_manager_instance, key_count, key_count_v);
                    REPORT_METRICS(cache_manager_instance, key_count, key_count_v);
                    GET_METRICS_(p, cache_manager_instance, byte_size, byte_size_v);
                    REPORT_METRICS(cache_manager_instance, byte_size, byte_size_v);
                }
            }
        }
    } while (false);
}

} // namespace kv_cache_manager
