#include "kv_cache_manager/metrics/kmonitor_metrics_reporter.h"

#include <atomic>
#include <limits>
#include <unordered_map>

#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/MetricsReporter.h"
#include "kv_cache_manager/common/common_util.h"
#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/config/registry_manager.h"
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
    DECLARE_METRICS(manager, get_cache_location_query_block_counter);
    DECLARE_METRICS(manager, get_cache_location_hit_block_counter);
    DECLARE_METRICS(manager, prefix_match_time_us);
    DECLARE_METRICS(manager, batch_get_time_us);
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
    DECLARE_METRICS(meta_searcher, host_projection_time_us);
    DECLARE_METRICS(meta_searcher, host_prefix_reduce_time_us);
    DECLARE_METRICS(meta_searcher, indexer_read_modify_write_block_time_us);
    DECLARE_METRICS(meta_searcher, indexer_read_modify_write_location_time_us);
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
    DECLARE_METRICS(meta_indexer, upsert_io_time_us);
    DECLARE_METRICS(meta_indexer, lock_wait_time_us);
    DECLARE_METRICS(meta_indexer, delete_io_time_us);
    DECLARE_METRICS(meta_indexer, get_io_time_us);
    DECLARE_METRICS(meta_indexer, rand_io_time_us);
    DECLARE_METRICS(meta_indexer, rmw_get_io_time_us);
    DECLARE_METRICS(meta_indexer, read_modify_write_put_key_count);
    DECLARE_METRICS(meta_indexer, read_modify_write_update_key_count);
    DECLARE_METRICS(meta_indexer, read_modify_write_skip_key_count);
    DECLARE_METRICS(meta_indexer, read_modify_write_delete_key_count);
    DECLARE_METRICS(meta_indexer, async_enqueue_timeout_key_count);
    DECLARE_METRICS(meta_indexer, async_enqueue_time_us);
    DECLARE_METRICS(meta_indexer, cache_backend_put_time_us);
    DECLARE_METRICS(meta_indexer, cache_backend_upsert_time_us);
    DECLARE_METRICS(meta_indexer, cache_backend_delete_time_us);

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
    DECLARE_METRICS(data_storage, write_bytes_dispatched_total);

    // schedule plan executor metrics
    DECLARE_METRICS(scheduler_plan_executor, waiting_task_count);
    DECLARE_METRICS(scheduler_plan_executor, executing_task_count);

    // cache reclaimer metrics
    DECLARE_METRICS(cache_reclaimer, reclaim_cron_count);
    DECLARE_METRICS(cache_reclaimer, reclaim_job_count);
    DECLARE_METRICS(cache_reclaimer, block_submit_count);
    DECLARE_METRICS(cache_reclaimer, location_submit_count);
    DECLARE_METRICS(cache_reclaimer, block_del_count);
    DECLARE_METRICS(cache_reclaimer, location_del_count);
    DECLARE_METRICS(cache_reclaimer, credit_timeout_count);
    DECLARE_METRICS(cache_reclaimer, pending_limit_reject_count);
    DECLARE_METRICS(cache_reclaimer, duplicate_pending_location_filtered_count);
    DECLARE_METRICS(cache_reclaimer, reclaim_no_progress_backoff_count);
    DECLARE_METRICS(cache_reclaimer, delete_submit_count);
    DECLARE_METRICS(cache_reclaimer, delete_complete_count);
    DECLARE_METRICS(cache_reclaimer, delete_fail_count);

    DECLARE_METRICS(cache_reclaimer, reclaim_cron_duration_us);
    DECLARE_METRICS(cache_reclaimer, reclaim_quota_duration_us);
    DECLARE_METRICS(cache_reclaimer, reclaim_job_duration_us);
    DECLARE_METRICS(cache_reclaimer, reclaim_res_duration_us);
    DECLARE_METRICS(cache_reclaimer, reclaim_lru_sample_duration_us);
    DECLARE_METRICS(cache_reclaimer, reclaim_lru_batch_duration_us);
    DECLARE_METRICS(cache_reclaimer, reclaim_lru_filter_duration_us);
    DECLARE_METRICS(cache_reclaimer, reclaim_lru_submit_duration_us);
    DECLARE_METRICS(cache_reclaimer, pending_delete_handler_count);
    DECLARE_METRICS(cache_reclaimer, pending_location_count);
    DECLARE_METRICS(cache_reclaimer, pending_delete_bytes);
    DECLARE_METRICS(cache_reclaimer, credited_delete_bytes);
    DECLARE_METRICS(cache_reclaimer, predicted_deleted_key_count);
    DECLARE_METRICS(cache_reclaimer, oldest_pending_request_age_ms);

    DECLARE_METRICS(cache_reclaimer, reclaim_batch_lru_age_min_us);
    DECLARE_METRICS(cache_reclaimer, reclaim_batch_lru_age_max_us);
    DECLARE_METRICS(cache_reclaimer, reclaim_batch_lru_age_avg_us);
    DECLARE_METRICS(cache_reclaimer, reclaim_batch_create_age_min_us);
    DECLARE_METRICS(cache_reclaimer, reclaim_batch_create_age_max_us);
    DECLARE_METRICS(cache_reclaimer, reclaim_batch_create_age_avg_us);

    // cache garbage collector metrics
    DECLARE_METRICS(cache_gc, scan_round_count);
    DECLARE_METRICS(cache_gc, scan_key_count);
    DECLARE_METRICS(cache_gc, candidate_count);
    DECLARE_METRICS(cache_gc, delete_target_count);
    DECLARE_METRICS(cache_gc, delete_result_count);
    DECLARE_METRICS(cache_gc, operation_error_count);
    DECLARE_METRICS(cache_gc, inflight_delete_count);
    DECLARE_METRICS(cache_gc, inflight_delete_age_ms);
    DECLARE_METRICS(cache_gc, round_duration_ms);

    // cache manager
    DECLARE_METRICS(cache_manager, write_location_expire_size);
    DECLARE_METRICS(cache_manager_group, usage_ratio);
    DECLARE_METRICS(cache_manager_instance, key_count);
    DECLARE_METRICS(cache_manager_instance, byte_size);
    DECLARE_METRICS(cache_manager_instance, async_queue_max_size);
    DECLARE_METRICS(cache_manager_instance, async_queue_avg_size);
    DECLARE_METRICS(cache_manager_instance, async_flush_key_count);
    DECLARE_METRICS(cache_manager_instance, async_batch_flush_time_us);
    DECLARE_METRICS(cache_manager_instance, async_pipeline_error_count);
    DECLARE_METRICS(cache_manager_instance, max_lru_age_us);

    DECLARE_METRICS(manager, location_lookup_keys_total);
    DECLARE_METRICS(manager, location_lookup_requests_total);
    DECLARE_METRICS(manager, location_lookup_request_keys_total);
    DECLARE_METRICS(service, final_result_requests_total);
    DECLARE_METRICS(data_storage, create_result_operations_total);
    DECLARE_METRICS(data_storage, create_result_keys_total);
    DECLARE_METRICS(data_storage, create_result_bytes_total);
    DECLARE_METRICS(write_session, blocks_total);
    DECLARE_METRICS(write_session, inflight_blocks);
    DECLARE_METRICS(write_session, oldest_age_seconds);
    DECLARE_METRICS(tair_backend, operations_total);
    DECLARE_METRICS(tair_backend, duration_us);

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

#define REGISTER_NAMED_GAUGE_METRIC(group, name, full_name)                                                            \
    do {                                                                                                               \
        std::string metric_name = (full_name);                                                                         \
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
    // Workaround: KMonitor only supports QPS and GAUGE metric types (no COUNTER).
    // These are semantically cumulative counters; registered as GAUGE to match
    // existing codebase convention (e.g. cache_reclaimer counters).
    REGISTER_GAUGE_METRIC(manager, get_cache_location_query_block_counter);
    REGISTER_GAUGE_METRIC(manager, get_cache_location_hit_block_counter);
    REGISTER_GAUGE_METRIC(manager, prefix_match_time_us);
    REGISTER_GAUGE_METRIC(manager, batch_get_time_us);
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
    REGISTER_GAUGE_METRIC(meta_searcher, host_projection_time_us);
    REGISTER_GAUGE_METRIC(meta_searcher, host_prefix_reduce_time_us);
    REGISTER_GAUGE_METRIC(meta_searcher, indexer_read_modify_write_block_time_us);
    REGISTER_GAUGE_METRIC(meta_searcher, indexer_read_modify_write_location_time_us);
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
    REGISTER_GAUGE_METRIC(meta_indexer, upsert_io_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, lock_wait_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, delete_io_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, get_io_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, rand_io_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, rmw_get_io_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, read_modify_write_put_key_count);
    REGISTER_GAUGE_METRIC(meta_indexer, read_modify_write_update_key_count);
    REGISTER_GAUGE_METRIC(meta_indexer, read_modify_write_skip_key_count);
    REGISTER_GAUGE_METRIC(meta_indexer, read_modify_write_delete_key_count);
    REGISTER_GAUGE_METRIC(meta_indexer, async_enqueue_timeout_key_count);
    REGISTER_GAUGE_METRIC(meta_indexer, async_enqueue_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, cache_backend_put_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, cache_backend_upsert_time_us);
    REGISTER_GAUGE_METRIC(meta_indexer, cache_backend_delete_time_us);

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
    REGISTER_GAUGE_METRIC(data_storage, write_bytes_dispatched_total);

    // schedule plan executor metrics
    REGISTER_GAUGE_METRIC(scheduler_plan_executor, waiting_task_count);
    REGISTER_GAUGE_METRIC(scheduler_plan_executor, executing_task_count);

    // cache reclaimer metrics
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_cron_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_job_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, block_submit_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, location_submit_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, block_del_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, location_del_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, credit_timeout_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, pending_limit_reject_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, duplicate_pending_location_filtered_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_no_progress_backoff_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, delete_submit_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, delete_complete_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, delete_fail_count);

    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_cron_duration_us);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_quota_duration_us);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_job_duration_us);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_res_duration_us);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_lru_sample_duration_us);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_lru_batch_duration_us);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_lru_filter_duration_us);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_lru_submit_duration_us);
    REGISTER_GAUGE_METRIC(cache_reclaimer, pending_delete_handler_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, pending_location_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, pending_delete_bytes);
    REGISTER_GAUGE_METRIC(cache_reclaimer, credited_delete_bytes);
    REGISTER_GAUGE_METRIC(cache_reclaimer, predicted_deleted_key_count);
    REGISTER_GAUGE_METRIC(cache_reclaimer, oldest_pending_request_age_ms);

    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_batch_lru_age_min_us);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_batch_lru_age_max_us);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_batch_lru_age_avg_us);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_batch_create_age_min_us);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_batch_create_age_max_us);
    REGISTER_GAUGE_METRIC(cache_reclaimer, reclaim_batch_create_age_avg_us);

    // Cache GC counters are registered as gauges, matching the existing
    // cumulative counter convention used for CacheReclaimer. Tagged series
    // share one KMonitor metric and are distinguished at report time.
    REGISTER_GAUGE_METRIC(cache_gc, scan_round_count);
    REGISTER_GAUGE_METRIC(cache_gc, scan_key_count);
    REGISTER_GAUGE_METRIC(cache_gc, candidate_count);
    REGISTER_GAUGE_METRIC(cache_gc, delete_target_count);
    REGISTER_GAUGE_METRIC(cache_gc, delete_result_count);
    REGISTER_GAUGE_METRIC(cache_gc, operation_error_count);
    REGISTER_GAUGE_METRIC(cache_gc, inflight_delete_count);
    REGISTER_GAUGE_METRIC(cache_gc, inflight_delete_age_ms);
    REGISTER_GAUGE_METRIC(cache_gc, round_duration_ms);

    // cache manager
    REGISTER_GAUGE_METRIC(cache_manager, write_location_expire_size);
    REGISTER_GAUGE_METRIC(cache_manager_group, usage_ratio);
    REGISTER_GAUGE_METRIC(cache_manager_instance, key_count);
    REGISTER_GAUGE_METRIC(cache_manager_instance, byte_size);
    REGISTER_GAUGE_METRIC(cache_manager_instance, async_queue_max_size);
    REGISTER_GAUGE_METRIC(cache_manager_instance, async_queue_avg_size);
    REGISTER_GAUGE_METRIC(cache_manager_instance, async_flush_key_count);
    REGISTER_GAUGE_METRIC(cache_manager_instance, async_batch_flush_time_us);
    REGISTER_GAUGE_METRIC(cache_manager_instance, async_pipeline_error_count);
    REGISTER_GAUGE_METRIC(cache_manager_instance, max_lru_age_us);

    // MetricsRegistry keeps *_total values as cumulative counters. KMonitor has
    // no COUNTER type, so publish their cumulative values as GAUGE, matching the
    // existing cache-reclaimer convention.
    REGISTER_NAMED_GAUGE_METRIC(manager, location_lookup_keys_total, "manager.location_lookup.keys_total");
    REGISTER_NAMED_GAUGE_METRIC(manager, location_lookup_requests_total, "manager.location_lookup.requests_total");
    REGISTER_NAMED_GAUGE_METRIC(
        manager, location_lookup_request_keys_total, "manager.location_lookup.request_keys_total");
    REGISTER_NAMED_GAUGE_METRIC(service, final_result_requests_total, "service.final_result.requests_total");
    REGISTER_NAMED_GAUGE_METRIC(
        data_storage, create_result_operations_total, "data_storage.create_result.operations_total");
    REGISTER_NAMED_GAUGE_METRIC(data_storage, create_result_keys_total, "data_storage.create_result.keys_total");
    REGISTER_NAMED_GAUGE_METRIC(data_storage, create_result_bytes_total, "data_storage.create_result.bytes_total");
    REGISTER_NAMED_GAUGE_METRIC(write_session, blocks_total, "write_session.blocks_total");
    REGISTER_NAMED_GAUGE_METRIC(write_session, inflight_blocks, "write_session.inflight_blocks");
    REGISTER_NAMED_GAUGE_METRIC(write_session, oldest_age_seconds, "write_session.oldest_age_seconds");
    REGISTER_NAMED_GAUGE_METRIC(tair_backend, operations_total, "tair_backend.operations_total");
    REGISTER_NAMED_GAUGE_METRIC(tair_backend, duration_us, "tair_backend.duration_us");

    return true;
}

#undef REGISTER_QPS_METRIC
#undef REGISTER_GAUGE_METRIC
#undef REGISTER_NAMED_GAUGE_METRIC

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
        REPORT_COLLECTED_METRICS(manager, get_cache_location_query_block_counter);
        REPORT_COLLECTED_METRICS(manager, get_cache_location_hit_block_counter);
        REPORT_COLLECTED_METRICS(manager, prefix_match_time_us);
        REPORT_COLLECTED_METRICS(manager, batch_get_time_us);
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
        REPORT_COLLECTED_METRICS(meta_searcher, host_projection_time_us);
        REPORT_COLLECTED_METRICS(meta_searcher, host_prefix_reduce_time_us);
        REPORT_STEAL_METRICS(meta_searcher, indexer_read_modify_write_block_time_us);
        REPORT_STEAL_METRICS(meta_searcher, indexer_read_modify_write_location_time_us);
        REPORT_STEAL_METRICS(meta_searcher, index_serialize_time_us);
        REPORT_COLLECTED_METRICS(meta_searcher, index_deserialize_time_us);
        REPORT_COLLECTED_METRICS(meta_searcher, indexer_query_times);

        // meta indexer metrics
        REPORT_COLLECTED_METRICS(meta_indexer, query_key_count);
        REPORT_STEAL_METRICS(meta_indexer, get_not_exist_key_count);
        REPORT_STEAL_METRICS(meta_indexer, query_batch_num);
        REPORT_STEAL_METRICS(meta_indexer, search_cache_hit_count);
        REPORT_STEAL_METRICS(meta_indexer, search_cache_miss_count);
        REPORT_STEAL_METRICS(meta_indexer, search_cache_hit_ratio);
        REPORT_STEAL_METRICS(meta_indexer, io_data_size);
        REPORT_STEAL_METRICS(meta_indexer, put_io_time_us);
        REPORT_STEAL_METRICS(meta_indexer, upsert_io_time_us);
        REPORT_STEAL_METRICS(meta_indexer, lock_wait_time_us);
        REPORT_STEAL_METRICS(meta_indexer, delete_io_time_us);
        REPORT_STEAL_METRICS(meta_indexer, get_io_time_us);
        REPORT_STEAL_METRICS(meta_indexer, rand_io_time_us);
        REPORT_STEAL_METRICS(meta_indexer, rmw_get_io_time_us);
        REPORT_STEAL_METRICS(meta_indexer, read_modify_write_put_key_count);
        REPORT_STEAL_METRICS(meta_indexer, read_modify_write_update_key_count);
        REPORT_STEAL_METRICS(meta_indexer, read_modify_write_skip_key_count);
        REPORT_STEAL_METRICS(meta_indexer, read_modify_write_delete_key_count);
        REPORT_STEAL_METRICS(meta_indexer, async_enqueue_timeout_key_count);
        REPORT_STEAL_METRICS(meta_indexer, async_enqueue_time_us);
        REPORT_STEAL_METRICS(meta_indexer, cache_backend_put_time_us);
        REPORT_STEAL_METRICS(meta_indexer, cache_backend_upsert_time_us);
        REPORT_STEAL_METRICS(meta_indexer, cache_backend_delete_time_us);
    } else if (dynamic_cast<EventReportMetricsCollector *>(collector)) {
        auto *p = dynamic_cast<EventReportMetricsCollector *>(collector);
        const kmonitor::MetricsTags tags = ctx_->GetKmonitorTags(p->GetMetricsTags());
        REPORT_METRICS(service, qps, 1.0);
        REPORT_METRICS(service, query_rt_us, p->GetRequestRtUsSample());

        const double error_code = p->GetErrorCodeSample();
        REPORT_METRICS_WHEN(service, error_qps, 1.0, !CommonUtil::IsZeroDouble(error_code));
        if (p->HasRequestKeyCountSample()) {
            REPORT_METRICS(manager, request_key_count, p->GetRequestKeyCountSample());
        }
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

    if (metrics_registry_) {
        const auto report_registry_metric = [this](kmonitor::MutableMetric *metric, const char *name) {
            if (metric == nullptr) {
                return;
            }
            VisitMetricSamples(name, [this, metric](const MetricsTags &base_tags, double sample) {
                const auto tags = ctx_->GetKmonitorTags(base_tags);
                metric->Report(&tags, sample);
            });
        };
        report_registry_metric(ctx_->manager_location_lookup_keys_total_metrics.get(),
                               "manager.location_lookup.keys_total");
        report_registry_metric(ctx_->manager_location_lookup_requests_total_metrics.get(),
                               "manager.location_lookup.requests_total");
        report_registry_metric(ctx_->manager_location_lookup_request_keys_total_metrics.get(),
                               "manager.location_lookup.request_keys_total");
        report_registry_metric(ctx_->service_final_result_requests_total_metrics.get(),
                               "service.final_result.requests_total");
        report_registry_metric(ctx_->data_storage_create_result_operations_total_metrics.get(),
                               "data_storage.create_result.operations_total");
        report_registry_metric(ctx_->data_storage_create_result_keys_total_metrics.get(),
                               "data_storage.create_result.keys_total");
        report_registry_metric(ctx_->data_storage_create_result_bytes_total_metrics.get(),
                               "data_storage.create_result.bytes_total");
        report_registry_metric(ctx_->write_session_blocks_total_metrics.get(), "write_session.blocks_total");
        report_registry_metric(ctx_->write_session_inflight_blocks_metrics.get(), "write_session.inflight_blocks");
        report_registry_metric(ctx_->write_session_oldest_age_seconds_metrics.get(),
                               "write_session.oldest_age_seconds");
        report_registry_metric(ctx_->tair_backend_operations_total_metrics.get(), "tair_backend.operations_total");
        report_registry_metric(ctx_->tair_backend_duration_us_metrics.get(), "tair_backend.duration_us");
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
        // for per-storage accumulative dispatched write bytes
        // the counter is updated in real-time on each backend's DataStorageMetricsCollector
        // (via DataStorageManager::RecordWriteBytes, covering both normal writes and migration
        // copies at their dispatch points), so read the current cumulative values and report
        // them to kmonitor periodically
        const auto registry_manager = cache_manager_->GetRegistryManager();
        if (!registry_manager) {
            break;
        }
        const auto data_storage_manager = registry_manager->data_storage_manager();
        if (!data_storage_manager) {
            break;
        }

        const auto storage_names = data_storage_manager->GetAllStorageNames();
        for (const auto &unique_name : storage_names) {
            const auto storage_backend = data_storage_manager->GetDataStorageBackend(unique_name);
            if (storage_backend == nullptr) {
                continue;
            }
            const auto collector = storage_backend->GetMetricsCollector();
            if (collector == nullptr) {
                continue;
            }
            std::uint64_t write_bytes_v;
            GET_METRICS_(collector, data_storage, write_bytes_dispatched_total, write_bytes_v);
            const kmonitor::MetricsTags tags = ctx_->GetKmonitorTags(collector->GetMetricsTags());
            REPORT_METRICS(data_storage, write_bytes_dispatched_total, static_cast<double>(write_bytes_v));
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

        double w_task_count_v;
        double e_task_count_v;

        GET_METRICS_(spe, schedule_plan_executor, waiting_task_count, w_task_count_v);
        GET_METRICS_(spe, schedule_plan_executor, executing_task_count, e_task_count_v);

        const kmonitor::MetricsTags tags;
        REPORT_METRICS(scheduler_plan_executor, waiting_task_count, w_task_count_v);
        REPORT_METRICS(scheduler_plan_executor, executing_task_count, e_task_count_v);
    } while (false);

    do {
        // cache reclaimer
        // note these metrics are reported to the local metrics registry
        // in real-time, but for kmonitor they are reported with time interval
        const auto cr = cache_manager_->cache_reclaimer();
        if (!cr || cr->IsPaused() || !cr->IsRunning()) {
            break;
        }

        std::uint64_t reclaim_cron_count_v;
        std::uint64_t reclaim_job_count_v;
        std::uint64_t blk_submit_count_v;
        std::uint64_t loc_submit_count_v;
        std::uint64_t blk_del_count_v;
        std::uint64_t loc_del_count_v;
        std::uint64_t credit_timeout_count_v;
        std::uint64_t pending_limit_reject_count_v;
        std::uint64_t duplicate_pending_location_filtered_count_v;
        std::uint64_t reclaim_no_progress_backoff_count_v;
        std::uint64_t delete_submit_count_v;
        std::uint64_t delete_complete_count_v;
        std::uint64_t delete_fail_count_v;

        double reclaim_cron_duration_us_v;
        double reclaim_quota_duration_us_v;
        double reclaim_job_duration_us_v;
        double reclaim_res_duration_us_v;
        double reclaim_lru_sample_duration_us_v;
        double reclaim_lru_batch_duration_us_v;
        double reclaim_lru_filter_duration_us_v;
        double reclaim_lru_submit_duration_us_v;
        double pending_delete_handler_count_v;
        double pending_location_count_v;
        double pending_delete_bytes_v;
        double credited_delete_bytes_v;
        double predicted_deleted_key_count_v;
        double oldest_pending_request_age_ms_v;

        double reclaim_batch_lru_age_min_us_v;
        double reclaim_batch_lru_age_max_us_v;
        double reclaim_batch_lru_age_avg_us_v;
        double reclaim_batch_create_age_min_us_v;
        double reclaim_batch_create_age_max_us_v;
        double reclaim_batch_create_age_avg_us_v;

        GET_METRICS_(cr, cache_reclaimer, reclaim_cron_count, reclaim_cron_count_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_job_count, reclaim_job_count_v);
        GET_METRICS_(cr, cache_reclaimer, block_submit_count, blk_submit_count_v);
        GET_METRICS_(cr, cache_reclaimer, location_submit_count, loc_submit_count_v);
        GET_METRICS_(cr, cache_reclaimer, block_del_count, blk_del_count_v);
        GET_METRICS_(cr, cache_reclaimer, location_del_count, loc_del_count_v);
        GET_METRICS_(cr, cache_reclaimer, credit_timeout_count, credit_timeout_count_v);
        GET_METRICS_(cr, cache_reclaimer, pending_limit_reject_count, pending_limit_reject_count_v);
        GET_METRICS_(cr,
                     cache_reclaimer,
                     duplicate_pending_location_filtered_count,
                     duplicate_pending_location_filtered_count_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_no_progress_backoff_count, reclaim_no_progress_backoff_count_v);
        GET_METRICS_(cr, cache_reclaimer, delete_submit_count, delete_submit_count_v);
        GET_METRICS_(cr, cache_reclaimer, delete_complete_count, delete_complete_count_v);
        GET_METRICS_(cr, cache_reclaimer, delete_fail_count, delete_fail_count_v);

        GET_METRICS_(cr, cache_reclaimer, reclaim_cron_duration_us, reclaim_cron_duration_us_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_quota_duration_us, reclaim_quota_duration_us_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_job_duration_us, reclaim_job_duration_us_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_res_duration_us, reclaim_res_duration_us_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_lru_sample_duration_us, reclaim_lru_sample_duration_us_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_lru_batch_duration_us, reclaim_lru_batch_duration_us_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_lru_filter_duration_us, reclaim_lru_filter_duration_us_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_lru_submit_duration_us, reclaim_lru_submit_duration_us_v);
        GET_METRICS_(cr, cache_reclaimer, pending_delete_handler_count, pending_delete_handler_count_v);
        GET_METRICS_(cr, cache_reclaimer, pending_location_count, pending_location_count_v);
        GET_METRICS_(cr, cache_reclaimer, pending_delete_bytes, pending_delete_bytes_v);
        GET_METRICS_(cr, cache_reclaimer, credited_delete_bytes, credited_delete_bytes_v);
        GET_METRICS_(cr, cache_reclaimer, predicted_deleted_key_count, predicted_deleted_key_count_v);
        GET_METRICS_(cr, cache_reclaimer, oldest_pending_request_age_ms, oldest_pending_request_age_ms_v);

        GET_METRICS_(cr, cache_reclaimer, reclaim_batch_lru_age_min_us, reclaim_batch_lru_age_min_us_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_batch_lru_age_max_us, reclaim_batch_lru_age_max_us_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_batch_lru_age_avg_us, reclaim_batch_lru_age_avg_us_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_batch_create_age_min_us, reclaim_batch_create_age_min_us_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_batch_create_age_max_us, reclaim_batch_create_age_max_us_v);
        GET_METRICS_(cr, cache_reclaimer, reclaim_batch_create_age_avg_us, reclaim_batch_create_age_avg_us_v);

        const kmonitor::MetricsTags tags;
        REPORT_METRICS(cache_reclaimer, reclaim_cron_count, static_cast<double>(reclaim_cron_count_v));
        REPORT_METRICS(cache_reclaimer, reclaim_job_count, static_cast<double>(reclaim_job_count_v));
        REPORT_METRICS(cache_reclaimer, block_submit_count, static_cast<double>(blk_submit_count_v));
        REPORT_METRICS(cache_reclaimer, location_submit_count, static_cast<double>(loc_submit_count_v));
        REPORT_METRICS(cache_reclaimer, block_del_count, static_cast<double>(blk_del_count_v));
        REPORT_METRICS(cache_reclaimer, location_del_count, static_cast<double>(loc_del_count_v));
        REPORT_METRICS(cache_reclaimer, credit_timeout_count, static_cast<double>(credit_timeout_count_v));
        REPORT_METRICS(cache_reclaimer, pending_limit_reject_count, static_cast<double>(pending_limit_reject_count_v));
        REPORT_METRICS(cache_reclaimer,
                       duplicate_pending_location_filtered_count,
                       static_cast<double>(duplicate_pending_location_filtered_count_v));
        REPORT_METRICS(cache_reclaimer,
                       reclaim_no_progress_backoff_count,
                       static_cast<double>(reclaim_no_progress_backoff_count_v));
        REPORT_METRICS(cache_reclaimer, delete_submit_count, static_cast<double>(delete_submit_count_v));
        REPORT_METRICS(cache_reclaimer, delete_complete_count, static_cast<double>(delete_complete_count_v));
        REPORT_METRICS(cache_reclaimer, delete_fail_count, static_cast<double>(delete_fail_count_v));

        REPORT_METRICS(cache_reclaimer, reclaim_cron_duration_us, reclaim_cron_duration_us_v);
        REPORT_METRICS(cache_reclaimer, reclaim_quota_duration_us, reclaim_quota_duration_us_v);
        REPORT_METRICS(cache_reclaimer, reclaim_job_duration_us, reclaim_job_duration_us_v);
        REPORT_METRICS(cache_reclaimer, reclaim_res_duration_us, reclaim_res_duration_us_v);
        REPORT_METRICS(cache_reclaimer, reclaim_lru_sample_duration_us, reclaim_lru_sample_duration_us_v);
        REPORT_METRICS(cache_reclaimer, reclaim_lru_batch_duration_us, reclaim_lru_batch_duration_us_v);
        REPORT_METRICS(cache_reclaimer, reclaim_lru_filter_duration_us, reclaim_lru_filter_duration_us_v);
        REPORT_METRICS(cache_reclaimer, reclaim_lru_submit_duration_us, reclaim_lru_submit_duration_us_v);
        REPORT_METRICS(cache_reclaimer, pending_delete_handler_count, pending_delete_handler_count_v);
        REPORT_METRICS(cache_reclaimer, pending_location_count, pending_location_count_v);
        REPORT_METRICS(cache_reclaimer, pending_delete_bytes, pending_delete_bytes_v);
        REPORT_METRICS(cache_reclaimer, credited_delete_bytes, credited_delete_bytes_v);
        REPORT_METRICS(cache_reclaimer, predicted_deleted_key_count, predicted_deleted_key_count_v);
        REPORT_METRICS(cache_reclaimer, oldest_pending_request_age_ms, oldest_pending_request_age_ms_v);

        REPORT_METRICS(cache_reclaimer, reclaim_batch_lru_age_min_us, reclaim_batch_lru_age_min_us_v);
        REPORT_METRICS(cache_reclaimer, reclaim_batch_lru_age_max_us, reclaim_batch_lru_age_max_us_v);
        REPORT_METRICS(cache_reclaimer, reclaim_batch_lru_age_avg_us, reclaim_batch_lru_age_avg_us_v);
        REPORT_METRICS(cache_reclaimer, reclaim_batch_create_age_min_us, reclaim_batch_create_age_min_us_v);
        REPORT_METRICS(cache_reclaimer, reclaim_batch_create_age_max_us, reclaim_batch_create_age_max_us_v);
        REPORT_METRICS(cache_reclaimer, reclaim_batch_create_age_avg_us, reclaim_batch_create_age_avg_us_v);
    } while (false);

    do {
        // Cache GC owns both untagged metrics and bounded tagged families
        // (candidate reason, delete status and error stage). Read the registry
        // snapshot so KMonitor preserves the same labels as Prometheus/local
        // reporting instead of maintaining a second hard-coded tag list.
        if (!metrics_registry_) {
            break;
        }
        const auto report_registry_metric = [this](kmonitor::MutableMetric *metric, const std::string &name) {
            const auto data = metrics_registry_->GetMetricsData(name);
            if (!metric || !data) {
                return;
            }
            for (const auto &[base_tags, value] : data->GetMetricsValues()) {
                if (!value || !value->touched.load(std::memory_order_relaxed)) {
                    continue;
                }
                double sample = 0.0;
                if (std::holds_alternative<CounterValue>(value->value)) {
                    sample = static_cast<double>(std::get<CounterValue>(value->value).load(std::memory_order_relaxed));
                } else {
                    sample = std::get<GaugeValue>(value->value).load(std::memory_order_relaxed);
                }
                const kmonitor::MetricsTags tags = ctx_->GetKmonitorTags(base_tags);
                metric->Report(&tags, sample);
            }
        };

        report_registry_metric(ctx_->cache_gc_scan_round_count_metrics.get(), "cache_gc.scan_round_count");
        report_registry_metric(ctx_->cache_gc_scan_key_count_metrics.get(), "cache_gc.scan_key_count");
        report_registry_metric(ctx_->cache_gc_candidate_count_metrics.get(), "cache_gc.candidate_count");
        report_registry_metric(ctx_->cache_gc_delete_target_count_metrics.get(), "cache_gc.delete_target_count");
        report_registry_metric(ctx_->cache_gc_delete_result_count_metrics.get(), "cache_gc.delete_result_count");
        report_registry_metric(ctx_->cache_gc_operation_error_count_metrics.get(), "cache_gc.operation_error_count");
        report_registry_metric(ctx_->cache_gc_inflight_delete_count_metrics.get(), "cache_gc.inflight_delete_count");
        report_registry_metric(ctx_->cache_gc_inflight_delete_age_ms_metrics.get(), "cache_gc.inflight_delete_age_ms");
        report_registry_metric(ctx_->cache_gc_round_duration_ms_metrics.get(), "cache_gc.round_duration_ms");
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
                    double key_count_v, byte_size_v, max_lru_age_us_v;
                    GET_METRICS_(p, cache_manager_instance, key_count, key_count_v);
                    REPORT_METRICS(cache_manager_instance, key_count, key_count_v);
                    GET_METRICS_(p, cache_manager_instance, byte_size, byte_size_v);
                    REPORT_METRICS(cache_manager_instance, byte_size, byte_size_v);
                    double async_queue_max_size_v, async_queue_avg_size_v;
                    GET_METRICS_(p, cache_manager_instance, async_queue_max_size, async_queue_max_size_v);
                    REPORT_METRICS(cache_manager_instance, async_queue_max_size, async_queue_max_size_v);
                    GET_METRICS_(p, cache_manager_instance, async_queue_avg_size, async_queue_avg_size_v);
                    REPORT_METRICS(cache_manager_instance, async_queue_avg_size, async_queue_avg_size_v);
                    double async_flush_key_count_v;
                    double async_batch_flush_time_us_v, async_pipeline_error_count_v;
                    GET_METRICS_(p, cache_manager_instance, async_flush_key_count, async_flush_key_count_v);
                    REPORT_METRICS(cache_manager_instance, async_flush_key_count, async_flush_key_count_v);
                    GET_METRICS_(p, cache_manager_instance, async_batch_flush_time_us, async_batch_flush_time_us_v);
                    REPORT_METRICS(cache_manager_instance, async_batch_flush_time_us, async_batch_flush_time_us_v);
                    GET_METRICS_(p, cache_manager_instance, async_pipeline_error_count, async_pipeline_error_count_v);
                    REPORT_METRICS(cache_manager_instance, async_pipeline_error_count, async_pipeline_error_count_v);
                    GET_METRICS_(p, cache_manager_instance, max_lru_age_us, max_lru_age_us_v);
                    REPORT_METRICS(cache_manager_instance, max_lru_age_us, max_lru_age_us_v);
                }
            }
        }
    } while (false);
}

void KmonitorMetricsReporter::VisitMetricSamples(const char *name, const MetricSampleConsumer &consumer) const {
    if (!metrics_registry_ || !consumer) {
        return;
    }
    const auto data = metrics_registry_->GetMetricsData(name);
    if (data == nullptr) {
        return;
    }
    for (const auto &[tags, value] : data->GetMetricsValues()) {
        if (!value || !value->touched.load(std::memory_order_relaxed)) {
            continue;
        }
        if (std::holds_alternative<CounterValue>(value->value)) {
            consumer(tags,
                     static_cast<double>(std::get<CounterValue>(value->value).load(std::memory_order_relaxed)));
        } else {
            consumer(tags, std::get<GaugeValue>(value->value).load(std::memory_order_relaxed));
        }
    }
}

} // namespace kv_cache_manager
