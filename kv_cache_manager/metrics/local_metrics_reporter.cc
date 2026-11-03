#include "kv_cache_manager/metrics/local_metrics_reporter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <utility>

#include "kv_cache_manager/common/common_util.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_manager_metrics_recorder.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/node_status_metrics.h"

namespace kv_cache_manager {

bool LocalMetricsReporter::Init(std::shared_ptr<CacheManager> cache_manager,
                                std::shared_ptr<MetricsRegistry> metrics_registry,
                                const std::string &config,
                                const MetricsTags &common_tags) {
    if (cache_manager == nullptr || metrics_registry == nullptr) {
        return false;
    }

    cache_manager_ = std::move(cache_manager);
    metrics_registry_ = std::move(metrics_registry);
    common_tags_ = common_tags;

    KVCM_MAKE_METRICS_COLLECTOR_(metrics_registry_, MetaIndexerAccumulative, MetaIndexerAccumulative, MetricsTags{});
    KVCM_MAKE_METRICS_COLLECTOR_(metrics_registry_, CacheManager, CacheManager, MetricsTags{});

    return true;
}

void LocalMetricsReporter::PublishNodeStatus(std::optional<bool> is_leader) {
    if (!is_leader.has_value()) {
        return;
    }

    std::shared_ptr<DataStorageManager> data_storage_manager;
    const auto registry_manager = cache_manager_->GetRegistryManager();
    if (registry_manager) {
        data_storage_manager = registry_manager->data_storage_manager();
    }
    NodeStatusMetrics::Publish(metrics_registry_, *is_leader, data_storage_manager);
}

void LocalMetricsReporter::ReportPerQuery(MetricsCollector *collector, std::optional<bool> request_failed) {
    if (dynamic_cast<ServiceMetricsCollector *>(collector)) {
        auto *p = dynamic_cast<ServiceMetricsCollector *>(collector);

        do {
            // service query_counter
            Counter service_query_counter;
            COPY_METRICS_(p, service, query_counter, service_query_counter);
            ++service_query_counter;
        } while (false);

        do {
            // service error_counter

            const bool failed = request_failed.value_or(!CommonUtil::IsZeroDouble(p->get_service_error_code_metrics()));
            if (!failed) {
                // no error, no report
                break;
            }

            // report
            Counter service_error_counter;
            COPY_METRICS_(p, service, error_counter, service_error_counter);
            ++service_error_counter;
        } while (false);
    } else if (dynamic_cast<EventReportMetricsCollector *>(collector)) {
        auto *p = dynamic_cast<EventReportMetricsCollector *>(collector);
        p->UpdateLocationLifetimeMetrics();
        Counter service_query_counter;
        COPY_METRICS_(p, service, query_counter, service_query_counter);
        ++service_query_counter;

        const double error_code = p->GetErrorCodeSample();
        if (!CommonUtil::IsZeroDouble(error_code)) {
            Counter service_error_counter;
            COPY_METRICS_(p, service, error_counter, service_error_counter);
            ++service_error_counter;
        }
    } else if (dynamic_cast<DataStorageMetricsCollector *>(collector)) {
        auto *p = dynamic_cast<DataStorageMetricsCollector *>(collector);

        do {
            // data_storage create_counter
            Counter ds_create_counter;
            COPY_METRICS_(p, data_storage, create_counter, ds_create_counter);
            ++ds_create_counter;
        } while (false);

        do {
            // data_storage create_keys_counter

            // first get the create_keys_qps value
            double ds_create_keys_qps_v;
            GET_METRICS_(p, data_storage, create_keys_qps, ds_create_keys_qps_v);

            // report
            Counter ds_create_keys_counter;
            COPY_METRICS_(p, data_storage, create_keys_counter, ds_create_keys_counter);
            ds_create_keys_counter += static_cast<std::uint64_t>(ds_create_keys_qps_v);
        } while (false);
    }
}

void LocalMetricsReporter::ReportInterval() {
    if (!cache_manager_) {
        return;
    }

    // hold a shared lifecycle lock across the entire metrics publication
    // so concurrent RemoveInstance/RemoveInstanceGroup/RemoveStorage
    // (which hold the lock as a writer) cannot interleave a tag-filter
    // purge with EmplaceMetricsCollector calls below
    std::shared_lock<std::shared_mutex> lifecycle_guard(cache_manager_->metrics_lifecycle()->mut_);

    // Sample the role once so node status and storage usage agree in this report.
    const auto &leader_state_provider = GetLeaderStateProvider();
    const std::optional<bool> is_leader =
        leader_state_provider ? std::optional<bool>(leader_state_provider()) : std::nullopt;
    PublishNodeStatus(is_leader);

    // Demotion cleanup can remove every backend before this interval runs.
    // Clear previously published ratios before consulting the storage manager,
    // otherwise its empty map would leave the old leader's readings exported.
    const bool is_follower = is_leader.has_value() && !*is_leader;
    if (is_follower) {
        const auto ratios = metrics_registry_->GetMetricsData("data_storage.storage_usage_ratio");
        if (ratios) {
            for (const auto &[tags, value] : ratios->GetMetricsValues()) {
                Gauge gauge(value);
                gauge = 0.0;
            }
        }
    }

    do {
        // for data storage metrics
        const auto registry_manager = cache_manager_->GetRegistryManager();
        if (!registry_manager) {
            break;
        }
        const auto data_storage_manager = registry_manager->data_storage_manager();
        if (!data_storage_manager) {
            break;
        }

        const auto storage_names = data_storage_manager->GetAllStorageNames();
        if (is_follower) {
            for (const auto &metric : {"data_storage.healthy_status", "data_storage.writable_status"}) {
                const auto data = metrics_registry_->GetMetricsData(metric);
                if (!data) {
                    continue;
                }
                for (const auto &[tags, value] : data->GetMetricsValues()) {
                    const auto name = tags.find("unique_name");
                    if (name != tags.end() &&
                        std::find(storage_names.begin(), storage_names.end(), name->second) == storage_names.end()) {
                        // Closed and removed backends must not retain health
                        // readings, even after the collector cache was reset.
                        Gauge gauge(value);
                        gauge = 0.0;
                    }
                }
            }
        }
        data_storage_interval_metrics_collectors_.Reset();
        for (const auto &unique_name : storage_names) {
            const auto storage_backend = data_storage_manager->GetDataStorageBackend(unique_name);
            if (storage_backend == nullptr) {
                continue;
            }
            const auto p =
                data_storage_interval_metrics_collectors_.EmplaceMetricsCollector<DataStorageIntervalMetricsCollector>(
                    metrics_registry_,
                    MetricsTags{{"type", ToString(storage_backend->GetType())}, {"unique_name", unique_name}});
            SET_METRICS_(p, data_storage, healthy_status, storage_backend->Available() ? 1.0 : 0.0);
            SET_METRICS_(p, data_storage, writable_status, storage_backend->Writable() ? 1.0 : 0.0);
            // Followers publish zero every interval, clearing any ratio retained
            // from their previous leader tenure. Health and writability above
            // still reflect the backend, including meta-service failures.
            if (is_follower) {
                SET_METRICS_(p, data_storage, storage_usage_ratio, 0.0);
            } else {
                // Preserve backend return semantics, including a default zero.
                // Defensively ignore negative values if a backend returns one.
                const double usage_ratio = storage_backend->GetStorageUsageRatio("trace_report_interval");
                if (usage_ratio >= 0.0) {
                    SET_METRICS_(p, data_storage, storage_usage_ratio, usage_ratio);
                }
            }
        }
    } while (false);

    do {
        // for meta indexer accumulative metrics
        const auto meta_indexer_manager = cache_manager_->meta_indexer_manager();
        if (!meta_indexer_manager) {
            break;
        }

        const auto p = KVCM_META_INDEXER_ACC_METRICS_COLLECTOR_PTR(MetaIndexerAccumulative);
        if (!p) {
            break;
        }

        const auto indexer_map = meta_indexer_manager->GetIndexers();
        std::size_t total_key_count_v = 0;
        std::size_t total_cache_usage_v = 0;
        for (auto &[_, indexer] : indexer_map) {
            if (indexer) {
                total_key_count_v += indexer->GetKeyCount();
                total_cache_usage_v += indexer->GetMemUsage();
            }
        }

        SET_METRICS_(p, meta_indexer, total_key_count, static_cast<double>(total_key_count_v));
        SET_METRICS_(p, meta_indexer, total_cache_usage, static_cast<double>(total_cache_usage_v));
    } while (false);

    do {
        const auto cache_manager_metrics_recorder = cache_manager_->metrics_recorder();
        if (!cache_manager_metrics_recorder) {
            break;
        }

        {
            const auto p = KVCM_CACHE_MANAGER_METRICS_COLLECTOR_PTR(CacheManager);
            if (!p) {
                break;
            }
            size_t expire_size = cache_manager_metrics_recorder->write_location_expire_size();
            SET_METRICS_(p, cache_manager, write_location_expire_size, static_cast<double>(expire_size));
        }

        {
            cache_manager_group_interval_metrics_collectors_.Reset();
            const auto group_usage_ratio_map = cache_manager_metrics_recorder->group_usage_ratio_map();
            for (const auto &[instance_group_name, usage_ratio] : group_usage_ratio_map) {
                const auto p = cache_manager_group_interval_metrics_collectors_
                                   .EmplaceMetricsCollector<CacheManagerGroupMetricsCollector>(
                                       metrics_registry_, MetricsTags{{"instance_group", instance_group_name}});
                SET_METRICS_(p, cache_manager_group, usage_ratio, usage_ratio);
            }
        }

        {
            cache_manager_instance_interval_metrics_collectors_.Reset();
            const auto group_instance_id_metric_map = cache_manager_metrics_recorder->group_instance_id_metric_map();
            for (const auto &[instance_group_name, instance_id_metric_map] : group_instance_id_metric_map) {
                for (const auto &[instance_id, instance_metric] : instance_id_metric_map) {
                    const auto p =
                        cache_manager_instance_interval_metrics_collectors_
                            .EmplaceMetricsCollector<CacheManagerInstanceMetricsCollector>(
                                metrics_registry_,
                                MetricsTags{{"instance_group", instance_group_name}, {"instance_id", instance_id}});
                    SET_METRICS_(p, cache_manager_instance, key_count, static_cast<double>(instance_metric.key_count));
                    SET_METRICS_(p, cache_manager_instance, byte_size, static_cast<double>(instance_metric.byte_size));
                    SET_METRICS_(
                        p, cache_manager_instance, max_lru_age_us, static_cast<double>(instance_metric.max_lru_age_us));
                    SET_METRICS_(p,
                                 cache_manager_instance,
                                 async_queue_max_size,
                                 static_cast<double>(instance_metric.async_queue_max_size));
                    SET_METRICS_(p,
                                 cache_manager_instance,
                                 async_queue_avg_size,
                                 static_cast<double>(instance_metric.async_queue_avg_size));
                    SET_METRICS_(p,
                                 cache_manager_instance,
                                 async_flush_key_count,
                                 static_cast<double>(instance_metric.async_flush_key_count));
                    SET_METRICS_(p,
                                 cache_manager_instance,
                                 async_batch_flush_time_us,
                                 static_cast<double>(instance_metric.async_batch_flush_time_us));
                    SET_METRICS_(p,
                                 cache_manager_instance,
                                 async_pipeline_error_count,
                                 static_cast<double>(instance_metric.async_pipeline_error_count));
                    SET_METRICS_(p,
                                 cache_manager_instance,
                                 async_dropped_key_count,
                                 static_cast<double>(instance_metric.async_dropped_key_count));
                    SET_METRICS_(p,
                                 cache_manager_instance,
                                 async_dropped_metadata_count,
                                 static_cast<double>(instance_metric.async_dropped_metadata_count));
                }
            }
        }

    } while (false);
}

std::shared_ptr<MetricsRegistry> LocalMetricsReporter::GetMetricsRegistry() { return metrics_registry_; }

} // namespace kv_cache_manager
