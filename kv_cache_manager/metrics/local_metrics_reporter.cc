#include "kv_cache_manager/metrics/local_metrics_reporter.h"

#include <cstddef>
#include <cstdint>
#include <memory>
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
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

bool LocalMetricsReporter::Init(std::shared_ptr<CacheManager> cache_manager,
                                std::shared_ptr<MetricsRegistry> metrics_registry,
                                const std::string &config) {
    if (cache_manager == nullptr || metrics_registry == nullptr) {
        return false;
    }

    cache_manager_ = std::move(cache_manager);
    metrics_registry_ = std::move(metrics_registry);

    KVCM_MAKE_METRICS_COLLECTOR_(metrics_registry_, MetaIndexerAccumulative, MetaIndexerAccumulative, MetricsTags{});
    KVCM_MAKE_METRICS_COLLECTOR_(metrics_registry_, CacheManager, CacheManager, MetricsTags{});

    return true;
}

void LocalMetricsReporter::ReportPerQuery(MetricsCollector *collector) {
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

            // first get the error_code value
            double service_error_code_v;
            GET_METRICS_(p, service, error_code, service_error_code_v);
            if (CommonUtil::IsZeroDouble(service_error_code_v)) {
                // no error, no report
                break;
            }

            // report
            Counter service_error_counter;
            COPY_METRICS_(p, service, error_counter, service_error_counter);
            ++service_error_counter;
        } while (false);
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

        data_storage_interval_metrics_collectors_.Reset();
        const auto storage_names = data_storage_manager->GetAllStorageNames();
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
            SET_METRICS_(
                p, data_storage, storage_usage_ratio, storage_backend->GetStorageUsageRatio("trace_report_interval"));
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
                total_cache_usage_v += indexer->GetCacheUsage();
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
                }
            }
        }

    } while (false);
}

std::shared_ptr<MetricsRegistry> LocalMetricsReporter::GetMetricsRegistry() { return metrics_registry_; }

} // namespace kv_cache_manager
