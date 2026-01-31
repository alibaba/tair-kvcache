#include "kv_cache_manager/metrics/metrics_collector.h"

#include <memory>
#include <type_traits>
#include <utility>

#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

MetricsCollector::MetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) noexcept
    : metrics_registry_(std::move(metrics_registry)) {}

MetricsCollector::MetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry, MetricsTags metrics_tags) noexcept
    : metrics_registry_(std::move(metrics_registry)), metrics_tags_(std::move(metrics_tags)) {}

const MetricsTags &MetricsCollector::GetMetricsTags() const noexcept { return metrics_tags_; }

void MetricsCollectors::Reset() noexcept { metrics_collectors_ = std::make_unique<mc_vec_t>(); }

void MetricsCollectors::Init() noexcept { this->Reset(); }

void MetricsCollectors::AddMetricsCollector(std::shared_ptr<MetricsCollector> metrics_collector) const noexcept {
    // it is the caller's responsibility to properly init the metrics
    // collector before it is added
    if (metrics_collectors_ != nullptr) {
        metrics_collectors_->emplace_back(std::move(metrics_collector));
    }
}

MetricsCollectors::mc_vec_t MetricsCollectors::GetMetricsCollectors() const noexcept {
    if (metrics_collectors_ == nullptr) {
        return mc_vec_t{};
    }
    return *metrics_collectors_;
}

/* ------------------- ServiceMetricsCollector ---------------------- */

// service metrics
#define DEFINE_METRICS_NAME_FOR_SERVICE(name) DEFINE_METRICS_NAME_(ServiceMetricsCollector, service, name)
#define REGISTER_COUNTER_METRICS_FOR_SERVICE(name)                                                                     \
    REGISTER_METRICS_W_TAGS_COUNTER_(metrics_registry_, service, name, metrics_tags_)
#define REGISTER_GAUGE_METRICS_FOR_SERVICE(name)                                                                       \
    REGISTER_METRICS_W_TAGS_GAUGE_(metrics_registry_, service, name, metrics_tags_)

DEFINE_METRICS_NAME_FOR_SERVICE(query_counter);
DEFINE_METRICS_NAME_FOR_SERVICE(query_rt_us);
DEFINE_METRICS_NAME_FOR_SERVICE(error_code);
DEFINE_METRICS_NAME_FOR_SERVICE(error_counter);
DEFINE_METRICS_NAME_FOR_SERVICE(request_queue_size);

// manager metrics
#define DEFINE_METRICS_NAME_FOR_MANAGER(name) DEFINE_METRICS_NAME_(ServiceMetricsCollector, manager, name)
#define REGISTER_GAUGE_METRICS_FOR_MANAGER(name)                                                                       \
    REGISTER_METRICS_W_TAGS_GAUGE_(metrics_registry_, manager, name, metrics_tags_)

DEFINE_METRICS_NAME_FOR_MANAGER(request_key_count);
DEFINE_METRICS_NAME_FOR_MANAGER(prefix_match_len);
DEFINE_METRICS_NAME_FOR_MANAGER(prefix_match_time_us);
DEFINE_METRICS_NAME_FOR_MANAGER(lock_write_location_retry_times);
DEFINE_METRICS_NAME_FOR_MANAGER(write_cache_io_cost_us);
DEFINE_METRICS_NAME_FOR_MANAGER(filter_write_cache_time_us);
DEFINE_METRICS_NAME_FOR_MANAGER(gen_write_location_us);
DEFINE_METRICS_NAME_FOR_MANAGER(put_write_location_manager_us);
DEFINE_METRICS_NAME_FOR_MANAGER(batch_get_location_time_us);
DEFINE_METRICS_NAME_FOR_MANAGER(batch_add_location_time_us);
DEFINE_METRICS_NAME_FOR_MANAGER(batch_update_location_time_us);

// meta searcher metrics
#define DEFINE_METRICS_NAME_FOR_META_SEARCHER(name) DEFINE_METRICS_NAME_(ServiceMetricsCollector, meta_searcher, name)
#define REGISTER_GAUGE_METRICS_FOR_META_SEARCHER(name)                                                                 \
    REGISTER_METRICS_W_TAGS_GAUGE_(metrics_registry_, meta_searcher, name, metrics_tags_)

DEFINE_METRICS_NAME_FOR_META_SEARCHER(indexer_get_time_us);
DEFINE_METRICS_NAME_FOR_META_SEARCHER(indexer_read_modify_write_time_us);
DEFINE_METRICS_NAME_FOR_META_SEARCHER(index_serialize_time_us);
DEFINE_METRICS_NAME_FOR_META_SEARCHER(index_deserialize_time_us);
DEFINE_METRICS_NAME_FOR_META_SEARCHER(indexer_query_times);

// meta indexer metrics
#define DEFINE_METRICS_NAME_FOR_META_INDEXER(name) DEFINE_METRICS_NAME_(ServiceMetricsCollector, meta_indexer, name)
#define REGISTER_GAUGE_METRICS_FOR_META_INDEXER(name)                                                                  \
    REGISTER_METRICS_W_TAGS_GAUGE_(metrics_registry_, meta_indexer, name, metrics_tags_)

DEFINE_METRICS_NAME_FOR_META_INDEXER(query_key_count);
DEFINE_METRICS_NAME_FOR_META_INDEXER(get_not_exist_key_count);
DEFINE_METRICS_NAME_FOR_META_INDEXER(query_batch_num);
DEFINE_METRICS_NAME_FOR_META_INDEXER(search_cache_hit_count);
DEFINE_METRICS_NAME_FOR_META_INDEXER(search_cache_miss_count);
DEFINE_METRICS_NAME_FOR_META_INDEXER(search_cache_hit_ratio);
DEFINE_METRICS_NAME_FOR_META_INDEXER(io_data_size);
DEFINE_METRICS_NAME_FOR_META_INDEXER(put_io_time_us);
DEFINE_METRICS_NAME_FOR_META_INDEXER(update_io_time_us);
DEFINE_METRICS_NAME_FOR_META_INDEXER(upsert_io_time_us);
DEFINE_METRICS_NAME_FOR_META_INDEXER(delete_io_time_us);
DEFINE_METRICS_NAME_FOR_META_INDEXER(get_io_time_us);
DEFINE_METRICS_NAME_FOR_META_INDEXER(read_modify_write_put_key_count);
DEFINE_METRICS_NAME_FOR_META_INDEXER(read_modify_write_update_key_count);
DEFINE_METRICS_NAME_FOR_META_INDEXER(read_modify_write_skip_key_count);
DEFINE_METRICS_NAME_FOR_META_INDEXER(read_modify_write_delete_key_count);

ServiceMetricsCollector::ServiceMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) noexcept
    : MetricsCollector(std::move(metrics_registry)) {}

ServiceMetricsCollector::ServiceMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry,
                                                 MetricsTags metrics_tags) noexcept
    : MetricsCollector(std::move(metrics_registry), std::move(metrics_tags)) {}

bool ServiceMetricsCollector::Init() {
    if (metrics_registry_ == nullptr) {
        return false;
    }

    // service metrics
    REGISTER_COUNTER_METRICS_FOR_SERVICE(query_counter);
    REGISTER_GAUGE_METRICS_FOR_SERVICE(query_rt_us);
    REGISTER_GAUGE_METRICS_FOR_SERVICE(error_code);
    REGISTER_COUNTER_METRICS_FOR_SERVICE(error_counter);
    REGISTER_GAUGE_METRICS_FOR_SERVICE(request_queue_size);

    // manager metrics
    REGISTER_GAUGE_METRICS_FOR_MANAGER(request_key_count);
    REGISTER_GAUGE_METRICS_FOR_MANAGER(prefix_match_len);
    REGISTER_GAUGE_METRICS_FOR_MANAGER(prefix_match_time_us);
    REGISTER_GAUGE_METRICS_FOR_MANAGER(lock_write_location_retry_times);
    REGISTER_GAUGE_METRICS_FOR_MANAGER(write_cache_io_cost_us);
    REGISTER_GAUGE_METRICS_FOR_MANAGER(filter_write_cache_time_us);
    REGISTER_GAUGE_METRICS_FOR_MANAGER(gen_write_location_us);
    REGISTER_GAUGE_METRICS_FOR_MANAGER(put_write_location_manager_us);
    REGISTER_GAUGE_METRICS_FOR_MANAGER(batch_get_location_time_us);
    REGISTER_GAUGE_METRICS_FOR_MANAGER(batch_add_location_time_us);
    REGISTER_GAUGE_METRICS_FOR_MANAGER(batch_update_location_time_us);

    // meta searcher metrics
    REGISTER_GAUGE_METRICS_FOR_META_SEARCHER(indexer_get_time_us);
    REGISTER_GAUGE_METRICS_FOR_META_SEARCHER(indexer_read_modify_write_time_us);
    REGISTER_GAUGE_METRICS_FOR_META_SEARCHER(index_serialize_time_us);
    REGISTER_GAUGE_METRICS_FOR_META_SEARCHER(index_deserialize_time_us);
    REGISTER_GAUGE_METRICS_FOR_META_SEARCHER(indexer_query_times);

    // meta indexer metrics
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(query_key_count);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(get_not_exist_key_count);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(query_batch_num);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(search_cache_hit_count);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(search_cache_miss_count);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(search_cache_hit_ratio);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(io_data_size);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(put_io_time_us);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(update_io_time_us);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(upsert_io_time_us);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(delete_io_time_us);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(get_io_time_us);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(read_modify_write_put_key_count);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(read_modify_write_update_key_count);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(read_modify_write_skip_key_count);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER(read_modify_write_delete_key_count);

    return true;
}

/* ------------------ DataStorageMetricsCollector ------------------- */

#define DEFINE_METRICS_NAME_FOR_DATA_STORAGE(name) DEFINE_METRICS_NAME_(DataStorageMetricsCollector, data_storage, name)
#define REGISTER_COUNTER_METRICS_FOR_DATA_STORAGE(name)                                                                \
    REGISTER_METRICS_W_TAGS_COUNTER_(metrics_registry_, data_storage, name, metrics_tags_)
#define REGISTER_GAUGE_METRICS_FOR_DATA_STORAGE(name)                                                                  \
    REGISTER_METRICS_W_TAGS_GAUGE_(metrics_registry_, data_storage, name, metrics_tags_)

DEFINE_METRICS_NAME_FOR_DATA_STORAGE(create_counter);
DEFINE_METRICS_NAME_FOR_DATA_STORAGE(create_keys_qps);
DEFINE_METRICS_NAME_FOR_DATA_STORAGE(create_keys_counter);
DEFINE_METRICS_NAME_FOR_DATA_STORAGE(create_time_us);

DataStorageMetricsCollector::DataStorageMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) noexcept
    : MetricsCollector(std::move(metrics_registry)) {}

DataStorageMetricsCollector::DataStorageMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry,
                                                         MetricsTags metrics_tags) noexcept
    : MetricsCollector(std::move(metrics_registry), std::move(metrics_tags)) {}

bool DataStorageMetricsCollector::Init() {
    if (metrics_registry_ == nullptr) {
        return false;
    }

    REGISTER_COUNTER_METRICS_FOR_DATA_STORAGE(create_counter);
    REGISTER_GAUGE_METRICS_FOR_DATA_STORAGE(create_keys_qps);
    REGISTER_COUNTER_METRICS_FOR_DATA_STORAGE(create_keys_counter);
    REGISTER_GAUGE_METRICS_FOR_DATA_STORAGE(create_time_us);

    return true;
}

/* --------------- DataStorageIntervalMetricsCollector ---------------- */

#define DEFINE_METRICS_NAME_FOR_DATA_STORAGE_INTERVAL(name)                                                            \
    DEFINE_METRICS_NAME_(DataStorageIntervalMetricsCollector, data_storage, name)

DEFINE_METRICS_NAME_FOR_DATA_STORAGE_INTERVAL(healthy_status);
DEFINE_METRICS_NAME_FOR_DATA_STORAGE_INTERVAL(storage_usage_ratio);

DataStorageIntervalMetricsCollector::DataStorageIntervalMetricsCollector(
    std::shared_ptr<MetricsRegistry> metrics_registry) noexcept
    : MetricsCollector(std::move(metrics_registry)) {}

DataStorageIntervalMetricsCollector::DataStorageIntervalMetricsCollector(
    std::shared_ptr<MetricsRegistry> metrics_registry, MetricsTags metrics_tags) noexcept
    : MetricsCollector(std::move(metrics_registry), std::move(metrics_tags)) {}

bool DataStorageIntervalMetricsCollector::Init() {
    if (metrics_registry_ == nullptr) {
        return false;
    }

    REGISTER_GAUGE_METRICS_FOR_DATA_STORAGE(healthy_status);
    REGISTER_GAUGE_METRICS_FOR_DATA_STORAGE(storage_usage_ratio);

    return true;
}

/* ------------ MetaIndexerAccumulativeMetricsCollector ------------- */

#define DEFINE_METRICS_NAME_FOR_META_INDEXER_ACC(name)                                                                 \
    DEFINE_METRICS_NAME_(MetaIndexerAccumulativeMetricsCollector, meta_indexer, name)
#define REGISTER_GAUGE_METRICS_FOR_META_INDEXER_ACC(name)                                                              \
    REGISTER_METRICS_W_TAGS_GAUGE_(metrics_registry_, meta_indexer, name, metrics_tags_)

DEFINE_METRICS_NAME_FOR_META_INDEXER_ACC(total_key_count);
DEFINE_METRICS_NAME_FOR_META_INDEXER_ACC(total_cache_usage);

MetaIndexerAccumulativeMetricsCollector::MetaIndexerAccumulativeMetricsCollector(
    std::shared_ptr<MetricsRegistry> metrics_registry) noexcept
    : MetricsCollector(std::move(metrics_registry)) {}

MetaIndexerAccumulativeMetricsCollector::MetaIndexerAccumulativeMetricsCollector(
    std::shared_ptr<MetricsRegistry> metrics_registry, MetricsTags metrics_tags) noexcept
    : MetricsCollector(std::move(metrics_registry), std::move(metrics_tags)) {}

bool MetaIndexerAccumulativeMetricsCollector::Init() {
    if (metrics_registry_ == nullptr) {
        return false;
    }

    REGISTER_GAUGE_METRICS_FOR_META_INDEXER_ACC(total_key_count);
    REGISTER_GAUGE_METRICS_FOR_META_INDEXER_ACC(total_cache_usage);

    return true;
}

/* ------------------ CacheManagerMetricsCollector ------------------- */

#define DEFINE_METRICS_NAME_FOR_CACHE_MANAGER(class, group, name) DEFINE_METRICS_NAME_(class, group, name)
#define REGISTER_GAUGE_METRICS_FOR_CACHE_MANAGER(group, name)                                                          \
    REGISTER_METRICS_W_TAGS_GAUGE_(metrics_registry_, group, name, metrics_tags_)

DEFINE_METRICS_NAME_FOR_CACHE_MANAGER(CacheManagerMetricsCollector, cache_manager, write_location_expire_size);

CacheManagerMetricsCollector::CacheManagerMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) noexcept
    : MetricsCollector(std::move(metrics_registry)) {}

CacheManagerMetricsCollector::CacheManagerMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry,
                                                           MetricsTags metrics_tags) noexcept
    : MetricsCollector(std::move(metrics_registry), std::move(metrics_tags)) {}

bool CacheManagerMetricsCollector::Init() {
    if (metrics_registry_ == nullptr) {
        return false;
    }

    REGISTER_GAUGE_METRICS_FOR_CACHE_MANAGER(cache_manager, write_location_expire_size);

    return true;
}

DEFINE_METRICS_NAME_FOR_CACHE_MANAGER(CacheManagerGroupMetricsCollector, cache_manager_group, usage_ratio);

CacheManagerGroupMetricsCollector::CacheManagerGroupMetricsCollector(
    std::shared_ptr<MetricsRegistry> metrics_registry) noexcept
    : MetricsCollector(std::move(metrics_registry)) {}

CacheManagerGroupMetricsCollector::CacheManagerGroupMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry,
                                                                     MetricsTags metrics_tags) noexcept
    : MetricsCollector(std::move(metrics_registry), std::move(metrics_tags)) {}

bool CacheManagerGroupMetricsCollector::Init() {
    if (metrics_registry_ == nullptr) {
        return false;
    }

    REGISTER_GAUGE_METRICS_FOR_CACHE_MANAGER(cache_manager_group, usage_ratio);

    return true;
}

DEFINE_METRICS_NAME_FOR_CACHE_MANAGER(CacheManagerInstanceMetricsCollector, cache_manager_instance, key_count);
DEFINE_METRICS_NAME_FOR_CACHE_MANAGER(CacheManagerInstanceMetricsCollector, cache_manager_instance, byte_size);

CacheManagerInstanceMetricsCollector::CacheManagerInstanceMetricsCollector(
    std::shared_ptr<MetricsRegistry> metrics_registry) noexcept
    : MetricsCollector(std::move(metrics_registry)) {}

CacheManagerInstanceMetricsCollector::CacheManagerInstanceMetricsCollector(
    std::shared_ptr<MetricsRegistry> metrics_registry, MetricsTags metrics_tags) noexcept
    : MetricsCollector(std::move(metrics_registry), std::move(metrics_tags)) {}

bool CacheManagerInstanceMetricsCollector::Init() {
    if (metrics_registry_ == nullptr) {
        return false;
    }

    REGISTER_GAUGE_METRICS_FOR_CACHE_MANAGER(cache_manager_instance, key_count);
    REGISTER_GAUGE_METRICS_FOR_CACHE_MANAGER(cache_manager_instance, byte_size);

    return true;
}

} // namespace kv_cache_manager
