#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

// RAII guard that records a begin timestamp on construction and writes
// (now - begin) into the target Gauge on explicit reset (move-assign) or,
// when auto_finish is true, on destruction.
//
// Two usage modes controlled by auto_finish:
//   - CHRONO_SCOPE (auto_finish=true):  records on destruction (scope guard)
//   - MARK_BEGIN/END (auto_finish=false): records only via MARK_END (move-assign);
//     if MARK_END is never reached (early return), destructor is a no-op.
//
// Stores begin_us_ per-instance (stack / member), eliminating the
// multi-thread race that existed when begin_ lived on a shared collector.
class ChronoScopeGuard {
public:
    ChronoScopeGuard() noexcept = default;

    explicit ChronoScopeGuard(Gauge *g, bool auto_finish = true) noexcept
        : gauge_(g), begin_us_(g ? TimestampUtil::GetCurrentTimeUs() : 0), auto_finish_(auto_finish) {}

    ChronoScopeGuard(const ChronoScopeGuard &) = delete;
    ChronoScopeGuard &operator=(const ChronoScopeGuard &) = delete;

    ChronoScopeGuard(ChronoScopeGuard &&o) noexcept
        : gauge_(o.gauge_), begin_us_(o.begin_us_), auto_finish_(o.auto_finish_) {
        o.gauge_ = nullptr;
    }

    ChronoScopeGuard &operator=(ChronoScopeGuard &&o) noexcept {
        if (this != &o) {
            Finish();
            gauge_ = o.gauge_;
            begin_us_ = o.begin_us_;
            auto_finish_ = o.auto_finish_;
            o.gauge_ = nullptr;
        }
        return *this;
    }

    ~ChronoScopeGuard() {
        if (auto_finish_) {
            Finish();
        }
    }

private:
    void Finish() noexcept {
        if (gauge_) {
            *gauge_ = static_cast<double>(TimestampUtil::GetCurrentTimeUs() - begin_us_);
            gauge_ = nullptr;
        }
    }

    Gauge *gauge_ = nullptr;
    std::int64_t begin_us_ = 0;
    bool auto_finish_ = false;
};

/* ---------------------- Generic Help Macros ----------------------- */

#ifndef KVCM_METRICS_COLLECTOR_
#define KVCM_METRICS_COLLECTOR_(name) metrics_collector_for_##name##_
#endif

#ifndef KVCM_DECLARE_METRICS_COLLECTOR_
#define KVCM_DECLARE_METRICS_COLLECTOR_(name) std::shared_ptr<MetricsCollector> KVCM_METRICS_COLLECTOR_(name)
#endif

#ifndef KVCM_MAKE_METRICS_COLLECTOR_
#define KVCM_MAKE_METRICS_COLLECTOR_(metrics_registry, name, type, tags)                                               \
    do {                                                                                                               \
        KVCM_METRICS_COLLECTOR_(name) = std::make_shared<type##MetricsCollector>(metrics_registry, tags);              \
        KVCM_METRICS_COLLECTOR_(name)->Init();                                                                         \
    } while (0)
#endif

#ifndef KVCM_METRICS_COLLECTOR_CHRONO_SCOPE
#define KVCM_METRICS_COLLECTOR_CHRONO_SCOPE(ptr, method) ((ptr) ? (ptr)->Make##method##Scope() : ChronoScopeGuard{})
#endif

#ifndef KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN
#define KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(ptr, method)                                                          \
    auto kvcm_chrono_scope_##method##_ = ((ptr) ? (ptr)->Make##method##Scope(false) : ChronoScopeGuard{})
#endif

#ifndef KVCM_METRICS_COLLECTOR_CHRONO_MARK_END
#define KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(ptr, method)                                                            \
    kvcm_chrono_scope_##method##_ = ChronoScopeGuard {}
#endif

#ifndef KVCM_METRICS_COLLECTOR_SET_METRICS
#define KVCM_METRICS_COLLECTOR_SET_METRICS(ptr, group, name, value)                                                    \
    do {                                                                                                               \
        if (!(ptr)) {                                                                                                  \
            break;                                                                                                     \
        }                                                                                                              \
        SET_METRICS_(ptr, group, name, value);                                                                         \
    } while (0)
#endif

#ifndef KVCM_METRICS_COLLECTOR_GET_METRICS
#define KVCM_METRICS_COLLECTOR_GET_METRICS(ptr, group, name, value)                                                    \
    do {                                                                                                               \
        if (!(ptr)) {                                                                                                  \
            break;                                                                                                     \
        }                                                                                                              \
        GET_METRICS_(ptr, group, name, value);                                                                         \
    } while (0)
#endif

class MetricsCollector {
public:
    MetricsCollector() = delete;
    virtual ~MetricsCollector() = default;

    virtual bool Init() = 0;

    [[nodiscard]] const MetricsTags &GetMetricsTags() const noexcept;

protected:
    explicit MetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) noexcept;
    MetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry, MetricsTags metrics_tags) noexcept;

    const std::shared_ptr<MetricsRegistry> metrics_registry_;
    const MetricsTags metrics_tags_;
};

class DummyMetricsCollector final : public MetricsCollector {
public:
    DummyMetricsCollector() : MetricsCollector(nullptr) {}
    bool Init() override { return true; }
};

// dynamic management for metrics collector
class MetricsCollectors {
public:
    using mc_vec_t = std::vector<std::shared_ptr<MetricsCollector>>;

    void Init() noexcept;
    void Reset() noexcept;

    void AddMetricsCollector(std::shared_ptr<MetricsCollector> metrics_collector) const noexcept;

    template <typename T>
    [[nodiscard]] std::shared_ptr<T>
    EmplaceMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) const noexcept;

    template <typename T>
    [[nodiscard]] std::shared_ptr<T> EmplaceMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry,
                                                             MetricsTags metrics_tags) const noexcept;

    [[nodiscard]] mc_vec_t GetMetricsCollectors() const noexcept;

private:
    std::unique_ptr<mc_vec_t> metrics_collectors_;
};

template <typename T>
[[nodiscard]] std::shared_ptr<T>
MetricsCollectors::EmplaceMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) const noexcept {
    if (metrics_collectors_ == nullptr) {
        return nullptr;
    }

    if (!std::is_base_of<MetricsCollector, T>::value) {
        return nullptr;
    }

    auto mc = std::make_shared<T>(std::move(metrics_registry));
    if (!mc->Init()) {
        return nullptr;
    }

    metrics_collectors_->emplace_back(mc);
    return mc;
}

template <typename T>
[[nodiscard]] std::shared_ptr<T>
MetricsCollectors::EmplaceMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry,
                                           MetricsTags metrics_tags) const noexcept {
    if (metrics_collectors_ == nullptr) {
        return nullptr;
    }

    if (!std::is_base_of<MetricsCollector, T>::value) {
        return nullptr;
    }

    auto mc = std::make_shared<T>(std::move(metrics_registry), std::move(metrics_tags));
    if (!mc->Init()) {
        return nullptr;
    }

    metrics_collectors_->emplace_back(mc);
    return mc;
}

/* ------------- Help Macros For Collector Definition --------------- */

#ifndef KVCM_COUNTER_METRICS
#define KVCM_COUNTER_METRICS(group, name)                                                                              \
public:                                                                                                                \
    DECLARE_METRICS_NAME_(group, name);                                                                                \
    DEFINE_COPY_METRICS_COUNTER_(group, name)                                                                          \
    DEFINE_SET_METRICS_COUNTER_(group, name)                                                                           \
    DEFINE_GET_METRICS_COUNTER_(group, name)                                                                           \
                                                                                                                       \
private:                                                                                                               \
    DECLARE_METRICS_COUNTER_(group, name);
#endif

#ifndef KVCM_GAUGE_METRICS
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
#endif

#ifndef KVCM_CHRONO_METRICS
#define KVCM_CHRONO_METRICS(group, name, method)                                                                       \
    KVCM_GAUGE_METRICS(group, name)                                                                                    \
public:                                                                                                                \
    ChronoScopeGuard Make##method##Scope(bool auto_finish = true) {                                                    \
        return ChronoScopeGuard(&METRICS_(group, name), auto_finish);                                                  \
    }
#endif


/* ------------------- ServiceMetricsCollector ---------------------- */

#ifndef KVCM_SERVICE_METRICS_COLLECTOR_PTR
#define KVCM_SERVICE_METRICS_COLLECTOR_PTR(name)                                                                       \
    std::dynamic_pointer_cast<ServiceMetricsCollector>(KVCM_METRICS_COLLECTOR_(name))
#endif

class ServiceMetricsCollector final : public MetricsCollector {
    // service metrics
    // no need for gauge metrics (service, qps) since the value always be 1.0
    KVCM_COUNTER_METRICS(service, query_counter) // for local metrics registry
    KVCM_CHRONO_METRICS(service, query_rt_us, ServiceQuery)
    KVCM_GAUGE_METRICS(service, error_code)
    // no need for gauge metrics (service, error_qps) since the value always be 1.0
    KVCM_COUNTER_METRICS(service, error_counter) // for local metrics registry
    KVCM_GAUGE_METRICS(service, request_queue_size)

    // manager metrics
    KVCM_GAUGE_METRICS(manager, request_key_count)
    KVCM_GAUGE_METRICS(manager, prefix_match_len)
    KVCM_CHRONO_METRICS(manager, prefix_match_time_us, ManagerPrefixMatch)
    KVCM_GAUGE_METRICS(manager, lock_write_location_retry_times)
    KVCM_GAUGE_METRICS(manager, write_cache_io_cost_us)
    KVCM_CHRONO_METRICS(manager, filter_write_cache_time_us, ManagerFilterWriteCache)
    KVCM_CHRONO_METRICS(manager, gen_write_location_us, GenWriteLocation)
    KVCM_CHRONO_METRICS(manager, put_write_location_manager_us, PutWriteLocationManager)
    KVCM_CHRONO_METRICS(manager, batch_get_location_time_us, ManagerBatchGetLocation)
    KVCM_CHRONO_METRICS(manager, batch_add_location_time_us, ManagerBatchAddLocation)
    KVCM_CHRONO_METRICS(manager, batch_update_location_time_us, ManagerBatchUpdateLocation)

    // meta searcher metrics
    KVCM_CHRONO_METRICS(meta_searcher, indexer_get_time_us, MetaSearcherIndexerGet)
    KVCM_CHRONO_METRICS(meta_searcher, indexer_read_modify_write_block_time_us, MetaSearcherIndexerReadModifyWriteBlock)
    KVCM_CHRONO_METRICS(meta_searcher,
                        indexer_read_modify_write_location_time_us,
                        MetaSearcherIndexerReadModifyWriteLocation)
    KVCM_GAUGE_METRICS(meta_searcher, index_serialize_time_us)
    KVCM_GAUGE_METRICS(meta_searcher, index_deserialize_time_us)
    KVCM_GAUGE_METRICS(meta_searcher, indexer_query_times)

    // meta indexer metrics
    KVCM_GAUGE_METRICS(meta_indexer, query_key_count)
    KVCM_GAUGE_METRICS(meta_indexer, get_not_exist_key_count)
    KVCM_GAUGE_METRICS(meta_indexer, query_batch_num)
    KVCM_GAUGE_METRICS(meta_indexer, search_cache_hit_count)
    KVCM_GAUGE_METRICS(meta_indexer, search_cache_miss_count)
    KVCM_GAUGE_METRICS(meta_indexer, search_cache_hit_ratio)
    KVCM_GAUGE_METRICS(meta_indexer, io_data_size)
    KVCM_GAUGE_METRICS(meta_indexer, put_io_time_us)
    KVCM_GAUGE_METRICS(meta_indexer, upsert_io_time_us)
    KVCM_GAUGE_METRICS(meta_indexer, lock_wait_time_us)
    KVCM_GAUGE_METRICS(meta_indexer, delete_io_time_us)
    KVCM_GAUGE_METRICS(meta_indexer, get_io_time_us)
    KVCM_GAUGE_METRICS(meta_indexer, rand_io_time_us)
    KVCM_GAUGE_METRICS(meta_indexer, rmw_get_io_time_us)
    KVCM_GAUGE_METRICS(meta_indexer, read_modify_write_put_key_count)
    KVCM_GAUGE_METRICS(meta_indexer, read_modify_write_update_key_count)
    KVCM_GAUGE_METRICS(meta_indexer, read_modify_write_skip_key_count)
    KVCM_GAUGE_METRICS(meta_indexer, read_modify_write_delete_key_count)
    KVCM_GAUGE_METRICS(meta_indexer, async_enqueue_timeout_key_count)
    KVCM_GAUGE_METRICS(meta_indexer, async_enqueue_time_us)
    KVCM_GAUGE_METRICS(meta_indexer, cache_backend_put_time_us)
    KVCM_GAUGE_METRICS(meta_indexer, cache_backend_upsert_time_us)
    KVCM_GAUGE_METRICS(meta_indexer, cache_backend_delete_time_us)

public:
    ServiceMetricsCollector() = delete;
    explicit ServiceMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) noexcept;
    ServiceMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry, MetricsTags metrics_tags) noexcept;
    ~ServiceMetricsCollector() override = default;

    bool Init() override;
};

/* ------------------ DataStorageMetricsCollector ------------------- */

#ifndef KVCM_DATA_STORAGE_METRICS_COLLECTOR_PTR
#define KVCM_DATA_STORAGE_METRICS_COLLECTOR_PTR(name)                                                                  \
    std::dynamic_pointer_cast<DataStorageMetricsCollector>(KVCM_METRICS_COLLECTOR_(name))
#endif

class DataStorageMetricsCollector final : public MetricsCollector {
    // no need for gauge metrics (data_storage, create_qps) since the value always be 1.0
    KVCM_COUNTER_METRICS(data_storage, create_counter) // for local metrics registry

    KVCM_GAUGE_METRICS(data_storage, create_keys_qps)       // this is necessary since its value may not be 1.0
    KVCM_COUNTER_METRICS(data_storage, create_keys_counter) // for local metrics registry

    KVCM_CHRONO_METRICS(data_storage, create_time_us, DataStorageCreate)

public:
    DataStorageMetricsCollector() = delete;
    explicit DataStorageMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) noexcept;
    DataStorageMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry, MetricsTags metrics_tags) noexcept;
    ~DataStorageMetricsCollector() override = default;

    bool Init() override;
};

/* --------------- DataStorageIntervalMetricsCollector ---------------- */

#ifndef KVCM_DATA_STORAGE_INTERVAL_METRICS_COLLECTOR_PTR
#define KVCM_DATA_STORAGE_INTERVAL_METRICS_COLLECTOR_PTR(name)                                                         \
    std::dynamic_pointer_cast<DataStorageIntervalMetricsCollector>(KVCM_METRICS_COLLECTOR_(name))
#endif

class DataStorageIntervalMetricsCollector final : public MetricsCollector {
    KVCM_GAUGE_METRICS(data_storage, healthy_status)
    KVCM_GAUGE_METRICS(data_storage, storage_usage_ratio)

public:
    DataStorageIntervalMetricsCollector() = delete;
    explicit DataStorageIntervalMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) noexcept;
    DataStorageIntervalMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry,
                                        MetricsTags metrics_tags) noexcept;
    ~DataStorageIntervalMetricsCollector() override = default;

    bool Init() override;
};

/* ------------ MetaIndexerAccumulativeMetricsCollector ------------- */

#ifndef KVCM_META_INDEXER_ACC_METRICS_COLLECTOR_PTR
#define KVCM_META_INDEXER_ACC_METRICS_COLLECTOR_PTR(name)                                                              \
    std::dynamic_pointer_cast<MetaIndexerAccumulativeMetricsCollector>(KVCM_METRICS_COLLECTOR_(name))
#endif

class MetaIndexerAccumulativeMetricsCollector final : public MetricsCollector {
    KVCM_GAUGE_METRICS(meta_indexer, total_key_count)
    KVCM_GAUGE_METRICS(meta_indexer, total_cache_usage)

public:
    MetaIndexerAccumulativeMetricsCollector() = delete;
    explicit MetaIndexerAccumulativeMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) noexcept;
    MetaIndexerAccumulativeMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry,
                                            MetricsTags metrics_tags) noexcept;
    ~MetaIndexerAccumulativeMetricsCollector() override = default;

    bool Init() override;
};

/* --------------- CacheManagerMetricsCollector ---------------- */

#ifndef KVCM_CACHE_MANAGER_METRICS_COLLECTOR_PTR
#define KVCM_CACHE_MANAGER_METRICS_COLLECTOR_PTR(name)                                                                 \
    std::dynamic_pointer_cast<CacheManagerMetricsCollector>(KVCM_METRICS_COLLECTOR_(name))
#endif

class CacheManagerMetricsCollector final : public MetricsCollector {
    KVCM_GAUGE_METRICS(cache_manager, write_location_expire_size)

public:
    CacheManagerMetricsCollector() = delete;
    explicit CacheManagerMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) noexcept;
    CacheManagerMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry, MetricsTags metrics_tags) noexcept;
    ~CacheManagerMetricsCollector() override = default;

    bool Init() override;
};

class CacheManagerGroupMetricsCollector final : public MetricsCollector {
    KVCM_GAUGE_METRICS(cache_manager_group, usage_ratio)

public:
    CacheManagerGroupMetricsCollector() = delete;
    explicit CacheManagerGroupMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) noexcept;
    CacheManagerGroupMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry,
                                      MetricsTags metrics_tags) noexcept;
    ~CacheManagerGroupMetricsCollector() override = default;

    bool Init() override;
};

class CacheManagerInstanceMetricsCollector final : public MetricsCollector {
    KVCM_GAUGE_METRICS(cache_manager_instance, key_count)
    KVCM_GAUGE_METRICS(cache_manager_instance, byte_size)
    KVCM_GAUGE_METRICS(cache_manager_instance, max_lru_age_us)
    KVCM_GAUGE_METRICS(cache_manager_instance, async_queue_max_size)
    KVCM_GAUGE_METRICS(cache_manager_instance, async_queue_avg_size)
    KVCM_GAUGE_METRICS(cache_manager_instance, async_flush_key_count)
    KVCM_GAUGE_METRICS(cache_manager_instance, async_batch_flush_time_us)
    KVCM_GAUGE_METRICS(cache_manager_instance, async_pipeline_error_count)

public:
    CacheManagerInstanceMetricsCollector() = delete;
    explicit CacheManagerInstanceMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry) noexcept;
    CacheManagerInstanceMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry,
                                         MetricsTags metrics_tags) noexcept;
    ~CacheManagerInstanceMetricsCollector() override = default;

    bool Init() override;
};

#undef KVCM_CHRONO_METRICS
#undef KVCM_GAUGE_METRICS

} // namespace kv_cache_manager
