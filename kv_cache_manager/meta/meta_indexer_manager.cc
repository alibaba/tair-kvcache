#include "kv_cache_manager/meta/meta_indexer_manager.h"

#include <algorithm>
#include <exception>
#include <mutex>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/query_executor.h"
#include "kv_cache_manager/metrics/revisit_interval_histogram.h"

namespace kv_cache_manager {

void MetaIndexerManager::SetRevisitHistogramConfig(std::shared_ptr<MetricsRegistry> registry,
                                                   const std::vector<double> &boundaries) {
    std::unique_lock lifecycle_guard(lifecycle_mutex_);
    metrics_registry_ = std::move(registry);
    revisit_boundaries_ = boundaries;
}

bool MetaIndexerManager::ConfigureQueryExecutor(std::size_t worker_count,
                                                std::size_t parallel_threshold,
                                                std::size_t chunk_size) {
    if (worker_count == 0 || worker_count > 64 || parallel_threshold == 0 || chunk_size == 0 ||
        chunk_size > parallel_threshold) {
        KVCM_LOG_ERROR("invalid query executor config: workers[%zu] threshold[%zu] chunk_size[%zu]",
                       worker_count,
                       parallel_threshold,
                       chunk_size);
        return false;
    }
    std::unique_lock lifecycle_guard(lifecycle_mutex_);
    {
        std::shared_lock read_guard(mutex_);
        if (!meta_indexers_.empty()) {
            KVCM_LOG_ERROR("query executor must be configured before creating meta indexers");
            return false;
        }
    }
    constexpr std::size_t kQueueSlotsPerWorker = 64;
    try {
        query_executor_ =
            std::make_shared<QueryExecutor>(worker_count,
                                            parallel_threshold,
                                            chunk_size,
                                            std::max<std::size_t>(64, worker_count * kQueueSlotsPerWorker));
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("failed to create query executor: %s", e.what());
        query_executor_.reset();
        return false;
    } catch (...) {
        KVCM_LOG_ERROR("failed to create query executor with unknown exception");
        query_executor_.reset();
        return false;
    }
    KVCM_LOG_INFO("configured query executor: workers[%zu] threshold[%zu] chunk_size[%zu]",
                  worker_count,
                  parallel_threshold,
                  chunk_size);
    return true;
}

ErrorCode MetaIndexerManager::CreateMetaIndexer(const std::string &instance_id,
                                                const std::shared_ptr<MetaIndexerConfig> &config,
                                                const std::vector<double> &boundaries) {
    std::shared_lock lifecycle_guard(lifecycle_mutex_);
    auto instance_mutex = GetInstanceMutex(instance_id);
    std::unique_lock instance_guard(*instance_mutex);
    if (GetMetaIndexer(instance_id)) {
        return ErrorCode::EC_EXIST;
    }
    if (!config) {
        return EC_BADARGS;
    }
    auto indexer = std::make_shared<MetaIndexer>();

    // Resolve boundaries: per-instance override > global default
    const std::vector<double> &resolved = boundaries.empty() ? revisit_boundaries_ : boundaries;
    indexer->initial_config_json_ = config->ToJsonString();
    indexer->initial_histogram_boundaries_ = resolved;
    indexer->SetQueryExecutor(query_executor_);
    auto ec = InitMetaIndexer(indexer, instance_id, config);
    if (ec != ErrorCode::EC_OK) {
        KVCM_LOG_ERROR("Init meta indexer failed, instance_id: %s", instance_id.c_str());
        return ec;
    }

    // Create and inject per-instance revisit interval histogram
    if (metrics_registry_ && !resolved.empty()) {
        auto histogram = std::make_shared<RevisitIntervalHistogram>();
        if (histogram->Init(metrics_registry_, resolved, instance_id)) {
            indexer->SetRevisitHistogram(histogram);
            KVCM_LOG_INFO("Created revisit interval histogram for instance_id: %s with %zu boundaries",
                          instance_id.c_str(),
                          resolved.size());
        } else {
            KVCM_LOG_WARN("Failed to create revisit interval histogram for instance_id: %s", instance_id.c_str());
        }
    }

    indexer->manager_initialized_ = true;
    {
        std::scoped_lock write_guard(mutex_);
        if (!meta_indexers_.emplace(instance_id, indexer).second) {
            return EC_EXIST;
        }
    }
    KVCM_LOG_INFO("Create meta indexer success, instance_id: %s", instance_id.c_str());
    return ErrorCode::EC_OK;
}

ErrorCode MetaIndexerManager::InitMetaIndexer(const std::shared_ptr<MetaIndexer> &indexer,
                                              const std::string &instance_id,
                                              const std::shared_ptr<MetaIndexerConfig> &config) {
    return indexer->Init(instance_id, config);
}

std::shared_ptr<MetaIndexer> MetaIndexerManager::GetMetaIndexer(const std::string &instance_id) const {
    std::shared_lock read_guard(mutex_);
    return GetMetaIndexerUnsafe(instance_id);
}

std::shared_ptr<MetaIndexer> MetaIndexerManager::GetMetaIndexerUnsafe(const std::string &instance_id) const {
    auto iter = meta_indexers_.find(instance_id);
    if (iter != meta_indexers_.end()) {
        return iter->second;
    }
    return nullptr;
}

ErrorCode MetaIndexerManager::GetMetaIndexerForReuse(const std::string &instance_id,
                                                     const std::shared_ptr<MetaIndexerConfig> &config,
                                                     const std::vector<double> &boundaries,
                                                     std::shared_ptr<MetaIndexer> &out_indexer) const {
    out_indexer.reset();
    if (!config) {
        return EC_BADARGS;
    }
    std::shared_lock lifecycle_guard(lifecycle_mutex_);
    auto instance_mutex = GetInstanceMutex(instance_id);
    std::unique_lock instance_guard(*instance_mutex);
    auto indexer = GetMetaIndexer(instance_id);
    if (!indexer) {
        return EC_NOENT;
    }
    if (!indexer->manager_initialized_ || !indexer->backend_manager_ || indexer->instance_id_ != instance_id) {
        KVCM_LOG_ERROR("cannot reuse meta indexer, instance[%s]: registration or initialization mismatch",
                       instance_id.c_str());
        return EC_MISMATCH;
    }
    const std::vector<double> &resolved = boundaries.empty() ? revisit_boundaries_ : boundaries;
    if (indexer->initial_config_json_ != config->ToJsonString() || indexer->initial_histogram_boundaries_ != resolved) {
        KVCM_LOG_ERROR("cannot reuse meta indexer, instance[%s]: configuration mismatch", instance_id.c_str());
        return EC_MISMATCH;
    }
    out_indexer = std::move(indexer);
    return EC_OK;
}

std::shared_ptr<std::mutex> MetaIndexerManager::GetInstanceMutex(const std::string &instance_id) const {
    std::scoped_lock write_guard(mutex_);
    for (auto it = instance_mutexes_.begin(); it != instance_mutexes_.end();) {
        if (it->second.expired()) {
            it = instance_mutexes_.erase(it);
        } else {
            ++it;
        }
    }
    auto &entry = instance_mutexes_[instance_id];
    auto instance_mutex = entry.lock();
    if (!instance_mutex) {
        instance_mutex = std::make_shared<std::mutex>();
        entry = instance_mutex;
    }
    return instance_mutex;
}

ErrorCode MetaIndexerManager::DeleteMetaIndexer(const std::string &instance_id) {
    // TODO : delete is dangerous, should carefully design
    return ErrorCode::EC_UNIMPLEMENTED;
    // size_t num = 0;
    // {
    //     std::scoped_lock write_guard(mutex_);
    //     num = meta_indexers_.erase(instance_id);
    // }
    // if (num == 0) {
    //     KVCM_LOG_WARN("Delete meta indexer failed, instance_id: %s", instance_id.c_str());
    //     return ErrorCode::EC_NOENT;
    // }
    // KVCM_LOG_INFO("Delete meta indexer success, instance_id: %s", instance_id.c_str());
    // return ErrorCode::EC_OK;
}

std::map<std::string, std::shared_ptr<MetaIndexer>> MetaIndexerManager::GetIndexers() const {
    std::shared_lock read_guard(mutex_);
    return meta_indexers_;
}
void MetaIndexerManager::DoCleanup() {
    std::unique_lock lifecycle_guard(lifecycle_mutex_);
    decltype(meta_indexers_) indexers;
    {
        std::scoped_lock write_guard(mutex_);
        indexers.swap(meta_indexers_);
        instance_mutexes_.clear();
    }
}

size_t MetaIndexerManager::GetIndexerSize() {
    std::shared_lock read_guard(mutex_);
    return meta_indexers_.size();
}

} // namespace kv_cache_manager
