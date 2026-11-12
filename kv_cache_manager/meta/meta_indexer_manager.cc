#include "kv_cache_manager/meta/meta_indexer_manager.h"

#include <algorithm>
#include <exception>

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
    std::scoped_lock write_guard(mutex_);
    if (!meta_indexers_.empty()) {
        KVCM_LOG_ERROR("query executor must be configured before creating meta indexers");
        return false;
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
    auto indexer = std::make_shared<MetaIndexer>();
    indexer->SetQueryExecutor(query_executor_);
    auto ec = InitMetaIndexer(indexer, instance_id, config);
    if (ec != ErrorCode::EC_OK) {
        KVCM_LOG_ERROR("Init meta indexer failed, instance_id: %s", instance_id.c_str());
        return ec;
    }

    // Per-instance boundaries override defaults protected by lifecycle_mutex_.
    const std::vector<double> &resolved = boundaries.empty() ? revisit_boundaries_ : boundaries;

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
    {
        std::scoped_lock write_guard(mutex_);
        meta_indexers_[instance_id] = indexer;
    }
    KVCM_LOG_INFO("Create meta indexer success, instance_id: %s", instance_id.c_str());
    return ErrorCode::EC_OK;
}

ErrorCode MetaIndexerManager::InitMetaIndexer(const std::shared_ptr<MetaIndexer> &indexer,
                                              const std::string &instance_id,
                                              const std::shared_ptr<MetaIndexerConfig> &config) {
    return indexer->Init(instance_id, config);
}

std::shared_ptr<std::mutex> MetaIndexerManager::GetInstanceMutex(const std::string &instance_id) {
    std::scoped_lock write_guard(mutex_);
    auto &instance_mutex = instance_mutexes_[instance_id];
    if (!instance_mutex) {
        instance_mutex = std::make_shared<std::mutex>();
    }
    return instance_mutex;
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

std::shared_ptr<MetaIndexer> MetaIndexerManager::ExtractMetaIndexer(const std::string &instance_id) {
    std::shared_lock lifecycle_guard(lifecycle_mutex_);
    auto instance_mutex = GetInstanceMutex(instance_id);
    std::unique_lock instance_guard(*instance_mutex);
    decltype(meta_indexers_)::node_type node;
    {
        std::scoped_lock write_guard(mutex_);
        node = meta_indexers_.extract(instance_id);
    }
    return node.empty() ? nullptr : std::move(node.mapped());
}

bool MetaIndexerManager::RestoreMetaIndexer(const std::string &instance_id, std::shared_ptr<MetaIndexer> indexer) {
    std::shared_lock lifecycle_guard(lifecycle_mutex_);
    auto instance_mutex = GetInstanceMutex(instance_id);
    std::unique_lock instance_guard(*instance_mutex);
    std::scoped_lock write_guard(mutex_);
    return meta_indexers_.emplace(instance_id, std::move(indexer)).second;
}

std::map<std::string, std::shared_ptr<MetaIndexer>> MetaIndexerManager::GetIndexers() const {
    std::shared_lock read_guard(mutex_);
    return meta_indexers_;
}

std::optional<MetaIndexerManager::StorageBackendConfigs> MetaIndexerManager::TryGetStorageBackendConfigs() const {
    std::shared_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        return std::nullopt;
    }
    StorageBackendConfigs configs;
    for (const auto &[id, indexer] : meta_indexers_) {
        configs.emplace(id, indexer->GetStorageBackendConfig());
    }
    return configs;
}
void MetaIndexerManager::DoCleanup() {
    std::unique_lock lifecycle_guard(lifecycle_mutex_);
    std::scoped_lock write_guard(mutex_);
    meta_indexers_.clear();
    instance_mutexes_.clear();
}

size_t MetaIndexerManager::GetIndexerSize() {
    std::shared_lock read_guard(mutex_);
    return meta_indexers_.size();
}

} // namespace kv_cache_manager
