#include "kv_cache_manager/meta/meta_indexer_manager.h"

#include <algorithm>
#include <exception>
#include <mutex>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/instance_info.h"
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
                                                const std::vector<double> &boundaries,
                                                std::shared_ptr<const InstanceInfo> instance_info) {
    std::shared_lock lifecycle_guard(lifecycle_mutex_);
    auto instance_mutex = GetInstanceMutex(instance_id);
    std::unique_lock instance_guard(*instance_mutex);
    if (instance_info && instance_info->instance_id() != instance_id) {
        return EC_BADARGS;
    }
    if (GetMetaIndexer(instance_id)) {
        return EC_EXIST;
    }
    if (!config) {
        return EC_BADARGS;
    }
    auto indexer = std::make_shared<MetaIndexer>(std::move(instance_info));
    const std::vector<double> &resolved = boundaries.empty() ? revisit_boundaries_ : boundaries;
    indexer->initial_config_json_ = config->ToJsonString();
    indexer->initial_histogram_boundaries_ = resolved;
    indexer->SetQueryExecutor(query_executor_);
    const ErrorCode ec = InitMetaIndexer(indexer, instance_id, config);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Init meta indexer failed, instance[%s] ec[%d]", instance_id.c_str(), ec);
        return ec;
    }
    if (indexer->SupportsWriteSession() && metrics_registry_ && !resolved.empty()) {
        auto histogram = std::make_shared<RevisitIntervalHistogram>();
        if (histogram->Init(metrics_registry_, resolved, instance_id)) {
            indexer->SetRevisitHistogram(std::move(histogram));
        } else {
            KVCM_LOG_WARN("Init revisit histogram failed, instance[%s]", instance_id.c_str());
        }
    }
    indexer->manager_initialized_ = true;
    {
        std::scoped_lock write_guard(mutex_);
        if (!meta_indexers_.emplace(instance_id, indexer).second) {
            return EC_EXIST;
        }
    }
    return EC_OK;
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
                                                     const std::shared_ptr<const InstanceInfo> &instance_info,
                                                     std::shared_ptr<MetaIndexer> &out_indexer) const {
    out_indexer.reset();
    if (!config || !instance_info || instance_info->instance_id() != instance_id) {
        return EC_BADARGS;
    }
    std::shared_lock lifecycle_guard(lifecycle_mutex_);
    auto instance_mutex = GetInstanceMutex(instance_id);
    std::unique_lock instance_guard(*instance_mutex);
    auto indexer = GetMetaIndexer(instance_id);
    if (!indexer) {
        return EC_NOENT;
    }
    const auto &original_instance = indexer->instance_info_;
    if (!indexer->manager_initialized_ || !indexer->backend_manager_ || indexer->instance_id_ != instance_id ||
        original_instance != instance_info || original_instance.owner_before(instance_info) ||
        instance_info.owner_before(original_instance)) {
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
    // Retain the local owner so a rejected indexer's slow teardown runs after releasing the map lock.
    return meta_indexers_.emplace(instance_id, indexer).second;
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
