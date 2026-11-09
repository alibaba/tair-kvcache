#include "kv_cache_manager/meta/meta_indexer_manager.h"

#include <algorithm>
#include <exception>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/query_executor.h"
#include "kv_cache_manager/metrics/revisit_interval_histogram.h"

namespace kv_cache_manager {

void MetaIndexerManager::SetRevisitHistogramConfig(std::shared_ptr<MetricsRegistry> registry,
                                                   const std::vector<double> &boundaries) {
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
                                                const std::vector<double> &boundaries,
                                                std::shared_ptr<const InstanceInfo> instance_info) {
    if (instance_info && instance_info->instance_id() != instance_id) {
        return EC_BADARGS;
    }
    std::scoped_lock write_guard(mutex_);
    if (GetMetaIndexerUnsafe(instance_id)) {
        return EC_EXIST;
    }
    auto indexer = std::make_shared<MetaIndexer>(std::move(instance_info));
    indexer->SetQueryExecutor(query_executor_);
    const ErrorCode ec = indexer->Init(instance_id, config);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Init meta indexer failed, instance[%s] ec[%d]", instance_id.c_str(), ec);
        return ec;
    }
    const std::vector<double> &resolved = boundaries.empty() ? revisit_boundaries_ : boundaries;
    if (indexer->SupportsWriteSession() && metrics_registry_ && !resolved.empty()) {
        auto histogram = std::make_shared<RevisitIntervalHistogram>();
        if (histogram->Init(metrics_registry_, resolved, instance_id)) {
            indexer->SetRevisitHistogram(std::move(histogram));
        } else {
            KVCM_LOG_WARN("Init revisit histogram failed, instance[%s]", instance_id.c_str());
        }
    }
    if (!meta_indexers_.emplace(instance_id, std::move(indexer)).second) {
        return EC_EXIST;
    }
    return EC_OK;
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
    decltype(meta_indexers_)::node_type node;
    {
        std::scoped_lock write_guard(mutex_);
        node = meta_indexers_.extract(instance_id);
    }
    return node.empty() ? nullptr : std::move(node.mapped());
}

bool MetaIndexerManager::RestoreMetaIndexer(const std::string &instance_id, std::shared_ptr<MetaIndexer> indexer) {
    std::scoped_lock write_guard(mutex_);
    return meta_indexers_.emplace(instance_id, std::move(indexer)).second;
}

std::map<std::string, std::shared_ptr<MetaIndexer>> MetaIndexerManager::GetIndexers() const {
    std::shared_lock read_guard(mutex_);
    return meta_indexers_;
}
void MetaIndexerManager::DoCleanup() {
    std::scoped_lock write_guard(mutex_);
    meta_indexers_.clear();
}

size_t MetaIndexerManager::GetIndexerSize() {
    std::shared_lock read_guard(mutex_);
    return meta_indexers_.size();
}

} // namespace kv_cache_manager
