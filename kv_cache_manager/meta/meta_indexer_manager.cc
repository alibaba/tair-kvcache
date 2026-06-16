#include "kv_cache_manager/meta/meta_indexer_manager.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/metrics/revisit_interval_histogram.h"

namespace kv_cache_manager {

void MetaIndexerManager::SetRevisitHistogramConfig(std::shared_ptr<MetricsRegistry> registry,
                                                   const std::vector<double> &boundaries) {
    metrics_registry_ = std::move(registry);
    revisit_boundaries_ = boundaries;
}

ErrorCode MetaIndexerManager::CreateMetaIndexer(const std::string &instance_id,
                                                const std::shared_ptr<MetaIndexerConfig> &config,
                                                const std::vector<double> &boundaries) {
    auto indexer = GetMetaIndexer(instance_id);
    if (indexer) {
        return ErrorCode::EC_EXIST;
    }
    {
        std::scoped_lock write_guard(mutex_);
        // double checkout
        auto indexer = GetMetaIndexerUnsafe(instance_id);
        if (indexer) {
            return ErrorCode::EC_EXIST;
        }
        indexer = std::make_shared<MetaIndexer>();
        auto ec = indexer->Init(instance_id, config);
        if (ec != ErrorCode::EC_OK) {
            KVCM_LOG_ERROR("Init meta indexer failed, instance_id: %s", instance_id.c_str());
            return ec;
        }

        // Resolve boundaries: per-instance override > global default
        const std::vector<double> &resolved = boundaries.empty() ? revisit_boundaries_ : boundaries;

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

        meta_indexers_[instance_id] = indexer;
    }
    KVCM_LOG_INFO("Create meta indexer success, instance_id: %s", instance_id.c_str());
    return ErrorCode::EC_OK;
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
    std::scoped_lock write_guard(mutex_);
    meta_indexers_.clear();
}

size_t MetaIndexerManager::GetIndexerSize() {
    std::shared_lock read_guard(mutex_);
    return meta_indexers_.size();
}

} // namespace kv_cache_manager
