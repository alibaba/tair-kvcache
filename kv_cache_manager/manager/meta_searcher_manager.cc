#include "kv_cache_manager/manager/meta_searcher_manager.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"

namespace kv_cache_manager {

MetaSearcherManager::MetaSearcherManager(std::shared_ptr<RegistryManager> registry_manager,
                                         std::shared_ptr<MetaIndexerManager> meta_indexer_manager)
    : registry_manager_(registry_manager), meta_indexer_manager_(meta_indexer_manager) {
    assert(registry_manager_);
    assert(meta_indexer_manager_);
}

MetaSearcherManager::~MetaSearcherManager() = default;

MetaSearcher *MetaSearcherManager::TryCreateMetaSearcher(RequestContext *request_context,
                                                         const std::string &instance_id) {
    MetaSearcher *meta_searcher = GetMetaSearcher(instance_id);
    if (meta_searcher) {
        return meta_searcher;
    }
    ErrorCode ec = ErrorCode::EC_UNKNOWN;
    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (instance_info == nullptr) {
        request_context->error_tracer()->AddErrorMsg("instance not registered");
        KVCM_LOG_ERROR("instance [%s] not registered", instance_id.c_str());
        return nullptr;
    }
    const std::string &instance_group = instance_info->instance_group_name();
    auto cache_config = registry_manager_->GetCacheConfig(instance_group);
    if (cache_config == nullptr) {
        request_context->error_tracer()->AddErrorMsg("instance group not found");
        KVCM_LOG_ERROR("instance group [%s] not found", instance_group.c_str());
        return nullptr;
    }
    {
        std::scoped_lock write_guard(mutex_);
        // double check
        meta_searcher = GetMetaSearcherUnsafe(instance_id);
        if (meta_searcher != nullptr) {
            return meta_searcher;
        }
        ec = meta_indexer_manager_->CreateMetaIndexer(instance_id, cache_config->meta_indexer_config());
        if (ec == ErrorCode::EC_OK) {
            if (auto pair = meta_searcher_map_.emplace(
                    instance_id, std::make_unique<MetaSearcher>(meta_indexer_manager_->GetMetaIndexer(instance_id)));
                pair.second) {
                return pair.first->second.get();
            }
        } else if (ec == ErrorCode::EC_EXIST) {
            KVCM_LOG_ERROR("meta indexer exist! IMPOSSIBLE!");
            assert(false);
        }
    }
    request_context->error_tracer()->AddErrorMsg("create meta indexer failed");
    KVCM_LOG_ERROR("create meta searcher fail, error code [%d]", ec);
    return nullptr;
}

MetaSearcher *MetaSearcherManager::GetMetaSearcher(const std::string &instance_id) const {
    std::shared_lock read_guard(mutex_);
    return GetMetaSearcherUnsafe(instance_id);
}
void MetaSearcherManager::DoCleanup() {
    std::scoped_lock write_guard(mutex_);
    meta_searcher_map_.clear();
}

MetaSearcher *MetaSearcherManager::GetMetaSearcherUnsafe(const std::string &instance_id) const {
    auto it = meta_searcher_map_.find(instance_id);
    if (it != meta_searcher_map_.end()) {
        KVCM_LOG_DEBUG("meta searcher [%s] exist", instance_id.c_str());
        return it->second.get();
    }
    return nullptr;
}

} // namespace kv_cache_manager
