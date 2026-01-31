#include "kv_cache_manager/meta/meta_indexer_manager.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/meta/meta_indexer.h"

namespace kv_cache_manager {

ErrorCode MetaIndexerManager::CreateMetaIndexer(const std::string &instance_id,
                                                const std::shared_ptr<MetaIndexerConfig> &config) {
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
