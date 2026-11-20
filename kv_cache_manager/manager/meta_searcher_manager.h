#pragma once

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "kv_cache_manager/manager/meta_searcher.h"

namespace kv_cache_manager {
class RegistryManager;
class MetaIndexer;
class MetaIndexerManager;
class RequestContext;

class MetaSearcherManager {
public:
    MetaSearcherManager(std::shared_ptr<RegistryManager> registry_manager,
                        std::shared_ptr<MetaIndexerManager> meta_indexer_manager);
    virtual ~MetaSearcherManager();
    MetaSearcher *TryCreateMetaSearcher(RequestContext *request_context,
                                        const std::string &instance_id,
                                        CheckLocDataExistFunc check_loc_data_exist,
                                        SubmitDelReqFunc submit_del_req);
    MetaSearcher *GetMetaSearcher(const std::string &instance_id) const;
    // Callers drain creation and instance management before cleanup.
    void DoCleanup();

protected:
    virtual std::unique_ptr<MetaSearcher> CreateMetaSearcher(const std::shared_ptr<MetaIndexer> &indexer,
                                                             const CheckLocDataExistFunc &check_loc_data_exist,
                                                             const SubmitDelReqFunc &submit_del_req);

private:
    MetaSearcher *GetMetaSearcherUnsafe(const std::string &instance_id) const;
    std::shared_ptr<std::mutex> GetInstanceMutex(const std::string &instance_id);

    // 需要清理 - 释放所有meta searcher
    std::unordered_map<std::string, std::unique_ptr<MetaSearcher>> meta_searcher_map_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<MetaIndexerManager> meta_indexer_manager_;
    mutable std::shared_mutex mutex_;
    // Lock order: instance -> map; initialize outside the map lock.
    // Operations retain the mutex; the next lookup removes expired entries.
    std::unordered_map<std::string, std::weak_ptr<std::mutex>> instance_mutexes_;
};

} // namespace kv_cache_manager
