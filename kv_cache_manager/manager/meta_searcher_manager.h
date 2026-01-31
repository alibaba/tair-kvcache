#pragma once

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "kv_cache_manager/manager/meta_searcher.h"

namespace kv_cache_manager {
class RegistryManager;
class MetaIndexerManager;
class RequestContext;

class MetaSearcherManager {
public:
    MetaSearcherManager(std::shared_ptr<RegistryManager> registry_manager,
                        std::shared_ptr<MetaIndexerManager> meta_indexer_manager);
    ~MetaSearcherManager();
    MetaSearcher *TryCreateMetaSearcher(RequestContext *request_context, const std::string &instance_id);
    MetaSearcher *GetMetaSearcher(const std::string &instance_id) const;
    void DoCleanup();

private:
    MetaSearcher *GetMetaSearcherUnsafe(const std::string &instance_id) const;

    // 需要清理 - 释放所有meta searcher
    std::unordered_map<std::string, std::unique_ptr<MetaSearcher>> meta_searcher_map_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<MetaIndexerManager> meta_indexer_manager_;
    mutable std::shared_mutex mutex_;
};

} // namespace kv_cache_manager