#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>

#include "kv_cache_manager/common/error_code.h"

namespace kv_cache_manager {

class MetaIndexerConfig;
class MetaIndexer;

class MetaIndexerManager {
public:
    MetaIndexerManager() = default;

    ~MetaIndexerManager() = default;

    ErrorCode CreateMetaIndexer(const std::string &instance_id, const std::shared_ptr<MetaIndexerConfig> &config);

    std::shared_ptr<MetaIndexer> GetMetaIndexer(const std::string &instance_id) const;

    ErrorCode DeleteMetaIndexer(const std::string &instance_id);

    size_t GetIndexerSize();

    std::map<std::string, std::shared_ptr<MetaIndexer>> GetIndexers() const;

    void DoCleanup();

private:
    std::shared_ptr<MetaIndexer> GetMetaIndexerUnsafe(const std::string &instance_id) const;

private:
    // instance_id -> MetaIndexer
    std::map<std::string, std::shared_ptr<MetaIndexer>> meta_indexers_;
    mutable std::shared_mutex mutex_;
};

} // namespace kv_cache_manager
