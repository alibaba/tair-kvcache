#pragma once

#include <memory>
#include <string>

#include "kv_cache_manager/meta/reclaim_indexer/reclaim_indexer.h"

namespace kv_cache_manager {

class ReclaimIndexerFactory {
public:
    static std::unique_ptr<ReclaimIndexer> Create(const std::string &type);
};

} // namespace kv_cache_manager
