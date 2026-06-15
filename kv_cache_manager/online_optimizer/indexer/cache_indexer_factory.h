#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/online_optimizer/indexer/cache_indexer.h"

namespace kv_cache_manager {

class CacheIndexerFactory {
public:
    static std::unique_ptr<CacheIndexer> CreateCacheIndexer(
        const std::string &indexer_type,
        int64_t max_key_count,
        const std::vector<double> &capacity_gb,
        int64_t size_full_only,
        int64_t size_full_linear,
        int32_t linear_step,
        int64_t ttl_seconds = 0);
};

} // namespace kv_cache_manager
