#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/optimizer/config/types.h"

namespace kv_cache_manager {

struct GetCacheLocationRes {
    std::string trace_id;
    int64_t kvcm_hit_length;
    std::vector<size_t> hit_indices;
    std::vector<int64_t> evicted_keys;
};

struct WriteCacheRes {
    std::string trace_id;
    int64_t kvcm_write_length;
    int64_t kvcm_write_hit_length;
    std::vector<int64_t> pool_source_write_keys;
    std::vector<int64_t> evicted_keys;
};

} // namespace kv_cache_manager
