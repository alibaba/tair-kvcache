#pragma once
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/optimizer/config/types.h"

namespace kv_cache_manager {

struct GetCacheLocationRes {
    std::string trace_id;
    int64_t kvcm_hit_length;
    std::vector<MaterializedKeySequence> evicted_materialized_sequences;
};

struct WriteCacheRes {
    std::string trace_id;
    int64_t kvcm_write_length;
    int64_t kvcm_write_hit_length;
    std::vector<MaterializedKeySequence> pool_source_write_sequences;
    std::vector<MaterializedKeySequence> evicted_materialized_sequences;
};

} // namespace kv_cache_manager
