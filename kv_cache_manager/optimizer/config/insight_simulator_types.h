#pragma once
#include <memory>
#include <string>
namespace kv_cache_manager {

struct GetCacheLocationRes {
    std::string trace_id;
    int64_t kvcm_hit_length;
};

struct WriteCacheRes {
    std::string trace_id;
    int64_t kvcm_write_length;
    int64_t kvcm_write_hit_length;
};

} // namespace kv_cache_manager