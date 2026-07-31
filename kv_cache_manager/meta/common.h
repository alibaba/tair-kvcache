#pragma once
#include <map>
#include <string>
#include <vector>

namespace kv_cache_manager {
static const std::string META_DUMMY_BACKEND_TYPE_STR = "dummy";
static const std::string META_LOCAL_BACKEND_TYPE_STR = "local";
static const std::string META_REDIS_BACKEND_TYPE_STR = "redis";
static const std::string META_ASYNC_REDIS_BACKEND_TYPE_STR = "async_redis";
static const std::string META_CACHED_BACKEND_TYPE_STR = "cached";

// Hash field prefixes: BP#{prop}, L#{location_id}, P#{location_id}#{prop}.
static const std::string PROPERTY_BLOCK_PREFIX = "BP#";
static const std::string PROPERTY_LOCATION_PREFIX = "L#";
static const std::string PROPERTY_LOC_SUB_PREFIX = "P#";

static const std::string PROPERTY_INNER_PREFIX = "__";

// Synthetic in-process key; not a real Redis field. Kept for API compatibility.
static const std::string PROPERTY_URI = "__uri__";

static const std::string PROPERTY_TTL = "BP#ttl";
static const std::string PROPERTY_HIT_COUNT = "BP#hit_count";
static const std::string PROPERTY_LRU_TIME = "BP#lru_time";
static const std::string PROPERTY_PREV_BLOCK_KEY = "BP#prev_key";

// Instance-level metadata (under "metadata" key, not per-block hashes).
static const std::string METADATA_PROPERTY_KEY_COUNT = "__key_count__";
static const std::string METADATA_PROPERTY_STORAGE_USAGE_DATA = "__storage_usage_data__";

static const std::string SCAN_BASE_CURSOR = "0";

inline bool IsInternalPropertyName(const std::string &name) noexcept {
    return name.rfind(PROPERTY_BLOCK_PREFIX, 0) == 0 || name.rfind(PROPERTY_LOCATION_PREFIX, 0) == 0 ||
           name.rfind(PROPERTY_LOC_SUB_PREFIX, 0) == 0 || name.rfind(PROPERTY_INNER_PREFIX, 0) == 0;
}

// MetaLocalBackend default constants
static const size_t META_LOCAL_BACKEND_DEFAULT_CAPACITY = 32ULL * 1024;
static const int32_t META_LOCAL_BACKEND_DEFAULT_NUM_SHARD_BITS = 10;
static const int32_t META_LOCAL_BACKEND_DEFAULT_SAMPLE_TIMES = 10;

} // namespace kv_cache_manager
