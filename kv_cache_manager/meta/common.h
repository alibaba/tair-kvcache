#pragma once
#include <map>
#include <string>
#include <vector>

namespace kv_cache_manager {
static const std::string META_DUMMY_BACKEND_TYPE_STR = "dummy";
static const std::string META_LOCAL_BACKEND_TYPE_STR = "local";
static const std::string META_REDIS_BACKEND_TYPE_STR = "redis";
static const std::string META_CACHED_BACKEND_TYPE_STR = "cached";

static const std::string PROPERTY_INNER_PREFIX = "__";
static const std::string PROPERTY_URI = "__uri__";
static const std::string PROPERTY_TTL = "__ttl__";
static const std::string PROPERTY_HIT_COUNT = "__hit_count__";
static const std::string PROPERTY_LRU_TIME = "__lru_time__";

static const std::string METADATA_PROPERTY_KEY_COUNT = "__key_count__";
static const std::string METADATA_PROPERTY_STORAGE_USAGE_DATA = "__storage_usage_data__";

static const std::string SCAN_BASE_CURSOR = "0";

// MetaLocalBackend default constants
static const size_t META_LOCAL_BACKEND_DEFAULT_CAPACITY = 32ULL * 1024;
static const int32_t META_LOCAL_BACKEND_DEFAULT_NUM_SHARD_BITS = 10;
static const int32_t META_LOCAL_BACKEND_DEFAULT_SAMPLE_TIMES = 10;

} // namespace kv_cache_manager
