#pragma once
#include <map>
#include <string>
#include <vector>

namespace kv_cache_manager {
static const std::string META_DUMMY_BACKEND_TYPE_STR = "dummy";
static const std::string META_LOCAL_BACKEND_TYPE_STR = "local";
static const std::string META_REDIS_BACKEND_TYPE_STR = "redis";
static const std::string META_CACHED_BACKEND_TYPE_STR = "cached";

// V8 §2.1.1 Hash field naming convention.
//
// Every block_key Redis Hash uses three classes of fields:
//   BP#{prop}              -- block-level property (e.g. BP#hit_count)
//   L#{location_id}        -- the JSON of a single CacheLocation
//   P#{location_id}#{prop} -- per-location auxiliary property
// '#' is reserved for KVCM internal use; '_' is left for users. Per-location
// property names MUST NOT contain '#' so the trailing '#' splits {prop} cleanly
// from {location_id}.
static const std::string PROPERTY_BLOCK_PREFIX = "BP#";
static const std::string PROPERTY_LOCATION_PREFIX = "L#";
static const std::string PROPERTY_LOC_SUB_PREFIX = "P#";

// Legacy "__"-prefixed property names predate V8 and are still recognised on
// the read path so a storage backend that has been around since V7 will not
// brick the code. New deployments use the BP# names below.
static const std::string PROPERTY_INNER_PREFIX = "__";

// PROPERTY_URI is no longer a real Redis field; MetaIndexer treats it as an
// in-process synthetic carrier between read/modify/write callers and the L#*
// field decomposition. Kept here so existing modifier signatures
// (`std::string &uri` referring to BlockCacheLocationsMeta JSON) continue to
// compile without a renaming churn.
static const std::string PROPERTY_URI = "__uri__";

// Block-level property names (V8 BP# layout).
static const std::string PROPERTY_TTL = "BP#ttl";
static const std::string PROPERTY_HIT_COUNT = "BP#hit_count";
static const std::string PROPERTY_LRU_TIME = "BP#lru_time";

// Top-level instance metadata kept under the "metadata" key (NOT the per-block
// hashes). Stays under the legacy "__" namespace because instance metadata is
// not bound by the BP#/L#/P# block layout.
static const std::string METADATA_PROPERTY_KEY_COUNT = "__key_count__";
static const std::string METADATA_PROPERTY_STORAGE_USAGE_DATA = "__storage_usage_data__";

static const std::string SCAN_BASE_CURSOR = "0";

// True if `name` is reserved for internal KVCM use (BP#, L#, P# or legacy __).
// Callers strip such fields before exposing properties to user-facing code.
inline bool IsInternalPropertyName(const std::string &name) noexcept {
    return name.rfind(PROPERTY_BLOCK_PREFIX, 0) == 0 || name.rfind(PROPERTY_LOCATION_PREFIX, 0) == 0 ||
           name.rfind(PROPERTY_LOC_SUB_PREFIX, 0) == 0 || name.rfind(PROPERTY_INNER_PREFIX, 0) == 0;
}

// MetaLocalBackend default constants
static const size_t META_LOCAL_BACKEND_DEFAULT_CAPACITY = 32ULL * 1024;
static const int32_t META_LOCAL_BACKEND_DEFAULT_NUM_SHARD_BITS = 10;
static const int32_t META_LOCAL_BACKEND_DEFAULT_SAMPLE_TIMES = 10;

} // namespace kv_cache_manager
