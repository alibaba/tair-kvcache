#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/hash/hash.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

inline uint64_t HashKey(KeyType key) noexcept {
    constexpr uint64_t kSeed = 0x9E3779B97F4A7C15ULL;
    return Hash64(reinterpret_cast<const char *>(&key), sizeof(key), kSeed);
}

// Maps a key to a shard index using a 64-bit hash so that adjacent keys
// (which often share low bits) are distributed evenly across shards.
// `shard_mask` MUST equal `shard_num - 1` where `shard_num` is a power of two.
inline int32_t GetShardIndex(KeyType key, size_t shard_mask) noexcept {
    assert(((shard_mask + 1) & shard_mask) == 0);
    return static_cast<int32_t>(HashKey(key) & static_cast<uint64_t>(shard_mask));
}

FieldMap SerializeToFieldMap(const CacheLocationMap &locations, const PropertyMap &properties);

ErrorCode DeserializeFieldMap(const FieldMap &field_map, CacheLocationMap &out_locations, PropertyMap &out_properties);

ErrorCode DeserializeLocations(const FieldMap &field_map, CacheLocationMap &out_locations);

void ExtractLocationIds(const FieldMap &field_map, std::vector<LocationId> &out_location_ids);

std::vector<std::string>
AppendPrefixToKeys(const std::string &cache_key_prefix, const KeyTypeVec &keys);

bool StripPrefixInKeys(const std::string &cache_key_prefix,
                       const std::string &instance_id,
                       const std::vector<std::string> &keys_with_prefix,
                       KeyTypeVec &out_keys);

} // namespace kv_cache_manager
