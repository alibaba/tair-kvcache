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

inline constexpr uint64_t kDefaultMetaShardHashSeed = 0x9E3779B97F4A7C15ULL;

inline uint64_t HashKey(KeyType key) noexcept {
    return Hash64(reinterpret_cast<const char *>(&key), sizeof(key), kDefaultMetaShardHashSeed);
}

// Maps a key to a shard index using a 64-bit hash so that adjacent keys
// (which often share low bits) are distributed evenly across shards.
// `shard_mask` MUST equal `shard_num - 1` where `shard_num` is a power of two.
inline int32_t GetShardIndex(KeyType key, size_t shard_mask, uint64_t hash_seed) noexcept {
    assert(((shard_mask + 1) & shard_mask) == 0);
    return static_cast<int32_t>(Hash64(reinterpret_cast<const char *>(&key), sizeof(key), hash_seed) &
                                static_cast<uint64_t>(shard_mask));
}

inline int32_t GetShardIndex(KeyType key, size_t shard_mask) noexcept {
    return GetShardIndex(key, shard_mask, kDefaultMetaShardHashSeed);
}

class RedisLocationCodec {
public:
    RedisLocationCodec() = default;
    explicit RedisLocationCodec(SharedLocationSpecName simple_spec_name);

    FieldMap SerializeToFieldMap(const CacheLocationMap &locations, PropertyMap properties) const;
    ErrorCode DeserializeFieldMap(const FieldMap &field_map,
                                  CacheLocationMap &out_locations,
                                  PropertyMap *out_properties,
                                  LocationIdInterner *id_interner,
                                  LocationSpecNameInterner *name_interner) const;
    bool DeserializeLocation(const LocationId &id,
                             const std::string &value,
                             LocationIdInterner *id_interner,
                             LocationSpecNameInterner *name_interner,
                             CacheLocation &out_location) const;

private:
    bool SerializeCompactLocation(const LocationId &id, const CacheLocation &location, std::string &out_value) const;

    SharedLocationSpecName simple_spec_name_;
};

void ExtractLocationIds(const FieldMap &field_map, std::vector<LocationId> &out_location_ids);

std::vector<std::string> AppendPrefixToKeys(const std::string &cache_key_prefix, const KeyTypeVec &keys);

bool StripPrefixInKeys(const std::string &cache_key_prefix,
                       const std::string &instance_id,
                       const std::vector<std::string> &keys_with_prefix,
                       KeyTypeVec &out_keys);

} // namespace kv_cache_manager
