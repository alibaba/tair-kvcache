#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

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

} // namespace kv_cache_manager
