#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "kv_cache_manager/common/hash/hash.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

// Maps a key to a shard-mutex index using a 64-bit hash so that adjacent keys
// (which often share low bits) are distributed evenly across shards. The hash
// is computed once per call and folded into [0, mutex_shard_mask] using a
// bit mask (`mutex_shard_mask` MUST equal `shard_num - 1` where `shard_num`
// is a power of two, i.e. mutex_shard_mask is a contiguous low-bit mask;
// validated at MetaIndexer init).
inline int32_t GetShardIndex(KeyType key, size_t mutex_shard_mask) noexcept {
    assert(((mutex_shard_mask + 1) & mutex_shard_mask) == 0);
    constexpr uint64_t kShardSeed = 0x9E3779B97F4A7C15ULL; // arbitrary, non-zero
    const uint64_t hash =
        Hash64(reinterpret_cast<const char *>(&key), sizeof(key), kShardSeed);
    return static_cast<int32_t>(hash & static_cast<uint64_t>(mutex_shard_mask));
}

} // namespace kv_cache_manager
