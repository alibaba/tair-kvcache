#pragma once

#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

// Reclaim indexer interface. Implementations decide the strategy (LRU, LFU, TTL, etc.).
// Thread-safe. Owned by MetaStorageBackendManager.
class ReclaimIndexer {
public:
    ReclaimIndexer() = default;
    virtual ~ReclaimIndexer() = default;

    ReclaimIndexer(const ReclaimIndexer &) = delete;
    ReclaimIndexer &operator=(const ReclaimIndexer &) = delete;

    // Index keys with their locations (write path / recover).
    // For each key[i]: if previous_error_codes[i] != EC_OK, skip and propagate
    // the prior error; otherwise perform the Add and return per-key error code.
    virtual std::vector<ErrorCode> Add(const KeyTypeVec &keys,
                                       const CacheLocationMapVector &locations,
                                       const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;

    // Mark keys as accessed (read path).
    virtual std::vector<ErrorCode> Touch(const KeyTypeVec &keys,
                                         const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;

    // Remove specific locations for each key (partial delete).
    virtual std::vector<ErrorCode> Remove(const KeyTypeVec &keys,
                                          const LocationIdsPerKey &location_ids,
                                          const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;

    // Remove keys entirely from the indexer (full delete).
    virtual std::vector<ErrorCode> Remove(const KeyTypeVec &keys,
                                          const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;

    // Sample up to `count` most-reclaimable keys for the given node.
    virtual ErrorCode
    Sample(size_t count, const std::unordered_set<std::string> &node_ids, KeyTypeVec &out_keys) noexcept = 0;
};

} // namespace kv_cache_manager
