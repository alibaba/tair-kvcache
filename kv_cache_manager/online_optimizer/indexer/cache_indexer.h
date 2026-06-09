#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace kv_cache_manager {

class CacheIndexer {
public:
    virtual ~CacheIndexer() = default;

    CacheIndexer() = default;
    CacheIndexer(const CacheIndexer &) = delete;
    CacheIndexer &operator=(const CacheIndexer &) = delete;

    // Returns stack distance for this key.
    // INT64_MAX means first-time access (cold miss).
    virtual int64_t ProcessKey(int64_t key) = 0;

    virtual int64_t unique_count() const = 0;

    virtual int64_t peak_unique_count() const = 0;

    // Number of keys evicted from the indexer.
    virtual int64_t eviction_count() const = 0;

    // Estimated memory usage in bytes of internal data structures.
    virtual int64_t memory_usage_bytes() const = 0;

    // Called after processing all keys in a query batch.
    // Subclasses may perform eviction, compaction, etc.
    virtual void PostQueryMaintenance() {}
};

std::unique_ptr<CacheIndexer> CreateCacheIndexer(const std::string &indexer_type, int64_t max_key_count);

} // namespace kv_cache_manager
