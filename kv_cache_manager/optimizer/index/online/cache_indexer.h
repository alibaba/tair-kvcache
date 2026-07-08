#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kv_cache_manager {

// Per-bucket hit count for cache age distribution.
// Each bucket covers hits whose age (now - last_access_time) falls within
// [0, threshold[0]), [threshold[0], threshold[1]), ..., [threshold[N-1], +inf).
struct HitAgeBucketInfo {
    int64_t threshold_seconds; // upper bound of this bucket (0 means "+inf")
    int64_t hit_count;
};

class CacheIndexer {
public:
    virtual ~CacheIndexer() = default;

    CacheIndexer() = default;
    CacheIndexer(const CacheIndexer &) = delete;
    CacheIndexer &operator=(const CacheIndexer &) = delete;

    // Initialize the indexer with capacity and size parameters.
    // capacity_gb: capacity tiers in GB.
    // size_full_only: byte size of a full-only block.
    // size_full_linear: byte size of a full+linear block.
    // linear_step: linear step factor (>=0).
    virtual void Init(const std::vector<double> &capacity_gb,
                      int64_t size_full_only,
                      int64_t size_full_linear,
                      int32_t linear_step) = 0;

    // Process a batch of key accesses and compute per-capacity prefix hit count.
    // keys: the block keys in query order.
    // hit_count: output vector sized to number of capacity tiers, filled with
    //            the count of contiguous prefix hits for each tier.
    // key_hits: optional output, one entry per input key. When provided, it is
    // filled with whether the key was resident before this query updated the
    // indexer. This is separate from prefix hit_count.
    virtual void ProcessKeys(const std::vector<int64_t> &keys,
                             std::vector<int64_t> &hit_count,
                             int64_t &max_hit_count,
                             std::vector<bool> *key_hits = nullptr) = 0;

    virtual int64_t unique_count() const = 0;

    // Number of keys evicted from the indexer.
    virtual int64_t eviction_count() const = 0;

    // Estimated memory usage in bytes of internal data structures.
    virtual int64_t memory_usage_bytes() const = 0;

    // Estimated total kv cache size in bytes for all keys currently tracked.
    virtual int64_t kv_cache_usage_bytes() const = 0;

    // Resident unique key count for each configured capacity. Implementations
    // that do not maintain per-capacity state may return an empty vector.
    virtual std::vector<int64_t> capacity_unique_counts() const { return {}; }

    // Remove a specific key from the indexer.
    // Returns true if the key existed and was removed.
    virtual bool RemoveKey(int64_t key) { return false; }

    // Number of keys evicted due to TTL expiration.
    virtual int64_t ttl_eviction_count() const { return 0; }

    // Called after processing all keys in a query batch.
    // Subclasses may perform eviction, compaction, etc.
    virtual void PostQueryMaintenance() {}

    // Return per-bucket hit counts for cache age distribution.
    // Default returns empty (no age tracking).
    virtual std::vector<HitAgeBucketInfo> GetHitAgeBuckets() const { return {}; }
};

} // namespace kv_cache_manager
