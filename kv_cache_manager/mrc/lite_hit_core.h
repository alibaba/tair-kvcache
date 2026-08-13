#pragma once

#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

namespace kv_cache_manager {

// Exact replication of the offline LiteHit Fenwick core's request-level
// semantics (kvs-profiler `LiteHit`), so online results stay directly
// comparable with the formal offline MRC reports:
//
//   Phase 1: the request's prefix hit curve is evaluated against the
//     request-start LRU snapshot. A cold key stops the prefix for every
//     capacity. Block i of the surviving prefix hits at capacity C iff
//     C >= threshold(i), where threshold is the running max of each
//     block's required_blocks (reuse distance + 1) kept strictly
//     increasing.
//   Phase 2: every complete block is committed tail-to-head, so a chain
//     head is always newer than its resident descendants and the global
//     LRU victim is always a leaf (vLLM free-queue / SGLang radix cache
//     eviction order).
//
// Not thread-safe; callers serialize access externally.
class LiteHitCore {
public:
    // max_tracked_blocks bounds the exact LRU stack. Results remain exact for
    // every simulated capacity <= max_tracked_blocks: entries older than that
    // cannot hit at any reported capacity and are therefore safe to forget.
    // A non-positive value keeps the historical unbounded behavior and is
    // intended only for offline validation.
    explicit LiteHitCore(int64_t initial_capacity = 4096, int64_t max_tracked_blocks = 0);

    LiteHitCore(const LiteHitCore &) = delete;
    LiteHitCore &operator=(const LiteHitCore &) = delete;

    // Process one request (all complete blocks, request order). Appends one
    // capacity threshold (in blocks, >= 1) per prefix hit block to
    // out_thresholds: the block hits an LRU cache of capacity C iff
    // C >= threshold. Cold and post-miss blocks emit nothing.
    void ProcessRequest(const std::vector<int64_t> &block_keys, std::vector<uint64_t> &out_thresholds);

    // Clears request history while retaining allocated buffers. Used when an
    // instance's metadata changes and its old hit curve is no longer valid.
    void Reset();

    int64_t unique_blocks() const { return static_cast<int64_t>(last_pos_.size()); }
    int64_t max_tracked_blocks() const { return max_tracked_blocks_; }
    int64_t memory_usage_bytes() const;

private:
    void FenwickAdd(int64_t pos, int64_t delta);
    int64_t FenwickPrefixSum(int64_t pos) const;
    void EnsureSlot();
    void Compact();
    void TrimToLimit();

    int64_t capacity_ = 0;
    int64_t next_pos_ = 1;
    int64_t max_tracked_blocks_ = 0;
    std::vector<int64_t> tree_;
    std::unordered_map<int64_t, int64_t> last_pos_;
    std::map<int64_t, int64_t> pos_to_key_;

    // Scratch buffers reused across requests.
    std::unordered_map<int64_t, uint64_t> snapshot_required_;
    std::unordered_map<int64_t, size_t> first_occurrence_;
};

} // namespace kv_cache_manager
