#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/liteHit/dynamic_fenwick_tree.h"

namespace kv_cache_manager {

// LiteHit is an exact multi-capacity LRU analyzer for equal-charge
// full-attention blocks. It keeps one global LRU stack plus aggregate hit
// counters, regardless of how many capacities are queried.
class LiteHit {
public:
    static constexpr int64_t kInfiniteCapacity = -1;

    struct TraceRequest {
        std::vector<int64_t> block_keys;
        uint64_t input_token_len = 0;
    };

    // The same value type is used for one request and for cumulative results.
    // hit_count is a prefix-hit block count; hit_tokens is the numerator used
    // by hit_rate.
    struct CapacityResult {
        int64_t capacity_blocks = 0;
        uint64_t hit_count = 0;
        uint64_t hit_tokens = 0;
        uint64_t input_tokens = 0;
        double hit_rate = 0.0;

        bool is_infinite() const { return capacity_blocks == kInfiniteCapacity; }
    };

    struct RequestResult {
        std::vector<CapacityResult> capacity_results;
        uint64_t input_token_len = 0;
    };

    struct AnalysisResult {
        std::vector<CapacityResult> capacity_results;
        uint64_t request_count = 0;
        uint64_t total_input_tokens = 0;
    };

    using RequestResultCallback = std::function<void(std::size_t request_index, const RequestResult &result)>;

    // capacity_blocks contains finite capacities and optionally -1 for
    // infinity. Capacity 0 is valid. block_size_tokens is fixed for the whole
    // analysis and is used only to convert hit blocks to hit tokens.
    LiteHit(std::vector<int64_t> capacity_blocks, uint64_t block_size_tokens);

    // One call is one request boundary. block_keys must contain exactly all
    // complete blocks of the request:
    //   block_keys.size() == floor(input_token_len / block_size_tokens).
    // Prefix hits are evaluated against the request-start LRU snapshot; every
    // complete block is then committed to the shared LRU state tail-to-head
    // (reverse request order), including blocks after the first miss. With
    // prefix-chained keys a chain head is therefore always newer than its
    // resident descendants, so the global LRU victim is always a leaf,
    // matching leaf-first eviction used by production prefix caches
    // (vLLM free-queue order, SGLang radix cache).
    RequestResult ProcessRequest(const std::vector<int64_t> &block_keys, uint64_t input_token_len);

    // Offline convenience API. It resets existing state, scans the requests
    // once, and returns the same result as streaming them in order.
    // on_request is invoked immediately after each request and lets an offline
    // adapter stream per-trace results without LiteHit retaining their history.
    AnalysisResult Analyze(const std::vector<TraceRequest> &requests,
                           const RequestResultCallback &on_request = RequestResultCallback{});

    AnalysisResult GetResult() const;
    void Reset();

    const std::vector<int64_t> &capacities() const { return input_capacities_; }
    uint64_t block_size_tokens() const { return block_size_tokens_; }
    uint64_t current_unique_blocks() const { return static_cast<uint64_t>(last_positions_.size()); }
    uint64_t request_count() const { return request_count_; }
    uint64_t total_input_tokens() const { return total_input_tokens_; }

    // Coarse memory estimate for observability. It is derived from the state
    // already required by the algorithm and does not retain extra trace data.
    uint64_t memory_usage_bytes() const;

private:
    struct SnapshotEntry {
        bool is_resident = false;
        uint64_t required_capacity = 0;
    };

    void ValidateRequest(const std::vector<int64_t> &block_keys, uint64_t input_token_len) const;
    std::vector<uint64_t> EvaluateRequestPrefix(const std::vector<int64_t> &block_keys) const;
    void CommitRequest(const std::vector<int64_t> &block_keys);
    void PruneToTrackedCapacity();
    void MaybeCompactPositions();
    std::size_t FirstCapacityAtLeast(uint64_t required_capacity) const;
    std::vector<uint64_t> ResolveHitDeltas(const std::vector<uint64_t> &hit_deltas) const;
    uint64_t ReuseDistance(std::size_t previous_position) const;
    std::vector<CapacityResult> MakeCapacityResults(const std::vector<uint64_t> &hit_counts,
                                                    uint64_t input_tokens) const;

    std::vector<int64_t> input_capacities_;
    std::vector<int64_t> finite_capacities_;
    std::vector<std::size_t> input_to_internal_capacity_;
    bool has_infinite_capacity_ = false;
    uint64_t largest_finite_capacity_ = 0;
    uint64_t block_size_tokens_ = 0;

    // hit_deltas_[i] counts prefix blocks whose minimum required capacity maps
    // to internal capacity i. Prefix summing it answers every configured
    // capacity without maintaining a separate LRU chain per capacity.
    std::vector<uint64_t> hit_deltas_;

    DynamicFenwickTree fenwick_;
    std::unordered_map<int64_t, std::size_t> last_positions_;
    // Monotonic position queue used to discard the oldest active marker when
    // only finite capacities are configured. Stale entries are removed lazily
    // and rebuilt during Fenwick compaction.
    std::deque<std::pair<std::size_t, int64_t>> position_order_;

    uint64_t request_count_ = 0;
    uint64_t total_input_tokens_ = 0;
};

} // namespace kv_cache_manager
