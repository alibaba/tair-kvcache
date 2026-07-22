#include <algorithm>
#include <cstdint>
#include <list>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/liteHit/lite_hit.h"

namespace kv_cache_manager {
namespace {

class NaivePrefixLru {
public:
    NaivePrefixLru(int64_t capacity_blocks, uint64_t block_size_tokens)
        : capacity_blocks_(capacity_blocks), block_size_tokens_(block_size_tokens) {}

    uint64_t Process(const LiteHit::TraceRequest &request) {
        // Prefix hits are evaluated against the request-start content. Hits
        // never change membership, so this matches sequential evaluation.
        uint64_t prefix_hits = 0;
        for (int64_t key : request.block_keys) {
            if (positions_.find(key) == positions_.end()) {
                break;
            }
            ++prefix_hits;
        }

        // State commits tail-to-head, mirroring LiteHit's reverse-order
        // touches: the chain head ends up most recent.
        for (auto it = request.block_keys.rbegin(); it != request.block_keys.rend(); ++it) {
            const int64_t key = *it;
            auto pos = positions_.find(key);
            if (pos != positions_.end()) {
                lru_.erase(pos->second);
                positions_.erase(pos);
            }
            if (capacity_blocks_ != 0) {
                lru_.push_back(key);
                positions_[key] = std::prev(lru_.end());
            }
            if (capacity_blocks_ != LiteHit::kInfiniteCapacity &&
                lru_.size() > static_cast<std::size_t>(capacity_blocks_)) {
                positions_.erase(lru_.front());
                lru_.pop_front();
            }
        }
        cumulative_hits_ += prefix_hits;
        cumulative_input_tokens_ += request.input_token_len;
        return prefix_hits;
    }

    uint64_t cumulative_hits() const { return cumulative_hits_; }
    uint64_t cumulative_hit_tokens() const { return cumulative_hits_ * block_size_tokens_; }
    uint64_t cumulative_input_tokens() const { return cumulative_input_tokens_; }

private:
    int64_t capacity_blocks_;
    uint64_t block_size_tokens_;
    std::list<int64_t> lru_;
    std::unordered_map<int64_t, std::list<int64_t>::iterator> positions_;
    uint64_t cumulative_hits_ = 0;
    uint64_t cumulative_input_tokens_ = 0;
};

std::vector<uint64_t> HitCounts(const std::vector<LiteHit::CapacityResult> &results) {
    std::vector<uint64_t> counts;
    counts.reserve(results.size());
    for (const auto &result : results) {
        counts.push_back(result.hit_count);
    }
    return counts;
}

std::vector<uint64_t> HitTokens(const std::vector<LiteHit::CapacityResult> &results) {
    std::vector<uint64_t> tokens;
    tokens.reserve(results.size());
    for (const auto &result : results) {
        tokens.push_back(result.hit_tokens);
    }
    return tokens;
}

} // namespace

TEST(LiteHitTest, RejectsInvalidConfiguration) {
    EXPECT_THROW(LiteHit({-2}, 16), std::invalid_argument);
    EXPECT_THROW(LiteHit({1}, 0), std::invalid_argument);
    EXPECT_NO_THROW(LiteHit({0, 1, LiteHit::kInfiniteCapacity}, 16));
}

TEST(LiteHitTest, ValidatesCompleteBlockInput) {
    LiteHit lite_hit({1}, 4);
    EXPECT_THROW(lite_hit.ProcessRequest({1}, 3), std::invalid_argument);
    EXPECT_THROW(lite_hit.ProcessRequest({}, 4), std::invalid_argument);
    EXPECT_EQ(0, lite_hit.request_count());

    EXPECT_NO_THROW(lite_hit.ProcessRequest({1}, 7));
    EXPECT_EQ(1, lite_hit.request_count());
}

TEST(LiteHitTest, EmptyAndTailOnlyRequestsUseTokenDenominator) {
    LiteHit lite_hit({0, 1, LiteHit::kInfiniteCapacity}, 4);

    const auto empty = lite_hit.ProcessRequest({}, 0);
    const auto tail_only = lite_hit.ProcessRequest({}, 3);
    EXPECT_EQ((std::vector<uint64_t>{0, 0, 0}), HitCounts(empty.capacity_results));
    EXPECT_EQ((std::vector<uint64_t>{0, 0, 0}), HitTokens(tail_only.capacity_results));

    const auto result = lite_hit.GetResult();
    EXPECT_EQ(2, result.request_count);
    EXPECT_EQ(3, result.total_input_tokens);
    for (const auto &capacity : result.capacity_results) {
        EXPECT_DOUBLE_EQ(0.0, capacity.hit_rate);
    }
}

TEST(LiteHitTest, RequestStartSnapshotProducesPrefixCapacityThresholds) {
    LiteHit lite_hit({1, 2, 3, LiteHit::kInfiniteCapacity}, 4);

    // Reverse-order commit makes the chain head most recent: the request-start
    // LRU is [3, 2, 1] from LRU to MRU.
    lite_hit.ProcessRequest({1, 2, 3}, 12);
    // Snapshot required capacities are [2:2, 1:1, 3:3], therefore prefix
    // thresholds are [2, 2, 3].
    const auto result = lite_hit.ProcessRequest({2, 1, 3}, 13);

    EXPECT_EQ((std::vector<uint64_t>{0, 2, 3, 3}), HitCounts(result.capacity_results));
    EXPECT_EQ((std::vector<uint64_t>{0, 8, 12, 12}), HitTokens(result.capacity_results));
    EXPECT_DOUBLE_EQ(12.0 / 13.0, result.capacity_results[2].hit_rate);
}

TEST(LiteHitTest, BlocksAfterFirstMissStillUpdateGlobalLru) {
    LiteHit lite_hit({1, 2, LiteHit::kInfiniteCapacity}, 4);

    lite_hit.ProcessRequest({1, 2}, 8);
    const auto mixed = lite_hit.ProcessRequest({1, 3, 2}, 12);
    EXPECT_EQ((std::vector<uint64_t>{1, 1, 1}), HitCounts(mixed.capacity_results));

    // The previous request committed all of [1, 3, 2] tail-to-head, so key 2
    // was committed even though it was located after that request's first
    // miss. It is the oldest of the three, hence resident only for infinity.
    const auto next = lite_hit.ProcessRequest({2}, 5);
    EXPECT_EQ((std::vector<uint64_t>{0, 0, 1}), HitCounts(next.capacity_results));
}

TEST(LiteHitTest, ReverseCommitKeepsChainHeadMostRecent) {
    LiteHit lite_hit({1, LiteHit::kInfiniteCapacity}, 4);
    lite_hit.ProcessRequest({1, 2, 3}, 12);

    // The chain head is MRU after the commit, so it hits even at capacity 1;
    // the chain leaf is the eviction victim side and only infinity hits it.
    const auto head = lite_hit.ProcessRequest({1}, 4);
    EXPECT_EQ((std::vector<uint64_t>{1, 1}), HitCounts(head.capacity_results));
    const auto leaf = lite_hit.ProcessRequest({3}, 4);
    EXPECT_EQ((std::vector<uint64_t>{0, 1}), HitCounts(leaf.capacity_results));
}

TEST(LiteHitTest, RepeatedKeysUseOneSnapshotAndSequentialFinalState) {
    LiteHit lite_hit({1, 2, LiteHit::kInfiniteCapacity}, 4);
    lite_hit.ProcessRequest({1, 2}, 8);

    // Snapshot LRU is [2, 1] from LRU to MRU, so the leading key 2 pins the
    // prefix threshold at 2 for the whole request.
    const auto repeated = lite_hit.ProcessRequest({2, 2, 1, 2}, 16);
    EXPECT_EQ((std::vector<uint64_t>{0, 4, 4}), HitCounts(repeated.capacity_results));

    const auto next = lite_hit.ProcessRequest({2, 1}, 8);
    EXPECT_EQ((std::vector<uint64_t>{1, 2, 2}), HitCounts(next.capacity_results));
}

TEST(LiteHitTest, PreservesInputCapacityOrderAndDuplicates) {
    LiteHit lite_hit({3, 1, 3, LiteHit::kInfiniteCapacity, 0}, 4);
    lite_hit.ProcessRequest({1, 2, 3}, 12);
    const auto result = lite_hit.ProcessRequest({1, 2, 3}, 15);

    EXPECT_EQ((std::vector<uint64_t>{3, 1, 3, 3, 0}), HitCounts(result.capacity_results));
    EXPECT_EQ(3, result.capacity_results[0].capacity_blocks);
    EXPECT_EQ(1, result.capacity_results[1].capacity_blocks);
    EXPECT_EQ(LiteHit::kInfiniteCapacity, result.capacity_results[3].capacity_blocks);
}

TEST(LiteHitTest, OfflineAndStreamingResultsMatch) {
    const std::vector<int64_t> capacities = {0, 1, 2, 5, LiteHit::kInfiniteCapacity};
    const std::vector<LiteHit::TraceRequest> requests = {
        {{1, 2, 3}, 14},
        {{1, 3}, 11},
        {{4, 2, 1, 4}, 16},
        {{4}, 7},
        {{}, 3},
    };

    LiteHit offline(capacities, 4);
    std::vector<std::vector<uint64_t>> offline_request_hits;
    const auto offline_result = offline.Analyze(requests, [&](std::size_t request_index, const auto &result) {
        EXPECT_EQ(offline_request_hits.size(), request_index);
        offline_request_hits.push_back(HitCounts(result.capacity_results));
    });

    LiteHit streaming(capacities, 4);
    std::vector<std::vector<uint64_t>> streaming_request_hits;
    for (const auto &request : requests) {
        streaming_request_hits.push_back(
            HitCounts(streaming.ProcessRequest(request.block_keys, request.input_token_len).capacity_results));
    }
    const auto streaming_result = streaming.GetResult();

    EXPECT_EQ(HitCounts(offline_result.capacity_results), HitCounts(streaming_result.capacity_results));
    EXPECT_EQ(HitTokens(offline_result.capacity_results), HitTokens(streaming_result.capacity_results));
    EXPECT_EQ(offline_result.total_input_tokens, streaming_result.total_input_tokens);
    EXPECT_EQ(offline_result.request_count, streaming_result.request_count);
    EXPECT_EQ(offline_request_hits, streaming_request_hits);
}

TEST(LiteHitTest, MatchesNaiveMultiCapacityOracleOnRandomRequests) {
    constexpr uint64_t kBlockSizeTokens = 8;
    const std::vector<int64_t> capacities = {0, 1, 2, 4, 9, LiteHit::kInfiniteCapacity, 2};
    LiteHit lite_hit(capacities, kBlockSizeTokens);

    std::vector<NaivePrefixLru> oracles;
    for (int64_t capacity : capacities) {
        oracles.emplace_back(capacity, kBlockSizeTokens);
    }

    std::mt19937_64 rng(20260713);
    for (int request_index = 0; request_index < 1000; ++request_index) {
        LiteHit::TraceRequest request;
        const std::size_t block_count = rng() % 12;
        request.block_keys.reserve(block_count);
        for (std::size_t i = 0; i < block_count; ++i) {
            request.block_keys.push_back(static_cast<int64_t>(rng() % 17));
        }
        request.input_token_len = block_count * kBlockSizeTokens + rng() % kBlockSizeTokens;

        const auto request_result = lite_hit.ProcessRequest(request.block_keys, request.input_token_len);
        ASSERT_EQ(capacities.size(), request_result.capacity_results.size());
        for (std::size_t i = 0; i < capacities.size(); ++i) {
            const uint64_t expected_hits = oracles[i].Process(request);
            EXPECT_EQ(expected_hits, request_result.capacity_results[i].hit_count)
                << "request=" << request_index << " capacity=" << capacities[i];
            EXPECT_EQ(expected_hits * kBlockSizeTokens, request_result.capacity_results[i].hit_tokens);
        }
    }

    const auto cumulative = lite_hit.GetResult();
    for (std::size_t i = 0; i < capacities.size(); ++i) {
        EXPECT_EQ(oracles[i].cumulative_hits(), cumulative.capacity_results[i].hit_count);
        EXPECT_EQ(oracles[i].cumulative_hit_tokens(), cumulative.capacity_results[i].hit_tokens);
        EXPECT_EQ(oracles[i].cumulative_input_tokens(), cumulative.capacity_results[i].input_tokens);
        const double expected_rate = static_cast<double>(oracles[i].cumulative_hit_tokens()) /
                                     static_cast<double>(oracles[i].cumulative_input_tokens());
        EXPECT_DOUBLE_EQ(expected_rate, cumulative.capacity_results[i].hit_rate);
    }
}

TEST(LiteHitTest, ResetClearsLruAndCumulativeStatistics) {
    LiteHit lite_hit({2, LiteHit::kInfiniteCapacity}, 4);
    lite_hit.ProcessRequest({1, 2}, 9);
    lite_hit.ProcessRequest({1, 2}, 8);
    ASSERT_EQ(2, lite_hit.GetResult().capacity_results[0].hit_count);

    lite_hit.Reset();
    const auto result = lite_hit.GetResult();
    EXPECT_EQ(0, result.request_count);
    EXPECT_EQ(0, result.total_input_tokens);
    EXPECT_EQ(0, lite_hit.current_unique_blocks());
    EXPECT_EQ((std::vector<uint64_t>{0, 0}), HitCounts(result.capacity_results));
}

TEST(LiteHitTest, CompactsDynamicPositionsWithoutChangingResults) {
    LiteHit lite_hit({1, LiteHit::kInfiniteCapacity}, 1);
    for (int64_t i = 0; i < 20000; ++i) {
        lite_hit.ProcessRequest({i % 7}, 1);
    }

    EXPECT_EQ(7, lite_hit.current_unique_blocks());
    EXPECT_LE(lite_hit.fenwick_.size(), 2 * lite_hit.last_positions_.size() + 4096);
    EXPECT_EQ((std::vector<uint64_t>{0, 19993}), HitCounts(lite_hit.GetResult().capacity_results));
}

TEST(LiteHitTest, FiniteOnlyAnalysisDropsKeysOlderThanLargestCapacity) {
    LiteHit lite_hit({1, 3}, 1);
    for (int64_t key = 0; key < 10000; ++key) {
        lite_hit.ProcessRequest({key}, 1);
    }

    EXPECT_EQ(3, lite_hit.current_unique_blocks());
    EXPECT_LE(lite_hit.fenwick_.size(), 2 * lite_hit.last_positions_.size() + 4096);

    const auto oldest_resident = lite_hit.ProcessRequest({9997}, 1);
    EXPECT_EQ((std::vector<uint64_t>{0, 1}), HitCounts(oldest_resident.capacity_results));
    const auto evicted = lite_hit.ProcessRequest({1}, 1);
    EXPECT_EQ((std::vector<uint64_t>{0, 0}), HitCounts(evicted.capacity_results));
}

} // namespace kv_cache_manager
