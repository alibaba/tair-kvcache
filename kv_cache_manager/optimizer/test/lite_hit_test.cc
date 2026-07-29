#include <algorithm>
#include <cstdint>
#include <list>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"
#include "kv_cache_manager/optimizer/liteHit/lite_hit.h"
#include "kv_cache_manager/optimizer/liteHit/request_preprocess.h"

namespace kv_cache_manager {
namespace {

constexpr int64_t kInfinite = -1;

// Naive single-capacity prefix LRU oracle with the same semantics as
// LiteHit: request-start snapshot evaluation + tail-to-head commit.
class NaivePrefixLru {
public:
    explicit NaivePrefixLru(int64_t capacity_blocks) : capacity_blocks_(capacity_blocks) {}

    uint64_t Process(const std::vector<int64_t> &block_keys) {
        uint64_t prefix_hits = 0;
        for (int64_t key : block_keys) {
            if (positions_.find(key) == positions_.end()) {
                break;
            }
            ++prefix_hits;
        }

        for (auto it = block_keys.rbegin(); it != block_keys.rend(); ++it) {
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
            if (capacity_blocks_ != kInfinite && lru_.size() > static_cast<std::size_t>(capacity_blocks_)) {
                positions_.erase(lru_.front());
                lru_.pop_front();
            }
        }
        cumulative_hits_ += prefix_hits;
        return prefix_hits;
    }

    uint64_t cumulative_hits() const { return cumulative_hits_; }

private:
    int64_t capacity_blocks_;
    std::list<int64_t> lru_;
    std::unordered_map<int64_t, std::list<int64_t>::iterator> positions_;
    uint64_t cumulative_hits_ = 0;
};

uint64_t Project(const RequestFact &fact, int64_t capacity_blocks) {
    if (capacity_blocks == kInfinite) {
        return HitCurveProjector::ProjectInfinite(fact);
    }
    return HitCurveProjector::ProjectBlocks(fact, static_cast<uint64_t>(capacity_blocks));
}

} // namespace

TEST(HitCurveProjectorTest, EmptyCurveHitsNothing) {
    const RequestFact fact;
    EXPECT_EQ(0, HitCurveProjector::ProjectBlocks(fact, 0));
    EXPECT_EQ(0, HitCurveProjector::ProjectBlocks(fact, 1000000));
    EXPECT_EQ(0, HitCurveProjector::ProjectInfinite(fact));
}

TEST(HitCurveProjectorTest, ProjectsSegmentBoundaries) {
    const RequestFact fact{{HitCurveSegment{1, 2}, HitCurveSegment{4, 1}}};
    EXPECT_EQ(0, HitCurveProjector::ProjectBlocks(fact, 0));
    EXPECT_EQ(1, HitCurveProjector::ProjectBlocks(fact, 1));
    EXPECT_EQ(2, HitCurveProjector::ProjectBlocks(fact, 2));
    EXPECT_EQ(2, HitCurveProjector::ProjectBlocks(fact, 3));
    EXPECT_EQ(3, HitCurveProjector::ProjectBlocks(fact, 4));
    EXPECT_EQ(3, HitCurveProjector::ProjectBlocks(fact, 100));
    EXPECT_EQ(3, HitCurveProjector::ProjectInfinite(fact));
}

TEST(HitCurveProjectorTest, ByteProjectionFloorsCapacity) {
    const RequestFact fact{{HitCurveSegment{2, 3}}};
    constexpr uint64_t kBlockBytes = 4096;
    EXPECT_EQ(0, HitCurveProjector::ProjectBytes(fact, 2 * kBlockBytes - 1, kBlockBytes));
    EXPECT_EQ(1, HitCurveProjector::ProjectBytes(fact, 2 * kBlockBytes, kBlockBytes));
    EXPECT_EQ(2, HitCurveProjector::ProjectBytes(fact, 4 * kBlockBytes - 1, kBlockBytes));
    EXPECT_EQ(3, HitCurveProjector::ProjectBytes(fact, 4 * kBlockBytes, kBlockBytes));
}

TEST(LiteHitTest, EmptyRequestIsNoOp) {
    LiteHit lite_hit;
    const RequestFact fact = lite_hit.ProcessRequest({});
    EXPECT_TRUE(fact.hit_curve.empty());
    EXPECT_EQ(0, lite_hit.current_unique_blocks());
}

TEST(LiteHitTest, ColdRequestProducesEmptyCurveButCommits) {
    LiteHit lite_hit;
    const RequestFact cold = lite_hit.ProcessRequest({1, 2, 3});
    EXPECT_TRUE(cold.hit_curve.empty());
    EXPECT_EQ(3, lite_hit.current_unique_blocks());
}

TEST(LiteHitTest, ContiguousChainReplayIsOneSegment) {
    LiteHit lite_hit;
    lite_hit.ProcessRequest({1, 2, 3});
    // Reverse commit puts the chain contiguously: depths 1, 2, 3.
    const RequestFact fact = lite_hit.ProcessRequest({1, 2, 3});
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 3}}), fact.hit_curve);
}

TEST(LiteHitTest, InterleavingChainBreaksSegments) {
    LiteHit lite_hit;
    lite_hit.ProcessRequest({1, 2, 3});
    // Fork [1, 2, 9] commits 9 between 2 and 3: LRU becomes [1, 2, 9, 3].
    lite_hit.ProcessRequest({1, 2, 9});
    const RequestFact fact = lite_hit.ProcessRequest({1, 2, 3});
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 2}, {4, 1}}), fact.hit_curve);
}

TEST(LiteHitTest, CurveStopsAtFirstColdKey) {
    LiteHit lite_hit;
    lite_hit.ProcessRequest({1, 2, 3});
    const RequestFact fact = lite_hit.ProcessRequest({1, 7, 3});
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 1}}), fact.hit_curve);
}

TEST(LiteHitTest, BlocksAfterFirstMissStillUpdateGlobalLru) {
    LiteHit lite_hit;
    lite_hit.ProcessRequest({1, 2});
    const RequestFact mixed = lite_hit.ProcessRequest({1, 3, 2});
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 1}}), mixed.hit_curve);

    // Key 2 was committed even though it was after that request's first miss.
    // It is the oldest of the three, so it needs the full capacity of 3.
    const RequestFact next = lite_hit.ProcessRequest({2});
    EXPECT_EQ((std::vector<HitCurveSegment>{{3, 1}}), next.hit_curve);
}

TEST(LiteHitTest, ReverseCommitKeepsChainHeadMostRecent) {
    LiteHit lite_hit;
    lite_hit.ProcessRequest({1, 2, 3});

    const RequestFact head = lite_hit.ProcessRequest({1});
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 1}}), head.hit_curve);
    // After the head query the LRU is [1, 2, 3] again; the leaf needs full
    // capacity.
    const RequestFact leaf = lite_hit.ProcessRequest({3});
    EXPECT_EQ((std::vector<HitCurveSegment>{{3, 1}}), leaf.hit_curve);
}

TEST(LiteHitTest, RepeatedKeysEncodeMonotonicallyAndNeverOptimistically) {
    LiteHit lite_hit;
    lite_hit.ProcessRequest({1, 2});
    // Snapshot depths: 1 -> 1, 2 -> 2. Request [2, 2, 1, 2] violates the
    // prefix-hash contract; thresholds before the guard are [2, 2, 2, 2].
    // The monotonic guard encodes strictly increasing thresholds [2,3,4,5],
    // which is pessimistic (never optimistic) versus the naive oracle.
    const RequestFact fact = lite_hit.ProcessRequest({2, 2, 1, 2});
    EXPECT_EQ((std::vector<HitCurveSegment>{{2, 4}}), fact.hit_curve);

    // Sequential final state is [2, 1] with 2 most recent.
    const RequestFact next = lite_hit.ProcessRequest({2, 1});
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 2}}), next.hit_curve);
}

TEST(LiteHitTest, ResetClearsLruState) {
    LiteHit lite_hit;
    lite_hit.ProcessRequest({1, 2});
    ASSERT_EQ(2, lite_hit.current_unique_blocks());

    lite_hit.Reset();
    EXPECT_EQ(0, lite_hit.current_unique_blocks());
    EXPECT_TRUE(lite_hit.ProcessRequest({1, 2}).hit_curve.empty());
}

TEST(LiteHitTest, MatchesNaiveMultiCapacityOracleOnRandomContractTraces) {
    // Generate contract-valid traces: every request is a prefix chain from a
    // random branching tree, keyed by rolling prefix hash.
    LiteHit lite_hit;
    const std::vector<int64_t> capacities = {0, 1, 2, 4, 9, kInfinite};
    std::vector<NaivePrefixLru> oracles;
    for (int64_t capacity : capacities) {
        oracles.emplace_back(capacity);
    }
    std::vector<uint64_t> cumulative_hits(capacities.size(), 0);

    std::mt19937_64 rng(20260722);
    for (int request_index = 0; request_index < 1500; ++request_index) {
        const std::size_t depth = 1 + rng() % 10;
        std::vector<int64_t> raw_keys;
        raw_keys.reserve(depth);
        for (std::size_t level = 0; level < depth; ++level) {
            // Small alphabet per level yields heavy prefix sharing.
            raw_keys.push_back(static_cast<int64_t>(rng() % 3));
        }
        const std::vector<int64_t> block_keys = ApplyPrefixHash(raw_keys);

        const RequestFact fact = lite_hit.ProcessRequest(block_keys);
        for (std::size_t i = 0; i < capacities.size(); ++i) {
            const uint64_t expected_hits = oracles[i].Process(block_keys);
            const uint64_t projected = Project(fact, capacities[i]);
            EXPECT_EQ(expected_hits, projected) << "request=" << request_index << " capacity=" << capacities[i];
            cumulative_hits[i] += projected;
        }
    }

    for (std::size_t i = 0; i < capacities.size(); ++i) {
        EXPECT_EQ(oracles[i].cumulative_hits(), cumulative_hits[i]);
    }
}

TEST(LiteHitTest, ProjectionIsNeverOptimisticOnNonContractTraces) {
    LiteHit lite_hit;
    const std::vector<int64_t> capacities = {0, 1, 2, 4, 9, kInfinite};
    std::vector<NaivePrefixLru> oracles;
    for (int64_t capacity : capacities) {
        oracles.emplace_back(capacity);
    }

    std::mt19937_64 rng(20260713);
    for (int request_index = 0; request_index < 1000; ++request_index) {
        const std::size_t block_count = rng() % 12;
        std::vector<int64_t> block_keys;
        block_keys.reserve(block_count);
        for (std::size_t i = 0; i < block_count; ++i) {
            block_keys.push_back(static_cast<int64_t>(rng() % 17));
        }

        const RequestFact fact = lite_hit.ProcessRequest(block_keys);
        for (std::size_t i = 0; i < capacities.size(); ++i) {
            const uint64_t expected_hits = oracles[i].Process(block_keys);
            const uint64_t projected = Project(fact, capacities[i]);
            EXPECT_LE(projected, expected_hits) << "request=" << request_index << " capacity=" << capacities[i];
            if (capacities[i] == kInfinite) {
                // Infinite capacity is exact even for duplicate keys.
                EXPECT_EQ(expected_hits, projected);
            }
        }
    }
}

TEST(LiteHitTest, CompactsDynamicPositionsWithoutChangingResults) {
    LiteHit lite_hit;
    for (int64_t i = 0; i < 20000; ++i) {
        lite_hit.ProcessRequest({i % 7});
    }

    EXPECT_EQ(7, lite_hit.current_unique_blocks());
    EXPECT_LE(lite_hit.fenwick_.size(), 2 * lite_hit.last_positions_.size() + 4096);
    const RequestFact fact = lite_hit.ProcessRequest({(20000 - 7) % 7});
    EXPECT_EQ((std::vector<HitCurveSegment>{{7, 1}}), fact.hit_curve);
}

TEST(LiteHitTest, RetainsAllActiveKeysWithoutCapacityPruning) {
    LiteHit lite_hit;
    for (int64_t key = 0; key < 10000; ++key) {
        lite_hit.ProcessRequest({key});
    }
    EXPECT_EQ(10000, lite_hit.current_unique_blocks());

    const RequestFact oldest = lite_hit.ProcessRequest({0});
    EXPECT_EQ((std::vector<HitCurveSegment>{{10000, 1}}), oldest.hit_curve);
}

TEST(RequestPreprocessTest, PrefixHashMatchesPythonGoldenVectors) {
    // Golden vectors produced by
    // optimizer/tools/trace_converter/utils/prefix_hash.py::apply_prefix_hash.
    EXPECT_EQ((std::vector<int64_t>{-7046029254386353130LL, -8366447756632839738LL, 8024461590772477417LL}),
              ApplyPrefixHash({1, 2, 3}));
    EXPECT_EQ((std::vector<int64_t>{-7046029254386353136LL, -8366447756632880699LL}), ApplyPrefixHash({-5, 7}));
    EXPECT_EQ((std::vector<int64_t>{-7046029254386353089LL}), ApplyPrefixHash({42}));
    EXPECT_EQ(-7046029254386353131LL, PrefixHashNext(0, 0));
    EXPECT_TRUE(ApplyPrefixHash({}).empty());
}

TEST(RequestPreprocessTest, ValidatesLengthContract) {
    EXPECT_THROW(NormalizeRequest({1}, 3, 4, false), std::invalid_argument);
    EXPECT_THROW(NormalizeRequest({}, 4, 4, false), std::invalid_argument);
    EXPECT_THROW(NormalizeRequest({1}, -1, 4, false), std::invalid_argument);
    EXPECT_THROW(NormalizeRequest({1}, 4, 0, false), std::invalid_argument);

    const NormalizedRequest tail = NormalizeRequest({1}, 7, 4, false);
    EXPECT_EQ(7, tail.input_token_len);
    EXPECT_EQ((std::vector<int64_t>{1}), tail.block_keys);

    const NormalizedRequest empty = NormalizeRequest({}, 0, 4, false);
    EXPECT_EQ(0, empty.input_token_len);
    EXPECT_TRUE(empty.block_keys.empty());
}

TEST(RequestPreprocessTest, DerivesMissingInputLengthFromKeys) {
    const NormalizedRequest derived = NormalizeRequest({1, 2, 3}, 0, 4, false);
    EXPECT_EQ(12, derived.input_token_len);
    EXPECT_EQ((std::vector<int64_t>{1, 2, 3}), derived.block_keys);
}

TEST(RequestPreprocessTest, AppliesRollingPrefixHashWhenEnabled) {
    const NormalizedRequest hashed = NormalizeRequest({1, 2, 3}, 13, 4, true);
    EXPECT_EQ(13, hashed.input_token_len);
    EXPECT_EQ(ApplyPrefixHash({1, 2, 3}), hashed.block_keys);

    const NormalizedRequest passthrough = NormalizeRequest({1, 2, 3}, 13, 4, false);
    EXPECT_EQ((std::vector<int64_t>{1, 2, 3}), passthrough.block_keys);
}

TEST(RequestPreprocessTest, ReblocksToCoarserGranularity) {
    // Trace granularity 4, analysis granularity 8: keep every 2nd chained
    // key, drop the incomplete tail; the token denominator is untouched.
    const NormalizedRequest coarse = NormalizeRequest({1, 2, 3, 4, 5}, 21, 8, false, 4);
    EXPECT_EQ(21, coarse.input_token_len);
    EXPECT_EQ((std::vector<int64_t>{2, 4}), coarse.block_keys);

    // stride 1 (explicit trace granularity == analysis granularity).
    const NormalizedRequest same = NormalizeRequest({1, 2, 3}, 13, 4, false, 4);
    EXPECT_EQ((std::vector<int64_t>{1, 2, 3}), same.block_keys);

    // Raw per-block hashes are chained at trace granularity first, then
    // sampled, so the coarse keys still encode their whole prefix.
    const NormalizedRequest hashed = NormalizeRequest({1, 2, 3, 4}, 16, 8, true, 4);
    const std::vector<int64_t> chained = ApplyPrefixHash({1, 2, 3, 4});
    EXPECT_EQ((std::vector<int64_t>{chained[1], chained[3]}), hashed.block_keys);

    // Fewer trace blocks than one coarse block yields an empty request.
    const NormalizedRequest tiny = NormalizeRequest({1}, 4, 8, false, 4);
    EXPECT_EQ(4, tiny.input_token_len);
    EXPECT_TRUE(tiny.block_keys.empty());

    // Length contract is checked at trace granularity.
    EXPECT_THROW(NormalizeRequest({1, 2}, 21, 8, false, 4), std::invalid_argument);
    // Coarsening only: analysis granularity must be an exact multiple.
    EXPECT_THROW(NormalizeRequest({1, 2}, 8, 6, false, 4), std::invalid_argument);
    EXPECT_THROW(NormalizeRequest({1}, 8, 4, false, 8), std::invalid_argument);
}

} // namespace kv_cache_manager
