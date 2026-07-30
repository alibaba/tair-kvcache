#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"
#include "kv_cache_manager/optimizer/liteHit/lite_hit.h"

namespace kv_cache_manager {

namespace {

// Naive joint LRU+TTL reference. Recency is a most-recent-first list; the
// hit prefix under (capacity C, fixed TTL) is the longest prefix whose
// blocks are all seen, alive (age strictly below TTL) and within the top C
// of the recency stack on the request-start snapshot. Commits touch
// tail-to-head (chain head most recent) and refresh last_access for every
// block, matching the LiteHit contract and the online TTL wrapper.
class NaiveLruTtlOracle {
public:
    uint64_t Evaluate(const std::vector<int64_t> &keys, int64_t now_ns, uint64_t ttl_ns, uint64_t capacity) const {
        uint64_t hits = 0;
        for (int64_t key : keys) {
            const auto it = last_access_.find(key);
            if (it == last_access_.end()) {
                break;
            }
            const uint64_t age = it->second < now_ns ? static_cast<uint64_t>(now_ns - it->second) : 0;
            if (age >= ttl_ns) {
                break;
            }
            const auto pos = std::find(recency_.begin(), recency_.end(), key);
            const uint64_t rank = static_cast<uint64_t>(pos - recency_.begin()) + 1;
            if (rank > capacity) {
                break;
            }
            ++hits;
        }
        return hits;
    }

    void Commit(const std::vector<int64_t> &keys, int64_t now_ns) {
        for (std::size_t i = keys.size(); i > 0; --i) {
            const int64_t key = keys[i - 1];
            const auto pos = std::find(recency_.begin(), recency_.end(), key);
            if (pos != recency_.end()) {
                recency_.erase(pos);
            }
            recency_.insert(recency_.begin(), key);
            last_access_[key] = now_ns;
        }
    }

private:
    std::vector<int64_t> recency_;
    std::unordered_map<int64_t, int64_t> last_access_;
};

} // namespace

class LiteHitTtlTest : public TESTBASE {};

TEST_F(LiteHitTtlTest, StrictDeadlineBoundary) {
    LiteHit core(2000);
    EXPECT_TRUE(core.ProcessRequest({1, 2, 3}, 1000).hit_curve.empty()); // cold
    // Ages 1999 < 2000: alive, normal LRU curve.
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 3}}), core.ProcessRequest({1, 2, 3}, 2999).hit_curve);
    // Ages exactly 2000: deadline reached, miss (matches the online wrapper).
    EXPECT_TRUE(core.ProcessRequest({1, 2, 3}, 4999).hit_curve.empty());
    // The expired access still refreshed last_access.
    EXPECT_EQ(3, HitCurveProjector::ProjectInfinite(core.ProcessRequest({1, 2, 3}, 5000)));
}

TEST_F(LiteHitTtlTest, ExpiredBlockStopsThePrefixLikeAColdOne) {
    LiteHit core(2500);
    core.ProcessRequest({10, 11}, 1000);
    core.ProcessRequest({10, 11, 12}, 3000); // all refreshed at 3000
    // At 5600 every block is 2600 old: dead despite being LRU-resident.
    EXPECT_TRUE(core.ProcessRequest({10, 11, 12}, 5600).hit_curve.empty());
}

TEST_F(LiteHitTtlTest, TtlZeroIsPureLru) {
    LiteHit core; // default: no TTL, timestamps ignored
    core.ProcessRequest({1, 2, 3}, 1000);
    const RequestFact fact = core.ProcessRequest({1, 2, 3}, 1000000000000);
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 3}}), fact.hit_curve);
}

TEST_F(LiteHitTtlTest, ResetClearsTtlState) {
    LiteHit core(1000000);
    core.ProcessRequest({1, 2}, 1000);
    EXPECT_EQ(2, core.current_unique_blocks());
    core.Reset();
    EXPECT_EQ(0, core.current_unique_blocks());
    EXPECT_TRUE(core.ProcessRequest({1, 2}, 2000).hit_curve.empty());
}

TEST_F(LiteHitTtlTest, RandomizedMatchesNaiveJointOracle) {
    std::mt19937_64 rng(20260730);
    std::uniform_int_distribution<int64_t> base_dist(0, 12);
    std::uniform_int_distribution<int> len_dist(1, 6);
    std::uniform_int_distribution<int64_t> delta_dist(0, 100);
    const std::vector<uint64_t> ttls = {1, 17, 50, 120, 1000};
    const std::vector<uint64_t> capacities = {1, 2, 3, 5, 8, 1000000};

    // One core per fixed TTL; commits are TTL-independent, so every core and
    // the single oracle share the same recency and last-access evolution.
    std::vector<std::unique_ptr<LiteHit>> cores;
    for (uint64_t ttl : ttls) {
        cores.push_back(std::make_unique<LiteHit>(ttl));
    }
    NaiveLruTtlOracle oracle;
    int64_t now = 0;
    for (int step = 0; step < 2000; ++step) {
        now += delta_dist(rng);
        // Prefix-chained keys: requests with the same base share prefixes,
        // key base*100+i identifies the whole chain prefix.
        const int64_t base = base_dist(rng);
        const int len = len_dist(rng);
        std::vector<int64_t> keys;
        keys.reserve(len);
        for (int i = 1; i <= len; ++i) {
            keys.push_back(base * 100 + i);
        }

        for (std::size_t t = 0; t < ttls.size(); ++t) {
            const RequestFact fact = cores[t]->ProcessRequest(keys, now);
            for (uint64_t capacity : capacities) {
                ASSERT_EQ(oracle.Evaluate(keys, now, ttls[t], capacity),
                          HitCurveProjector::ProjectBlocks(fact, capacity))
                    << "step " << step << " ttl " << ttls[t] << " capacity " << capacity;
            }
        }
        oracle.Commit(keys, now);
    }
}

} // namespace kv_cache_manager
