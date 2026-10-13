#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"
#include "kv_cache_manager/optimizer/liteHit/lite_hit_ttl.h"

namespace kv_cache_manager {

namespace {

// Naive joint LRU+TTL reference. Recency is a most-recent-first list; the
// hit prefix under (capacity C, fixed TTL) is the longest prefix whose
// blocks are all seen, alive (age strictly below TTL) and within the top C
// of the recency stack on the request-start snapshot. Commits touch
// tail-to-head (chain head most recent) and refresh last_access for every
// block, matching the LiteHit contract and the TTL decorator.
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

    // Keys whose age at now_ns is strictly below the TTL (0 = no TTL).
    uint64_t AliveCount(int64_t now_ns, uint64_t ttl_ns) const {
        if (ttl_ns == 0) {
            return static_cast<uint64_t>(last_access_.size());
        }
        uint64_t alive = 0;
        for (const auto &[key, last_ns] : last_access_) {
            const uint64_t age = last_ns < now_ns ? static_cast<uint64_t>(now_ns - last_ns) : 0;
            if (age < ttl_ns) {
                ++alive;
            }
        }
        return alive;
    }

private:
    std::vector<int64_t> recency_;
    std::unordered_map<int64_t, int64_t> last_access_;
};

} // namespace

class LiteHitTtlTest : public TESTBASE {};

TEST_F(LiteHitTtlTest, StrictDeadlineBoundary) {
    TtlLiteHit core(2000);
    EXPECT_TRUE(core.ProcessFullRequest({1, 2, 3}, 1000).hit_curve.empty()); // cold
    // Ages 1999 < 2000: alive, normal LRU curve.
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 3}}), core.ProcessFullRequest({1, 2, 3}, 2999).hit_curve);
    // Ages exactly 2000: deadline reached, miss (matches the online wrapper).
    EXPECT_TRUE(core.ProcessFullRequest({1, 2, 3}, 4999).hit_curve.empty());
    // The expired access still refreshed last_access.
    EXPECT_EQ(3, HitCurveProjector::ProjectFullInfinite(core.ProcessFullRequest({1, 2, 3}, 5000)));
}

TEST_F(LiteHitTtlTest, ExpiredBlockStopsThePrefixLikeAColdOne) {
    TtlLiteHit core(2500);
    core.ProcessFullRequest({10, 11}, 1000);
    core.ProcessFullRequest({10, 11, 12}, 3000); // all refreshed at 3000
    // At 5600 every block is 2600 old: dead despite being LRU-resident.
    EXPECT_TRUE(core.ProcessFullRequest({10, 11, 12}, 5600).hit_curve.empty());
}

TEST_F(LiteHitTtlTest, TtlZeroIsPureLru) {
    TtlLiteHit core; // default: no TTL, timestamps ignored
    core.ProcessFullRequest({1, 2, 3}, 1000);
    const FullRequestFact fact = core.ProcessFullRequest({1, 2, 3}, 1000000000000);
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 3}}), fact.hit_curve);
}

TEST_F(LiteHitTtlTest, DisabledDecoratorMatchesBareLinearCore) {
    LiteHit::CacheObjectConfig object_config;
    object_config.full_charge_bytes = 10;
    object_config.linear_charge_bytes = 2;
    object_config.linear_step_blocks = 2;
    LiteHit bare(object_config);
    TtlLiteHit decorated(object_config, /*ttl_ns=*/0);

    const std::vector<std::vector<int64_t>> requests = {{1, 2, 3}, {1, 2}, {1, 2, 3}, {4, 5}};
    int64_t now_ns = 1000;
    for (const auto &request : requests) {
        EXPECT_EQ(bare.ProcessRequest(request).points, decorated.ProcessRequest(request, now_ns).points);
        EXPECT_EQ(bare.current_unique_blocks(), decorated.current_unique_blocks());
        EXPECT_EQ(bare.resident_bytes(), decorated.resident_bytes());
        now_ns += 1000000;
    }
    EXPECT_EQ(0, decorated.ttl_expired_blocks());
}

TEST_F(LiteHitTtlTest, ResetClearsTtlState) {
    TtlLiteHit core(1000000);
    core.ProcessFullRequest({1, 2}, 1000);
    EXPECT_EQ(2, core.current_unique_blocks());
    core.Reset();
    EXPECT_EQ(0, core.current_unique_blocks());
    EXPECT_TRUE(core.ProcessFullRequest({1, 2}, 2000).hit_curve.empty());
}

TEST_F(LiteHitTtlTest, UniqueBlocksExcludeExpired) {
    TtlLiteHit core(1000);
    core.ProcessFullRequest({1, 2, 3}, 0);
    EXPECT_EQ(3, core.current_unique_blocks());
    // All three expire; only the new key is alive even though the expired
    // table entries linger until compaction.
    core.ProcessFullRequest({100}, 5000);
    EXPECT_EQ(1, core.current_unique_blocks());
}

TEST_F(LiteHitTtlTest, CompactionDropsExpiredEntries) {
    TtlLiteHit core(1000);
    for (int64_t r = 0; r < 600; ++r) {
        std::vector<int64_t> keys;
        keys.reserve(10);
        for (int64_t i = 0; i < 10; ++i) {
            keys.push_back(1000000 + r * 100 + i);
        }
        core.ProcessFullRequest(keys, 0);
    }
    EXPECT_EQ(6000, core.current_unique_blocks());
    // Every block expires; the next request pushes the state past the
    // compaction threshold and the dead entries are dropped, not carried.
    core.ProcessFullRequest({1, 2, 3}, 5000);
    EXPECT_EQ(3, core.current_unique_blocks());
    EXPECT_EQ(3u, core.core_.pool_.full_positions_.size());
    // The bucket array shrinks with the entries instead of staying at the
    // 6000-key high-water mark.
    EXPECT_LT(core.core_.pool_.full_positions_.bucket_count(), 1000u);
    // Semantics survive the cleanup: dropped keys stay cold, alive ones hit
    // (the re-committed cold key occupies MRU, shifting thresholds by one).
    EXPECT_TRUE(core.ProcessFullRequest({1000000}, 5001).hit_curve.empty());
    EXPECT_EQ((std::vector<HitCurveSegment>{{2, 3}}), core.ProcessFullRequest({1, 2, 3}, 5001).hit_curve);
}

TEST_F(LiteHitTtlTest, CompactionCollapsesEmptyEpochs) {
    TtlLiteHit core(1000000000); // TTL far beyond the replayed horizon
    // A single hot key with a distinct timestamp per request: every commit
    // moves the marker and leaves the previous epoch's range empty. Without
    // the compaction dedupe the deque would hold one epoch per request.
    for (int64_t t = 1; t <= 10000; ++t) {
        core.ProcessFullRequest({7}, t);
    }
    EXPECT_LE(core.ttl_state_.position_epochs_.size(), 4200u);
    EXPECT_EQ(1, core.current_unique_blocks());
    // Exactness survives the collapse: alive within TTL, dead at the strict
    // deadline measured from the true last access.
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 1}}), core.ProcessFullRequest({7}, 10001).hit_curve);
    EXPECT_TRUE(core.ProcessFullRequest({7}, 10001 + 1000000000).hit_curve.empty());
}

TEST_F(LiteHitTtlTest, AdvanceTimeRefreshesUniqueCount) {
    TtlLiteHit core(1000);
    core.ProcessFullRequest({1, 2}, 0);
    EXPECT_EQ(2, core.current_unique_blocks());
    core.AdvanceTime(999); // ages 999 < 1000: still alive
    EXPECT_EQ(2, core.current_unique_blocks());
    core.AdvanceTime(1000); // strict deadline: dead without any request
    EXPECT_EQ(0, core.current_unique_blocks());
    // Re-access after the observational advance still revives normally.
    core.ProcessFullRequest({1}, 1000);
    EXPECT_EQ(1, core.current_unique_blocks());

    TtlLiteHit pure_lru;
    pure_lru.ProcessFullRequest({1, 2}, 0);
    pure_lru.AdvanceTime(1000000000); // no TTL: a pure no-op
    EXPECT_EQ(2, pure_lru.current_unique_blocks());
}

TEST_F(LiteHitTtlTest, CountsTtlExpiredBlocks) {
    TtlLiteHit core(1000);
    core.ProcessFullRequest({1, 2, 3}, 0);
    EXPECT_EQ(0, core.ttl_expired_blocks());
    // All three reached the deadline; key 1 revives after being counted.
    core.ProcessFullRequest({1}, 2000);
    EXPECT_EQ(3, core.ttl_expired_blocks());
    // Repeated advances must not double count the already-swept markers.
    core.AdvanceTime(2500);
    EXPECT_EQ(3, core.ttl_expired_blocks());
    // The revived key expires again and counts again, via AdvanceTime alone.
    core.AdvanceTime(3000);
    EXPECT_EQ(4, core.ttl_expired_blocks());
    core.Reset();
    EXPECT_EQ(0, core.ttl_expired_blocks());
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
    std::vector<std::unique_ptr<TtlLiteHit>> cores;
    for (uint64_t ttl : ttls) {
        cores.push_back(std::make_unique<TtlLiteHit>(ttl));
    }
    NaiveLruTtlOracle oracle;
    // Naive harvest model per TTL: a tracked key leaves the alive set and
    // counts once when its age reaches the TTL at a step boundary; a commit
    // re-adds it, so a revived key counts again on its next expiry.
    std::unordered_map<int64_t, int64_t> naive_last_access;
    std::vector<std::unordered_set<int64_t>> naive_alive(ttls.size());
    std::vector<uint64_t> naive_expired(ttls.size(), 0);
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
            for (auto it = naive_alive[t].begin(); it != naive_alive[t].end();) {
                const int64_t last_ns = naive_last_access.at(*it);
                const uint64_t age = last_ns < now ? static_cast<uint64_t>(now - last_ns) : 0;
                if (age >= ttls[t]) {
                    it = naive_alive[t].erase(it);
                    ++naive_expired[t];
                } else {
                    ++it;
                }
            }
            const FullRequestFact fact = cores[t]->ProcessFullRequest(keys, now);
            ASSERT_EQ(naive_expired[t], cores[t]->ttl_expired_blocks()) << "step " << step << " ttl " << ttls[t];
            for (uint64_t capacity : capacities) {
                ASSERT_EQ(oracle.Evaluate(keys, now, ttls[t], capacity),
                          HitCurveProjector::ProjectFullBlocks(fact, capacity))
                    << "step " << step << " ttl " << ttls[t] << " capacity " << capacity;
            }
            naive_alive[t].insert(keys.begin(), keys.end());
        }
        for (int64_t key : keys) {
            naive_last_access[key] = now;
        }
        oracle.Commit(keys, now);
        for (std::size_t t = 0; t < ttls.size(); ++t) {
            ASSERT_EQ(oracle.AliveCount(now, ttls[t]), cores[t]->current_unique_blocks())
                << "step " << step << " ttl " << ttls[t];
        }
    }
}

} // namespace kv_cache_manager
