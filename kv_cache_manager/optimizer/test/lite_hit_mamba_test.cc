#include <cstdint>
#include <list>
#include <random>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"
#include "kv_cache_manager/optimizer/liteHit/lite_hit_mamba.h"

namespace kv_cache_manager {

namespace {

// Naive reference of the Mamba recovery semantics: unbounded recency lists
// with per-type charges; residency for a capacity is decided by
// required-bytes thresholds (Mattson inclusion), evaluated on the
// request-start snapshot; the commit set is fixed and tail-to-head.
class NaiveMambaOracle {
public:
    NaiveMambaOracle(const LiteHitMamba::Config &config) : config_(config) {}

    // Hits of `keys` for every capacity in `capacities`, then commit.
    std::vector<uint64_t> ProcessRequest(const std::vector<int64_t> &keys, const std::vector<uint64_t> &capacities) {
        std::vector<uint64_t> hits;
        hits.reserve(capacities.size());
        for (const uint64_t capacity : capacities) {
            hits.push_back(EvaluateSnapshot(keys, capacity));
        }
        Commit(keys);
        return hits;
    }

private:
    struct Entry {
        CacheObjectType type;
        int64_t key;
    };
    using Recency = std::list<Entry>; // front = most recent

    bool IsCheckpoint(std::size_t position, std::size_t total) const {
        return (position + 1) % config_.step_blocks == 0 || position == total - 1;
    }

    uint64_t Charge(CacheObjectType type) const {
        return type == CacheObjectType::kFull ? config_.full_charge_bytes : config_.mamba_charge_bytes;
    }

    bool RequiredBytes(const Recency &list, CacheObjectType type, int64_t key, uint64_t &required) const {
        uint64_t newer = 0;
        for (const Entry &entry : list) {
            if (entry.type == type && entry.key == key) {
                required = newer + Charge(type);
                return true;
            }
            newer += Charge(entry.type);
        }
        return false;
    }

    uint64_t EvaluateSnapshot(const std::vector<int64_t> &keys, uint64_t total_capacity) const {
        std::size_t covered = 0;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            uint64_t required = 0;
            if (!RequiredBytes(recency_, CacheObjectType::kFull, keys[i], required) || required > total_capacity) {
                break;
            }
            covered = i + 1;
        }
        uint64_t best = 0;
        for (std::size_t p = 0; p < covered; ++p) {
            if (!IsCheckpoint(p, keys.size())) {
                continue;
            }
            uint64_t required = 0;
            if (RequiredBytes(recency_, CacheObjectType::kMamba, keys[p], required) && required <= total_capacity) {
                best = p + 1;
            }
        }
        return best;
    }

    void Touch(CacheObjectType type, int64_t key) {
        recency_.remove_if([&](const Entry &entry) { return entry.type == type && entry.key == key; });
        recency_.push_front({type, key});
    }

    void Commit(const std::vector<int64_t> &keys) {
        for (std::size_t i = keys.size(); i > 0; --i) {
            const std::size_t position = i - 1;
            if (IsCheckpoint(position, keys.size())) {
                Touch(CacheObjectType::kMamba, keys[position]);
            }
            Touch(CacheObjectType::kFull, keys[position]);
        }
    }

    LiteHitMamba::Config config_;
    Recency recency_;
};

// Generates prefix-chained request key sequences: each request is a path of
// a random trie, so equal keys imply equal prefixes.
class ChainTraceGenerator {
public:
    explicit ChainTraceGenerator(uint32_t seed) : rng_(seed) {}

    std::vector<int64_t> NextRequest(std::size_t max_len) {
        std::vector<int64_t> keys;
        if (!history_.empty() && std::uniform_int_distribution<int>(0, 2)(rng_) != 0) {
            const auto &base = history_[std::uniform_int_distribution<std::size_t>(0, history_.size() - 1)(rng_)];
            const std::size_t take = std::uniform_int_distribution<std::size_t>(0, base.size())(rng_);
            keys.assign(base.begin(), base.begin() + take);
        }
        const std::size_t fresh = std::uniform_int_distribution<std::size_t>(0, max_len)(rng_);
        for (std::size_t i = 0; i < fresh; ++i) {
            keys.push_back(next_key_++);
        }
        if (!keys.empty()) {
            history_.push_back(keys);
        }
        return keys;
    }

private:
    std::mt19937 rng_;
    int64_t next_key_ = 1;
    std::vector<std::vector<int64_t>> history_;
};

void RunOracleComparison(const LiteHitMamba::Config &config,
                         const std::vector<uint64_t> &capacities,
                         uint32_t seed,
                         int requests) {
    LiteHitMamba core(config);
    NaiveMambaOracle oracle(config);
    ChainTraceGenerator generator(seed);

    for (int r = 0; r < requests; ++r) {
        const std::vector<int64_t> keys = generator.NextRequest(12);
        const MambaRequestFact fact = core.ProcessRequest(keys);
        const std::vector<uint64_t> expected = oracle.ProcessRequest(keys, capacities);

        // Envelope invariants: strictly increasing in both fields.
        for (std::size_t i = 1; i < fact.points.size(); ++i) {
            ASSERT_LT(fact.points[i - 1].min_total_capacity_bytes, fact.points[i].min_total_capacity_bytes);
            ASSERT_LT(fact.points[i - 1].hit_blocks, fact.points[i].hit_blocks);
        }

        for (std::size_t c = 0; c < capacities.size(); ++c) {
            ASSERT_EQ(expected[c], HitCurveProjector::ProjectMambaBytes(fact, capacities[c]))
                << "request " << r << " capacity " << capacities[c];
        }
        ASSERT_EQ(HitCurveProjector::ProjectMambaBytes(fact, std::numeric_limits<uint64_t>::max() / 4),
                  HitCurveProjector::ProjectMambaInfinite(fact));
    }
}

} // namespace

class LiteHitMambaTest : public TESTBASE {};

TEST_F(LiteHitMambaTest, EmptyRequestYieldsEmptyFactAndNoState) {
    LiteHitMamba core({/*full=*/8, /*mamba=*/4, /*step_blocks=*/2});
    const MambaRequestFact fact = core.ProcessRequest({});
    EXPECT_TRUE(fact.points.empty());
    EXPECT_EQ(0u, core.current_unique_blocks());
    EXPECT_EQ(0u, core.resident_bytes());
}

TEST_F(LiteHitMambaTest, ColdRequestMissesThenRepeatsHit) {
    LiteHitMamba core({/*full=*/8, /*mamba=*/4, /*step_blocks=*/2});

    // 5 blocks, step 2 -> checkpoints at positions 1, 3 and forced 4.
    const std::vector<int64_t> keys = {10, 20, 30, 40, 50};
    const MambaRequestFact cold = core.ProcessRequest(keys);
    EXPECT_TRUE(cold.points.empty());
    EXPECT_EQ(5u, core.current_unique_blocks());
    // 5 Full * 8 + 3 Mamba * 4.
    EXPECT_EQ(52u, core.resident_bytes());

    const MambaRequestFact warm = core.ProcessRequest(keys);
    ASSERT_FALSE(warm.points.empty());
    // Unbounded capacity recovers the forced last checkpoint: all 5 blocks.
    EXPECT_EQ(5u, HitCurveProjector::ProjectMambaInfinite(warm));
    // Zero capacity recovers nothing.
    EXPECT_EQ(0u, HitCurveProjector::ProjectMambaBytes(warm, 0));
}

TEST_F(LiteHitMambaTest, RecoveryRequiresBothFullCoverageAndCheckpoint) {
    // One block per checkpoint keeps the arithmetic small.
    LiteHitMamba core({/*full=*/10, /*mamba=*/1, /*step_blocks=*/1});

    core.ProcessRequest({1, 2, 3});
    // Recency (new->old): F1 M1 F2 M2 F3 M3; required bytes:
    // F1=10, M1=11, F2=21, M2=22, F3=32, M3=33.
    const MambaRequestFact fact = core.ProcessRequest({1, 2, 3});
    // Recover position 0 needs F1 and M1 -> 11; position 1 adds F2/M2 -> 22;
    // position 2 -> 33.
    ASSERT_EQ(3u, fact.points.size());
    EXPECT_EQ(MambaCurvePoint({11, 1}), fact.points[0]);
    EXPECT_EQ(MambaCurvePoint({22, 2}), fact.points[1]);
    EXPECT_EQ(MambaCurvePoint({33, 3}), fact.points[2]);
    EXPECT_EQ(0u, HitCurveProjector::ProjectMambaBytes(fact, 10));
    EXPECT_EQ(1u, HitCurveProjector::ProjectMambaBytes(fact, 11));
    EXPECT_EQ(2u, HitCurveProjector::ProjectMambaBytes(fact, 32));
    EXPECT_EQ(3u, HitCurveProjector::ProjectMambaBytes(fact, 33));
}

TEST_F(LiteHitMambaTest, SharedOracleComparison) {
    RunOracleComparison({/*full=*/48, /*mamba=*/16, /*step_blocks=*/3},
                        {0, 16, 48, 64, 100, 160, 320, 640, 1000, 5000, 100000},
                        20260701,
                        600);
}

TEST_F(LiteHitMambaTest, SharedStepOneOracleComparison) {
    RunOracleComparison(
        {/*full=*/7, /*mamba=*/13, /*step_blocks=*/1}, {0, 7, 13, 20, 39, 77, 200, 1000, 40000}, 20260702, 600);
}

TEST_F(LiteHitMambaTest, ResetClearsEverything) {
    LiteHitMamba core({/*full=*/8, /*mamba=*/4, /*step_blocks=*/2});
    core.ProcessRequest({1, 2, 3, 4});
    EXPECT_GT(core.resident_bytes(), 0u);
    core.Reset();
    EXPECT_EQ(0u, core.current_unique_blocks());
    EXPECT_EQ(0u, core.resident_bytes());
    EXPECT_TRUE(core.ProcessRequest({1, 2}).points.empty());
}

} // namespace kv_cache_manager
