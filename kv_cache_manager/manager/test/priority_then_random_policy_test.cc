#include <set>

#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/priority_then_random_policy.h"

using namespace kv_cache_manager;

#define D_3FS DataStorageType::DATA_STORAGE_TYPE_HF3FS
#define D_MOONCAKE DataStorageType::DATA_STORAGE_TYPE_MOONCAKE
#define D_NFS DataStorageType::DATA_STORAGE_TYPE_NFS
#define D_MEMPOOL DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL
#define D_VINEYARD DataStorageType::DATA_STORAGE_TYPE_VINEYARD
#define D_UNKNOWN DataStorageType::DATA_STORAGE_TYPE_UNKNOWN

static CheckLocDataExistFunc dummy_check = [](const CacheLocation &) -> LocCheckResult {
    return LocCheckResult::EXIST;
};
static std::vector<std::string> dummy_loc_ids;

class PriorityThenRandomSLPolicyTest : public TESTBASE {
public:
    struct FakeLocationMeta {
        CacheLocationStatus status;
        DataStorageType type;
        std::string unique_name;
    };

    CacheLocationMap GenLocationMap(const std::vector<FakeLocationMeta> &metas) {
        CacheLocationMap location_map;
        for (const auto &meta : metas) {
            auto id = StringUtil::GenerateRandomString(8);
            location_map[id] = GenFakeLocation(id, meta);
        }
        return location_map;
    }

    CacheLocationConstPtr GenFakeLocation(const std::string &id, const FakeLocationMeta &meta) const {
        std::string uri = ToString(meta.type) + "://" + meta.unique_name + "/" + id;
        auto location = std::make_shared<CacheLocation>();
        location->set_id(id);
        location->set_status(meta.status);
        location->set_type(meta.type);
        location->set_spec_size(1);
        location->set_location_specs({LocationSpec("tp0", uri)});
        return location;
    }
};

// (1) Cross-type deterministic: VINEYARD(weight=10) always beats
// TAIR_MEMPOOL(weight=3). Run many iterations to make the deterministic
// claim concrete -- a weighted-random policy would let TAIR slip through.
TEST_F(PriorityThenRandomSLPolicyTest, CrossTypeDeterministicTopTier) {
    PriorityThenRandomSLPolicy policy;
    auto location_map = GenLocationMap({
        {CLS_SERVING, D_VINEYARD, "v6d_a"},
        {CLS_SERVING, D_MEMPOOL, "pace_01"},
        {CLS_SERVING, D_3FS, "3fs_01"},
        {CLS_SERVING, D_NFS, "nfs_01"},
    });
    for (int i = 0; i < 200; ++i) {
        std::vector<std::string> prune;
        auto picked = policy.SelectForMatch(location_map, dummy_check, prune);
        ASSERT_TRUE(picked && !picked->id().empty());
        ASSERT_EQ(picked->type(), D_VINEYARD) << "iteration " << i << " unexpectedly picked non-VINEYARD type";
    }
}

// (2) Same-weight random: with three VINEYARD nodes sharing top weight, every
// node must be observed across enough iterations. We don't assert any precise
// distribution -- only "no node is starved" -- to keep the test robust.
TEST_F(PriorityThenRandomSLPolicyTest, SameWeightUniformRandom) {
    PriorityThenRandomSLPolicy policy;
    auto location_map = GenLocationMap({
        {CLS_SERVING, D_VINEYARD, "v6d_a"},
        {CLS_SERVING, D_VINEYARD, "v6d_b"},
        {CLS_SERVING, D_VINEYARD, "v6d_c"},
        {CLS_SERVING, D_MEMPOOL, "pace_01"},
    });
    std::set<std::string> seen_ids;
    for (int i = 0; i < 500; ++i) {
        std::vector<std::string> prune;
        auto picked = policy.SelectForMatch(location_map, dummy_check, prune);
        ASSERT_TRUE(picked && !picked->id().empty());
        ASSERT_EQ(picked->type(), D_VINEYARD);
        seen_ids.insert(picked->id());
    }
    // All three v6d nodes should be hit at least once across 500 draws.
    // Probability of missing one node in 500 uniform draws is ~(2/3)^500 ≈ 0.
    ASSERT_EQ(seen_ids.size(), 3u);
}

// (3) Stale check at top tier: when every VINEYARD is judged stale, prune list
// is complete and the policy falls back to the next-best tier (TAIR_MEMPOOL).
TEST_F(PriorityThenRandomSLPolicyTest, StaleTopTierFallsBackToLower) {
    PriorityThenRandomSLPolicy policy;
    auto location_map = GenLocationMap({
        {CLS_SERVING, D_VINEYARD, "v6d_a"},
        {CLS_SERVING, D_VINEYARD, "v6d_b"},
        {CLS_SERVING, D_MEMPOOL, "pace_01"},
        {CLS_SERVING, D_3FS, "3fs_01"},
    });
    auto stale_check = [](const CacheLocation &loc) -> LocCheckResult {
        return loc.type() == D_VINEYARD ? LocCheckResult::NOT_EXIST : LocCheckResult::EXIST;
    };
    for (int i = 0; i < 100; ++i) {
        std::vector<std::string> prune;
        auto picked = policy.SelectForMatch(location_map, stale_check, prune);
        ASSERT_TRUE(picked && !picked->id().empty());
        // Both VINEYARD locations must be in the prune list every iteration.
        ASSERT_EQ(prune.size(), 2u);
        for (const auto &id : prune) {
            ASSERT_EQ(location_map.at(id)->type(), D_VINEYARD);
        }
        // After pruning VINEYARD, TAIR_MEMPOOL(3) and HF3FS(3) share the top
        // weight; either is acceptable.
        ASSERT_THAT(picked->type(), AnyOf(D_MEMPOOL, D_3FS));
    }
}

// (4) Stale-everything: every CLS_SERVING is stale, returns nullptr and the
// prune list contains all of them.
TEST_F(PriorityThenRandomSLPolicyTest, AllStaleReturnsNullptr) {
    PriorityThenRandomSLPolicy policy;
    auto location_map = GenLocationMap({
        {CLS_SERVING, D_VINEYARD, "v6d_a"},
        {CLS_SERVING, D_MEMPOOL, "pace_01"},
    });
    auto stale_check = [](const CacheLocation &) -> LocCheckResult { return LocCheckResult::NOT_EXIST; };
    std::vector<std::string> prune;
    auto picked = policy.SelectForMatch(location_map, stale_check, prune);
    ASSERT_TRUE(picked->id().empty());
    ASSERT_EQ(prune.size(), 2u);
}

// (5) Non-CLS_SERVING entries (CLS_WRITING / CLS_NOT_FOUND) are ignored
// entirely -- not pruned, not selected.
TEST_F(PriorityThenRandomSLPolicyTest, IgnoresNonServingEntries) {
    PriorityThenRandomSLPolicy policy;
    auto location_map = GenLocationMap({
        {CLS_WRITING, D_VINEYARD, "v6d_writing"},
        {CLS_NOT_FOUND, D_VINEYARD, "v6d_notfound"},
        {CLS_SERVING, D_MEMPOOL, "pace_01"},
    });
    for (int i = 0; i < 100; ++i) {
        std::vector<std::string> prune;
        auto picked = policy.SelectForMatch(location_map, dummy_check, prune);
        ASSERT_TRUE(picked && !picked->id().empty());
        // VINEYARD entries are non-SERVING so they don't participate; only the
        // MEMPOOL one is eligible.
        ASSERT_EQ(picked->type(), D_MEMPOOL);
        ASSERT_TRUE(prune.empty());
    }
}

// (6) Empty map: returns nullptr safely without mutating prune list.
TEST_F(PriorityThenRandomSLPolicyTest, EmptyMapReturnsNullptr) {
    PriorityThenRandomSLPolicy policy;
    CacheLocationMap location_map;
    std::vector<std::string> prune{"prefilled_should_be_cleared"};
    auto picked = policy.SelectForMatch(location_map, dummy_check, prune);
    ASSERT_TRUE(picked->id().empty());
    // SelectForMatch contract: prune is overwritten -- callers must not rely
    // on prior contents.
    ASSERT_TRUE(prune.empty());
}

// (7) Zero-weight types are treated as non-candidates: D_UNKNOWN has weight 1
// in the static table, so we use a custom weight slot via DynamicWeight to
// nail down the "weight=0 skipped" branch. Using PriorityThenRandom directly
// (which inherits StaticWeight weights) means we just verify UNKNOWN is not
// returned when a higher-weight peer exists.
TEST_F(PriorityThenRandomSLPolicyTest, NonZeroPeerBeatsDefaultWeight) {
    PriorityThenRandomSLPolicy policy;
    auto location_map = GenLocationMap({
        {CLS_SERVING, D_UNKNOWN, "unknown_01"}, // weight=1 (DEFAULT)
        {CLS_SERVING, D_VINEYARD, "v6d_a"},     // weight=10
    });
    for (int i = 0; i < 100; ++i) {
        std::vector<std::string> prune;
        auto picked = policy.SelectForMatch(location_map, dummy_check, prune);
        ASSERT_TRUE(picked && !picked->id().empty());
        ASSERT_EQ(picked->type(), D_VINEYARD);
    }
}

// (8) ExistsForWrite: PriorityThenRandom inherits from StaticWeightSLPolicy
// without overriding the write path, so any CLS_SERVING with weight>0 should
// satisfy the "already written" predicate. This test guards against an
// accidental override regression.
TEST_F(PriorityThenRandomSLPolicyTest, ExistsForWriteInheritsBase) {
    PriorityThenRandomSLPolicy policy;
    {
        auto location_map = GenLocationMap({
            {CLS_SERVING, D_VINEYARD, "v6d_a"},
        });
        std::vector<std::string> prune;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, nullptr, prune));
    }
    {
        // Only CLS_NOT_FOUND -> nothing exists.
        auto location_map = GenLocationMap({
            {CLS_NOT_FOUND, D_VINEYARD, "v6d_a"},
            {CLS_NOT_FOUND, D_MEMPOOL, "pace_01"},
        });
        std::vector<std::string> prune;
        ASSERT_FALSE(policy.ExistsForWrite(location_map, nullptr, prune));
    }
}
