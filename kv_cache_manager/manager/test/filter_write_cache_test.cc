#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/select_location_policy.h"

using namespace kv_cache_manager;

#define D_VINEYARD DataStorageType::DATA_STORAGE_TYPE_VINEYARD
#define D_3FS DataStorageType::DATA_STORAGE_TYPE_HF3FS
#define D_NFS DataStorageType::DATA_STORAGE_TYPE_NFS
#define D_UNKNOWN DataStorageType::DATA_STORAGE_TYPE_UNKNOWN

// V8 §2.6 / 8.4 #5 -- this test fixture exercises the predicate that drives
// V6D's "skip remote write if n_total >= min_replica_count" decision inside
// CacheManager::FilterWriteCacheWithMinReplica. We target
// WeightSLPolicy::ExistsForWriteWithMinCount directly so the predicate is
// nailed down without spinning up a CacheManager / MetaSearcher fixture.
class FilterWriteCachePredicateTest : public TESTBASE {
public:
    struct Meta {
        CacheLocationStatus status;
        DataStorageType type;
        std::string unique_name;
    };

    static CacheLocationMap MakeMap(const std::vector<Meta> &metas) {
        CacheLocationMap m;
        for (const auto &meta : metas) {
            auto id = StringUtil::GenerateRandomString(8);
            CacheLocation loc;
            loc.set_id(id);
            loc.set_status(meta.status);
            loc.set_type(meta.type);
            loc.set_spec_size(1);
            std::string uri = ToString(meta.type) + "://" + meta.unique_name + "/" + id;
            loc.set_location_specs({LocationSpec("tp0", uri)});
            m[id] = std::move(loc);
        }
        return m;
    }
};

// (1) n_total >= min_replica_count -> true (skip remote write).
//     Two V6D replicas already cover the V6D eviction case.
TEST_F(FilterWriteCachePredicateTest, TwoVineyardReplicasSatisfyMinTwo) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_SERVING, D_VINEYARD, "v6d_a"},
        {CLS_SERVING, D_VINEYARD, "v6d_b"},
    });
    EXPECT_TRUE(policy.ExistsForWriteWithMinCount(m, /*min*/ 2, nullptr));
}

// (2) n_total < min_replica_count -> false (must write to remote).
//     Only V6D-self replica exists; the eviction path needs to allocate
//     a HF3FS / Mooncake / NFS write target.
TEST_F(FilterWriteCachePredicateTest, SingleVineyardReplicaTriggersRemoteWrite) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_SERVING, D_VINEYARD, "v6d_a"},
    });
    EXPECT_FALSE(policy.ExistsForWriteWithMinCount(m, /*min*/ 2, nullptr));
}

// (3) Mixed types still count as long as weight > 0.
TEST_F(FilterWriteCachePredicateTest, VineyardPlusHf3fsCountsAsTwo) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_SERVING, D_VINEYARD, "v6d_a"},
        {CLS_SERVING, D_3FS, "3fs_01"},
    });
    EXPECT_TRUE(policy.ExistsForWriteWithMinCount(m, /*min*/ 2, nullptr));
}

// (4) UNKNOWN / weight=0 entries do NOT count.
//     With a custom DynamicWeight that zero-weights NFS, an NFS replica
//     should not satisfy the predicate even though its status is SERVING.
TEST_F(FilterWriteCachePredicateTest, ZeroWeightDoesNotCount) {
    StaticWeightSLPolicy::WeightArray w{};
    // index layout: [0]UNKNOWN [1]HF3FS [2]MOONCAKE [3]TAIR [4]NFS
    //               [5]VCNS_HF3FS [6]DUMMY [7]VINEYARD
    w[static_cast<size_t>(D_VINEYARD)] = 10;
    w[static_cast<size_t>(D_NFS)] = 0; // explicitly zero
    DynamicWeightSLPoliy policy(w);

    auto m = MakeMap({
        {CLS_SERVING, D_VINEYARD, "v6d_a"}, {CLS_SERVING, D_NFS, "nfs_01"}, // weight=0, ignored
    });
    EXPECT_FALSE(policy.ExistsForWriteWithMinCount(m, /*min*/ 2, nullptr));
}

// (5) CLS_NOT_FOUND entries are excluded.
TEST_F(FilterWriteCachePredicateTest, NotFoundLocationsAreExcluded) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_SERVING, D_VINEYARD, "v6d_a"}, {CLS_NOT_FOUND, D_VINEYARD, "v6d_b"}, // not counted
    });
    EXPECT_FALSE(policy.ExistsForWriteWithMinCount(m, /*min*/ 2, nullptr));
}

// (6) check_loc_data_exist must subtract stale replicas from n_total.
//     Two V6D entries exist on paper, but the stale check fails one of them
//     -> only one effective replica -> n_total < 2 -> false (write needed).
//     The NOT_EXIST entry must appear in out_prune_loc_ids.
TEST_F(FilterWriteCachePredicateTest, StaleCheckRemovesUnhealthyReplica) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_SERVING, D_VINEYARD, "v6d_alive"},
        {CLS_SERVING, D_VINEYARD, "v6d_stale"},
    });
    auto stale_check = [](const CacheLocation &loc) -> LocCheckResult {
        for (const auto &spec : loc.location_specs()) {
            if (spec.uri().find("v6d_stale") != std::string::npos) {
                return LocCheckResult::NOT_EXIST;
            }
        }
        return LocCheckResult::EXIST;
    };
    std::vector<std::string> prune_loc_ids;
    EXPECT_FALSE(policy.ExistsForWriteWithMinCount(m, /*min*/ 2, stale_check, prune_loc_ids));
    EXPECT_EQ(prune_loc_ids.size(), 1u);
    // Same map, min=1 still satisfied because v6d_alive remains.
    EXPECT_TRUE(policy.ExistsForWriteWithMinCount(m, /*min*/ 1, stale_check, prune_loc_ids));
    EXPECT_EQ(prune_loc_ids.size(), 1u);
}

// (7) When all replicas are stale, the predicate must return false even if
//     the underlying type/weight would otherwise qualify.
//     All entries must appear in out_prune_loc_ids.
TEST_F(FilterWriteCachePredicateTest, AllStaleReturnsFalse) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_SERVING, D_VINEYARD, "v6d_a"},
        {CLS_SERVING, D_VINEYARD, "v6d_b"},
    });
    auto stale_check = [](const CacheLocation &) -> LocCheckResult { return LocCheckResult::NOT_EXIST; };
    std::vector<std::string> prune_loc_ids;
    EXPECT_FALSE(policy.ExistsForWriteWithMinCount(m, /*min*/ 1, stale_check, prune_loc_ids));
    EXPECT_EQ(prune_loc_ids.size(), 2u);
}

// (8) CLS_WRITING bypasses stale check (an in-flight write is conceptually
//     "already covered"; we don't second-guess it via data existence). This
//     mirrors WeightSLPolicy::ExistsForWrite's treatment of CLS_WRITING.
TEST_F(FilterWriteCachePredicateTest, WritingPlaceholderBypassesStaleCheck) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_WRITING, D_VINEYARD, "v6d_writing"},
    });
    auto stale_check = [](const CacheLocation &) -> LocCheckResult { return LocCheckResult::NOT_EXIST; };
    EXPECT_TRUE(policy.ExistsForWriteWithMinCount(m, /*min*/ 1, stale_check));
}
