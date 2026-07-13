#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/select_location_policy.h"

using namespace kv_cache_manager;

#define D_EVENT_REPORT DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT
#define D_3FS DataStorageType::DATA_STORAGE_TYPE_HF3FS
#define D_NFS DataStorageType::DATA_STORAGE_TYPE_NFS
#define D_UNKNOWN DataStorageType::DATA_STORAGE_TYPE_UNKNOWN

// Unit tests for WeightSLPolicy::ExistsForWriteWithMinCount.
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
            auto loc = std::make_shared<CacheLocation>();
            loc->set_id(id);
            loc->set_status(meta.status);
            loc->set_type(meta.type);
            loc->set_spec_size(1);
            std::string uri = ToString(meta.type) + "://" + meta.unique_name + "/" + id;
            loc->set_location_specs({LocationSpec("tp0", uri)});
            m[id] = loc;
        }
        return m;
    }
};

// (1) n_total >= min -> true.
TEST_F(FilterWriteCachePredicateTest, TwoEventReportReplicasSatisfyMinTwo) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_SERVING, D_EVENT_REPORT, "er_a"},
        {CLS_SERVING, D_EVENT_REPORT, "er_b"},
    });
    std::vector<std::string> prune;
    EXPECT_TRUE(policy.ExistsForWriteWithMinCount(m, /*min*/ 2, {}, nullptr, prune));
}

// (2) n_total < min -> false.
TEST_F(FilterWriteCachePredicateTest, SingleEventReportReplicaTriggersRemoteWrite) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_SERVING, D_EVENT_REPORT, "er_a"},
    });
    std::vector<std::string> prune;
    EXPECT_FALSE(policy.ExistsForWriteWithMinCount(m, /*min*/ 2, {}, nullptr, prune));
}

// (3) Mixed types still count as long as weight > 0.
TEST_F(FilterWriteCachePredicateTest, EventReportPlusHf3fsCountsAsTwo) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_SERVING, D_EVENT_REPORT, "er_a"},
        {CLS_SERVING, D_3FS, "3fs_01"},
    });
    std::vector<std::string> prune;
    EXPECT_TRUE(policy.ExistsForWriteWithMinCount(m, /*min*/ 2, {}, nullptr, prune));
}

// (4) weight=0 entries do NOT count.
TEST_F(FilterWriteCachePredicateTest, ZeroWeightDoesNotCount) {
    StaticWeightSLPolicy::WeightArray w{};
    // index layout: [0]UNKNOWN [1]HF3FS [2]MOONCAKE [3]TAIR [4]NFS
    //               [5]VCNS_HF3FS [6]DUMMY [7]EVENT_REPORT
    w[static_cast<size_t>(D_EVENT_REPORT)] = 10;
    w[static_cast<size_t>(D_NFS)] = 0; // explicitly zero
    DynamicWeightSLPoliy policy(w);

    auto m = MakeMap({
        {CLS_SERVING, D_EVENT_REPORT, "er_a"}, {CLS_SERVING, D_NFS, "nfs_01"}, // weight=0, ignored
    });
    std::vector<std::string> prune;
    EXPECT_FALSE(policy.ExistsForWriteWithMinCount(m, /*min*/ 2, {}, nullptr, prune));
}

// (5) CLS_NOT_FOUND entries are excluded.
TEST_F(FilterWriteCachePredicateTest, NotFoundLocationsAreExcluded) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_SERVING, D_EVENT_REPORT, "er_a"}, {CLS_NOT_FOUND, D_EVENT_REPORT, "er_b"}, // not counted
    });
    std::vector<std::string> prune;
    EXPECT_FALSE(policy.ExistsForWriteWithMinCount(m, /*min*/ 2, {}, nullptr, prune));
}

// (6) Stale replicas are subtracted; event report locations are filtered but not pruned.
TEST_F(FilterWriteCachePredicateTest, StaleCheckRemovesUnhealthyReplica) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_SERVING, D_EVENT_REPORT, "er_alive"},
        {CLS_SERVING, D_EVENT_REPORT, "er_stale"},
    });
    auto stale_check = [](const CacheLocation &loc) -> bool {
        for (const auto &spec : loc.location_specs()) {
            if (spec.uri().find("er_stale") != std::string::npos) {
                return false;
            }
        }
        return true;
    };
    std::vector<std::string> prune_loc_ids;
    EXPECT_FALSE(policy.ExistsForWriteWithMinCount(m, /*min*/ 2, {}, stale_check, prune_loc_ids));
    // D_EVENT_REPORT locations are not pruned (only filtered)
    EXPECT_EQ(prune_loc_ids.size(), 0u);
    // Same map, min=1 still satisfied because er_alive remains.
    EXPECT_TRUE(policy.ExistsForWriteWithMinCount(m, /*min*/ 1, {}, stale_check, prune_loc_ids));
    EXPECT_EQ(prune_loc_ids.size(), 0u);
}

// (7) All stale -> false; event report locations are not pruned.
TEST_F(FilterWriteCachePredicateTest, AllStaleReturnsFalse) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_SERVING, D_EVENT_REPORT, "er_a"},
        {CLS_SERVING, D_EVENT_REPORT, "er_b"},
    });
    auto stale_check = [](const CacheLocation &) -> bool { return false; };
    std::vector<std::string> prune_loc_ids;
    EXPECT_FALSE(policy.ExistsForWriteWithMinCount(m, /*min*/ 1, {}, stale_check, prune_loc_ids));
    // D_EVENT_REPORT locations are not pruned (only filtered)
    EXPECT_EQ(prune_loc_ids.size(), 0u);
}

// (8) CLS_WRITING bypasses stale check.
TEST_F(FilterWriteCachePredicateTest, WritingPlaceholderBypassesStaleCheck) {
    StaticWeightSLPolicy policy;
    auto m = MakeMap({
        {CLS_WRITING, D_EVENT_REPORT, "er_writing"},
    });
    auto stale_check = [](const CacheLocation &) -> bool { return false; };
    std::vector<std::string> unused_prune;
    EXPECT_TRUE(policy.ExistsForWriteWithMinCount(m, /*min*/ 1, {}, stale_check, unused_prune));
}
