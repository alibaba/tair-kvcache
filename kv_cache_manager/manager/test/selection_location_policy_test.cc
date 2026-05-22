#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/select_location_policy.h"

using namespace kv_cache_manager;

#define D_3FS DataStorageType::DATA_STORAGE_TYPE_HF3FS
#define D_MOONCAKE DataStorageType::DATA_STORAGE_TYPE_MOONCAKE
#define D_NFS DataStorageType::DATA_STORAGE_TYPE_NFS
#define D_MEMPOOL DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL
#define D_UNKNOWN DataStorageType::DATA_STORAGE_TYPE_UNKNOWN

static CheckLocDataExistFunc dummy_check_loc_data_exist = [](const CacheLocation &) -> bool { return true; };
static std::vector<std::string> dummy_loc_ids;

class SelectLocationPolicyTest : public TESTBASE {
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
        location->set_spec_size(1000);
        location->set_location_specs({LocationSpec("tp0", uri)});
        return location;
    }

    struct FakeLocationMetaWithSpecs {
        CacheLocationStatus status;
        DataStorageType type;
        std::string unique_name;
        std::vector<std::string> spec_names;
    };

    CacheLocationConstPtr GenFakeLocationWithSpecs(const std::string &id, const FakeLocationMetaWithSpecs &meta) const {
        auto location = std::make_shared<CacheLocation>();
        location->set_id(id);
        location->set_status(meta.status);
        location->set_type(meta.type);
        location->set_spec_size(meta.spec_names.size());
        std::vector<LocationSpec> specs;
        for (const auto &name : meta.spec_names) {
            std::string uri = ToString(meta.type) + "://" + meta.unique_name + "/" + id + "/" + name;
            specs.emplace_back(name, uri);
        }
        location->set_location_specs(std::move(specs));
        return location;
    }

    CacheLocationMap GenLocationMapWithSpecs(const std::vector<FakeLocationMetaWithSpecs> &metas) {
        CacheLocationMap location_map;
        for (const auto &meta : metas) {
            auto id = StringUtil::GenerateRandomString(8);
            location_map[id] = GenFakeLocationWithSpecs(id, meta);
        }
        return location_map;
    }
};

TEST_F(SelectLocationPolicyTest, TestStaticWeightSLPolicySelectForMatch) {
    StaticWeightSLPolicy policy;
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocationConstPtr location =
                policy.SelectForMatch(location_map, dummy_check_loc_data_exist, dummy_loc_ids);
            ASSERT_THAT(location->type(), AnyOf(D_MEMPOOL, D_3FS, D_NFS));
        }
    }
    {
        auto location_map = GenLocationMap(
            {{CLS_WRITING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocationConstPtr location =
                policy.SelectForMatch(location_map, dummy_check_loc_data_exist, dummy_loc_ids);
            ASSERT_THAT(location->type(), AnyOf(D_3FS, D_NFS));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_MEMPOOL, "pace_01"},
                                            {CLS_SERVING, D_3FS, "3fs_01"},
                                            {CLS_SERVING, D_3FS, "3fs_02"},
                                            {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocationConstPtr location =
                policy.SelectForMatch(location_map, dummy_check_loc_data_exist, dummy_loc_ids);
            ASSERT_THAT(location->type(), AnyOf(D_3FS, D_NFS));
            ASSERT_THAT(location->location_specs().front().uri(),
                        AnyOf(HasSubstr("3fs_01"), HasSubstr("3fs_02"), HasSubstr("nfs_01")));
        }
    }
}

TEST_F(SelectLocationPolicyTest, TestStaticWeightSLPolicySelectForMatchWithStaleCheck) {
    StaticWeightSLPolicy policy;
    // (a) all CLS_SERVING stale ->
    //     all pruned, returns nullptr
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        CacheLocationConstPtr location = policy.SelectForMatch(location_map, stale_check, prune_loc_ids);
        ASSERT_TRUE(location->location_specs().empty());
        ASSERT_EQ(prune_loc_ids.size(), 2u);
        for (const auto &id : prune_loc_ids) {
            ASSERT_EQ(location_map.at(id)->status(), CLS_SERVING);
        }
    }
    // (b) mix of stale and valid CLS_SERVING ->
    //     stale entries pruned, returns a valid one
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}, {CLS_SERVING, D_MEMPOOL, "pace_01"}});
        // 3fs is stale, nfs and mempool are valid
        auto stale_check = [](const CacheLocation &loc) -> bool {
            return loc.type() != DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        };
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            CacheLocationConstPtr location = policy.SelectForMatch(location_map, stale_check, prune_loc_ids);
            ASSERT_FALSE(location->location_specs().empty());
            ASSERT_THAT(location->type(), AnyOf(D_NFS, D_MEMPOOL));
            ASSERT_EQ(prune_loc_ids.size(), 1u);
            ASSERT_EQ(location_map.at(prune_loc_ids[0])->type(), D_3FS);
        }
    }
    // (c) non-CLS_SERVING entries are ignored entirely ->
    //     not selected, only CLS_SERVING pruned
    {
        auto location_map = GenLocationMap(
            {{CLS_WRITING, D_3FS, "3fs_01"}, {CLS_NOT_FOUND, D_NFS, "nfs_01"}, {CLS_SERVING, D_MEMPOOL, "pace_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        CacheLocationConstPtr location = policy.SelectForMatch(location_map, stale_check, prune_loc_ids);
        // only the CLS_SERVING entry is checked; it is stale and pruned
        ASSERT_TRUE(location->location_specs().empty());
        ASSERT_EQ(prune_loc_ids.size(), 1u);
        ASSERT_EQ(location_map.at(prune_loc_ids[0])->status(), CLS_SERVING);
    }
    // (d) single stale CLS_SERVING entry ->
    //     pruned, returns nullptr
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        CacheLocationConstPtr location = policy.SelectForMatch(location_map, stale_check, prune_loc_ids);
        ASSERT_TRUE(location->location_specs().empty());
        ASSERT_EQ(prune_loc_ids.size(), 1u);
    }
}

TEST_F(SelectLocationPolicyTest, TestStaticWeightSLPolicyExistsForWrite) {
    StaticWeightSLPolicy policy;
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            ASSERT_TRUE(policy.ExistsForWrite(location_map, nullptr, prune_loc_ids));
        }
    }
    {
        auto location_map = GenLocationMap(
            {{CLS_NOT_FOUND, D_MEMPOOL, "pace_01"}, {CLS_NOT_FOUND, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            ASSERT_TRUE(policy.ExistsForWrite(location_map, nullptr, prune_loc_ids));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_NOT_FOUND, D_MEMPOOL, "pace_01"},
                                            {CLS_NOT_FOUND, D_3FS, "3fs_01"},
                                            {CLS_NOT_FOUND, D_3FS, "3fs_02"},
                                            {CLS_NOT_FOUND, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            ASSERT_FALSE(policy.ExistsForWrite(location_map, nullptr, prune_loc_ids));
        }
    }
}

TEST_F(SelectLocationPolicyTest, TestStaticWeightSLPolicyExistsForWriteWithStaleCheck) {
    StaticWeightSLPolicy policy;
    // (a) single stale CLS_SERVING entry ->
    //     prune list populated, returns false
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_FALSE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_EQ(prune_loc_ids.size(), 1u);
        ASSERT_EQ(prune_loc_ids[0], location_map.begin()->first);
    }
    // (b) all CLS_SERVING stale ->
    //     all pruned, returns false
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_FALSE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_EQ(prune_loc_ids.size(), 2u);
    }
    // (c) CLS_WRITING entry alone ->
    //     not pruned, bypasses stale check, returns true
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_3FS, "3fs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_TRUE(prune_loc_ids.empty());
    }
    // (d) all CLS_SERVING stale + CLS_NOT_FOUND ->
    //     returns false, only CLS_SERVING pruned
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}, {CLS_NOT_FOUND, D_MEMPOOL, "pace_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_FALSE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_EQ(prune_loc_ids.size(), 2u);
    }
    // (e) CLS_WRITING + all CLS_SERVING stale ->
    //     all stale CLS_SERVING entries pruned, CLS_WRITING not pruned,
    //     returns true (full scan guarantees complete prune list)
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_NFS, "nfs_01"}, {CLS_SERVING, D_MEMPOOL, "pace_01"}, {CLS_WRITING, D_3FS, "3fs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_EQ(prune_loc_ids.size(), 2u);
        for (const auto &id : prune_loc_ids) {
            ASSERT_EQ(location_map.at(id)->status(), CLS_SERVING);
        }
    }
    // (f) mix of stale and valid CLS_SERVING ->
    //     stale entry pruned, returns true, prune list complete
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        auto stale_check = [](const CacheLocation &loc) -> bool {
            return loc.type() != DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        };
        std::vector<std::string> prune_loc_ids;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_EQ(prune_loc_ids.size(), 1u);
        ASSERT_EQ(location_map.at(prune_loc_ids[0])->type(), D_3FS);
    }
}

TEST_F(SelectLocationPolicyTest, TestDynamicWeightSLPolicySelectForMatch) {
    DynamicWeightSLPoliy policy({
        0, // default
        1, // 3fs
        1, // mooncake
        1, // pace
        0  // nfs
    });
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocationConstPtr location =
                policy.SelectForMatch(location_map, dummy_check_loc_data_exist, dummy_loc_ids);
            ASSERT_THAT(location->type(), AnyOf(D_MEMPOOL, D_3FS));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_MEMPOOL, "pace_01"},
                                            {CLS_SERVING, D_3FS, "3fs_01"},
                                            {CLS_SERVING, D_MOONCAKE, "mooncake_01"},
                                            {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocationConstPtr location =
                policy.SelectForMatch(location_map, dummy_check_loc_data_exist, dummy_loc_ids);
            ASSERT_THAT(location->type(), AnyOf(D_3FS, D_MOONCAKE));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_MEMPOOL, "pace_01"},
                                            {CLS_SERVING, D_3FS, "3fs_01"},
                                            {CLS_SERVING, D_MOONCAKE, "mooncake_01"},
                                            {CLS_SERVING, D_MOONCAKE, "mooncake_02"},
                                            {CLS_SERVING, D_NFS, "nfs_01"},
                                            {CLS_SERVING, D_NFS, "nfs_02"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocationConstPtr location =
                policy.SelectForMatch(location_map, dummy_check_loc_data_exist, dummy_loc_ids);
            ASSERT_THAT(location->type(), AnyOf(D_3FS, D_MOONCAKE));
            ASSERT_THAT(location->location_specs().front().uri(),
                        AnyOf(HasSubstr("3fs_01"), HasSubstr("mooncake_01"), HasSubstr("mooncake_02")));
        }
    }
}

TEST_F(SelectLocationPolicyTest, TestDynamicWeightSLPolicySelectForMatchWithStaleCheck) {
    DynamicWeightSLPoliy policy({
        0, // default
        1, // 3fs
        1, // mooncake
        1, // pace
        0  // nfs
    });
    // (a) all CLS_SERVING stale ->
    //     all pruned, returns nullptr
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_MOONCAKE, "mooncake_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        CacheLocationConstPtr location = policy.SelectForMatch(location_map, stale_check, prune_loc_ids);
        ASSERT_TRUE(location->location_specs().empty());
        ASSERT_EQ(prune_loc_ids.size(), 2u);
    }
    // (b) mix of stale and valid ->
    //     stale pruned, returns valid
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"},
                                            {CLS_SERVING, D_MOONCAKE, "mooncake_01"},
                                            {CLS_SERVING, D_MEMPOOL, "pace_01"}});
        // 3fs is stale, mooncake and mempool are valid
        auto stale_check = [](const CacheLocation &loc) -> bool {
            return loc.type() != DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        };
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            CacheLocationConstPtr location = policy.SelectForMatch(location_map, stale_check, prune_loc_ids);
            ASSERT_FALSE(location->location_specs().empty());
            ASSERT_THAT(location->type(), AnyOf(D_MOONCAKE, D_MEMPOOL));
            ASSERT_EQ(prune_loc_ids.size(), 1u);
            ASSERT_EQ(location_map.at(prune_loc_ids[0])->type(), D_3FS);
        }
    }
    // (c) non-CLS_SERVING entries are ignored entirely ->
    //     not selected, only CLS_SERVING pruned
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_3FS, "3fs_01"}, {CLS_SERVING, D_MOONCAKE, "mooncake_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        CacheLocationConstPtr location = policy.SelectForMatch(location_map, stale_check, prune_loc_ids);
        ASSERT_TRUE(location->location_specs().empty());
        // only the CLS_SERVING mooncake entry is pruned
        ASSERT_EQ(prune_loc_ids.size(), 1u);
        ASSERT_EQ(location_map.at(prune_loc_ids[0])->status(), CLS_SERVING);
    }
    // (d) stale CLS_SERVING entry with weight 0 (nfs) ->
    //     still pruned
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_NFS, "nfs_01"}, {CLS_SERVING, D_3FS, "3fs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        CacheLocationConstPtr location = policy.SelectForMatch(location_map, stale_check, prune_loc_ids);
        ASSERT_TRUE(location->location_specs().empty());
        // both are CLS_SERVING and stale, both pruned (weight check
        // happens after the stale-data check)
        ASSERT_EQ(prune_loc_ids.size(), 2u);
    }
}

TEST_F(SelectLocationPolicyTest, TestDynamicWeightSLPolicyExistsForWrite) {
    DynamicWeightSLPoliy policy({
        0, // default
        1, // 3fs
        1, // mooncake
        1, // pace
        0  // nfs
    });
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            ASSERT_TRUE(policy.ExistsForWrite(location_map, nullptr, prune_loc_ids));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            ASSERT_FALSE(policy.ExistsForWrite(location_map, nullptr, prune_loc_ids));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_NFS, "nfs_01"}, {CLS_SERVING, D_NFS, "nfs_02"}});
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            ASSERT_FALSE(policy.ExistsForWrite(location_map, nullptr, prune_loc_ids));
        }
    }
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}, {CLS_SERVING, D_NFS, "nfs_02"}});
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            ASSERT_TRUE(policy.ExistsForWrite(location_map, nullptr, prune_loc_ids));
        }
    }
}

TEST_F(SelectLocationPolicyTest, TestDynamicWeightSLPolicyExistsForWriteWithStaleCheck) {
    DynamicWeightSLPoliy policy({
        0, // default
        1, // 3fs
        1, // mooncake
        1, // pace
        0  // nfs
    });
    // (a) single stale CLS_SERVING entry ->
    //     pruned, returns false
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_FALSE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_EQ(prune_loc_ids.size(), 1u);
    }
    // (b) all CLS_SERVING stale ->
    //     all pruned, returns false
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_MOONCAKE, "mooncake_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_FALSE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_EQ(prune_loc_ids.size(), 2u);
    }
    // (c) CLS_WRITING alone ->
    //     bypasses stale check, not pruned, returns true
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_3FS, "3fs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_TRUE(prune_loc_ids.empty());
    }
    // (d) CLS_WRITING + all CLS_SERVING stale ->
    //     all stale CLS_SERVING entries pruned, returns true
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_MOONCAKE, "mooncake_01"},
                                            {CLS_SERVING, D_3FS, "3fs_01"},
                                            {CLS_WRITING, D_MEMPOOL, "pace_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_EQ(prune_loc_ids.size(), 2u);
        for (const auto &id : prune_loc_ids) {
            ASSERT_EQ(location_map.at(id)->status(), CLS_SERVING);
        }
    }
    // (e) mix of stale and valid CLS_SERVING ->
    //     stale entry pruned, returns true, prune list complete
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_MOONCAKE, "mooncake_01"}});
        auto stale_check = [](const CacheLocation &loc) -> bool {
            return loc.type() != DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        };
        std::vector<std::string> prune_loc_ids;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_EQ(prune_loc_ids.size(), 1u);
        ASSERT_EQ(location_map.at(prune_loc_ids[0])->type(), D_3FS);
    }
}

TEST_F(SelectLocationPolicyTest, TestNamedStorageWeightedSLPolicySelectForMatch) {
    NamedStorageWeightedSLPolicy policy({{"pace_01", 1}, {"3fs_01", 1}, {"3fs_02", 0}, {"nfs_01", 1}});
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocationConstPtr location =
                policy.SelectForMatch(location_map, dummy_check_loc_data_exist, dummy_loc_ids);
            ASSERT_THAT(location->type(), AnyOf(D_MEMPOOL, D_3FS, D_NFS));
            ASSERT_THAT(location->location_specs().front().uri(),
                        AnyOf(HasSubstr("pace_01"), HasSubstr("3fs_01"), HasSubstr("nfs_01")));
        }
    }
    {
        auto location_map = GenLocationMap(
            {{CLS_WRITING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocationConstPtr location =
                policy.SelectForMatch(location_map, dummy_check_loc_data_exist, dummy_loc_ids);
            ASSERT_THAT(location->type(), AnyOf(D_3FS, D_NFS));
            ASSERT_THAT(location->location_specs().front().uri(), AnyOf(HasSubstr("3fs_01"), HasSubstr("nfs_01")));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_MEMPOOL, "pace_01"},
                                            {CLS_SERVING, D_3FS, "3fs_01"},
                                            {CLS_SERVING, D_3FS, "3fs_02"},
                                            {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocationConstPtr location =
                policy.SelectForMatch(location_map, dummy_check_loc_data_exist, dummy_loc_ids);
            ASSERT_THAT(location->type(), AnyOf(D_3FS, D_NFS));
            ASSERT_THAT(location->location_specs().front().uri(), AnyOf(HasSubstr("3fs_01"), HasSubstr("nfs_01")));
        }
    }
}

TEST_F(SelectLocationPolicyTest, TestNamedStorageWeightedSLPolicySelectForMatchWithStaleCheck) {
    NamedStorageWeightedSLPolicy policy({{"pace_01", 1}, {"3fs_01", 1}, {"3fs_02", 0}, {"nfs_01", 1}});
    // (a) all CLS_SERVING stale ->
    //     all pruned, returns nullptr
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        CacheLocationConstPtr location = policy.SelectForMatch(location_map, stale_check, prune_loc_ids);
        ASSERT_TRUE(location->location_specs().empty());
        ASSERT_EQ(prune_loc_ids.size(), 2u);
    }
    // (b) mix of stale and valid ->
    //     stale pruned, returns valid
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}, {CLS_SERVING, D_MEMPOOL, "pace_01"}});
        // 3fs is stale, nfs and mempool are valid
        auto stale_check = [](const CacheLocation &loc) -> bool {
            return loc.type() != DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        };
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            CacheLocationConstPtr location = policy.SelectForMatch(location_map, stale_check, prune_loc_ids);
            ASSERT_FALSE(location->location_specs().empty());
            ASSERT_THAT(location->type(), AnyOf(D_NFS, D_MEMPOOL));
            ASSERT_EQ(prune_loc_ids.size(), 1u);
            ASSERT_EQ(location_map.at(prune_loc_ids[0])->type(), D_3FS);
        }
    }
    // (c) non-CLS_SERVING entries ignored ->
    //     stale CLS_SERVING pruned
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        CacheLocationConstPtr location = policy.SelectForMatch(location_map, stale_check, prune_loc_ids);
        ASSERT_TRUE(location->location_specs().empty());
        // only the CLS_SERVING nfs entry is pruned
        ASSERT_EQ(prune_loc_ids.size(), 1u);
        ASSERT_EQ(location_map.at(prune_loc_ids[0])->status(), CLS_SERVING);
    }
    // (d) stale entry with weight 0 (3fs_02) ->
    //     still pruned (stale-data check runs before the weight check)
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_02"}, {CLS_SERVING, D_3FS, "3fs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        CacheLocationConstPtr location = policy.SelectForMatch(location_map, stale_check, prune_loc_ids);
        ASSERT_TRUE(location->location_specs().empty());
        ASSERT_EQ(prune_loc_ids.size(), 2u);
    }
}

TEST_F(SelectLocationPolicyTest, TestNamedStorageWeightedSLPolicyExistsForWrite) {
    NamedStorageWeightedSLPolicy policy({{"pace_01", 0}, {"3fs_01", 1}, {"3fs_02", 0}, {"nfs_01", 1}});
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            ASSERT_TRUE(policy.ExistsForWrite(location_map, nullptr, prune_loc_ids));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_MEMPOOL, "pace_01"}});
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            ASSERT_FALSE(policy.ExistsForWrite(location_map, nullptr, prune_loc_ids));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_02"}});
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            ASSERT_FALSE(policy.ExistsForWrite(location_map, nullptr, prune_loc_ids));
        }
    }
    {
        auto location_map = GenLocationMap(
            {{CLS_WRITING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_02"}, {CLS_SERVING, D_3FS, "3fs_01"}});
        for (int i = 0; i < 100; ++i) {
            std::vector<std::string> prune_loc_ids;
            ASSERT_TRUE(policy.ExistsForWrite(location_map, nullptr, prune_loc_ids));
        }
    }
}

TEST_F(SelectLocationPolicyTest, TestNamedStorageWeightedSLPolicyExistsForWriteWithStaleCheck) {
    NamedStorageWeightedSLPolicy policy({{"pace_01", 0}, {"3fs_01", 1}, {"3fs_02", 0}, {"nfs_01", 1}});
    // (a) single stale CLS_SERVING entry with weight > 0 ->
    //     pruned, returns false
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_FALSE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_EQ(prune_loc_ids.size(), 1u);
    }
    // (b) all CLS_SERVING stale ->
    //     all pruned, returns false
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_FALSE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_EQ(prune_loc_ids.size(), 2u);
    }
    // (c) CLS_WRITING alone ->
    //     bypasses stale check, not pruned, returns true
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_3FS, "3fs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_TRUE(prune_loc_ids.empty());
    }
    // (d) CLS_WRITING with weight 0 ->
    //     does not make ExistsForWrite return true
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_MEMPOOL, "pace_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_FALSE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_TRUE(prune_loc_ids.empty());
    }
    // (e) CLS_WRITING + all CLS_SERVING stale ->
    //     all stale CLS_SERVING entries pruned, returns true
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}, {CLS_WRITING, D_3FS, "3fs_01"}});
        auto stale_check = [](const CacheLocation &) -> bool { return false; };
        std::vector<std::string> prune_loc_ids;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_EQ(prune_loc_ids.size(), 2u);
        for (const auto &id : prune_loc_ids) {
            ASSERT_EQ(location_map.at(id)->status(), CLS_SERVING);
        }
    }
    // (f) mix of stale and valid CLS_SERVING ->
    //     stale entry pruned, returns true, prune list complete
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        auto stale_check = [](const CacheLocation &loc) -> bool {
            return loc.type() != DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        };
        std::vector<std::string> prune_loc_ids;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, stale_check, prune_loc_ids));
        ASSERT_EQ(prune_loc_ids.size(), 1u);
        ASSERT_EQ(location_map.at(prune_loc_ids[0])->type(), D_3FS);
    }
}

TEST_F(SelectLocationPolicyTest, TestStaticWeightExistsForWriteWithSpecNames) {
    StaticWeightSLPolicy policy;
    // empty spec_names falls back to original 1-param overload
    {
        auto location_map = GenLocationMapWithSpecs({{CLS_SERVING, D_3FS, "3fs_01", {"tp0"}}});
        std::vector<std::string> prune_loc_ids;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, {}, nullptr, prune_loc_ids));
    }
    // single location covers all requested specs
    {
        auto location_map = GenLocationMapWithSpecs({{CLS_SERVING, D_3FS, "3fs_01", {"tp0_F0", "tp1_F0", "tp0_L1"}}});
        std::vector<std::string> prune_loc_ids;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, {"tp0_F0", "tp1_F0"}, nullptr, prune_loc_ids));
    }
    // no single location covers all requested specs (split across two locations)
    {
        auto location_map = GenLocationMapWithSpecs({{CLS_SERVING, D_3FS, "3fs_01", {"tp0_F0", "tp1_F0"}},
                                                     {CLS_SERVING, D_NFS, "nfs_01", {"tp0_L1", "tp1_L1"}}});
        std::vector<std::string> prune_loc_ids;
        ASSERT_FALSE(policy.ExistsForWrite(location_map, {"tp0_F0", "tp0_L1"}, nullptr, prune_loc_ids));
    }
    // NOT_FOUND status location is skipped even if specs match
    {
        auto location_map = GenLocationMapWithSpecs({{CLS_NOT_FOUND, D_3FS, "3fs_01", {"tp0_F0", "tp1_F0"}}});
        std::vector<std::string> prune_loc_ids;
        ASSERT_FALSE(policy.ExistsForWrite(location_map, {"tp0_F0"}, nullptr, prune_loc_ids));
    }
}

TEST_F(SelectLocationPolicyTest, TestDynamicWeightExistsForWriteWithSpecNames) {
    DynamicWeightSLPoliy policy({
        0, // default
        1, // 3fs
        1, // mooncake
        1, // pace
        0  // nfs
    });
    // NFS covers all specs but weight=0 → false
    {
        auto location_map = GenLocationMapWithSpecs({{CLS_SERVING, D_NFS, "nfs_01", {"tp0_F0", "tp1_F0"}}});
        std::vector<std::string> prune_loc_ids;
        ASSERT_FALSE(policy.ExistsForWrite(location_map, {"tp0_F0"}, nullptr, prune_loc_ids));
    }
    // 3FS covers spec and weight>0 → true
    {
        auto location_map = GenLocationMapWithSpecs({{CLS_SERVING, D_3FS, "3fs_01", {"tp0_F0"}}});
        std::vector<std::string> prune_loc_ids;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, {"tp0_F0"}, nullptr, prune_loc_ids));
    }
}

TEST_F(SelectLocationPolicyTest, TestNamedStorageWeightedExistsForWriteWithSpecNames) {
    NamedStorageWeightedSLPolicy policy({{"3fs_01", 1}, {"3fs_02", 0}});
    // 3fs_02 weight=0, covers spec → false
    {
        auto location_map = GenLocationMapWithSpecs({{CLS_SERVING, D_3FS, "3fs_02", {"tp0_F0"}}});
        std::vector<std::string> prune_loc_ids;
        ASSERT_FALSE(policy.ExistsForWrite(location_map, {"tp0_F0"}, nullptr, prune_loc_ids));
    }
    // 3fs_01 weight=1, covers spec → true
    {
        auto location_map = GenLocationMapWithSpecs({{CLS_SERVING, D_3FS, "3fs_01", {"tp0_F0"}}});
        std::vector<std::string> prune_loc_ids;
        ASSERT_TRUE(policy.ExistsForWrite(location_map, {"tp0_F0"}, nullptr, prune_loc_ids));
    }
}

TEST_F(SelectLocationPolicyTest, TestIsSameDataStorageRejectsCrossType) {
    StaticWeightSLPolicy policy;
    // Same type, same host → true
    {
        CacheLocation nfs_a;
        nfs_a.set_type(D_NFS);
        nfs_a.set_location_specs({LocationSpec("tp0", "nfs://host01/a/tp0")});
        CacheLocation nfs_b;
        nfs_b.set_type(D_NFS);
        nfs_b.set_location_specs({LocationSpec("tp1", "nfs://host01/b/tp1")});
        EXPECT_TRUE(policy.IsSameDataStorage(nfs_a, nfs_b));
    }
    // Different type (NFS vs Mooncake), same host → false
    {
        CacheLocation nfs_loc;
        nfs_loc.set_type(D_NFS);
        nfs_loc.set_location_specs({LocationSpec("tp0", "nfs://host01/a/tp0")});
        CacheLocation mc_loc;
        mc_loc.set_type(D_MOONCAKE);
        mc_loc.set_location_specs({LocationSpec("tp0", "mooncake://host01/b/tp0")});
        EXPECT_FALSE(policy.IsSameDataStorage(nfs_loc, mc_loc));
    }
    // Same type, different host → false
    {
        CacheLocation nfs_a;
        nfs_a.set_type(D_NFS);
        nfs_a.set_location_specs({LocationSpec("tp0", "nfs://host01/a/tp0")});
        CacheLocation nfs_b;
        nfs_b.set_type(D_NFS);
        nfs_b.set_location_specs({LocationSpec("tp0", "nfs://host02/b/tp0")});
        EXPECT_FALSE(policy.IsSameDataStorage(nfs_a, nfs_b));
    }
    // Both empty specs → true (vacuously same)
    {
        CacheLocation empty_a;
        empty_a.set_type(D_NFS);
        CacheLocation empty_b;
        empty_b.set_type(D_NFS);
        EXPECT_TRUE(policy.IsSameDataStorage(empty_a, empty_b));
    }
    // One empty, one non-empty → false
    {
        CacheLocation empty_loc;
        empty_loc.set_type(D_NFS);
        CacheLocation nfs_loc;
        nfs_loc.set_type(D_NFS);
        nfs_loc.set_location_specs({LocationSpec("tp0", "nfs://host01/a/tp0")});
        EXPECT_FALSE(policy.IsSameDataStorage(empty_loc, nfs_loc));
    }
}
