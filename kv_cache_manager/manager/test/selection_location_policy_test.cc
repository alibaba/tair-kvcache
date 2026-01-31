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

    CacheLocation GenFakeLocation(const std::string &id, const FakeLocationMeta &meta) const {

        std::string uri = ToString(meta.type) + "://" + meta.unique_name + "/" + id;
        CacheLocation location;
        location.set_id(id);
        location.set_status(meta.status);
        location.set_type(meta.type);
        location.set_spec_size(1000);
        location.set_location_specs({LocationSpec("tp0", uri)});
        return location;
    }
};

TEST_F(SelectLocationPolicyTest, TestStaticWeightSLPolicySelectForMatch) {
    StaticWeightSLPolicy policy;
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocation *location = policy.SelectForMatch(location_map);
            ASSERT_THAT(location->type(), AnyOf(D_MEMPOOL, D_3FS, D_NFS));
        }
    }
    {
        auto location_map = GenLocationMap(
            {{CLS_WRITING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocation *location = policy.SelectForMatch(location_map);
            ASSERT_THAT(location->type(), AnyOf(D_3FS, D_NFS));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_MEMPOOL, "pace_01"},
                                            {CLS_SERVING, D_3FS, "3fs_01"},
                                            {CLS_SERVING, D_3FS, "3fs_02"},
                                            {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocation *location = policy.SelectForMatch(location_map);
            ASSERT_THAT(location->type(), AnyOf(D_3FS, D_NFS));
            ASSERT_THAT(location->location_specs().front().uri(),
                        AnyOf(HasSubstr("3fs_01"), HasSubstr("3fs_02"), HasSubstr("nfs_01")));
        }
    }
}

TEST_F(SelectLocationPolicyTest, TestStaticWeightSLPolicyExistsForWrite) {
    StaticWeightSLPolicy policy;
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            ASSERT_TRUE(policy.ExistsForWrite(location_map));
        }
    }
    {
        auto location_map = GenLocationMap(
            {{CLS_NOT_FOUND, D_MEMPOOL, "pace_01"}, {CLS_NOT_FOUND, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            ASSERT_TRUE(policy.ExistsForWrite(location_map));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_NOT_FOUND, D_MEMPOOL, "pace_01"},
                                            {CLS_NOT_FOUND, D_3FS, "3fs_01"},
                                            {CLS_NOT_FOUND, D_3FS, "3fs_02"},
                                            {CLS_NOT_FOUND, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            ASSERT_FALSE(policy.ExistsForWrite(location_map));
        }
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
            CacheLocation *location = policy.SelectForMatch(location_map);
            ASSERT_THAT(location->type(), AnyOf(D_MEMPOOL, D_3FS));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_MEMPOOL, "pace_01"},
                                            {CLS_SERVING, D_3FS, "3fs_01"},
                                            {CLS_SERVING, D_MOONCAKE, "mooncake_01"},
                                            {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocation *location = policy.SelectForMatch(location_map);
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
            CacheLocation *location = policy.SelectForMatch(location_map);
            ASSERT_THAT(location->type(), AnyOf(D_3FS, D_MOONCAKE));
            ASSERT_THAT(location->location_specs().front().uri(),
                        AnyOf(HasSubstr("3fs_01"), HasSubstr("mooncake_01"), HasSubstr("mooncake_02")));
        }
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
            ASSERT_TRUE(policy.ExistsForWrite(location_map));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            ASSERT_FALSE(policy.ExistsForWrite(location_map));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_SERVING, D_NFS, "nfs_01"}, {CLS_SERVING, D_NFS, "nfs_02"}});
        for (int i = 0; i < 100; ++i) {
            ASSERT_FALSE(policy.ExistsForWrite(location_map));
        }
    }
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}, {CLS_SERVING, D_NFS, "nfs_02"}});
        for (int i = 0; i < 100; ++i) {
            ASSERT_TRUE(policy.ExistsForWrite(location_map));
        }
    }
}

TEST_F(SelectLocationPolicyTest, TestNamedStorageWeightedSLPolicySelectForMatch) {
    NamedStorageWeightedSLPolicy policy({{"pace_01", 1}, {"3fs_01", 1}, {"3fs_02", 0}, {"nfs_01", 1}});
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocation *location = policy.SelectForMatch(location_map);
            ASSERT_THAT(location->type(), AnyOf(D_MEMPOOL, D_3FS, D_NFS));
            ASSERT_THAT(location->location_specs().front().uri(),
                        AnyOf(HasSubstr("pace_01"), HasSubstr("3fs_01"), HasSubstr("nfs_01")));
        }
    }
    {
        auto location_map = GenLocationMap(
            {{CLS_WRITING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            CacheLocation *location = policy.SelectForMatch(location_map);
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
            CacheLocation *location = policy.SelectForMatch(location_map);
            ASSERT_THAT(location->type(), AnyOf(D_3FS, D_NFS));
            ASSERT_THAT(location->location_specs().front().uri(), AnyOf(HasSubstr("3fs_01"), HasSubstr("nfs_01")));
        }
    }
}

TEST_F(SelectLocationPolicyTest, TestNamedStorageWeightedSLPolicyExistsForWrite) {
    NamedStorageWeightedSLPolicy policy({{"pace_01", 0}, {"3fs_01", 1}, {"3fs_02", 0}, {"nfs_01", 1}});
    {
        auto location_map = GenLocationMap(
            {{CLS_SERVING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_01"}, {CLS_SERVING, D_NFS, "nfs_01"}});
        for (int i = 0; i < 100; ++i) {
            ASSERT_TRUE(policy.ExistsForWrite(location_map));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_MEMPOOL, "pace_01"}});
        for (int i = 0; i < 100; ++i) {
            ASSERT_FALSE(policy.ExistsForWrite(location_map));
        }
    }
    {
        auto location_map = GenLocationMap({{CLS_WRITING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_02"}});
        for (int i = 0; i < 100; ++i) {
            ASSERT_FALSE(policy.ExistsForWrite(location_map));
        }
    }
    {
        auto location_map = GenLocationMap(
            {{CLS_WRITING, D_MEMPOOL, "pace_01"}, {CLS_SERVING, D_3FS, "3fs_02"}, {CLS_SERVING, D_3FS, "3fs_01"}});
        for (int i = 0; i < 100; ++i) {
            ASSERT_TRUE(policy.ExistsForWrite(location_map));
        }
    }
}