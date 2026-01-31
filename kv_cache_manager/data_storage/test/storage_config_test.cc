#include <gtest/gtest.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/storage_config.h"

using namespace kv_cache_manager;

class StorageConfigTest : public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}
};

// TODO 只测试了NFS，其他类型再加吧
TEST_F(StorageConfigTest, TestNfsStorageSpecJsonize) {
    NfsStorageSpec spec;
    spec.set_root_path("/mnt/nfs");
    spec.set_key_count_per_file(10);
    std::string json = spec.ToJsonString();
    EXPECT_NE(json.find("root_path"), std::string::npos);
    EXPECT_NE(json.find("key_count_per_file"), std::string::npos);
    ASSERT_EQ(R"({"root_path":"/mnt/nfs","key_count_per_file":10})", json);
    NfsStorageSpec spec2;
    spec2.FromJsonString(json);
    EXPECT_EQ(spec.root_path(), spec2.root_path());
    EXPECT_EQ(spec.key_count_per_file(), spec2.key_count_per_file());
}

TEST_F(StorageConfigTest, TestStorageConfigJsonizeNfs) {
    std::shared_ptr<NfsStorageSpec> nfs_spec_ptr(new NfsStorageSpec());
    auto &nfs_spec = *nfs_spec_ptr;
    nfs_spec.set_root_path("/mnt/nfs");
    nfs_spec.set_key_count_per_file(5);
    StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_1", nfs_spec_ptr);
    std::string json = config.ToJsonString();
    ASSERT_NE(json.find("file"), std::string::npos);
    ASSERT_NE(json.find("test_1"), std::string::npos);
    ASSERT_NE(json.find("root_path"), std::string::npos);
    ASSERT_EQ(
        R"({"type":"file","is_available":true,"global_unique_name":"test_1","storage_spec":{"root_path":"/mnt/nfs","key_count_per_file":5}})",
        json);
    StorageConfig config2;
    config2.FromJsonString(json);
    EXPECT_EQ(config.type(), config2.type());
    EXPECT_EQ(config.global_unique_name(), config2.global_unique_name());
    auto &storage_spec = config2.storage_spec();
    auto nfs_spec2_ptr = std::dynamic_pointer_cast<NfsStorageSpec>(storage_spec);
    ASSERT_TRUE(nfs_spec2_ptr);
    auto &nfs_spec2 = *nfs_spec2_ptr;
    EXPECT_EQ(nfs_spec2.root_path(), nfs_spec.root_path());
    EXPECT_EQ(nfs_spec2.key_count_per_file(), nfs_spec.key_count_per_file());
}
