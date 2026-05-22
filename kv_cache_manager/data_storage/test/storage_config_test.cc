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

TEST_F(StorageConfigTest, TestTairMemPoolStorageSpecParseNewSchema) {
    // 新版 schema：直接用 service_discovery_url，不带任何老字段。
    const std::string json =
        R"({"domain":"pace.meta","timeout":5000,"service_discovery_url":"spectrum://v-xx?cache_time=30"})";
    TairMemPoolStorageSpec spec;
    ASSERT_TRUE(spec.FromJsonString(json));
    EXPECT_EQ(spec.domain(), "pace.meta");
    EXPECT_EQ(spec.timeout(), 5000);
    EXPECT_EQ(spec.service_discovery_url(), "spectrum://v-xx?cache_time=30");
}

TEST_F(StorageConfigTest, TestTairMemPoolStorageSpecMigrateLegacyVipserverFields) {
    // 老 admin/老持久化数据：只有 enable_vipserver + vipserver_domain，没有 service_discovery_url。
    // 新版应当自动迁移成 service_discovery_url=vipserver://<domain>。
    const std::string json =
        R"({"domain":"pace.meta","timeout":5000,"enable_vipserver":true,"vipserver_domain":"pace.meta.vipserver"})";
    TairMemPoolStorageSpec spec;
    ASSERT_TRUE(spec.FromJsonString(json));
    EXPECT_EQ(spec.domain(), "pace.meta");
    EXPECT_EQ(spec.timeout(), 5000);
    EXPECT_EQ(spec.service_discovery_url(), "vipserver://pace.meta.vipserver");
}

TEST_F(StorageConfigTest, TestTairMemPoolStorageSpecLegacyEnableFalseDoesNotMigrate) {
    // enable_vipserver=false 时不应迁移，service_discovery_url 保持空。
    const std::string json =
        R"({"domain":"pace.meta","timeout":5000,"enable_vipserver":false,"vipserver_domain":"pace.meta.vipserver"})";
    TairMemPoolStorageSpec spec;
    ASSERT_TRUE(spec.FromJsonString(json));
    EXPECT_EQ(spec.service_discovery_url(), "");
}

TEST_F(StorageConfigTest, TestTairMemPoolStorageSpecNewSchemaTakesPrecedenceOverLegacy) {
    // 同时有 service_discovery_url 与老字段时，以 service_discovery_url 为准。
    const std::string json =
        R"({"domain":"pace.meta","timeout":5000,"service_discovery_url":"spectrum://v-yy",)"
        R"("enable_vipserver":true,"vipserver_domain":"pace.meta.vipserver"})";
    TairMemPoolStorageSpec spec;
    ASSERT_TRUE(spec.FromJsonString(json));
    EXPECT_EQ(spec.service_discovery_url(), "spectrum://v-yy");
}

TEST_F(StorageConfigTest, TestTairMemPoolStorageSpecLegacyEnabledButEmptyDomainNoMigrate) {
    // enable_vipserver=true 但 vipserver_domain 为空时，不应生成无意义的 vipserver:// URL。
    const std::string json = R"({"domain":"pace.meta","timeout":5000,"enable_vipserver":true,"vipserver_domain":""})";
    TairMemPoolStorageSpec spec;
    ASSERT_TRUE(spec.FromJsonString(json));
    EXPECT_EQ(spec.service_discovery_url(), "");
}

TEST_F(StorageConfigTest, TestTairMemPoolStorageSpecToJsonOmitsLegacyFields) {
    // ToJsonString 必须只输出新版字段（domain / timeout / service_discovery_url），
    // 不应再输出已废弃的 enable_vipserver / vipserver_domain，避免污染老 client 的解析路径。
    TairMemPoolStorageSpec spec;
    spec.set_domain("pace.meta");
    spec.set_timeout(5000);
    spec.set_service_discovery_url("spectrum://v-zz?cache_time=30");
    const std::string json = spec.ToJsonString();
    EXPECT_NE(json.find("\"domain\":\"pace.meta\""), std::string::npos);
    EXPECT_NE(json.find("\"service_discovery_url\":\"spectrum://v-zz?cache_time=30\""), std::string::npos);
    EXPECT_EQ(json.find("enable_vipserver"), std::string::npos);
    EXPECT_EQ(json.find("vipserver_domain"), std::string::npos);
}

TEST_F(StorageConfigTest, TestTairMemPoolStorageSpecRoundTrip) {
    // 端到端 round-trip：模拟 server 序列化 → client 反序列化场景，验证 service_discovery_url 不丢。
    TairMemPoolStorageSpec spec;
    spec.set_domain("pace.meta");
    spec.set_timeout(5000);
    spec.set_service_discovery_url("static://10.0.0.1:8080,10.0.0.2:8080");
    const std::string json = spec.ToJsonString();

    TairMemPoolStorageSpec parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_EQ(parsed.domain(), spec.domain());
    EXPECT_EQ(parsed.timeout(), spec.timeout());
    EXPECT_EQ(parsed.service_discovery_url(), spec.service_discovery_url());
}
