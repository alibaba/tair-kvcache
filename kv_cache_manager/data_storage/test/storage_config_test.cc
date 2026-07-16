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
        R"({"type":"file","is_available":true,"global_unique_name":"test_1","storage_spec":{"root_path":"/mnt/nfs","key_count_per_file":5},"integrity":{"enable_meta_checksum":false,"enable_inline_header":false,"inline_header_version":0,"algo":"crc32_xor_int64"}})",
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

TEST_F(StorageConfigTest, TestDataIntegrityConfigDefaults) {
    DataIntegrityConfig integrity;
    EXPECT_FALSE(integrity.enable_meta_checksum());
    EXPECT_FALSE(integrity.enable_inline_header());
    EXPECT_EQ(integrity.inline_header_version(), 0u);
    EXPECT_EQ(integrity.algo(), ChecksumAlgo::CA_CRC32_XOR_INT64);

    std::string invalid_fields;
    EXPECT_TRUE(integrity.ValidateRequiredFields(invalid_fields));
    EXPECT_TRUE(invalid_fields.empty());
}

TEST_F(StorageConfigTest, TestDataIntegrityConfigJsonRoundTrip) {
    DataIntegrityConfig integrity;
    integrity.set_enable_meta_checksum(true);
    integrity.set_algo(ChecksumAlgo::CA_CRC32_XOR_INT64);
    const std::string json = integrity.ToJsonString();
    EXPECT_NE(json.find("enable_meta_checksum"), std::string::npos);
    EXPECT_NE(json.find("crc32_xor_int64"), std::string::npos);

    DataIntegrityConfig parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_EQ(parsed.enable_meta_checksum(), integrity.enable_meta_checksum());
    EXPECT_EQ(parsed.algo(), integrity.algo());
}

TEST_F(StorageConfigTest, TestDataIntegrityConfigRejectInlineHeader) {
    // 方案 B 拒绝防线：enable_inline_header=true 必须被 Validate 拒绝，
    // 这样 server / client 启动时不会把 "B 已实现" 的错觉传到上层。
    DataIntegrityConfig integrity;
    integrity.set_enable_inline_header(true);
    std::string invalid_fields;
    EXPECT_FALSE(integrity.ValidateRequiredFields(invalid_fields));
    EXPECT_NE(invalid_fields.find("enable_inline_header is reserved"), std::string::npos);
}

TEST_F(StorageConfigTest, TestDataIntegrityConfigRejectVersionWithoutHeader) {
    // 配置矛盾：inline_header_version != 0 但开关没开。
    DataIntegrityConfig integrity;
    integrity.set_inline_header_version(1);
    std::string invalid_fields;
    EXPECT_FALSE(integrity.ValidateRequiredFields(invalid_fields));
    EXPECT_NE(invalid_fields.find("inline_header_version requires enable_inline_header=true"), std::string::npos);
}

TEST_F(StorageConfigTest, TestStorageConfigOmittedIntegrityBackwardCompatible) {
    // 老 JSON 配置不带 integrity 字段 -> 解析成功，integrity 全部走默认 (关闭)。
    const std::string old_json =
        R"({"type":"file","is_available":true,"global_unique_name":"old_cfg","storage_spec":{"root_path":"/mnt/nfs","key_count_per_file":5}})";
    StorageConfig config;
    ASSERT_TRUE(config.FromJsonString(old_json));
    EXPECT_FALSE(config.integrity().enable_meta_checksum());
    EXPECT_FALSE(config.integrity().enable_inline_header());
}

TEST_F(StorageConfigTest, TestStorageConfigPropagatesIntegrityRejection) {
    // StorageConfig::ValidateRequiredFields 应该把 DataIntegrityConfig 的拒绝传出来。
    std::shared_ptr<NfsStorageSpec> nfs_spec_ptr(new NfsStorageSpec());
    nfs_spec_ptr->set_root_path("/mnt/nfs");
    StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_NFS, "cfg", nfs_spec_ptr);
    config.mutable_integrity().set_enable_inline_header(true);
    std::string invalid_fields;
    EXPECT_FALSE(config.ValidateRequiredFields(invalid_fields));
    EXPECT_NE(invalid_fields.find("DataIntegrityConfig"), std::string::npos);
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
    const std::string json = R"({"domain":"pace.meta","timeout":5000,"service_discovery_url":"spectrum://v-yy",)"
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

// integrity 字段类型错乱 (bool 位置放了 string) 时 StorageConfig 必须整体解析失败,
// 而不是把 integrity 静默降级为全默认后照收。
TEST_F(StorageConfigTest, TestStorageConfigRejectsMalformedIntegrity) {
    const std::string malformed_json = R"({
        "type": "file",
        "global_unique_name": "bad_integrity",
        "storage_spec": {"root_path": "/tmp/x", "key_count_per_file": 1},
        "integrity": {"enable_meta_checksum": "true", "enable_inline_header": false}
    })";
    StorageConfig config;
    EXPECT_FALSE(config.FromJsonString(malformed_json));
}

TEST_F(StorageConfigTest, TestStorageConfigRejectsNonObjectIntegrity) {
    const std::string malformed_json = R"({
        "type": "file",
        "global_unique_name": "bad_integrity",
        "storage_spec": {"root_path": "/tmp/x", "key_count_per_file": 1},
        "integrity": "bad"
    })";
    StorageConfig config;
    EXPECT_FALSE(config.FromJsonString(malformed_json));
}
