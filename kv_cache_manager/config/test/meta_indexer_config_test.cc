#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"

using namespace kv_cache_manager;

class MetaIndexerConfigTest : public TESTBASE {
public:
    void SetUp() override { config_ = std::make_shared<MetaIndexerConfig>(); }
    void TearDown() override {}

private:
    std::shared_ptr<MetaIndexerConfig> config_;
};

TEST_F(MetaIndexerConfigTest, TestSimple) {
    ASSERT_FALSE(config_->FromJsonString(""));

    std::string configStr = R"({
        "meta_storage_backend_config": {
            "storage_type": "local"
        }
    })";
    ASSERT_TRUE(config_->FromJsonString(configStr));
    ASSERT_EQ(MetaIndexerConfig::kDefaultMaxKeyCount, config_->GetMaxKeyCount());
    ASSERT_EQ(MetaIndexerConfig::kDefaultMutexShardNum, config_->GetMutexShardNum());
    ASSERT_EQ(MetaIndexerConfig::kDefaultPersistMetaDataIntervalTimeMs, config_->GetPersistMetaDataIntervalTimeMs());
    ASSERT_EQ("local", config_->GetMetaStorageBackendConfig()->GetStorageType());
    ASSERT_FALSE(config_->GetMetaStorageBackendConfig()->GetMemoryPrimary());
    ASSERT_TRUE(config_->GetMetaStorageBackendConfig()->GetForceDeletingAsyncEnqueue());
    EXPECT_NE(std::string::npos, config_->ToJsonString().find("meta_cache_policy_config"));

    configStr = R"({
        "max_key_count": 1000,
        "mutex_shard_num": 100,
        "meta_storage_backend_config": {
            "storage_type": "redis"
        }
    })";
    ASSERT_FALSE(config_->FromJsonString(configStr));

    configStr = R"({
        "max_key_count": 1000,
        "mutex_shard_num": 0,
        "meta_storage_backend_config": {
            "storage_type": "redis"
        }
    })";
    ASSERT_FALSE(config_->FromJsonString(configStr));

    configStr = R"({
        "max_key_count": 1000,
        "mutex_shard_num": 64,
        "persist_metadata_interval_time_ms": 2000,
        "meta_storage_backend_config": {
            "storage_type": "redis"
        }
    })";
    ASSERT_TRUE(config_->FromJsonString(configStr));
    ASSERT_EQ(1000, config_->GetMaxKeyCount());
    ASSERT_EQ(64, config_->GetMutexShardNum());
    ASSERT_EQ(2000, config_->GetPersistMetaDataIntervalTimeMs());
    ASSERT_EQ("redis", config_->GetMetaStorageBackendConfig()->GetStorageType());
}

TEST_F(MetaIndexerConfigTest, TestMemoryPrimaryJsonRoundTripAndDefault) {
    MetaStorageBackendConfig backend;
    EXPECT_FALSE(backend.GetMemoryPrimary());
    EXPECT_TRUE(backend.GetForceDeletingAsyncEnqueue());
    ASSERT_TRUE(backend.FromJsonString(
        R"({"storage_type":"cached","storage_uri":"redis://backup:6379/?persistent_type=async_redis","memory_primary":true,"force_deleting_async_enqueue":false})"));
    EXPECT_TRUE(backend.GetMemoryPrimary());
    EXPECT_FALSE(backend.GetForceDeletingAsyncEnqueue());
    MetaStorageBackendConfig restored;
    ASSERT_TRUE(restored.FromJsonString(backend.ToJsonString()));
    EXPECT_TRUE(restored.GetMemoryPrimary());
    EXPECT_FALSE(restored.GetForceDeletingAsyncEnqueue());
    EXPECT_EQ(backend.GetStorageUri(), restored.GetStorageUri());
    ASSERT_TRUE(restored.FromJsonString(R"({"storage_type":"local"})"));
    EXPECT_FALSE(restored.GetMemoryPrimary());
    EXPECT_TRUE(restored.GetForceDeletingAsyncEnqueue());
    EXPECT_FALSE(restored.FromJsonString(R"({"storage_type":"local","memory_primary":"true"})"));
    EXPECT_FALSE(restored.FromJsonString(R"({"storage_type":"local","force_deleting_async_enqueue":"false"})"));
}

TEST_F(MetaIndexerConfigTest, TestLegacySearchCacheValuesRoundTripWithoutRuntimeValidation) {
    ASSERT_TRUE(config_->FromJsonString(R"({
        "meta_storage_backend_config": {"storage_type": "local"},
        "meta_cache_policy_config": {
            "type": "removed", "capacity": 1024, "cache_shard_bits": 23, "high_pri_pool_ratio": 0.25
        }
    })"));
    std::string invalid_fields;
    EXPECT_TRUE(config_->ValidateRequiredFields(invalid_fields));
    EXPECT_TRUE(invalid_fields.empty());
    EXPECT_EQ("local", config_->GetMetaStorageBackendConfig()->GetStorageType());
    rapidjson::Document json;
    json.Parse(config_->ToJsonString().c_str());
    ASSERT_FALSE(json.HasParseError());
    ASSERT_TRUE(json.HasMember("meta_cache_policy_config"));
    const rapidjson::Value &policy = json["meta_cache_policy_config"];
    ASSERT_TRUE(policy.IsObject());
    EXPECT_STREQ("LRU", policy["type"].GetString());
    EXPECT_EQ(1024, policy["capacity"].GetInt());
    EXPECT_EQ(23, policy["cache_shard_bits"].GetInt());
    EXPECT_DOUBLE_EQ(0.0, policy["high_pri_pool_ratio"].GetDouble());
    MetaIndexerConfig restored;
    ASSERT_TRUE(restored.FromJsonString(config_->ToJsonString()));
    EXPECT_EQ(config_->ToJsonString(), restored.ToJsonString());
    ASSERT_TRUE(restored.FromJsonString(R"({"meta_storage_backend_config":{"storage_type":"local"}})"));
    EXPECT_EQ(0u, restored.GetLegacySearchCacheCapacity());
    EXPECT_EQ(6, restored.GetLegacySearchCacheShardBits());
}
