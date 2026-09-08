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
    ASSERT_TRUE(config_->GetMutexEnabled());
    ASSERT_EQ(MetaIndexerConfig::kDefaultPersistMetaDataIntervalTimeMs, config_->GetPersistMetaDataIntervalTimeMs());
    ASSERT_EQ("local", config_->GetMetaStorageBackendConfig()->GetStorageType());

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

TEST_F(MetaIndexerConfigTest, TestMutexEnabled) {
    EXPECT_TRUE(config_->GetMutexEnabled());
    const MetaIndexerConfig constructed_config(100, 8, 16, std::make_shared<MetaStorageBackendConfig>());
    EXPECT_TRUE(constructed_config.GetMutexEnabled());

    for (const bool enabled : {false, true}) {
        ASSERT_TRUE(config_->FromJsonString(enabled ? R"({"mutex_enabled":true})" : R"({"mutex_enabled":false})"));
        EXPECT_EQ(enabled, config_->GetMutexEnabled());
        MetaIndexerConfig restored_config;
        ASSERT_TRUE(restored_config.FromJsonString(config_->ToJsonString()));
        EXPECT_EQ(enabled, restored_config.GetMutexEnabled());
    }

    config_->SetMutexEnabled(false);
    ASSERT_TRUE(config_->FromJsonString("{}"));
    EXPECT_TRUE(config_->GetMutexEnabled());
    EXPECT_FALSE(config_->FromJsonString(R"({"mutex_enabled":"false"})"));
}
