#include <thread>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/meta//test/meta_indexer_test_base.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_search_cache.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"

namespace kv_cache_manager {

class MetaIndexerTest : public MetaIndexerTestBase, public TESTBASE {
public:
    void SetUp() override;

    void TearDown() override {}

    ErrorCode InitIndexer(const std::string &configStr);
};

void MetaIndexerTest::SetUp() {
    meta_indexer_ = std::make_shared<MetaIndexer>();
    request_context_ = std::make_shared<RequestContext>("test_trace_id");
}

ErrorCode MetaIndexerTest::InitIndexer(const std::string &configStr) {
    auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    meta_indexer_config->FromJsonString(configStr);
    std::string local_path = GetPrivateTestRuntimeDataPath() + "meta_local_backend_file1";
    meta_indexer_config->meta_storage_backend_config_->SetStorageUri("file://" + local_path);
    return meta_indexer_->Init(/*instance_id*/ "test", meta_indexer_config);
}

TEST_F(MetaIndexerTest, TestInit) {
    // test success
    std::string configStr = R"({
        "max_key_count" : 100, "mutex_shard_num" : 8,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : {}
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(100, meta_indexer_->max_key_count_);
    ASSERT_EQ(8, meta_indexer_->mutex_shard_num_);
    ASSERT_TRUE(meta_indexer_->cache_);
    ASSERT_EQ(MetaCachePolicyConfig::kDefaultCacheCapacity, meta_indexer_->cache_->cache_size_);
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR, meta_indexer_->storage_->GetStorageType());

    // test failed
    ASSERT_EQ(ErrorCode::EC_BADARGS, meta_indexer_->Init(/*instance_id*/ "test", nullptr));

    auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    meta_indexer_config->meta_storage_backend_config_ = nullptr;
    ASSERT_EQ(EC_BADARGS, meta_indexer_->Init(/*instance_id*/ "test", meta_indexer_config));

    configStr = R"({
        "meta_storage_backend_config" : { "storage_type" : "test" },
        "meta_cache_policy_config" : {}
    })";
    ASSERT_EQ(EC_ERROR, InitIndexer(configStr));

    configStr = R"({
        "max_key_count" : 100, "mutex_shard_num" : 10,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : {}
    })";
    ASSERT_EQ(EC_CONFIG_ERROR, InitIndexer(configStr));

    configStr = R"({
        "max_key_count" : 100, "mutex_shard_num" : 0,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : {}
    })";
    ASSERT_EQ(EC_CONFIG_ERROR, InitIndexer(configStr));

    configStr = R"({
        "max_key_count" : 100, "mutex_shard_num" : 128,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : {}
    })";
    ASSERT_EQ(EC_CONFIG_ERROR, InitIndexer(configStr));

    configStr = R"({
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "type" : "fifo" }
    })";
    ASSERT_EQ(EC_ERROR, InitIndexer(configStr));
}

TEST_F(MetaIndexerTest, TestMakeBatches) {
    std::string configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,
        "batch_key_size" : 2,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(100, meta_indexer_->max_key_count_);
    ASSERT_EQ(8, meta_indexer_->mutex_shard_num_);
    ASSERT_EQ(2, meta_indexer_->batch_key_size_);
    ASSERT_FALSE(meta_indexer_->cache_);
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR, meta_indexer_->storage_->GetStorageType());

    MetaIndexer::BatchMetaData batch_data;
    PropertyMapVector empty_properties;
    KeyVector keys = {0, 1, 2, 3, 4, 8, 9, 80, 800};
    meta_indexer_->MakeBatches(keys, empty_properties, batch_data);
    ASSERT_EQ(4, batch_data.batch_shard_indexs.size());
    ASSERT_EQ(4, batch_data.batch_indexs.size());
    ASSERT_EQ(4, batch_data.batch_keys.size());

    std::vector<std::vector<int32_t>> expect_batch_shard_indexs = {{0}, {1}, {2, 3}, {4}};
    std::vector<std::vector<int32_t>> expect_batch_indexs = {{0, 5, 7, 8}, {1, 6}, {2, 3}, {4}};
    std::vector<KeyVector> expect_batch_keys = {{0, 8, 80, 800}, {1, 9}, {2, 3}, {4}};
    ASSERT_EQ(expect_batch_shard_indexs, batch_data.batch_shard_indexs);
    ASSERT_EQ(expect_batch_indexs, batch_data.batch_indexs);
    ASSERT_EQ(expect_batch_keys, batch_data.batch_keys);
    ASSERT_TRUE(batch_data.batch_properties.empty());
}

TEST_F(MetaIndexerTest, TestMakeBatches2) {
    std::string configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 16,
        "batch_key_size" : 3,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(100, meta_indexer_->max_key_count_);
    ASSERT_EQ(16, meta_indexer_->mutex_shard_num_);
    ASSERT_EQ(3, meta_indexer_->batch_key_size_);
    ASSERT_FALSE(meta_indexer_->cache_);
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR, meta_indexer_->storage_->GetStorageType());

    MetaIndexer::BatchMetaData batch_data;
    KeyVector keys = {0, 4, 7, 16, 20, 32, 33, 34, 35, 64};
    PropertyMapVector properties = {{{"uri", "0"}},
                                    {{"uri", "4"}},
                                    {{"uri", "7"}},
                                    {{"uri", "16"}},
                                    {{"uri", "20"}},
                                    {{"uri", "32"}},
                                    {{"uri", "33"}},
                                    {{"uri", "34"}},
                                    {{"uri", "35"}},
                                    {{"uri", "64"}}};
    meta_indexer_->MakeBatches(keys, properties, batch_data);
    ASSERT_EQ(3, batch_data.batch_shard_indexs.size());
    ASSERT_EQ(3, batch_data.batch_indexs.size());
    ASSERT_EQ(3, batch_data.batch_keys.size());

    std::vector<std::vector<int32_t>> expect_batch_shard_indexs = {{0}, {1, 2, 3}, {4, 7}};
    std::vector<std::vector<int32_t>> expect_batch_indexs = {{0, 3, 5, 9}, {6, 7, 8}, {1, 4, 2}};
    std::vector<KeyVector> expect_batch_keys = {{0, 16, 32, 64}, {33, 34, 35}, {4, 20, 7}};
    std::vector<PropertyMapVector> expect_batch_properties = {
        {{{"uri", "0"}}, {{"uri", "16"}}, {{"uri", "32"}}, {{"uri", "64"}}},
        {{{"uri", "33"}}, {{"uri", "34"}}, {{"uri", "35"}}},
        {{{"uri", "4"}}, {{"uri", "20"}}, {{"uri", "7"}}}};
    ASSERT_EQ(expect_batch_shard_indexs, batch_data.batch_shard_indexs);
    ASSERT_EQ(expect_batch_indexs, batch_data.batch_indexs);
    ASSERT_EQ(expect_batch_keys, batch_data.batch_keys);
    ASSERT_EQ(expect_batch_properties, batch_data.batch_properties);
}

TEST_F(MetaIndexerTest, TestLocalSimple) {
    std::string configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,        
        "batch_key_size" : 2,
        "meta_storage_backend_config" : {
            "storage_type" : "local"
        },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(100, meta_indexer_->max_key_count_);
    ASSERT_EQ(8, meta_indexer_->mutex_shard_num_);
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR, meta_indexer_->storage_->GetStorageType());
    ASSERT_FALSE(meta_indexer_->cache_);
    DoSimpleTest();

    configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,        
        "batch_key_size" : 2,
        "meta_storage_backend_config" : {
            "storage_type" : "local"
        },
        "meta_cache_policy_config" : { "capacity" : 16 }
    })";
    meta_indexer_ = std::make_unique<MetaIndexer>();
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_TRUE(meta_indexer_->cache_);
    ASSERT_EQ(16, meta_indexer_->cache_->cache_size_);
    DoSimpleTest();
}

TEST_F(MetaIndexerTest, TestMultiThread) {
    std::string configStr = R"({
        "max_key_count" : 10000,
        "mutex_shard_num" : 16,        
        "batch_key_size" : 4,
        "meta_storage_backend_config" : {
            "storage_type" : "local"
        },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(10000, meta_indexer_->max_key_count_);
    ASSERT_EQ(16, meta_indexer_->mutex_shard_num_);
    ASSERT_EQ(4, meta_indexer_->batch_key_size_);
    ASSERT_FALSE(meta_indexer_->cache_);
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR, meta_indexer_->storage_->GetStorageType());
    DoMultiThreadTest();

    configStr = R"({
        "max_key_count" : 10000,
        "mutex_shard_num" : 16,        
        "batch_key_size" : 4,
        "meta_storage_backend_config" : {
            "storage_type" : "local"
        },
        "meta_cache_policy_config" : { "capacity" : 16 }
    })";
    meta_indexer_ = std::make_unique<MetaIndexer>();
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_TRUE(meta_indexer_->cache_);
    ASSERT_EQ(16, meta_indexer_->cache_->cache_size_);
    DoMultiThreadTest();
}

} // namespace kv_cache_manager
