#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_redis_backend.h"
#include "kv_cache_manager/meta/meta_search_cache.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"
#include "kv_cache_manager/meta/test/meta_indexer_test_base.h"

namespace kv_cache_manager {

class MetaIndexerRedisTest : public MetaIndexerTestBase, public TESTBASE {
public:
    void SetUp() override;

    void TearDown() override {}

    ErrorCode InitIndexer(const std::string &configStr);
};

void MetaIndexerRedisTest::SetUp() {
    meta_indexer_ = std::make_shared<MetaIndexer>();
    request_context_ = std::make_shared<RequestContext>("test_trace_id");
}

ErrorCode MetaIndexerRedisTest::InitIndexer(const std::string &configStr) {
    auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    meta_indexer_config->FromJsonString(configStr);
    return meta_indexer_->Init(/*instance_id*/ "test", meta_indexer_config);
}

TEST_F(MetaIndexerRedisTest, TestInit) {
    std::string configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,
        "meta_storage_backend_config" : { 
            "storage_type" : "redis",
            "storage_uri" : "redis://test_redis_user:test_redis_password@localhost:6379/?timeout_ms=1000&retry_count=3&client_pool_size=2"
        },
        "meta_cache_policy_config" : {
            "type" : "lru",
            "capacity" : 1024
        }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(100, meta_indexer_->max_key_count_);
    ASSERT_EQ(8, meta_indexer_->mutex_shard_num_);
    ASSERT_TRUE(meta_indexer_->cache_);
    ASSERT_EQ(1024, meta_indexer_->cache_->cache_size_);
    ASSERT_EQ(META_REDIS_BACKEND_TYPE_STR, meta_indexer_->storage_->GetStorageType());

    auto *redis_storage = dynamic_cast<MetaRedisBackend *>(meta_indexer_->storage_.get());
    ASSERT_TRUE(redis_storage);
    const StandardUri &storage_uri = redis_storage->storage_uri_;
    ASSERT_EQ("test_redis_user:test_redis_password", storage_uri.GetUserInfo());
    ASSERT_EQ("localhost", storage_uri.GetHostName());
    ASSERT_EQ(6379, storage_uri.GetPort());
    ASSERT_EQ("1000", storage_uri.GetParam("timeout_ms"));
    ASSERT_EQ("3", storage_uri.GetParam("retry_count"));
    ASSERT_EQ("2", storage_uri.GetParam("client_pool_size"));
}

TEST_F(MetaIndexerRedisTest, TestRedisSimple) {
    std::string configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,
        "meta_storage_backend_config" : { 
            "storage_type" : "redis",
            "storage_uri" : "redis://test_redis_user:test_redis_password@localhost:6379/?timeout_ms=1000&retry_count=3&client_pool_size=2"
        },
        "meta_cache_policy_config" : {
            "type" : "lru",
            "capacity" : 1024
        }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(100, meta_indexer_->max_key_count_);
    ASSERT_EQ(8, meta_indexer_->mutex_shard_num_);
    ASSERT_EQ(META_REDIS_BACKEND_TYPE_STR, meta_indexer_->storage_->GetStorageType());
    ASSERT_TRUE(meta_indexer_->cache_);
    ASSERT_EQ(1024, meta_indexer_->cache_->cache_size_);
    DoSimpleTest();

    // test redis recover
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
    KVData data;
    int32_t key_count = 3;
    MakeKVData(/*start*/ 0, /*end*/ 3, data);
    auto result = meta_indexer_->Put(request_context_.get(), data.keys, data.uris, data.properties);
    ASSERT_EQ(key_count, meta_indexer_->GetKeyCount());
    ASSERT_EQ(EC_OK, result.ec);
    AssertSearchCacheGet({0, 1, 2}, {"", "", ""}, {EC_NOENT, EC_NOENT, EC_NOENT});
    Result expect_result(3);
    UriVector expect_uris = {"uri_0", "uri_1", "uri_2"};
    AssertGet({0, 1, 2}, expect_uris, expect_result);
    AssertSearchCacheGet({0, 1, 2}, expect_uris, {EC_OK, EC_OK, EC_OK});

    meta_indexer_->PersistMetaData();
    meta_indexer_ = std::make_unique<MetaIndexer>();
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(key_count, meta_indexer_->GetKeyCount());
    AssertSearchCacheGet({0, 1, 2}, {"", "", ""}, {EC_NOENT, EC_NOENT, EC_NOENT});
    AssertGet({0, 1, 2}, expect_uris, expect_result);
    AssertSearchCacheGet({0, 1, 2}, expect_uris, {EC_OK, EC_OK, EC_OK});
}

TEST_F(MetaIndexerRedisTest, TestMultiThread) {
    std::string configStr = R"({
        "max_key_count" : 10000,
        "mutex_shard_num" : 8,
        "batch_key_size" : 8,
        "meta_storage_backend_config" : { 
            "storage_type" : "redis",
            "storage_uri" : "redis://test_redis_user:test_redis_password@localhost:6379/?timeout_ms=1000&retry_count=3&client_pool_size=16"
        },
        "meta_cache_policy_config" : {
            "type" : "lru",
            "capacity" : 1024
        }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(10000, meta_indexer_->max_key_count_);
    ASSERT_EQ(8, meta_indexer_->mutex_shard_num_);
    ASSERT_EQ(8, meta_indexer_->batch_key_size_);
    ASSERT_EQ(META_REDIS_BACKEND_TYPE_STR, meta_indexer_->storage_->GetStorageType());
    ASSERT_TRUE(meta_indexer_->cache_);
    ASSERT_EQ(1024, meta_indexer_->cache_->cache_size_);
    DoMultiThreadTest();
}

} // namespace kv_cache_manager
