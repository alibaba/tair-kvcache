#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_cache_policy_config.h"
#include "kv_cache_manager/meta/meta_search_cache.h"

namespace kv_cache_manager {

class MetaSearchCacheTest : public TESTBASE {
public:
    using KeyType = MetaSearchCache::KeyType;

public:
    void SetUp() override;

    void TearDown() override {}

    ErrorCode InitSearchCache(const std::string &configStr);

    std::shared_ptr<MetaSearchCache> search_cache_;
};

void MetaSearchCacheTest::SetUp() { search_cache_ = std::make_shared<MetaSearchCache>(); }

ErrorCode MetaSearchCacheTest::InitSearchCache(const std::string &configStr) {
    auto meta_cache_policy_config = std::make_shared<MetaCachePolicyConfig>();
    meta_cache_policy_config->FromJsonString(configStr);
    return search_cache_->Init(meta_cache_policy_config);
}

TEST_F(MetaSearchCacheTest, TestInit) {
    // test success
    std::string configStr = R"({
        "capacity" : 128,
        "type" : "lru",
        "cache_shard_bits" : 8,
        "high_pri_pool_ratio" : 0.5
    })";
    ASSERT_EQ(EC_OK, InitSearchCache(configStr));
    ASSERT_EQ(128, search_cache_->cache_size_);
    ASSERT_TRUE(search_cache_->cache_);
    ASSERT_TRUE(search_cache_->cache_item_helper_);

    // test default
    configStr = R"({})";
    ASSERT_EQ(EC_OK, InitSearchCache(configStr));
    ASSERT_EQ(MetaCachePolicyConfig::kDefaultCacheCapacity, search_cache_->cache_size_);
    ASSERT_TRUE(search_cache_->cache_);
    ASSERT_TRUE(search_cache_->cache_item_helper_);

    // test failed
    configStr = R"({
        "type" : "fifo"
    })";
    ASSERT_EQ(EC_ERROR, InitSearchCache(configStr));
    configStr = R"({
        "cache_shard_bits" : 100
    })";
    ASSERT_EQ(EC_ERROR, InitSearchCache(configStr));
    configStr = R"({
        "high_pri_pool_ratio" : -0.1
    })";
    ASSERT_EQ(EC_ERROR, InitSearchCache(configStr));
    configStr = R"({
        "high_pri_pool_ratio" : 1.1
    })";
    ASSERT_EQ(EC_ERROR, InitSearchCache(configStr));
}

TEST_F(MetaSearchCacheTest, TestSimple) {
    std::string configStr = R"({
        "capacity" : 1,
        "type" : "lru",
        "cache_shard_bits" : 6,
        "high_pri_pool_ratio" : 0
    })";
    ASSERT_EQ(EC_OK, InitSearchCache(configStr));
    ASSERT_EQ(1, search_cache_->cache_size_);
    ASSERT_TRUE(search_cache_->cache_);
    ASSERT_TRUE(search_cache_->cache_item_helper_);

    // test put
    int32_t count = 100;
    for (int32_t i = 0; i < count; i++) {
        auto ec = search_cache_->Put(i, "test_" + std::to_string(i));
        ASSERT_EQ(EC_OK, ec);
    }
    // test get
    for (int32_t i = 0; i < count; i++) {
        std::string value;
        auto ec = search_cache_->Get(i, &value);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ("test_" + std::to_string(i), value);
    }

    // test delete
    int32_t delete_count = 10;
    for (int32_t i = 0; i < delete_count; i++) {
        search_cache_->Delete(i);
        std::string value;
        auto ec = search_cache_->Get(i, &value);
        ASSERT_EQ(EC_NOENT, ec);
        ASSERT_EQ("", value);
    }
    for (int32_t i = delete_count; i < count; i++) {
        std::string value;
        auto ec = search_cache_->Get(i, &value);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ("test_" + std::to_string(i), value);
    }
}

TEST_F(MetaSearchCacheTest, TestPutFull) {
    std::string configStr = R"({
        "capacity" : 1,
        "type" : "lru",
        "cache_shard_bits" : 6,
        "high_pri_pool_ratio" : 0
    })";
    ASSERT_EQ(EC_OK, InitSearchCache(configStr));
    ASSERT_EQ(1, search_cache_->cache_size_);
    ASSERT_TRUE(search_cache_->cache_);
    ASSERT_TRUE(search_cache_->cache_item_helper_);

    // test put until lru full
    int32_t count = 20000;
    for (int32_t i = 0; i < count; i++) {
        auto ec = search_cache_->Put(i, "test_" + std::to_string(i));
        ASSERT_EQ(EC_OK, ec);
    }
    // test get evict key
    int32_t evict_count = 5000;
    for (int32_t i = 0; i < evict_count; i++) {
        std::string value;
        auto ec = search_cache_->Get(i, &value);
        ASSERT_EQ(EC_NOENT, ec);
        ASSERT_EQ("", value);
    }
}

TEST_F(MetaSearchCacheTest, TestPutLargeValue) {
    // test for only one lru shard
    std::string configStr = R"({
        "capacity" : 1,
        "type" : "lru",
        "cache_shard_bits" : 0,
        "high_pri_pool_ratio" : 0
    })";
    ASSERT_EQ(EC_OK, InitSearchCache(configStr));
    ASSERT_EQ(1, search_cache_->cache_size_);
    ASSERT_TRUE(search_cache_->cache_);
    ASSERT_TRUE(search_cache_->cache_item_helper_);

    // 1. test put some keys
    int32_t count = 10;
    for (int32_t i = 0; i < count; i++) {
        auto ec = search_cache_->Put(i, "test_" + std::to_string(i));
        ASSERT_EQ(EC_OK, ec);
    }
    for (int32_t i = 0; i < count; i++) {
        std::string value;
        auto ec = search_cache_->Get(i, &value);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ("test_" + std::to_string(i), value);
    }

    // 2. test put large value > 1mb
    // lru will evict all unpinned keys in single shard, then free large value, finally return ok
    KeyType large_key = count;
    int32_t large_size = 1 * 1024 * 1024;
    std::string large_value(large_size, 'A');
    auto ec = search_cache_->Put(large_key, large_value);
    ASSERT_EQ(EC_OK, ec);
    // all keys were evicted
    ASSERT_EQ(0, search_cache_->cache_->GetUsage());
    for (int32_t i = 0; i < count; i++) {
        std::string value;
        auto ec = search_cache_->Get(i, &value);
        ASSERT_EQ(EC_NOENT, ec);
        ASSERT_EQ("", value);
    }
}

} // namespace kv_cache_manager
