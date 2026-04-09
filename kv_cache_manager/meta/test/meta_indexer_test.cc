#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/meta//test/meta_indexer_test_base.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_search_cache.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"

using namespace kv_cache_manager;

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

TEST_F(MetaIndexerTest, TestMetadataPersistAndRecover) {
    const std::string configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,
        "persist_metadata_interval_time_ms" : 0,
        "meta_storage_backend_config" : { "storage_type" : "dummy" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    const auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    meta_indexer_config->FromJsonString(configStr);
    const std::string path = GetPrivateTestRuntimeDataPath() + "meta_dummy_backend_file";
    meta_indexer_config->meta_storage_backend_config_->SetStorageUri("file://" + path);

    // verify fresh init behavior
    {
        meta_indexer_ = std::make_shared<MetaIndexer>();
        ASSERT_EQ(ErrorCode::EC_OK, meta_indexer_->Init(/* instance_id */ "test_instance_01", meta_indexer_config));

        ASSERT_EQ(MetaIndexer::InstanceVersion::VERSION_1, meta_indexer_->GetVersion());
        ASSERT_EQ(0, meta_indexer_->GetKeyCount());
        for (auto &v : meta_indexer_->storage_usage_data_.storage_usage_by_type_) {
            ASSERT_EQ(0, v.load());
        }
    }

    // persist
    meta_indexer_->key_count_.store(3);
    const std::vector<std::uint64_t> expected_usage_vec{1, 100, 200, 300, 400, 500};
    ASSERT_EQ(expected_usage_vec.size(), meta_indexer_->storage_usage_data_.storage_usage_by_type_.size());
    for (std::size_t i = 0; i != meta_indexer_->storage_usage_data_.storage_usage_by_type_.size(); ++i) {
        meta_indexer_->storage_usage_data_.storage_usage_by_type_.at(i).store(expected_usage_vec.at(i));
    }
    meta_indexer_->PersistMetaData();

    // verify recovery behavior
    {
        meta_indexer_ = std::make_shared<MetaIndexer>();
        ASSERT_EQ(ErrorCode::EC_OK, meta_indexer_->Init(/* instance_id */ "test_instance_01", meta_indexer_config));

        ASSERT_EQ(MetaIndexer::InstanceVersion::VERSION_1, meta_indexer_->GetVersion());
        ASSERT_EQ(3, meta_indexer_->GetKeyCount());
        for (std::size_t i = 0; i != meta_indexer_->storage_usage_data_.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), meta_indexer_->storage_usage_data_.storage_usage_by_type_.at(i).load());
        }
    }
}

TEST_F(MetaIndexerTest, TestStorageUsageDataManipulation) {
    std::string configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,
        "persist_metadata_interval_time_ms" : 0,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";

    ASSERT_EQ(EC_OK, InitIndexer(configStr));

    // test get/set
    {
        meta_indexer_->storage_usage_data_.Reset();
        ASSERT_EQ(0, meta_indexer_->GetStorageUsage());

        auto type = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
        std::vector<std::uint64_t> expected_usage_vec{0, 100, 200, 300, 400, 0};

        type = DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));

        type = DataStorageType::DATA_STORAGE_TYPE_MOONCAKE;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));

        type = DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));

        type = DataStorageType::DATA_STORAGE_TYPE_NFS;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));

        for (std::size_t i = 0; i != meta_indexer_->storage_usage_data_.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), meta_indexer_->storage_usage_data_.storage_usage_by_type_.at(i).load());
        }

        type = DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));

        type = DataStorageType::DATA_STORAGE_TYPE_MOONCAKE;
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));

        type = DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL;
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));

        type = DataStorageType::DATA_STORAGE_TYPE_NFS;
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));

        std::uint64_t expect_usage = 0;
        for (const auto &v : expected_usage_vec) {
            expect_usage += v;
        }
        ASSERT_EQ(expect_usage, meta_indexer_->GetStorageUsage());
    }

    // test add/sub
    {
        meta_indexer_->storage_usage_data_.Reset();
        auto type = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
        std::vector<std::uint64_t> expected_usage_vec{0, 100, 200, 300, 400, 0};

        type = DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));
        meta_indexer_->AddStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)) + 16,
                  meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 1024);         // would underflow
        ASSERT_EQ(0, meta_indexer_->GetStorageUsageByType(type)); // expect to be proper handled

        type = DataStorageType::DATA_STORAGE_TYPE_MOONCAKE;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));
        meta_indexer_->AddStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)) + 16,
                  meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 1024);         // would underflow
        ASSERT_EQ(0, meta_indexer_->GetStorageUsageByType(type)); // expect to be proper handled

        type = DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));
        meta_indexer_->AddStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)) + 16,
                  meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 1024);         // would underflow
        ASSERT_EQ(0, meta_indexer_->GetStorageUsageByType(type)); // expect to be proper handled

        type = DataStorageType::DATA_STORAGE_TYPE_NFS;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));
        meta_indexer_->AddStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)) + 16,
                  meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 1024);         // would underflow
        ASSERT_EQ(0, meta_indexer_->GetStorageUsageByType(type)); // expect to be proper handled
    }

    // test special case: DATA_STORAGE_TYPE_VCNS_HF3FS behavior as DATA_STORAGE_TYPE_HF3FS
    {
        meta_indexer_->storage_usage_data_.Reset();
        std::vector<std::uint64_t> expected_usage_vec{0, 128, 0, 0, 0, 0};

        meta_indexer_->SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS, 128);
        ASSERT_EQ(128, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
        ASSERT_EQ(128, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        for (std::size_t i = 0; i != meta_indexer_->storage_usage_data_.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), meta_indexer_->storage_usage_data_.storage_usage_by_type_.at(i).load());
        }

        meta_indexer_->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS, 16);
        ASSERT_EQ(128 + 16, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
        ASSERT_EQ(128 + 16, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));

        meta_indexer_->SubStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS, 16);
        ASSERT_EQ(128, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
        ASSERT_EQ(128, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        for (std::size_t i = 0; i != meta_indexer_->storage_usage_data_.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), meta_indexer_->storage_usage_data_.storage_usage_by_type_.at(i).load());
        }

        meta_indexer_->SubStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS, 1024); // would underflow
        ASSERT_EQ(0, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
        ASSERT_EQ(0, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        expected_usage_vec[1] = 0;
        for (std::size_t i = 0; i != meta_indexer_->storage_usage_data_.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), meta_indexer_->storage_usage_data_.storage_usage_by_type_.at(i).load());
        }
    }
}

TEST_F(MetaIndexerTest, TestStorageUsageDataSeriDeseri) {
    MetaIndexer::StorageUsageData storage_usage_data;

    // Successful round-trip: serialize then deserialize
    {
        std::vector<std::uint64_t> expected_usage_vec{1, 100, 200, 300, 400, 500};

        storage_usage_data.Reset();
        for (std::size_t i = 0; i != expected_usage_vec.size(); ++i) {
            storage_usage_data.storage_usage_by_type_.at(i).store(expected_usage_vec.at(i));
        }

        std::string serialized = storage_usage_data.Serialize();
        ASSERT_EQ(R"({"unknown":1,"hf3fs":100,"mooncake":200,"pace":300,"file":400,"vcns_hf3fs":500})", serialized);

        storage_usage_data.Reset();
        ASSERT_EQ(EC_OK, storage_usage_data.Deserialize(serialized));
        for (std::size_t i = 0; i != storage_usage_data.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), storage_usage_data.storage_usage_by_type_.at(i).load());
        }
    }

    // Legal input: keys in different order
    {
        const std::vector<std::uint64_t> expected_usage_vec{1, 2, 3, 4, 5, 6};
        storage_usage_data.Reset();
        ASSERT_EQ(
            EC_OK,
            storage_usage_data.Deserialize(R"({"vcns_hf3fs":6,"file":5,"pace":4,"mooncake":3,"hf3fs":2,"unknown":1})"));
        for (std::size_t i = 0; i != storage_usage_data.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), storage_usage_data.storage_usage_by_type_.at(i).load());
        }
    }

    // Legal input: partial JSON (missing keys default to 0)
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 99);
        ASSERT_EQ(EC_OK, storage_usage_data.Deserialize(R"({"hf3fs":100,"mooncake":200})"));
        ASSERT_EQ(100, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        ASSERT_EQ(200, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
        ASSERT_EQ(0, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
    }

    // Legal input: whitespace-padded JSON
    {
        const std::vector<std::uint64_t> expected_usage_vec{1, 2, 3, 4, 5, 6};
        storage_usage_data.Reset();
        ASSERT_EQ(EC_OK,
                  storage_usage_data.Deserialize(
                      "  {\"unknown\":1,\"hf3fs\":2,\"mooncake\":3,\"pace\":4,\"file\":5,\"vcns_hf3fs\":6}  "));
        for (std::size_t i = 0; i != storage_usage_data.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), storage_usage_data.storage_usage_by_type_.at(i).load());
        }
    }

    // Empty string
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 888);
        ASSERT_EQ(EC_ERROR, storage_usage_data.Deserialize(""));
        // Original data must not be modified on error
        ASSERT_EQ(888, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }

    // Malformed data: not valid JSON (old comma-separated format)
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 777);
        ASSERT_EQ(EC_ERROR, storage_usage_data.Deserialize("0,abc,2,3,4,5"));
        // Original data must not be modified on error
        ASSERT_EQ(777, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }

    // Malformed data: JSON array instead of object
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 777);
        ASSERT_EQ(EC_ERROR, storage_usage_data.Deserialize("[1,2,3,4,5,6]"));
        // Original data must not be modified on error
        ASSERT_EQ(777, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }

    // Malformed data: non-integer value for a key
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 777);
        ASSERT_EQ(EC_ERROR,
                  storage_usage_data.Deserialize(
                      R"({"unknown":0,"hf3fs":"not_a_number","mooncake":2,"pace":3,"file":4,"vcns_hf3fs":5})"));
        // Original data must not be modified on error
        ASSERT_EQ(777, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }

    // Malformed data: floating-point value
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 777);
        ASSERT_EQ(EC_ERROR,
                  storage_usage_data.Deserialize(
                      R"({"unknown":0,"hf3fs":1.5,"mooncake":2,"pace":3,"file":4,"vcns_hf3fs":5})"));
        // Original data must not be modified on error
        ASSERT_EQ(777, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }

    // Unrecognized key: must be rejected and leave data unchanged
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 888);
        ASSERT_EQ(EC_ERROR, storage_usage_data.Deserialize(R"({"future_type": 999, "hf3fs": 10})"));
        // Original data must not be modified on error
        ASSERT_EQ(888, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }

    // "unknown" is a legitimate key and must still be accepted.
    {
        storage_usage_data.Reset();
        ASSERT_EQ(EC_OK, storage_usage_data.Deserialize(R"({"unknown":7,"hf3fs":10})"));
        ASSERT_EQ(7u,
                  storage_usage_data.storage_usage_by_type_
                      .at(static_cast<std::size_t>(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN))
                      .load());
        ASSERT_EQ(10, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }
}
