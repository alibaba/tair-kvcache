#include <thread>

#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_redis_backend.h"

namespace kv_cache_manager {
class MetaRedisBackendRealServiceTest : public TESTBASE {
public:
    void SetUp() override;

    void TearDown() override {}

    void ConstructMetaRedisBackend();
    void ConstructMetaStorageBackendConfig();

private:
    using KeyType = MetaStorageBackend::KeyType;
    using KeyTypeVec = MetaStorageBackend::KeyTypeVec;
    using FieldMap = MetaStorageBackend::FieldMap;
    using FieldMapVec = MetaStorageBackend::FieldMapVec;
    std::unique_ptr<MetaRedisBackend> meta_redis_backend_;
    std::shared_ptr<MetaStorageBackendConfig> meta_storage_backend_config_;
};

void MetaRedisBackendRealServiceTest::SetUp() {
    ConstructMetaRedisBackend();
    ConstructMetaStorageBackendConfig();
}

void MetaRedisBackendRealServiceTest::ConstructMetaRedisBackend() {
    meta_redis_backend_ = std::make_unique<MetaRedisBackend>();
}

void MetaRedisBackendRealServiceTest::ConstructMetaStorageBackendConfig() {
    meta_storage_backend_config_ = std::make_shared<MetaStorageBackendConfig>();
    meta_storage_backend_config_->SetStorageType(META_REDIS_BACKEND_TYPE_STR);
    meta_storage_backend_config_->SetStorageUri(
        "redis://test_redis_user:test_redis_password@localhost:6379/?client_max_pool_size=4");
}

TEST_F(MetaRedisBackendRealServiceTest, TestOpenAndClose) {
    meta_storage_backend_config_->SetStorageUri(
        "redis://test_redis_user:test_redis_password@localhost:6379/?client_max_pool_size=2");
    ASSERT_EQ(EC_OK, meta_redis_backend_->Init("test_open_and_close", meta_storage_backend_config_));
    // open first
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());
    {
        auto handle1 = meta_redis_backend_->client_pool_->AcquireClient();
        ASSERT_TRUE(handle1);
        auto handle2 = meta_redis_backend_->client_pool_->AcquireClient();
        ASSERT_TRUE(handle2);
        auto handle3 = meta_redis_backend_->client_pool_->AcquireClient();
        ASSERT_FALSE(handle3);
    }
    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
    ASSERT_TRUE(meta_redis_backend_->client_pool_ == nullptr);
    // reopen second
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());
    {
        auto handle1 = meta_redis_backend_->client_pool_->AcquireClient();
        ASSERT_TRUE(handle1);
        auto handle2 = meta_redis_backend_->client_pool_->AcquireClient();
        ASSERT_TRUE(handle2);
        auto handle3 = meta_redis_backend_->client_pool_->AcquireClient();
        ASSERT_FALSE(handle3);
    }
    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
    ASSERT_TRUE(meta_redis_backend_->client_pool_ == nullptr);
}

TEST_F(MetaRedisBackendRealServiceTest, TestMultiThreadSimple) {
    ASSERT_EQ(EC_OK, meta_redis_backend_->Init("test_multi_thread_simple", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());

    auto MyKey = [](int64_t i) { return i; };
    auto MyMap = [](int64_t i) {
        return FieldMap{{"f1", "v" + std::to_string(i) + "_1"}, {"f2", "v" + std::to_string(i) + "_2"}};
    };
    constexpr int COUNT = 20;
    std::vector<KeyTypeVec> keys_vec;
    std::vector<FieldMapVec> field_maps_vec;
    keys_vec.reserve(COUNT);
    field_maps_vec.reserve(COUNT);
    for (int i = 0; i < COUNT; ++i) {
        keys_vec.emplace_back(KeyTypeVec{MyKey(i), MyKey(i + 100)});
        field_maps_vec.emplace_back(FieldMapVec{MyMap(i), MyMap(i + 100)});
    }

    // put
    std::atomic<int> put_index = 0;
    auto put_task = [&, this]() {
        int i = put_index++;
        auto ec_per_key = meta_redis_backend_->Put(keys_vec[i], field_maps_vec[i]);
        ASSERT_EQ(std::vector<ErrorCode>(keys_vec[i].size(), EC_OK), ec_per_key);
    };
    std::vector<std::thread> put_threads;
    for (int i = 0; i < COUNT; ++i) {
        put_threads.emplace_back(put_task);
    }
    for (auto &thread : put_threads) {
        thread.join();
    }

    // get
    std::atomic<int> get_index = 0;
    auto get_task = [&, this]() {
        int i = get_index++;
        FieldMapVec out_field_map;
        auto ec_per_key = meta_redis_backend_->Get(keys_vec[i], {"f1", "f2"}, out_field_map);
        ASSERT_EQ(std::vector<ErrorCode>(keys_vec[i].size(), EC_OK), ec_per_key);
        ASSERT_EQ(field_maps_vec[i], out_field_map);
    };
    std::vector<std::thread> get_threads;
    for (int i = 0; i < COUNT; ++i) {
        get_threads.emplace_back(get_task);
    }
    for (auto &thread : get_threads) {
        thread.join();
    }

    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
}

TEST_F(MetaRedisBackendRealServiceTest, TestConcurrentMixedOperations) {
    ASSERT_EQ(EC_OK, meta_redis_backend_->Init("test_concurrent_mixed_operations", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());

    auto MyKey = [](int64_t i) { return i; };
    auto MyMap = [](int64_t i) {
        return FieldMap{{"name", "item" + std::to_string(i)}, {"value", std::to_string(i * 10)}, {"updated", "false"}};
    };

    constexpr int COUNT = 12;
    std::vector<KeyTypeVec> keys_vec;
    std::vector<FieldMapVec> field_maps_vec;
    keys_vec.reserve(COUNT);
    field_maps_vec.reserve(COUNT);
    for (int i = 0; i < COUNT; ++i) {
        keys_vec.emplace_back(KeyTypeVec{MyKey(i), MyKey(i + 1000)});
        field_maps_vec.emplace_back(FieldMapVec{MyMap(i), MyMap(i + 1000)});
        auto ec_per_key = meta_redis_backend_->Delete(keys_vec.back());
        for (const auto &ec : ec_per_key) {
            ASSERT_TRUE(ec == EC_OK || ec == EC_NOENT);
        }
    }

    // Mix of PUT, GET, UPDATE, and EXISTS operations
    std::atomic<int> operation_count{0};
    auto mixed_task = [&, this]() {
        int op_id = operation_count++;
        int item_id = op_id % COUNT;
        KeyTypeVec &keys = keys_vec[item_id];
        FieldMapVec &field_maps = field_maps_vec[item_id];

        switch (op_id % 4) {
        case 0: // PUT
        {
            auto ec_per_key = meta_redis_backend_->Put(keys, field_maps);
            ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
        } break;
        case 1: // GET
        {
            FieldMapVec out_field_maps;
            std::vector<std::string> field_names = {"name", "value", "updated", "ts"};
            auto ec_per_key = meta_redis_backend_->Get(keys, field_names, out_field_maps);
            ASSERT_EQ(keys.size(), ec_per_key.size());
            ASSERT_EQ(keys.size(), out_field_maps.size());
            for (int i = 0; i < ec_per_key.size(); ++i) {
                auto ec = ec_per_key[i];
                ASSERT_TRUE(ec == EC_OK) << ec;
                if (!out_field_maps[i]["name"].empty()) {
                    ASSERT_EQ(field_maps[i]["name"], out_field_maps[i]["name"]);
                    ASSERT_EQ(field_maps[i]["value"], out_field_maps[i]["value"]);
                    ASSERT_TRUE(out_field_maps[i]["updated"] == "true" || out_field_maps[i]["updated"] == "false");
                    bool is_updated = out_field_maps[i]["updated"] == "true";
                    ASSERT_EQ(is_updated ? std::string("1234") : std::string(), out_field_maps[i]["ts"]);
                } else {
                    ASSERT_EQ(std::string(), out_field_maps[i]["name"]);
                    ASSERT_EQ(std::string(), out_field_maps[i]["value"]);
                    ASSERT_EQ(std::string(), out_field_maps[i]["updated"]);
                    ASSERT_EQ(std::string(), out_field_maps[i]["ts"]);
                }
            }
        } break;
        case 2: // UPDATE
        {
            FieldMap update_map = {{"updated", "true"}, {"ts", "1234"}};
            FieldMapVec update_maps(keys.size(), update_map);
            auto ec_per_key = meta_redis_backend_->UpdateFields(keys, update_maps);
            ASSERT_EQ(keys.size(), ec_per_key.size());
            for (auto ec : ec_per_key) {
                ASSERT_TRUE(ec == EC_OK || ec == EC_NOENT) << ec;
            }
        } break;
        case 3: // EXISTS
        {
            std::vector<bool> out_is_exist_vec;
            auto ec_per_key = meta_redis_backend_->Exists(keys, out_is_exist_vec);
            ASSERT_EQ(keys.size(), out_is_exist_vec.size());
            ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
        } break;
        }
    };

    // Create many threads to increase concurrency
    std::vector<std::thread> threads;
    for (int i = 0; i < COUNT * 4; ++i) {
        threads.emplace_back(mixed_task);
    }
    for (auto &thread : threads) {
        thread.join();
    }

    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
}

TEST_F(MetaRedisBackendRealServiceTest, TestConcurrentListAndRandomOperations) {
    ASSERT_EQ(EC_OK, meta_redis_backend_->Init("test_concurrent_list_random", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());

    auto MyKey = [](int64_t i) { return i; };
    auto MyMap = [](int64_t i) {
        return FieldMap{{"name", "item" + std::to_string(i)}, {"value", std::to_string(i * 10)}};
    };

    constexpr int COUNT = 10;
    for (int i = 0; i < COUNT; ++i) {
        KeyTypeVec keys = {MyKey(i), MyKey(i + COUNT)};
        FieldMapVec field_maps = {MyMap(i), MyMap(i + COUNT)};
        auto ec_per_key = meta_redis_backend_->Put(keys, field_maps);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    }

    std::atomic<int> operation_count{0};
    auto list_random_task = [&, this]() {
        int op_id = operation_count++;
        if (op_id % 2 == 0) {
            std::string next_cursor;
            KeyTypeVec out_keys;
            ErrorCode ec = meta_redis_backend_->ListKeys(SCAN_BASE_CURSOR, 10, next_cursor, out_keys);
            ASSERT_EQ(EC_OK, ec);
            for (const KeyType &key : out_keys) {
                ASSERT_TRUE(key >= 0 && key <= COUNT * 2);
            }
        } else {
            std::vector<KeyType> out_keys;
            ErrorCode ec = meta_redis_backend_->RandomSample(5, out_keys);
            ASSERT_EQ(EC_OK, ec);
            for (const KeyType &key : out_keys) {
                ASSERT_TRUE(key >= 0 && key <= COUNT * 2);
            }
        }
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < 20; ++i) {
        threads.emplace_back(list_random_task);
    }
    for (auto &thread : threads) {
        thread.join();
    }

    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
}

TEST_F(MetaRedisBackendRealServiceTest, TestScanUntilBaseCursor) {
    ASSERT_EQ(EC_OK, meta_redis_backend_->Init("test_scan_until_base_cursor", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());

    auto MyKey = [](int64_t i) { return i; };
    auto MyMap = [](int64_t i) {
        return FieldMap{{"f1", "v" + std::to_string(i) + "_1"}, {"f2", "v" + std::to_string(i) + "_2"}};
    };

    // Create 50 keys for testing
    constexpr int COUNT = 5;
    std::vector<KeyTypeVec> keys_vec;
    std::vector<FieldMapVec> field_maps_vec;
    keys_vec.reserve(COUNT);
    field_maps_vec.reserve(COUNT);

    for (int i = 0; i < COUNT; ++i) {
        keys_vec.emplace_back(KeyTypeVec{MyKey(i), MyKey(i + 1000)});
        field_maps_vec.emplace_back(FieldMapVec{MyMap(i), MyMap(i + 1000)});
    }

    // Put all keys
    for (int i = 0; i < COUNT; ++i) {
        auto ec_per_key = meta_redis_backend_->Put(keys_vec[i], field_maps_vec[i]);
        ASSERT_EQ(std::vector<ErrorCode>(keys_vec[i].size(), EC_OK), ec_per_key);
    }

    // List keys in a loop until next cursor is base cursor
    std::vector<KeyType> all_listed_keys;
    std::string current_cursor = SCAN_BASE_CURSOR;
    std::string next_cursor;
    int loop_count = 0;
    const int max_loops = 30; // Safety limit to prevent infinite loop

    do {
        KeyTypeVec batch_keys;
        auto ec = meta_redis_backend_->ListKeys(current_cursor, /*limit*/ 50, next_cursor, batch_keys);
        ASSERT_EQ(EC_OK, ec);

        // Check that batch_keys doesn't contain any keys that are already in all_listed_keys
        for (const auto &key : batch_keys) {
            auto it = std::find(all_listed_keys.begin(), all_listed_keys.end(), key);
            ASSERT_TRUE(it == all_listed_keys.end()) << "Duplicate key " << key << " found in batch_keys";
        }

        // Add listed keys to our collection
        all_listed_keys.insert(all_listed_keys.end(), batch_keys.begin(), batch_keys.end());

        // Move to next cursor
        current_cursor = next_cursor;
        loop_count++;

        // Safety check to prevent infinite loop
        ASSERT_LT(loop_count, max_loops) << "ListKeys loop exceeded maximum iterations";

    } while (next_cursor != SCAN_BASE_CURSOR);

    // Verify that we've listed all keys
    ASSERT_EQ(COUNT * 2, all_listed_keys.size());

    // Verify that all inserted keys are in the listed results
    for (const auto &keys : keys_vec) {
        for (const auto &key : keys) {
            auto it = std::find(all_listed_keys.begin(), all_listed_keys.end(), key);
            ASSERT_TRUE(it != all_listed_keys.end()) << "Key " << key << " not found in listed results";
        }
    }

    // Clean up: delete all keys
    for (int i = 0; i < COUNT; ++i) {
        auto ec_per_key = meta_redis_backend_->Delete(keys_vec[i]);
        ASSERT_EQ(std::vector<ErrorCode>(keys_vec[i].size(), EC_OK), ec_per_key);
    }

    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
}

TEST_F(MetaRedisBackendRealServiceTest, TestMultiOpen) {
    ASSERT_EQ(EC_OK, meta_redis_backend_->Init("test_multi_thread_simple", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());

    auto MyKey = [](int64_t i) { return i; };
    auto MyMap = [](int64_t i) {
        return FieldMap{{"f1", "v" + std::to_string(i) + "_1"}, {"f2", "v" + std::to_string(i) + "_2"}};
    };
    constexpr int COUNT = 20;
    std::vector<KeyTypeVec> keys_vec;
    std::vector<FieldMapVec> field_maps_vec;
    keys_vec.reserve(COUNT);
    field_maps_vec.reserve(COUNT);
    for (int i = 0; i < COUNT; ++i) {
        keys_vec.emplace_back(KeyTypeVec{MyKey(i), MyKey(i + 100)});
        field_maps_vec.emplace_back(FieldMapVec{MyMap(i), MyMap(i + 100)});
    }

    // put
    std::atomic<int> put_index = 0;
    auto put_task = [&, this]() {
        int i = put_index++;
        auto ec_per_key = meta_redis_backend_->Put(keys_vec[i], field_maps_vec[i]);
        ASSERT_EQ(std::vector<ErrorCode>(keys_vec[i].size(), EC_OK), ec_per_key);
    };
    std::vector<std::thread> put_threads;
    for (int i = 0; i < COUNT / 2; ++i) {
        put_threads.emplace_back(put_task);
    }
    for (auto &thread : put_threads) {
        thread.join();
    }
    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());
    put_threads.clear();
    for (int i = COUNT / 2; i < COUNT; ++i) {
        put_threads.emplace_back(put_task);
    }
    for (auto &thread : put_threads) {
        thread.join();
    }

    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());

    // get
    std::atomic<int> get_index = 0;
    auto get_task = [&, this]() {
        int i = get_index++;
        FieldMapVec out_field_map;
        auto ec_per_key = meta_redis_backend_->Get(keys_vec[i], {"f1", "f2"}, out_field_map);
        ASSERT_EQ(std::vector<ErrorCode>(keys_vec[i].size(), EC_OK), ec_per_key);
        ASSERT_EQ(field_maps_vec[i], out_field_map);
    };
    std::vector<std::thread> get_threads;
    for (int i = 0; i < COUNT; ++i) {
        get_threads.emplace_back(get_task);
    }
    for (auto &thread : get_threads) {
        thread.join();
    }
    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());
    get_threads.clear();
    get_index = 0;
    for (int i = 0; i < COUNT; ++i) {
        get_threads.emplace_back(get_task);
    }
    for (auto &thread : get_threads) {
        thread.join();
    }

    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
}

} // namespace kv_cache_manager
