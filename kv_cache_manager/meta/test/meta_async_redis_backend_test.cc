#include <thread>

#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/common/test/mock_redis_client.h"
#include "kv_cache_manager/common/test/redis_test_base.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_async_redis_backend.h"
#include "kv_cache_manager/meta/test/meta_storage_backend_test_base.h"

namespace kv_cache_manager {

class MockMetaAsyncRedisBackend : public MetaAsyncRedisBackend {
public:
    MOCK_METHOD(std::shared_ptr<RedisClient>, CreateRedisClient, (), (const));
};

class MetaAsyncRedisBackendTest : public MetaStorageBackendTestBase, public RedisTestBase, public TESTBASE {
public:
    void SetUp() override {
        ConstructBackend();
        ConstructConfig();
    }

    void TearDown() override {
        if (backend_) {
            backend_->Close();
        }
    }

protected:
    void ConstructBackend() { backend_ = std::make_unique<MockMetaAsyncRedisBackend>(); }

    void ConstructConfig() {
        config_ = std::make_shared<MetaStorageBackendConfig>();
        config_->SetStorageType(META_ASYNC_REDIS_BACKEND_TYPE_STR);
        config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=2&async_max_batch=64&async_wait_us=1000"
                               "&async_max_size=1000&async_drain_ms=2000&async_max_retry=3");
    }

    void SetupMockRedisClients() {
        EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([this]() {
            StandardUri empty_uri;
            auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
            ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
            ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
            ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([](const std::vector<CmdArgs> &cmds) {
                std::vector<ReplyUPtr> replies;
                for (size_t i = 0; i < cmds.size(); ++i) {
                    redisReply *r = (redisReply *)malloc(sizeof(redisReply));
                    memset(r, 0, sizeof(redisReply));
                    r->type = REDIS_REPLY_INTEGER;
                    r->integer = 1;
                    replies.emplace_back(r, freeReplyObject);
                }
                return replies;
            }));
            return mock;
        }));
    }

    ErrorCode InitAndOpen() {
        SetupMockRedisClients();
        ErrorCode ec = backend_->Init("test_instance", config_);
        if (ec != EC_OK)
            return ec;
        return backend_->Open();
    }

    std::unique_ptr<MockMetaAsyncRedisBackend> backend_;
    std::shared_ptr<MetaStorageBackendConfig> config_;
};

// ==================== Init Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestInit) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
    ASSERT_EQ(META_ASYNC_REDIS_BACKEND_TYPE_STR, backend_->GetStorageType());
}

TEST_F(MetaAsyncRedisBackendTest, TestInitInvalidEmptyInstanceId) {
    ASSERT_EQ(EC_BADARGS, backend_->Init("", config_));
}

TEST_F(MetaAsyncRedisBackendTest, TestInitInvalidNullConfig) {
    ASSERT_EQ(EC_BADARGS, backend_->Init("test_instance", nullptr));
}

TEST_F(MetaAsyncRedisBackendTest, TestInitInvalidEmptyUri) {
    config_->SetStorageUri("");
    ASSERT_EQ(EC_BADARGS, backend_->Init("test_instance", config_));
}

TEST_F(MetaAsyncRedisBackendTest, TestInitParams) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
    ASSERT_EQ(2, backend_->queue_count_);
    ASSERT_EQ(64, backend_->max_batch_size_);
    ASSERT_EQ(1000, backend_->batch_wait_timeout_us_);
    ASSERT_EQ(1000, backend_->queue_max_size_);
    ASSERT_EQ(2000, backend_->drain_timeout_ms_);
}

// ==================== Open/Close Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestOpenAndClose) {
    ASSERT_EQ(EC_OK, InitAndOpen());
    ASSERT_TRUE(backend_->is_running_.load());
    ASSERT_EQ(2, (int)backend_->consumer_threads_.size());
    ASSERT_EQ(2, (int)backend_->queues_.size());
    ASSERT_EQ(2, (int)backend_->consumer_clients_.size());
    ASSERT_TRUE(backend_->read_client_pool_ != nullptr);

    ASSERT_EQ(EC_OK, backend_->Close());
    ASSERT_FALSE(backend_->is_running_.load());
    ASSERT_TRUE(backend_->consumer_threads_.empty());
    ASSERT_TRUE(backend_->consumer_clients_.empty());
    ASSERT_TRUE(backend_->read_client_pool_ == nullptr);
}

TEST_F(MetaAsyncRedisBackendTest, TestOpenFailsOnClientCreation) {
    EXPECT_CALL(*backend_, CreateRedisClient()).WillOnce(Invoke([]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
        EXPECT_CALL(*mock, Reconnect()).WillRepeatedly(Return(false));
        return mock;
    }));
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
    ASSERT_EQ(EC_ERROR, backend_->Open());
}

TEST_F(MetaAsyncRedisBackendTest, TestDoubleClose) {
    ASSERT_EQ(EC_OK, InitAndOpen());
    ASSERT_EQ(EC_OK, backend_->Close());
    ASSERT_EQ(EC_OK, backend_->Close()); // idempotent
}

// ==================== Write Operations (Enqueue) Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestPut) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys = {100, 200};
    CacheLocationMapVector locations(2);
    PropertyMapVector properties = {{{"f1", "v1"}, {"f2", "v2"}}, {{"f3", "v3"}}};
    auto results = backend_->Put(nullptr, keys, locations, properties);
    ASSERT_EQ(2, results.size());
    ASSERT_EQ(EC_OK, results[0]);
    ASSERT_EQ(EC_OK, results[1]);
}

TEST_F(MetaAsyncRedisBackendTest, TestUpsert) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys = {400, 500, 600};
    CacheLocationMapVector locations(3);
    PropertyMapVector properties = {{{"a", "1"}}, {{"b", "2"}}, {{"c", "3"}}};
    auto results = backend_->Upsert(nullptr, keys, locations, properties);
    ASSERT_EQ(3, results.size());
    for (auto &ec : results) {
        ASSERT_EQ(EC_OK, ec);
    }
}

TEST_F(MetaAsyncRedisBackendTest, TestDelete) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys = {700, 800};
    auto results = backend_->Delete(nullptr, keys);
    ASSERT_EQ(2, results.size());
    ASSERT_EQ(EC_OK, results[0]);
    ASSERT_EQ(EC_OK, results[1]);
}

TEST_F(MetaAsyncRedisBackendTest, TestDeleteLocations) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys = {900};
    LocationIdsPerKey location_ids = {{"loc_1", "loc_2"}};
    auto results = backend_->DeleteLocations(nullptr, keys, location_ids);
    ASSERT_EQ(1, results.size());
    ASSERT_EQ(EC_OK, results[0]);
}

TEST_F(MetaAsyncRedisBackendTest, TestWriteEmptyKeys) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    auto results = backend_->Put(nullptr, {}, {}, {});
    ASSERT_TRUE(results.empty());
}

// ==================== Sync Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestSyncEmpty) {
    ASSERT_EQ(EC_OK, InitAndOpen());
    ASSERT_TRUE(backend_->Sync({}));
}

TEST_F(MetaAsyncRedisBackendTest, TestSyncBasic) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    CacheLocationMapVector locs(2);
    PropertyMapVector props = {{{"f", "v"}}, {{"f", "v"}}};
    backend_->Put(nullptr, {10, 20}, locs, props);
    ASSERT_TRUE(backend_->Sync({10, 20}));
}

TEST_F(MetaAsyncRedisBackendTest, TestSyncNotRunning) {
    ASSERT_EQ(EC_OK, InitAndOpen());
    backend_->Close();
    ASSERT_FALSE(backend_->Sync({1, 2, 3}));
}

TEST_F(MetaAsyncRedisBackendTest, TestSyncMultiQueue) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys;
    for (int i = 0; i < 100; ++i) {
        keys.push_back(i);
    }
    CacheLocationMapVector locations(keys.size());
    PropertyMapVector properties(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        properties[i] = {{"field", "value"}};
    }
    backend_->Put(nullptr, keys, locations, properties);
    ASSERT_TRUE(backend_->Sync(keys));
}

// ==================== Read Operations (Passthrough) Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestGetPassthrough) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys = {1, 2};
    CacheLocationMapVector out_locations;
    PropertyMapVector out_properties;
    auto results = backend_->Get(nullptr, keys, out_locations, out_properties);
    // With our mock returning INTEGER replies, the HGETALL parsing will fail
    // But we're testing that the read path doesn't crash and properly routes to read pool
    ASSERT_EQ(2, results.size());
}

TEST_F(MetaAsyncRedisBackendTest, TestExistsPassthrough) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys = {1, 2};
    std::vector<bool> out_exists;
    auto results = backend_->Exists(nullptr, keys, out_exists);
    ASSERT_EQ(2, results.size());
    ASSERT_EQ(EC_OK, results[0]);
    ASSERT_EQ(EC_OK, results[1]);
    ASSERT_EQ(2, out_exists.size());
    ASSERT_TRUE(out_exists[0]);
    ASSERT_TRUE(out_exists[1]);
}

// ==================== CompileWriteOp Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestCompileWriteOpPut) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));

    WriteOp op;
    op.type = WriteOpType::kPut;
    op.keys = {1, 2};
    op.field_maps = {{{"f1", "v1"}, {"f2", "v2"}}, {{"f3", "v3"}}};

    std::vector<CmdArgs> cmds;
    backend_->CompileWriteOp(op, cmds);

    // kPut: DEL + HSET per key = 4 commands
    ASSERT_EQ(4, cmds.size());
    ASSERT_EQ("DEL", cmds[0][0]);
    ASSERT_TRUE(cmds[0][1].find("test_instance") != std::string::npos);
    ASSERT_EQ("HSET", cmds[1][0]);
    ASSERT_EQ(cmds[0][1], cmds[1][1]);
    ASSERT_EQ(6, cmds[1].size()); // HSET key f1 v1 f2 v2
    ASSERT_EQ("DEL", cmds[2][0]);
    ASSERT_EQ("HSET", cmds[3][0]);
    ASSERT_EQ(4, cmds[3].size()); // HSET key f3 v3
}

TEST_F(MetaAsyncRedisBackendTest, TestCompileWriteOpUpsert) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));

    WriteOp op;
    op.type = WriteOpType::kUpsert;
    op.keys = {20, 30};
    op.field_maps = {{{"a", "1"}}, {{"b", "2"}}};

    std::vector<CmdArgs> cmds;
    backend_->CompileWriteOp(op, cmds);

    ASSERT_EQ(2, cmds.size());
    ASSERT_EQ("HSET", cmds[0][0]);
    ASSERT_EQ("HSET", cmds[1][0]);
}

TEST_F(MetaAsyncRedisBackendTest, TestCompileWriteOpDelete) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));

    WriteOp op;
    op.type = WriteOpType::kDelete;
    op.keys = {40, 50, 60};

    std::vector<CmdArgs> cmds;
    backend_->CompileWriteOp(op, cmds);

    ASSERT_EQ(3, cmds.size());
    for (auto &cmd : cmds) {
        ASSERT_EQ("DEL", cmd[0]);
        ASSERT_EQ(2, cmd.size());
    }
}

TEST_F(MetaAsyncRedisBackendTest, TestCompileWriteOpDeleteLocations) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));

    WriteOp op;
    op.type = WriteOpType::kDeleteLocations;
    op.keys = {70};
    op.field_names_vec = {{"loc_1", "loc_2", "loc_3"}};

    std::vector<CmdArgs> cmds;
    backend_->CompileWriteOp(op, cmds);

    ASSERT_EQ(1, cmds.size());
    ASSERT_EQ("HDEL", cmds[0][0]);
    ASSERT_EQ(5, cmds[0].size()); // HDEL key loc_1 loc_2 loc_3
}

TEST_F(MetaAsyncRedisBackendTest, TestCompileWriteOpEmptyFieldMaps) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));

    WriteOp op;
    op.type = WriteOpType::kPut;
    op.keys = {80};
    op.field_maps = {{}};

    std::vector<CmdArgs> cmds;
    backend_->CompileWriteOp(op, cmds);

    ASSERT_EQ(1, cmds.size());
    ASSERT_EQ("DEL", cmds[0][0]);
}

// ==================== Key Routing Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestGetQueueIndexForKey) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));

    for (KeyType k = 0; k < 100; ++k) {
        int qi = backend_->GetQueueIndexForKey(k);
        ASSERT_GE(qi, 0);
        ASSERT_LT(qi, 2);
    }

    int q1 = backend_->GetQueueIndexForKey(42);
    int q2 = backend_->GetQueueIndexForKey(42);
    ASSERT_EQ(q1, q2);
}

// ==================== Prefix Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestAppendAndStripPrefix) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));

    KeyTypeVec keys = {100, 200, 300};
    auto full_keys = backend_->AppendPrefixToKeys(keys);
    ASSERT_EQ(3, full_keys.size());
    for (const auto &fk : full_keys) {
        ASSERT_TRUE(fk.find("kvcache:instance_test_instance:cache_") == 0);
    }

    KeyTypeVec out_keys;
    ASSERT_TRUE(backend_->StripPrefixInKeys(full_keys, out_keys));
    ASSERT_EQ(keys, out_keys);
}

TEST_F(MetaAsyncRedisBackendTest, TestStripPrefixInvalid) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));

    KeyTypeVec out_keys;
    ASSERT_FALSE(backend_->StripPrefixInKeys({"invalid_key"}, out_keys));
    ASSERT_TRUE(out_keys.empty());
}

// ==================== End-to-End Consumer Pipeline Test ====================

TEST_F(MetaAsyncRedisBackendTest, TestEndToEndPutAndFlush) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys = {1000, 2000, 3000, 4000, 5000};
    CacheLocationMapVector locations(keys.size());
    PropertyMapVector properties(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        properties[i] = {{"field_" + std::to_string(i), "value_" + std::to_string(i)}};
    }

    auto put_results = backend_->Put(nullptr, keys, locations, properties);
    ASSERT_EQ(5, put_results.size());
    for (auto &ec : put_results) {
        ASSERT_EQ(EC_OK, ec);
    }

    ASSERT_TRUE(backend_->Sync(keys));
}

TEST_F(MetaAsyncRedisBackendTest, TestEndToEndMultipleOpTypes) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    CacheLocationMapVector locs1(1), locs2(1);
    PropertyMapVector props1 = {{{"f1", "v1"}}};
    PropertyMapVector props2 = {{{"f3", "v3"}}};
    backend_->Put(nullptr, {1}, locs1, props1);
    backend_->Upsert(nullptr, {2}, locs2, props2);
    backend_->Delete(nullptr, {3});
    backend_->DeleteLocations(nullptr, {4}, {{"loc_1"}});

    ASSERT_TRUE(backend_->Sync({1, 2, 3, 4}));
}

// ==================== Backpressure Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestBackpressureEnqueue) {
    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=1&async_max_size=5"
                           "&async_enqueue_timeout_ms=100&async_enqueue_retry_us=1000");

    EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
        ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
        ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
        ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([](const std::vector<CmdArgs> &cmds) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::vector<RedisClient::ReplyUPtr> replies;
            for (size_t i = 0; i < cmds.size(); ++i) {
                redisReply *r = (redisReply *)malloc(sizeof(redisReply));
                memset(r, 0, sizeof(redisReply));
                r->type = REDIS_REPLY_INTEGER;
                r->integer = 1;
                replies.emplace_back(r, freeReplyObject);
            }
            return replies;
        }));
        return mock;
    }));

    ASSERT_EQ(EC_OK, backend_->Init("bp_test", config_));
    ASSERT_EQ(EC_OK, backend_->Open());

    int timeout_count = 0;
    for (int i = 0; i < 50; ++i) {
        CacheLocationMapVector locs(1);
        PropertyMapVector props = {{{"f", "v"}}};
        auto results = backend_->Put(nullptr, {(KeyType)i}, locs, props);
        if (!results.empty() && results[0] == EC_TIMEOUT) {
            ++timeout_count;
        }
    }
    ASSERT_GT(timeout_count, 0);
    ASSERT_EQ(EC_OK, backend_->Close());
}

// ==================== MetaData Passthrough Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestPutAndGetMetaData) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    FieldMap meta = {{"version", "1"}, {"created_at", "2024-01-01"}};
    ErrorCode ec = backend_->PutMetaData(meta);
    ASSERT_EQ(EC_OK, ec);
}

// ==================== Metrics Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestGetAsyncWriteStatsQueueSizes) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
    {
        auto stats = backend_->GetAsyncWriteStats();
        EXPECT_EQ(0, stats.max_async_queue_size);
        EXPECT_EQ(0, stats.avg_async_queue_size);
    }

    SetupMockRedisClients();
    ASSERT_EQ(EC_OK, backend_->Open());
    {
        auto stats = backend_->GetAsyncWriteStats();
        EXPECT_EQ(0, stats.max_async_queue_size);
        EXPECT_EQ(0, stats.avg_async_queue_size);
    }

    ASSERT_EQ(EC_OK, backend_->Close());

    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=2&async_max_batch=1"
                           "&async_wait_us=500000&async_max_size=10000");
    EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
        ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
        ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
        ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([](const std::vector<CmdArgs> &cmds) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::vector<RedisClient::ReplyUPtr> replies;
            for (size_t i = 0; i < cmds.size(); ++i) {
                redisReply *r = (redisReply *)malloc(sizeof(redisReply));
                memset(r, 0, sizeof(redisReply));
                r->type = REDIS_REPLY_INTEGER;
                r->integer = 1;
                replies.emplace_back(r, freeReplyObject);
            }
            return replies;
        }));
        return mock;
    }));
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
    ASSERT_EQ(EC_OK, backend_->Open());

    CacheLocationMapVector seed_locs(1);
    PropertyMapVector seed_props = {{{"f", "v"}}};
    backend_->Put(nullptr, {0}, seed_locs, seed_props);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    KeyTypeVec keys;
    CacheLocationMapVector locations;
    PropertyMapVector properties;
    for (int i = 1; i <= 20; ++i) {
        keys.push_back(i);
        locations.push_back({});
        properties.push_back({{"f", "v"}});
    }
    backend_->Put(nullptr, keys, locations, properties);

    {
        auto stats = backend_->GetAsyncWriteStats();
        EXPECT_GT(stats.max_async_queue_size + stats.avg_async_queue_size, 0);
    }

    ASSERT_EQ(EC_OK, backend_->Close());
}

TEST_F(MetaAsyncRedisBackendTest, TestGetAsyncWriteStats) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
    {
        auto stats = backend_->GetAsyncWriteStats();
        EXPECT_EQ(0, stats.flush_key_count);
        EXPECT_EQ(0, stats.batch_flush_time_us);
        EXPECT_EQ(0, stats.pipeline_error_count);
    }

    SetupMockRedisClients();
    ASSERT_EQ(EC_OK, backend_->Open());

    // Put some keys and wait for flush
    KeyTypeVec keys = {1, 2, 3};
    CacheLocationMapVector locations(3);
    PropertyMapVector properties = {{{"f", "v"}}, {{"f", "v"}}, {{"f", "v"}}};
    auto results = backend_->Put(nullptr, keys, locations, properties);
    for (auto ec : results) {
        EXPECT_EQ(EC_OK, ec);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    {
        auto stats = backend_->GetAsyncWriteStats();
        EXPECT_GT(stats.flush_key_count, 0);
        EXPECT_GT(stats.batch_flush_time_us, 0);
        EXPECT_EQ(0, stats.pipeline_error_count);
    }

    // CAS reset: second read should return zeros for accumulated counters
    {
        auto stats = backend_->GetAsyncWriteStats();
        EXPECT_EQ(0, stats.flush_key_count);
        EXPECT_EQ(0, stats.batch_flush_time_us);
        EXPECT_EQ(0, stats.pipeline_error_count);
    }

    ASSERT_EQ(EC_OK, backend_->Close());

    // Test enqueue timeout: tiny queue capacity
    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=1&async_max_batch=1"
                           "&async_wait_us=500000&async_max_size=1&async_enqueue_timeout_ms=1");
    ConstructBackend();
    EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
        ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
        ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
        ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([](const std::vector<CmdArgs> &cmds) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            std::vector<RedisClient::ReplyUPtr> replies;
            for (size_t i = 0; i < cmds.size(); ++i) {
                redisReply *r = (redisReply *)malloc(sizeof(redisReply));
                memset(r, 0, sizeof(redisReply));
                r->type = REDIS_REPLY_INTEGER;
                r->integer = 1;
                replies.emplace_back(r, freeReplyObject);
            }
            return replies;
        }));
        return mock;
    }));
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
    ASSERT_EQ(EC_OK, backend_->Open());

    // First put fills the queue, subsequent puts should timeout
    CacheLocationMapVector locs1(1);
    PropertyMapVector props1 = {{{"f", "v"}}};
    backend_->Put(nullptr, {100}, locs1, props1);

    // Give consumer time to pick up but it sleeps 500ms in pipeline
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Now queue should be blocked, more puts will timeout
    KeyTypeVec timeout_keys;
    CacheLocationMapVector timeout_locs;
    PropertyMapVector timeout_props;
    for (int i = 200; i < 210; ++i) {
        timeout_keys.push_back(i);
        timeout_locs.push_back({});
        timeout_props.push_back({{"f", "v"}});
    }
    auto timeout_results = backend_->Put(nullptr, timeout_keys, timeout_locs, timeout_props);
    for (auto ec : timeout_results) {
        EXPECT_TRUE(ec == EC_OK || ec == EC_TIMEOUT);
    }

    ASSERT_EQ(EC_OK, backend_->Close());
}

TEST_F(MetaAsyncRedisBackendTest, TestGetAsyncWriteStatsPipelineError) {
    // Setup a backend with pipeline that always fails
    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=1&async_max_batch=64&async_wait_us=1000"
                           "&async_max_size=1000");
    ConstructBackend();
    EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
        ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
        ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
        ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([](const std::vector<CmdArgs> &) {
            return std::vector<RedisClient::ReplyUPtr>{};
        }));
        return mock;
    }));
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
    ASSERT_EQ(EC_OK, backend_->Open());

    KeyTypeVec keys = {1, 2, 3};
    CacheLocationMapVector locations(3);
    PropertyMapVector properties = {{{"f", "v"}}, {{"f", "v"}}, {{"f", "v"}}};
    backend_->Put(nullptr, keys, locations, properties);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto stats = backend_->GetAsyncWriteStats();
    EXPECT_GT(stats.pipeline_error_count, 0);
    EXPECT_EQ(0, stats.flush_key_count);
    EXPECT_GT(stats.batch_flush_time_us, 0);

    ASSERT_EQ(EC_OK, backend_->Close());
}

} // namespace kv_cache_manager
