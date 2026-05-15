#include <thread>

#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/common/test/mock_redis_client.h"
#include "kv_cache_manager/common/test/redis_test_base.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
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
            auto mock = std::make_shared<MockRedisClient>(empty_uri);
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
        auto mock = std::make_shared<MockRedisClient>(empty_uri);
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
    FieldMapVec field_maps = {
        {{"f1", "v1"}, {"f2", "v2"}},
        {{"f3", "v3"}},
    };
    auto results = backend_->Put(keys, field_maps);
    ASSERT_EQ(2, results.size());
    ASSERT_EQ(EC_OK, results[0]);
    ASSERT_EQ(EC_OK, results[1]);
}

TEST_F(MetaAsyncRedisBackendTest, TestUpdateFields) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys = {300};
    FieldMapVec field_maps = {{{"f1", "updated"}}};
    auto results = backend_->UpdateFields(keys, field_maps);
    ASSERT_EQ(1, results.size());
    ASSERT_EQ(EC_OK, results[0]);
}

TEST_F(MetaAsyncRedisBackendTest, TestUpsert) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys = {400, 500, 600};
    FieldMapVec field_maps = {{{"a", "1"}}, {{"b", "2"}}, {{"c", "3"}}};
    auto results = backend_->Upsert(keys, field_maps);
    ASSERT_EQ(3, results.size());
    for (auto &ec : results) {
        ASSERT_EQ(EC_OK, ec);
    }
}

TEST_F(MetaAsyncRedisBackendTest, TestDelete) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys = {700, 800};
    auto results = backend_->Delete(keys);
    ASSERT_EQ(2, results.size());
    ASSERT_EQ(EC_OK, results[0]);
    ASSERT_EQ(EC_OK, results[1]);
}

TEST_F(MetaAsyncRedisBackendTest, TestDeleteFields) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys = {900};
    std::vector<std::vector<std::string>> field_names_vec = {{"loc_1", "loc_2"}};
    auto results = backend_->DeleteFields(keys, field_names_vec);
    ASSERT_EQ(1, results.size());
    ASSERT_EQ(EC_OK, results[0]);
}

TEST_F(MetaAsyncRedisBackendTest, TestWriteEmptyKeys) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    auto results = backend_->Put({}, {});
    ASSERT_TRUE(results.empty());
}

// ==================== Sync Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestSyncEmpty) {
    ASSERT_EQ(EC_OK, InitAndOpen());
    ASSERT_TRUE(backend_->Sync({}));
}

TEST_F(MetaAsyncRedisBackendTest, TestSyncBasic) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    // Put some data first
    backend_->Put({10, 20}, {{{"f", "v"}}, {{"f", "v"}}});
    // Sync should wait for the consumer to process
    ASSERT_TRUE(backend_->Sync({10, 20}));
}

TEST_F(MetaAsyncRedisBackendTest, TestSyncNotRunning) {
    ASSERT_EQ(EC_OK, InitAndOpen());
    backend_->Close();
    ASSERT_FALSE(backend_->Sync({1, 2, 3}));
}

TEST_F(MetaAsyncRedisBackendTest, TestSyncMultiQueue) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    // Generate keys that hash to different queues (with 2 queues)
    KeyTypeVec keys;
    for (int i = 0; i < 100; ++i) {
        keys.push_back(i);
    }
    FieldMapVec field_maps;
    for (size_t i = 0; i < keys.size(); ++i) {
        field_maps.push_back({{"field", "value"}});
    }
    backend_->Put(keys, field_maps);

    // Sync all keys - should touch both queues
    ASSERT_TRUE(backend_->Sync(keys));
}

// ==================== Read Operations (Passthrough) Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestGetPassthrough) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys = {1, 2};
    std::vector<std::string> field_names = {"f1", "f2"};
    FieldMapVec out_field_maps;
    auto results = backend_->Get(keys, field_names, out_field_maps);
    // With our mock returning INTEGER replies, the HMGET parsing will fail
    // But we're testing that the read path doesn't crash and properly routes to read pool
    ASSERT_EQ(2, results.size());
}

TEST_F(MetaAsyncRedisBackendTest, TestExistsPassthrough) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    KeyTypeVec keys = {1, 2};
    std::vector<bool> out_exists;
    auto results = backend_->Exists(keys, out_exists);
    // Mock returns INTEGER=1 for EXISTS, so keys exist
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
    // Key 1: DEL
    ASSERT_EQ("DEL", cmds[0][0]);
    ASSERT_TRUE(cmds[0][1].find("test_instance") != std::string::npos);
    // Key 1: HSET
    ASSERT_EQ("HSET", cmds[1][0]);
    ASSERT_EQ(cmds[0][1], cmds[1][1]); // same key
    ASSERT_EQ(6, cmds[1].size());      // HSET key f1 v1 f2 v2
    // Key 2: DEL
    ASSERT_EQ("DEL", cmds[2][0]);
    // Key 2: HSET
    ASSERT_EQ("HSET", cmds[3][0]);
    ASSERT_EQ(4, cmds[3].size()); // HSET key f3 v3
}

TEST_F(MetaAsyncRedisBackendTest, TestCompileWriteOpUpdateFields) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));

    WriteOp op;
    op.type = WriteOpType::kUpdateFields;
    op.keys = {10};
    op.field_maps = {{{"field", "value"}}};

    std::vector<CmdArgs> cmds;
    backend_->CompileWriteOp(op, cmds);

    // kUpdateFields: HSET only
    ASSERT_EQ(1, cmds.size());
    ASSERT_EQ("HSET", cmds[0][0]);
    ASSERT_EQ(4, cmds[0].size()); // HSET key field value
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

TEST_F(MetaAsyncRedisBackendTest, TestCompileWriteOpDeleteFields) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));

    WriteOp op;
    op.type = WriteOpType::kDeleteFields;
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

    // kPut with empty field_maps - should still DEL but no HSET
    WriteOp op;
    op.type = WriteOpType::kPut;
    op.keys = {80};
    op.field_maps = {{}}; // empty map for key 80

    std::vector<CmdArgs> cmds;
    backend_->CompileWriteOp(op, cmds);

    // BuildSetCmds generates DEL always, and skips HSET when field_map is empty
    ASSERT_EQ(1, cmds.size());
    ASSERT_EQ("DEL", cmds[0][0]);
}

// ==================== Key Routing Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestGetQueueIndexForKey) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));

    // With 2 queues, all keys should map to 0 or 1
    for (KeyType k = 0; k < 100; ++k) {
        int qi = backend_->GetQueueIndexForKey(k);
        ASSERT_GE(qi, 0);
        ASSERT_LT(qi, 2);
    }

    // Same key always routes to same queue
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

    // Write multiple keys
    KeyTypeVec keys = {1000, 2000, 3000, 4000, 5000};
    FieldMapVec field_maps;
    for (size_t i = 0; i < keys.size(); ++i) {
        field_maps.push_back({{"field_" + std::to_string(i), "value_" + std::to_string(i)}});
    }

    auto put_results = backend_->Put(keys, field_maps);
    ASSERT_EQ(5, put_results.size());
    for (auto &ec : put_results) {
        ASSERT_EQ(EC_OK, ec);
    }

    // Sync ensures all are consumed
    ASSERT_TRUE(backend_->Sync(keys));
}

TEST_F(MetaAsyncRedisBackendTest, TestEndToEndMultipleOpTypes) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    backend_->Put({1}, {{{"f1", "v1"}}});
    backend_->UpdateFields({1}, {{{"f2", "v2"}}});
    backend_->Upsert({2}, {{{"f3", "v3"}}});
    backend_->Delete({3});
    backend_->DeleteFields({4}, {{"loc_1"}});

    // Flush all
    ASSERT_TRUE(backend_->Sync({1, 2, 3, 4}));
}

// ==================== Backpressure Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestBackpressureEnqueue) {
    // Use very small queue to trigger backpressure
    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=1&async_max_size=5"
                           "&async_enqueue_timeout_ms=100&async_enqueue_retry_us=1000");

    // Make consumer block by using a slow TryExecPipeline
    EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<MockRedisClient>(empty_uri);
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

    // Fill queue beyond max_size rapidly
    int timeout_count = 0;
    for (int i = 0; i < 50; ++i) {
        auto results = backend_->Put({(KeyType)i}, {{{"f", "v"}}});
        if (!results.empty() && results[0] == EC_TIMEOUT) {
            ++timeout_count;
        }
    }
    // With max_size=5 and slow consumer, we should see some timeouts
    ASSERT_GT(timeout_count, 0);
    ASSERT_EQ(EC_OK, backend_->Close());
}

// ==================== MetaData Passthrough Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestPutAndGetMetaData) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    FieldMap meta = {{"version", "1"}, {"created_at", "2024-01-01"}};
    // PutMetaData uses Set which returns INTEGER replies - our mock returns integer=1
    ErrorCode ec = backend_->PutMetaData(meta);
    ASSERT_EQ(EC_OK, ec);
}

// ==================== Metrics Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestGetAsyncQueueSizes) {
    // Before Open(), queues_ is empty
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
    EXPECT_TRUE(backend_->GetAsyncQueueSizes().empty());

    // After Open(), should have queue_count (2) entries, all zero
    SetupMockRedisClients();
    ASSERT_EQ(EC_OK, backend_->Open());
    {
        auto sizes = backend_->GetAsyncQueueSizes();
        ASSERT_EQ(2, sizes.size());
        EXPECT_EQ(0, sizes[0]);
        EXPECT_EQ(0, sizes[1]);
    }

    // After Close() + reopen with slow consumer, enqueue and observe non-zero sizes
    ASSERT_EQ(EC_OK, backend_->Close());

    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=2&async_max_batch=64"
                           "&async_wait_us=500000&async_max_size=10000");
    EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<MockRedisClient>(empty_uri);
        ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
        ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
        ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([](const std::vector<CmdArgs> &cmds) {
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

    KeyTypeVec keys;
    FieldMapVec field_maps;
    for (int i = 0; i < 20; ++i) {
        keys.push_back(i);
        field_maps.push_back({{"f", "v"}});
    }
    backend_->Put(keys, field_maps);

    {
        auto sizes = backend_->GetAsyncQueueSizes();
        ASSERT_EQ(2, sizes.size());
        EXPECT_GT(sizes[0] + sizes[1], 0);
    }

    ASSERT_EQ(EC_OK, backend_->Close());
}

} // namespace kv_cache_manager
