#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/common/test/mock_redis_client.h"
#include "kv_cache_manager/common/test/redis_test_base.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_async_redis_backend.h"
#include "kv_cache_manager/meta/test/meta_storage_backend_test_base.h"
#include "kv_cache_manager/meta/utils.h"

namespace kv_cache_manager {

class MockMetaAsyncRedisBackend : public MetaAsyncRedisBackend {
public:
    MOCK_METHOD(std::shared_ptr<RedisClient>, CreateRedisClient, (), (const));
};

class TrackingCacheLocation : public CacheLocation {
public:
    TrackingCacheLocation(std::atomic<int> &serialize_count, std::string serialized_value)
        : serialize_count_(serialize_count), serialized_value_(std::move(serialized_value)) {}

    std::string ToJsonString() const noexcept override {
        serialize_count_.fetch_add(1, std::memory_order_relaxed);
        return serialized_value_;
    }

private:
    std::atomic<int> &serialize_count_;
    std::string serialized_value_;
};

class ScriptedReclaimMetaAsyncRedisBackend : public MetaAsyncRedisBackend {
public:
    ErrorCode RandomSample(RequestContext *, const int64_t, KeyTypeVec &out_keys) noexcept override {
        out_keys = sampled_keys;
        return sample_result;
    }

    std::vector<ErrorCode> GetProperties(RequestContext *,
                                         const KeyTypeVec &,
                                         const std::vector<std::string> &field_names,
                                         PropertyMapVector &out_properties) noexcept override {
        requested_fields = field_names;
        out_properties = properties;
        return property_results;
    }

    ErrorCode sample_result = EC_OK;
    KeyTypeVec sampled_keys;
    std::vector<ErrorCode> property_results;
    PropertyMapVector properties;
    std::vector<std::string> requested_fields;
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
                               "&async_max_size=1000&async_drain_ms=2000");
    }

    void SetupMockRedisClients() {
        EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([this]() {
            StandardUri empty_uri;
            auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
            ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
            ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
            ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([](const std::vector<CmdArgs> &cmds) {
                // Inject minimal delay to ensure batch_flush_time_us > 0 in stats
                std::this_thread::sleep_for(std::chrono::microseconds(10));
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

    void InitMemoryPrimaryWithoutConsumers(int64_t capacity = 2) {
        config_->SetMemoryPrimary(true);
        SetupMockRedisClients();
        ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
        backend_->queue_max_size_ = capacity;
        for (int i = 0; i < backend_->queue_count_; ++i) {
            backend_->queues_.push_back(std::make_unique<MpscWriteQueue>());
            backend_->consumer_clients_.push_back(backend_->CreateRedisClient());
        }
        backend_->is_running_.store(true);
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

TEST_F(MetaAsyncRedisBackendTest, TestSampleReclaimCandidatesUsesReadOnlyTimestampQuery) {
    ScriptedReclaimMetaAsyncRedisBackend backend;
    backend.sampled_keys = {1, 2};
    backend.property_results = {EC_OK, EC_OK};
    backend.properties = {{{PROPERTY_LRU_TIME, "101"}}, {{PROPERTY_LRU_TIME, "malformed"}}};

    ReclaimCandidateVector candidates;
    ASSERT_EQ(EC_OK, backend.SampleReclaimCandidates(nullptr, 2, candidates));
    ASSERT_EQ((std::vector<std::string>{PROPERTY_LRU_TIME}), backend.requested_fields);
    ASSERT_EQ(2, candidates.size());
    EXPECT_EQ(1, candidates[0].key);
    EXPECT_EQ(101, candidates[0].last_access_time_us);
    EXPECT_EQ(2, candidates[1].key);
    EXPECT_EQ(0, candidates[1].last_access_time_us);
}

TEST_F(MetaAsyncRedisBackendTest, TestSampleReclaimCandidatesPropertyFailureDegradesAllTimestamps) {
    ScriptedReclaimMetaAsyncRedisBackend backend;
    backend.sampled_keys = {1, 2};
    backend.property_results = {EC_ERROR, EC_OK};
    backend.properties = {{{PROPERTY_LRU_TIME, "101"}}, {{PROPERTY_LRU_TIME, "202"}}};

    ReclaimCandidateVector candidates;
    ASSERT_EQ(EC_OK, backend.SampleReclaimCandidates(nullptr, 2, candidates));
    ASSERT_EQ(2, candidates.size());
    EXPECT_EQ(0, candidates[0].last_access_time_us);
    EXPECT_EQ(0, candidates[1].last_access_time_us);
}

TEST_F(MetaAsyncRedisBackendTest, TestGroupLruStrictCandidateReadsPreserveLegacyFallback) {
    AssertReclaimCandidateReadModes<MetaAsyncRedisBackend>();
}

TEST_F(MetaAsyncRedisBackendTest, TestInitInvalidAsyncConfig) {
    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=0&async_max_batch=64");
    ASSERT_EQ(EC_BADARGS, backend_->Init("test_instance", config_));

    ConstructBackend();
    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=2&async_max_batch=0");
    ASSERT_EQ(EC_BADARGS, backend_->Init("test_instance", config_));
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

    std::atomic<int> location_serialize_count{0};
    std::atomic<int> collision_serialize_count{0};
    WriteOp op;
    op.type = WriteOpType::kPut;
    op.keys = {1, 2};
    op.field_maps = {{{"property", "value"}}, {{"L#collision", "property-wins"}}};
    op.locations = {
        {{"location", std::make_shared<TrackingCacheLocation>(location_serialize_count, "location-json")}},
        {{"collision", std::make_shared<TrackingCacheLocation>(collision_serialize_count, "must-not-serialize")}}};

    std::vector<CmdArgs> cmds;
    backend_->CompileWriteOp(op, cmds);

    EXPECT_THAT(
        cmds,
        ElementsAre(
            ElementsAre("DEL", "kvcache:instance_test_instance:cache_1"),
            ElementsAre(
                "HSET", "kvcache:instance_test_instance:cache_1", "L#location", "location-json", "property", "value"),
            ElementsAre("DEL", "kvcache:instance_test_instance:cache_2"),
            ElementsAre("HSET", "kvcache:instance_test_instance:cache_2", "L#collision", "property-wins")));
    EXPECT_EQ(1, location_serialize_count.load(std::memory_order_relaxed));
    EXPECT_EQ(0, collision_serialize_count.load(std::memory_order_relaxed));
    EXPECT_TRUE(op.field_maps.empty());
    EXPECT_TRUE(op.locations.empty());
}

TEST_F(MetaAsyncRedisBackendTest, TestCompileWriteOpUpsert) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));

    std::atomic<int> serialize_count{0};
    WriteOp op;
    op.type = WriteOpType::kUpsert;
    op.keys = {20, 30};
    op.field_maps = {{{"property", "value"}}, {}};
    op.locations = {{{"location", std::make_shared<TrackingCacheLocation>(serialize_count, "upsert-location-json")}},
                    {{"null", nullptr}}};

    std::vector<CmdArgs> cmds;
    backend_->CompileWriteOp(op, cmds);

    EXPECT_THAT(cmds,
                ElementsAre(ElementsAre("HSET",
                                        "kvcache:instance_test_instance:cache_20",
                                        "L#location",
                                        "upsert-location-json",
                                        "property",
                                        "value"),
                            ElementsAre("HSET", "kvcache:instance_test_instance:cache_30", "L#null", "")));
    EXPECT_EQ(1, serialize_count.load(std::memory_order_relaxed));
    EXPECT_TRUE(op.field_maps.empty());
    EXPECT_TRUE(op.locations.empty());
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
    op.locations = {{}};

    std::vector<CmdArgs> cmds;
    backend_->CompileWriteOp(op, cmds);

    ASSERT_EQ(1, cmds.size());
    ASSERT_EQ("DEL", cmds[0][0]);
}

TEST_F(MetaAsyncRedisBackendTest, TestPutAndUpsertSerializeAfterDequeueWithOwnedSnapshot) {
    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=1&async_max_batch=64"
                           "&async_wait_us=1000&async_max_size=1000&async_sync_timeout_ms=5000");

    std::atomic<int> pipeline_call_count{0};
    std::mutex pipeline_mutex;
    std::condition_variable pipeline_cv;
    bool pipeline_entered = false;
    bool can_proceed = false;
    std::mutex flushed_cmds_mutex;
    std::vector<CmdArgs> flushed_cmds;
    EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([&]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
        ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
        ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
        ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([&](const std::vector<CmdArgs> &cmds) {
            if (pipeline_call_count.fetch_add(1, std::memory_order_relaxed) == 0) {
                std::unique_lock<std::mutex> lock(pipeline_mutex);
                pipeline_entered = true;
                pipeline_cv.notify_all();
                pipeline_cv.wait(lock, [&]() { return can_proceed; });
            } else {
                std::lock_guard<std::mutex> lock(flushed_cmds_mutex);
                flushed_cmds.insert(flushed_cmds.end(), cmds.begin(), cmds.end());
            }

            std::vector<ReplyUPtr> replies;
            for (size_t i = 0; i < cmds.size(); ++i) {
                redisReply *reply = (redisReply *)malloc(sizeof(redisReply));
                memset(reply, 0, sizeof(redisReply));
                reply->type = REDIS_REPLY_INTEGER;
                reply->integer = 1;
                replies.emplace_back(reply, freeReplyObject);
            }
            return replies;
        }));
        return mock;
    }));

    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
    ASSERT_EQ(EC_OK, backend_->Open());

    CacheLocationMapVector seed_locations(1);
    PropertyMapVector seed_properties = {{{"seed", "value"}}};
    EXPECT_THAT(backend_->Put(nullptr, {0}, seed_locations, seed_properties), ElementsAre(EC_OK));
    bool consumer_blocked = false;
    {
        std::unique_lock<std::mutex> lock(pipeline_mutex);
        consumer_blocked = pipeline_cv.wait_for(lock, std::chrono::seconds(5), [&]() { return pipeline_entered; });
    }
    if (!consumer_blocked) {
        {
            std::lock_guard<std::mutex> lock(pipeline_mutex);
            can_proceed = true;
        }
        pipeline_cv.notify_all();
        EXPECT_EQ(EC_OK, backend_->Close());
        FAIL() << "consumer did not enter the seed pipeline";
    }

    std::atomic<int> put_serialize_count{0};
    {
        CacheLocationMapVector locations = {
            {{"put", std::make_shared<TrackingCacheLocation>(put_serialize_count, "put-location-json")}}};
        PropertyMapVector properties = {{{"put_property", "put-value"}}};
        EXPECT_THAT(backend_->Put(nullptr, {101}, locations, properties), ElementsAre(EC_OK));
        locations[0].clear();
        properties[0]["put_property"] = "mutated-put-value";
    }

    std::atomic<int> upsert_serialize_count{0};
    {
        CacheLocationMapVector locations = {
            {{"upsert", std::make_shared<TrackingCacheLocation>(upsert_serialize_count, "upsert-location-json")}}};
        PropertyMapVector properties = {{{"upsert_property", "upsert-value"}}};
        EXPECT_THAT(backend_->Upsert(nullptr, {102}, locations, properties), ElementsAre(EC_OK));
        locations[0].clear();
        properties[0]["upsert_property"] = "mutated-upsert-value";
    }

    EXPECT_EQ(0, put_serialize_count.load(std::memory_order_relaxed));
    EXPECT_EQ(0, upsert_serialize_count.load(std::memory_order_relaxed));

    {
        std::lock_guard<std::mutex> lock(pipeline_mutex);
        can_proceed = true;
    }
    pipeline_cv.notify_all();
    EXPECT_TRUE(backend_->Sync({101, 102}));
    EXPECT_EQ(1, put_serialize_count.load(std::memory_order_relaxed));
    EXPECT_EQ(1, upsert_serialize_count.load(std::memory_order_relaxed));

    std::vector<CmdArgs> actual_cmds;
    {
        std::lock_guard<std::mutex> lock(flushed_cmds_mutex);
        actual_cmds = flushed_cmds;
    }
    EXPECT_THAT(actual_cmds,
                ElementsAre(ElementsAre("DEL", "kvcache:instance_test_instance:cache_101"),
                            ElementsAre("HSET",
                                        "kvcache:instance_test_instance:cache_101",
                                        "L#put",
                                        "put-location-json",
                                        "put_property",
                                        "put-value"),
                            ElementsAre("HSET",
                                        "kvcache:instance_test_instance:cache_102",
                                        "L#upsert",
                                        "upsert-location-json",
                                        "upsert_property",
                                        "upsert-value")));
    EXPECT_EQ(EC_OK, backend_->Close());
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

    const std::string &prefix = backend_->cache_key_prefix_;
    const std::string &instance_id = backend_->instance_id_;

    KeyTypeVec keys = {100, 200, 300};
    auto full_keys = AppendPrefixToKeys(prefix, keys);
    ASSERT_EQ(3, full_keys.size());
    for (const auto &fk : full_keys) {
        ASSERT_TRUE(fk.find("kvcache:instance_test_instance:cache_") == 0);
    }

    KeyTypeVec out_keys;
    ASSERT_TRUE(StripPrefixInKeys(prefix, instance_id, full_keys, out_keys));
    ASSERT_EQ(keys, out_keys);
}

TEST_F(MetaAsyncRedisBackendTest, TestStripPrefixInvalid) {
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));

    const std::string &prefix = backend_->cache_key_prefix_;
    const std::string &instance_id = backend_->instance_id_;

    KeyTypeVec out_keys;
    ASSERT_FALSE(StripPrefixInKeys(prefix, instance_id, {"invalid_key"}, out_keys));
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
                           "&async_enqueue_timeout_ms=100");

    std::atomic<bool> pipeline_entered{false};
    std::atomic<bool> can_proceed{false};
    EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([&]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
        ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
        ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
        ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([&](const std::vector<CmdArgs> &cmds) {
            pipeline_entered.store(true, std::memory_order_release);
            while (!can_proceed.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
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

    // Seed: consumer picks this up and blocks in pipeline
    CacheLocationMapVector seed_locs(1);
    PropertyMapVector seed_props = {{{"f", "v"}}};
    backend_->Put(nullptr, {0}, seed_locs, seed_props);

    while (!pipeline_entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    // Consumer is blocked — queue will fill up, causing enqueue timeouts
    int timeout_count = 0;
    for (int i = 1; i <= 50; ++i) {
        CacheLocationMapVector locs(1);
        PropertyMapVector props = {{{"f", "v"}}};
        auto results = backend_->Put(nullptr, {(KeyType)i}, locs, props);
        if (!results.empty() && results[0] == EC_TIMEOUT) {
            ++timeout_count;
        }
    }
    ASSERT_EQ(timeout_count, 45);

    can_proceed.store(true, std::memory_order_release);
    ASSERT_EQ(EC_OK, backend_->Close());
}

// ==================== MetaData Passthrough Tests ====================

TEST_F(MetaAsyncRedisBackendTest, TestPutMetaData) {
    ASSERT_EQ(EC_OK, InitAndOpen());

    FieldMap meta = {{"version", "1"}, {"created_at", "2024-01-01"}};
    ASSERT_EQ(EC_OK, backend_->PutMetaData(meta));

    FieldMap out_meta;
    backend_->GetMetaData(out_meta);
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

    // Use 1 queue with a blocking mock: consumer blocks in pipeline so subsequent keys stay in queue
    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=1&async_max_batch=64"
                           "&async_wait_us=500000&async_max_size=10000");
    std::atomic<bool> pipeline_entered{false};
    std::atomic<bool> can_proceed{false};
    EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([&]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
        ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
        ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
        ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([&](const std::vector<CmdArgs> &cmds) {
            pipeline_entered.store(true, std::memory_order_release);
            while (!can_proceed.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
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

    // Seed put: consumer picks it up and blocks in pipeline
    CacheLocationMapVector seed_locs(1);
    PropertyMapVector seed_props = {{{"f", "v"}}};
    backend_->Put(nullptr, {0}, seed_locs, seed_props);

    // Deterministic handshake: wait until consumer enters pipeline
    while (!pipeline_entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    // Consumer is blocked in pipeline — new keys stay in queue
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
        EXPECT_GT(stats.max_async_queue_size, 0);
        EXPECT_GT(stats.avg_async_queue_size, 0);
    }

    can_proceed.store(true, std::memory_order_release);
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

    KeyTypeVec keys = {1, 2, 3};
    CacheLocationMapVector locations(3);
    PropertyMapVector properties = {{{"f", "v"}}, {{"f", "v"}}, {{"f", "v"}}};
    auto results = backend_->Put(nullptr, keys, locations, properties);
    for (auto ec : results) {
        EXPECT_EQ(EC_OK, ec);
    }

    ASSERT_TRUE(backend_->Sync(keys));

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

    // Test enqueue timeout: tiny queue capacity with blocked consumer
    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=1&async_max_batch=1"
                           "&async_wait_us=500000&async_max_size=1&async_enqueue_timeout_ms=1");
    ConstructBackend();
    std::atomic<bool> timeout_pipeline_entered{false};
    std::atomic<bool> timeout_can_proceed{false};
    EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([&]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
        ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
        ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
        ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([&](const std::vector<CmdArgs> &cmds) {
            timeout_pipeline_entered.store(true, std::memory_order_release);
            while (!timeout_can_proceed.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
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

    // Seed: consumer picks this up and blocks in pipeline
    CacheLocationMapVector locs1(1);
    PropertyMapVector props1 = {{{"f", "v"}}};
    backend_->Put(nullptr, {100}, locs1, props1);

    while (!timeout_pipeline_entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    // Consumer is blocked — queue will fill up (max_size=1), enqueue will timeout
    // Must put keys one-by-one: WaitForCapacity is checked once per EnqueueWriteOp per queue,
    // so a batch put would pass the check (key_size_=0 < 1) and push all keys at once.
    int timeout_count = 0;
    for (int i = 200; i < 210; ++i) {
        CacheLocationMapVector locs(1);
        PropertyMapVector props = {{{"f", "v"}}};
        auto results = backend_->Put(nullptr, {(KeyType)i}, locs, props);
        if (!results.empty() && results[0] == EC_TIMEOUT) {
            ++timeout_count;
        }
    }
    ASSERT_EQ(timeout_count, 9);

    timeout_can_proceed.store(true, std::memory_order_release);
    ASSERT_EQ(EC_OK, backend_->Close());
}

TEST_F(MetaAsyncRedisBackendTest, TestGetAsyncWriteStatsPipelineError) {
    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=1&async_max_batch=64&async_wait_us=1000"
                           "&async_max_size=1000&async_sync_timeout_ms=5000");
    ConstructBackend();

    std::atomic<int> pipeline_call_count{0};
    std::atomic<bool> first_pipeline_entered{false};
    std::atomic<bool> can_proceed{false};

    EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([&]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
        ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
        ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
        ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([&](const std::vector<CmdArgs> &) {
            int n = pipeline_call_count.fetch_add(1, std::memory_order_acq_rel);
            if (n == 0) {
                first_pipeline_entered.store(true, std::memory_order_release);
                while (!can_proceed.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
            }
            return std::vector<RedisClient::ReplyUPtr>{};
        }));
        return mock;
    }));
    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
    ASSERT_EQ(EC_OK, backend_->Open());

    // Seed: consumer picks this up and blocks in pipeline
    CacheLocationMapVector seed_locs(1);
    PropertyMapVector seed_props = {{{"f", "v"}}};
    backend_->Put(nullptr, {0}, seed_locs, seed_props);

    while (!first_pipeline_entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    // Consumer is blocked. Push WriteOp + barrier deterministically.
    KeyTypeVec keys = {1, 2, 3};
    CacheLocationMapVector locations(3);
    PropertyMapVector properties = {{{"f", "v"}}, {{"f", "v"}}, {{"f", "v"}}};
    backend_->Put(nullptr, keys, locations, properties);

    auto barrier = std::make_shared<BarrierContext>();
    barrier->remain.store(1, std::memory_order_release);
    backend_->queues_[0]->PushBarrier(SyncBarrierItem{barrier});

    can_proceed.store(true, std::memory_order_release);

    ASSERT_FALSE(barrier->Wait(std::chrono::milliseconds{5000}));

    ASSERT_EQ(EC_OK, backend_->Close());

    auto stats = backend_->GetAsyncWriteStats();
    EXPECT_GT(stats.pipeline_error_count, 0);
    EXPECT_EQ(0, stats.flush_key_count);
    EXPECT_GT(stats.batch_flush_time_us, 0);
}

TEST_F(MetaAsyncRedisBackendTest, TestSyncPartialPipelineFailure) {
    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=1&async_max_batch=64&async_wait_us=1000"
                           "&async_max_size=1000&async_sync_timeout_ms=5000");
    ConstructBackend();

    std::atomic<int> pipeline_call_count{0};
    std::atomic<bool> first_pipeline_entered{false};
    std::atomic<bool> can_proceed{false};

    EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([&]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
        ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
        ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
        ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([&](const std::vector<CmdArgs> &cmds) {
            int n = pipeline_call_count.fetch_add(1, std::memory_order_acq_rel);
            if (n == 0) {
                first_pipeline_entered.store(true, std::memory_order_release);
                while (!can_proceed.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                std::vector<RedisClient::ReplyUPtr> replies;
                for (size_t i = 0; i < cmds.size(); ++i) {
                    redisReply *r = (redisReply *)malloc(sizeof(redisReply));
                    memset(r, 0, sizeof(redisReply));
                    r->type = REDIS_REPLY_INTEGER;
                    r->integer = 1;
                    replies.emplace_back(r, freeReplyObject);
                }
                return replies;
            }
            std::vector<RedisClient::ReplyUPtr> replies;
            for (size_t i = 0; i < cmds.size(); ++i) {
                redisReply *r = (redisReply *)malloc(sizeof(redisReply));
                memset(r, 0, sizeof(redisReply));
                if (i == cmds.size() - 1) {
                    r->type = REDIS_REPLY_ERROR;
                    r->str = strdup("ERR partial failure");
                    r->len = static_cast<int>(strlen(r->str));
                } else {
                    r->type = REDIS_REPLY_INTEGER;
                    r->integer = 1;
                }
                replies.emplace_back(r, freeReplyObject);
            }
            return replies;
        }));
        return mock;
    }));

    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
    ASSERT_EQ(EC_OK, backend_->Open());

    // Seed: consumer picks this up and blocks in pipeline
    CacheLocationMapVector seed_locs(1);
    PropertyMapVector seed_props = {{{"f", "v"}}};
    backend_->Put(nullptr, {0}, seed_locs, seed_props);

    while (!first_pipeline_entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    // Consumer is blocked. Push WriteOp + barrier deterministically.
    KeyTypeVec keys = {1, 2, 3};
    CacheLocationMapVector locations(3);
    PropertyMapVector properties = {{{"f1", "v1"}}, {{"f2", "v2"}}, {{"f3", "v3"}}};
    backend_->Put(nullptr, keys, locations, properties);

    auto barrier = std::make_shared<BarrierContext>();
    barrier->remain.store(1, std::memory_order_release);
    backend_->queues_[0]->PushBarrier(SyncBarrierItem{barrier});

    can_proceed.store(true, std::memory_order_release);

    // Barrier segment covers all 6 commands; last command fails → segment_ok = false
    ASSERT_FALSE(barrier->Wait(std::chrono::milliseconds{5000}));

    ASSERT_EQ(EC_OK, backend_->Close());

    auto stats = backend_->GetAsyncWriteStats();
    EXPECT_EQ(1, stats.pipeline_error_count);
    // Seed batch flushed 1 key; partial failure batch flushed 0 (segment containing error)
    EXPECT_EQ(1, stats.flush_key_count);
}

TEST_F(MetaAsyncRedisBackendTest, TestBatchFlushMultiSegmentPartialFailure) {
    config_->SetStorageUri("redis://user:pass@host:6379/?async_queue_count=1&async_max_batch=1024&async_wait_us=500000"
                           "&async_max_size=10000&async_sync_timeout_ms=5000");
    ConstructBackend();

    std::atomic<int> pipeline_call_count{0};
    std::atomic<bool> first_pipeline_entered{false};
    std::atomic<bool> can_proceed{false};

    EXPECT_CALL(*backend_, CreateRedisClient()).WillRepeatedly(Invoke([&]() {
        StandardUri empty_uri;
        auto mock = std::make_shared<::testing::NiceMock<MockRedisClient>>(empty_uri);
        ON_CALL(*mock, IsContextOk()).WillByDefault(Return(true));
        ON_CALL(*mock, Reconnect()).WillByDefault(Return(true));
        ON_CALL(*mock, TryExecPipeline(_)).WillByDefault(Invoke([&](const std::vector<CmdArgs> &cmds) {
            int call_num = pipeline_call_count.fetch_add(1, std::memory_order_acq_rel);

            if (call_num == 0) {
                first_pipeline_entered.store(true, std::memory_order_release);
                while (!can_proceed.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                std::vector<RedisClient::ReplyUPtr> replies;
                for (size_t i = 0; i < cmds.size(); ++i) {
                    redisReply *r = (redisReply *)malloc(sizeof(redisReply));
                    memset(r, 0, sizeof(redisReply));
                    r->type = REDIS_REPLY_INTEGER;
                    r->integer = 1;
                    replies.emplace_back(r, freeReplyObject);
                }
                return replies;
            }

            // Second call: fail last 2 commands (second segment's DEL + HSET)
            std::vector<RedisClient::ReplyUPtr> replies;
            for (size_t i = 0; i < cmds.size(); ++i) {
                redisReply *r = (redisReply *)malloc(sizeof(redisReply));
                memset(r, 0, sizeof(redisReply));
                if (i >= cmds.size() - 2) {
                    r->type = REDIS_REPLY_ERROR;
                    r->str = strdup("ERR partial");
                    r->len = static_cast<int>(strlen(r->str));
                } else {
                    r->type = REDIS_REPLY_INTEGER;
                    r->integer = 1;
                }
                replies.emplace_back(r, freeReplyObject);
            }
            return replies;
        }));
        return mock;
    }));

    ASSERT_EQ(EC_OK, backend_->Init("test_instance", config_));
    ASSERT_EQ(EC_OK, backend_->Open());

    // Seed: consumer picks this up and blocks in first pipeline call
    CacheLocationMapVector seed_locs(1);
    PropertyMapVector seed_props = {{{"f", "v"}}};
    backend_->Put(nullptr, {0}, seed_locs, seed_props);

    while (!first_pipeline_entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    // Consumer is blocked. Push items for a multi-segment batch:
    //   WriteOp({1,2}) → 4 cmds | Barrier1 | WriteOp({3}) → 2 cmds | Barrier2
    // Mock will fail last 2 cmds → segment 1 ok, segment 2 fail

    CacheLocationMapVector locs1(2);
    PropertyMapVector props1 = {{{"f1", "v1"}}, {{"f2", "v2"}}};
    backend_->Put(nullptr, {1, 2}, locs1, props1);

    auto barrier1 = std::make_shared<BarrierContext>();
    barrier1->remain.store(1, std::memory_order_release);
    backend_->queues_[0]->PushBarrier(SyncBarrierItem{barrier1});

    CacheLocationMapVector locs2(1);
    PropertyMapVector props2 = {{{"f3", "v3"}}};
    backend_->Put(nullptr, {3}, locs2, props2);

    auto barrier2 = std::make_shared<BarrierContext>();
    barrier2->remain.store(1, std::memory_order_release);
    backend_->queues_[0]->PushBarrier(SyncBarrierItem{barrier2});

    // Release consumer — first batch (seed) succeeds, second batch has partial failure
    can_proceed.store(true, std::memory_order_release);

    ASSERT_TRUE(barrier1->Wait(std::chrono::milliseconds{5000}));
    ASSERT_FALSE(barrier2->Wait(std::chrono::milliseconds{5000}));

    // Close joins consumer thread, ensuring all stats are flushed
    ASSERT_EQ(EC_OK, backend_->Close());

    auto stats = backend_->GetAsyncWriteStats();
    EXPECT_EQ(1, stats.pipeline_error_count);
    // seed batch: 1 key (all_ok path) + segment 1: 2 keys (partial ok) = 3
    EXPECT_EQ(3, stats.flush_key_count);
}

TEST_F(MetaAsyncRedisBackendTest, TestMemoryPrimaryFullQueueDoesNotWait) {
    InitMemoryPrimaryWithoutConsumers(1);
    backend_->enqueue_timeout_ms_ = 5000;
    const KeyType key = 7;
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, backend_->Delete(nullptr, {key}, {EC_OK}));
    auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(std::vector<ErrorCode>{EC_TIMEOUT}, backend_->Put(nullptr, {key}, {{}}, {{{"p", "v"}}}, {EC_OK}));
    EXPECT_EQ(std::vector<ErrorCode>{EC_TIMEOUT}, backend_->Upsert(nullptr, {key}, {{}}, {{{"p", "v"}}}, {EC_OK}));
    EXPECT_EQ(std::vector<ErrorCode>{EC_TIMEOUT}, backend_->DeleteLocations(nullptr, {key}, {{"loc"}}, {EC_OK}));
    EXPECT_EQ(std::vector<ErrorCode>{EC_TIMEOUT}, backend_->Delete(nullptr, {key}, {EC_OK}));
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(500));
    EXPECT_FALSE(backend_->Sync({key}));
    auto stats = backend_->GetAsyncWriteStats();
    EXPECT_EQ(4, stats.dropped_key_count);
}

TEST_F(MetaAsyncRedisBackendTest, TestForceUpsertBypassesQueueCapacity) {
    InitMemoryPrimaryWithoutConsumers(1);
    const KeyType key = 7;
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, backend_->Delete(nullptr, {key}, {EC_OK}));
    EXPECT_EQ(std::vector<ErrorCode>{EC_TIMEOUT}, backend_->Upsert(nullptr, {key}, {{}}, {{{"p", "v"}}}, {EC_OK}));
    EXPECT_EQ(std::vector<ErrorCode>{EC_OK}, backend_->ForceUpsert(nullptr, {key}, {{}}, {{{"p", "v"}}}));

    auto &queue = backend_->queues_[backend_->GetQueueIndexForKey(key)];
    EXPECT_EQ(2, queue->GetKeySize());
    int64_t taken = 0;
    auto items = queue->PopBatch(10, taken);
    ASSERT_EQ(2, items.size());
    EXPECT_EQ(2, taken);
    EXPECT_EQ(WriteOpType::kDelete, std::get<WriteOp>(items[0]).type);
    EXPECT_EQ(WriteOpType::kUpsert, std::get<WriteOp>(items[1]).type);
    EXPECT_EQ(0, queue->GetKeySize());
}

TEST_F(MetaAsyncRedisBackendTest, TestMemoryPrimaryPrimaryWriteKeepsOriginalBackpressure) {
    InitMemoryPrimaryWithoutConsumers(1);
    backend_->enqueue_timeout_ms_ = 0;
    const KeyType key = 7;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), backend_->Delete(nullptr, {key, key}));
    EXPECT_EQ(2, backend_->queues_[backend_->GetQueueIndexForKey(key)]->GetKeySize());
    EXPECT_EQ(std::vector<ErrorCode>{EC_TIMEOUT}, backend_->Delete(nullptr, {key}));
    const auto stats = backend_->GetAsyncWriteStats();
    EXPECT_EQ(0, stats.dropped_key_count);
}

TEST_F(MetaAsyncRedisBackendTest, TestConditionalSecondaryWritesFilterDuringQueueGrouping) {
    InitMemoryPrimaryWithoutConsumers(10);
    const KeyTypeVec keys{1, 2, 3};
    CacheLocationMapVector locations(3);
    PropertyMapVector properties{{{"p", "one"}}, {{"p", "two"}}, {{"p", "three"}}};
    const std::vector<ErrorCode> put_gate{EC_OK, EC_NOSPC, EC_OK};
    EXPECT_EQ(put_gate, backend_->Put(nullptr, keys, locations, properties, put_gate));

    std::vector<std::pair<KeyType, std::string>> queued;
    for (const auto &queue : backend_->queues_) {
        int64_t taken = 0;
        auto items = queue->PopBatch(10, taken);
        for (auto &item : items) {
            auto &op = std::get<WriteOp>(item);
            ASSERT_EQ(op.keys.size(), op.field_maps.size());
            for (size_t i = 0; i < op.keys.size(); ++i) {
                queued.emplace_back(op.keys[i], op.field_maps[i].at("p"));
            }
        }
    }
    std::sort(queued.begin(), queued.end());
    EXPECT_EQ((std::vector<std::pair<KeyType, std::string>>{{1, "one"}, {3, "three"}}), queued);

    const std::vector<ErrorCode> delete_gate{EC_NOENT, EC_ERROR, EC_NOSPC};
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK, EC_ERROR, EC_NOSPC}), backend_->Delete(nullptr, keys, delete_gate));
    KeyTypeVec delete_keys;
    for (const auto &queue : backend_->queues_) {
        int64_t taken = 0;
        auto items = queue->PopBatch(10, taken);
        for (auto &item : items) {
            auto &op = std::get<WriteOp>(item);
            delete_keys.insert(delete_keys.end(), op.keys.begin(), op.keys.end());
        }
    }
    EXPECT_EQ(KeyTypeVec{1}, delete_keys);

    const LocationIdsPerKey location_ids{{"a"}, {"b", "c"}, {"d"}};
    const std::vector<ErrorCode> delete_locations_gate{EC_NOSPC, EC_NOENT, EC_OK};
    EXPECT_EQ((std::vector<ErrorCode>{EC_NOSPC, EC_OK, EC_OK}),
              backend_->DeleteLocations(nullptr, keys, location_ids, delete_locations_gate));
    std::vector<std::pair<KeyType, std::vector<std::string>>> queued_location_deletes;
    for (const auto &queue : backend_->queues_) {
        int64_t taken = 0;
        auto items = queue->PopBatch(10, taken);
        for (auto &item : items) {
            auto &op = std::get<WriteOp>(item);
            ASSERT_EQ(op.keys.size(), op.field_names_vec.size());
            for (size_t i = 0; i < op.keys.size(); ++i) {
                queued_location_deletes.emplace_back(op.keys[i], std::move(op.field_names_vec[i]));
            }
        }
    }
    std::sort(queued_location_deletes.begin(), queued_location_deletes.end());
    EXPECT_EQ((std::vector<std::pair<KeyType, std::vector<std::string>>>{
                  {2, {PROPERTY_LOCATION_PREFIX + "b", PROPERTY_LOCATION_PREFIX + "c"}},
                  {3, {PROPERTY_LOCATION_PREFIX + "d"}},
              }),
              queued_location_deletes);

    const std::vector<ErrorCode> rejected(keys.size(), EC_NOSPC);
    EXPECT_EQ(rejected, backend_->Upsert(nullptr, keys, locations, properties, rejected));
    for (const auto &queue : backend_->queues_) {
        EXPECT_EQ(0, queue->GetKeySize());
    }
    const auto stats = backend_->GetAsyncWriteStats();
    EXPECT_EQ(0, stats.dropped_key_count);
}

TEST_F(MetaAsyncRedisBackendTest, TestMemoryPrimaryOversizeAndSyncKeepsOriginalAdmission) {
    InitMemoryPrimaryWithoutConsumers(1);
    EXPECT_EQ((std::vector<ErrorCode>{EC_TIMEOUT, EC_TIMEOUT}), backend_->Delete(nullptr, {5, 5}, {EC_OK, EC_OK}));
    EXPECT_EQ(0, backend_->queues_[backend_->GetQueueIndexForKey(5)]->GetKeySize());
    KeyType key = 6;
    while (backend_->GetQueueIndexForKey(key) == backend_->GetQueueIndexForKey(5)) {
        ++key;
    }
    EXPECT_EQ(std::vector<ErrorCode>{EC_OK}, backend_->Delete(nullptr, {key}, {EC_OK}));
    EXPECT_FALSE(backend_->Sync({key})); // no consumer to complete the barrier
    EXPECT_EQ(1, backend_->queues_[backend_->GetQueueIndexForKey(key)]->GetKeySize());
}

TEST_F(MetaAsyncRedisBackendTest, TestMemoryPrimaryMetadataUsesBoundedConsumerAndRealKey) {
    InitMemoryPrimaryWithoutConsumers(1);
    EXPECT_EQ(EC_OK, backend_->PutMetaData({{"key_count", "19"}}));
    EXPECT_EQ(EC_TIMEOUT, backend_->PutMetaData({{"key_count", "20"}}));
    EXPECT_EQ(1, backend_->queues_[0]->GetKeySize());
    int64_t taken = 0;
    auto items = backend_->queues_[0]->PopBatch(10, taken);
    ASSERT_EQ(1, items.size());
    auto &op = std::get<WriteOp>(items.front());
    EXPECT_EQ(WriteOpType::kPutMetaData, op.type);
    EXPECT_TRUE(op.keys.empty());
    std::vector<CmdArgs> commands;
    backend_->CompileWriteOp(op, commands);
    ASSERT_FALSE(commands.empty());
    for (const auto &command : commands) {
        ASSERT_GE(command.size(), 2);
        EXPECT_EQ(backend_->metadata_key_, command[1]);
    }
    EXPECT_EQ(1, backend_->GetAsyncWriteStats().dropped_metadata_count);
    EXPECT_EQ(EC_OK, backend_->PutMetaData({{"key_count", "21"}}));
    items = backend_->queues_[0]->PopBatch(10, taken);
    backend_->BatchFlush(0, items, taken);
    EXPECT_EQ(0, backend_->GetAsyncWriteStats().flush_key_count);
}

TEST_F(MetaAsyncRedisBackendTest, TestMemoryPrimaryHealthySyncAndMetadata) {
    config_->SetMemoryPrimary(true);
    ASSERT_EQ(EC_OK, InitAndOpen());
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, backend_->Put(nullptr, {7}, {{}}, {{{"p", "v"}}}));
    ASSERT_EQ(EC_OK, backend_->PutMetaData({{"key_count", "1"}}));
    EXPECT_TRUE(backend_->Sync({7}));
    ASSERT_EQ(EC_OK, backend_->Close());
    auto stats = backend_->GetAsyncWriteStats();
    EXPECT_EQ(1, stats.flush_key_count);
}

TEST_F(MetaAsyncRedisBackendTest, TestMemoryPrimaryPreservesFifoAndConsumerSerialization) {
    InitMemoryPrimaryWithoutConsumers(10);
    std::atomic<int> serialized{0};
    auto location = std::make_shared<TrackingCacheLocation>(serialized, "location_json");
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, backend_->Put(nullptr, {42}, {{{"loc", location}}}, {{}}));
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, backend_->Upsert(nullptr, {42}, {{{"loc", location}}}, {{}}));
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, backend_->DeleteLocations(nullptr, {42}, {{"loc"}}));
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, backend_->Delete(nullptr, {42}));
    location.reset();
    EXPECT_EQ(0, serialized.load());
    int64_t taken = 0;
    auto items = backend_->queues_[backend_->GetQueueIndexForKey(42)]->PopBatch(10, taken);
    ASSERT_EQ(4, items.size());
    const std::vector<WriteOpType> expected = {
        WriteOpType::kPut, WriteOpType::kUpsert, WriteOpType::kDeleteLocations, WriteOpType::kDelete};
    std::vector<CmdArgs> commands;
    for (size_t i = 0; i < items.size(); ++i) {
        auto &op = std::get<WriteOp>(items[i]);
        EXPECT_EQ(expected[i], op.type);
        backend_->CompileWriteOp(op, commands);
    }
    EXPECT_EQ(2, serialized.load());
    EXPECT_EQ(4, taken);
}

TEST_F(MetaAsyncRedisBackendTest, TestMemoryPrimaryDrainDropsAreVisibleAndFailBarriers) {
    InitMemoryPrimaryWithoutConsumers(10);
    backend_->drain_timeout_ms_ = 1;
    backend_->max_batch_size_ = 1;
    KeyType key = 1;
    while (backend_->GetQueueIndexForKey(key) != 0) {
        ++key;
    }
    auto *client = static_cast<MockRedisClient *>(backend_->consumer_clients_[0].get());
    EXPECT_CALL(*client, TryExecPipeline(_)).WillOnce(Invoke([](const std::vector<CmdArgs> &) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::vector<ReplyUPtr> replies;
        replies.push_back(MakeFakeReplyInteger(1));
        return replies;
    }));
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, backend_->Delete(nullptr, {key}));
    ASSERT_EQ(EC_OK, backend_->PutMetaData({{"key_count", "1"}}));
    auto barrier = std::make_shared<BarrierContext>();
    barrier->remain.store(1);
    backend_->queues_[0]->PushBarrier(SyncBarrierItem{barrier});
    backend_->DrainQueue(0);
    EXPECT_FALSE(barrier->Wait(std::chrono::milliseconds(10)));
    auto stats = backend_->GetAsyncWriteStats();
    EXPECT_EQ(1, stats.flush_key_count);
    EXPECT_EQ(1, stats.dropped_metadata_count);
}

} // namespace kv_cache_manager
