#include "kv_cache_manager/common/test/mock_redis_client.h"
#include "kv_cache_manager/common/test/redis_test_base.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_cached_backend.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_redis_backend.h"
#include "kv_cache_manager/meta/test/meta_storage_backend_test_base.h"

namespace kv_cache_manager {

// A mock redis backend that overrides CreateRedisClient to return a MockRedisClient
class MockMetaRedisBackendForCached : public MetaRedisBackend {
public:
    MOCK_METHOD(std::shared_ptr<RedisClient>, CreateRedisClient, (), (const));
};

// A testable MetaCachedBackend that overrides CreatePersistentBackend
// to return a MockMetaRedisBackendForCached (already init'd by the mock factory)
class TestableMetaCachedBackend : public MetaCachedBackend {
public:
    void SetMockPersistentBackendFactory(std::function<std::unique_ptr<MockMetaRedisBackendForCached>()> factory) {
        mock_persistent_factory_ = std::move(factory);
    }

    void SetRecoverState(RecoverState state) { recover_state_.store(state, std::memory_order_release); }

protected:
    std::unique_ptr<MetaStorageBackend>
    CreatePersistentBackend(const std::string &instance_id,
                            const std::shared_ptr<MetaStorageBackendConfig> &config) const override {
        if (mock_persistent_factory_) {
            auto backend = mock_persistent_factory_();
            // Init the mock backend with the provided config (same as factory would do)
            backend->Init(instance_id, config);
            return backend;
        }
        return MetaCachedBackend::CreatePersistentBackend(instance_id, config);
    }

private:
    std::function<std::unique_ptr<MockMetaRedisBackendForCached>()> mock_persistent_factory_;
};

class MetaCachedBackendTest : public MetaStorageBackendTestBase, public RedisTestBase, public TESTBASE {
public:
    void SetUp() override;
    void TearDown() override {}

    void ConstructBackend();
    void ConstructConfig();

    // Setup mock expectations for a MockRedisClient that stores data in an in-memory map
    // This simulates a real redis for the test
    std::shared_ptr<MockRedisClient> CreateInMemoryMockRedisClient();

private:
    std::shared_ptr<TestableMetaCachedBackend> backend_;
    std::shared_ptr<MetaStorageBackendConfig> config_;

    // Shared in-memory store to simulate redis persistence across backend recreations
    std::shared_ptr<std::map<std::string, FieldMap>> redis_store_;
    std::string instance_id_ = "test_instance_0";
    std::string cache_key_prefix_;
    std::string metadata_key_;
};

void MetaCachedBackendTest::SetUp() {
    redis_store_ = std::make_shared<std::map<std::string, FieldMap>>();
    cache_key_prefix_ = "kvcache:instance_" + instance_id_ + ":cache_";
    metadata_key_ = "kvcache:instance_" + instance_id_ + ":metadata";
    ConstructBackend();
    ConstructConfig();
}

void MetaCachedBackendTest::ConstructBackend() {
    backend_ = std::make_shared<TestableMetaCachedBackend>();
    auto store = redis_store_;
    auto prefix = cache_key_prefix_;
    auto metadata_key = metadata_key_;

    backend_->SetMockPersistentBackendFactory([store, prefix, metadata_key]() {
        auto mock_backend = std::make_unique<MockMetaRedisBackendForCached>();
        auto store_captured = store;
        auto prefix_captured = prefix;
        auto metadata_key_captured = metadata_key;

        EXPECT_CALL(*mock_backend, CreateRedisClient())
            .WillRepeatedly([store_captured, prefix_captured, metadata_key_captured]() {
                StandardUri empty_uri;
                auto mock_client = std::make_shared<MockRedisClient>(empty_uri);
                EXPECT_CALL(*mock_client, IsContextOk()).WillRepeatedly(Return(true));
                EXPECT_CALL(*mock_client, Reconnect()).WillRepeatedly(Return(true));

                // Implement TryExecPipeline using the shared in-memory store
                EXPECT_CALL(*mock_client, TryExecPipeline(testing::_))
                    .WillRepeatedly([store_captured, prefix_captured, metadata_key_captured](
                                        const std::vector<CmdArgs> &commands) -> std::vector<ReplyUPtr> {
                        std::vector<ReplyUPtr> replies;
                        for (const auto &cmd : commands) {
                            if (cmd.empty()) {
                                replies.emplace_back(MakeFakeReplyInteger(0));
                                continue;
                            }
                            const std::string &op = cmd[0];
                            if (op == "DEL" && cmd.size() >= 2) {
                                store_captured->erase(cmd[1]);
                                replies.emplace_back(MakeFakeReplyInteger(1));
                            } else if (op == "HSET" && cmd.size() >= 4) {
                                const std::string &key = cmd[1];
                                for (size_t i = 2; i + 1 < cmd.size(); i += 2) {
                                    (*store_captured)[key][cmd[i]] = cmd[i + 1];
                                }
                                replies.emplace_back(MakeFakeReplyInteger(1));
                            } else if (op == "HMGET" && cmd.size() >= 3) {
                                const std::string &key = cmd[1];
                                std::vector<std::optional<std::string>> values;
                                auto it = store_captured->find(key);
                                for (size_t i = 2; i < cmd.size(); ++i) {
                                    if (it != store_captured->end()) {
                                        auto field_it = it->second.find(cmd[i]);
                                        if (field_it != it->second.end()) {
                                            values.emplace_back(field_it->second);
                                        } else {
                                            values.emplace_back(std::nullopt);
                                        }
                                    } else {
                                        values.emplace_back(std::nullopt);
                                    }
                                }
                                replies.emplace_back(MakeFakeReplyArrayString(values));
                            } else if (op == "HGETALL" && cmd.size() >= 2) {
                                const std::string &key = cmd[1];
                                std::vector<std::optional<std::string>> kv_pairs;
                                auto it = store_captured->find(key);
                                if (it != store_captured->end()) {
                                    for (const auto &[field, value] : it->second) {
                                        kv_pairs.emplace_back(field);
                                        kv_pairs.emplace_back(value);
                                    }
                                }
                                replies.emplace_back(MakeFakeReplyArrayString(kv_pairs));
                            } else if (op == "EXISTS" && cmd.size() >= 2) {
                                const std::string &key = cmd[1];
                                int64_t exists = store_captured->count(key) > 0 ? 1 : 0;
                                replies.emplace_back(MakeFakeReplyInteger(exists));
                            } else if (op == "SCAN" && cmd.size() >= 6) {
                                // SCAN cursor MATCH pattern COUNT count
                                std::string match_pattern = cmd[3];
                                // Remove trailing '*' for prefix matching
                                std::string match_prefix = match_pattern.substr(0, match_pattern.size() - 1);
                                std::vector<std::optional<std::string>> matched_keys;
                                for (const auto &[key, _] : *store_captured) {
                                    if (key.compare(0, match_prefix.size(), match_prefix) == 0) {
                                        matched_keys.emplace_back(key);
                                    }
                                }
                                replies.emplace_back(MakeFakeReplyScan("0", matched_keys));
                            } else if (op == "RANDOMKEY") {
                                if (!store_captured->empty()) {
                                    replies.emplace_back(
                                        MakeFakeReply(REDIS_REPLY_STRING, store_captured->begin()->first));
                                } else {
                                    replies.emplace_back(MakeFakeReplyInteger(0));
                                }
                            } else {
                                replies.emplace_back(MakeFakeReplyInteger(0));
                            }
                        }
                        return replies;
                    });
                return mock_client;
            });
        return mock_backend;
    });
}

void MetaCachedBackendTest::ConstructConfig() {
    config_ = std::make_shared<MetaStorageBackendConfig>();
    config_->SetStorageType(META_CACHED_BACKEND_TYPE_STR);
    config_->SetStorageUri("redis://test_redis_user:test_redis_password@test_redis_host:0/?client_max_pool_size=1");
}

// Helper: wait until backend reaches kRunning state (recover completes).
static void WaitForRunning(TestableMetaCachedBackend *backend) {
    while (backend->GetRecoverState() != MetaCachedBackend::RecoverState::kRunning) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

static void ForceRecoverState(TestableMetaCachedBackend *backend) {
    backend->SetRecoverState(MetaCachedBackend::RecoverState::kRecover);
}

TEST_F(MetaCachedBackendTest, TestSimple) {
    ASSERT_EQ(META_CACHED_BACKEND_TYPE_STR, backend_->GetStorageType());

    // Init and Open
    ASSERT_EQ(EC_OK, backend_->Init(instance_id_, config_));
    ASSERT_EQ(EC_OK, backend_->Open());

    // Put one key
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Put({1}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}}));

    // Read back via Get
    AssertGet(backend_.get(), {1}, {"f1", "f2"}, {EC_OK}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}});

    // Read back via GetAllFields
    AssertGetAllFields(backend_.get(), {1}, {EC_OK}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}});

    // Read back via Exists
    AssertExists(backend_.get(), {1, 2}, {EC_OK, EC_OK}, {true, false});

    // Close
    ASSERT_EQ(EC_OK, backend_->Close());

    // Recreate the backend object (simulating restart), redis_store_ persists
    ConstructBackend();
    ASSERT_EQ(EC_OK, backend_->Init(instance_id_, config_));
    ASSERT_EQ(EC_OK, backend_->Open());
    WaitForRunning(backend_.get());

    // After recovery from persistent backend, the key should still be readable from cache
    AssertGet(backend_.get(), {1}, {"f1", "f2"}, {EC_OK}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}});

    AssertExists(backend_.get(), {1, 2}, {EC_OK, EC_OK}, {true, false});

    ASSERT_EQ(EC_OK, backend_->Close());
}

// ==================== Recover phase: read operations ====================

TEST_F(MetaCachedBackendTest, RecoverReadLocalHit) {
    ASSERT_EQ(EC_OK, backend_->Init(instance_id_, config_));
    ASSERT_EQ(EC_OK, backend_->Open());
    WaitForRunning(backend_.get());

    // Put key into both local and persistent (dual-write in Running state)
    backend_->Put({1}, {{{"f1", "v1"}}});

    // Force back to Recover state
    ForceRecoverState(backend_.get());
    ASSERT_EQ(MetaCachedBackend::RecoverState::kRecover, backend_->GetRecoverState());

    // Key exists in local → should return from local without hitting persistent
    AssertGet(backend_.get(), {1}, {"f1"}, {EC_OK}, {{{"f1", "v1"}}});
    AssertGetAllFields(backend_.get(), {1}, {EC_OK}, {{{"f1", "v1"}}});
    AssertExists(backend_.get(), {1}, {EC_OK}, {true});

    ASSERT_EQ(EC_OK, backend_->Close());
}

TEST_F(MetaCachedBackendTest, RecoverReadPersistentFallback) {
    ASSERT_EQ(EC_OK, backend_->Init(instance_id_, config_));
    ASSERT_EQ(EC_OK, backend_->Open());
    WaitForRunning(backend_.get());

    // Put key 1 into both, put key 2 only into persistent (via redis_store_ directly)
    backend_->Put({1}, {{{"f1", "v1"}}});
    (*redis_store_)[cache_key_prefix_ + "2"] = {{"f1", "v2"}};

    // Force back to Recover state
    ForceRecoverState(backend_.get());

    // Key 1 hits local, key 2 misses local → falls back to persistent for key 2 only
    AssertGet(backend_.get(), {1, 2}, {"f1"}, {EC_OK, EC_OK}, {{{"f1", "v1"}}, {{"f1", "v2"}}});
    AssertGetAllFields(backend_.get(), {1, 2}, {EC_OK, EC_OK}, {{{"f1", "v1"}}, {{"f1", "v2"}}});
    AssertExists(backend_.get(), {1, 2}, {EC_OK, EC_OK}, {true, true});

    // Key 3 doesn't exist anywhere
    AssertExists(backend_.get(), {3}, {EC_OK}, {false});

    ASSERT_EQ(EC_OK, backend_->Close());
}

// ==================== Recover phase: write operations ====================

TEST_F(MetaCachedBackendTest, RecoverWriteDualWrite) {
    ASSERT_EQ(EC_OK, backend_->Init(instance_id_, config_));
    ASSERT_EQ(EC_OK, backend_->Open());
    WaitForRunning(backend_.get());

    ForceRecoverState(backend_.get());

    // Put in Recover state → should dual-write to both persistent and local
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Put({10}, {{{"fa", "va"}}}));

    // Readable from both phases
    AssertGet(backend_.get(), {10}, {"fa"}, {EC_OK}, {{{"fa", "va"}}});

    // UpdateFields in Recover state (triggers EnsureKeyInLocal)
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->UpdateFields({10}, {{{"fa", "va-updated"}}}));
    AssertGet(backend_.get(), {10}, {"fa"}, {EC_OK}, {{{"fa", "va-updated"}}});

    // Delete in Recover state
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Delete({10}));
    AssertExists(backend_.get(), {10}, {EC_OK}, {false});

    ASSERT_EQ(EC_OK, backend_->Close());
}

// ==================== Running phase: read operations ====================

TEST_F(MetaCachedBackendTest, RunningReadLocalOnly) {
    ASSERT_EQ(EC_OK, backend_->Init(instance_id_, config_));
    ASSERT_EQ(EC_OK, backend_->Open());
    WaitForRunning(backend_.get());

    // Put key into both via dual-write
    backend_->Put({1}, {{{"f1", "v1"}}});

    // Put key 2 only into persistent (simulating stale data in Redis)
    (*redis_store_)[cache_key_prefix_ + "2"] = {{"f2", "v2"}};

    // In Running state, reads should only come from local
    // Key 1 is in local → found
    AssertGet(backend_.get(), {1}, {"f1"}, {EC_OK}, {{{"f1", "v1"}}});
    AssertGetAllFields(backend_.get(), {1}, {EC_OK}, {{{"f1", "v1"}}});
    AssertExists(backend_.get(), {1}, {EC_OK}, {true});

    // Key 2 is only in persistent → NOT found in Running state (local-only reads)
    AssertExists(backend_.get(), {2}, {EC_OK}, {false});

    ASSERT_EQ(EC_OK, backend_->Close());
}

// ==================== Running phase: write operations ====================

TEST_F(MetaCachedBackendTest, RunningWriteDualWrite) {
    ASSERT_EQ(EC_OK, backend_->Init(instance_id_, config_));
    ASSERT_EQ(EC_OK, backend_->Open());
    WaitForRunning(backend_.get());

    // Put
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Put({1}, {{{"f1", "v1"}}}));
    AssertGet(backend_.get(), {1}, {"f1"}, {EC_OK}, {{{"f1", "v1"}}});

    // UpdateFields
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->UpdateFields({1}, {{{"f1", "v1-updated"}}}));
    AssertGet(backend_.get(), {1}, {"f1"}, {EC_OK}, {{{"f1", "v1-updated"}}});

    // Upsert
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Upsert({1}, {{{"f1", "v1-upserted"}, {"f3", "v3"}}}));
    AssertGet(backend_.get(), {1}, {"f1", "f3"}, {EC_OK}, {{{"f1", "v1-upserted"}, {"f3", "v3"}}});

    // Delete
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Delete({1}));
    AssertExists(backend_.get(), {1}, {EC_OK}, {false});

    // Verify persistent is also updated: force Recover and read from persistent fallback
    ForceRecoverState(backend_.get());
    AssertExists(backend_.get(), {1}, {EC_OK}, {false});

    ASSERT_EQ(EC_OK, backend_->Close());
}

// ==================== storage_uri parsing tests ====================

TEST_F(MetaCachedBackendTest, TestParseStorageUriDefaultTypes) {
    // Default URI without persistent_type/cache_type params should use defaults (redis/local)
    auto config = std::make_shared<MetaStorageBackendConfig>();
    config->SetStorageType(META_CACHED_BACKEND_TYPE_STR);
    config->SetStorageUri("redis://test_redis_user:test_redis_password@test_redis_host:0/?client_max_pool_size=1");
    auto backend = std::make_shared<MetaCachedBackend>();
    ASSERT_EQ(EC_OK, backend->Init(instance_id_, config));

    auto local_backend = dynamic_cast<MetaLocalBackend *>(backend->local_backend_.get());
    ASSERT_TRUE(local_backend);
    ASSERT_EQ(META_REDIS_BACKEND_TYPE_STR, backend->persistent_backend_->GetStorageType());
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR, local_backend->GetStorageType());
    ASSERT_EQ(META_LOCAL_BACKEND_DEFAULT_CAPACITY * 1024 * 1024, local_backend->cache_->GetCapacity());
    uint32_t expected_shard_mask = (1 << META_LOCAL_BACKEND_DEFAULT_NUM_SHARD_BITS) - 1;
    ASSERT_EQ(expected_shard_mask, local_backend->shard_mask_);
    ASSERT_EQ(META_LOCAL_BACKEND_DEFAULT_SAMPLE_TIMES, local_backend->sample_times_);
}

TEST_F(MetaCachedBackendTest, TestParseStorageUriWithExplicitTypes) {
    // URI with explicit persistent_type and cache_type params
    auto config = std::make_shared<MetaStorageBackendConfig>();
    config->SetStorageType(META_CACHED_BACKEND_TYPE_STR);
    config->SetStorageUri("redis://test_redis_user:test_redis_password@test_redis_host:0/"
                          "?client_max_pool_size=1&persistent_type=redis&cache_type=local");
    auto backend = std::make_shared<MetaCachedBackend>();
    ASSERT_EQ(EC_OK, backend->Init(instance_id_, config));
    ASSERT_EQ(META_REDIS_BACKEND_TYPE_STR, backend->persistent_backend_->GetStorageType());
    auto local_backend = dynamic_cast<MetaLocalBackend *>(backend->local_backend_.get());
    ASSERT_TRUE(local_backend);
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR, local_backend->GetStorageType());

    config->SetStorageUri("redis://test_redis_user:test_redis_password@test_redis_host:0/"
                          "?client_max_pool_size=1&persistent_type=invalid&cache_type=local");
    ASSERT_EQ(EC_ERROR, backend->Init(instance_id_, config));

    config->SetStorageUri("redis://test_redis_user:test_redis_password@test_redis_host:0/"
                          "?client_max_pool_size=1&persistent_type=redis&cache_type=invalid");
    ASSERT_EQ(EC_ERROR, backend->Init(instance_id_, config));
}

TEST_F(MetaCachedBackendTest, TestParseStorageUriWithLocalParams) {
    // URI with local backend params (capacity, num_shard_bits, sample_times)
    // These are parsed by MetaLocalBackend::Init, not MetaCachedBackend::Init,
    // but the URI is forwarded to the local backend via cache_config->SetStorageUri.
    auto config = std::make_shared<MetaStorageBackendConfig>();
    config->SetStorageType(META_CACHED_BACKEND_TYPE_STR);
    config->SetStorageUri("redis://test_redis_user:test_redis_password@test_redis_host:0/"
                          "?client_max_pool_size=1&capacity=1024&num_shard_bits=4&sample_times=50");
    auto backend = std::make_shared<MetaCachedBackend>();
    ASSERT_EQ(EC_OK, backend->Init(instance_id_, config));
    auto local_backend = dynamic_cast<MetaLocalBackend *>(backend->local_backend_.get());
    ASSERT_TRUE(local_backend);
    ASSERT_EQ(1024 * 1024 * 1024, local_backend->cache_->GetCapacity());
    uint32_t expected_shard_mask = (1 << 4) - 1;
    ASSERT_EQ(expected_shard_mask, local_backend->shard_mask_);
    ASSERT_EQ(50, local_backend->sample_times_);
}

} // namespace kv_cache_manager
