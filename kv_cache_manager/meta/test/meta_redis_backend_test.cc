#include <thread>

#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/common/test/mock_redis_client.h"
#include "kv_cache_manager/common/test/redis_test_base.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_redis_backend.h"
#include "kv_cache_manager/meta/test/meta_storage_backend_test_base.h"

namespace kv_cache_manager {
class MockMetaRedisBackend : public MetaRedisBackend {
public:
    MOCK_METHOD(std::unique_ptr<RedisClient>, CreateRedisClient, (), (const));
};

class MetaRedisBackendTest : public MetaStorageBackendTestBase, public RedisTestBase, public TESTBASE {
public:
    void SetUp() override;

    void TearDown() override {}

    void ConstructMetaRedisBackend();
    void ConstructMetaStorageBackendConfig();

private:
    std::unique_ptr<MockMetaRedisBackend> meta_redis_backend_;
    std::shared_ptr<MetaStorageBackendConfig> meta_storage_backend_config_;
};

void MetaRedisBackendTest::SetUp() {
    ConstructMetaRedisBackend();
    ConstructMetaStorageBackendConfig();
}

void MetaRedisBackendTest::ConstructMetaRedisBackend() {
    meta_redis_backend_ = std::make_unique<MockMetaRedisBackend>();
}

void MetaRedisBackendTest::ConstructMetaStorageBackendConfig() {
    meta_storage_backend_config_ = std::make_shared<MetaStorageBackendConfig>();
    meta_storage_backend_config_->SetStorageType(META_REDIS_BACKEND_TYPE_STR);
    meta_storage_backend_config_->SetStorageUri(
        "redis://test_redis_user:test_redis_password@test_redis_host:0/?client_pool_size=1");
}

TEST_F(MetaRedisBackendTest, TestInit) {
    ASSERT_EQ(EC_OK, meta_redis_backend_->Init("instance_0", meta_storage_backend_config_));
    ASSERT_EQ(META_REDIS_BACKEND_TYPE_STR, meta_redis_backend_->GetStorageType());

    {
        // invalid nullptr config
        ConstructMetaRedisBackend();
        ASSERT_EQ(EC_BADARGS, meta_redis_backend_->Init("instance_0", /*config*/ nullptr));
    }
    {
        // invalid empty storage uri
        ConstructMetaRedisBackend();
        ConstructMetaStorageBackendConfig();
        meta_storage_backend_config_->SetStorageUri("");
        ASSERT_EQ(EC_BADARGS, meta_redis_backend_->Init("instance_0", meta_storage_backend_config_));
    }
    {
        // invalid empty instance id
        ConstructMetaRedisBackend();
        ASSERT_EQ(EC_BADARGS, meta_redis_backend_->Init(/*instance_id*/ "", meta_storage_backend_config_));
    }
}

TEST_F(MetaRedisBackendTest, TestInvalidClientPoolSize) {
    EXPECT_CALL(*meta_redis_backend_, CreateRedisClient()).WillRepeatedly(Invoke([]() {
        StandardUri empty_storage_uri;
        auto mock_redis_client = std::make_unique<MockRedisClient>(empty_storage_uri);
        EXPECT_CALL(*mock_redis_client, Reconnect()).WillRepeatedly(Return(true));
        return mock_redis_client;
    }));
    meta_storage_backend_config_->SetStorageUri(
        "redis://test_redis_user:test_redis_password@test_redis_host:0/?client_pool_size=-1");
    ASSERT_EQ(EC_OK, meta_redis_backend_->Init("instance_0", meta_storage_backend_config_));
    // open first
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());
    ASSERT_EQ(16, meta_redis_backend_->GetPoolState()->client_pool_.size());
    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
}

TEST_F(MetaRedisBackendTest, TestOpenAndClose) {
    EXPECT_CALL(*meta_redis_backend_, CreateRedisClient()).WillRepeatedly(Invoke([]() {
        StandardUri empty_storage_uri;
        auto mock_redis_client = std::make_unique<MockRedisClient>(empty_storage_uri);
        EXPECT_CALL(*mock_redis_client, Reconnect()).WillRepeatedly(Return(true));
        return mock_redis_client;
    }));
    meta_storage_backend_config_->SetStorageUri(
        "redis://test_redis_user:test_redis_password@test_redis_host:0/?client_pool_size=2");
    ASSERT_EQ(EC_OK, meta_redis_backend_->Init("instance_0", meta_storage_backend_config_));
    // open first
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());
    {
        MetaRedisBackend::ClientHandle handle1 = meta_redis_backend_->AcquireClientFromPool();
        ASSERT_TRUE(handle1);
        MetaRedisBackend::ClientHandle handle2 = meta_redis_backend_->AcquireClientFromPool();
        ASSERT_TRUE(handle2);
        MetaRedisBackend::ClientHandle handle3 = meta_redis_backend_->AcquireClientFromPool();
        ASSERT_FALSE(handle3);
    }
    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
    {
        MetaRedisBackend::ClientHandle handle1 = meta_redis_backend_->AcquireClientFromPool();
        ASSERT_FALSE(handle1);
    }
    // reopen second
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());
    {
        MetaRedisBackend::ClientHandle handle1 = meta_redis_backend_->AcquireClientFromPool();
        ASSERT_TRUE(handle1);
        MetaRedisBackend::ClientHandle handle2 = meta_redis_backend_->AcquireClientFromPool();
        ASSERT_TRUE(handle2);
        MetaRedisBackend::ClientHandle handle3 = meta_redis_backend_->AcquireClientFromPool();
        ASSERT_FALSE(handle3);
    }
    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
    {
        MetaRedisBackend::ClientHandle handle1 = meta_redis_backend_->AcquireClientFromPool();
        ASSERT_FALSE(handle1);
    }

    {
        ConstructMetaRedisBackend();
        ConstructMetaStorageBackendConfig();
        meta_storage_backend_config_->SetStorageUri(
            "redis://test_redis_user:test_redis_password@test_redis_host:0/?client_pool_size=2");
        EXPECT_CALL(*meta_redis_backend_, CreateRedisClient()).WillOnce(Invoke([]() {
            StandardUri empty_storage_uri;
            auto mock_redis_client = std::make_unique<MockRedisClient>(empty_storage_uri);
            EXPECT_CALL(*mock_redis_client, Reconnect()).WillOnce(Return(false));
            return mock_redis_client;
        }));
        ASSERT_EQ(EC_OK, meta_redis_backend_->Init("instance_0", meta_storage_backend_config_));
        ASSERT_EQ(EC_ERROR, meta_redis_backend_->Open());
    }
}

TEST_F(MetaRedisBackendTest, TestSimple) {
    EXPECT_CALL(*meta_redis_backend_, CreateRedisClient()).WillOnce(Invoke([]() {
        StandardUri empty_storage_uri;
        auto mock_redis_client = std::make_unique<MockRedisClient>(empty_storage_uri);
        EXPECT_CALL(*mock_redis_client, IsContextOk()).WillRepeatedly(Return(true));
        EXPECT_CALL(*mock_redis_client, Reconnect()).WillRepeatedly(Return(true));
        std::vector<ReplyUPtr> put_replies;
        put_replies.emplace_back(MakeFakeReplyInteger(1));
        put_replies.emplace_back(MakeFakeReplyInteger(1));
        put_replies.emplace_back(MakeFakeReplyInteger(1));
        put_replies.emplace_back(MakeFakeReplyInteger(1));
        EXPECT_CALL(
            *mock_redis_client,
            TryExecPipeline(ElementsAre(
                ElementsAre(StrEq("DEL"), StrEq("kvcache:instance_instance_0:cache_1")),
                ElementsAre(StrEq("HSET"), StrEq("kvcache:instance_instance_0:cache_1"), StrEq("f1"), StrEq("v1-1")),
                ElementsAre(StrEq("DEL"), ("kvcache:instance_instance_0:cache_2")),
                ElementsAre(StrEq("HSET"), StrEq("kvcache:instance_instance_0:cache_2"), StrEq("f1"), StrEq("v2-1")))))
            .WillOnce(Return(ByMove(std::move(put_replies))));

        std::vector<ReplyUPtr> update_replies_1;
        update_replies_1.emplace_back(MakeFakeReplyInteger(1));
        update_replies_1.emplace_back(MakeFakeReplyInteger(1));
        update_replies_1.emplace_back(MakeFakeReplyInteger(0));
        EXPECT_CALL(
            *mock_redis_client,
            TryExecPipeline(ElementsAre(ElementsAre(StrEq("EXISTS"), StrEq("kvcache:instance_instance_0:cache_1")),
                                        ElementsAre(StrEq("EXISTS"), StrEq("kvcache:instance_instance_0:cache_2")),
                                        ElementsAre(StrEq("EXISTS"), StrEq("kvcache:instance_instance_0:cache_3")))))
            .WillOnce(Return(ByMove(std::move(update_replies_1))));
        std::vector<ReplyUPtr> update_replies_2;
        update_replies_2.emplace_back(MakeFakeReplyInteger(1));
        update_replies_2.emplace_back(MakeFakeReplyInteger(1));
        EXPECT_CALL(
            *mock_redis_client,
            TryExecPipeline(ElementsAre(
                ElementsAre(StrEq("HSET"), StrEq("kvcache:instance_instance_0:cache_1"), StrEq("f2"), StrEq("v1-2")),
                ElementsAre(StrEq("HSET"), StrEq("kvcache:instance_instance_0:cache_2"), StrEq("f2"), StrEq("v2-2")))))
            .WillOnce(Return(ByMove(std::move(update_replies_2))));

        std::vector<ReplyUPtr> exist_replies;
        exist_replies.emplace_back(MakeFakeReplyInteger(1));
        exist_replies.emplace_back(MakeFakeReplyInteger(1));
        exist_replies.emplace_back(MakeFakeReplyInteger(0));
        EXPECT_CALL(
            *mock_redis_client,
            TryExecPipeline(ElementsAre(ElementsAre(StrEq("EXISTS"), StrEq("kvcache:instance_instance_0:cache_1")),
                                        ElementsAre(StrEq("EXISTS"), StrEq("kvcache:instance_instance_0:cache_2")),
                                        ElementsAre(StrEq("EXISTS"), StrEq("kvcache:instance_instance_0:cache_4")))))
            .WillOnce(Return(ByMove(std::move(exist_replies))));

        std::vector<ReplyUPtr> get_replies;
        get_replies.emplace_back(MakeFakeReplyArrayString({"v1-1", "v1-2"}));
        get_replies.emplace_back(MakeFakeReplyArrayString({"v2-1", "v2-2"}));
        EXPECT_CALL(
            *mock_redis_client,
            TryExecPipeline(ElementsAre(
                ElementsAre(StrEq("HMGET"), StrEq("kvcache:instance_instance_0:cache_1"), StrEq("f1"), StrEq("f2")),
                ElementsAre(StrEq("HMGET"), StrEq("kvcache:instance_instance_0:cache_2"), StrEq("f1"), StrEq("f2")))))
            .WillOnce(Return(ByMove(std::move(get_replies))));
        std::vector<ReplyUPtr> get_all_fields_replies;
        get_all_fields_replies.emplace_back(MakeFakeReplyArrayString({"f1", "v1-1", "f2", "v1-2"}));
        get_all_fields_replies.emplace_back(MakeFakeReplyArrayString({"f1", "v2-1", "f2", "v2-2"}));
        EXPECT_CALL(
            *mock_redis_client,
            TryExecPipeline(ElementsAre(ElementsAre(StrEq("HGETALL"), StrEq("kvcache:instance_instance_0:cache_1")),
                                        ElementsAre(StrEq("HGETALL"), StrEq("kvcache:instance_instance_0:cache_2")))))
            .WillOnce(Return(ByMove(std::move(get_all_fields_replies))));

        std::vector<ReplyUPtr> list_keys_replies;
        list_keys_replies.emplace_back(MakeFakeReplyScan(
            /*next_cursor*/ "5", {"kvcache:instance_instance_0:cache_1", "kvcache:instance_instance_0:cache_2"}));
        EXPECT_CALL(*mock_redis_client,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("SCAN"),
                                                            StrEq("0"),
                                                            StrEq("MATCH"),
                                                            StrEq("kvcache:instance_instance_0:cache_*"),
                                                            StrEq("COUNT"),
                                                            StrEq("5")))))
            .WillOnce(Return(ByMove(std::move(list_keys_replies))));

        std::vector<ReplyUPtr> random_replies;
        for (int i = 0; i < 20; ++i) {
            random_replies.emplace_back(MakeFakeReply(REDIS_REPLY_STRING, "some_other_key"));
        }
        random_replies[0] = MakeFakeReply(REDIS_REPLY_STRING, "kvcache:instance_instance_0:cache_1");
        random_replies[9] = MakeFakeReply(REDIS_REPLY_STRING, "kvcache:instance_instance_0:cache_2");
        std::vector<std::vector<std::string>> randomkey_commands(20, {"RANDOMKEY"});
        EXPECT_CALL(*mock_redis_client, TryExecPipeline(ElementsAreArray(randomkey_commands)))
            .WillOnce(Return(ByMove(std::move(random_replies))));

        // test upsert
        std::vector<ReplyUPtr> upsert_replies;
        upsert_replies.emplace_back(MakeFakeReplyInteger(1));
        upsert_replies.emplace_back(MakeFakeReplyInteger(1));
        EXPECT_CALL(
            *mock_redis_client,
            TryExecPipeline(ElementsAre(
                ElementsAre(StrEq("HSET"), StrEq("kvcache:instance_instance_0:cache_2"), StrEq("f1"), StrEq("v2-1-2")),
                ElementsAre(StrEq("HSET"), StrEq("kvcache:instance_instance_0:cache_3"), StrEq("f1"), StrEq("v3-1")))))
            .WillOnce(Return(ByMove(std::move(upsert_replies))));

        std::vector<ReplyUPtr> get_replies2;
        get_replies2.emplace_back(MakeFakeReplyArrayString({"v2-1-2", "v2-2"}));
        get_replies2.emplace_back(MakeFakeReplyArrayString({"v3-1", std::nullopt}));
        EXPECT_CALL(
            *mock_redis_client,
            TryExecPipeline(ElementsAre(
                ElementsAre(StrEq("HMGET"), StrEq("kvcache:instance_instance_0:cache_2"), StrEq("f1"), StrEq("f2")),
                ElementsAre(StrEq("HMGET"), StrEq("kvcache:instance_instance_0:cache_3"), StrEq("f1"), StrEq("f2")))))
            .WillOnce(Return(ByMove(std::move(get_replies2))));
        return mock_redis_client;
    }));

    ASSERT_EQ(EC_OK, meta_redis_backend_->Init("instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_redis_backend_->Put({1, 2}, {{{"f1", "v1-1"}}, {{"f1", "v2-1"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_NOENT}),
              meta_redis_backend_->UpdateFields({1, 2, 3}, {{{"f2", "v1-2"}}, {{"f2", "v2-2"}}, {{"f3", "v3-2"}}}));

    AssertExists(meta_redis_backend_.get(), {1, 2, 4}, {EC_OK, EC_OK, EC_OK}, /*is_exist*/ {true, true, false});
    AssertGet(meta_redis_backend_.get(),
              {1, 2},
              {"f1", "f2"},
              {EC_OK, EC_OK},
              {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});
    AssertGetAllFields(meta_redis_backend_.get(),
                       {1, 2},
                       {EC_OK, EC_OK},
                       {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});
    AssertListKeys(
        meta_redis_backend_.get(), SCAN_BASE_CURSOR, /*limit*/ 5, EC_OK, /*expected_next_cursor*/ "5", {1, 2});
    AssertRandomSample(meta_redis_backend_.get(), /*count*/ 2, EC_OK, {1, 2});

    // test upsert
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_redis_backend_->Upsert({2, 3}, {{{"f1", "v2-1-2"}}, {{"f1", "v3-1"}}}));
    AssertGet(meta_redis_backend_.get(),
              {2, 3},
              {"f1", "f2"},
              {EC_OK, EC_OK},
              {{{"f1", "v2-1-2"}, {"f2", "v2-2"}}, {{"f1", "v3-1"}, {"f2", ""}}});

    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
}

TEST_F(MetaRedisBackendTest, TestRedisError) {
    EXPECT_CALL(*meta_redis_backend_, CreateRedisClient()).WillOnce(Invoke([]() {
        StandardUri empty_storage_uri;
        auto mock_redis_client = std::make_unique<MockRedisClient>(empty_storage_uri);
        EXPECT_CALL(*mock_redis_client, Reconnect()).WillOnce(Return(true)).WillRepeatedly(Return(false));
        EXPECT_CALL(*mock_redis_client, IsContextOk()).WillRepeatedly(Return(false));
        return mock_redis_client;
    }));

    ASSERT_EQ(EC_OK, meta_redis_backend_->Init("instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());
    ASSERT_EQ((std::vector<ErrorCode>{EC_ERROR, EC_ERROR}),
              meta_redis_backend_->Put({1, 2}, {{{"f1", "v1-1"}}, {{"f1", "v2-1"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_ERROR, EC_ERROR, EC_ERROR}),
              meta_redis_backend_->UpdateFields({1, 2, 3}, {{{"f2", "v1-2"}}, {{"f2", "v2-2"}}, {{"f3", "v3-2"}}}));

    std::vector<bool> is_exist_vec;
    ASSERT_EQ((std::vector<ErrorCode>{EC_ERROR, EC_ERROR}), meta_redis_backend_->Exists({1, 2}, is_exist_vec));
    ASSERT_EQ(std::vector<bool>(2, false), is_exist_vec);

    FieldMapVec field_maps;
    ASSERT_EQ((std::vector<ErrorCode>{EC_ERROR, EC_ERROR}), meta_redis_backend_->Get({1, 2}, {"f1", "f2"}, field_maps));
    ASSERT_EQ(std::vector<FieldMap>(2), field_maps);
    ASSERT_EQ((std::vector<ErrorCode>{EC_ERROR, EC_ERROR}), meta_redis_backend_->GetAllFields({1, 2}, field_maps));
    ASSERT_EQ(std::vector<FieldMap>(2), field_maps);

    std::string next_cursor;
    KeyTypeVec keys;
    ASSERT_EQ(EC_ERROR, meta_redis_backend_->ListKeys(SCAN_BASE_CURSOR, /*limit*/ 5, next_cursor, keys));
    ASSERT_TRUE(next_cursor.empty());
    ASSERT_TRUE(keys.empty());

    ASSERT_EQ(EC_ERROR, meta_redis_backend_->RandomSample(/*limit*/ 5, keys));
    ASSERT_TRUE(keys.empty());

    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
}

TEST_F(MetaRedisBackendTest, TestMultiThreadSimple) {
    auto make_mock_client = []() {
        StandardUri empty_storage_uri;
        auto mock_redis_client = std::make_unique<MockRedisClient>(empty_storage_uri);
        EXPECT_CALL(*mock_redis_client, Reconnect()).WillRepeatedly(Return(true));
        EXPECT_CALL(*mock_redis_client, IsContextOk()).WillRepeatedly(Return(true));
        EXPECT_CALL(
            *mock_redis_client,
            TryExecPipeline(ElementsAre(
                ElementsAre(StrEq("HMGET"), StrEq("kvcache:instance_instance_0:cache_1"), StrEq("f1"), StrEq("f2")),
                ElementsAre(StrEq("HMGET"), StrEq("kvcache:instance_instance_0:cache_2"), StrEq("f1"), StrEq("f2")))))
            .WillRepeatedly(Invoke([]() {
                usleep(5 * 1000); // assume network use 5ms
                std::vector<ReplyUPtr> get_replies_2;
                get_replies_2.emplace_back(MakeFakeReplyArrayString({"v1-1", "v1-2"}));
                get_replies_2.emplace_back(MakeFakeReplyArrayString({"v2-1", "v2-2"}));
                return get_replies_2;
            }));
        return mock_redis_client;
    };
    EXPECT_CALL(*meta_redis_backend_, CreateRedisClient())
        .WillOnce(Invoke(make_mock_client))
        .WillOnce(Invoke(make_mock_client));
    meta_storage_backend_config_->SetStorageUri(
        "redis://test_redis_user:test_redis_password@test_redis_host:0/?client_pool_size=2");
    ASSERT_EQ(EC_OK, meta_redis_backend_->Init("instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_redis_backend_->Open());

    auto get_task = [this]() {
        AssertGet(meta_redis_backend_.get(),
                  {1, 2},
                  {"f1", "f2"},
                  {EC_OK, EC_OK},
                  {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(get_task);
        usleep(2 * 1000);
    }
    for (auto &thread : threads) {
        thread.join();
    }
    ASSERT_EQ(EC_OK, meta_redis_backend_->Close());
}

} // namespace kv_cache_manager
