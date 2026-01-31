#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/common/test/mock_redis_client.h"
#include "kv_cache_manager/common/test/redis_test_base.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class RedisClientTest : public RedisTestBase, public TESTBASE {
public:
    void SetUp() override;

    void TearDown() override {}

public:
    void PrepareMockRedisClient(const std::string &user_info, const std::string &host, int32_t port);
    void CheckReplyType(const redisReply *r, const int expected_type) const;

private:
    std::unique_ptr<MockRedisClient> redis_client_;
};

void RedisClientTest::SetUp() { PrepareMockRedisClient("test_user:test_password", "test_host", /*port*/ 0); }

void RedisClientTest::PrepareMockRedisClient(const std::string &user_info, const std::string &host, int32_t port) {
    StandardUri storage_uri;
    storage_uri.user_info_ = user_info;
    storage_uri.hostname_ = host;
    storage_uri.port_ = port;
    storage_uri.params_["timeout_ms"] = "2000";
    storage_uri.params_["retry_count"] = "2";

    redis_client_ = std::make_unique<MockRedisClient>(storage_uri);
    ON_CALL(*redis_client_, IsContextOk()).WillByDefault(Invoke(redis_client_.get(), &RedisClient::IsContextOk));
    ON_CALL(*redis_client_, Reconnect()).WillByDefault(Invoke(redis_client_.get(), &RedisClient::Reconnect));
    ON_CALL(*redis_client_, TryExecPipeline(_))
        .WillByDefault(Invoke(redis_client_.get(), &RedisClient::TryExecPipeline));
}

void RedisClientTest::CheckReplyType(const redisReply *r, const int expected_type) const {
    ASSERT_TRUE(r);
    ASSERT_EQ(expected_type, r->type);
}

TEST_F(RedisClientTest, TestCommandPipelineSimple) {
    EXPECT_CALL(*redis_client_, IsContextOk()).WillRepeatedly(Return(true));

    std::vector<ReplyUPtr> prepared_replies;
    prepared_replies.emplace_back(MakeFakeReply(REDIS_REPLY_STATUS, "OK"));
    prepared_replies.emplace_back(MakeFakeReply(REDIS_REPLY_STATUS, "OK"));
    EXPECT_CALL(*redis_client_, TryExecPipeline(_)).WillOnce(Return(ByMove(std::move(prepared_replies))));

    std::vector<CmdArgs> cmds{{"fake_valid_cmd1"}, {"fake_valid_cmd2"}};
    std::vector<ReplyUPtr> replies = redis_client_->CommandPipeline(cmds);
    ASSERT_EQ(2, replies.size());
    CheckReplyType(replies[0].get(), REDIS_REPLY_STATUS);
    ASSERT_EQ("OK", std::string(replies[0]->str));
    CheckReplyType(replies[1].get(), REDIS_REPLY_STATUS);
    ASSERT_EQ("OK", std::string(replies[1]->str));
}

TEST_F(RedisClientTest, TestCommandPipelineError) {
    {
        // connection error always
        EXPECT_CALL(*redis_client_, IsContextOk()).WillRepeatedly(Return(false));
        EXPECT_CALL(*redis_client_, Reconnect()).WillRepeatedly(Return(false));

        std::vector<CmdArgs> cmds{{"fake_valid_cmd1"}, {"fake_valid_cmd2"}};
        std::vector<ReplyUPtr> replies = redis_client_->CommandPipeline(cmds);
        ASSERT_EQ(0, replies.size());
    }
    {
        // connection error and reconnect
        EXPECT_CALL(*redis_client_, IsContextOk()).WillOnce(Return(false)).WillRepeatedly(Return(true));
        EXPECT_CALL(*redis_client_, Reconnect()).WillRepeatedly(Return(true));

        std::vector<ReplyUPtr> prepared_replies;
        prepared_replies.emplace_back(MakeFakeReply(REDIS_REPLY_STATUS, "OK"));
        prepared_replies.emplace_back(MakeFakeReply(REDIS_REPLY_STATUS, "OK"));
        EXPECT_CALL(*redis_client_, TryExecPipeline(_)).WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<CmdArgs> cmds{{"fake_valid_cmd1"}, {"fake_valid_cmd2"}};
        std::vector<ReplyUPtr> replies = redis_client_->CommandPipeline(cmds);
        ASSERT_EQ(2, replies.size());
        CheckReplyType(replies[0].get(), REDIS_REPLY_STATUS);
        ASSERT_EQ("OK", std::string(replies[0]->str));
        CheckReplyType(replies[1].get(), REDIS_REPLY_STATUS);
        ASSERT_EQ("OK", std::string(replies[1]->str));
    }
    {
        // connection error and reconnect error
        EXPECT_CALL(*redis_client_, IsContextOk()).WillOnce(Return(false)).WillRepeatedly(Return(true));
        EXPECT_CALL(*redis_client_, Reconnect()).WillOnce(Return(false)).WillRepeatedly(Return(true));

        std::vector<CmdArgs> cmds{{"fake_valid_cmd1"}, {"fake_valid_cmd2"}};
        std::vector<ReplyUPtr> replies = redis_client_->CommandPipeline(cmds);
        ASSERT_EQ(0, replies.size());
    }
    {
        // connection error in first pipeline, ok in second
        bool is_connection_ok = true;
        EXPECT_CALL(*redis_client_, IsContextOk()).WillRepeatedly(ReturnPointee(&is_connection_ok));
        EXPECT_CALL(*redis_client_, Reconnect()).WillRepeatedly(DoAll(Assign(&is_connection_ok, true), Return(true)));

        std::vector<ReplyUPtr> prepared_replies_0;
        std::vector<ReplyUPtr> prepared_replies_1;
        prepared_replies_1.emplace_back(MakeFakeReply(REDIS_REPLY_STATUS, "OK"));
        prepared_replies_1.emplace_back(MakeFakeReply(REDIS_REPLY_STATUS, "OK"));
        EXPECT_CALL(*redis_client_, TryExecPipeline(_))
            .WillOnce(DoAll(Assign(&is_connection_ok, false), Return(ByMove(std::move(prepared_replies_0)))))
            .WillOnce(Return(ByMove(std::move(prepared_replies_1))));

        std::vector<CmdArgs> cmds{{"fake_valid_cmd1"}, {"fake_valid_cmd2"}};
        std::vector<ReplyUPtr> replies = redis_client_->CommandPipeline(cmds);
        ASSERT_EQ(2, replies.size());
        CheckReplyType(replies[0].get(), REDIS_REPLY_STATUS);
        ASSERT_EQ("OK", std::string(replies[0]->str));
        CheckReplyType(replies[1].get(), REDIS_REPLY_STATUS);
        ASSERT_EQ("OK", std::string(replies[1]->str));
    }
    {
        // connection ok but cmds is illegal
        EXPECT_CALL(*redis_client_, IsContextOk()).WillRepeatedly(Return(true));
        EXPECT_CALL(*redis_client_, Reconnect()).WillRepeatedly(Return(true));

        std::vector<ReplyUPtr> prepared_replies;
        EXPECT_CALL(*redis_client_, TryExecPipeline(_)).WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<CmdArgs> cmds{{"fake_valid_cmd1"}, {"fake_valid_cmd2"}};
        std::vector<ReplyUPtr> replies = redis_client_->CommandPipeline(cmds);
        ASSERT_EQ(0, replies.size());
    }
    {
        // cmds is empty
        std::vector<CmdArgs> cmds;
        std::vector<ReplyUPtr> replies = redis_client_->CommandPipeline(cmds);
        ASSERT_EQ(0, replies.size());
    }
}

TEST_F(RedisClientTest, TestSet) {
    EXPECT_CALL(*redis_client_, IsContextOk()).WillRepeatedly(Return(true));
    {
        std::vector<ReplyUPtr> prepared_replies;
        prepared_replies.emplace_back(MakeFakeReplyInteger(1));
        prepared_replies.emplace_back(MakeFakeReplyInteger(2));
        prepared_replies.emplace_back(MakeFakeReplyInteger(1));
        prepared_replies.emplace_back(MakeFakeReplyInteger(2));
        EXPECT_CALL(
            *redis_client_,
            TryExecPipeline(ElementsAre(
                ElementsAre(StrEq("DEL"), StrEq("key1")),
                ElementsAre(StrEq("HSET"), StrEq("key1"), StrEq("f1"), StrEq("v1-1-0"), StrEq("f2"), StrEq("v1-2-0")),
                ElementsAre(StrEq("DEL"), StrEq("key2")),
                ElementsAre(StrEq("HSET"), StrEq("key2"), StrEq("f1"), StrEq("v2-1-0"), StrEq("f2"), StrEq("v2-2-0")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<std::map<std::string, std::string>> field_maps{{{"f1", "v1-1-0"}, {"f2", "v1-2-0"}},
                                                                   {{"f1", "v2-1-0"}, {"f2", "v2-2-0"}}};
        auto ec_per_key = redis_client_->Set(keys, field_maps);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    }
    {
        std::vector<ReplyUPtr> prepared_replies;
        prepared_replies.emplace_back(MakeFakeReplyInteger(-1));
        prepared_replies.emplace_back(MakeFakeReplyInteger(2));
        prepared_replies.emplace_back(MakeFakeReplyInteger(1));
        prepared_replies.emplace_back(MakeFakeReplyInteger(-2));
        EXPECT_CALL(
            *redis_client_,
            TryExecPipeline(ElementsAre(
                ElementsAre(StrEq("DEL"), StrEq("key1")),
                ElementsAre(StrEq("HSET"), StrEq("key1"), StrEq("f1"), StrEq("v1-1-0"), StrEq("f2"), StrEq("v1-2-0")),
                ElementsAre(StrEq("DEL"), StrEq("key2")),
                ElementsAre(StrEq("HSET"), StrEq("key2"), StrEq("f1"), StrEq("v2-1-0"), StrEq("f2"), StrEq("v2-2-0")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<std::map<std::string, std::string>> field_maps{{{"f1", "v1-1-0"}, {"f2", "v1-2-0"}},
                                                                   {{"f1", "v2-1-0"}, {"f2", "v2-2-0"}}};
        auto ec_per_key = redis_client_->Set(keys, field_maps);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_ERROR), ec_per_key);
    }
    {
        std::vector<ReplyUPtr> prepared_replies;
        EXPECT_CALL(
            *redis_client_,
            TryExecPipeline(ElementsAre(
                ElementsAre(StrEq("DEL"), StrEq("key1")),
                ElementsAre(StrEq("HSET"), StrEq("key1"), StrEq("f1"), StrEq("v1-1-0"), StrEq("f2"), StrEq("v1-2-0")),
                ElementsAre(StrEq("DEL"), StrEq("key2")),
                ElementsAre(StrEq("HSET"), StrEq("key2"), StrEq("f1"), StrEq("v2-1-0"), StrEq("f2"), StrEq("v2-2-0")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<std::map<std::string, std::string>> field_maps{{{"f1", "v1-1-0"}, {"f2", "v1-2-0"}},
                                                                   {{"f1", "v2-1-0"}, {"f2", "v2-2-0"}}};
        auto ec_per_key = redis_client_->Set(keys, field_maps);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_ERROR), ec_per_key);
    }
}

TEST_F(RedisClientTest, TestUpdate) {
    EXPECT_CALL(*redis_client_, IsContextOk()).WillRepeatedly(Return(true));
    {
        std::vector<ReplyUPtr> prepared_exists_replies;
        prepared_exists_replies.emplace_back(MakeFakeReplyInteger(1));
        prepared_exists_replies.emplace_back(MakeFakeReplyInteger(1));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("EXISTS"), StrEq("key1")),
                                                ElementsAre(StrEq("EXISTS"), StrEq("key2")))))
            .WillOnce(Return(ByMove(std::move(prepared_exists_replies))));

        std::vector<ReplyUPtr> prepared_hset_replies;
        prepared_hset_replies.emplace_back(MakeFakeReplyInteger(2));
        prepared_hset_replies.emplace_back(MakeFakeReplyInteger(2));
        EXPECT_CALL(
            *redis_client_,
            TryExecPipeline(ElementsAre(
                ElementsAre(StrEq("HSET"), StrEq("key1"), StrEq("f1"), StrEq("v1-1-0"), StrEq("f2"), StrEq("v1-2-0")),
                ElementsAre(StrEq("HSET"), StrEq("key2"), StrEq("f1"), StrEq("v2-1-0"), StrEq("f2"), StrEq("v2-2-0")))))
            .WillOnce(Return(ByMove(std::move(prepared_hset_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<std::map<std::string, std::string>> field_maps{{{"f1", "v1-1-0"}, {"f2", "v1-2-0"}},
                                                                   {{"f1", "v2-1-0"}, {"f2", "v2-2-0"}}};
        auto ec_per_key = redis_client_->Update(keys, field_maps);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    }
    {
        std::vector<ReplyUPtr> prepared_exists_replies;
        prepared_exists_replies.emplace_back(MakeFakeReplyInteger(0));
        prepared_exists_replies.emplace_back(MakeFakeReplyInteger(1));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("EXISTS"), StrEq("key1")),
                                                ElementsAre(StrEq("EXISTS"), StrEq("key2")))))
            .WillOnce(Return(ByMove(std::move(prepared_exists_replies))));

        std::vector<ReplyUPtr> prepared_hset_replies;
        prepared_hset_replies.emplace_back(MakeFakeReplyInteger(0));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(
                        StrEq("HSET"), StrEq("key2"), StrEq("f1"), StrEq("v2-1-0"), StrEq("f2"), StrEq("v2-2-0")))))
            .WillOnce(Return(ByMove(std::move(prepared_hset_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<std::map<std::string, std::string>> field_maps{{{"f1", "v1-1-0"}, {"f2", "v1-2-0"}},
                                                                   {{"f1", "v2-1-0"}, {"f2", "v2-2-0"}}};
        auto ec_per_key = redis_client_->Update(keys, field_maps);
        std::vector<ErrorCode> expected_ec_per_key{EC_NOENT, EC_OK};
        ASSERT_EQ(expected_ec_per_key, ec_per_key);
    }
    {
        std::vector<ReplyUPtr> prepared_exists_replies;
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("EXISTS"), StrEq("key1")),
                                                ElementsAre(StrEq("EXISTS"), StrEq("key2")))))
            .WillOnce(Return(ByMove(std::move(prepared_exists_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<std::map<std::string, std::string>> field_maps{{{"f1", "v1-1-0"}, {"f2", "v1-2-0"}},
                                                                   {{"f1", "v2-1-0"}, {"f2", "v2-2-0"}}};
        auto ec_per_key = redis_client_->Update(keys, field_maps);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_ERROR), ec_per_key);
    }
}

TEST_F(RedisClientTest, TestDelete) {
    EXPECT_CALL(*redis_client_, IsContextOk()).WillRepeatedly(Return(true));
    {
        std::vector<ReplyUPtr> prepared_replies;
        prepared_replies.emplace_back(MakeFakeReplyInteger(1));
        prepared_replies.emplace_back(MakeFakeReplyInteger(1));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("DEL"), StrEq("key1")),
                                                ElementsAre(StrEq("DEL"), StrEq("key2")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        auto ec_per_key = redis_client_->Delete(keys);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    }
    {
        std::vector<ReplyUPtr> prepared_replies;
        prepared_replies.emplace_back(MakeFakeReplyInteger(-1));
        prepared_replies.emplace_back(MakeFakeReplyInteger(0));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("DEL"), StrEq("key1")),
                                                ElementsAre(StrEq("DEL"), StrEq("key2")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        auto ec_per_key = redis_client_->Delete(keys);
        std::vector<ErrorCode> expected_ec_per_key{EC_ERROR, EC_NOENT};
        ASSERT_EQ(expected_ec_per_key, ec_per_key);
    }
    {
        std::vector<ReplyUPtr> prepared_replies;
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("DEL"), StrEq("key1")),
                                                ElementsAre(StrEq("DEL"), StrEq("key2")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        auto ec_per_key = redis_client_->Delete(keys);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_ERROR), ec_per_key);
    }
}

TEST_F(RedisClientTest, TestGet) {
    EXPECT_CALL(*redis_client_, IsContextOk()).WillRepeatedly(Return(true));
    {
        std::vector<ReplyUPtr> prepared_hmget_replies;
        prepared_hmget_replies.emplace_back(MakeFakeReplyArrayString({"v1-1-0", "v1-2-0"}));
        prepared_hmget_replies.emplace_back(MakeFakeReplyArrayString({"v2-1-0", "v2-2-0"}));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("HMGET"), StrEq("key1"), StrEq("f1"), StrEq("f2")),
                                                ElementsAre(StrEq("HMGET"), StrEq("key2"), StrEq("f1"), StrEq("f2")))))
            .WillOnce(Return(ByMove(std::move(prepared_hmget_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<std::map<std::string, std::string>> field_maps;
        std::vector<std::map<std::string, std::string>> expected_field_maps{{{"f1", "v1-1-0"}, {"f2", "v1-2-0"}},
                                                                            {{"f1", "v2-1-0"}, {"f2", "v2-2-0"}}};
        auto ec_per_key = redis_client_->Get(keys, /*field_names*/ {"f1", "f2"}, field_maps);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
        ASSERT_EQ(expected_field_maps, field_maps);
    }
    {
        std::vector<ReplyUPtr> prepared_hmget_replies;
        prepared_hmget_replies.emplace_back(MakeFakeReplyArrayString({std::nullopt, std::nullopt})); // key not exist
        prepared_hmget_replies.emplace_back(MakeFakeReplyArrayString({std::nullopt, "v2-2-0"}));     // f1 not exist
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("HMGET"), StrEq("key1"), StrEq("f1"), StrEq("f2")),
                                                ElementsAre(StrEq("HMGET"), StrEq("key2"), StrEq("f1"), StrEq("f2")))))
            .WillOnce(Return(ByMove(std::move(prepared_hmget_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<std::map<std::string, std::string>> field_maps;
        std::vector<std::map<std::string, std::string>> expected_field_maps{{{"f1", ""}, {"f2", ""}},
                                                                            {{"f1", ""}, {"f2", "v2-2-0"}}};
        std::vector<ErrorCode> expected_ec_per_key{EC_OK, EC_OK};
        auto ec_per_key = redis_client_->Get(keys, /*field_names*/ {"f1", "f2"}, field_maps);
        ASSERT_EQ(expected_ec_per_key, ec_per_key);
        ASSERT_EQ(expected_field_maps, field_maps);
    }
    {
        // all key not exist
        std::vector<ReplyUPtr> prepared_hmget_replies;
        prepared_hmget_replies.emplace_back(MakeFakeReplyArrayString({std::nullopt, std::nullopt}));
        prepared_hmget_replies.emplace_back(MakeFakeReplyArrayString({std::nullopt, std::nullopt}));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("HMGET"), StrEq("key1"), StrEq("f1"), StrEq("f2")),
                                                ElementsAre(StrEq("HMGET"), StrEq("key2"), StrEq("f1"), StrEq("f2")))))
            .WillOnce(Return(ByMove(std::move(prepared_hmget_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<std::map<std::string, std::string>> field_maps;
        std::vector<std::map<std::string, std::string>> expected_field_maps{{{"f1", ""}, {"f2", ""}},
                                                                            {{"f1", ""}, {"f2", ""}}};
        std::vector<ErrorCode> expected_ec_per_key{EC_OK, EC_OK};
        auto ec_per_key = redis_client_->Get(keys, /*field_names*/ {"f1", "f2"}, field_maps);
        ASSERT_EQ(expected_ec_per_key, ec_per_key);
        ASSERT_EQ(expected_field_maps, field_maps);
    }
    {
        std::vector<ReplyUPtr> prepared_hmget_replies;
        prepared_hmget_replies.emplace_back(MakeFakeReplyInteger(1)); // reply is not array
        prepared_hmget_replies.emplace_back(MakeFakeReplyArrayString({std::nullopt, "v2-2-0"}));
        prepared_hmget_replies[1]->element[1]->type = REDIS_REPLY_ERROR; // f2 reply error
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("HMGET"), StrEq("key1"), StrEq("f1"), StrEq("f2")),
                                                ElementsAre(StrEq("HMGET"), StrEq("key2"), StrEq("f1"), StrEq("f2")))))
            .WillOnce(Return(ByMove(std::move(prepared_hmget_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<std::map<std::string, std::string>> field_maps;
        std::vector<std::map<std::string, std::string>> expected_field_maps(2);
        std::vector<ErrorCode> expected_ec_per_key{EC_ERROR, EC_ERROR};
        auto ec_per_key = redis_client_->Get(keys, /*field_names*/ {"f1", "f2"}, field_maps);
        ASSERT_EQ(expected_ec_per_key, ec_per_key);
        ASSERT_EQ(expected_field_maps, field_maps);
    }
}

TEST_F(RedisClientTest, TestGetAllFields) {
    EXPECT_CALL(*redis_client_, IsContextOk()).WillRepeatedly(Return(true));
    {
        std::vector<ReplyUPtr> prepared_hgetall_replies;
        prepared_hgetall_replies.emplace_back(MakeFakeReplyArrayString({"f1", "v1-1-0", "f2", "v1-2-0"}));
        prepared_hgetall_replies.emplace_back(MakeFakeReplyArrayString({"f1", "v2-1-0", "f2", "v2-2-0"}));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("HGETALL"), StrEq("key1")),
                                                ElementsAre(StrEq("HGETALL"), StrEq("key2")))))
            .WillOnce(Return(ByMove(std::move(prepared_hgetall_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<std::map<std::string, std::string>> field_maps;
        std::vector<std::map<std::string, std::string>> expected_field_maps{{{"f1", "v1-1-0"}, {"f2", "v1-2-0"}},
                                                                            {{"f1", "v2-1-0"}, {"f2", "v2-2-0"}}};
        auto ec_per_key = redis_client_->GetAllFields(keys, field_maps);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
        ASSERT_EQ(expected_field_maps, field_maps);
    }
    {
        std::vector<ReplyUPtr> prepared_hgetall_replies;
        prepared_hgetall_replies.emplace_back(MakeFakeReplyArrayString({})); // not exist
        prepared_hgetall_replies.emplace_back(MakeFakeReplyInteger(1));      // error
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("HGETALL"), StrEq("key1")),
                                                ElementsAre(StrEq("HGETALL"), StrEq("key2")))))
            .WillOnce(Return(ByMove(std::move(prepared_hgetall_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<std::map<std::string, std::string>> field_maps;
        std::vector<std::map<std::string, std::string>> expected_field_maps{{}, {}};
        std::vector<ErrorCode> expected_ec_per_key{EC_NOENT, EC_ERROR};
        auto ec_per_key = redis_client_->GetAllFields(keys, field_maps);
        ASSERT_EQ(expected_ec_per_key, ec_per_key);
        ASSERT_EQ(expected_field_maps, field_maps);
    }
    {
        std::vector<ReplyUPtr> prepared_hgetall_replies;
        prepared_hgetall_replies.emplace_back(MakeFakeReplyArrayString({"f1", "v1-1-0"}));
        prepared_hgetall_replies.emplace_back(MakeFakeReplyArrayString({"f1", "v2-1-0", "f2"})); // invalid element 3
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("HGETALL"), StrEq("key1")),
                                                ElementsAre(StrEq("HGETALL"), StrEq("key2")))))
            .WillOnce(Return(ByMove(std::move(prepared_hgetall_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<std::map<std::string, std::string>> field_maps;
        std::vector<std::map<std::string, std::string>> expected_field_maps{{{"f1", "v1-1-0"}}, {}};
        std::vector<ErrorCode> expected_ec_per_key{EC_OK, EC_ERROR};
        auto ec_per_key = redis_client_->GetAllFields(keys, field_maps);
        ASSERT_EQ(expected_ec_per_key, ec_per_key);
        ASSERT_EQ(expected_field_maps, field_maps);
    }
}

TEST_F(RedisClientTest, TestExists) {
    EXPECT_CALL(*redis_client_, IsContextOk()).WillRepeatedly(Return(true));
    {
        std::vector<ReplyUPtr> prepared_replies;
        prepared_replies.emplace_back(MakeFakeReplyInteger(1));
        prepared_replies.emplace_back(MakeFakeReplyInteger(1));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("EXISTS"), StrEq("key1")),
                                                ElementsAre(StrEq("EXISTS"), StrEq("key2")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<bool> is_exist_vec;
        auto ec_per_key = redis_client_->Exists(keys, is_exist_vec);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
        ASSERT_EQ(std::vector<bool>(keys.size(), true), is_exist_vec);
    }
    {
        std::vector<ReplyUPtr> prepared_replies;
        prepared_replies.emplace_back(MakeFakeReply(REDIS_REPLY_ERROR, "ERROR")); // error
        prepared_replies.emplace_back(MakeFakeReplyInteger(0));                   // not exist
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("EXISTS"), StrEq("key1")),
                                                ElementsAre(StrEq("EXISTS"), StrEq("key2")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<bool> is_exist_vec;
        auto ec_per_key = redis_client_->Exists(keys, is_exist_vec);
        std::vector<ErrorCode> expected_ec_per_key{EC_ERROR, EC_OK};
        ASSERT_EQ(expected_ec_per_key, ec_per_key);
        std::vector<bool> expected_is_exist_vec{false, false};
        ASSERT_EQ(expected_is_exist_vec, is_exist_vec);
    }
    {
        std::vector<ReplyUPtr> prepared_replies;
        prepared_replies.emplace_back(MakeFakeReply(REDIS_REPLY_ERROR, "ERROR")); // all error
        prepared_replies.emplace_back(MakeFakeReply(REDIS_REPLY_ERROR, "ERROR")); // all error
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("EXISTS"), StrEq("key1")),
                                                ElementsAre(StrEq("EXISTS"), StrEq("key2")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys{"key1", "key2"};
        std::vector<bool> is_exist_vec;
        auto ec_per_key = redis_client_->Exists(keys, is_exist_vec);
        std::vector<ErrorCode> expected_ec_per_key{EC_ERROR, EC_ERROR};
        ASSERT_EQ(expected_ec_per_key, ec_per_key);
        std::vector<bool> expected_is_exist_vec{false, false};
        ASSERT_EQ(expected_is_exist_vec, is_exist_vec);
    }
}

TEST_F(RedisClientTest, TestScan) {
    EXPECT_CALL(*redis_client_, IsContextOk()).WillRepeatedly(Return(true));
    {
        std::vector<ReplyUPtr> prepared_replies;
        std::vector<std::optional<std::string>> expected_keys_opt{"instance:key1", "instance:key2", "instance:key3"};
        prepared_replies.emplace_back(MakeFakeReplyScan(/*next_cursor*/ "5", expected_keys_opt));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(
                        StrEq("SCAN"), StrEq("0"), StrEq("MATCH"), StrEq("instance:*"), StrEq("COUNT"), StrEq("5")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::string next_cursor;
        std::vector<std::string> keys;
        std::vector<std::string> expected_keys{"instance:key1", "instance:key2", "instance:key3"};
        auto ec = redis_client_->Scan(/*matching_prefix*/ "instance:", /*cursor*/ "0", /*limit*/ 5, next_cursor, keys);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ("5", next_cursor);
        ASSERT_EQ(expected_keys, keys);
    }
    {
        std::vector<ReplyUPtr> prepared_replies;
        std::vector<std::optional<std::string>> expected_keys_opt{};
        prepared_replies.emplace_back(MakeFakeReplyScan(/*next_cursor*/ "5", expected_keys_opt));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(
                        StrEq("SCAN"), StrEq("0"), StrEq("MATCH"), StrEq("instance:*"), StrEq("COUNT"), StrEq("5")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::string next_cursor;
        std::vector<std::string> keys;
        std::vector<std::string> expected_keys{};
        auto ec = redis_client_->Scan(/*matching_prefix*/ "instance:", /*cursor*/ "0", /*limit*/ 5, next_cursor, keys);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ("5", next_cursor);
        ASSERT_EQ(expected_keys, keys);
    }
    {
        std::vector<ReplyUPtr> prepared_replies;
        std::vector<std::optional<std::string>> expected_keys_opt{std::nullopt}; // invalid nil key
        prepared_replies.emplace_back(MakeFakeReplyScan(/*next_cursor*/ "5", expected_keys_opt));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(
                        StrEq("SCAN"), StrEq("0"), StrEq("MATCH"), StrEq("instance:*"), StrEq("COUNT"), StrEq("5")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::string next_cursor;
        std::vector<std::string> keys;
        std::vector<std::string> expected_keys{};
        auto ec = redis_client_->Scan(/*matching_prefix*/ "instance:", /*cursor*/ "0", /*limit*/ 5, next_cursor, keys);
        ASSERT_EQ(EC_ERROR, ec);
        ASSERT_EQ("", next_cursor);
        ASSERT_EQ(expected_keys, keys);
    }
    {
        std::vector<ReplyUPtr> prepared_replies;
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(
                        StrEq("SCAN"), StrEq("0"), StrEq("MATCH"), StrEq("instance:*"), StrEq("COUNT"), StrEq("5")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::string next_cursor;
        std::vector<std::string> keys;
        std::vector<std::string> expected_keys{};
        auto ec = redis_client_->Scan(/*matching_prefix*/ "instance:", /*cursor*/ "0", /*limit*/ 5, next_cursor, keys);
        ASSERT_EQ(EC_ERROR, ec);
        ASSERT_EQ("", next_cursor);
        ASSERT_EQ(expected_keys, keys);
    }
}

TEST_F(RedisClientTest, TestRand) {
    EXPECT_CALL(*redis_client_, IsContextOk()).WillRepeatedly(Return(true));
    {
        std::vector<ReplyUPtr> prepared_replies;
        for (int i = 0; i < 20; ++i) {
            if (i & 1) {
                prepared_replies.emplace_back(MakeFakeReply(REDIS_REPLY_STRING, "some_other_key"));
            } else {
                prepared_replies.emplace_back(MakeFakeReply(REDIS_REPLY_NIL, ""));
            }
        }
        prepared_replies[0] = MakeFakeReply(REDIS_REPLY_STRING, "instance1:key1");
        prepared_replies[9] = MakeFakeReply(REDIS_REPLY_STRING, "instance2:key2");
        prepared_replies[19] = MakeFakeReply(REDIS_REPLY_STRING, "instance1:key3");
        std::vector<CmdArgs> expected_cmds(20, CmdArgs{"RANDOMKEY"});
        EXPECT_CALL(*redis_client_, TryExecPipeline(ElementsAreArray(expected_cmds)))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys;
        auto ec = redis_client_->Rand(/*matching_prefix*/ "instance1:", /*count*/ 2, keys);
        ASSERT_EQ(EC_OK, ec);
        std::vector<std::string> expected_keys{"instance1:key1", "instance1:key3"};
        ASSERT_EQ(expected_keys, keys);
    }
    {
        std::vector<ReplyUPtr> prepared_replies_0;
        for (int i = 0; i < 20; ++i) {
            prepared_replies_0.emplace_back(MakeFakeReply(REDIS_REPLY_STRING, "some_other_key"));
        }
        std::vector<ReplyUPtr> prepared_replies_1;
        for (int i = 0; i < 20; ++i) {
            prepared_replies_1.emplace_back(MakeFakeReply(REDIS_REPLY_STRING, "some_other_key"));
        }
        std::vector<ReplyUPtr> prepared_replies_2;
        for (int i = 0; i < 20; ++i) {
            prepared_replies_2.emplace_back(MakeFakeReply(REDIS_REPLY_STRING, "some_other_key"));
        }
        std::vector<ReplyUPtr> prepared_replies_3;
        for (int i = 0; i < 20; ++i) {
            prepared_replies_3.emplace_back(MakeFakeReply(REDIS_REPLY_STRING, "some_other_key"));
        }
        prepared_replies_0[0] = MakeFakeReply(REDIS_REPLY_STRING, "instance1:key1");
        prepared_replies_0[9] = MakeFakeReply(REDIS_REPLY_STRING, "instance2:key2");
        prepared_replies_0[19] = MakeFakeReply(REDIS_REPLY_STRING, "instance1:key3");
        std::vector<CmdArgs> expected_cmds(20, CmdArgs{"RANDOMKEY"});
        EXPECT_CALL(*redis_client_, TryExecPipeline(ElementsAreArray(expected_cmds)))
            .WillOnce(Return(ByMove(std::move(prepared_replies_0))))
            .WillOnce(Return(ByMove(std::move(prepared_replies_1))))
            .WillOnce(Return(ByMove(std::move(prepared_replies_2))))
            .WillOnce(Return(ByMove(std::move(prepared_replies_3))));

        std::vector<std::string> keys;
        auto ec = redis_client_->Rand(/*matching_prefix*/ "instance1:", /*count*/ 3, keys);
        ASSERT_EQ(EC_OK, ec);
        std::vector<std::string> expected_keys{"instance1:key1", "instance1:key3"};
        ASSERT_EQ(expected_keys, keys);
    }
    {
        std::vector<ReplyUPtr> prepared_replies_0;
        for (int i = 0; i < 20; ++i) {
            prepared_replies_0.emplace_back(MakeFakeReply(REDIS_REPLY_STRING, "some_other_key"));
        }
        prepared_replies_0[0] = MakeFakeReply(REDIS_REPLY_STRING, "instance1:key1");
        prepared_replies_0[9] = MakeFakeReply(REDIS_REPLY_STRING, "instance2:key2");
        prepared_replies_0[19] = MakeFakeReply(REDIS_REPLY_STRING, "instance1:key3");
        std::vector<ReplyUPtr> prepared_replies_1; // empty, error happened
        std::vector<CmdArgs> expected_cmds_0(20, CmdArgs{"RANDOMKEY"});
        std::vector<CmdArgs> expected_cmds_1(20, CmdArgs{"RANDOMKEY"});
        EXPECT_CALL(*redis_client_, TryExecPipeline(ElementsAreArray(expected_cmds_0)))
            .WillOnce(Return(ByMove(std::move(prepared_replies_0))))
            .WillOnce(Return(ByMove(std::move(prepared_replies_1))));

        std::vector<std::string> keys;
        auto ec = redis_client_->Rand(/*matching_prefix*/ "instance1:", /*count*/ 3, keys);
        ASSERT_EQ(EC_ERROR, ec);
        std::vector<std::string> expected_keys{};
        ASSERT_EQ(expected_keys, keys);
    }
}

TEST_F(RedisClientTest, TestKeysAndFieldsWithSpaces) {
    EXPECT_CALL(*redis_client_, IsContextOk()).WillRepeatedly(Return(true));

    // Test key with spaces for Set method
    {
        std::vector<ReplyUPtr> prepared_replies;
        prepared_replies.emplace_back(MakeFakeReplyInteger(1)); // DEL reply
        prepared_replies.emplace_back(MakeFakeReplyInteger(2)); // HSET reply
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("DEL"), StrEq("key with spaces")),
                                                ElementsAre(StrEq("HSET"),
                                                            StrEq("key with spaces"),
                                                            StrEq("another field"),
                                                            StrEq("another value"),
                                                            StrEq("field with spaces"),
                                                            StrEq("value with spaces")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys{"key with spaces"};
        std::vector<std::map<std::string, std::string>> field_maps{
            {{"field with spaces", "value with spaces"}, {"another field", "another value"}}};
        auto ec_per_key = redis_client_->Set(keys, field_maps);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    }

    // Test key with spaces for Update method
    {
        std::vector<ReplyUPtr> prepared_exists_replies;
        prepared_exists_replies.emplace_back(MakeFakeReplyInteger(1));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("EXISTS"), StrEq("key with spaces")))))
            .WillOnce(Return(ByMove(std::move(prepared_exists_replies))));

        std::vector<ReplyUPtr> prepared_hset_replies;
        prepared_hset_replies.emplace_back(MakeFakeReplyInteger(2));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("HSET"),
                                                            StrEq("key with spaces"),
                                                            StrEq("another field"),
                                                            StrEq("another value"),
                                                            StrEq("field with spaces"),
                                                            StrEq("value with spaces")))))
            .WillOnce(Return(ByMove(std::move(prepared_hset_replies))));

        std::vector<std::string> keys{"key with spaces"};
        std::vector<std::map<std::string, std::string>> field_maps{
            {{"field with spaces", "value with spaces"}, {"another field", "another value"}}};
        auto ec_per_key = redis_client_->Update(keys, field_maps);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    }

    // Test key with spaces for Delete method
    {
        std::vector<ReplyUPtr> prepared_replies;
        prepared_replies.emplace_back(MakeFakeReplyInteger(1));
        EXPECT_CALL(*redis_client_, TryExecPipeline(ElementsAre(ElementsAre(StrEq("DEL"), StrEq("key with spaces")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys{"key with spaces"};
        auto ec_per_key = redis_client_->Delete(keys);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    }

    // Test key with spaces for Get method
    {
        std::vector<ReplyUPtr> prepared_hmget_replies;
        prepared_hmget_replies.emplace_back(MakeFakeReplyArrayString({"value with spaces", "another value"}));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(
                        StrEq("HMGET"), StrEq("key with spaces"), StrEq("field with spaces"), StrEq("another field")))))
            .WillOnce(Return(ByMove(std::move(prepared_hmget_replies))));

        std::vector<std::string> keys{"key with spaces"};
        std::vector<std::map<std::string, std::string>> field_maps;
        std::vector<std::map<std::string, std::string>> expected_field_maps{
            {{"field with spaces", "value with spaces"}, {"another field", "another value"}}};
        auto ec_per_key = redis_client_->Get(keys, /*field_names*/ {"field with spaces", "another field"}, field_maps);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
        ASSERT_EQ(expected_field_maps, field_maps);
    }

    // Test key with spaces for GetAllFields method
    {
        std::vector<ReplyUPtr> prepared_hgetall_replies;
        prepared_hgetall_replies.emplace_back(
            MakeFakeReplyArrayString({"another field", "another value", "field with spaces", "value with spaces"}));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("HGETALL"), StrEq("key with spaces")))))
            .WillOnce(Return(ByMove(std::move(prepared_hgetall_replies))));

        std::vector<std::string> keys{"key with spaces"};
        std::vector<std::map<std::string, std::string>> field_maps;
        std::vector<std::map<std::string, std::string>> expected_field_maps{
            {{"field with spaces", "value with spaces"}, {"another field", "another value"}}};
        auto ec_per_key = redis_client_->GetAllFields(keys, field_maps);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
        ASSERT_EQ(expected_field_maps, field_maps);
    }

    // Test key with spaces for Exists method
    {
        std::vector<ReplyUPtr> prepared_replies;
        prepared_replies.emplace_back(MakeFakeReplyInteger(1));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(StrEq("EXISTS"), StrEq("key with spaces")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys{"key with spaces"};
        std::vector<bool> is_exist_vec;
        auto ec_per_key = redis_client_->Exists(keys, is_exist_vec);
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
        ASSERT_EQ(std::vector<bool>(keys.size(), true), is_exist_vec);
    }

    // Test key with spaces for Scan method
    {
        std::vector<ReplyUPtr> prepared_replies;
        std::vector<std::optional<std::string>> expected_keys_opt{"key with spaces", "another key with spaces"};
        prepared_replies.emplace_back(MakeFakeReplyScan(/*next_cursor*/ "0", expected_keys_opt));
        EXPECT_CALL(*redis_client_,
                    TryExecPipeline(ElementsAre(ElementsAre(
                        StrEq("SCAN"), StrEq("0"), StrEq("MATCH"), StrEq("key with*"), StrEq("COUNT"), StrEq("10")))))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::string next_cursor;
        std::vector<std::string> keys;
        std::vector<std::string> expected_keys{"key with spaces", "another key with spaces"};
        auto ec = redis_client_->Scan(/*matching_prefix*/ "key with", /*cursor*/ "0", /*limit*/ 10, next_cursor, keys);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ("0", next_cursor);
        ASSERT_EQ(expected_keys, keys);
    }

    // Test key with spaces for Rand method
    {
        std::vector<ReplyUPtr> prepared_replies;
        for (int i = 0; i < 20; ++i) {
            if (i & 1) {
                prepared_replies.emplace_back(MakeFakeReply(REDIS_REPLY_STRING, "some other key"));
            } else {
                prepared_replies.emplace_back(MakeFakeReply(REDIS_REPLY_NIL, ""));
            }
        }
        prepared_replies[0] = MakeFakeReply(REDIS_REPLY_STRING, "key with spaces");
        prepared_replies[9] = MakeFakeReply(REDIS_REPLY_STRING, "key with spaces another");
        prepared_replies[19] = MakeFakeReply(REDIS_REPLY_STRING, "key with spaces yet another");
        std::vector<CmdArgs> expected_cmds(20, CmdArgs{"RANDOMKEY"});
        EXPECT_CALL(*redis_client_, TryExecPipeline(ElementsAreArray(expected_cmds)))
            .WillOnce(Return(ByMove(std::move(prepared_replies))));

        std::vector<std::string> keys;
        auto ec = redis_client_->Rand(/*matching_prefix*/ "key with", /*count*/ 2, keys);
        ASSERT_EQ(EC_OK, ec);
        std::vector<std::string> expected_keys{"key with spaces", "key with spaces another"};
        ASSERT_EQ(expected_keys, keys);
    }
}

} // namespace kv_cache_manager
