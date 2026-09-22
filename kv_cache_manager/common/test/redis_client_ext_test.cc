#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/redis_client_ext.h"
#include "kv_cache_manager/common/test/redis_test_base.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {
namespace {

class RecoverableRedisClientExt : public RedisClientExt, private RedisTestBase {
public:
    explicit RecoverableRedisClientExt(const StandardUri &storage_uri) : RedisClientExt(storage_uri) {}

    void BreakConnection() { context_ok_ = false; }
    void ReturnErrorOnce(const std::string &error) { next_error_ = error; }
    void SetReconnectSucceeds(bool succeeds) { reconnect_succeeds_ = succeeds; }
    int32_t ReconnectCount() const { return reconnect_count_; }

protected:
    bool IsContextOk() const override { return context_ok_; }

    bool Reconnect() override {
        ++reconnect_count_;
        context_ok_ = reconnect_succeeds_;
        return reconnect_succeeds_;
    }

    std::vector<RedisClient::ReplyUPtr> TryExecPipeline(const std::vector<RedisClient::CmdArgs> &cmds) override {
        std::vector<RedisClient::ReplyUPtr> replies;
        replies.reserve(cmds.size());
        for (const auto &cmd : cmds) {
            replies.emplace_back(MakeReply(cmd));
        }
        return replies;
    }

private:
    RedisClient::ReplyUPtr MakeReply(const RedisClient::CmdArgs &cmd) {
        if (!next_error_.empty()) {
            auto error = std::move(next_error_);
            return MakeFakeReply(REDIS_REPLY_ERROR, error);
        }
        if (cmd[0] == "EVAL" || cmd[0] == "EVALSHA") {
            return MakeFakeReplyInteger(1);
        }
        if (cmd[0] == "SCRIPT" && cmd[1] == "LOAD") {
            return MakeFakeReply(REDIS_REPLY_STRING, "script_sha1");
        }
        if (cmd[0] == "SCRIPT" && cmd[1] == "EXISTS") {
            return MakeIntegerArrayReply(1);
        }
        if (cmd[0] == "GET") {
            return MakeFakeReply(REDIS_REPLY_STRING, "value");
        }
        if (cmd[0] == "SET" || cmd[0] == "FLUSHALL") {
            return MakeFakeReply(REDIS_REPLY_STATUS, "OK");
        }
        return MakeFakeReplyInteger(cmd[0] == "PTTL" ? 1000 : 1);
    }

    static RedisClient::ReplyUPtr MakeIntegerArrayReply(int64_t value) {
        auto *reply = static_cast<redisReply *>(std::malloc(sizeof(redisReply)));
        std::memset(reply, 0, sizeof(redisReply));
        reply->type = REDIS_REPLY_ARRAY;
        reply->elements = 1;
        reply->element = static_cast<redisReply **>(std::malloc(sizeof(redisReply *)));
        reply->element[0] = MakeFakeReplyInteger(value).release();
        return RedisClient::ReplyUPtr(reply, freeReplyObject);
    }

private:
    bool context_ok_{true};
    std::string next_error_;
    bool reconnect_succeeds_{true};
    int32_t reconnect_count_{0};
};

TEST(RedisClientExtTest, EveryCommandRecoversAStaleConnectionThroughCommandPipeline) {
    RecoverableRedisClientExt client(StandardUri::FromUri("redis://localhost:6379?retry_count=2"));
    int32_t expected_reconnect_count = 0;

    std::string result;
    client.BreakConnection();
    EXPECT_EQ(EC_OK, client.Eval("return 1", {"key"}, {"arg"}, result));
    EXPECT_EQ("1", result);
    EXPECT_EQ(++expected_reconnect_count, client.ReconnectCount());

    client.BreakConnection();
    EXPECT_EQ(EC_OK, client.EvalSha("script_sha1", {"key"}, {"arg"}, result));
    EXPECT_EQ("1", result);
    EXPECT_EQ(++expected_reconnect_count, client.ReconnectCount());

    std::string sha1;
    client.BreakConnection();
    EXPECT_EQ(EC_OK, client.ScriptLoad("return 1", sha1));
    EXPECT_EQ("script_sha1", sha1);
    EXPECT_EQ(++expected_reconnect_count, client.ReconnectCount());

    bool exists = false;
    client.BreakConnection();
    EXPECT_EQ(EC_OK, client.ScriptExists("script_sha1", exists));
    EXPECT_TRUE(exists);
    EXPECT_EQ(++expected_reconnect_count, client.ReconnectCount());

    std::string value;
    client.BreakConnection();
    EXPECT_EQ(EC_OK, client.Get("key", value));
    EXPECT_EQ("value", value);
    EXPECT_EQ(++expected_reconnect_count, client.ReconnectCount());

    client.BreakConnection();
    EXPECT_EQ(EC_OK, client.Set("key", "value", 1000));
    EXPECT_EQ(++expected_reconnect_count, client.ReconnectCount());

    int64_t ttl_ms = 0;
    client.BreakConnection();
    EXPECT_EQ(EC_OK, client.Pttl("key", ttl_ms));
    EXPECT_EQ(1000, ttl_ms);
    EXPECT_EQ(++expected_reconnect_count, client.ReconnectCount());

    client.BreakConnection();
    EXPECT_EQ(EC_OK, client.Del("key"));
    EXPECT_EQ(++expected_reconnect_count, client.ReconnectCount());

    client.BreakConnection();
    EXPECT_EQ(EC_OK, client.Pexpire("key", 1000));
    EXPECT_EQ(++expected_reconnect_count, client.ReconnectCount());

    client.BreakConnection();
    EXPECT_EQ(EC_OK, client.FlushAll());
    EXPECT_EQ(++expected_reconnect_count, client.ReconnectCount());
}

TEST(RedisClientExtTest, EvalShaReportsMissingCachedScript) {
    RecoverableRedisClientExt client(StandardUri::FromUri("redis://localhost:6379"));
    client.ReturnErrorOnce("NOSCRIPT No matching script. Please use EVAL.");

    std::string result;
    EXPECT_EQ(EC_NOSCRIPT, client.EvalSha("stale_sha1", {"key"}, {"arg"}, result));
}

TEST(RedisClientExtTest, ExecuteScriptReloadsMissingCachedScript) {
    RecoverableRedisClientExt client(StandardUri::FromUri("redis://localhost:6379"));
    client.ReturnErrorOnce("NOSCRIPT No matching script. Please use EVAL.");

    std::string cached_sha1 = "stale_sha1";
    std::string result;
    EXPECT_EQ(EC_OK, client.ExecuteScriptWithFallback("return 1", {"key"}, {"arg"}, cached_sha1, result));
    EXPECT_EQ("script_sha1", cached_sha1);
    EXPECT_EQ("1", result);
}

TEST(RedisClientExtTest, NextCommandRecoversAfterRedisBecomesReachable) {
    RecoverableRedisClientExt client(StandardUri::FromUri("redis://localhost:6379"));
    client.SetReconnectSucceeds(false);
    client.BreakConnection();

    std::string value;
    EXPECT_EQ(EC_IO_ERROR, client.Get("key", value));
    EXPECT_EQ(1, client.ReconnectCount());

    client.SetReconnectSucceeds(true);
    EXPECT_EQ(EC_OK, client.Get("key", value));
    EXPECT_EQ("value", value);
    EXPECT_EQ(2, client.ReconnectCount());
}

TEST(RedisClientExtTest, ScriptTransportFailureDoesNotStartASecondReconnectCycle) {
    RecoverableRedisClientExt client(StandardUri::FromUri("redis://localhost:6379"));
    client.SetReconnectSucceeds(false);
    client.BreakConnection();

    std::string cached_sha1 = "script_sha1";
    std::string result;
    EXPECT_EQ(EC_IO_ERROR, client.ExecuteScriptWithFallback("return 1", {"key"}, {"arg"}, cached_sha1, result));
    EXPECT_EQ(1, client.ReconnectCount());
}

TEST(RedisClientExtTest, EvalShaDoesNotTreatNoScriptOutsideErrorPrefixAsCacheMiss) {
    RecoverableRedisClientExt client(StandardUri::FromUri("redis://localhost:6379"));
    client.ReturnErrorOnce("ERR script returned the word NOSCRIPT");

    std::string result;
    EXPECT_EQ(EC_ERROR, client.EvalSha("script_sha1", {"key"}, {"arg"}, result));
}

} // namespace
} // namespace kv_cache_manager
