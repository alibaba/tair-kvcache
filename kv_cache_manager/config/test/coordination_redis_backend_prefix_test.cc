#include "kv_cache_manager/config/coordination_redis_backend.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/redis_client_ext.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

namespace {

StandardUri RedisUriWithParams(const std::string &params) {
    return StandardUri::FromUri("redis://localhost:6379/?" + params);
}

class PrefixRedisClient : public RedisClientExt {
public:
    using RedisClientExt::RedisClientExt;

protected:
    bool IsContextOk() const override { return true; }
    bool Reconnect() override { return true; }

    std::vector<ReplyUPtr> TryExecPipeline(const std::vector<CmdArgs> &cmds) override {
        std::vector<ReplyUPtr> replies;
        replies.reserve(cmds.size());
        for (const auto &cmd : cmds) {
            if (cmd.size() >= 2 && cmd[0] == "SCRIPT" && cmd[1] == "LOAD") {
                replies.emplace_back(MakeStringReply("fake_script_sha1"));
            } else {
                replies.emplace_back(MakeStringReply("OK"));
            }
        }
        return replies;
    }

private:
    static ReplyUPtr MakeStringReply(const std::string &value) {
        auto *reply = static_cast<redisReply *>(std::malloc(sizeof(redisReply)));
        std::memset(reply, 0, sizeof(redisReply));
        reply->type = REDIS_REPLY_STRING;
        reply->len = value.size();
        reply->str = static_cast<char *>(std::malloc(value.size() + 1));
        std::memcpy(reply->str, value.data(), value.size());
        reply->str[value.size()] = '\0';
        return ReplyUPtr(reply, freeReplyObject);
    }
};

class TestCoordinationRedisBackend : public CoordinationRedisBackend {
protected:
    std::shared_ptr<RedisClientExt> CreateRedisClient(const StandardUri &storage_uri) const override {
        return std::make_shared<PrefixRedisClient>(storage_uri);
    }
};

ErrorCode InitBackend(CoordinationRedisBackend &backend, const StandardUri &uri) { return backend.Init(uri); }

} // namespace

TEST(CoordinationRedisBackendPrefixTest, InitUsesLegacyPrefixWhenClusterNameIsMissing) {
    TestCoordinationRedisBackend backend;
    ASSERT_EQ(EC_OK, InitBackend(backend, RedisUriWithParams("timeout_ms=1000")));

    EXPECT_EQ("kvcm_lock:leader", backend.GetRedisLockKey("leader"));
    EXPECT_EQ("kvcm_kv:node_endpoint", backend.GetRedisKVKey("node_endpoint"));
}

TEST(CoordinationRedisBackendPrefixTest, InitUsesLegacyPrefixWhenClusterNameIsEmpty) {
    TestCoordinationRedisBackend backend;
    ASSERT_EQ(EC_OK, InitBackend(backend, RedisUriWithParams("cluster_name=")));

    EXPECT_EQ("kvcm_lock:leader", backend.GetRedisLockKey("leader"));
    EXPECT_EQ("kvcm_kv:node_endpoint", backend.GetRedisKVKey("node_endpoint"));
}

TEST(CoordinationRedisBackendPrefixTest, InitScopesGeneratedKeysByClusterName) {
    TestCoordinationRedisBackend backend;
    ASSERT_EQ(EC_OK, InitBackend(backend, RedisUriWithParams("cluster_name=test_cluster")));

    EXPECT_EQ("kvcm_test_cluster_lock:leader", backend.GetRedisLockKey("leader"));
    EXPECT_EQ("kvcm_test_cluster_kv:node_endpoint/127.0.0.1:6381",
              backend.GetRedisKVKey("node_endpoint/127.0.0.1:6381"));
}

TEST(CoordinationRedisBackendPrefixTest, OtherRedisParamsDoNotAffectClusterScope) {
    TestCoordinationRedisBackend backend;
    ASSERT_EQ(EC_OK,
              InitBackend(backend, RedisUriWithParams("timeout_ms=1000&retry_count=3&db=2&cluster_name=test_cluster")));

    EXPECT_EQ("kvcm_test_cluster_lock:leader", backend.GetRedisLockKey("leader"));
    EXPECT_EQ("kvcm_test_cluster_kv:leader", backend.GetRedisKVKey("leader"));
}

TEST(CoordinationRedisBackendPrefixTest, LockAndKVKeysAreSeparatedWithinSameCluster) {
    TestCoordinationRedisBackend backend;
    ASSERT_EQ(EC_OK, InitBackend(backend, RedisUriWithParams("cluster_name=test_cluster")));

    EXPECT_NE(backend.GetRedisLockKey("leader"), backend.GetRedisKVKey("leader"));
    EXPECT_EQ("kvcm_test_cluster_lock:leader", backend.GetRedisLockKey("leader"));
    EXPECT_EQ("kvcm_test_cluster_kv:leader", backend.GetRedisKVKey("leader"));
}

TEST(CoordinationRedisBackendPrefixTest, LogicalKeysAreAppendedWithoutMutation) {
    TestCoordinationRedisBackend backend;
    ASSERT_EQ(EC_OK, InitBackend(backend, RedisUriWithParams("cluster_name=test_cluster")));

    EXPECT_EQ("kvcm_test_cluster_lock:model/a/b:0", backend.GetRedisLockKey("model/a/b:0"));
    EXPECT_EQ("kvcm_test_cluster_kv:node_endpoint/10.0.0.1:6381",
              backend.GetRedisKVKey("node_endpoint/10.0.0.1:6381"));
}

TEST(CoordinationRedisBackendPrefixTest, DifferentClustersGenerateDifferentRedisKeysForSameLogicalKey) {
    TestCoordinationRedisBackend first;
    TestCoordinationRedisBackend second;
    ASSERT_EQ(EC_OK, InitBackend(first, RedisUriWithParams("cluster_name=cluster_a")));
    ASSERT_EQ(EC_OK, InitBackend(second, RedisUriWithParams("cluster_name=cluster_b")));

    EXPECT_NE(first.GetRedisLockKey("leader"), second.GetRedisLockKey("leader"));
    EXPECT_NE(first.GetRedisKVKey("node_endpoint"), second.GetRedisKVKey("node_endpoint"));
    EXPECT_EQ("kvcm_cluster_a_lock:leader", first.GetRedisLockKey("leader"));
    EXPECT_EQ("kvcm_cluster_b_lock:leader", second.GetRedisLockKey("leader"));
}

TEST(CoordinationRedisBackendPrefixTest, SameClusterNameGeneratesSameScopeAcrossDifferentRedisUris) {
    TestCoordinationRedisBackend first;
    TestCoordinationRedisBackend second;
    StandardUri first_uri = StandardUri::FromUri(
        "redis://user:pwd@redis-a:6379/?timeout_ms=1000&cluster_name=test_cluster");
    StandardUri second_uri = StandardUri::FromUri(
        "redis://redis-b:6380/?db=3&retry_count=5&cluster_name=test_cluster");
    ASSERT_EQ(EC_OK, InitBackend(first, first_uri));
    ASSERT_EQ(EC_OK, InitBackend(second, second_uri));

    EXPECT_EQ(first.GetRedisLockKey("leader"), second.GetRedisLockKey("leader"));
    EXPECT_EQ(first.GetRedisKVKey("node_endpoint"), second.GetRedisKVKey("node_endpoint"));
}

} // namespace kv_cache_manager
