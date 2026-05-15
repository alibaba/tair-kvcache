#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/redis_client_ext.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/coordination_redis_backend.h"

namespace kv_cache_manager {
namespace {

struct RedisClientConcurrencyState {
    std::atomic<int32_t> concurrent_pipeline_count{0};
};

class TrackingRedisClient : public RedisClientExt {
public:
    TrackingRedisClient(const StandardUri &storage_uri, std::shared_ptr<RedisClientConcurrencyState> state)
        : RedisClientExt(storage_uri), state_(std::move(state)) {}

protected:
    bool IsContextOk() const override { return true; }
    bool Reconnect() override { return true; }

    std::vector<ReplyUPtr> TryExecPipeline(const std::vector<CmdArgs> &cmds) override {
        int32_t active_count = active_pipeline_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (active_count > 1) {
            state_->concurrent_pipeline_count.fetch_add(1, std::memory_order_relaxed);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        std::vector<ReplyUPtr> replies;
        replies.reserve(cmds.size());
        for (const auto &cmd : cmds) {
            replies.emplace_back(MakeReply(cmd));
        }

        active_pipeline_count_.fetch_sub(1, std::memory_order_acq_rel);
        return replies;
    }

private:
    static ReplyUPtr MakeFakeReply(int type, const std::string &str) {
        redisReply *r = (redisReply *)malloc(sizeof(redisReply));
        memset(r, 0, sizeof(redisReply));
        r->type = type;
        if (type == REDIS_REPLY_STRING || type == REDIS_REPLY_STATUS || type == REDIS_REPLY_ERROR || !str.empty()) {
            r->str = strdup(str.c_str());
            r->len = str.size();
        }
        return ReplyUPtr(r, freeReplyObject);
    }

    static ReplyUPtr MakeFakeReplyInteger(const int64_t integer) {
        redisReply *r = (redisReply *)malloc(sizeof(redisReply));
        memset(r, 0, sizeof(redisReply));
        r->type = REDIS_REPLY_INTEGER;
        r->integer = integer;
        return ReplyUPtr(r, freeReplyObject);
    }

    ReplyUPtr MakeReply(const CmdArgs &cmd) {
        if (cmd.empty()) {
            return MakeFakeReply(REDIS_REPLY_ERROR, "empty command");
        }
        if (cmd[0] == "SCRIPT") {
            return MakeFakeReply(REDIS_REPLY_STRING, "fake_script_sha1");
        }
        if (cmd[0] == "EVALSHA" || cmd[0] == "EVAL") {
            return MakeFakeReplyInteger(1);
        }
        if (cmd[0] == "GET") {
            return MakeFakeReply(REDIS_REPLY_STRING, "node_info_json");
        }
        if (cmd[0] == "PTTL") {
            return MakeFakeReplyInteger(1000);
        }
        if (cmd[0] == "SET") {
            return MakeFakeReply(REDIS_REPLY_STATUS, "OK");
        }
        return MakeFakeReplyInteger(1);
    }

private:
    std::shared_ptr<RedisClientConcurrencyState> state_;
    std::atomic<int32_t> active_pipeline_count_{0};
};

class TestCoordinationRedisBackend : public CoordinationRedisBackend {
public:
    explicit TestCoordinationRedisBackend(std::shared_ptr<RedisClientConcurrencyState> state)
        : state_(std::move(state)) {}

protected:
    std::shared_ptr<RedisClientExt> CreateRedisClient(const StandardUri &storage_uri) const override {
        return std::make_shared<TrackingRedisClient>(storage_uri, state_);
    }

private:
    std::shared_ptr<RedisClientConcurrencyState> state_;
};

TEST(CoordinationRedisBackendUnitTest, PoolSerializesDifferentApisOnSameClient) {
    auto state = std::make_shared<RedisClientConcurrencyState>();
    TestCoordinationRedisBackend backend(state);
    StandardUri uri = StandardUri::FromUri("redis://localhost:6379?client_min_pool_size=1&client_max_pool_size=1");
    ASSERT_EQ(EC_OK, backend.Init(uri));

    std::atomic<bool> go{false};
    std::thread renew_thread([&]() {
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        EXPECT_EQ(EC_OK, backend.RenewLock("leader_lock", "node_1", 1000));
    });
    std::thread get_thread([&]() {
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::string value;
        EXPECT_EQ(EC_OK, backend.GetValue("_TAIR_KVCM_NODE_INFO_node_1", value));
        EXPECT_EQ("node_info_json", value);
    });

    go.store(true, std::memory_order_release);
    renew_thread.join();
    get_thread.join();

    EXPECT_EQ(0, state->concurrent_pipeline_count.load(std::memory_order_relaxed));
}

} // namespace
} // namespace kv_cache_manager
