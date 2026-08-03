#include <atomic>
#include <future>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/redis/redis_cluster_executor.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class RedisClusterExecutorTest : public TESTBASE {
protected:
    static std::shared_ptr<RedisClusterExecutor> CreateExecutor(const std::string &uri_text) {
        StandardUri uri(uri_text);
        std::string error;
        std::shared_ptr<RedisClusterExecutor> executor = RedisClusterExecutor::GetOrCreate(uri, error);
        EXPECT_TRUE(error.empty()) << error;
        return executor;
    }

    static redisReply MakeErrorReply(std::string &message) {
        redisReply reply{};
        reply.type = REDIS_REPLY_ERROR;
        reply.str = message.data();
        reply.len = message.size();
        return reply;
    }
};

TEST_F(RedisClusterExecutorTest, TestCreateAndShareExecutor) {
    const std::string uri_text =
        "redis_cluster://test-user:test-token@127.0.0.1:6379/?db=0&cluster_pipeline_worker_count=2";
    std::shared_ptr<RedisClusterExecutor> first = CreateExecutor(uri_text);
    std::shared_ptr<RedisClusterExecutor> second = CreateExecutor(uri_text);

    ASSERT_NE(nullptr, first);
    ASSERT_EQ(first, second);

    std::shared_ptr<RedisClusterExecutor> different =
        CreateExecutor("redis_cluster://test-user:other-token@127.0.0.1:6379/?db=0&cluster_pipeline_worker_count=2");
    ASSERT_NE(nullptr, different);
    ASSERT_NE(first, different);
}

TEST_F(RedisClusterExecutorTest, TestRejectInvalidUri) {
    const std::vector<std::string> invalid_uris = {
        "redis://127.0.0.1:6379/?db=0",
        "redis_cluster://:6379/?db=0",
        "redis_cluster://127.0.0.1:6379/?db=1",
        "redis_cluster://127.0.0.1:6379/?connect_timeout_ms=0",
        "redis_cluster://test-user@127.0.0.1:6379/",
        "redis_cluster://:test-token@127.0.0.1:6379/",
        "redis_cluster://test-user:@127.0.0.1:6379/",
        "redis_cluster://127.0.0.1:6379/?tls=true",
        "redis_cluster://127.0.0.1:6379/?cluster_pipeline_worker_count=0",
    };

    for (const std::string &uri_text : invalid_uris) {
        StandardUri uri(uri_text);
        std::string error;
        std::shared_ptr<RedisClusterExecutor> executor = RedisClusterExecutor::GetOrCreate(uri, error);
        EXPECT_EQ(nullptr, executor) << uri_text;
        EXPECT_FALSE(error.empty()) << uri_text;
    }
}

TEST_F(RedisClusterExecutorTest, TestExecuteBatchBeforeOpen) {
    std::shared_ptr<RedisClusterExecutor> executor =
        CreateExecutor("redis_cluster://127.0.0.1:6379/?db=0&cluster_pipeline_worker_count=2");
    ASSERT_NE(nullptr, executor);
    ASSERT_FALSE(executor->IsReady());

    const std::vector<RedisCmdArgs> commands = {{"GET", "key"}};
    const std::vector<RedisReplyUPtr> replies = executor->ExecuteBatch(commands);
    ASSERT_TRUE(replies.empty());
}

TEST_F(RedisClusterExecutorTest, TestExtractRouteKey) {
    std::string route_key;
    std::string error;
    ASSERT_TRUE(RedisClusterExecutor::ExtractRouteKey({"HSET", "hash-key", "field", "value"}, route_key, error));
    ASSERT_EQ("hash-key", route_key);

    route_key.clear();
    error.clear();
    ASSERT_TRUE(RedisClusterExecutor::ExtractRouteKey({"EVAL", "return 1", "1", "script-key"}, route_key, error));
    ASSERT_EQ("script-key", route_key);

    route_key.clear();
    error.clear();
    ASSERT_FALSE(RedisClusterExecutor::ExtractRouteKey({"HSET"}, route_key, error));
    ASSERT_FALSE(error.empty());

    error.clear();
    ASSERT_FALSE(RedisClusterExecutor::ExtractRouteKey({"PING"}, route_key, error));
    ASSERT_FALSE(error.empty());
}

TEST_F(RedisClusterExecutorTest, TestRedirectReply) {
    std::string moved_message = "MOVED 1 127.0.0.1:6380";
    redisReply moved_reply = MakeErrorReply(moved_message);
    ASSERT_TRUE(RedisClusterExecutor::IsRedirectReply(&moved_reply));

    std::string ask_message = "ASK 1 127.0.0.1:6380";
    redisReply ask_reply = MakeErrorReply(ask_message);
    ASSERT_TRUE(RedisClusterExecutor::IsRedirectReply(&ask_reply));

    std::string ordinary_message = "ERR failed";
    redisReply ordinary_reply = MakeErrorReply(ordinary_message);
    ASSERT_FALSE(RedisClusterExecutor::IsRedirectReply(&ordinary_reply));
    ASSERT_FALSE(RedisClusterExecutor::IsRedirectReply(nullptr));
}

TEST_F(RedisClusterExecutorTest, TestClusterScanCursorRoundTrip) {
    std::map<std::string, RedisClusterExecutor::ClusterScanNodeState> states;
    states["node-a"].cursor = 12;
    states["node-b"].done = true;

    const std::string cursor = RedisClusterExecutor::EncodeClusterScanCursor(states);
    std::map<std::string, RedisClusterExecutor::ClusterScanNodeState> parsed;
    std::string error;
    ASSERT_TRUE(RedisClusterExecutor::ParseClusterScanCursor(cursor, parsed, error));
    ASSERT_EQ(2, parsed.size());
    ASSERT_EQ(12, parsed["node-a"].cursor);
    ASSERT_FALSE(parsed["node-a"].done);
    ASSERT_TRUE(parsed["node-b"].done);

    parsed.clear();
    error.clear();
    ASSERT_FALSE(RedisClusterExecutor::ParseClusterScanCursor("invalid", parsed, error));
    ASSERT_FALSE(error.empty());
}

TEST_F(RedisClusterExecutorTest, TestPipelineWorkers) {
    std::shared_ptr<RedisClusterExecutor> executor =
        CreateExecutor("redis_cluster://127.0.0.1:6379/?db=0&cluster_pipeline_worker_count=2");
    ASSERT_NE(nullptr, executor);
    ASSERT_TRUE(executor->StartPipelineWorkers());
    ASSERT_EQ(2, executor->pipeline_workers_.size());
    ASSERT_TRUE(executor->StartPipelineWorkers());
    ASSERT_EQ(2, executor->pipeline_workers_.size());

    std::atomic<bool> first_completed{false};
    std::atomic<bool> second_completed{false};
    std::packaged_task<void()> first_task(
        [&first_completed]() { first_completed.store(true, std::memory_order_relaxed); });
    std::future<void> first_future = first_task.get_future();
    ASSERT_TRUE(executor->SubmitPipelineTask(std::move(first_task)));

    std::packaged_task<void()> second_task([&second_completed]() {
        second_completed.store(true, std::memory_order_relaxed);
        throw std::runtime_error("task failed");
    });
    std::future<void> second_future = second_task.get_future();
    ASSERT_TRUE(executor->SubmitPipelineTask(std::move(second_task)));

    first_future.get();
    ASSERT_THROW(second_future.get(), std::runtime_error);
    ASSERT_TRUE(first_completed.load(std::memory_order_relaxed));
    ASSERT_TRUE(second_completed.load(std::memory_order_relaxed));

    executor->StopPipelineWorkers();
    ASSERT_TRUE(executor->pipeline_workers_.empty());

    std::packaged_task<void()> stopped_task([]() {});
    ASSERT_FALSE(executor->SubmitPipelineTask(std::move(stopped_task)));
}

} // namespace kv_cache_manager
