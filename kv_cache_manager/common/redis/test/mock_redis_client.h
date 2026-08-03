#pragma once

#include <gmock/gmock.h>
#include <memory>
#include <utility>

#include "kv_cache_manager/common/redis/redis_client.h"

namespace kv_cache_manager {

class MockRedisExecutor : public IRedisExecutor {
public:
    MOCK_METHOD(bool, Open, (), (override));
    MOCK_METHOD(bool, IsReady, (), (const, noexcept, override));
    MOCK_METHOD(std::vector<RedisReplyUPtr>, ExecuteBatch, (const std::vector<RedisCmdArgs> &), (override));
};

class MockRedisClient : public RedisClient {
public:
    explicit MockRedisClient(const StandardUri &storage_uri)
        : MockRedisClient(storage_uri, std::make_shared<::testing::NiceMock<MockRedisExecutor>>()) {}

    MockRedisExecutor &Executor() { return *executor_; }

private:
    MockRedisClient(const StandardUri &storage_uri, std::shared_ptr<MockRedisExecutor> executor)
        : RedisClient(storage_uri, executor), executor_(std::move(executor)) {}

    std::shared_ptr<MockRedisExecutor> executor_;
};

} // namespace kv_cache_manager
