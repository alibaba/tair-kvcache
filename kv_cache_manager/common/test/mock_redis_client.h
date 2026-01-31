#pragma once

#include <gmock/gmock.h>

#include "kv_cache_manager/common/redis_client.h"

namespace kv_cache_manager {
class MockRedisClient : public RedisClient {
public:
    MockRedisClient(const StandardUri &storage_uri) : RedisClient(storage_uri) {}
    ~MockRedisClient() = default;

public:
    MOCK_METHOD(bool, IsContextOk, (), (const));
    MOCK_METHOD(bool, Reconnect, ());
    MOCK_METHOD(std::vector<ReplyUPtr>, TryExecPipeline, (const std::vector<CmdArgs> &));
};
} // namespace kv_cache_manager
