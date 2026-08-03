#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kv_cache_manager/common/redis/redis_executor.h"
#include "kv_cache_manager/common/standard_uri.h"

namespace kv_cache_manager {

class HiredisExecutor : public IRedisExecutor {
public:
    explicit HiredisExecutor(const StandardUri &storage_uri);
    ~HiredisExecutor() override;

    HiredisExecutor(const HiredisExecutor &) = delete;
    HiredisExecutor &operator=(const HiredisExecutor &) = delete;

    bool Open() override;
    bool IsReady() const noexcept override;
    std::vector<RedisReplyUPtr> ExecuteBatch(const std::vector<RedisCmdArgs> &cmds) override;

private:
    bool Connect();
    bool Reconnect();
    void Disconnect();
    std::vector<RedisReplyUPtr> TryExecuteBatch(const std::vector<RedisCmdArgs> &cmds);

    redisContext *context_ = nullptr;
    std::string user_info_;
    std::string host_;
    int64_t port_ = 0;
    int64_t db_ = 0;
    int64_t timeout_ms_ = 2000;
    int64_t retry_count_ = 2;
};

} // namespace kv_cache_manager
