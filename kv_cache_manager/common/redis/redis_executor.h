#pragma once

#include <memory>
#include <string>
#include <sw/redis++/reply.h>
#include <vector>

namespace kv_cache_manager {

using RedisReplyUPtr = sw::redis::ReplyUPtr;
using RedisCmdArgs = std::vector<std::string>;

class IRedisExecutor {
public:
    virtual ~IRedisExecutor() = default;

    virtual bool Open() = 0;
    virtual bool IsReady() const noexcept = 0;
    virtual std::vector<RedisReplyUPtr> ExecuteBatch(const std::vector<RedisCmdArgs> &cmds) = 0;
};

} // namespace kv_cache_manager
