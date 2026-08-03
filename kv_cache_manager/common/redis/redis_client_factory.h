#pragma once

#include <memory>

#include "kv_cache_manager/common/redis/redis_client.h"

namespace kv_cache_manager {

class RedisClientFactory {
public:
    static std::unique_ptr<RedisClient> Create(const StandardUri &uri);
};

} // namespace kv_cache_manager
