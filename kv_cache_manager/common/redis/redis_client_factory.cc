#include "kv_cache_manager/common/redis/redis_client_factory.h"

#include <string>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/redis/hiredis_executor.h"
#include "kv_cache_manager/common/redis/redis_cluster_executor.h"

namespace kv_cache_manager {

std::unique_ptr<RedisClient> RedisClientFactory::Create(const StandardUri &uri) {
    std::shared_ptr<IRedisExecutor> executor;
    if (uri.GetProtocol() == "redis") {
        executor = std::make_shared<HiredisExecutor>(uri);
    } else if (uri.GetProtocol() == "redis_cluster") {
        std::string error;
        std::shared_ptr<RedisClusterExecutor> cluster_executor = RedisClusterExecutor::GetOrCreate(uri, error);
        if (!cluster_executor) {
            KVCM_LOG_ERROR("create redis cluster client failed: %s", error.c_str());
            return nullptr;
        }
        executor = std::move(cluster_executor);
    } else {
        KVCM_LOG_ERROR("unsupported redis uri protocol[%s]", uri.GetProtocol().c_str());
        return nullptr;
    }
    return std::make_unique<RedisClient>(uri, std::move(executor));
}

} // namespace kv_cache_manager
