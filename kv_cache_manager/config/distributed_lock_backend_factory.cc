#include "kv_cache_manager/config/distributed_lock_backend_factory.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/config/distributed_lock_file_backend.h"
#include "kv_cache_manager/config/distributed_lock_memory_backend.h"
#include "kv_cache_manager/config/distributed_lock_redis_backend.h"

namespace kv_cache_manager {
static const std::string LOCK_REDIS_BACKEND_TYPE_STR = "redis";
static const std::string LOCK_FILE_BACKEND_TYPE_STR = "file";
static const std::string LOCK_MEMORY_BACKEND_TYPE_STR = "memory";

std::unique_ptr<DistributedLockBackend>
DistributedLockBackendFactory::CreateAndInitDistributedLockBackend(const std::string &lock_backend_uri) {
    auto standard_uri = StandardUri::FromUri(lock_backend_uri);
    std::unique_ptr<DistributedLockBackend> storage_backend;
    if (standard_uri.GetProtocol() == LOCK_REDIS_BACKEND_TYPE_STR) {
        storage_backend = std::make_unique<DistributedLockRedisBackend>();
    } else if (standard_uri.GetProtocol() == LOCK_FILE_BACKEND_TYPE_STR) {
        storage_backend = std::make_unique<DistributedLockFileBackend>();
    } else if (standard_uri.GetProtocol() == LOCK_MEMORY_BACKEND_TYPE_STR) {
        storage_backend = std::make_unique<DistributedLockMemoryBackend>();
    } else if (lock_backend_uri.empty()) {
        KVCM_LOG_WARN("distributed lock uri not configured, use distributed lock memory backend");
        storage_backend = std::make_unique<DistributedLockMemoryBackend>();
        standard_uri = StandardUri::FromUri("memory://");
    } else {
        KVCM_LOG_ERROR("create distributed lock failed, unknown registry storage type[%s]",
                       standard_uri.GetProtocol().c_str());
        return nullptr;
    }
    if (storage_backend->Init(standard_uri) != EC_OK) {
        KVCM_LOG_ERROR("distributed lock backend init failed, type[%s], uri[%s]",
                       standard_uri.GetProtocol().c_str(),
                       lock_backend_uri.c_str());
        return nullptr;
    }
    KVCM_LOG_INFO("distributed lock backend create and init success, type[%s], uri[%s]",
                  standard_uri.GetProtocol().c_str(),
                  lock_backend_uri.c_str());
    return storage_backend;
}
} // namespace kv_cache_manager
