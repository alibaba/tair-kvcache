#include "kv_cache_manager/config/registry_storage_backend_factory.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/config/registry_local_backend.h"
#include "kv_cache_manager/config/registry_redis_backend.h"

namespace kv_cache_manager {
static const std::string REGISTRY_REDIS_BACKEND_TYPE_STR = "redis";
static const std::string REGISTRY_LOCAL_BACKEND_TYPE_STR = "local";

std::unique_ptr<RegistryStorageBackend>
RegistryStorageBackendFactory::CreateAndInitStorageBackend(const std::string &registry_storage_uri) {
    auto standard_uri = StandardUri::FromUri(registry_storage_uri);
    std::unique_ptr<RegistryStorageBackend> storage_backend;
    if (standard_uri.GetProtocol() == REGISTRY_REDIS_BACKEND_TYPE_STR) {
        storage_backend = std::make_unique<RegistryRedisBackend>();
    } else if (standard_uri.GetProtocol() == REGISTRY_LOCAL_BACKEND_TYPE_STR) {
        storage_backend = std::make_unique<RegistryLocalBackend>();
    } else if (registry_storage_uri.empty()) {
        KVCM_LOG_WARN("registry storage uri not configured, use registry local backend");
        storage_backend = std::make_unique<RegistryLocalBackend>();
    } else {
        KVCM_LOG_ERROR("create registry storage backedn fail, unknown registry storage type[%s]",
                       standard_uri.GetProtocol().c_str());
        return nullptr;
    }
    if (storage_backend->Init(standard_uri) != EC_OK) {
        KVCM_LOG_ERROR("registry storage backend init failed, type[%s], uri[%s]",
                       standard_uri.GetProtocol().c_str(),
                       registry_storage_uri.c_str());
        return nullptr;
    }
    KVCM_LOG_INFO("registry storage backend create and init success, type[%s], uri[%s]",
                  standard_uri.GetProtocol().c_str(),
                  registry_storage_uri.c_str());
    return storage_backend;
}
} // namespace kv_cache_manager
