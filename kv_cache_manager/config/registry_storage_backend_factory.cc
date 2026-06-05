#include "kv_cache_manager/config/registry_storage_backend_factory.h"

#include <map>
#include <mutex>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/config/registry_local_backend.h"
#include "kv_cache_manager/config/registry_redis_backend.h"

namespace kv_cache_manager {

namespace {

std::mutex &RegistryMutex() {
    static std::mutex mu;
    return mu;
}

std::map<std::string, RegistryStorageBackendFactory::CreatorFunc> &RegistryMap() {
    static std::map<std::string, RegistryStorageBackendFactory::CreatorFunc> m;
    return m;
}

} // namespace

void RegistryStorageBackendFactory::RegisterType(const std::string &protocol, CreatorFunc creator) {
    std::lock_guard<std::mutex> g(RegistryMutex());
    RegistryMap()[protocol] = std::move(creator);
}

std::unique_ptr<RegistryStorageBackend>
RegistryStorageBackendFactory::CreateAndInitStorageBackend(const std::string &registry_storage_uri) {
    auto standard_uri = StandardUri::FromUri(registry_storage_uri);
    std::unique_ptr<RegistryStorageBackend> storage_backend;
    const std::string &protocol = standard_uri.GetProtocol();

    if (protocol == "redis") {
        storage_backend = std::make_unique<RegistryRedisBackend>();
    } else if (protocol == "local") {
        storage_backend = std::make_unique<RegistryLocalBackend>();
    } else if (registry_storage_uri.empty()) {
        KVCM_LOG_WARN("registry storage uri not configured, use registry local backend");
        storage_backend = std::make_unique<RegistryLocalBackend>();
    } else {
        std::lock_guard<std::mutex> g(RegistryMutex());
        auto it = RegistryMap().find(protocol);
        if (it != RegistryMap().end()) {
            storage_backend = it->second();
        } else {
            KVCM_LOG_ERROR("create registry storage backend fail, unknown registry storage type[%s]",
                           protocol.c_str());
            return nullptr;
        }
    }

    if (storage_backend->Init(standard_uri) != EC_OK) {
        KVCM_LOG_ERROR("registry storage backend init failed, type[%s], uri[%s]",
                       protocol.c_str(),
                       registry_storage_uri.c_str());
        return nullptr;
    }
    KVCM_LOG_INFO("registry storage backend create and init success, type[%s], uri[%s]",
                  protocol.c_str(),
                  registry_storage_uri.c_str());
    return storage_backend;
}

} // namespace kv_cache_manager
