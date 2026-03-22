#include "kv_cache_manager/config/coordination_backend_factory.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/config/coordination_file_backend.h"
#include "kv_cache_manager/config/coordination_memory_backend.h"
#include "kv_cache_manager/config/coordination_redis_backend.h"

namespace kv_cache_manager {
static const std::string COORDINATION_REDIS_BACKEND_TYPE_STR = "redis";
static const std::string COORDINATION_FILE_BACKEND_TYPE_STR = "file";
static const std::string COORDINATION_MEMORY_BACKEND_TYPE_STR = "memory";

std::unique_ptr<CoordinationBackend>
CoordinationBackendFactory::CreateAndInitCoordinationBackend(const std::string &coordination_backend_uri) {
    auto standard_uri = StandardUri::FromUri(coordination_backend_uri);
    std::unique_ptr<CoordinationBackend> backend;
    if (standard_uri.GetProtocol() == COORDINATION_REDIS_BACKEND_TYPE_STR) {
        backend = std::make_unique<CoordinationRedisBackend>();
    } else if (standard_uri.GetProtocol() == COORDINATION_FILE_BACKEND_TYPE_STR) {
        backend = std::make_unique<CoordinationFileBackend>();
    } else if (standard_uri.GetProtocol() == COORDINATION_MEMORY_BACKEND_TYPE_STR) {
        backend = std::make_unique<CoordinationMemoryBackend>();
    } else if (coordination_backend_uri.empty()) {
        KVCM_LOG_WARN("coordination backend uri not configured, use coordination memory backend");
        backend = std::make_unique<CoordinationMemoryBackend>();
        standard_uri = StandardUri::FromUri("memory://");
    } else {
        KVCM_LOG_ERROR("create coordination backend failed, unknown backend type[%s]",
                       standard_uri.GetProtocol().c_str());
        return nullptr;
    }
    if (backend->Init(standard_uri) != EC_OK) {
        KVCM_LOG_ERROR("coordination backend init failed, type[%s], uri[%s]",
                       standard_uri.GetProtocol().c_str(),
                       coordination_backend_uri.c_str());
        return nullptr;
    }
    KVCM_LOG_INFO("coordination backend create and init success, type[%s], uri[%s]",
                  standard_uri.GetProtocol().c_str(),
                  coordination_backend_uri.c_str());
    return backend;
}
} // namespace kv_cache_manager
