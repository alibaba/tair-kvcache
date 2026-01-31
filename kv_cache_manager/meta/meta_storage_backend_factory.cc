#include "kv_cache_manager/meta/meta_storage_backend_factory.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_redis_backend.h"

namespace kv_cache_manager {

std::unique_ptr<MetaStorageBackend>
MetaStorageBackendFactory::CreateAndInitStorageBackend(const std::string &instance_id,
                                                       const std::shared_ptr<MetaStorageBackendConfig> &config) {
    std::unique_ptr<MetaStorageBackend> storage_backend;
    if (config->GetStorageType() == META_REDIS_BACKEND_TYPE_STR) {
        storage_backend = std::make_unique<MetaRedisBackend>();
    } else if (config->GetStorageType() == META_LOCAL_BACKEND_TYPE_STR) {
        storage_backend = std::make_unique<MetaLocalBackend>();
    } else {
        KVCM_LOG_ERROR("meta storage backend create failed, unknown meta storage type[%s]",
                       config->GetStorageType().c_str());
        return nullptr;
    }
    if (storage_backend->Init(instance_id, config) != EC_OK) {
        KVCM_LOG_ERROR("meta storage backend init failed, type[%s]", config->GetStorageType().c_str());
        return nullptr;
    }
    return storage_backend;
}

} // namespace kv_cache_manager
