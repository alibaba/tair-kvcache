#pragma once

#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

class SdkFactory {
public:
    static SdkFactory *GetInstance();
    static std::shared_ptr<SdkInterface> CreateSdk(const DataStorageType &type,
                                                   const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                                   const std::shared_ptr<StorageConfig> &storage_config);
};

} // namespace kv_cache_manager