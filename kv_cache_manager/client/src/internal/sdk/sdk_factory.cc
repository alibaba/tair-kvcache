#include "kv_cache_manager/client/src/internal/sdk/sdk_factory.h"

#ifdef ENABLE_HF3FS
#include "kv_cache_manager/client/src/internal/sdk/hf3fs_sdk.h"
#endif
#include "kv_cache_manager/client/src/internal/sdk/local_file_sdk.h"
#ifdef ENABLE_MOONCAKE
#include "kv_cache_manager/client/src/internal/sdk/mooncake_sdk.h"
#endif
#ifdef ENABLE_TAIR_MEMPOOL
#include "stub_source/kv_cache_manager/client/src/internal/sdk/tair_mempool_sdk.h"
#endif
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

SdkFactory *SdkFactory::GetInstance() {
    static SdkFactory instance;
    return &instance;
}

std::shared_ptr<SdkInterface> SdkFactory::CreateSdk(const DataStorageType &type,
                                                    const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                                    const std::shared_ptr<StorageConfig> &storage_config) {
    if (!sdk_backend_config || !storage_config) {
        KVCM_LOG_WARN("sdk_backend_config or storage_config is null");
        return nullptr;
    }
    std::shared_ptr<SdkInterface> sdk;
    switch (type) {
#ifdef ENABLE_HF3FS
    case DataStorageType::DATA_STORAGE_TYPE_HF3FS:
        sdk = std::make_shared<Hf3fsSdk>();
        break;
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS:
        sdk = std::make_shared<Hf3fsSdk>();
        break;
#endif
#ifdef ENABLE_MOONCAKE
    case DataStorageType::DATA_STORAGE_TYPE_MOONCAKE:
        sdk = std::make_shared<MooncakeSdk>();
        break;
#endif
#ifdef ENABLE_TAIR_MEMPOOL
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL:
        sdk = std::make_shared<TairMempoolSdk>();
        break;
#endif
    case DataStorageType::DATA_STORAGE_TYPE_NFS:
        sdk = std::make_shared<LocalFileSdk>();
        break;
    default:
        KVCM_LOG_WARN("unsupported sdk type: %s", ToString(type).c_str());
        return nullptr;
    }
    if (!sdk) {
        KVCM_LOG_WARN("create sdk failed, sdk is null, type:%s", ToString(type).c_str());
        return nullptr;
    }
    auto ec = sdk->Init(sdk_backend_config, storage_config);
    if (ec != ER_OK) {
        KVCM_LOG_WARN("init sdk failed, type:%s, sdk backend config: %s, storage config: %s, errorcode: %d",
                      ToString(type).c_str(),
                      sdk_backend_config->ToJsonString().c_str(),
                      storage_config->ToString().c_str(),
                      ec);
        return nullptr;
    }
    return sdk;
}

} // namespace kv_cache_manager