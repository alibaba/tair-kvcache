#include "kv_cache_manager/client/src/internal/sdk/caller_node_provider_factory.h"

#include "kv_cache_manager/client/src/internal/sdk/nfs_caller_node_provider.h"

#ifdef ENABLE_TAIR_MEMPOOL
#include "stub_source/kv_cache_manager/client/src/internal/sdk/tair_mempool_caller_node_provider.h"
#endif

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

std::unique_ptr<CallerNodeProvider>
CallerNodeProviderFactory::Create(const std::vector<std::shared_ptr<StorageConfig>> &storage_configs,
                                  std::chrono::seconds refresh_interval) {
    if (refresh_interval.count() <= 0) {
        refresh_interval = std::chrono::seconds(30);
    }
    for (const auto &storage_config : storage_configs) {
        if (!storage_config) {
            continue;
        }
        const DataStorageType type = storage_config->type();
        switch (type) {
#ifdef ENABLE_TAIR_MEMPOOL
        case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL: {
            auto provider = std::make_unique<TairMempoolCallerNodeProvider>(refresh_interval);
            if (provider->Init(storage_config) == ER_OK) {
                KVCM_LOG_INFO("caller_node_provider: using TairMempoolCallerNodeProvider for storage [%s]",
                              storage_config->global_unique_name().c_str());
                return provider;
            }
            KVCM_LOG_INFO("caller_node_provider: TairMempoolCallerNodeProvider init failed for storage [%s], skipping",
                          storage_config->global_unique_name().c_str());
            break;
        }
#endif
        case DataStorageType::DATA_STORAGE_TYPE_NFS: {
            auto provider = std::make_unique<NfsCallerNodeProvider>();
            KVCM_LOG_INFO("caller_node_provider: using NfsCallerNodeProvider for storage [%s]",
                          storage_config->global_unique_name().c_str());
            return provider;
        }
        default:
            // No concrete provider for this storage type in this build —
            // fall through and try the next storage config (if any).
            break;
        }
    }
    KVCM_LOG_INFO("caller_node_provider: falling back to NoopCallerNodeProvider");
    return std::make_unique<NoopCallerNodeProvider>();
}

} // namespace kv_cache_manager
