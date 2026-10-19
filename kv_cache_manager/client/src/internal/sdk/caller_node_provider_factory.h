#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include "kv_cache_manager/client/src/internal/sdk/caller_node_provider.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

// Builds a CallerNodeProvider for the meta client based on the storage
// configs returned by RegisterInstance.
//
// Selection rule:
//   - Walk the storage_configs in declared order.
//   - For each entry, dispatch on `StorageConfig::type()`. If a concrete
//     provider is compiled in for that type and accepts the configuration,
//     use it. Initialization does not probe provider health or identity.
//   - Otherwise fall back to NoopCallerNodeProvider.
//
// TAIR_MEMPOOL and TAIR_MEMPOOL_SSD share PACE's numeric node identity;
// NFS uses the local IP. Other storage types have no concrete provider.
class CallerNodeProviderFactory {
public:
    // Returns a non-null CallerNodeProvider. Falls back to
    // NoopCallerNodeProvider if no concrete provider is applicable.
    static std::unique_ptr<CallerNodeProvider>
    Create(const std::vector<std::shared_ptr<StorageConfig>> &storage_configs,
           std::chrono::seconds refresh_interval = std::chrono::seconds(0));
};

} // namespace kv_cache_manager
