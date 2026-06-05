#pragma once

#include <memory>
#include <vector>

#include "kv_cache_manager/client/src/internal/sdk/caller_node_provider.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

// Builds a CallerNodeProvider for the meta client based on the storage
// configs returned by RegisterInstance.
//
// Selection rule (v1):
//   - Walk the storage_configs in declared order.
//   - For each entry, dispatch on `StorageConfig::type()`. If a concrete
//     provider is compiled in for that type and successfully initialises
//     (i.e. the local backend is actually reachable on this machine), use it.
//   - Otherwise fall back to NoopCallerNodeProvider.
//
// v1 only knows how to produce a real provider for TAIR_MEMPOOL — that is
// the only backend whose SDK can introspect the local pace-server's node id.
// All other types (HF3FS / Mooncake / NFS / Dummy / Vineyard) get the noop
// provider, exactly matching the pre-affinity behaviour.
class CallerNodeProviderFactory {
public:
    // Returns a non-null CallerNodeProvider. Falls back to
    // NoopCallerNodeProvider if no concrete provider is applicable.
    static std::unique_ptr<CallerNodeProvider>
    Create(const std::vector<std::shared_ptr<StorageConfig>> &storage_configs);
};

} // namespace kv_cache_manager
