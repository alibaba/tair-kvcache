#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/pace_reclaim_free.h"

namespace kv_cache_manager {

class DataStorageBackend;
class DataStorageManager;
class StorageConfig;

bool ParsePaceReclaimGA(const DataStorageUri &uri, uint64_t &ga);

PaceReclaimFreeResult FreePaceGAsForReclaim(const DataStorageManager &storage,
                                            const std::string &trace_id,
                                            const std::string &name,
                                            const std::vector<uint64_t> &gas,
                                            std::weak_ptr<DataStorageBackend> &original,
                                            bool retry,
                                            const std::function<bool()> &can_send);

// Identifies the target Meta independently of client-side discovery timeouts.
std::string GetPaceClusterEndpoint(const StorageConfig &config);

// Reads the backend's last health-poll result without issuing a request.
PaceProviderCount GetPaceProviderCount(const std::weak_ptr<DataStorageBackend> &source);

} // namespace kv_cache_manager
