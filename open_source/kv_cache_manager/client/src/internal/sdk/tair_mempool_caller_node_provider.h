#pragma once

#include <chrono>
#include <memory>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/sdk/caller_node_provider.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

class TairMempoolCallerNodeProvider : public CallerNodeProvider {
public:
    explicit TairMempoolCallerNodeProvider(std::chrono::seconds = std::chrono::seconds(30)) {}
    ~TairMempoolCallerNodeProvider() override = default;

    ClientErrorCode Init(const std::shared_ptr<StorageConfig> &) { return ER_SDKINIT_ERROR; }

    ClientCallerNode GetCallerNode() override { return {}; }
};

} // namespace kv_cache_manager
