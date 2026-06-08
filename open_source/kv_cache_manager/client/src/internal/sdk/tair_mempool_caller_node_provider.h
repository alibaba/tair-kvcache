#pragma once

#include <memory>
#include <string>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/sdk/caller_node_provider.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

// Open-source stub: always returns empty CallerNode (no pace-mp available).
class TairMempoolCallerNodeProvider : public CallerNodeProvider {
public:
    TairMempoolCallerNodeProvider() = default;
    ~TairMempoolCallerNodeProvider() override = default;

    ClientErrorCode Init(const std::shared_ptr<StorageConfig> &storage_config);

    CallerNode GetCallerNode() const override { return cached_caller_node_; }

private:
    CallerNode cached_caller_node_;
};

} // namespace kv_cache_manager
