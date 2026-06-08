#include "stub_source/kv_cache_manager/client/src/internal/sdk/tair_mempool_caller_node_provider.h"

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

ClientErrorCode TairMempoolCallerNodeProvider::Init(const std::shared_ptr<StorageConfig> & /*storage_config*/) {
    KVCM_LOG_INFO("TairMempoolCallerNodeProvider::Init: stub (no pace-mp in open-source build)");
    return ER_SDKINIT_ERROR;
}

} // namespace kv_cache_manager
