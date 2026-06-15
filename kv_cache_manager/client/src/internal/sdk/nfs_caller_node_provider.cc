#include "kv_cache_manager/client/src/internal/sdk/nfs_caller_node_provider.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/net_util.h"

namespace kv_cache_manager {

NfsCallerNodeProvider::NfsCallerNodeProvider() : caller_{NetUtil::GetLocalIp(), ""} {
    KVCM_LOG_INFO("NfsCallerNodeProvider caller_node node_id [%s]", caller_.node_id.c_str());
}

} // namespace kv_cache_manager
