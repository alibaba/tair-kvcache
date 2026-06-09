#include "kv_cache_manager/meta/reclaim_indexer/reclaim_indexer_factory.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/meta/reclaim_indexer/node_lru_reclaim_indexer.h"

namespace kv_cache_manager {

std::unique_ptr<ReclaimIndexer> ReclaimIndexerFactory::Create(const std::string &type) {
    if (type == "node_lru") {
        return std::make_unique<NodeLruReclaimIndexer>();
    }
    KVCM_LOG_ERROR("unknown reclaim indexer type: %s", type.c_str());
    return nullptr;
}

} // namespace kv_cache_manager
