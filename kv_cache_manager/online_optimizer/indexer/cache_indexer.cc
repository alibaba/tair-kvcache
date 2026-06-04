#include "kv_cache_manager/online_optimizer/indexer/cache_indexer.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/online_optimizer/indexer/bst_cache_indexer.h"
#include "kv_cache_manager/online_optimizer/indexer/fenwick_cache_indexer.h"

namespace kv_cache_manager {

std::unique_ptr<CacheIndexer> CreateCacheIndexer(const std::string &indexer_type, int64_t max_key_count) {
    if (indexer_type == "bst_lru") {
        return std::make_unique<BSTCacheIndexer>(max_key_count);
    }
    if (indexer_type == "fenwick_lru") {
        return std::make_unique<FenwickCacheIndexer>(max_key_count);
    }
    KVCM_LOG_ERROR("CreateCacheIndexer: unknown indexer_type[%s]", indexer_type.c_str());
    return nullptr;
}

} // namespace kv_cache_manager
