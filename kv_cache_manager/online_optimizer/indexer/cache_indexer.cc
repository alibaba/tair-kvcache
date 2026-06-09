#include "kv_cache_manager/online_optimizer/indexer/cache_indexer.h"

#include <algorithm>
#include <cmath>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/online_optimizer/indexer/bst_cache_indexer.h"
#include "kv_cache_manager/online_optimizer/indexer/fenwick_cache_indexer.h"
#include "kv_cache_manager/online_optimizer/indexer/lru_cache_indexer.h"

namespace kv_cache_manager {

std::unique_ptr<CacheIndexer> CreateCacheIndexer(const std::string &indexer_type,
                                                  int64_t max_key_count,
                                                  const std::vector<double> &capacity_gb,
                                                  int64_t size_full_only,
                                                  int64_t size_full_linear,
                                                  int32_t linear_step) {
    if (max_key_count > 0) {
        linear_step = std::max(linear_step, int32_t(1));
        int64_t avg = (linear_step <= 1) ? size_full_linear
            : ((linear_step - 1) * size_full_only + size_full_linear) / linear_step;
        if (avg > 0) {
            for (double cap : capacity_gb) {
                int64_t blocks = static_cast<int64_t>(cap * 1024.0 * 1024.0 * 1024.0) / avg;
                max_key_count = std::max(max_key_count, blocks);
            }
        }
    }

    std::unique_ptr<CacheIndexer> indexer;
    if (indexer_type == "bst_lru") {
        indexer = std::make_unique<BSTCacheIndexer>(max_key_count);
    } else if (indexer_type == "fenwick_lru") {
        indexer = std::make_unique<FenwickCacheIndexer>(max_key_count);
    } else if (indexer_type == "lru") {
        indexer = std::make_unique<LruCacheIndexer>(max_key_count);
    } else {
        KVCM_LOG_ERROR("CreateCacheIndexer: unknown indexer_type[%s]", indexer_type.c_str());
        return nullptr;
    }
    indexer->Init(capacity_gb, size_full_only, size_full_linear, linear_step);
    return indexer;
}

} // namespace kv_cache_manager
