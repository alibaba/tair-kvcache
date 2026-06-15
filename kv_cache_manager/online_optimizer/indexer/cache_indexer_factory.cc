#include "kv_cache_manager/online_optimizer/indexer/cache_indexer_factory.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/online_optimizer/indexer/bst_cache_indexer.h"
#include "kv_cache_manager/online_optimizer/indexer/fenwick_cache_indexer.h"
#include "kv_cache_manager/online_optimizer/indexer/lru_cache_indexer.h"
#include "kv_cache_manager/online_optimizer/indexer/ttl_cache_indexer_wrapper.h"

namespace kv_cache_manager {

static void ApplyHitAgeBucketThresholdsFromEnv(TtlCacheIndexerWrapper *wrapper) {
    std::string env_value = EnvUtil::GetEnv("KVCM_HIT_AGE_BUCKET_THRESHOLDS", std::string(""));
    if (env_value.empty()) {
        return;
    }
    std::vector<int64_t> thresholds;
    std::istringstream stream(env_value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        try {
            int64_t value = std::stoll(token);
            if (value > 0) {
                thresholds.push_back(value);
            }
        } catch (...) {
            KVCM_LOG_WARN("Invalid token in KVCM_HIT_AGE_BUCKET_THRESHOLDS: [%s]", token.c_str());
        }
    }
    if (!thresholds.empty()) {
        wrapper->SetHitAgeBucketThresholds(thresholds);
        KVCM_LOG_INFO("Applied custom hit age bucket thresholds from env, count=%zu", thresholds.size());
    }
}

std::unique_ptr<CacheIndexer> CacheIndexerFactory::CreateCacheIndexer(
    const std::string &indexer_type,
    int64_t max_key_count,
    const std::vector<double> &capacity_gb,
    int64_t size_full_only,
    int64_t size_full_linear,
    int32_t linear_step,
    int64_t ttl_seconds) {
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

    if (ttl_seconds > 0) {
        auto ttl_wrapper = std::make_unique<TtlCacheIndexerWrapper>(
            std::move(indexer), ttl_seconds);
        ApplyHitAgeBucketThresholdsFromEnv(ttl_wrapper.get());
        return ttl_wrapper;
    }

    return indexer;
}

} // namespace kv_cache_manager
