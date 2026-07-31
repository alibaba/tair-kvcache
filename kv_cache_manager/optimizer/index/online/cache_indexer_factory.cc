#include "kv_cache_manager/optimizer/index/online/cache_indexer_factory.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/index/online/lru_cache_indexer.h"
#include "kv_cache_manager/optimizer/index/online/ttl_cache_indexer_wrapper.h"

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
        } catch (...) { KVCM_LOG_WARN("Invalid token in KVCM_HIT_AGE_BUCKET_THRESHOLDS: [%s]", token.c_str()); }
    }
    if (!thresholds.empty()) {
        std::sort(thresholds.begin(), thresholds.end());
        wrapper->SetHitAgeBucketThresholds(thresholds);
        KVCM_LOG_INFO("Applied custom hit age bucket thresholds from env, count=%zu", thresholds.size());
    }
}

std::unique_ptr<CacheIndexer> CacheIndexerFactory::CreateCacheIndexer(const std::string &eviction_policy,
                                                                      bool enable_theoretical_max_cache,
                                                                      const std::vector<double> &capacity_gb,
                                                                      int64_t size_full_only,
                                                                      int64_t size_full_linear,
                                                                      int32_t linear_step,
                                                                      int64_t ttl_seconds) {
    if (eviction_policy != "lru") {
        KVCM_LOG_ERROR("CreateCacheIndexer: unsupported eviction_policy[%s]", eviction_policy.c_str());
        return nullptr;
    }
    if (ttl_seconds < 0) {
        KVCM_LOG_ERROR("CreateCacheIndexer: ttl_seconds must be non-negative[%ld]", ttl_seconds);
        return nullptr;
    }

    std::unique_ptr<CacheIndexer> indexer = std::make_unique<LruCacheIndexer>(enable_theoretical_max_cache);
    indexer->Init(capacity_gb, size_full_only, size_full_linear, linear_step);

    if (ttl_seconds > 0) {
        auto ttl_wrapper = std::make_unique<TtlCacheIndexerWrapper>(std::move(indexer), ttl_seconds);
        ApplyHitAgeBucketThresholdsFromEnv(ttl_wrapper.get());
        return ttl_wrapper;
    }

    return indexer;
}

} // namespace kv_cache_manager
