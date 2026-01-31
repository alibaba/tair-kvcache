#include "kv_cache_manager/optimizer/eviction_policy/policy_factory.h"

#include <variant>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
std::shared_ptr<EvictionPolicy> EvictionPolicyFactory::CreatePolicy(EvictionPolicyType type,
                                                                    const std::string &name,
                                                                    const int32_t eviction_batch_size_per_instance,
                                                                    EvictionPolicyParam param) {
    KVCM_LOG_INFO("Creating eviction policy for tier %s, type=%s", name.c_str(), ToString(type).c_str());
    switch (type) {
    case EvictionPolicyType::POLICY_LRU: {
        if (!std::holds_alternative<LruParams>(param)) {
            KVCM_LOG_ERROR("Invalid parameters for LRU eviction policy on tier %s", name.c_str());
            return nullptr;
        }
        const LruParams &lru_params = std::get<LruParams>(param);
        KVCM_LOG_INFO("Creating LRU policy with sample_rate=%f", lru_params.sample_rate);
        return std::make_shared<LruEvictionPolicy>(name, lru_params);
    }
    case EvictionPolicyType::POLICY_RANDOM_LRU: {
        if (!std::holds_alternative<RandomLruParams>(param)) {
            KVCM_LOG_ERROR("Invalid parameters for Random LRU eviction policy on tier %s", name.c_str());
            return nullptr;
        }
        const RandomLruParams &random_lru_params = std::get<RandomLruParams>(param);
        KVCM_LOG_INFO("Creating Random LRU policy with sample_rate=%f", random_lru_params.sample_rate);
        return std::make_shared<RandomLruEvictionPolicy>(name, random_lru_params, eviction_batch_size_per_instance);
    }
    case EvictionPolicyType::POLICY_LEAF_AWARE_LRU: {
        if (!std::holds_alternative<LruParams>(param)) {
            KVCM_LOG_ERROR("Invalid parameters for Leaf Aware LRU eviction policy on tier %s", name.c_str());
            return nullptr;
        }
        const LruParams &lru_params = std::get<LruParams>(param);
        KVCM_LOG_INFO("Creating Leaf Aware LRU policy with sample_rate=%f", lru_params.sample_rate);
        return std::make_shared<LeafAwareLruEvictionPolicy>(name, lru_params);
    }
    case EvictionPolicyType::POLICY_UNSPECIFIED:
    default:
        KVCM_LOG_ERROR("Unsupported eviction policy type for tier %s", name.c_str());
        return nullptr;
    }
}

} // namespace kv_cache_manager