#include "kv_cache_manager/optimizer/config/types.h"

namespace kv_cache_manager {
EvictionPolicyType ToEvictionPolicyType(const std::string &str) {
    if (str == "lru") {
        return EvictionPolicyType::POLICY_LRU;
    } else if (str == "random_lru") {
        return EvictionPolicyType::POLICY_RANDOM_LRU;
    } else if (str == "leaf_aware_lru") {
        return EvictionPolicyType::POLICY_LEAF_AWARE_LRU;
    } else {
        return EvictionPolicyType::POLICY_UNSPECIFIED;
    }
}
std::string ToString(const EvictionPolicyType &type) {
    switch (type) {
    case EvictionPolicyType::POLICY_LRU:
        return "lru";
    case EvictionPolicyType::POLICY_RANDOM_LRU:
        return "random_lru";
    case EvictionPolicyType::POLICY_LEAF_AWARE_LRU:
        return "leaf_aware_lru";
    default:
        return "unspecified";
    }
}

} // namespace kv_cache_manager