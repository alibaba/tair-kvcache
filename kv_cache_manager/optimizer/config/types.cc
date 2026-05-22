#include "kv_cache_manager/optimizer/config/types.h"

namespace kv_cache_manager {
EvictionPolicyType ToEvictionPolicyType(const std::string &str) {
    if (str == "lru") {
        return EvictionPolicyType::POLICY_LRU;
    } else if (str == "random_lru") {
        return EvictionPolicyType::POLICY_RANDOM_LRU;
    } else if (str == "leaf_aware_lru") {
        return EvictionPolicyType::POLICY_LEAF_AWARE_LRU;
    } else if (str == "ttl") {
        return EvictionPolicyType::POLICY_TTL;
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
    case EvictionPolicyType::POLICY_TTL:
        return "ttl";
    default:
        return "unspecified";
    }
}

TierWriteMode ToTierWriteMode(const std::string &str) {
    if (str == "write_through") {
        return TierWriteMode::WRITE_THROUGH;
    } else if (str == "cascading") {
        return TierWriteMode::CASCADING;
    } else if (str == "write_through_selective") {
        return TierWriteMode::WRITE_THROUGH_SELECTIVE;
    }
    // 非法/缺省值回退到 WRITE_THROUGH，保证向后兼容
    return TierWriteMode::WRITE_THROUGH;
}

bool IsValidTierWriteMode(const std::string &str) {
    return str == "write_through" || str == "cascading" || str == "write_through_selective";
}

std::string ToString(const TierWriteMode &mode) {
    switch (mode) {
    case TierWriteMode::WRITE_THROUGH:
        return "write_through";
    case TierWriteMode::CASCADING:
        return "cascading";
    case TierWriteMode::WRITE_THROUGH_SELECTIVE:
        return "write_through_selective";
    default:
        return "write_through";
    }
}

} // namespace kv_cache_manager
