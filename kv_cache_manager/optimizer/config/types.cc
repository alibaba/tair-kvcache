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

TraceType ToTraceType(const std::string &str) {
    if (str == "publisher_log") {
        return TraceType::TRACE_PUBLISHER_LOG;
    } else if (str == "qwen_bailian") {
        return TraceType::TRACE_QWEN_BAILIAN;
    } else if (str == "optimizer_schema") {
        return TraceType::TRACE_OPTIMIZER_SCHEMA;
    } else {
        return TraceType::TRACE_UNSPECIFIED;
    }
}
std::string ToString(const TraceType &type) {
    switch (type) {
    case TraceType::TRACE_PUBLISHER_LOG:
        return "publisher_log";
    case TraceType::TRACE_QWEN_BAILIAN:
        return "qwen_bailian";
    case TraceType::TRACE_OPTIMIZER_SCHEMA:
        return "optimizer_schema";
    default:
        return "unspecified";
    }
}
} // namespace kv_cache_manager