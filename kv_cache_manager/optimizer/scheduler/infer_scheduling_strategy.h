#pragma once

#include <string>

namespace kv_cache_manager {

struct InferSchedulingStrategy {
    static constexpr const char *kPreserveTrace = "preserve_trace";
    static constexpr const char *kRoundRobin = "round_robin";
    static constexpr const char *kPrefixHit = "prefix_hit";
};

[[nodiscard]] bool IsSupportedInferSchedulingStrategy(const std::string &strategy);
[[nodiscard]] bool UsesTraceInferAssignment(const std::string &strategy);

} // namespace kv_cache_manager
