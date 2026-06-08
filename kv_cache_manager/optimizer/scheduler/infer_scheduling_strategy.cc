#include "kv_cache_manager/optimizer/scheduler/infer_scheduling_strategy.h"

namespace kv_cache_manager {

bool IsSupportedInferSchedulingStrategy(const std::string &strategy) {
    return strategy == InferSchedulingStrategy::kPreserveTrace || strategy == InferSchedulingStrategy::kRoundRobin ||
           strategy == InferSchedulingStrategy::kPrefixHit;
}

bool UsesTraceInferAssignment(const std::string &strategy) {
    return strategy == InferSchedulingStrategy::kPreserveTrace || strategy == InferSchedulingStrategy::kPrefixHit;
}

} // namespace kv_cache_manager
