#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kv_cache_manager {

// Decision-only projection of the online MRC window. The coverage value is
// limited to Optimizer ingress and does not claim DashTrace-to-KVCM coverage.
struct OnlineMrcCurvePoint {
    uint64_t capacity_bytes = 0;
    uint64_t input_tokens = 0;
    uint64_t hit_tokens = 0;
};

struct OnlineMrcSourceSnapshot {
    std::string source_id;
    int64_t newest_event_time_ns = 0;
    uint64_t accepted_facts = 0;
    std::vector<OnlineMrcCurvePoint> curve;

    // Exact online shadow observation at the quota that was acknowledged by
    // KVCM while these facts were consumed. Unlike the MRC curve, this is one
    // enforced capacity and therefore represents the hit rate the shadow
    // cache actually delivered for this traffic window.
    int64_t enforced_shadow_quota_bytes = 0;
    uint64_t enforced_shadow_generation = 0;
    uint64_t enforced_shadow_accepted_facts = 0;
    uint64_t enforced_shadow_input_tokens = 0;
    uint64_t enforced_shadow_hit_tokens = 0;
};

struct OnlineMrcDecisionSnapshot {
    uint64_t snapshot_id = 0;
    int64_t created_at_ns = 0;
    std::vector<OnlineMrcSourceSnapshot> sources;
};

} // namespace kv_cache_manager
