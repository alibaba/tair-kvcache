#pragma once

#include <array>
#include <cstdint>

namespace kv_cache_manager {

enum class DeleteSource {
    Reclaimer,
    GC,
    Trim
};

struct DeleteSize {
    uint64_t locations{0};
    uint64_t bytes{0};
};

struct DeleteSourceLimit {
    uint64_t requests{0};
    DeleteSize size;
};

// Zero disables that source ceiling. Cluster and legacy non-PACE ceilings are positive.
struct DeleteLimits {
    static constexpr uint32_t kDefaultMaxBackoffMs = 5000;
    DeleteSize cluster{24576, 1280ULL << 30};
    DeleteSize non_pace_reclaim{20000, 1ULL << 40};
    std::array<DeleteSourceLimit, 3> sources{{{1024, {0, 4ULL << 40}}, {64, {}}, {32, {}}}};
    bool protocol{true};
    uint64_t initial_backoff_ms{50};
    uint64_t retry_memory_bytes{128ULL << 20};
    uint32_t max_backoff_ms{kDefaultMaxBackoffMs};
};

} // namespace kv_cache_manager
