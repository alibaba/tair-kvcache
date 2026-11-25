#pragma once

#include <chrono>
#include <cstdint>

namespace kv_cache_manager {

// A successful PACE topology poll. A default timestamp means the count is unknown.
struct PaceProviderCount {
    uint32_t count{0};
    std::chrono::steady_clock::time_point observed_at{};
};

// Without a generation the GA came from a legacy allocation.
struct PaceReclaimGA {
    uint64_t ga{0};
    uint32_t node_id{0};
    uint64_t generation{0};

    bool HasIdentity() const { return generation != 0; }
};

inline bool IsValidPaceNodeId(uint64_t id) { return id > 0 && id <= 0x7fffff && id != UINT16_MAX; }

struct PaceReclaimFreeResult {
    enum Status {
        kSuccess,
        kError,
        kBusy,
        kTooLarge,
        kBackendChanged,
        kRejected,
        kNotSent
    };
    Status status{kError};
    uint64_t retry_after_ms{0};
    uint64_t max_gas{0};
    bool recovery_confirmed{false};
};

} // namespace kv_cache_manager
