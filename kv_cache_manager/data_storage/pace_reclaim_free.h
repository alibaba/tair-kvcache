#pragma once

#include <chrono>
#include <cstdint>

namespace kv_cache_manager {

// A successful PACE topology poll. A default timestamp means the count is unknown.
struct PaceProviderCount {
    uint32_t count{0};
    std::chrono::steady_clock::time_point observed_at{};
};

struct PaceReclaimFreeResult {
    enum Status {
        kSuccess,
        kError,
        kBusy,
        kTooLarge,
        kBackendChanged,
        kNotSent
    };
    Status status{kError};
    uint64_t retry_after_ms{0};
    uint64_t max_gas{0};
    bool provider_free_acknowledged{false};
};

} // namespace kv_cache_manager
