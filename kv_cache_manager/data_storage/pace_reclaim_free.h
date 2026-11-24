#pragma once

#include <cstdint>

namespace kv_cache_manager {

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
