#pragma once

#include <charconv>
#include <cstdint>
#include <string_view>

namespace kv_cache_manager {

struct TairReclaimFreeResult {
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

// The list endpoint also supports administrative cleanup. Only emit canonical,
// nonzero decimal GAs from a complete URI path on the watermark-eviction path.
inline bool ParseReclaimGA(std::string_view path, uint64_t &ga) {
    if (path.size() < 2 || path.front() != '/') {
        return false;
    }
    path.remove_prefix(1);
    const auto end = path.data() + path.size();
    const auto result = std::from_chars(path.data(), end, ga);
    return result.ec == std::errc{} && result.ptr == end && ga != 0;
}

} // namespace kv_cache_manager
