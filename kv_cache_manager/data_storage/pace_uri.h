#pragma once

#include <charconv>
#include <cstdint>
#include <string_view>

namespace kv_cache_manager {

// PACE stores the decimal GA in the URI path. Zero denotes an invalid address.
inline uint64_t ParsePaceGAPath(std::string_view path) noexcept {
    if (path.size() < 2 || path.front() != '/') {
        return 0;
    }
    uint64_t ga = 0;
    const char *end = path.data() + path.size();
    const auto [ptr, ec] = std::from_chars(path.data() + 1, end, ga);
    return ec == std::errc{} && ptr == end ? ga : 0;
}

} // namespace kv_cache_manager
