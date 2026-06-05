#pragma once

#include <string_view>

namespace kv_cache_manager {

class AuthUtil {
public:
    // length-revealing constant-time equality compare; returns true
    // iff a and b have the same length and the same bytes.  the
    // comparison cost is O(min(len(a), len(b))) regardless of where
    // the first mismatch occurs, defeating naive timing oracles on
    // the matching prefix.  callers should keep secrets at a
    // bounded length to avoid leaking length itself
    static bool ConstantTimeEquals(std::string_view a, std::string_view b);

    // case-insensitive ASCII equality, used to match the scheme
    // name "Bearer" per RFC 7235 §2.1 (scheme is case-insensitive)
    static bool ICaseEqualsAscii(std::string_view a, std::string_view b);
};

} // namespace kv_cache_manager
