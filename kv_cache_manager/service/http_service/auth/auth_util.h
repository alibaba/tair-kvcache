#pragma once

#include <string_view>

namespace kv_cache_manager {

class AuthUtil {
public:
    // length-revealing equality compare; returns true if a and b
    // have the same length and the same bytes.  early-rejects on
    // size mismatch (so length is leaked through timing); when the
    // sizes match the loop scans every byte without short-circuiting
    // on the first differing one, so timing does not depend on the
    // position of the first mismatch.  callers should keep secrets
    // at a bounded length to avoid leaking length itself
    static bool ConstantTimeEquals(std::string_view a, std::string_view b);

    // case-insensitive ASCII equality, used to match the scheme
    // name "Bearer" per RFC 7235 §2.1 (scheme is case-insensitive)
    static bool ICaseEqualsAscii(std::string_view a, std::string_view b);
};

} // namespace kv_cache_manager
