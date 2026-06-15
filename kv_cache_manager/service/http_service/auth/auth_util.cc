#include "kv_cache_manager/service/http_service/auth/auth_util.h"

namespace kv_cache_manager {

bool AuthUtil::ConstantTimeEquals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    unsigned char diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff |= static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i]);
    }
    return diff == 0;
}

bool AuthUtil::ICaseEqualsAscii(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        unsigned char ca = static_cast<unsigned char>(a[i]);
        unsigned char cb = static_cast<unsigned char>(b[i]);
        if (ca >= 'A' && ca <= 'Z') {
            ca = static_cast<unsigned char>(ca + ('a' - 'A'));
        }
        if (cb >= 'A' && cb <= 'Z') {
            cb = static_cast<unsigned char>(cb + ('a' - 'A'));
        }
        if (ca != cb) {
            return false;
        }
    }
    return true;
}

} // namespace kv_cache_manager
