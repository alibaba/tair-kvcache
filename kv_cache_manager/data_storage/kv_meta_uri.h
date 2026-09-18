#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

#include "kv_cache_manager/data_storage/data_storage_uri.h"

namespace kv_cache_manager {

inline constexpr std::size_t kMaxKvMetaLocationUriBytes = 64 * 1024;
inline constexpr std::size_t kMaxKvMetaLocationUriQueryParams = 64;

// StandardUri intentionally canonicalizes query parameters into a map.  A
// KVMeta object URI is also a persisted ownership record, so accepting text
// that loses information during that canonicalization would let different
// layers disagree about security-sensitive fields such as size and blkid.
// Keep this stricter rule scoped to the KVMeta side path; ordinary storage
// URI parsing remains backward compatible.
inline bool HasUnambiguousKvMetaUriText(std::string_view uri_text) noexcept {
    for (const char ch : uri_text) {
        const auto byte = static_cast<unsigned char>(ch);
        if (byte <= 0x1f || byte == 0x7f) {
            return false;
        }
    }
    if (uri_text.find('#') != std::string_view::npos) {
        return false;
    }
    const auto query_position = uri_text.find('?');
    if (query_position == std::string_view::npos) {
        return true;
    }
    // A fixed view array makes validation allocation-free and keeps duplicate
    // detection bounded even for an untrusted persisted URI. The caller-owned
    // URI remains alive for the duration of this function.
    std::array<std::string_view, kMaxKvMetaLocationUriQueryParams> keys{};
    std::size_t begin = query_position + 1;
    std::size_t parameter_count = 0;
    while (begin < uri_text.size()) {
        if (parameter_count == keys.size()) {
            return false;
        }
        const auto end = uri_text.find('&', begin);
        const auto item_end = end == std::string_view::npos ? uri_text.size() : end;
        const auto equals = uri_text.find('=', begin);
        const auto key_end = equals != std::string_view::npos && equals < item_end ? equals : item_end;
        const std::string_view key = uri_text.substr(begin, key_end - begin);
        bool duplicate = false;
        for (std::size_t i = 0; i < parameter_count; ++i) {
            if (keys[i] == key) {
                duplicate = true;
                break;
            }
        }
        if (key.empty() || duplicate) {
            return false;
        }
        keys[parameter_count++] = key;
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    return begin < uri_text.size();
}

// SDKs may reorder query parameters when serializing StandardUri, but an
// exact-object write must never be redirected to another allocation. Compare
// only after rejecting raw text that canonicalization would make ambiguous.
inline bool HasSameCanonicalKvMetaUri(const std::string &expected, const std::string &actual) {
    if (expected.size() > kMaxKvMetaLocationUriBytes || actual.size() > kMaxKvMetaLocationUriBytes ||
        !HasUnambiguousKvMetaUriText(expected) || !HasUnambiguousKvMetaUriText(actual)) {
        return false;
    }
    const DataStorageUri expected_uri(expected);
    const DataStorageUri actual_uri(actual);
    return expected_uri.Valid() && actual_uri.Valid() && !expected_uri.GetHostName().empty() &&
           !actual_uri.GetHostName().empty() && expected_uri.ToUriString() == actual_uri.ToUriString();
}

} // namespace kv_cache_manager
