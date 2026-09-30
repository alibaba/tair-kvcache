#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

inline constexpr std::size_t kMaxKvMetaLocationUriBytes = 64 * 1024;
inline constexpr std::size_t kMaxKvMetaLocationUriQueryParams = 64;
inline constexpr std::size_t kMaxKvMetaBackendNameBytes = 512;
inline constexpr std::int64_t kKvMetaMaxWriteTimeoutSeconds = 1800;
inline constexpr std::size_t kKvMetaObjectNonceBytes = 32;

inline bool IsCanonicalKvMetaBackendName(std::string_view name) noexcept {
    if (name.empty() || name.size() > kMaxKvMetaBackendNameBytes) {
        return false;
    }
    return std::all_of(name.begin(), name.end(), [](char ch) {
        const auto byte = static_cast<unsigned char>(ch);
        return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || (byte >= '0' && byte <= '9') ||
               byte == '.' || byte == '_' || byte == '-';
    });
}

inline bool HasCanonicalKvMetaAuthority(const DataStorageUri &uri) noexcept {
    return uri.Valid() && IsCanonicalKvMetaBackendName(uri.GetHostName()) && uri.GetUserInfo().empty() &&
           uri.GetPort() == 0;
}

// StandardUri stores query parameters in a map. Reject text that would lose
// information while parsing, especially duplicate ownership fields.
inline bool HasUnambiguousKvMetaUriText(std::string_view uri_text) noexcept {
    if (uri_text.empty() || uri_text.size() > kMaxKvMetaLocationUriBytes ||
        uri_text.find('#') != std::string_view::npos || std::any_of(uri_text.begin(), uri_text.end(), [](char ch) {
            const auto byte = static_cast<unsigned char>(ch);
            return byte <= 0x1f || byte == 0x7f;
        })) {
        return false;
    }

    const auto scheme_end = uri_text.find("://");
    if (scheme_end == std::string_view::npos || scheme_end == 0) {
        return false;
    }
    const std::size_t authority_begin = scheme_end + 3;
    const auto path_begin = uri_text.find('/', authority_begin);
    const auto query_begin = uri_text.find('?', authority_begin);
    const std::size_t authority_end = std::min(path_begin == std::string_view::npos ? uri_text.size() : path_begin,
                                               query_begin == std::string_view::npos ? uri_text.size() : query_begin);
    const auto authority = uri_text.substr(authority_begin, authority_end - authority_begin);
    if (!IsCanonicalKvMetaBackendName(authority)) {
        return false;
    }
    if (query_begin == std::string_view::npos) {
        return true;
    }

    std::array<std::string_view, kMaxKvMetaLocationUriQueryParams> keys{};
    std::size_t count = 0;
    std::size_t begin = query_begin + 1;
    while (begin < uri_text.size()) {
        if (count == keys.size()) {
            return false;
        }
        const auto separator = uri_text.find('&', begin);
        const auto end = separator == std::string_view::npos ? uri_text.size() : separator;
        const auto equals = uri_text.find('=', begin);
        if (equals == std::string_view::npos || equals == begin || equals >= end) {
            return false;
        }
        const auto key = uri_text.substr(begin, equals - begin);
        if (std::find(keys.begin(), keys.begin() + count, key) != keys.begin() + count) {
            return false;
        }
        keys[count++] = key;
        if (separator == std::string_view::npos) {
            return true;
        }
        begin = separator + 1;
    }
    return false;
}

inline bool HasSameCanonicalKvMetaUri(const std::string &expected, const std::string &actual) {
    if (!HasUnambiguousKvMetaUriText(expected) || !HasUnambiguousKvMetaUriText(actual)) {
        return false;
    }
    const DataStorageUri expected_uri(expected);
    const DataStorageUri actual_uri(actual);
    return HasCanonicalKvMetaAuthority(expected_uri) && HasCanonicalKvMetaAuthority(actual_uri) &&
           expected_uri.ToUriString() == actual_uri.ToUriString();
}

inline bool TryGetExactTairMempoolOffset(const DataStorageUri &uri, std::uint64_t &offset) noexcept {
    const std::string &path = uri.GetPath();
    if (path.size() < 2 || path.front() != '/') {
        return false;
    }
    const char *begin = path.data() + 1;
    const char *end = path.data() + path.size();
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end || (end - begin > 1 && *begin == '0')) {
        return false;
    }
    offset = parsed;
    return true;
}

inline bool
TryGetCanonicalUint16Param(const DataStorageUri &uri, const std::string &name, std::uint16_t &value) noexcept {
    if (!uri.HasParam(name)) {
        return false;
    }
    const std::string text = uri.GetParam(name);
    std::uint16_t parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        text != std::to_string(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

inline bool HasExactTairMempoolAddress(const DataStorageUri &uri) noexcept {
    std::uint64_t offset = 0;
    std::uint16_t node_id = 0;
    std::uint16_t media_type = 0;
    std::uint16_t range_id = 0;
    return uri.GetProtocol() == kTairMempoolUriScheme && TryGetExactTairMempoolOffset(uri, offset) && offset != 0 &&
           TryGetCanonicalUint16Param(uri, "node_id", node_id) && node_id != 0 &&
           TryGetCanonicalUint16Param(uri, "media_type", media_type) &&
           TryGetCanonicalUint16Param(uri, "range_id", range_id);
}

inline bool HasOwnedKvMetaAllocationShape(const DataStorageUri &uri, DataStorageType storage_type) noexcept {
    return SupportsKvMetaAdmission(storage_type) && HasExactTairMempoolAddress(uri);
}

inline bool IsKvMetaTairMempoolMediaCompatible(std::uint16_t configured, std::uint16_t actual) noexcept {
    if (configured == kTairMemPoolMediaTypeUnspecified) {
        return actual == kTairMemPoolMediaTypeUnspecified || actual == kTairMemPoolMediaTypeDram ||
               actual == kTairMemPoolMediaTypeSsd;
    }
    return actual == configured;
}

inline bool UriMatchesConfiguredKvMetaNamespace(const DataStorageUri &uri,
                                                DataStorageType storage_type,
                                                const StorageConfig &config) {
    if (!HasCanonicalKvMetaAuthority(uri) || !HasOwnedKvMetaAllocationShape(uri, storage_type) ||
        config.type() != storage_type || config.global_unique_name() != uri.GetHostName()) {
        return false;
    }
    const auto spec = std::dynamic_pointer_cast<TairMemPoolStorageSpec>(config.storage_spec());
    std::uint16_t media_type = 0;
    return spec && TryGetCanonicalUint16Param(uri, "media_type", media_type) &&
           IsKvMetaTairMempoolMediaCompatible(spec->media_type(), media_type);
}

inline bool HasSafeConfiguredKvMetaNamespace(const StorageConfig &config,
                                             std::size_t max_uri_bytes = kMaxKvMetaLocationUriBytes) {
    if (!SupportsKvMetaAdmission(config.type()) || max_uri_bytes == 0 || max_uri_bytes > kMaxKvMetaLocationUriBytes ||
        !IsCanonicalKvMetaBackendName(config.global_unique_name())) {
        return false;
    }
    const auto spec = std::dynamic_pointer_cast<TairMemPoolStorageSpec>(config.storage_spec());
    if (!spec || (config.type() == DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD &&
                  spec->media_type() != kTairMemPoolMediaTypeSsd)) {
        return false;
    }

    DataStorageUri sample;
    sample.SetProtocol(kTairMempoolUriScheme);
    sample.SetHostName(config.global_unique_name());
    sample.SetPath("/" + std::to_string(std::numeric_limits<std::uint64_t>::max()));
    sample.SetParam("node_id", std::to_string(std::numeric_limits<std::uint16_t>::max()));
    sample.SetParam("media_type", std::to_string(spec->media_type()));
    sample.SetParam("range_id", std::to_string(std::numeric_limits<std::uint16_t>::max()));
    sample.SetParam("size", std::to_string(std::numeric_limits<std::uint64_t>::max()));
    const std::string text = sample.ToUriString();
    return text.size() <= max_uri_bytes && HasUnambiguousKvMetaUriText(text) &&
           UriMatchesConfiguredKvMetaNamespace(DataStorageUri(text), config.type(), config);
}

} // namespace kv_cache_manager
