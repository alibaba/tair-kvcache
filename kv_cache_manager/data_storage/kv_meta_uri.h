#pragma once

#include <charconv>
#include <cstdint>
#include <string>

#include "kv_cache_manager/common/kv_meta_constants.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

template <typename T>
inline bool ParseKvMetaUriNumber(const DataStorageUri &uri, const char *name, T &value) noexcept {
    if (!uri.HasParam(name)) {
        return false;
    }
    const std::string text = uri.GetParam(name);
    T parsed{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return false;
    }
    value = parsed;
    return true;
}

// Validate the canonical fields emitted by KVCM's PACE backend. Zero is a
// valid value for address components such as node_id; KVCM must not impose
// allocation policy beyond parseability and a non-empty object size.
inline bool GetKvMetaObjectSize(const DataStorageUri &uri, std::uint64_t &size) noexcept {
    size = 0;
    if (!uri.Valid() || uri.GetProtocol() != kTairMempoolUriScheme || uri.GetHostName().empty() ||
        !uri.GetUserInfo().empty() || uri.GetPort() != 0) {
        return false;
    }
    const std::string &path = uri.GetPath();
    if (path.size() < 2 || path.front() != '/') {
        return false;
    }
    std::uint64_t offset = 0;
    const auto offset_result = std::from_chars(path.data() + 1, path.data() + path.size(), offset);
    std::uint16_t node_id = 0;
    std::uint16_t media_type = 0;
    std::uint16_t range_id = 0;
    return offset_result.ec == std::errc{} && offset_result.ptr == path.data() + path.size() &&
           ParseKvMetaUriNumber(uri, "size", size) && size != 0 && ParseKvMetaUriNumber(uri, "node_id", node_id) &&
           ParseKvMetaUriNumber(uri, "media_type", media_type) && ParseKvMetaUriNumber(uri, "range_id", range_id);
}

inline bool IsValidKvMetaLocation(const std::string &uri_text, std::uint64_t &size) noexcept {
    if (uri_text.empty() || uri_text.size() > kKvMetaMaxLocationUriBytes) {
        return false;
    }
    const DataStorageUri uri(uri_text);
    // KVCM's backend always emits StandardUri's canonical representation.
    // Requiring it here also rejects duplicate query keys discarded by the
    // parser without maintaining a second URI parser.
    return uri.ToUriString() == uri_text && GetKvMetaObjectSize(uri, size);
}

} // namespace kv_cache_manager
