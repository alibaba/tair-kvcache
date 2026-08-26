#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "kv_cache_manager/data_storage/data_storage_uri.h"

namespace kv_cache_manager {

// Identifies the one reporter whose complete cache set is replaced together.
// This is an implementation key for versioning and mutual exclusion, not a
// separately visible protocol object.
struct ReporterSnapshotKey {
    std::string instance_id;
    std::string host_ip_port;

    bool operator==(const ReporterSnapshotKey &other) const noexcept {
        return instance_id == other.instance_id && host_ip_port == other.host_ip_port;
    }

    bool operator!=(const ReporterSnapshotKey &other) const noexcept { return !(*this == other); }
};

struct ReporterSnapshotKeyHash {
    size_t operator()(const ReporterSnapshotKey &key) const {
        size_t seed = std::hash<std::string>{}(key.instance_id);
        seed ^= std::hash<std::string>{}(key.host_ip_port) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct SnapshotUriInfo {
    std::string version;
};

struct CanonicalSnapshotUriAppendInfo {
    size_t insertion_offset = 0;
    std::uint64_t size = 0;
    bool has_query = false;
};

class SnapshotUriUtils {
public:
    inline static constexpr const char *kSnapshotVersionParam = "s_version";

    static bool IsValidLocationIdComponent(const std::string &value) {
        return !value.empty() && value.find('#') == std::string::npos;
    }

    static size_t CountUriParam(const std::string &uri_text, const std::string &key) {
        const size_t query_begin = uri_text.find('?');
        if (query_begin == std::string::npos) {
            return 0;
        }
        size_t count = 0;
        size_t begin = query_begin + 1;
        while (begin <= uri_text.size()) {
            size_t end = uri_text.find('&', begin);
            if (end == std::string::npos) {
                end = uri_text.size();
            }
            const size_t equals = uri_text.find('=', begin);
            const size_t key_end = equals != std::string::npos && equals < end ? equals : end;
            if (uri_text.compare(begin, key_end - begin, key) == 0 && key_end - begin == key.size()) {
                ++count;
            }
            if (end == uri_text.size()) {
                break;
            }
            begin = end + 1;
        }
        return count;
    }

    static bool IsValidSnapshotVersionToken(const std::string &version) {
        if (version.size() != 32) {
            return false;
        }
        for (const unsigned char ch : version) {
            if (!IsAsciiHexDigit(ch)) {
                return false;
            }
        }
        return true;
    }

    // GetHostCacheState only needs to know whether an EventReport URI is
    // structurally usable and which snapshot generation it belongs to. A full
    // DataStorageUri parse would copy every URI component and allocate one
    // std::map node per query parameter for every (key, spec) visited. Scan the
    // immutable URI in place instead. An empty out_version means legacy
    // metadata without s_version; malformed or duplicate s_version parameters
    // fail closed.
    //
    // The returned view borrows uri_text and must not outlive it.
    static bool InspectSnapshotUriForVisibility(std::string_view uri_text,
                                                std::string_view &out_version,
                                                bool uri_structure_prevalidated = false) noexcept {
        out_version = {};

        // Match the observable validity conditions used by StandardUri
        // without rebuilding its strings and parameter map. In addition to a
        // non-empty protocol, a textual port must be a non-negative int64.
        // Host/path/query contents otherwise remain intentionally permissive,
        // just like StandardUri::Parse.
        const size_t protocol_end = uri_text.find("://");
        if (protocol_end == std::string_view::npos || protocol_end == 0) {
            return false;
        }

        const size_t authority_begin = protocol_end + 3;
        const size_t query_begin = uri_text.find('?');
        if (query_begin != std::string_view::npos && query_begin < authority_begin) {
            return false;
        }
        if (!uri_structure_prevalidated) {
            const size_t path_begin = uri_text.find('/', authority_begin);
            const size_t authority_end =
                std::min(path_begin == std::string_view::npos ? uri_text.size() : path_begin,
                         query_begin == std::string_view::npos ? uri_text.size() : query_begin);
            size_t host_begin = authority_begin;
            const size_t user_info_end = uri_text.find('@', authority_begin);
            if (user_info_end != std::string_view::npos && user_info_end < authority_end) {
                host_begin = user_info_end + 1;
            }
            const size_t port_separator = uri_text.find(':', host_begin);
            if (port_separator != std::string_view::npos && port_separator < authority_end) {
                std::uint64_t port = 0;
                if (!ParseDecimalUint64(uri_text.substr(port_separator + 1, authority_end - port_separator - 1),
                                        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
                                        port)) {
                    return false;
                }
            }
        }

        if (query_begin == std::string_view::npos) {
            return true;
        }

        bool found_version = false;
        size_t begin = query_begin + 1;
        while (begin <= uri_text.size()) {
            size_t end = uri_text.find('&', begin);
            if (end == std::string_view::npos) {
                end = uri_text.size();
            }
            const size_t equals = uri_text.find('=', begin);
            const size_t key_end = equals != std::string_view::npos && equals < end ? equals : end;
            constexpr std::string_view version_key{kSnapshotVersionParam};
            if (key_end - begin == version_key.size() &&
                uri_text.compare(begin, version_key.size(), version_key) == 0) {
                if (found_version || equals == std::string_view::npos || equals >= end) {
                    return false;
                }
                found_version = true;
                out_version = uri_text.substr(equals + 1, end - equals - 1);
            }
            if (end == uri_text.size()) {
                break;
            }
            begin = end + 1;
        }

        if (!found_version) {
            return true;
        }
        if (out_version.size() != 32) {
            return false;
        }
        for (const unsigned char ch : out_version) {
            if (!IsAsciiHexDigit(ch)) {
                return false;
            }
        }
        return true;
    }

    static bool HasEventReportInternalUriMetadata(const DataStorageUri &uri) {
        return uri.HasParam(kSnapshotVersionParam);
    }

    // Allocation-free fast parser for an already canonical URI. It accepts
    // exactly the textual form StandardUri::ToUriString() produces: a valid
    // positive canonical port, explicit `key=value` query entries, and unique
    // keys in strict lexical order. Noncanonical-but-valid input returns false
    // so callers can use the full StandardUri fallback without changing wire
    // compatibility. The returned offset inserts s_version in sorted order.
    static bool ParseCanonicalUriForSnapshotAppend(std::string_view uri, CanonicalSnapshotUriAppendInfo &out) noexcept {
        out = {};
        const size_t protocol_end = uri.find("://");
        if (protocol_end == std::string_view::npos || protocol_end == 0) {
            return false;
        }
        const size_t authority_start = protocol_end + 3;
        const size_t path_start = uri.find('/', authority_start);
        const size_t query_start = uri.find('?', authority_start);
        if (path_start != std::string_view::npos && query_start != std::string_view::npos && query_start < path_start) {
            return false;
        }
        const size_t host_end = std::min(path_start == std::string_view::npos ? uri.size() : path_start,
                                         query_start == std::string_view::npos ? uri.size() : query_start);
        size_t host_start = authority_start;
        const size_t user_info_end = uri.find('@', authority_start);
        if (user_info_end != std::string_view::npos && user_info_end < host_end) {
            // StandardUri omits an empty user-info when serializing, so this
            // raw spelling is valid but not canonical.
            if (user_info_end == authority_start) {
                return false;
            }
            host_start = user_info_end + 1;
        }
        const size_t port_separator = uri.find(':', host_start);
        if (port_separator != std::string_view::npos && port_separator < host_end) {
            const std::string_view port_text = uri.substr(port_separator + 1, host_end - port_separator - 1);
            std::uint64_t port = 0;
            if (port_text.empty() || port_text.front() == '0' ||
                !ParseDecimalUint64(
                    port_text, static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()), port)) {
                return false;
            }
        }

        out.insertion_offset = uri.size();
        if (query_start == std::string_view::npos) {
            return true;
        }
        out.has_query = true;
        size_t param_start = query_start + 1;
        if (param_start == uri.size()) {
            return false;
        }
        std::string_view previous_key;
        bool has_previous_key = false;
        while (param_start < uri.size()) {
            size_t param_end = uri.find('&', param_start);
            if (param_end == std::string_view::npos) {
                param_end = uri.size();
            }
            const size_t equals = uri.find('=', param_start);
            if (equals == std::string_view::npos || equals >= param_end) {
                return false;
            }
            const std::string_view key = uri.substr(param_start, equals - param_start);
            const std::string_view value = uri.substr(equals + 1, param_end - equals - 1);
            if ((has_previous_key && !(previous_key < key)) || key == kSnapshotVersionParam) {
                return false;
            }
            if (out.insertion_offset == uri.size() && std::string_view(kSnapshotVersionParam) < key) {
                out.insertion_offset = param_start;
            }
            if (key == "size") {
                std::uint64_t parsed_size = 0;
                if (ParseDecimalUint64(value, std::numeric_limits<std::uint64_t>::max(), parsed_size)) {
                    out.size = parsed_size;
                }
            }
            previous_key = key;
            has_previous_key = true;
            if (param_end == uri.size()) {
                break;
            }
            param_start = param_end + 1;
            if (param_start == uri.size()) {
                return false;
            }
        }
        return true;
    }

    static bool AddSnapshotVersionToCanonicalUri(std::string_view uri,
                                                 const CanonicalSnapshotUriAppendInfo &info,
                                                 const std::string &version,
                                                 std::string &out_uri) {
        out_uri.clear();
        if (!IsValidSnapshotVersionToken(version)) {
            return false;
        }
        return AddPrevalidatedSnapshotVersionToCanonicalUri(uri, info, version, out_uri);
    }

    // ReportEvent obtains one KVCM-generated, already validated generation
    // token and appends it to tens of thousands of independently validated
    // specs. Keep the public checked helper above for general callers; this
    // variant avoids rescanning the same 32-byte token for every block.
    static bool AddPrevalidatedSnapshotVersionToCanonicalUri(std::string_view uri,
                                                             const CanonicalSnapshotUriAppendInfo &info,
                                                             const std::string &version,
                                                             std::string &out_uri) {
        out_uri.clear();
        if (info.insertion_offset > uri.size()) {
            return false;
        }
        out_uri.reserve(uri.size() + std::char_traits<char>::length(kSnapshotVersionParam) + version.size() + 2);
        if (info.insertion_offset == uri.size()) {
            out_uri.assign(uri.data(), uri.size());
            out_uri.push_back(info.has_query ? '&' : '?');
            out_uri.append(kSnapshotVersionParam).push_back('=');
            out_uri.append(version);
            return true;
        }
        out_uri.assign(uri.data(), info.insertion_offset);
        out_uri.append(kSnapshotVersionParam).push_back('=');
        out_uri.append(version).push_back('&');
        out_uri.append(uri.data() + info.insertion_offset, uri.size() - info.insertion_offset);
        return true;
    }

    // DataStorageUri stores query parameters in a map, so duplicate keys from
    // the original text are no longer observable here. Raw metadata and
    // protocol input must use the string overload below, which enforces that
    // s_version occurs exactly once.
    static bool ParseSnapshotUriInfo(const DataStorageUri &uri, SnapshotUriInfo &out) {
        out.version.clear();
        if (!uri.Valid()) {
            return false;
        }
        const std::string version = uri.GetParam(kSnapshotVersionParam);
        if (!IsValidSnapshotVersionToken(version)) {
            return false;
        }
        out.version = version;
        return true;
    }

    static bool ParseSnapshotUriInfo(const std::string &uri_text, SnapshotUriInfo &out) {
        out.version.clear();
        if (CountUriParam(uri_text, kSnapshotVersionParam) != 1) {
            return false;
        }
        return ParseSnapshotUriInfo(DataStorageUri(uri_text), out);
    }

    static bool AddSnapshotVersionToUri(const std::string &raw_uri, const std::string &version, std::string &out_uri) {
        out_uri.clear();
        DataStorageUri uri(raw_uri);
        if (!uri.Valid() || !IsValidSnapshotVersionToken(version) ||
            CountUriParam(raw_uri, kSnapshotVersionParam) != 0) {
            return false;
        }
        return AddSnapshotVersionToUri(std::move(uri), version, out_uri);
    }

    // ReportEvent validates and parses every URI before it acquires the
    // snapshot/delta fence. Reusing that parsed value avoids parsing the same
    // URI again merely to append KVCM's internal generation token.
    static bool AddSnapshotVersionToUri(DataStorageUri uri, const std::string &version, std::string &out_uri) {
        out_uri.clear();
        if (!uri.Valid() || !IsValidSnapshotVersionToken(version) || HasEventReportInternalUriMetadata(uri)) {
            return false;
        }
        out_uri = uri.ToUriStringWithExtraParam(kSnapshotVersionParam, version);
        return !out_uri.empty();
    }

    // Zero-copy parser for the per-block EventReport location id hot path.
    // Returned views borrow location_id and must not outlive it.
    static bool ParseEventReportLocationIdView(std::string_view location_id,
                                               std::string_view &out_storage_type,
                                               std::string_view &out_medium,
                                               std::string_view &out_host_ip_port) noexcept {
        out_storage_type = {};
        out_medium = {};
        out_host_ip_port = {};
        constexpr std::string_view root_prefix{"kvs#"};
        constexpr size_t root_prefix_size = root_prefix.size();
        if (location_id.size() <= root_prefix_size || location_id.compare(0, root_prefix_size, root_prefix) != 0) {
            return false;
        }
        const size_t type_end = location_id.find('#', root_prefix_size);
        if (type_end == std::string::npos || type_end == root_prefix_size) {
            return false;
        }
        const std::string_view storage_type = location_id.substr(root_prefix_size, type_end - root_prefix_size);
        if (storage_type != "event_report_l1p5" && storage_type != "event_report_l2") {
            return false;
        }
        const size_t medium_begin = type_end + 1;
        const size_t separator = location_id.find('#', medium_begin);
        if (separator == std::string::npos || separator == medium_begin || separator + 1 >= location_id.size()) {
            return false;
        }
        const std::string_view medium = location_id.substr(medium_begin, separator - medium_begin);
        const std::string_view host_ip_port = location_id.substr(separator + 1);
        if (host_ip_port.empty() || host_ip_port.find('#') != std::string::npos) {
            return false;
        }
        out_storage_type = storage_type;
        out_medium = medium;
        out_host_ip_port = host_ip_port;
        return true;
    }

    static bool ParseEventReportLocationId(const std::string &location_id,
                                           std::string &out_storage_type,
                                           std::string &out_medium,
                                           std::string &out_host_ip_port) {
        std::string_view storage_type;
        std::string_view medium;
        std::string_view host_ip_port;
        if (!ParseEventReportLocationIdView(location_id, storage_type, medium, host_ip_port)) {
            out_storage_type.clear();
            out_medium.clear();
            out_host_ip_port.clear();
            return false;
        }
        out_storage_type.assign(storage_type.data(), storage_type.size());
        out_medium.assign(medium.data(), medium.size());
        out_host_ip_port.assign(host_ip_port.data(), host_ip_port.size());
        return true;
    }

    static bool
    ParseEventReportLocationId(const std::string &location_id, std::string &out_medium, std::string &out_host_ip_port) {
        std::string storage_type;
        return ParseEventReportLocationId(location_id, storage_type, out_medium, out_host_ip_port);
    }

private:
    static bool ParseDecimalUint64(std::string_view text, std::uint64_t limit, std::uint64_t &out) noexcept {
        if (text.empty()) {
            return false;
        }
        std::uint64_t value = 0;
        for (const unsigned char ch : text) {
            if (ch < '0' || ch > '9') {
                return false;
            }
            const std::uint64_t digit = ch - '0';
            if (value > (limit - digit) / 10) {
                return false;
            }
            value = value * 10 + digit;
        }
        out = value;
        return true;
    }

    static constexpr bool IsAsciiHexDigit(unsigned char ch) noexcept {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F');
    }

    SnapshotUriUtils() = delete;
};

} // namespace kv_cache_manager
