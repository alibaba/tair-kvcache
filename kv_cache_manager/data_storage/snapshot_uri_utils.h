#pragma once

#include <cctype>
#include <functional>
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
            if (!std::isxdigit(ch)) {
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
    static bool InspectSnapshotUriForVisibility(std::string_view uri_text, std::string_view &out_version) noexcept {
        out_version = {};

        // This is the observable validity condition used by StandardUri::Valid
        // for a freshly constructed URI: it must have a non-empty protocol
        // before "://". ReportEvent performs the full parse before persisting
        // metadata; this read-side check additionally protects recovered or
        // otherwise malformed metadata without rebuilding the parsed object.
        const size_t protocol_end = uri_text.find("://");
        if (protocol_end == std::string_view::npos || protocol_end == 0) {
            return false;
        }

        const size_t query_begin = uri_text.find('?');
        if (query_begin == std::string_view::npos) {
            return true;
        }
        if (query_begin < protocol_end + 3) {
            return false;
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
            const bool is_hex_digit = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F');
            if (!is_hex_digit) {
                return false;
            }
        }
        return true;
    }

    static bool HasEventReportInternalUriMetadata(const DataStorageUri &uri) {
        return uri.HasParam(kSnapshotVersionParam);
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
        uri.SetParam(kSnapshotVersionParam, version);
        out_uri = uri.ToUriString();
        return !out_uri.empty();
    }

    static bool ParseEventReportLocationId(const std::string &location_id,
                                           std::string &out_storage_type,
                                           std::string &out_medium,
                                           std::string &out_host_ip_port) {
        out_storage_type.clear();
        out_medium.clear();
        out_host_ip_port.clear();
        constexpr const char *root_prefix = "kvs#";
        constexpr size_t root_prefix_size = 4;
        if (location_id.size() <= root_prefix_size || location_id.compare(0, root_prefix_size, root_prefix) != 0) {
            return false;
        }
        const size_t type_end = location_id.find('#', root_prefix_size);
        if (type_end == std::string::npos || type_end == root_prefix_size) {
            return false;
        }
        const std::string storage_type = location_id.substr(root_prefix_size, type_end - root_prefix_size);
        if (storage_type != "event_report_l1p5" && storage_type != "event_report_l2") {
            return false;
        }
        const size_t medium_begin = type_end + 1;
        const size_t separator = location_id.find('#', medium_begin);
        if (separator == std::string::npos || separator == medium_begin || separator + 1 >= location_id.size()) {
            return false;
        }
        const std::string medium = location_id.substr(medium_begin, separator - medium_begin);
        const std::string host_ip_port = location_id.substr(separator + 1);
        if (host_ip_port.empty() || host_ip_port.find('#') != std::string::npos) {
            return false;
        }
        out_storage_type = storage_type;
        out_medium = medium;
        out_host_ip_port = host_ip_port;
        return true;
    }

    static bool
    ParseEventReportLocationId(const std::string &location_id, std::string &out_medium, std::string &out_host_ip_port) {
        std::string storage_type;
        return ParseEventReportLocationId(location_id, storage_type, out_medium, out_host_ip_port);
    }

private:
    SnapshotUriUtils() = delete;
};

} // namespace kv_cache_manager
