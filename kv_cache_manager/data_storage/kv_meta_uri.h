#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/kv_meta_identity.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

inline constexpr std::size_t kMaxKvMetaLocationUriBytes = 64 * 1024;
inline constexpr std::size_t kMaxKvMetaLocationUriQueryParams = 64;
inline constexpr std::size_t kMaxKvMetaBackendNameBytes = 512;
inline constexpr std::int64_t kKvMetaMaxWriteTimeoutSeconds = 1800;
inline constexpr std::int64_t kKvMetaMaxFailedWriteCleanupGraceSeconds = 180;
// Exact remote control calls are bounded separately from the caller's data-I/O
// lease. StartWrite checks its deadline between singleton commits, so at most
// one request can cross that boundary; exact rollback is then bounded by the
// same per-request limit. The Provider's provisional lease must cover that
// entire reconcile window. These constraints apply only to KVMeta-capable
// backends and never change the fixed-block KV-cache path.
inline constexpr std::int64_t kKvMetaMaxExactControlRpcTimeoutSeconds = 120;
// One exact rollback performs at most two mutation attempts and two
// authoritative queries after each attempt. Backends may finish earlier, but
// must not add an unbounded retry loop behind the extension interface.
inline constexpr std::int64_t kKvMetaMaxExactCleanupControlRequests = 6;
inline constexpr std::uint64_t kKvMetaMinimumExactAllocationLeaseSeconds = 3600;
static_assert(kKvMetaMinimumExactAllocationLeaseSeconds >
              static_cast<std::uint64_t>(kKvMetaMaxWriteTimeoutSeconds + kKvMetaMaxFailedWriteCleanupGraceSeconds));
static_assert(kKvMetaMinimumExactAllocationLeaseSeconds >
              static_cast<std::uint64_t>(kKvMetaMaxWriteTimeoutSeconds +
                                         kKvMetaMaxExactControlRpcTimeoutSeconds *
                                             (1 + kKvMetaMaxExactCleanupControlRequests)));
// The authority is an internal DataStorageManager key, not an arbitrary DNS
// endpoint. Keep its persisted spelling stable and URI-safe so Create, client
// dispatch, and Delete cannot parse one backend name in different ways.
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

// Backend names, rather than arbitrary network authorities, select the SDK
// and deletion provider. Generated KVMeta URIs never carry user-info or a
// port; accepting either creates textual aliases that StandardUri may later
// normalize differently (notably an explicit port zero).
inline bool HasCanonicalKvMetaAuthority(const DataStorageUri &uri) noexcept {
    return uri.Valid() && IsCanonicalKvMetaBackendName(uri.GetHostName()) && uri.GetUserInfo().empty() &&
           uri.GetPort() == 0;
}

// File-like backends delete exactly the path in the ownership record. Require
// a lexical canonical absolute object path: an empty segment, trailing slash,
// or dot segment can alias another spelling (and "/tmp/.." can name the
// filesystem root). Such aliases must never reach a backend Delete call when
// metadata is damaged.
inline bool HasOwnedKvMetaFilePath(const DataStorageUri &uri) noexcept {
    const std::string &path = uri.GetPath();
    if (path.size() <= 1 || path.front() != '/' || path.back() == '/') {
        return false;
    }
    std::size_t segment_begin = 1;
    while (segment_begin < path.size()) {
        const std::size_t segment_end = path.find('/', segment_begin);
        const std::size_t end = segment_end == std::string::npos ? path.size() : segment_end;
        const std::string_view segment(path.data() + segment_begin, end - segment_begin);
        if (segment.empty() || segment == "." || segment == "..") {
            return false;
        }
        if (segment_end == std::string::npos) {
            break;
        }
        segment_begin = segment_end + 1;
    }
    return true;
}

// An exact-object NFS root is a durable namespace boundary shared by the
// control plane and every data-plane client. Keep one unambiguous absolute
// spelling: relative roots depend on process cwd, dot/empty segments alias a
// different path, and "/" is too broad a deletion boundary for cache-owned
// objects. The trailing slash remains part of the existing NfsBackend
// concatenation contract.
inline bool HasCanonicalKvMetaNfsRootPath(std::string_view root_path) noexcept {
    if (root_path.size() <= 1 || root_path.front() != '/' || root_path.back() != '/' ||
        root_path.find('\0') != std::string_view::npos) {
        return false;
    }
    std::size_t segment_begin = 1;
    while (segment_begin < root_path.size()) {
        const std::size_t segment_end = root_path.find('/', segment_begin);
        if (segment_end == std::string_view::npos) {
            return false;
        }
        const std::string_view segment = root_path.substr(segment_begin, segment_end - segment_begin);
        if (segment.empty() || segment == "." || segment == "..") {
            return false;
        }
        if (segment_end == root_path.size() - 1) {
            return true;
        }
        segment_begin = segment_end + 1;
    }
    return false;
}

// Server-generated object keys are deliberately self-describing and use only
// canonical lowercase text. Besides reducing accidental namespace overlap,
// this lets a data-plane client reject a corrupted metadata response before
// it can read or overwrite an arbitrary file inside the configured root.
inline bool HasCanonicalKvMetaObjectKey(std::string_view object_key) noexcept {
    constexpr std::string_view kPrefix = "kvmeta/";
    if (object_key.size() <= kPrefix.size() || object_key.compare(0, kPrefix.size(), kPrefix) != 0) {
        return false;
    }
    const std::size_t instance_end = object_key.find('/', kPrefix.size());
    if (instance_end == std::string_view::npos) {
        return false;
    }
    const std::size_t key_end = object_key.find('/', instance_end + 1);
    if (key_end == std::string_view::npos || object_key.find('/', key_end + 1) != std::string_view::npos) {
        return false;
    }
    const auto is_legacy_canonical_hex = [](std::string_view value) {
        return !value.empty() && value.size() <= kKvMetaCanonicalUint64MaxHexChars &&
               (value.size() == 1 || value.front() != '0') && std::all_of(value.begin(), value.end(), [](char ch) {
                   const auto byte = static_cast<unsigned char>(ch);
                   return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f');
               });
    };
    const auto is_new_key_fingerprint = [](std::string_view value) {
        return value.size() == kKvMetaObjectFingerprintHexChars && std::all_of(value.begin(), value.end(), [](char ch) {
                   const auto byte = static_cast<unsigned char>(ch);
                   return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f');
               });
    };
    const std::string_view instance_hash = object_key.substr(kPrefix.size(), instance_end - kPrefix.size());
    const std::string_view key_hash = object_key.substr(instance_end + 1, key_end - instance_end - 1);
    const std::string_view nonce = object_key.substr(key_end + 1);
    // The short key-hash form is read only for rolling cleanup compatibility.
    // Every new allocation uses the fixed-width, instance-bound fingerprint.
    return is_legacy_canonical_hex(instance_hash) &&
           (is_new_key_fingerprint(key_hash) || is_legacy_canonical_hex(key_hash)) &&
           nonce.size() == kKvMetaObjectNonceBytes && std::all_of(nonce.begin(), nonce.end(), [](char ch) {
               const auto byte = static_cast<unsigned char>(ch);
               return (byte >= 'a' && byte <= 'z') || (byte >= '0' && byte <= '9');
           });
}

inline bool TryGetCanonicalKvMetaObjectKeyFromPath(std::string_view path, std::string_view &object_key) noexcept {
    object_key = {};
    if (path.empty() || path.front() != '/') {
        return false;
    }
    std::size_t component_boundary = path.size();
    // Walk back over nonce, key hash, instance hash and the `kvmeta`
    // component. The remaining prefix is the backend-configured root.
    for (int i = 0; i < 4; ++i) {
        if (component_boundary == 0) {
            return false;
        }
        component_boundary = path.rfind('/', component_boundary - 1);
        if (component_boundary == std::string_view::npos) {
            return false;
        }
    }
    object_key = path.substr(component_boundary + 1);
    if (!HasCanonicalKvMetaObjectKey(object_key)) {
        object_key = {};
        return false;
    }
    return true;
}

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
    const auto scheme_end = uri_text.find("://");
    if (scheme_end == std::string_view::npos || scheme_end == 0) {
        return false;
    }
    const std::size_t authority_begin = scheme_end + 3;
    const auto path_position = uri_text.find('/', authority_begin);
    const auto query_position = uri_text.find('?', authority_begin);
    const std::size_t authority_end =
        std::min(path_position == std::string_view::npos ? uri_text.size() : path_position,
                 query_position == std::string_view::npos ? uri_text.size() : query_position);
    const std::string_view authority = uri_text.substr(authority_begin, authority_end - authority_begin);
    if (authority.empty() || authority.find('@') != std::string_view::npos ||
        authority.find(':') != std::string_view::npos) {
        return false;
    }
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
        if (equals == std::string_view::npos || equals >= item_end) {
            // StandardUri serializes both `flag` and `flag=` as `flag=`.
            // Require the explicit delimiter so canonical comparison cannot
            // erase that distinction in an ownership record.
            return false;
        }
        const auto key_end = equals;
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
    return HasCanonicalKvMetaAuthority(expected_uri) && HasCanonicalKvMetaAuthority(actual_uri) &&
           expected_uri.ToUriString() == actual_uri.ToUriString();
}

// TairMempool's physical byte offset is encoded as the complete URI path
// ("/<uint64>"). StandardUri deliberately leaves path interpretation to its
// caller, while the SDK's historical convenience parser used zero for both a
// real offset 0 and every malformed/missing value. KVMeta must distinguish
// those cases before dispatching any data-plane I/O: otherwise an untrusted
// or corrupted location can silently redirect a read/write to allocation 0.
inline bool TryGetExactTairMempoolOffset(const DataStorageUri &uri, std::uint64_t &offset) noexcept {
    const std::string &path = uri.GetPath();
    if (path.size() < 2 || path.front() != '/') {
        return false;
    }
    const char *begin = path.data() + 1;
    const char *end = path.data() + path.size();
    std::uint64_t parsed_offset = 0;
    const auto parsed = std::from_chars(begin, end, parsed_offset);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        return false;
    }
    offset = parsed_offset;
    return true;
}

inline bool HasExactTairMempoolOffset(const DataStorageUri &uri) noexcept {
    std::uint64_t ignored = 0;
    return TryGetExactTairMempoolOffset(uri, ignored);
}

inline bool
TryGetTairMempoolUint16ParamOrDefault(const DataStorageUri &uri, const std::string &name, std::uint16_t &value) {
    value = 0;
    if (!uri.HasParam(name)) {
        return true;
    }
    const std::string text = uri.GetParam(name);
    std::uint16_t parsed_value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), parsed_value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        return false;
    }
    value = parsed_value;
    return true;
}

inline bool HasExactOptionalTairMempoolUint16Param(const DataStorageUri &uri, const std::string &name) {
    std::uint16_t ignored = 0;
    return TryGetTairMempoolUint16ParamOrDefault(uri, name, ignored);
}

// The three query fields are optional for compatibility and default to zero
// in the PACE SDK. If present, however, they must be a complete uint16 value;
// otherwise malformed text would silently alias the real zero-valued address.
inline bool HasExactTairMempoolAddress(const DataStorageUri &uri) {
    return HasExactTairMempoolOffset(uri) && HasExactOptionalTairMempoolUint16Param(uri, "node_id") &&
           HasExactOptionalTairMempoolUint16Param(uri, "media_type") &&
           HasExactOptionalTairMempoolUint16Param(uri, "range_id");
}

inline bool HasCanonicalTairMempoolProviderIncarnation(const DataStorageUri &uri) {
    if (!uri.HasParam("provider_incarnation")) {
        return false;
    }
    const std::string value = uri.GetParam("provider_incarnation");
    if (value.size() != 36) {
        return false;
    }
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (value[i] != '-')
                return false;
        } else if (!((value[i] >= '0' && value[i] <= '9') || (value[i] >= 'a' && value[i] <= 'f'))) {
            return false;
        }
    }
    return true;
}

// Stable Provider identity is serialized as an unescaped URI query value and
// is also used as a MetaService cache key. Keep one narrow canonical alphabet
// across KVCM, MetaService and Provider so percent-decoding, whitespace or
// delimiter handling cannot make the same route compare differently.
inline bool IsCanonicalTairMempoolProviderUuid(std::string_view value) noexcept {
    if (value.empty() || value.size() >= 64) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](char ch) {
        const auto byte = static_cast<unsigned char>(ch);
        return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || (byte >= '0' && byte <= '9') ||
               ch == '-' || ch == '_' || ch == '.';
    });
}

// New exact-allocation responses persist the Provider's stable routing id so
// MetaService can route a live durable owner after a numeric node-id change.
// The UUID is never deletion authority when owner/proof state is missing. The
// field remains optional only for rolling-upgrade compatibility with already
// persisted URIs whose durable owner still supplies the route identity.
inline bool HasSafeOptionalTairMempoolProviderUuid(const DataStorageUri &uri) {
    if (!uri.HasParam("provider_uuid")) {
        return true;
    }
    return IsCanonicalTairMempoolProviderUuid(uri.GetParam("provider_uuid"));
}

// A newly allocated PACE object must carry a stable route. Persisted objects
// created by an older version are allowed through the optional validator
// above and are reconciled using the durable owner service, but accepting a
// fresh allocation without this field would create new unrouteable GC debt.
inline bool HasRequiredTairMempoolProviderUuid(const DataStorageUri &uri) {
    return uri.HasParam("provider_uuid") && IsCanonicalTairMempoolProviderUuid(uri.GetParam("provider_uuid"));
}

// The allocation token is the per-object generation capability. Provider GA
// values may be reused within one process for explicit/local DRAM, so node,
// offset and process incarnation alone are insufficient to authorize GC.
inline bool HasCanonicalTairMempoolAllocationToken(const DataStorageUri &uri) {
    return uri.HasParam("allocation_token") && HasCanonicalKvMetaObjectKey(uri.GetParam("allocation_token"));
}

// Validate the backend fields that make one URI an independently deletable
// KVMeta allocation. Keep this rule in the data-storage layer so the manager,
// service serializer, and exact-size SDK cannot drift to different ownership
// interpretations. Authority, configured namespace, logical size, and exact
// metadata identity remain separate checks at their respective trust
// boundaries.
inline bool HasOwnedKvMetaAllocationShape(const DataStorageUri &uri, DataStorageType storage_type) {
    if (storage_type == DataStorageType::DATA_STORAGE_TYPE_MOONCAKE) {
        return uri.HasParam("key") && HasCanonicalKvMetaObjectKey(uri.GetParam("key"));
    }
    if (IsTairMempoolStorageType(storage_type)) {
        return HasExactTairMempoolAddress(uri) && HasCanonicalTairMempoolProviderIncarnation(uri) &&
               HasSafeOptionalTairMempoolProviderUuid(uri) && HasCanonicalTairMempoolAllocationToken(uri);
    }
    switch (storage_type) {
    case DataStorageType::DATA_STORAGE_TYPE_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_NFS:
    case DataStorageType::DATA_STORAGE_TYPE_DUMMY:
        break;
    case DataStorageType::DATA_STORAGE_TYPE_UNKNOWN:
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5:
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2:
    case DataStorageType::COUNT:
    default:
        return false;
    }
    if (!HasOwnedKvMetaFilePath(uri)) {
        return false;
    }
    if (!uri.HasParam("blkid")) {
        return true;
    }
    const std::string block_id_text = uri.GetParam("blkid");
    std::uint64_t block_id = 0;
    const auto parsed = std::from_chars(block_id_text.data(), block_id_text.data() + block_id_text.size(), block_id);
    return !block_id_text.empty() && parsed.ec == std::errc{} &&
           parsed.ptr == block_id_text.data() + block_id_text.size() && block_id == 0;
}

inline bool
TryGetOwnedKvMetaObjectKey(const DataStorageUri &uri, DataStorageType storage_type, std::string &object_key) {
    object_key.clear();
    if (storage_type == DataStorageType::DATA_STORAGE_TYPE_MOONCAKE) {
        if (!uri.HasParam("key")) {
            return false;
        }
        object_key = uri.GetParam("key");
    } else if (IsTairMempoolStorageType(storage_type)) {
        if (!uri.HasParam("allocation_token")) {
            return false;
        }
        object_key = uri.GetParam("allocation_token");
    } else {
        std::string_view key_view;
        if (!TryGetCanonicalKvMetaObjectKeyFromPath(uri.GetPath(), key_view)) {
            return false;
        }
        object_key.assign(key_view.data(), key_view.size());
    }
    if (!HasCanonicalKvMetaObjectKey(object_key)) {
        object_key.clear();
        return false;
    }
    return true;
}

inline bool KvMetaObjectKeyBelongsToInstance(std::string_view object_key,
                                             std::string_view expected_instance_path_hash) noexcept {
    constexpr std::string_view kPrefix = "kvmeta/";
    if (!HasCanonicalKvMetaObjectKey(object_key) || expected_instance_path_hash.empty()) {
        return false;
    }
    const std::size_t instance_end = object_key.find('/', kPrefix.size());
    return instance_end != std::string_view::npos &&
           object_key.substr(kPrefix.size(), instance_end - kPrefix.size()) == expected_instance_path_hash;
}

inline bool KvMetaUriBelongsToInstance(const DataStorageUri &uri,
                                       DataStorageType storage_type,
                                       std::string_view expected_instance_path_hash) {
    std::string object_key;
    return TryGetOwnedKvMetaObjectKey(uri, storage_type, object_key) &&
           KvMetaObjectKeyBelongsToInstance(object_key, expected_instance_path_hash);
}

inline bool KvMetaUriBelongsToObjectKey(const DataStorageUri &uri,
                                        DataStorageType storage_type,
                                        std::string_view expected_object_key_prefix) {
    if (expected_object_key_prefix.empty()) {
        return false;
    }
    std::string object_key;
    return TryGetOwnedKvMetaObjectKey(uri, storage_type, object_key) &&
           object_key.size() == expected_object_key_prefix.size() + kKvMetaObjectNonceBytes &&
           object_key.compare(0, expected_object_key_prefix.size(), expected_object_key_prefix) == 0;
}

// Physical-slot identity intentionally excludes the PACE allocation token.
// It detects a provider returning one address to two logical objects in the
// same operation; generation identity below adds the token for durable delete
// ownership. Length-prefix textual components to avoid delimiter aliases.
inline bool
GetKvMetaPhysicalAllocationIdentity(DataStorageType type, const DataStorageUri &uri, std::string &identity) {
    identity.clear();
    if (!HasCanonicalKvMetaAuthority(uri)) {
        return false;
    }
    const auto append_component = [&identity](std::string_view component) {
        identity.append(std::to_string(component.size()));
        identity.push_back(':');
        identity.append(component.data(), component.size());
    };
    identity.append(std::to_string(static_cast<int>(type)));
    identity.push_back('|');
    append_component(uri.GetHostName());
    switch (type) {
    case DataStorageType::DATA_STORAGE_TYPE_MOONCAKE:
        if (!uri.HasParam("key") || uri.GetParam("key").empty()) {
            return false;
        }
        append_component(uri.GetParam("key"));
        return true;
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL:
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD: {
        std::uint64_t offset = 0;
        if (!HasOwnedKvMetaAllocationShape(uri, type) || !TryGetExactTairMempoolOffset(uri, offset)) {
            return false;
        }
        std::uint16_t node = 0;
        std::uint16_t media = 0;
        std::uint16_t range = 0;
        if (!TryGetTairMempoolUint16ParamOrDefault(uri, "node_id", node) ||
            !TryGetTairMempoolUint16ParamOrDefault(uri, "media_type", media) ||
            !TryGetTairMempoolUint16ParamOrDefault(uri, "range_id", range)) {
            return false;
        }
        identity.push_back('|');
        identity.append(std::to_string(offset));
        identity.push_back('|');
        identity.append(std::to_string(node));
        identity.push_back('|');
        identity.append(std::to_string(media));
        identity.push_back('|');
        identity.append(std::to_string(range));
        identity.push_back('|');
        identity.append(uri.GetParam("provider_incarnation"));
        // Stable provider routing is part of a physical address when present.
        // Legacy persisted locations may omit it, while every fresh allocation
        // is validated by HasRequiredTairMempoolProviderUuid before admission.
        append_component(uri.GetParam("provider_uuid"));
        return true;
    }
    case DataStorageType::DATA_STORAGE_TYPE_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_NFS:
    case DataStorageType::DATA_STORAGE_TYPE_DUMMY:
        if (!HasOwnedKvMetaFilePath(uri)) {
            return false;
        }
        append_component(uri.GetPath());
        return true;
    case DataStorageType::DATA_STORAGE_TYPE_UNKNOWN:
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5:
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2:
    case DataStorageType::COUNT:
    default:
        return false;
    }
}

inline bool
GetKvMetaPhysicalAllocationGenerationIdentity(DataStorageType type, const DataStorageUri &uri, std::string &identity) {
    if (!GetKvMetaPhysicalAllocationIdentity(type, uri, identity)) {
        return false;
    }
    if (IsTairMempoolStorageType(type)) {
        if (!HasCanonicalTairMempoolAllocationToken(uri)) {
            identity.clear();
            return false;
        }
        const std::string token = uri.GetParam("allocation_token");
        identity.push_back('|');
        identity.append(std::to_string(token.size()));
        identity.push_back(':');
        identity.append(token);
    }
    return true;
}

inline bool
BuildConfiguredKvMetaObjectPath(const StorageConfig &config, std::string_view object_key, std::string &object_path) {
    object_path.clear();
    if (!HasCanonicalKvMetaObjectKey(object_key)) {
        return false;
    }
    const std::string key(object_key);
    switch (config.type()) {
    case DataStorageType::DATA_STORAGE_TYPE_NFS: {
        const auto spec = std::dynamic_pointer_cast<NfsStorageSpec>(config.storage_spec());
        if (!spec || !HasCanonicalKvMetaNfsRootPath(spec->root_path())) {
            return false;
        }
        // NfsBackend uses raw concatenation rather than filesystem::path.
        object_path = spec->root_path() + key;
        return true;
    }
    case DataStorageType::DATA_STORAGE_TYPE_HF3FS: {
        const auto spec = std::dynamic_pointer_cast<ThreeFSStorageSpec>(config.storage_spec());
        if (!spec) {
            return false;
        }
        object_path = (std::filesystem::path(spec->mountpoint()) / spec->root_dir() / key).string();
        return true;
    }
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS: {
        const auto spec = std::dynamic_pointer_cast<VcnsThreeFSStorageSpec>(config.storage_spec());
        if (!spec) {
            return false;
        }
        object_path = (std::filesystem::path(spec->mountpoint()) / spec->root_dir() / key).string();
        return true;
    }
    case DataStorageType::DATA_STORAGE_TYPE_DUMMY: {
        const auto spec = std::dynamic_pointer_cast<DummyStorageSpec>(config.storage_spec());
        if (!spec) {
            return false;
        }
        const std::filesystem::path root = spec->root_path() == "memory://" ? std::filesystem::path("/memory")
                                                                            : std::filesystem::path(spec->root_path());
        object_path = (root / key).string();
        return true;
    }
    case DataStorageType::DATA_STORAGE_TYPE_UNKNOWN:
    case DataStorageType::DATA_STORAGE_TYPE_MOONCAKE:
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL:
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5:
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2:
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD:
    case DataStorageType::COUNT:
    default:
        return false;
    }
}

inline bool UriMatchesConfiguredKvMetaNamespace(const DataStorageUri &uri,
                                                DataStorageType storage_type,
                                                const StorageConfig &config) {
    if (config.type() != storage_type || config.global_unique_name() != uri.GetHostName()) {
        return false;
    }
    if (storage_type == DataStorageType::DATA_STORAGE_TYPE_MOONCAKE) {
        return uri.HasParam("key") && HasCanonicalKvMetaObjectKey(uri.GetParam("key"));
    }
    if (IsTairMempoolStorageType(storage_type)) {
        // PACE addresses are allocator-owned opaque offsets, but media_type
        // is part of their physical pool identity.  Missing legacy URI fields
        // decode to zero and therefore match only an explicitly unspecified
        // backend; an SSD/DRAM backend must return its configured media.
        const auto spec = std::dynamic_pointer_cast<TairMemPoolStorageSpec>(config.storage_spec());
        std::uint16_t uri_media_type = 0;
        return spec && TryGetTairMempoolUint16ParamOrDefault(uri, "media_type", uri_media_type) &&
               uri_media_type == spec->media_type();
    }
    std::string_view object_key;
    std::string expected_path;
    return TryGetCanonicalKvMetaObjectKeyFromPath(uri.GetPath(), object_key) &&
           BuildConfiguredKvMetaObjectPath(config, object_key, expected_path) && uri.GetPath() == expected_path;
}

inline bool HasSafeConfiguredKvMetaNamespace(const StorageConfig &config,
                                             std::size_t max_uri_bytes = kMaxKvMetaLocationUriBytes) {
    if (max_uri_bytes == 0 || max_uri_bytes > kMaxKvMetaLocationUriBytes ||
        !IsCanonicalKvMetaBackendName(config.global_unique_name())) {
        return false;
    }

    // Exercise the longest URI that KVCM itself can generate rather than a
    // short happy-path sample. This makes registration fail before a backend
    // allocation when a long configured root fits the sample but not the
    // instance hash, 256-bit key fingerprint, size, or singleton blkid in a
    // real location.
    const std::string object_key = "kvmeta/" + std::string(kKvMetaCanonicalUint64MaxHexChars, 'f') + "/" +
                                   std::string(kKvMetaObjectFingerprintHexChars, 'f') + "/" +
                                   std::string(kKvMetaObjectNonceBytes, 'a');
    if (object_key.size() != kKvMetaMaxPhysicalObjectKeyBytes) {
        return false;
    }
    const std::string max_uint64 = std::to_string(std::numeric_limits<std::uint64_t>::max());

    DataStorageUri sample;
    sample.SetHostName(config.global_unique_name());
    switch (config.type()) {
    case DataStorageType::DATA_STORAGE_TYPE_MOONCAKE:
        if (!std::dynamic_pointer_cast<MooncakeStorageSpec>(config.storage_spec())) {
            return false;
        }
        sample.SetProtocol(ToString(config.type()));
        sample.SetPath("/");
        sample.SetParam("key", object_key);
        break;
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL:
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD: {
        const auto spec = std::dynamic_pointer_cast<TairMemPoolStorageSpec>(config.storage_spec());
        if (!spec || (config.type() == DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD &&
                      spec->media_type() != kTairMemPoolMediaTypeSsd)) {
            return false;
        }
        sample.SetProtocol(kTairMempoolUriScheme);
        sample.SetPath("/" + max_uint64);
        sample.SetParam("node_id", std::to_string(std::numeric_limits<std::uint16_t>::max()));
        sample.SetParam("provider_uuid", "ffffffff-ffff-4fff-bfff-ffffffffffff");
        sample.SetParam("media_type", std::to_string(spec->media_type()));
        sample.SetParam("allocation_token", object_key);
        sample.SetParam("provider_incarnation", "ffffffff-ffff-4fff-bfff-ffffffffffff");
        sample.SetParam("range_id", std::to_string(std::numeric_limits<std::uint16_t>::max()));
        break;
    }
    case DataStorageType::DATA_STORAGE_TYPE_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_NFS:
    case DataStorageType::DATA_STORAGE_TYPE_DUMMY: {
        std::string object_path;
        if (!BuildConfiguredKvMetaObjectPath(config, object_key, object_path)) {
            return false;
        }
        sample.SetProtocol(ToString(config.type()));
        sample.SetPath(object_path);
        // Existing file providers include blkid=0 whenever their configured
        // pack size is greater than one, even for a singleton Create call.
        bool includes_block_id = false;
        if (config.type() == DataStorageType::DATA_STORAGE_TYPE_HF3FS) {
            includes_block_id =
                std::dynamic_pointer_cast<ThreeFSStorageSpec>(config.storage_spec())->key_count_per_file() > 1;
        } else if (config.type() == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS) {
            includes_block_id =
                std::dynamic_pointer_cast<VcnsThreeFSStorageSpec>(config.storage_spec())->key_count_per_file() > 1;
        } else if (config.type() == DataStorageType::DATA_STORAGE_TYPE_NFS) {
            includes_block_id =
                std::dynamic_pointer_cast<NfsStorageSpec>(config.storage_spec())->key_count_per_file() > 1;
        } else {
            includes_block_id =
                std::dynamic_pointer_cast<DummyStorageSpec>(config.storage_spec())->key_count_per_file() > 1;
        }
        if (includes_block_id) {
            sample.SetParam("blkid", "0");
        }
        break;
    }
    case DataStorageType::DATA_STORAGE_TYPE_UNKNOWN:
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5:
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2:
    case DataStorageType::COUNT:
    default:
        return false;
    }

    sample.SetParam("size", max_uint64);
    const std::string uri_text = sample.ToUriString();
    if (uri_text.size() > max_uri_bytes || !HasUnambiguousKvMetaUriText(uri_text)) {
        return false;
    }
    const DataStorageUri parsed(uri_text);
    if (!HasCanonicalKvMetaAuthority(parsed) || !UriMatchesConfiguredKvMetaNamespace(parsed, config.type(), config)) {
        return false;
    }
    if (config.type() == DataStorageType::DATA_STORAGE_TYPE_MOONCAKE) {
        return parsed.GetPath() == "/" && parsed.GetParam("key") == object_key;
    }
    if (IsTairMempoolStorageType(config.type())) {
        return HasOwnedKvMetaAllocationShape(parsed, config.type());
    }
    return HasOwnedKvMetaFilePath(parsed);
}

} // namespace kv_cache_manager
