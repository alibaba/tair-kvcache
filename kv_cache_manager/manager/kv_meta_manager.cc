#include "kv_cache_manager/manager/kv_meta_manager.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/hash/hash.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/kv_meta_uri.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/data_storage_selector.h"
#include "kv_cache_manager/manager/kv_meta_instance.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"

namespace kv_cache_manager {

namespace {

constexpr std::uint64_t kObjectKeyHashSeed = 0x8bc5'1f2d'671a'94e3ULL;
constexpr std::uint64_t kInstancePathHashSeed = 0x6e91'ca34'0bd7'52f8ULL;
constexpr std::size_t kRecoveryScanBatchSize = 1000;
// Keep every side-path physical cleanup request bounded. In particular, the
// deployed PACE delete API encodes addresses in a URL query, so inheriting an
// arbitrarily large fixed-block CacheReclaimer batch would exceed practical
// request-line limits. This limit does not alter the fixed-block reclaimer.
constexpr std::size_t kKvMetaDeleteBatchSize = 256;
constexpr std::int64_t kMicrosecondsPerSecond = 1'000'000;
constexpr std::int64_t kLeaseDeadlineTag = std::int64_t{1} << 62;
// The first retirement CAS must stop new readers before a finite grace
// deadline is chosen.  If leadership changes between the two transitions,
// recovery treats this tagged far-future value conservatively (subject to its
// bounded force deadline) instead of deleting an object whose last reader may
// only just have obtained the URI.
constexpr std::int64_t kRetirementFenceDeadline = std::numeric_limits<std::int64_t>::max();
constexpr auto kRecoveryWaitPollInterval = std::chrono::milliseconds(100);

bool EncodeLeaseDeadline(std::int64_t now_us, std::int64_t timeout_seconds, std::int64_t &encoded_deadline) {
    encoded_deadline = 0;
    if (now_us <= 0 || timeout_seconds <= 0 || now_us >= kLeaseDeadlineTag ||
        timeout_seconds > (kLeaseDeadlineTag - 1 - now_us) / kMicrosecondsPerSecond) {
        return false;
    }
    encoded_deadline = kLeaseDeadlineTag + now_us + timeout_seconds * kMicrosecondsPerSecond;
    return true;
}

bool EncodeTaggedDeadlineUs(std::int64_t now_us, std::int64_t timeout_us, std::int64_t &encoded_deadline) {
    encoded_deadline = 0;
    if (now_us <= 0 || timeout_us < 0 || now_us >= kLeaseDeadlineTag || timeout_us > kLeaseDeadlineTag - 1 - now_us) {
        return false;
    }
    encoded_deadline = kLeaseDeadlineTag + now_us + timeout_us;
    return true;
}

bool DecodeLeaseDeadline(std::int64_t encoded_deadline, std::int64_t &deadline_us) {
    deadline_us = 0;
    if (encoded_deadline <= kLeaseDeadlineTag) {
        return false;
    }
    deadline_us = encoded_deadline - kLeaseDeadlineTag;
    return deadline_us > 0 && deadline_us < kLeaseDeadlineTag;
}

bool DecodeRecoveryLeaseDeadline(std::int64_t marker,
                                 std::int64_t max_write_timeout_seconds,
                                 std::int64_t &deadline_us) {
    if (DecodeLeaseDeadline(marker, deadline_us)) {
        return true;
    }
    // Rolling-upgrade compatibility: the first KVMeta implementation stored
    // the wall-clock allocation time directly as its positive active marker.
    // It did not persist the caller's timeout, so use the configured maximum
    // to avoid reclaiming an old leader's still-valid allocation early. The
    // recovery-wide steady-clock force deadline remains the hard upper bound
    // for corrupt or far-future values.
    deadline_us = 0;
    if (marker <= 0 || marker >= kLeaseDeadlineTag || max_write_timeout_seconds <= 0) {
        return false;
    }
    if (max_write_timeout_seconds > (kLeaseDeadlineTag - 1 - marker) / kMicrosecondsPerSecond) {
        deadline_us = kLeaseDeadlineTag - 1;
    } else {
        deadline_us = marker + max_write_timeout_seconds * kMicrosecondsPerSecond;
    }
    return true;
}

std::string HexEncode(std::string_view input) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string output;
    output.resize(input.size() * 2);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto value = static_cast<unsigned char>(input[i]);
        output[2 * i] = kHex[value >> 4];
        output[2 * i + 1] = kHex[value & 0x0f];
    }
    return output;
}

bool HexDecode(std::string_view input, std::string &output) {
    if (input.empty() || (input.size() & 1U) != 0) {
        return false;
    }
    const auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        return -1;
    };
    output.resize(input.size() / 2);
    for (std::size_t i = 0; i < output.size(); ++i) {
        const int high = nibble(input[2 * i]);
        const int low = nibble(input[2 * i + 1]);
        if (high < 0 || low < 0) {
            output.clear();
            return false;
        }
        output[i] = static_cast<char>((high << 4) | low);
    }
    return true;
}

void AddError(RequestContext *request_context, const std::string &message) {
    if (request_context && request_context->error_tracer()) {
        request_context->error_tracer()->AddErrorMsg(message);
    }
}

ErrorCode FirstHardError(ErrorCode current, ErrorCode candidate) {
    if (current != EC_OK) {
        return current;
    }
    return candidate == EC_OK || candidate == EC_NOENT ? EC_OK : candidate;
}

bool IsKnownPreAllocationKvMetaCreateFailure(ErrorCode ec) noexcept {
    // These codes are the KVMeta side-interface contract for a rejection that
    // occurred before an allocation was created. EC_OUTCOME_UNKNOWN and all
    // other errors remain fail-closed because an unreported object may exist.
    switch (ec) {
    case EC_NOSPC:
    case EC_NOENT:
    case EC_BADARGS:
    case EC_UNIMPLEMENTED:
    case EC_CONFIG_ERROR:
    case EC_OUT_OF_LIMIT:
    case EC_CORRUPTION:
        return true;
    default:
        return false;
    }
}

std::vector<std::vector<std::size_t>> MakeUniqueKeyLayers(const std::vector<std::int64_t> &keys) {
    std::vector<std::vector<std::size_t>> layers;
    std::vector<std::unordered_set<std::int64_t>> layer_keys;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        std::size_t layer = 0;
        for (; layer < layer_keys.size(); ++layer) {
            if (layer_keys[layer].insert(keys[i]).second) {
                layers[layer].push_back(i);
                break;
            }
        }
        if (layer == layer_keys.size()) {
            layer_keys.emplace_back();
            layer_keys.back().insert(keys[i]);
            layers.push_back({i});
        }
    }
    return layers;
}

bool ReadLogicalSize(const CacheLocation &location, std::uint64_t &out_size) {
    out_size = 0;
    if (location.spec_size() != 1 || location.location_specs().size() != 1 ||
        location.location_specs().front().name() != kKvMetaValueSpecName) {
        return false;
    }
    const std::string &uri_text = location.location_specs().front().uri();
    if (uri_text.size() > kMaxKvMetaLocationUriBytes || !HasUnambiguousKvMetaUriText(uri_text)) {
        return false;
    }

    const DataStorageUri uri(uri_text);
    if (!uri.Valid() || uri.GetHostName().empty() || !uri.HasParam("size")) {
        return false;
    }
    const std::string size_text = uri.GetParam("size");
    std::uint64_t size = 0;
    const auto parsed = std::from_chars(size_text.data(), size_text.data() + size_text.size(), size);
    if (size_text.empty() || parsed.ec != std::errc{} || parsed.ptr != size_text.data() + size_text.size() ||
        size == 0 || size_text != std::to_string(size)) {
        return false;
    }
    out_size = size;
    return true;
}

bool HasSupportedKvMetaReclaimConfiguration(const InstanceGroup &group,
                                            std::int64_t max_write_timeout_seconds) noexcept {
    const auto cache_config = group.cache_config();
    const auto strategy = cache_config ? cache_config->reclaim_strategy() : nullptr;
    if (!strategy || strategy->reclaim_policy() != ReclaimPolicy::POLICY_LRU) {
        return false;
    }
    const double threshold = strategy->trigger_strategy().used_percentage();
    const std::int64_t max_delete_delay_ms = max_write_timeout_seconds > std::numeric_limits<std::int64_t>::max() / 1000
                                                 ? std::numeric_limits<std::int64_t>::max()
                                                 : max_write_timeout_seconds * 1000;
    return std::isfinite(threshold) && threshold >= 0.0 && threshold <= 1.0 &&
           strategy->delay_before_delete_ms() >= 0 && strategy->delay_before_delete_ms() <= max_delete_delay_ms;
}

bool ParseSupportedCachedKvMetaBackendTypes(const MetaStorageBackendConfig &backend_config,
                                            std::string &persistent_type) noexcept {
    persistent_type.clear();
    const std::string &storage_uri = backend_config.GetStorageUri();
    // Cached mode defaults to Redis + local, but Redis itself rejects an empty
    // URI. Do not certify a configuration that the real backend cannot open.
    if (storage_uri.empty()) {
        return false;
    }
    try {
        const StandardUri uri = StandardUri::FromUri(storage_uri);
        if (!uri.Valid()) {
            return false;
        }
        persistent_type = uri.GetParam("persistent_type");
        if (persistent_type.empty()) {
            persistent_type = META_REDIS_BACKEND_TYPE_STR;
        }
        std::string cache_type = uri.GetParam("cache_type");
        if (cache_type.empty()) {
            cache_type = META_LOCAL_BACKEND_TYPE_STR;
        }
        // Keep this list aligned with MetaStorageBackendFactory. Only the
        // local cache backend supplies no-touch maintenance reads plus normal
        // access-time refresh; dummy is a supported persistent test double.
        const bool supported_persistent = persistent_type == META_REDIS_BACKEND_TYPE_STR ||
                                          persistent_type == META_ASYNC_REDIS_BACKEND_TYPE_STR ||
                                          persistent_type == META_DUMMY_BACKEND_TYPE_STR;
        return cache_type == META_LOCAL_BACKEND_TYPE_STR && supported_persistent;
    } catch (const std::exception &) {
        // Configuration validation must fail closed rather than allowing a
        // malformed/provider-controlled URI to escape into service startup.
        persistent_type.clear();
        return false;
    } catch (...) {
        persistent_type.clear();
        return false;
    }
}

bool HasSupportedKvMetaReadHeatTracking(const InstanceGroup &group) noexcept {
    const auto cache_config = group.cache_config();
    const auto indexer_config = cache_config ? cache_config->meta_indexer_config() : nullptr;
    const auto backend_config = indexer_config ? indexer_config->GetMetaStorageBackendConfig() : nullptr;
    if (!backend_config) {
        return false;
    }
    const std::string &type = backend_config->GetStorageType();
    if (type == META_LOCAL_BACKEND_TYPE_STR || type == META_DUMMY_BACKEND_TYPE_STR) {
        return true;
    }
    // Direct Redis reads do not refresh PROPERTY_LRU_TIME. Admitting such a
    // group under POLICY_LRU would silently turn equal/zero timestamps into a
    // sampled key-order eviction policy. Cached mode is valid only when its
    // actual hot layer is local and both configured factories are supported.
    // Keep this restriction KVMeta-only; ordinary fixed-block instances retain
    // their existing backend choices.
    std::string persistent_type;
    return type == META_CACHED_BACKEND_TYPE_STR &&
           ParseSupportedCachedKvMetaBackendTypes(*backend_config, persistent_type);
}

bool HasCrashRecoverableKvMetaMetadata(const InstanceGroup &group) noexcept {
    const auto cache_config = group.cache_config();
    const auto indexer_config = cache_config ? cache_config->meta_indexer_config() : nullptr;
    const auto backend_config = indexer_config ? indexer_config->GetMetaStorageBackendConfig() : nullptr;
    if (!backend_config || backend_config->GetStorageType() != META_CACHED_BACKEND_TYPE_STR) {
        return false;
    }

    // A dummy persistent layer is useful in tests, but neither it nor a
    // process-local single backend can recover allocation ownership after a
    // KVCM restart.
    std::string persistent_type;
    if (!ParseSupportedCachedKvMetaBackendTypes(*backend_config, persistent_type)) {
        return false;
    }
    return persistent_type == META_REDIS_BACKEND_TYPE_STR || persistent_type == META_ASYNC_REDIS_BACKEND_TYPE_STR;
}

bool UriMatchesStorageBackend(const DataStorageUri &uri,
                              const std::string &storage_name,
                              DataStorageType storage_type) {
    if (!HasCanonicalKvMetaAuthority(uri) || uri.GetHostName() != storage_name) {
        return false;
    }
    if (IsTairMempoolStorageType(storage_type)) {
        return uri.GetProtocol() == kTairMempoolUriScheme;
    }
    const DataStorageType uri_type = ToDataStorageType(uri.GetProtocol());
    return uri_type != DataStorageType::DATA_STORAGE_TYPE_UNKNOWN && ToBaseType(uri_type) == ToBaseType(storage_type);
}

bool UriNamesCreatedKvMetaObject(const DataStorageUri &uri,
                                 const std::shared_ptr<DataStorageBackend> &backend,
                                 DataStorageType storage_type,
                                 std::string_view object_key) {
    if (!backend || object_key.empty()) {
        return false;
    }
    const StorageConfig &config = backend->GetStorageConfig();
    if (backend->GetType() != storage_type || config.type() != storage_type ||
        config.global_unique_name() != uri.GetHostName() ||
        !UriMatchesConfiguredKvMetaNamespace(uri, storage_type, config)) {
        return false;
    }
    if (storage_type == DataStorageType::DATA_STORAGE_TYPE_MOONCAKE) {
        return uri.GetParam("key") == object_key;
    }
    if (IsTairMempoolStorageType(storage_type)) {
        // The allocator returns an opaque physical address; current PACE URIs
        // do not carry the logical allocation key. Address shape and exact
        // operation cardinality are the strongest V1 ownership evidence.
        return true;
    }
    switch (storage_type) {
    case DataStorageType::DATA_STORAGE_TYPE_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_NFS:
    case DataStorageType::DATA_STORAGE_TYPE_DUMMY: {
        std::string expected_path;
        return BuildConfiguredKvMetaObjectPath(config, object_key, expected_path) && uri.GetPath() == expected_path;
    }
    case DataStorageType::DATA_STORAGE_TYPE_UNKNOWN:
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5:
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2:
    case DataStorageType::COUNT:
    default:
        return false;
    }
}

bool UriBelongsToKvMetaNamespace(const DataStorageUri &uri,
                                 const std::shared_ptr<DataStorageBackend> &backend,
                                 DataStorageType storage_type,
                                 const std::string &internal_instance_id,
                                 std::int64_t internal_key) {
    if (!backend) {
        return false;
    }
    const StorageConfig &config = backend->GetStorageConfig();
    if (backend->GetType() != storage_type || config.type() != storage_type ||
        config.global_unique_name() != uri.GetHostName() ||
        !UriMatchesConfiguredKvMetaNamespace(uri, storage_type, config)) {
        return false;
    }
    if (IsTairMempoolStorageType(storage_type)) {
        return true;
    }
    const std::uint64_t instance_path_hash =
        Hash64(internal_instance_id.data(), internal_instance_id.size(), kInstancePathHashSeed);
    const std::string object_key_prefix = "kvmeta/" + StringUtil::Uint64ToHex(instance_path_hash) + "/" +
                                          StringUtil::Uint64ToHex(static_cast<std::uint64_t>(internal_key)) + "/";
    const auto get_expected_key =
        [&](std::string_view value, bool require_leading_slash, std::string_view &object_key) {
            object_key = {};
            const std::size_t expected_size = object_key_prefix.size() + kKvMetaObjectNonceBytes;
            if (value.size() < expected_size) {
                return false;
            }
            const std::size_t key_begin = value.size() - expected_size;
            if ((require_leading_slash && (key_begin == 0 || value[key_begin - 1] != '/')) ||
                value.compare(key_begin, object_key_prefix.size(), object_key_prefix) != 0) {
                return false;
            }
            if (value.substr(key_begin + object_key_prefix.size()).size() != kKvMetaObjectNonceBytes ||
                !HasCanonicalKvMetaObjectKey(value.substr(key_begin, expected_size))) {
                return false;
            }
            object_key = value.substr(key_begin, expected_size);
            return true;
        };
    if (storage_type == DataStorageType::DATA_STORAGE_TYPE_MOONCAKE) {
        const std::string &physical_key = uri.GetParam("key");
        std::string_view object_key;
        return get_expected_key(physical_key, false, object_key) && object_key.size() == physical_key.size() &&
               uri.GetParam("key") == object_key;
    }
    switch (storage_type) {
    case DataStorageType::DATA_STORAGE_TYPE_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_NFS:
    case DataStorageType::DATA_STORAGE_TYPE_DUMMY: {
        std::string_view object_key;
        std::string expected_path;
        return get_expected_key(uri.GetPath(), true, object_key) &&
               BuildConfiguredKvMetaObjectPath(config, object_key, expected_path) && uri.GetPath() == expected_path;
    }
    case DataStorageType::DATA_STORAGE_TYPE_UNKNOWN:
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5:
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2:
    case DataStorageType::COUNT:
    default:
        return false;
    }
}

bool HasMatchingStorageBackend(const CacheLocation &location,
                               const std::shared_ptr<DataStorageManager> &data_storage_manager) {
    if (!data_storage_manager || !IsKvMetaObjectStorageType(location.type()) || location.location_specs().size() != 1 ||
        location.location_specs().front().uri().size() > kMaxKvMetaLocationUriBytes ||
        !HasUnambiguousKvMetaUriText(location.location_specs().front().uri())) {
        return false;
    }
    const DataStorageUri uri(location.location_specs().front().uri());
    if (!HasCanonicalKvMetaAuthority(uri)) {
        return false;
    }
    const auto backend = data_storage_manager->GetDataStorageBackend(uri.GetHostName());
    if (!backend || backend->GetType() != location.type()) {
        return false;
    }
    const StorageConfig &config = backend->GetStorageConfig();
    return config.type() == location.type() && config.global_unique_name() == uri.GetHostName() &&
           UriMatchesStorageBackend(uri, uri.GetHostName(), location.type()) &&
           HasOwnedKvMetaAllocationShape(uri, location.type()) &&
           UriMatchesConfiguredKvMetaNamespace(uri, location.type(), config);
}

bool ToValueLocation(const CacheLocation &location, KvMetaManager::ValueLocation &out) {
    std::uint64_t size = 0;
    if (!ReadLogicalSize(location, size)) {
        return false;
    }
    out = {};
    out.type = location.type();
    out.value_size = size;
    out.specs.reserve(location.location_specs().size());
    for (const auto &spec : location.location_specs()) {
        out.specs.emplace_back(spec.name(), spec.uri());
    }
    return true;
}

bool IsCommittedObject(const CacheLocation &location) {
    // CLS_NEW keeps this path invisible to the KV-cache reclaimer/migration
    // machinery. A negative create_time is the KVMeta-private commit marker;
    // an in-flight allocation always has a positive wall-clock timestamp.
    return location.status() == CLS_NEW && location.create_time() < 0;
}

bool IsRetiredObject(const CacheLocation &location) {
    return location.status() == CLS_DELETING && location.create_time() > kLeaseDeadlineTag;
}

bool GetPhysicalAllocationIdentity(DataStorageType type, const DataStorageUri &uri, std::string &identity) {
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
        // Mooncake Delete addresses an object only by its key; path and size
        // are not part of physical identity.
        if (uri.GetParam("key").empty()) {
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
        // The deployed PACE delete API names an allocation by GA only. Query
        // fields are validated routing/data-plane metadata, not a generation
        // capability and therefore must not make the same GA look distinct.
        identity.push_back('|');
        identity.append(std::to_string(offset));
        return true;
    }
    case DataStorageType::DATA_STORAGE_TYPE_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_NFS:
    case DataStorageType::DATA_STORAGE_TYPE_DUMMY:
        // These backends delete the file/object selected by path. KVMeta only
        // accepts singleton blkid=0, so query size is metadata, not identity.
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

bool GetPhysicalAllocationIdentity(const CacheLocation &location, std::string &identity) {
    if (location.location_specs().size() != 1) {
        identity.clear();
        return false;
    }
    return GetPhysicalAllocationIdentity(
        location.type(), DataStorageUri(location.location_specs().front().uri()), identity);
}

bool SamePhysicalAllocation(const CacheLocation &lhs, const CacheLocation &rhs) {
    std::string lhs_identity;
    std::string rhs_identity;
    return GetPhysicalAllocationIdentity(lhs, lhs_identity) && GetPhysicalAllocationIdentity(rhs, rhs_identity) &&
           lhs_identity == rhs_identity;
}

std::shared_ptr<CacheLocation> MakeCommittedLocation(const CacheLocation &location) {
    auto committed = std::make_shared<CacheLocation>(location);
    const std::int64_t create_time = location.create_time();
    committed->set_create_time(create_time > 0 ? -create_time : -1);
    return committed;
}

} // namespace

struct KvMetaManager::SessionItem {
    std::size_t request_index = 0;
    std::string original_key;
    std::int64_t internal_key = 0;
    std::string location_id;
    // Exact value expected by metadata CAS/delete.
    CacheLocationConstPtr metadata_location;
    // Physical allocation owned by this metadata value. It can differ from
    // metadata_location while reconciling a failed start operation.
    CacheLocationConstPtr data_location;
    std::uint64_t value_size = 0;
};

struct KvMetaManager::ExactLocation {
    std::int64_t internal_key = 0;
    std::string location_id;
    ErrorCode ec = EC_UNKNOWN;
    CacheLocationConstPtr location;
};

class KvMetaWriteSessionManager {
public:
    using Clock = std::chrono::steady_clock;

    enum class TakeResult {
        kOk,
        kNotFound,
        kInstanceMismatch,
        kSizeMismatch,
        kExpired,
        kDeferred,
        kExpiredDeferred,
        kAborted
    };
    enum class PutResult {
        kOk,
        kDuplicate,
        kStopped,
        kFull,
        kExpired
    };

    struct Session {
        std::string internal_instance_id;
        std::size_t quota_shard = 0;
        Clock::time_point commit_deadline;
        Clock::time_point cleanup_deadline;
        bool defer_failed_cleanup = false;
        std::vector<KvMetaManager::SessionItem> items;
    };

    struct FinalizationState {
        std::mutex mutex;
        std::unordered_map<std::string, std::size_t> instances;
    };

    class FinalizationGuard {
    public:
        FinalizationGuard() = default;
        FinalizationGuard(std::shared_ptr<FinalizationState> state, std::string internal_instance_id)
            : state_(std::move(state)), internal_instance_id_(std::move(internal_instance_id)) {}
        ~FinalizationGuard() { Reset(); }

        FinalizationGuard(const FinalizationGuard &) = delete;
        FinalizationGuard &operator=(const FinalizationGuard &) = delete;
        FinalizationGuard(FinalizationGuard &&other) noexcept = default;
        FinalizationGuard &operator=(FinalizationGuard &&other) noexcept {
            if (this != &other) {
                Reset();
                state_ = std::move(other.state_);
                internal_instance_id_ = std::move(other.internal_instance_id_);
            }
            return *this;
        }

    private:
        void Reset() {
            // Keep an owning reference alive until after the mutex has been
            // unlocked. A guard is deliberately allowed to outlive the
            // session manager during shutdown; resetting the last shared_ptr
            // while lock_guard still referred to state_->mutex would destroy
            // a locked mutex and make the subsequent unlock undefined.
            auto state = std::move(state_);
            if (state) {
                std::lock_guard<std::mutex> lock(state->mutex);
                const auto it = state->instances.find(internal_instance_id_);
                if (it != state->instances.end()) {
                    if (it->second <= 1) {
                        state->instances.erase(it);
                    } else {
                        --it->second;
                    }
                }
            }
        }

        std::shared_ptr<FinalizationState> state_;
        std::string internal_instance_id_;
    };

    KvMetaWriteSessionManager(KvMetaManager *owner, std::size_t max_sessions)
        : owner_(owner), max_sessions_(max_sessions) {}
    ~KvMetaWriteSessionManager() { StopAndDiscard(); }

    bool Start() {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        if (thread_.joinable()) {
            return !stopping_;
        }
        stopping_ = false;
        try {
            thread_ = std::thread([this]() { ExpireLoop(); });
        } catch (const std::exception &e) {
            stopping_ = true;
            KVCM_LOG_ERROR("failed to start KVMeta write-session expiry worker: %s", e.what());
            return false;
        }
        return true;
    }

    PutResult Put(const std::string &session_id,
                  const std::string &internal_instance_id,
                  std::size_t quota_shard,
                  std::vector<KvMetaManager::SessionItem> items,
                  Clock::time_point commit_deadline,
                  Clock::time_point cleanup_deadline,
                  bool defer_failed_cleanup) {
        auto entry = std::make_shared<Entry>();
        entry->session_id = session_id;
        entry->deadline = cleanup_deadline;
        entry->session.internal_instance_id = internal_instance_id;
        entry->session.quota_shard = quota_shard;
        entry->session.commit_deadline = commit_deadline;
        entry->session.cleanup_deadline = cleanup_deadline;
        entry->session.defer_failed_cleanup = defer_failed_cleanup;
        entry->session.items = std::move(items);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return PutResult::kStopped;
            }
            if (commit_deadline <= Clock::now() || cleanup_deadline < commit_deadline) {
                return PutResult::kExpired;
            }
            if (sessions_.size() >= max_sessions_) {
                return PutResult::kFull;
            }
            if (sessions_.find(session_id) != sessions_.end()) {
                return PutResult::kDuplicate;
            }
            entry->sequence = next_sequence_++;
            auto [session_it, session_inserted] = sessions_.emplace(session_id, entry);
            if (!session_inserted) {
                return PutResult::kDuplicate;
            }
            try {
                const bool deadline_inserted =
                    deadlines_.emplace(DeadlineKey{entry->deadline, entry->sequence}, entry).second;
                if (!deadline_inserted) {
                    sessions_.erase(session_it);
                    return PutResult::kDuplicate;
                }
            } catch (...) {
                // Keep publication atomic across the lookup and deadline
                // indexes. Otherwise an allocation failure here leaves a
                // session that clients can consume but the expiry worker can
                // never discover.
                sessions_.erase(session_it);
                throw;
            }
        }
        condition_.notify_all();
        return PutResult::kOk;
    }

    PutResult Availability() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            return PutResult::kStopped;
        }
        return sessions_.size() < max_sessions_ ? PutResult::kOk : PutResult::kFull;
    }

    // Close admission and wake the expiry loop without joining it. Server
    // demotion uses this first so the existing KV-cache drain/GC/migration
    // sequence never waits behind KVMeta backend I/O. StopAndDiscard performs
    // the eventual join before CacheManager teardown.
    void RequestStop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
    }

    std::pair<TakeResult, FinalizationGuard> Take(const std::string &session_id,
                                                  const std::string &internal_instance_id,
                                                  std::optional<std::size_t> expected_size,
                                                  bool all_success,
                                                  Session &out) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            return {TakeResult::kNotFound, FinalizationGuard{}};
        }
        const auto &entry = it->second;
        if (entry->session.internal_instance_id != internal_instance_id) {
            return {TakeResult::kInstanceMismatch, FinalizationGuard{}};
        }
        if (expected_size && entry->session.items.size() != *expected_size) {
            return {TakeResult::kSizeMismatch, FinalizationGuard{}};
        }
        if (entry->aborted) {
            return {all_success ? TakeResult::kAborted : TakeResult::kDeferred, FinalizationGuard{}};
        }
        const bool expired = entry->session.commit_deadline <= Clock::now();
        if (entry->session.defer_failed_cleanup && (!all_success || expired)) {
            // A failed/timed-out PACE write may still have remote DMA in flight.
            // Keep both the metadata owner and allocation charged until the
            // backend's quarantine deadline. The expiry worker is the sole
            // consumer, so a repeated failed Finish is idempotent and a late
            // successful Finish can never resurrect an aborted value.
            entry->aborted = true;
            return {expired && all_success ? TakeResult::kExpiredDeferred : TakeResult::kDeferred, FinalizationGuard{}};
        }
        FinalizationGuard finalization = BeginFinalizationLocked(entry->session.internal_instance_id);
        deadlines_.erase(DeadlineKey{entry->deadline, entry->sequence});
        out = std::move(entry->session);
        sessions_.erase(it);
        condition_.notify_all();
        return {expired ? TakeResult::kExpired : TakeResult::kOk, std::move(finalization)};
    }

    bool HasSessionForInstance(const std::string &internal_instance_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool has_session = std::any_of(sessions_.begin(), sessions_.end(), [&](const auto &entry) {
            return entry.second && entry.second->session.internal_instance_id == internal_instance_id;
        });
        if (has_session) {
            return true;
        }
        std::lock_guard<std::mutex> finalization_lock(finalization_state_->mutex);
        return finalization_state_->instances.count(internal_instance_id) != 0;
    }

    void StopAndDiscard() {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        std::thread expiry_thread;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            if (thread_.joinable()) {
                expiry_thread = std::move(thread_);
            }
        }
        condition_.notify_all();
        if (expiry_thread.joinable()) {
            expiry_thread.join();
        }
        std::lock_guard<std::mutex> lock(mutex_);
        sessions_.clear();
        deadlines_.clear();
    }

private:
    using DeadlineKey = std::pair<Clock::time_point, std::uint64_t>;

    struct Entry {
        std::string session_id;
        Clock::time_point deadline;
        std::uint64_t sequence = 0;
        bool aborted = false;
        Session session;
    };

    void ExpireLoop() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stopping_) {
            if (deadlines_.empty()) {
                condition_.wait(lock, [this]() { return stopping_ || !deadlines_.empty(); });
                continue;
            }
            const auto deadline = deadlines_.begin()->first.first;
            condition_.wait_until(lock, deadline);
            if (stopping_) {
                break;
            }
            const auto now = Clock::now();
            std::optional<Session> expired;
            std::optional<FinalizationGuard> finalization;
            if (!deadlines_.empty() && deadlines_.begin()->first.first <= now) {
                auto entry = deadlines_.begin()->second;
                deadlines_.erase(deadlines_.begin());
                const auto session_it = sessions_.find(entry->session_id);
                if (session_it != sessions_.end() && session_it->second == entry) {
                    finalization.emplace(BeginFinalizationLocked(entry->session.internal_instance_id));
                    expired.emplace(std::move(entry->session));
                    sessions_.erase(session_it);
                }
            }
            if (expired) {
                lock.unlock();
                Expire(std::move(*expired));
                finalization.reset();
                lock.lock();
            }
        }
    }

    FinalizationGuard BeginFinalizationLocked(const std::string &internal_instance_id) {
        // Prepare every potentially throwing guard field before publishing the
        // count. Once incremented, constructing the returned guard is move-only
        // and noexcept, so an allocation failure cannot leave a permanent
        // phantom finalizer that blocks Trim.
        std::string guard_instance_id = internal_instance_id;
        std::lock_guard<std::mutex> lock(finalization_state_->mutex);
        ++finalization_state_->instances[internal_instance_id];
        return FinalizationGuard(finalization_state_, std::move(guard_instance_id));
    }

    void Expire(Session session) {
        if (!owner_) {
            return;
        }
        if (session.items.empty()) {
            KVCM_LOG_ERROR("KVMeta write-session cleanup found an empty owned session; recovery is required");
            owner_->CancelMaintenance();
            return;
        }
        if (owner_->maintenance_cancelled_.load(std::memory_order_acquire)) {
            return;
        }
        if (session.quota_shard >= owner_->quota_admission_mutexes_.size()) {
            KVCM_LOG_ERROR("KVMeta write-session cleanup has an invalid quota shard; recovery is required");
            owner_->CancelMaintenance();
            return;
        }
        // Serialize metadata removal and physical deletion with the next
        // StartWrite admission. Otherwise timeout cleanup can erase metadata,
        // expose the key as missing, and still be deleting the old allocation
        // while a new generation is admitted for the same group.
        std::unique_lock<std::mutex> quota_lock(owner_->quota_admission_mutexes_[session.quota_shard]);
        if (owner_->maintenance_cancelled_.load(std::memory_order_acquire)) {
            return;
        }
        RequestContext request_context("kv_meta_write_session_expired");
        const std::vector<bool> failed(session.items.size(), false);
        ErrorCode ec = EC_IO_ERROR;
        const char *failure_kind = "error_code";
        bool cleanup_threw = false;
        try {
            ec = owner_->FinishWriteInternal(&request_context, session.internal_instance_id, failed, session.items);
        } catch (const std::exception &) {
            failure_kind = "standard_exception";
            cleanup_threw = true;
        } catch (...) {
            failure_kind = "unknown_exception";
            cleanup_threw = true;
        }
        if (cleanup_threw) {
            // The session has already been removed from the owner table. An
            // exception can occur on either side of a metadata mutation, so
            // only recovery may safely classify the allocation now.
            owner_->CancelMaintenance();
            ec = EC_OUTCOME_UNKNOWN;
        }
        if (ec != EC_OK) {
            // FinishWriteInternal first persists a read-invisible tombstone,
            // then applies the backend's deletion policy. Retry-safe paths
            // retain that ledger for recovery; a legacy reusable address is
            // attempted once and an uncertain result closes KVMeta instead of
            // replaying a stale delete. Never downgrade either case to a clean
            // session expiry.
            KVCM_LOG_WARN("KVMeta write-session cleanup did not complete; durable cleanup recovery is required, "
                          "item_count[%zu], failure[%s], ec[%d]",
                          session.items.size(),
                          failure_kind,
                          ec);
        }
    }

    KvMetaManager *owner_ = nullptr;
    const std::size_t max_sessions_;
    std::mutex lifecycle_mutex_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool stopping_ = true;
    std::uint64_t next_sequence_ = 0;
    std::unordered_map<std::string, std::shared_ptr<Entry>> sessions_;
    // A guard shares this small tracker instead of calling back through the
    // session manager, so its destructor remains safe across a concurrent
    // manager stop. The map contains only currently finalizing instances.
    std::shared_ptr<FinalizationState> finalization_state_ = std::make_shared<FinalizationState>();
    std::map<DeadlineKey, std::shared_ptr<Entry>> deadlines_;
    std::thread thread_;
};

// KVMeta deliberately does not enter CacheReclaimer's fixed-block state
// machine or its shared deletion executor. This worker reuses the same
// Instance Group watermark/LRU configuration, but samples and retires only
// reserved KVMeta instances. Keeping the worker here also lets it use the
// exact-value guards and group admission shards that protect KVMeta's stable
// location ids from ABA races.
class KvMetaReclaimer {
public:
    explicit KvMetaReclaimer(KvMetaManager *owner) : owner_(owner) { RegisterMetrics(); }
    ~KvMetaReclaimer() { StopAndJoin(); }

    KvMetaReclaimer(const KvMetaReclaimer &) = delete;
    KvMetaReclaimer &operator=(const KvMetaReclaimer &) = delete;

    bool Start() {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (thread_.joinable()) {
                return !stopping_;
            }
            stopping_ = false;
            wake_requested_ = true;
        }
        try {
            thread_ = std::thread([this]() { Loop(); });
        } catch (const std::exception &e) {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            KVCM_LOG_ERROR("failed to start KVMeta reclaimer: %s", e.what());
            return false;
        }
        condition_.notify_all();
        return true;
    }

    void RequestStop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
    }

    void StopAndJoin() noexcept {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        std::thread worker;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            wake_requested_ = false;
            if (thread_.joinable()) {
                worker = std::move(thread_);
            }
        }
        condition_.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        std::lock_guard<std::mutex> lock(mutex_);
        pending_instances_.clear();
        pending_locations_.clear();
        pending_credits_.clear();
        admission_demands_.clear();
        pending_batches_.clear();
        sampling_rotation_by_group_.clear();
        pending_batch_count_ = 0;
        pending_object_count_ = 0;
        pending_bytes_ = 0;
        UpdatePendingMetricsLocked();
        UpdateAdmissionDemandMetricsLocked();
    }

    void Wake() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            wake_requested_ = true;
        }
        condition_.notify_all();
    }

    bool HasPendingForInstance(const std::string &internal_instance_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = pending_instances_.find(internal_instance_id);
        return it != pending_instances_.end() && it->second != 0;
    }

    bool HasPendingLocation(const std::string &internal_instance_id, const std::string &location_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = pending_locations_.find({internal_instance_id, location_id});
        return it != pending_locations_.end() && it->second != 0;
    }

    bool IsAdmissionBlocked(const std::string &instance_group) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = pending_credits_.find(instance_group);
        return it != pending_credits_.end() && it->second.blocked_batch_count != 0;
    }

    bool HasExecutableTuning() const noexcept {
        const auto [sampling_size, batch_size] = SamplingAndBatchSize();
        return sampling_size != 0 && batch_size != 0;
    }

    // Record the largest request that was rejected only because existing
    // cache entries consume its hard byte/key capacity. A wake-up alone is
    // insufficient below the configured watermark: the worker must know how
    // much headroom this concrete request needs. Max (rather than sum) avoids
    // evicting the whole cache for a burst of equivalent retries.
    void RequestAdmissionCapacity(const std::string &instance_group,
                                  DataStorageType storage_type,
                                  std::uint64_t requested_bytes,
                                  const std::string &internal_instance_id = {},
                                  std::uint64_t requested_keys = 0,
                                  bool backend_capacity_failure = false) noexcept {
        if (instance_group.empty() || (requested_bytes == 0 && requested_keys == 0) ||
            (requested_keys != 0 && internal_instance_id.empty()) ||
            (backend_capacity_failure &&
             (requested_bytes == 0 || storage_type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN ||
              requested_keys != 0))) {
            return;
        }
        try {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stopping_) {
                    return;
                }
                // Allocate every map node before changing the published
                // demand. A failed allocation must not leave a sequence-zero
                // entry that can neither be observed as new nor cleared.
                const auto [demand_it, inserted_group] = admission_demands_.try_emplace(instance_group);
                try {
                    if (requested_keys != 0) {
                        demand_it->second.requested_keys_by_instance.try_emplace(internal_instance_id, 0);
                    }
                } catch (...) {
                    if (inserted_group) {
                        admission_demands_.erase(demand_it);
                    }
                    throw;
                }
                auto &demand = demand_it->second;
                if (!backend_capacity_failure) {
                    demand.requested_group_bytes = std::max(demand.requested_group_bytes, requested_bytes);
                }
                if (requested_bytes != 0 && storage_type != DataStorageType::DATA_STORAGE_TYPE_UNKNOWN) {
                    const std::size_t type_index = ToIndex(ToBaseType(storage_type));
                    if (type_index < demand.requested_bytes_by_type.size()) {
                        auto &type_bytes = backend_capacity_failure ? demand.backend_reclaim_bytes_by_type[type_index]
                                                                    : demand.requested_bytes_by_type[type_index];
                        // A retry burst represents overlapping demand, not
                        // additive capacity. Keep the largest unsatisfied
                        // request so retries cannot evict the entire cache.
                        type_bytes = std::max(type_bytes, requested_bytes);
                    }
                }
                if (requested_keys != 0) {
                    auto &instance_keys = demand.requested_keys_by_instance.find(internal_instance_id)->second;
                    instance_keys = std::max(instance_keys, requested_keys);
                }
                // Zero means "no demand" in pressure snapshots. Keep it
                // reserved even after the practically unreachable uint64 wrap.
                if (++next_admission_demand_sequence_ == 0) {
                    ++next_admission_demand_sequence_;
                }
                demand.sequence = next_admission_demand_sequence_;
                ++admission_demand_count_metrics_;
                if (backend_capacity_failure) {
                    ++backend_capacity_demand_count_metrics_;
                }
                UpdateAdmissionDemandMetricsLocked();
                wake_requested_ = true;
            }
            condition_.notify_all();
        } catch (const std::exception &e) {
            KVCM_LOG_WARN("failed to publish KVMeta admission demand: %s", e.what());
        } catch (...) { KVCM_LOG_WARN("failed to publish KVMeta admission demand with unknown exception"); }
    }

    // Publish physical-backend pressure only after a backend has
    // authoritatively rejected an allocation without creating an object.
    // This is deliberately separate from logical quota admission: a shared
    // Provider can be full while this KVCM group remains below its quota.
    void RequestBackendCapacity(const std::string &instance_group,
                                DataStorageType storage_type,
                                std::uint64_t requested_bytes) noexcept {
        // A retry while this group already has a retired object of the same
        // type must wait for that physical delete to settle. Publishing a new
        // demand here would be causally ambiguous: the Provider may have
        // evaluated the allocation before or after the concurrent delete. A
        // retry after pending completion will publish a fresh demand if the
        // Provider is still full.
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto credit_it = pending_credits_.find(instance_group);
            const std::size_t type_index = ToIndex(ToBaseType(storage_type));
            if (credit_it != pending_credits_.end() && type_index < credit_it->second.bytes_by_type.size() &&
                credit_it->second.bytes_by_type[type_index] != 0) {
                return;
            }
        } catch (const std::exception &e) {
            KVCM_LOG_WARN("failed to inspect pending KVMeta capacity release: %s", e.what());
            return;
        } catch (...) {
            KVCM_LOG_WARN("failed to inspect pending KVMeta capacity release with unknown exception");
            return;
        }
        RequestAdmissionCapacity(instance_group, storage_type, requested_bytes, {}, 0, true);
    }

    // A successful foreground Remove releases the same Provider capacity as
    // an asynchronous reclaim batch. The caller holds the group-admission
    // shard across deletion, so consuming demand cannot race a newer NOSPC.
    void ConfirmExternalCapacityFreed(const std::string &instance_group,
                                      const std::vector<KvMetaManager::SessionItem> &items) noexcept {
        if (instance_group.empty() || items.empty()) {
            return;
        }
        try {
            std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> freed_by_type{};
            for (const auto &item : items) {
                if (!item.data_location) {
                    continue;
                }
                const std::size_t type_index = ToIndex(ToBaseType(item.data_location->type()));
                if (type_index < freed_by_type.size()) {
                    freed_by_type[type_index] = SaturatingAdd(freed_by_type[type_index], item.value_size);
                }
            }
            ConfirmBackendCapacityFreed(instance_group, freed_by_type);
        } catch (const std::exception &e) {
            KVCM_LOG_WARN("failed to account foreground KVMeta capacity release: %s", e.what());
        } catch (...) {
            KVCM_LOG_WARN("failed to account foreground KVMeta capacity release with unknown exception");
        }
    }

private:
    void FailClosedMaintenance() noexcept {
        if (!owner_) {
            return;
        }
        owner_->maintenance_cancelled_.store(true, std::memory_order_release);
        if (owner_->write_session_manager_) {
            // Match demotion semantics: reject publication of any concurrently
            // allocating session, while sessions already published may still
            // Finish and release their own allocation safely.
            owner_->write_session_manager_->RequestStop();
        }
    }

    void RegisterMetrics() noexcept {
        try {
            const auto registry =
                owner_ && owner_->cache_manager_ ? owner_->cache_manager_->metrics_registry() : nullptr;
            if (!registry) {
                return;
            }
            round_count_metrics_ = registry->GetCounter("kv_meta_reclaimer.round_count");
            retired_object_count_metrics_ = registry->GetCounter("kv_meta_reclaimer.retired_object_count");
            reclaimed_object_count_metrics_ = registry->GetCounter("kv_meta_reclaimer.reclaimed_object_count");
            reclaimed_bytes_metrics_ = registry->GetCounter("kv_meta_reclaimer.reclaimed_bytes");
            retry_count_metrics_ = registry->GetCounter("kv_meta_reclaimer.retry_count");
            error_count_metrics_ = registry->GetCounter("kv_meta_reclaimer.error_count");
            pending_limit_reject_count_metrics_ = registry->GetCounter("kv_meta_reclaimer.pending_limit_reject_count");
            admission_demand_count_metrics_ = registry->GetCounter("kv_meta_reclaimer.admission_demand_count");
            backend_capacity_demand_count_metrics_ =
                registry->GetCounter("kv_meta_reclaimer.backend_capacity_demand_count");
            physical_delete_attempted_object_count_metrics_ =
                registry->GetCounter("kv_meta_reclaimer.physical_delete_attempted_object_count");
            physical_delete_uncertain_object_count_metrics_ =
                registry->GetCounter("kv_meta_reclaimer.physical_delete_uncertain_object_count");
            physical_delete_uncertain_bytes_metrics_ =
                registry->GetCounter("kv_meta_reclaimer.physical_delete_uncertain_bytes");
            maintenance_touch_key_count_metrics_ =
                registry->GetCounter("kv_meta_reclaimer.maintenance_touch_key_count");
            pending_batch_count_metrics_ = registry->GetGauge("kv_meta_reclaimer.pending_batch_count");
            pending_object_count_metrics_ = registry->GetGauge("kv_meta_reclaimer.pending_object_count");
            pending_bytes_metrics_ = registry->GetGauge("kv_meta_reclaimer.pending_bytes");
            blocked_group_count_metrics_ = registry->GetGauge("kv_meta_reclaimer.blocked_group_count");
            admission_demand_group_count_metrics_ =
                registry->GetGauge("kv_meta_reclaimer.admission_demand_group_count");
            backend_capacity_demand_bytes_metrics_ =
                registry->GetGauge("kv_meta_reclaimer.backend_capacity_demand_bytes");
            UpdatePendingMetricsLocked();
            UpdateAdmissionDemandMetricsLocked();
        } catch (const std::exception &e) {
            KVCM_LOG_WARN("failed to register KVMeta reclaimer metrics: %s", e.what());
        } catch (...) { KVCM_LOG_WARN("failed to register KVMeta reclaimer metrics with unknown exception"); }
    }

    void UpdatePendingMetricsLocked() noexcept {
        pending_batch_count_metrics_ = static_cast<double>(pending_batch_count_);
        pending_object_count_metrics_ = static_cast<double>(pending_object_count_);
        pending_bytes_metrics_ = static_cast<double>(pending_bytes_);
        const auto blocked_groups =
            std::count_if(pending_credits_.begin(), pending_credits_.end(), [](const auto &entry) {
                return entry.second.blocked_batch_count != 0;
            });
        blocked_group_count_metrics_ = static_cast<double>(blocked_groups);
        admission_demand_group_count_metrics_ = static_cast<double>(admission_demands_.size());
    }

    void UpdateAdmissionDemandMetricsLocked() noexcept {
        std::uint64_t backend_bytes = 0;
        for (const auto &[_, demand] : admission_demands_) {
            for (const std::uint64_t bytes : demand.backend_reclaim_bytes_by_type) {
                backend_bytes = SaturatingAdd(backend_bytes, bytes);
            }
        }
        admission_demand_group_count_metrics_ = static_cast<double>(admission_demands_.size());
        backend_capacity_demand_bytes_metrics_ = static_cast<double>(backend_bytes);
    }

    struct Pressure {
        std::uint64_t group_bytes = 0;
        std::uint64_t keys = 0;
        std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> bytes_by_type{};
        std::map<std::string, std::uint64_t> keys_by_instance;

        bool Any() const noexcept {
            if (group_bytes != 0 || keys != 0) {
                return true;
            }
            return std::any_of(
                       bytes_by_type.begin(), bytes_by_type.end(), [](std::uint64_t value) { return value != 0; }) ||
                   std::any_of(keys_by_instance.begin(), keys_by_instance.end(), [](const auto &entry) {
                       return entry.second != 0;
                   });
        }

        void Consume(DataStorageType type, std::uint64_t bytes) noexcept {
            group_bytes = bytes >= group_bytes ? 0 : group_bytes - bytes;
            const std::size_t type_index = ToIndex(ToBaseType(type));
            if (type_index < bytes_by_type.size()) {
                bytes_by_type[type_index] = bytes >= bytes_by_type[type_index] ? 0 : bytes_by_type[type_index] - bytes;
            }
        }

        void ConsumeKey(const std::string &internal_instance_id) noexcept {
            if (keys != 0) {
                --keys;
            }
            const auto it = keys_by_instance.find(internal_instance_id);
            if (it != keys_by_instance.end() && it->second != 0) {
                --it->second;
            }
        }
    };

    struct Candidate {
        std::string internal_instance_id;
        std::int64_t internal_key = 0;
        std::string location_id;
        CacheLocationConstPtr location;
        std::uint64_t value_size = 0;
        bool removes_metadata_key = false;
    };

    struct CandidateKey {
        std::string internal_instance_id;
        std::int64_t internal_key = 0;
        std::int64_t last_access_time_us = 0;
        bool all_locations_committed = false;
        std::vector<Candidate> objects;
    };

    struct RetiredItem {
        std::string internal_instance_id;
        KvMetaManager::SessionItem item;
        bool removes_metadata_key = false;
        bool metadata_durable = false;
        // Set only after this pending batch's exact compare-and-delete was
        // accepted. A subsequent EC_NOENT is then a legitimate Sync retry;
        // EC_NOENT before this evidence is an unexpected owner transition.
        bool metadata_delete_applied = false;
    };

    struct PendingCredit {
        std::uint64_t bytes = 0;
        std::uint64_t keys = 0;
        std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> bytes_by_type{};
        std::map<std::string, std::uint64_t> keys_by_instance;
        // The entry is created when the batch is enqueued, before a metadata
        // cleanup can fail. Finalization can therefore close admission without
        // allocating memory on the error path.
        std::size_t blocked_batch_count = 0;
    };

    struct AdmissionDemand {
        std::uint64_t requested_group_bytes = 0;
        std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> requested_bytes_by_type{};
        // Remaining bytes that a concrete backend EC_NOSPC asked us to free.
        // Pending tombstones cover this demand while their physical deletion
        // is in flight; only a successful backend release consumes it.
        std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> backend_reclaim_bytes_by_type{};
        std::map<std::string, std::uint64_t> requested_keys_by_instance;
        std::uint64_t sequence = 0;
    };

    struct PendingBatch {
        std::string instance_group;
        std::size_t quota_shard = 0;
        std::chrono::steady_clock::time_point deadline;
        std::uint64_t sequence = 0;
        std::vector<RetiredItem> items;
        std::vector<std::string> instances;
        std::vector<std::pair<std::string, std::string>> locations;
        bool admission_blocked = false;
        // Path-unique backends may replay a delete from the durable retired
        // record. Legacy PACE GAs are reusable and carry no generation token,
        // so an ambiguous attempt is terminal and metadata-only finalization
        // must never send that address again.
        bool delete_policy_valid = true;
        bool delete_retry_safe = false;
        bool physical_delete_finished = false;
        std::uint32_t retry_count = 0;
    };

    using PendingDeadline = std::pair<std::chrono::steady_clock::time_point, std::uint64_t>;

    struct PendingCapacity {
        std::size_t batch_slots = 0;
        std::size_t object_count = 0;
        std::uint64_t bytes = 0;
    };

    static constexpr std::uint64_t kPendingBatchLimit = 1024;
    static constexpr std::uint64_t kPendingObjectLimit = 20'000;
    static constexpr std::uint64_t kPendingBytesLimit = 4ULL * 1024 * 1024 * 1024 * 1024;
    static constexpr long double kWatermarkEpsilon = 1e-9L;

    static std::size_t DeleteBatchCount(std::size_t item_count) noexcept {
        return item_count / kKvMetaDeleteBatchSize + (item_count % kKvMetaDeleteBatchSize != 0);
    }

    static std::uint64_t SaturatingAdd(std::uint64_t lhs, std::uint64_t rhs) noexcept {
        return rhs > std::numeric_limits<std::uint64_t>::max() - lhs ? std::numeric_limits<std::uint64_t>::max()
                                                                     : lhs + rhs;
    }

    static std::uint64_t SaturatingSub(std::uint64_t lhs, std::uint64_t rhs) noexcept {
        return rhs >= lhs ? 0 : lhs - rhs;
    }

    static std::uint64_t BytesToFree(std::int64_t capacity, double threshold, std::uint64_t used) noexcept {
        if (used == 0) {
            return 0;
        }
        if (capacity <= 0) {
            return used;
        }
        // Match CacheReclaimer's ratio + 1e-9 > threshold trigger. A
        // successful round must settle at or below threshold - epsilon instead
        // of continuously firing at an equal (or epsilon-close) boundary.
        const long double adjusted_threshold =
            std::max<long double>(0.0L, static_cast<long double>(threshold) - kWatermarkEpsilon);
        const long double raw_allowed = static_cast<long double>(capacity) * adjusted_threshold;
        const std::uint64_t allowed =
            raw_allowed <= 0
                ? 0
                : static_cast<std::uint64_t>(std::min<long double>(
                      std::floor(raw_allowed), static_cast<long double>(std::numeric_limits<std::uint64_t>::max())));
        return used > allowed ? used - allowed : 0;
    }

    static std::uint64_t
    BytesToFit(std::int64_t capacity, std::uint64_t used, std::uint64_t requested, bool &possible) noexcept {
        if (requested == 0) {
            return 0;
        }
        if (capacity < 0 || requested > static_cast<std::uint64_t>(capacity)) {
            possible = false;
            return 0;
        }
        const std::uint64_t allowed_before_admission = static_cast<std::uint64_t>(capacity) - requested;
        return used > allowed_before_admission ? used - allowed_before_admission : 0;
    }

    bool ShouldStop() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopping_ || !owner_ || owner_->maintenance_cancelled_.load(std::memory_order_acquire);
    }

    std::uint32_t IdleIntervalMs() const noexcept {
        try {
            if (!owner_ || !owner_->cache_manager_ || !owner_->cache_manager_->cache_reclaimer()) {
                return 100;
            }
            RequestContext request_context("kv_meta_reclaimer_config");
            return std::max<std::uint32_t>(
                1, owner_->cache_manager_->cache_reclaimer()->GetSleepIntervalMs(&request_context));
        } catch (...) { return 100; }
    }

    std::pair<std::size_t, std::size_t> SamplingAndBatchSize() const noexcept {
        try {
            if (!owner_ || !owner_->cache_manager_ || !owner_->cache_manager_->cache_reclaimer()) {
                return {0, 0};
            }
            RequestContext request_context("kv_meta_reclaimer_config");
            return {owner_->cache_manager_->cache_reclaimer()->GetSamplingSize(&request_context),
                    owner_->cache_manager_->cache_reclaimer()->GetBatchingSize(&request_context)};
        } catch (...) { return {0, 0}; }
    }

    PendingCredit GetPendingCredit(const std::string &instance_group) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = pending_credits_.find(instance_group);
        return it == pending_credits_.end() ? PendingCredit{} : it->second;
    }

    AdmissionDemand GetAdmissionDemand(const std::string &instance_group) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = admission_demands_.find(instance_group);
        return it == admission_demands_.end() ? AdmissionDemand{} : it->second;
    }

    void ClearAdmissionDemandIfCurrent(const std::string &instance_group, std::uint64_t sequence) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = admission_demands_.find(instance_group);
            // A concurrent rejected request may have raised the high-water
            // demand after this pressure snapshot. Never erase that newer
            // request's signal.
            if (it != admission_demands_.end() && it->second.sequence == sequence) {
                admission_demands_.erase(it);
                UpdateAdmissionDemandMetricsLocked();
            }
        } catch (...) {
            // Clearing a satisfied/impossible optimization signal must not
            // terminate the worker. Keeping it is safe and will be retried.
        }
    }

    void ConfirmBackendCapacityFreed(const std::string &instance_group,
                                     const std::vector<RetiredItem> &items) noexcept {
        if (instance_group.empty() || items.empty()) {
            return;
        }
        std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> freed_by_type{};
        for (const auto &item : items) {
            if (!item.item.data_location) {
                continue;
            }
            const std::size_t type_index = ToIndex(ToBaseType(item.item.data_location->type()));
            if (type_index < freed_by_type.size()) {
                freed_by_type[type_index] = SaturatingAdd(freed_by_type[type_index], item.item.value_size);
            }
        }
        ConfirmBackendCapacityFreed(instance_group, freed_by_type);
    }

    void ConfirmBackendCapacityFreed(
        const std::string &instance_group,
        const std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> &freed_by_type) noexcept {
        try {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                const auto demand_it = admission_demands_.find(instance_group);
                if (demand_it == admission_demands_.end()) {
                    return;
                }
                for (std::size_t i = 0; i < freed_by_type.size(); ++i) {
                    demand_it->second.backend_reclaim_bytes_by_type[i] =
                        SaturatingSub(demand_it->second.backend_reclaim_bytes_by_type[i], freed_by_type[i]);
                }
                UpdateAdmissionDemandMetricsLocked();
                // Let the worker clear a now-satisfied demand or continue any
                // uncovered request immediately after the Provider capacity
                // has actually become reusable.
                wake_requested_ = true;
            }
            condition_.notify_all();
        } catch (...) {
            // This path performs no allocation today. If a future container
            // change introduces one, retaining a conservative demand is safe:
            // it may reclaim extra cache, but cannot admit over capacity.
        }
    }

    bool HasPendingCapacity(const std::vector<Candidate> &candidates, std::size_t required_batch_count) const {
        if (required_batch_count == 0 || required_batch_count > kPendingBatchLimit) {
            return false;
        }
        std::uint64_t candidate_bytes = 0;
        for (const auto &candidate : candidates) {
            candidate_bytes = SaturatingAdd(candidate_bytes, candidate.value_size);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_batch_count_ <= kPendingBatchLimit - required_batch_count &&
               candidates.size() <= kPendingObjectLimit &&
               pending_object_count_ <= kPendingObjectLimit - candidates.size() &&
               candidate_bytes <= kPendingBytesLimit && pending_bytes_ <= kPendingBytesLimit - candidate_bytes;
    }

    PendingCapacity RemainingPendingCapacity() const {
        std::lock_guard<std::mutex> lock(mutex_);
        PendingCapacity capacity;
        if (pending_batch_count_ < kPendingBatchLimit) {
            capacity.batch_slots = static_cast<std::size_t>(kPendingBatchLimit - pending_batch_count_);
        }
        if (capacity.batch_slots == 0 || pending_object_count_ >= kPendingObjectLimit ||
            pending_bytes_ >= kPendingBytesLimit) {
            return capacity;
        }
        capacity.object_count = static_cast<std::size_t>(kPendingObjectLimit - pending_object_count_);
        capacity.bytes = kPendingBytesLimit - pending_bytes_;
        return capacity;
    }

    bool ReadPressure(RequestContext *request_context,
                      const InstanceGroup &group,
                      const std::vector<InstanceInfoConstPtr> &instances,
                      double threshold,
                      Pressure &out) {
        out = {};
        std::uint64_t group_usage = 0;
        std::uint64_t key_count = 0;
        std::uint64_t max_key_count = 0;
        std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> usage_by_type{};
        std::map<std::string, std::uint64_t> key_count_by_instance;
        std::map<std::string, std::uint64_t> max_key_count_by_instance;
        for (const auto &instance : instances) {
            if (!instance || !IsKvMetaInstance(*instance)) {
                AddError(request_context, "KVMeta reclaimer requires a dedicated generic-object instance group");
                return false;
            }
            const auto indexer =
                owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance->instance_id());
            if (!indexer) {
                AddError(request_context, "KVMeta reclaimer could not read an instance indexer");
                return false;
            }
            group_usage = SaturatingAdd(group_usage, indexer->GetStorageUsage());
            key_count = SaturatingAdd(key_count, static_cast<std::uint64_t>(indexer->GetKeyCount()));
            max_key_count = SaturatingAdd(max_key_count, static_cast<std::uint64_t>(indexer->GetMaxKeyCount()));
            key_count_by_instance[instance->instance_id()] = static_cast<std::uint64_t>(indexer->GetKeyCount());
            max_key_count_by_instance[instance->instance_id()] = static_cast<std::uint64_t>(indexer->GetMaxKeyCount());
            for (std::size_t i = 1; i < usage_by_type.size(); ++i) {
                const auto type = static_cast<DataStorageType>(i);
                if (ToBaseType(type) != type) {
                    continue;
                }
                usage_by_type[i] = SaturatingAdd(usage_by_type[i], indexer->GetStorageUsageByType(type));
            }
        }

        const PendingCredit credit = GetPendingCredit(group.name());
        group_usage = SaturatingSub(group_usage, credit.bytes);
        key_count = SaturatingSub(key_count, credit.keys);
        for (auto &[instance_id, instance_key_count] : key_count_by_instance) {
            const auto credit_it = credit.keys_by_instance.find(instance_id);
            if (credit_it != credit.keys_by_instance.end()) {
                instance_key_count = SaturatingSub(instance_key_count, credit_it->second);
            }
        }
        for (std::size_t i = 0; i < usage_by_type.size(); ++i) {
            usage_by_type[i] = SaturatingSub(usage_by_type[i], credit.bytes_by_type[i]);
        }

        out.group_bytes = BytesToFree(group.quota().capacity(), threshold, group_usage);
        out.keys = BytesToFree(max_key_count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
                                   ? std::numeric_limits<std::int64_t>::max()
                                   : static_cast<std::int64_t>(max_key_count),
                               threshold,
                               key_count);
        const AdmissionDemand demand = GetAdmissionDemand(group.name());
        bool demand_possible = true;
        bool demand_satisfied = demand.sequence != 0;
        const std::uint64_t group_demand_pressure =
            BytesToFit(group.quota().capacity(), group_usage, demand.requested_group_bytes, demand_possible);
        out.group_bytes = std::max(out.group_bytes, group_demand_pressure);
        demand_satisfied = demand_satisfied && group_demand_pressure == 0;
        for (const auto &quota : group.quota().quota_config()) {
            const auto base_type = ToBaseType(quota.storage_spec());
            const std::size_t type_index = ToIndex(base_type);
            if (base_type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN || type_index >= usage_by_type.size()) {
                continue;
            }
            out.bytes_by_type[type_index] = std::max(
                out.bytes_by_type[type_index], BytesToFree(quota.capacity(), threshold, usage_by_type[type_index]));
            const std::uint64_t type_demand_pressure = BytesToFit(quota.capacity(),
                                                                  usage_by_type[type_index],
                                                                  demand.requested_bytes_by_type[type_index],
                                                                  demand_possible);
            out.bytes_by_type[type_index] = std::max(out.bytes_by_type[type_index], type_demand_pressure);
            demand_satisfied = demand_satisfied && type_demand_pressure == 0;
        }

        // Backend capacity is a physical constraint, independent of whether
        // the group configured an optional logical per-storage-type quota.
        // A Provider can return an authoritative no-allocation EC_NOSPC while
        // the group has ample total quota and no type quota at all. Target the
        // failed physical type in that case too; otherwise the demand would be
        // discarded and a valid cache could never make progress.
        for (std::size_t type_index = 1; type_index < demand.backend_reclaim_bytes_by_type.size(); ++type_index) {
            const std::uint64_t backend_remaining = demand.backend_reclaim_bytes_by_type[type_index];
            if (backend_remaining == 0) {
                continue;
            }
            const auto type = static_cast<DataStorageType>(type_index);
            if (ToBaseType(type) != type) {
                demand_possible = false;
                demand_satisfied = false;
                continue;
            }
            // A retired object is already unavailable to readers but its
            // bytes do not help a physically full Provider until exact
            // deletion is confirmed. Treat pending bytes as reserved work so
            // request retries cannot schedule duplicate eviction.
            demand_satisfied = false;
            const std::uint64_t inflight = credit.bytes_by_type[type_index];
            const std::uint64_t uncovered = SaturatingSub(backend_remaining, inflight);
            if (uncovered == 0) {
                continue;
            }
            const std::uint64_t reclaimable = std::min(uncovered, usage_by_type[type_index]);
            out.bytes_by_type[type_index] = std::max(out.bytes_by_type[type_index], reclaimable);
            if (reclaimable == 0) {
                // There is no object of this type left that this KVCM group is
                // authorized to delete. A retry may publish a fresh demand
                // after external capacity changes.
                demand_possible = false;
            }
        }
        for (const auto &[instance_id, requested_keys] : demand.requested_keys_by_instance) {
            const auto used_it = key_count_by_instance.find(instance_id);
            const auto capacity_it = max_key_count_by_instance.find(instance_id);
            if (used_it == key_count_by_instance.end() || capacity_it == max_key_count_by_instance.end() ||
                requested_keys > capacity_it->second) {
                demand_possible = false;
                continue;
            }
            const std::uint64_t allowed_before_admission = capacity_it->second - requested_keys;
            if (used_it->second > allowed_before_admission) {
                out.keys_by_instance[instance_id] = used_it->second - allowed_before_admission;
                demand_satisfied = false;
            }
        }
        if (!demand_possible || demand_satisfied) {
            // An impossible demand must not remain armed and continuously scan
            // or evict the cache after a quota/configuration change. A caller
            // with a still-viable request will publish a fresh demand on retry.
            ClearAdmissionDemandIfCurrent(group.name(), demand.sequence);
        }
        return true;
    }

    bool CollectCandidates(RequestContext *request_context,
                           const std::string &instance_group,
                           const std::vector<InstanceInfoConstPtr> &instances,
                           const Pressure &pressure,
                           std::size_t sampling_size,
                           std::vector<CandidateKey> &out) {
        out.clear();
        std::vector<std::pair<InstanceInfoConstPtr, std::size_t>> eligible;
        std::uint64_t total_key_count = 0;
        for (const auto &instance : instances) {
            if (!instance) {
                return false;
            }
            const auto indexer =
                owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance->instance_id());
            if (!indexer) {
                return false;
            }
            const std::size_t count = indexer->GetKeyCount();
            if (count != 0) {
                eligible.emplace_back(instance, count);
            }
        }
        if (eligible.empty() || sampling_size == 0) {
            return true;
        }

        std::sort(eligible.begin(), eligible.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.first->instance_id() < rhs.first->instance_id();
        });
        const std::size_t instance_budget = std::min(sampling_size, eligible.size());
        const std::size_t rotation = sampling_rotation_by_group_[instance_group] % eligible.size();
        std::vector<std::pair<InstanceInfoConstPtr, std::size_t>> selected_instances;
        selected_instances.reserve(instance_budget);
        std::set<std::string> selected_instance_ids;
        // A rejected request at one full instance must not wait for a group
        // round-robin over every peer. Reserve the bounded sampling slots for
        // specifically pressured instances first, then retain rotation for the
        // remaining periodic/group/type work.
        for (const auto &entry : eligible) {
            const auto pressure_it = pressure.keys_by_instance.find(entry.first->instance_id());
            if (pressure_it != pressure.keys_by_instance.end() && pressure_it->second != 0 &&
                selected_instances.size() < instance_budget) {
                selected_instances.push_back(entry);
                selected_instance_ids.insert(entry.first->instance_id());
            }
        }
        for (std::size_t offset = 0; offset < eligible.size() && selected_instances.size() < instance_budget;
             ++offset) {
            const auto &entry = eligible[(rotation + offset) % eligible.size()];
            if (selected_instance_ids.insert(entry.first->instance_id()).second) {
                selected_instances.push_back(entry);
            }
        }
        for (const auto &entry : selected_instances) {
            total_key_count = SaturatingAdd(total_key_count, entry.second);
        }
        sampling_rotation_by_group_[instance_group] = (rotation + instance_budget) % eligible.size();

        // Give each selected instance one slot, then distribute the remaining
        // strict per-round budget by key count. Rotation prevents groups with
        // more instances than sampling slots from starving their tail.
        const std::size_t total_budget = sampling_size;
        std::size_t remaining_budget = total_budget;
        std::uint64_t remaining_weight = total_key_count;
        std::uint64_t observed_location_count = 0;
        std::set<std::tuple<std::string, std::int64_t>> sampled_keys;
        for (std::size_t instance_index = 0; instance_index < selected_instances.size(); ++instance_index) {
            const auto &[instance, key_count_for_instance] = selected_instances[instance_index];
            const std::size_t instances_left = selected_instances.size() - instance_index;
            std::size_t key_budget = 1;
            if (remaining_budget > instances_left && remaining_weight != 0) {
                const long double weighted = static_cast<long double>(remaining_budget) *
                                             static_cast<long double>(key_count_for_instance) /
                                             static_cast<long double>(remaining_weight);
                key_budget = std::max<std::size_t>(1, static_cast<std::size_t>(weighted));
            }
            key_budget = std::min({key_budget, key_count_for_instance, remaining_budget - instances_left + 1});
            remaining_budget -= key_budget;
            remaining_weight =
                key_count_for_instance >= remaining_weight ? 0 : remaining_weight - key_count_for_instance;

            const auto indexer =
                owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance->instance_id());
            ReclaimCandidateVector sampled;
            if (indexer->SampleReclaimCandidates(
                    request_context, static_cast<std::int64_t>(key_budget), sampled, true) != EC_OK) {
                return false;
            }
            if (sampled.size() > key_budget) {
                AddError(request_context, "KVMeta reclaim sampling exceeded its requested bound");
                ++error_count_metrics_;
                return false;
            }
            KeyVector keys;
            std::vector<std::int64_t> access_times;
            keys.reserve(sampled.size());
            access_times.reserve(sampled.size());
            for (const auto &candidate : sampled) {
                if (sampled_keys.emplace(instance->instance_id(), candidate.key).second) {
                    keys.push_back(candidate.key);
                    access_times.push_back(candidate.last_access_time_us);
                }
            }
            if (keys.empty()) {
                continue;
            }
            CacheLocationMapVector location_maps;
            const auto get_result = indexer->GetLocationMapsForMaintenance(request_context, keys, location_maps);
            if ((get_result.ec != EC_OK && get_result.ec != EC_PARTIAL_OK) || location_maps.size() != keys.size() ||
                get_result.error_codes.size() != keys.size()) {
                return false;
            }
            KeyVector rejected_keys;
            rejected_keys.reserve(keys.size());
            for (std::size_t key_index = 0; key_index < keys.size(); ++key_index) {
                if (get_result.error_codes[key_index] == EC_NOENT) {
                    continue;
                }
                if (get_result.error_codes[key_index] != EC_OK || location_maps[key_index].empty()) {
                    return false;
                }
                CandidateKey key_candidate;
                key_candidate.internal_instance_id = instance->instance_id();
                key_candidate.internal_key = keys[key_index];
                key_candidate.last_access_time_us = access_times[key_index];
                key_candidate.all_locations_committed = true;
                if (location_maps[key_index].size() > kPendingObjectLimit - observed_location_count) {
                    AddError(request_context, "KVMeta reclaim candidate locations exceeded the bounded scan limit");
                    ++error_count_metrics_;
                    return false;
                }
                observed_location_count += location_maps[key_index].size();
                bool valid_key = true;
                for (const auto &[location_id, location] : location_maps[key_index]) {
                    if (!location) {
                        valid_key = false;
                        break;
                    }
                    std::uint64_t value_size = 0;
                    if (owner_->ValidateOwnedLocation(request_context,
                                                      instance->instance_id(),
                                                      keys[key_index],
                                                      location_id,
                                                      *location,
                                                      value_size) != EC_OK) {
                        valid_key = false;
                        break;
                    }
                    if (!IsCommittedObject(*location)) {
                        key_candidate.all_locations_committed = false;
                        continue;
                    }
                    key_candidate.objects.push_back(
                        Candidate{instance->instance_id(), keys[key_index], location_id, location, value_size});
                }
                if (!valid_key) {
                    ++error_count_metrics_;
                    KVCM_INTERVAL_LOG_WARN(10,
                                           "KVMeta reclaimer skipped a corrupt candidate in instance [%s]",
                                           instance->instance_id().c_str());
                    rejected_keys.push_back(keys[key_index]);
                    continue;
                }
                const auto instance_pressure = pressure.keys_by_instance.find(instance->instance_id());
                const bool relieves_key_pressure =
                    key_candidate.all_locations_committed &&
                    (pressure.keys != 0 || (instance_pressure != pressure.keys_by_instance.end() &&
                                            instance_pressure->second != 0));
                const bool relieves_type_pressure =
                    std::any_of(key_candidate.objects.begin(), key_candidate.objects.end(), [&](const auto &item) {
                        const std::size_t type_index = ToIndex(ToBaseType(item.location->type()));
                        return type_index < pressure.bytes_by_type.size() && pressure.bytes_by_type[type_index] != 0;
                    });
                if (!key_candidate.objects.empty() &&
                    (pressure.group_bytes != 0 || relieves_key_pressure || relieves_type_pressure)) {
                    out.push_back(std::move(key_candidate));
                } else {
                    // Sampling is LRU ordered. An active/tombstoned object, or
                    // an object that cannot relieve the current constrained
                    // pressure, must yield its cold slot or a small sample can
                    // rediscover it forever and starve reclaimable successors.
                    // This maintenance-only touch is supported solely by a
                    // complete local/cached sampling source; other backends
                    // return zero without mutating metadata.
                    rejected_keys.push_back(keys[key_index]);
                }
            }
            if (!rejected_keys.empty()) {
                maintenance_touch_key_count_metrics_ += indexer->TouchKeysForMaintenance(rejected_keys);
            }
        }
        return true;
    }

    static std::vector<Candidate> SelectCandidates(std::vector<CandidateKey> candidates,
                                                   Pressure pressure,
                                                   std::size_t batch_size,
                                                   std::uint64_t byte_budget) {
        std::sort(candidates.begin(), candidates.end(), [](const CandidateKey &lhs, const CandidateKey &rhs) {
            return std::tie(lhs.last_access_time_us, lhs.internal_instance_id, lhs.internal_key) <
                   std::tie(rhs.last_access_time_us, rhs.internal_instance_id, rhs.internal_key);
        });
        std::vector<Candidate> selected;
        selected.reserve(std::min<std::size_t>(batch_size, kPendingObjectLimit));
        std::vector<std::vector<bool>> object_selected;
        object_selected.reserve(candidates.size());
        std::vector<std::size_t> selected_per_key(candidates.size(), 0);
        std::vector<bool> key_credit_applied(candidates.size(), false);
        std::uint64_t selected_bytes = 0;
        for (const auto &candidate : candidates) {
            object_selected.emplace_back(candidate.objects.size(), false);
        }

        const auto has_type_pressure = [&pressure](const CandidateKey &key_candidate) {
            return std::any_of(key_candidate.objects.begin(), key_candidate.objects.end(), [&](const Candidate &item) {
                const std::size_t type_index = ToIndex(ToBaseType(item.location->type()));
                return type_index < pressure.bytes_by_type.size() && pressure.bytes_by_type[type_index] != 0;
            });
        };
        const auto select_object = [&](std::size_t key_index, std::size_t object_index) {
            if (selected.size() >= batch_size || object_selected[key_index][object_index]) {
                return false;
            }
            const Candidate &candidate = candidates[key_index].objects[object_index];
            if (selected_bytes > byte_budget || candidate.value_size > byte_budget - selected_bytes) {
                return false;
            }
            object_selected[key_index][object_index] = true;
            ++selected_per_key[key_index];
            selected.push_back(candidate);
            selected_bytes += candidate.value_size;
            pressure.Consume(candidate.location->type(), candidate.value_size);
            if (!key_credit_applied[key_index] && candidates[key_index].all_locations_committed &&
                selected_per_key[key_index] == candidates[key_index].objects.size()) {
                // Whichever constraint selected the last location has removed
                // the complete primary metadata key. Credit it exactly once so
                // a simultaneous key-count pressure cannot over-evict.
                selected.back().removes_metadata_key = true;
                pressure.ConsumeKey(candidates[key_index].internal_instance_id);
                key_credit_applied[key_index] = true;
            }
            return true;
        };
        const auto select_whole_key = [&](std::size_t key_index) {
            const auto &key_candidate = candidates[key_index];
            if (!key_candidate.all_locations_committed || key_candidate.objects.empty() ||
                key_credit_applied[key_index] ||
                key_candidate.objects.size() - selected_per_key[key_index] > batch_size - selected.size()) {
                return false;
            }
            std::uint64_t remaining_key_bytes = 0;
            for (std::size_t object_index = 0; object_index < key_candidate.objects.size(); ++object_index) {
                if (object_selected[key_index][object_index]) {
                    continue;
                }
                const std::uint64_t object_bytes = key_candidate.objects[object_index].value_size;
                if (selected_bytes > byte_budget || remaining_key_bytes > byte_budget - selected_bytes ||
                    object_bytes > byte_budget - selected_bytes - remaining_key_bytes) {
                    return false;
                }
                remaining_key_bytes += object_bytes;
            }
            for (std::size_t object_index = 0; object_index < key_candidate.objects.size(); ++object_index) {
                select_object(key_index, object_index);
            }
            return static_cast<bool>(key_credit_applied[key_index]);
        };
        const auto select_key_for_key_pressure = [&](std::size_t key_index) {
            if (select_whole_key(key_index)) {
                return true;
            }
            const auto &key_candidate = candidates[key_index];
            if (!key_candidate.all_locations_committed || key_candidate.objects.empty() ||
                key_credit_applied[key_index]) {
                return false;
            }
            // A hash bucket can contain more exact locations than one reclaim
            // batch. Requiring the whole bucket to fit would make key-count
            // pressure stall forever. Drain a committed oversized bucket in
            // bounded chunks; only the batch containing its final location
            // receives key credit.
            bool made_progress = false;
            for (std::size_t object_index = 0;
                 object_index < key_candidate.objects.size() && selected.size() < batch_size;
                 ++object_index) {
                made_progress = select_object(key_index, object_index) || made_progress;
            }
            return made_progress;
        };

        // Specific constraints are subsets of the group constraint. Satisfy
        // them first so the same retired bytes also reduce group pressure. A
        // single global-LRU pass can otherwise retire an unrelated old object
        // for the group and then a second object for the constrained type or
        // instance, even though one retirement was sufficient.
        for (std::size_t key_index = 0; key_index < candidates.size() && selected.size() < batch_size; ++key_index) {
            const auto pressure_it = pressure.keys_by_instance.find(candidates[key_index].internal_instance_id);
            if (pressure_it != pressure.keys_by_instance.end() && pressure_it->second != 0) {
                select_key_for_key_pressure(key_index);
            }
        }

        // For aggregate key pressure, prefer a whole key that also relieves an
        // outstanding storage-type pressure, preserving LRU order within that
        // more useful class. Then fall back to the oldest remaining whole key.
        for (const bool require_type_overlap : {true, false}) {
            for (std::size_t key_index = 0;
                 pressure.keys != 0 && key_index < candidates.size() && selected.size() < batch_size;
                 ++key_index) {
                if ((!require_type_overlap || has_type_pressure(candidates[key_index])) &&
                    !key_credit_applied[key_index]) {
                    select_key_for_key_pressure(key_index);
                }
            }
        }

        // Storage-type pressure is narrower than group pressure. Select the
        // oldest matching locations before using arbitrary bytes for the group.
        for (std::size_t key_index = 0; key_index < candidates.size() && selected.size() < batch_size; ++key_index) {
            for (std::size_t object_index = 0;
                 object_index < candidates[key_index].objects.size() && selected.size() < batch_size;
                 ++object_index) {
                const auto &candidate = candidates[key_index].objects[object_index];
                const std::size_t type_index = ToIndex(ToBaseType(candidate.location->type()));
                if (type_index < pressure.bytes_by_type.size() && pressure.bytes_by_type[type_index] != 0) {
                    select_object(key_index, object_index);
                }
            }
        }

        for (std::size_t key_index = 0;
             pressure.group_bytes != 0 && key_index < candidates.size() && selected.size() < batch_size;
             ++key_index) {
            for (std::size_t object_index = 0;
                 pressure.group_bytes != 0 && object_index < candidates[key_index].objects.size() &&
                 selected.size() < batch_size;
                 ++object_index) {
                select_object(key_index, object_index);
            }
        }
        return selected;
    }

    std::vector<RetiredItem> RetireCandidates(RequestContext *request_context,
                                              const std::vector<Candidate> &candidates,
                                              std::chrono::milliseconds delay,
                                              std::chrono::steady_clock::time_point &finalization_deadline) {
        finalization_deadline = {};
        std::map<std::string, std::vector<std::size_t>> by_instance;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            by_instance[candidates[i].internal_instance_id].push_back(i);
        }

        // Phase 1 is the reader fence.  Use a tagged far-future deadline until
        // every successful candidate has become durably unreadable. A failover
        // at any point in this phase can retain an object longer, but can never
        // shorten the grace period for a reader that just received its URI.
        std::vector<bool> retired(candidates.size(), false);
        std::vector<CacheLocationConstPtr> fenced_locations(candidates.size());
        for (const auto &[internal_instance_id, indices] : by_instance) {
            const auto indexer = owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
            if (!indexer) {
                continue;
            }
            KeyVector fenced_keys;
            fenced_keys.reserve(indices.size());
            std::vector<std::int64_t> keys;
            keys.reserve(indices.size());
            for (const std::size_t index : indices) {
                keys.push_back(candidates[index].internal_key);
            }
            for (const auto &layer : MakeUniqueKeyLayers(keys)) {
                KeyVector layer_keys;
                LocationIdsPerKey layer_ids;
                std::vector<CacheLocationConstPtr> expected;
                std::vector<CacheLocationConstPtr> replacements;
                layer_keys.reserve(layer.size());
                layer_ids.reserve(layer.size());
                expected.reserve(layer.size());
                replacements.reserve(layer.size());
                for (const std::size_t relative_index : layer) {
                    const Candidate &candidate = candidates[indices[relative_index]];
                    layer_keys.push_back(candidate.internal_key);
                    layer_ids.push_back({candidate.location_id});
                    expected.push_back(candidate.location);
                    auto replacement = std::make_shared<CacheLocation>(*candidate.location);
                    replacement->set_status(CLS_DELETING);
                    replacement->set_create_time(kRetirementFenceDeadline);
                    replacements.push_back(std::move(replacement));
                }
                std::vector<bool> layer_retired(layer.size(), false);
                auto modifier = [&expected, &replacements, &layer_retired](const std::vector<ErrorCode> &get_ecs,
                                                                           const LocationIdVector &,
                                                                           std::size_t key_index,
                                                                           CacheLocationVector &locations,
                                                                           PropertyMap &) -> LocationModifierResult {
                    if (get_ecs.size() != 1 || locations.size() != 1 || key_index >= expected.size()) {
                        return {MA_FAIL, {EC_MISMATCH}};
                    }
                    if (get_ecs[0] != EC_OK) {
                        return {MA_FAIL, {get_ecs[0]}};
                    }
                    if (!locations[0] || locations[0]->ToJsonString() != expected[key_index]->ToJsonString() ||
                        !IsCommittedObject(*locations[0])) {
                        return {MA_SKIP, {EC_MISMATCH}};
                    }
                    locations[0] = replacements[key_index];
                    layer_retired[key_index] = true;
                    return {MA_OK, {EC_OK}};
                };
                const auto result = indexer->ReadModifyWriteLocationsForMaintenance(
                    request_context, layer_keys, layer_ids, modifier, false);
                if (result.per_location_error_codes.size() != layer.size()) {
                    throw std::runtime_error("KVMeta retirement fence returned a malformed result");
                }
                for (std::size_t i = 0; i < layer.size(); ++i) {
                    if (result.per_location_error_codes[i].size() != 1) {
                        throw std::runtime_error("KVMeta retirement fence returned a malformed location result");
                    }
                    if (result.per_location_error_codes[i][0] == EC_OK && layer_retired[i]) {
                        const std::size_t candidate_index = indices[layer[i]];
                        retired[candidate_index] = true;
                        fenced_locations[candidate_index] = replacements[i];
                        fenced_keys.push_back(candidates[candidate_index].internal_key);
                    } else if (result.per_location_error_codes[i][0] == EC_OK || layer_retired[i]) {
                        // A successful result without a modifier transition, or
                        // a failed result after the modifier authorized one,
                        // has an unknown metadata outcome. Never continue toward
                        // physical deletion from such a response.
                        throw std::runtime_error("KVMeta retirement fence returned an inconsistent result");
                    }
                }
            }
            if (!fenced_keys.empty()) {
                std::sort(fenced_keys.begin(), fenced_keys.end());
                fenced_keys.erase(std::unique(fenced_keys.begin(), fenced_keys.end()), fenced_keys.end());
                if (!indexer->Sync(fenced_keys)) {
                    // No finite grace deadline has been published yet. Stop
                    // KVMeta maintenance and leave the durable/uncertain fence
                    // to leader recovery; issuing Delete here could race a
                    // reader admitted by the pre-fence metadata generation.
                    throw std::runtime_error("could not persist KVMeta retirement fence");
                }
            }
        }

        if (std::none_of(retired.begin(), retired.end(), [](bool value) { return value; })) {
            return {};
        }

        // Phase 2 chooses one finite deadline only after phase 1 has stopped
        // every new reader in this batch and persisted every fence. Capture
        // wall time first, then steady time, so normal in-process deletion is
        // not scheduled before the corresponding persisted wall deadline.
        const std::int64_t now_us = TimestampUtil::GetCurrentTimeUs();
        const auto steady_anchor = std::chrono::steady_clock::now();
        const auto delay_us = std::chrono::duration_cast<std::chrono::microseconds>(delay).count();
        std::int64_t retire_deadline = 0;
        if (!EncodeTaggedDeadlineUs(now_us, delay_us, retire_deadline)) {
            throw std::runtime_error("could not encode KVMeta retirement grace deadline");
        }
        finalization_deadline = steady_anchor + delay;

        std::vector<CacheLocationConstPtr> finalized_locations(candidates.size());
        std::map<std::string, bool> metadata_durable_by_instance;
        for (const auto &[internal_instance_id, indices] : by_instance) {
            const auto indexer = owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
            if (!indexer) {
                if (std::any_of(indices.begin(), indices.end(), [&](std::size_t index) { return retired[index]; })) {
                    throw std::runtime_error("KVMeta indexer disappeared after retirement fence");
                }
                continue;
            }

            std::vector<std::size_t> retired_indices;
            std::vector<std::int64_t> retired_keys;
            retired_indices.reserve(indices.size());
            retired_keys.reserve(indices.size());
            for (const std::size_t candidate_index : indices) {
                if (retired[candidate_index]) {
                    retired_indices.push_back(candidate_index);
                    retired_keys.push_back(candidates[candidate_index].internal_key);
                }
            }
            if (retired_indices.empty()) {
                continue;
            }

            for (const auto &layer : MakeUniqueKeyLayers(retired_keys)) {
                KeyVector layer_keys;
                LocationIdsPerKey layer_ids;
                std::vector<CacheLocationConstPtr> expected;
                std::vector<CacheLocationConstPtr> replacements;
                layer_keys.reserve(layer.size());
                layer_ids.reserve(layer.size());
                expected.reserve(layer.size());
                replacements.reserve(layer.size());
                for (const std::size_t relative_index : layer) {
                    const std::size_t candidate_index = retired_indices[relative_index];
                    const Candidate &candidate = candidates[candidate_index];
                    if (!fenced_locations[candidate_index]) {
                        throw std::logic_error("missing KVMeta retirement fence location");
                    }
                    layer_keys.push_back(candidate.internal_key);
                    layer_ids.push_back({candidate.location_id});
                    expected.push_back(fenced_locations[candidate_index]);
                    auto replacement = std::make_shared<CacheLocation>(*fenced_locations[candidate_index]);
                    replacement->set_create_time(retire_deadline);
                    replacements.push_back(std::move(replacement));
                }
                std::vector<bool> layer_finalized(layer.size(), false);
                auto modifier = [&expected, &replacements, &layer_finalized](const std::vector<ErrorCode> &get_ecs,
                                                                             const LocationIdVector &,
                                                                             std::size_t key_index,
                                                                             CacheLocationVector &locations,
                                                                             PropertyMap &) -> LocationModifierResult {
                    if (get_ecs.size() != 1 || locations.size() != 1 || key_index >= expected.size()) {
                        return {MA_FAIL, {EC_MISMATCH}};
                    }
                    if (get_ecs[0] != EC_OK) {
                        return {MA_FAIL, {get_ecs[0]}};
                    }
                    if (!locations[0] || locations[0]->ToJsonString() != expected[key_index]->ToJsonString() ||
                        !IsRetiredObject(*locations[0]) || locations[0]->create_time() != kRetirementFenceDeadline) {
                        return {MA_FAIL, {EC_MISMATCH}};
                    }
                    locations[0] = replacements[key_index];
                    layer_finalized[key_index] = true;
                    return {MA_OK, {EC_OK}};
                };
                const auto result = indexer->ReadModifyWriteLocationsForMaintenance(
                    request_context, layer_keys, layer_ids, modifier, false);
                if (result.per_location_error_codes.size() != layer.size()) {
                    throw std::runtime_error("KVMeta retirement deadline update returned a malformed result");
                }
                for (std::size_t i = 0; i < layer.size(); ++i) {
                    if (result.per_location_error_codes[i].size() != 1 ||
                        result.per_location_error_codes[i][0] != EC_OK || !layer_finalized[i]) {
                        throw std::runtime_error("KVMeta retirement deadline update failed");
                    }
                    finalized_locations[retired_indices[layer[i]]] = replacements[i];
                }
            }

            KeyVector sync_keys = retired_keys;
            std::sort(sync_keys.begin(), sync_keys.end());
            sync_keys.erase(std::unique(sync_keys.begin(), sync_keys.end()), sync_keys.end());
            const bool metadata_durable = indexer->Sync(sync_keys);
            metadata_durable_by_instance.emplace(internal_instance_id, metadata_durable);
            if (!metadata_durable) {
                // The in-memory view may already be retired, but without a
                // persistence barrier it is not safe to release the physical
                // allocation. Keep it in the pending queue and retry only the
                // persistence barrier; no physical Delete is issued first.
                KVCM_LOG_WARN("KVMeta reclaimer could not persist retired metadata for instance [%s]",
                              internal_instance_id.c_str());
            }
        }

        std::vector<RetiredItem> retired_items;
        retired_items.reserve(candidates.size());
        for (const auto &[internal_instance_id, indices] : by_instance) {
            std::vector<bool> removes_metadata_key(indices.size(), false);
            std::map<std::int64_t, std::vector<std::size_t>> retired_by_key;
            for (std::size_t i = 0; i < indices.size(); ++i) {
                retired_by_key[candidates[indices[i]].internal_key].push_back(i);
            }
            for (const auto &[_, relative_indices] : retired_by_key) {
                const bool selected_complete_key = std::any_of(
                    relative_indices.begin(), relative_indices.end(), [&](const std::size_t relative_index) {
                        return candidates[indices[relative_index]].removes_metadata_key;
                    });
                const bool all_retired =
                    std::all_of(relative_indices.begin(),
                                relative_indices.end(),
                                [&](const std::size_t relative_index) { return retired[indices[relative_index]]; });
                if (selected_complete_key && all_retired) {
                    removes_metadata_key[relative_indices.back()] = true;
                }
            }
            for (std::size_t i = 0; i < indices.size(); ++i) {
                const std::size_t candidate_index = indices[i];
                if (!retired[candidate_index]) {
                    continue;
                }
                const Candidate &candidate = candidates[candidate_index];
                const auto retired_location = finalized_locations[candidate_index];
                if (!retired_location) {
                    throw std::logic_error("missing finalized KVMeta retirement location");
                }
                retired_items.push_back(RetiredItem{internal_instance_id,
                                                    KvMetaManager::SessionItem{0,
                                                                               {},
                                                                               candidate.internal_key,
                                                                               candidate.location_id,
                                                                               retired_location,
                                                                               retired_location,
                                                                               candidate.value_size},
                                                    removes_metadata_key[i],
                                                    metadata_durable_by_instance.at(internal_instance_id)});
            }
        }
        return retired_items;
    }

    void AddPendingBatch(const std::string &instance_group,
                         std::size_t quota_shard,
                         std::chrono::steady_clock::time_point deadline,
                         std::vector<RetiredItem> items) {
        if (items.empty()) {
            throw std::invalid_argument("cannot publish an empty KVMeta pending batch");
        }
        auto batch = std::make_shared<PendingBatch>();
        batch->instance_group = instance_group;
        batch->quota_shard = quota_shard;
        batch->deadline = deadline;
        batch->items = std::move(items);
        std::optional<bool> batch_retry_safe;
        std::set<std::string> instances;
        for (const auto &item : batch->items) {
            instances.insert(item.internal_instance_id);
            batch->locations.emplace_back(item.internal_instance_id, item.item.location_id);
            bool item_retry_safe = false;
            if (!item.item.data_location ||
                !owner_->TryGetKvMetaDeleteRetrySafety(*item.item.data_location, item_retry_safe)) {
                batch->delete_policy_valid = false;
            } else if (batch_retry_safe && *batch_retry_safe != item_retry_safe) {
                // Mixing reusable and generation-safe addresses in one batch
                // would make a partial provider failure impossible to retry
                // safely. ReclaimGroup partitions them before publication;
                // keep this defensive check at the queue boundary too.
                batch->delete_policy_valid = false;
            } else {
                batch_retry_safe = item_retry_safe;
            }
        }
        batch->delete_retry_safe = batch_retry_safe.value_or(false);
        batch->instances.assign(instances.begin(), instances.end());
        std::uint64_t batch_bytes = 0;
        for (const auto &item : batch->items) {
            batch_bytes = SaturatingAdd(batch_bytes, item.item.value_size);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            batch->sequence = next_pending_sequence_;
            // Allocate every map node before publishing counters. This gives
            // the in-memory indexes a strong exception guarantee: either the
            // complete pending batch is visible, or no zero/partial marker is
            // left behind.
            try {
                for (const auto &instance : batch->instances) {
                    pending_instances_.try_emplace(instance, 0);
                }
                for (const auto &location : batch->locations) {
                    pending_locations_.try_emplace(location, 0);
                }
                pending_credits_.try_emplace(instance_group, PendingCredit{});
                auto &prepared_credit = pending_credits_.find(instance_group)->second;
                for (const auto &item : batch->items) {
                    if (item.removes_metadata_key) {
                        prepared_credit.keys_by_instance.try_emplace(item.internal_instance_id, 0);
                    }
                }
                const auto [_, inserted] =
                    pending_batches_.emplace(PendingDeadline{batch->deadline, batch->sequence}, batch);
                if (!inserted) {
                    throw std::logic_error("duplicate KVMeta pending sequence");
                }
            } catch (...) {
                for (const auto &instance : batch->instances) {
                    const auto it = pending_instances_.find(instance);
                    if (it != pending_instances_.end() && it->second == 0) {
                        pending_instances_.erase(it);
                    }
                }
                for (const auto &location : batch->locations) {
                    const auto it = pending_locations_.find(location);
                    if (it != pending_locations_.end() && it->second == 0) {
                        pending_locations_.erase(it);
                    }
                }
                const auto credit_it = pending_credits_.find(instance_group);
                if (credit_it != pending_credits_.end()) {
                    for (auto it = credit_it->second.keys_by_instance.begin();
                         it != credit_it->second.keys_by_instance.end();) {
                        if (it->second == 0) {
                            it = credit_it->second.keys_by_instance.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
                if (credit_it != pending_credits_.end() && credit_it->second.bytes == 0 &&
                    credit_it->second.keys == 0 && credit_it->second.blocked_batch_count == 0 &&
                    credit_it->second.keys_by_instance.empty() &&
                    std::none_of(credit_it->second.bytes_by_type.begin(),
                                 credit_it->second.bytes_by_type.end(),
                                 [](std::uint64_t value) { return value != 0; })) {
                    pending_credits_.erase(credit_it);
                }
                FailClosedMaintenance();
                throw;
            }
            ++next_pending_sequence_;
            for (const auto &instance : batch->instances) {
                ++pending_instances_.find(instance)->second;
            }
            for (const auto &location : batch->locations) {
                ++pending_locations_.find(location)->second;
            }
            ++pending_batch_count_;
            auto &credit = pending_credits_.find(instance_group)->second;
            for (const auto &item : batch->items) {
                credit.bytes = SaturatingAdd(credit.bytes, item.item.value_size);
                if (item.removes_metadata_key) {
                    credit.keys = SaturatingAdd(credit.keys, 1);
                    const auto instance_it = credit.keys_by_instance.find(item.internal_instance_id);
                    if (instance_it == credit.keys_by_instance.end()) {
                        FailClosedMaintenance();
                        throw std::logic_error("missing prepared KVMeta per-instance pending credit");
                    }
                    instance_it->second = SaturatingAdd(instance_it->second, 1);
                }
                if (item.item.data_location) {
                    const std::size_t type_index = ToIndex(ToBaseType(item.item.data_location->type()));
                    if (type_index < credit.bytes_by_type.size()) {
                        credit.bytes_by_type[type_index] =
                            SaturatingAdd(credit.bytes_by_type[type_index], item.item.value_size);
                    }
                }
            }
            pending_object_count_ = SaturatingAdd(pending_object_count_, batch->items.size());
            pending_bytes_ = SaturatingAdd(pending_bytes_, batch_bytes);
            retired_object_count_metrics_ += batch->items.size();
            UpdatePendingMetricsLocked();
            wake_requested_ = true;
        }
        condition_.notify_all();
    }

    void CompletePending(const std::shared_ptr<PendingBatch> &batch) noexcept {
        if (!batch) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &instance : batch->instances) {
            const auto it = pending_instances_.find(instance);
            if (it == pending_instances_.end() || it->second <= 1) {
                pending_instances_.erase(instance);
            } else {
                --it->second;
            }
        }
        for (const auto &location : batch->locations) {
            const auto it = pending_locations_.find(location);
            if (it == pending_locations_.end() || it->second <= 1) {
                pending_locations_.erase(location);
            } else {
                --it->second;
            }
        }
        if (batch->admission_blocked) {
            const auto it = pending_credits_.find(batch->instance_group);
            if (it != pending_credits_.end() && it->second.blocked_batch_count != 0) {
                --it->second.blocked_batch_count;
            }
            batch->admission_blocked = false;
        }
        const auto credit_it = pending_credits_.find(batch->instance_group);
        if (credit_it != pending_credits_.end()) {
            auto &credit = credit_it->second;
            for (const auto &item : batch->items) {
                credit.bytes = SaturatingSub(credit.bytes, item.item.value_size);
                if (item.removes_metadata_key) {
                    credit.keys = SaturatingSub(credit.keys, 1);
                    const auto instance_it = credit.keys_by_instance.find(item.internal_instance_id);
                    if (instance_it != credit.keys_by_instance.end()) {
                        instance_it->second = SaturatingSub(instance_it->second, 1);
                        if (instance_it->second == 0) {
                            credit.keys_by_instance.erase(instance_it);
                        }
                    }
                }
                if (item.item.data_location) {
                    const std::size_t type_index = ToIndex(ToBaseType(item.item.data_location->type()));
                    if (type_index < credit.bytes_by_type.size()) {
                        credit.bytes_by_type[type_index] =
                            SaturatingSub(credit.bytes_by_type[type_index], item.item.value_size);
                    }
                }
            }
            const bool has_type_credit = std::any_of(credit.bytes_by_type.begin(),
                                                     credit.bytes_by_type.end(),
                                                     [](std::uint64_t value) { return value != 0; });
            if (credit.bytes == 0 && credit.keys == 0 && credit.keys_by_instance.empty() && !has_type_credit &&
                credit.blocked_batch_count == 0) {
                pending_credits_.erase(credit_it);
            }
        }
        pending_object_count_ = SaturatingSub(pending_object_count_, batch->items.size());
        pending_batch_count_ = SaturatingSub(pending_batch_count_, 1);
        for (const auto &item : batch->items) {
            pending_bytes_ = SaturatingSub(pending_bytes_, item.item.value_size);
        }
        UpdatePendingMetricsLocked();
    }

    void BlockAdmission(const std::shared_ptr<PendingBatch> &batch) noexcept {
        if (!batch) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!batch->admission_blocked) {
            const auto credit_it = pending_credits_.find(batch->instance_group);
            if (credit_it == pending_credits_.end()) {
                // Losing this fence after an ambiguous metadata persistence
                // outcome could admit a successor allocation and create an ABA
                // delete race. Fail closed for KVMeta only; ordinary KV-cache
                // traffic does not consult maintenance_cancelled_.
                KVCM_LOG_ERROR("KVMeta reclaimer lost pending credit while blocking admission");
                ++error_count_metrics_;
                FailClosedMaintenance();
                return;
            }
            batch->admission_blocked = true;
            if (credit_it->second.blocked_batch_count != std::numeric_limits<std::size_t>::max()) {
                ++credit_it->second.blocked_batch_count;
            }
            UpdatePendingMetricsLocked();
        }
    }

    std::vector<std::shared_ptr<PendingBatch>> TakeDueBatches() {
        std::vector<std::shared_ptr<PendingBatch>> due;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = std::chrono::steady_clock::now();
        std::size_t due_count = 0;
        for (auto it = pending_batches_.begin(); it != pending_batches_.end() && it->first.first <= now; ++it) {
            ++due_count;
        }
        // Reserve before mutating the queue. If allocation fails, every batch
        // remains indexed and the outer loop can retry without losing its
        // pending markers or quota credit.
        due.reserve(due_count);
        while (!pending_batches_.empty() && pending_batches_.begin()->first.first <= now) {
            due.push_back(std::move(pending_batches_.begin()->second));
            pending_batches_.erase(pending_batches_.begin());
        }
        return due;
    }

    void ReschedulePending(const std::shared_ptr<PendingBatch> &batch) {
        if (!batch) {
            return;
        }
        constexpr std::uint64_t kMinimumRetryDelayMs = 100;
        constexpr std::uint64_t kMaximumRetryDelayMs = 30'000;
        const std::uint64_t base_delay_ms = std::max<std::uint64_t>(kMinimumRetryDelayMs, IdleIntervalMs());
        const std::uint32_t shift = std::min<std::uint32_t>(batch->retry_count, 8);
        const std::uint64_t retry_delay_ms = std::min<std::uint64_t>(kMaximumRetryDelayMs, base_delay_ms << shift);
        const auto retry_delay = std::chrono::milliseconds(retry_delay_ms);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            batch->deadline = std::chrono::steady_clock::now() + retry_delay;
            batch->sequence = next_pending_sequence_++;
            if (batch->retry_count != std::numeric_limits<std::uint32_t>::max()) {
                ++batch->retry_count;
            }
            try {
                const auto [_, inserted] =
                    pending_batches_.emplace(PendingDeadline{batch->deadline, batch->sequence}, batch);
                if (!inserted) {
                    throw std::logic_error("duplicate KVMeta retry sequence");
                }
            } catch (...) {
                // The durable retired marker must never be left without an
                // in-memory finalizer while this leader continues admitting
                // writes. Recovery on the next leader owns it after this
                // KVMeta-only fail-closed transition.
                FailClosedMaintenance();
                throw;
            }
            wake_requested_ = true;
        }
        condition_.notify_all();
    }

    bool EnsureRetiredMetadataDurable(const std::shared_ptr<PendingBatch> &batch) {
        std::map<std::string, KeyVector> keys_by_instance;
        for (const auto &retired : batch->items) {
            if (!retired.metadata_durable) {
                keys_by_instance[retired.internal_instance_id].push_back(retired.item.internal_key);
            }
        }
        for (auto &[internal_instance_id, keys] : keys_by_instance) {
            std::sort(keys.begin(), keys.end());
            keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
            const auto indexer = owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
            if (!indexer || !indexer->Sync(keys)) {
                return false;
            }
            for (auto &retired : batch->items) {
                if (retired.internal_instance_id == internal_instance_id) {
                    retired.metadata_durable = true;
                }
            }
        }
        return true;
    }

    void FinalizePending(const std::shared_ptr<PendingBatch> &batch) {
        if (!batch) {
            return;
        }
        if (ShouldStop()) {
            CompletePending(batch);
            return;
        }
        if (batch->quota_shard >= owner_->quota_admission_mutexes_.size()) {
            KVCM_LOG_ERROR("KVMeta reclaimer pending batch has an invalid quota shard");
            // This batch already owns durable retired metadata. Dropping its
            // only finalizer while continuing admission would strand that
            // ownership transition outside both the pending indexes and the
            // recovery scan contract. Close only KVMeta and let recovery
            // rebuild the state from metadata.
            FailClosedMaintenance();
            CompletePending(batch);
            return;
        }

        std::map<std::string, std::vector<std::size_t>> item_indices_by_instance;
        std::vector<KvMetaManager::SessionItem> all_items;
        all_items.reserve(batch->items.size());
        for (std::size_t i = 0; i < batch->items.size(); ++i) {
            item_indices_by_instance[batch->items[i].internal_instance_id].push_back(i);
            all_items.push_back(batch->items[i].item);
        }

        std::unique_lock<std::mutex> quota_lock(owner_->quota_admission_mutexes_[batch->quota_shard]);
        if (ShouldStop()) {
            CompletePending(batch);
            return;
        }
        if (!batch->delete_policy_valid) {
            ++error_count_metrics_;
            KVCM_LOG_ERROR("KVMeta reclaimer cannot bind a retired location to a valid delete policy; "
                           "retaining its durable tombstone and failing maintenance closed");
            FailClosedMaintenance();
            CompletePending(batch);
            return;
        }
        if (!EnsureRetiredMetadataDurable(batch)) {
            ++retry_count_metrics_;
            quota_lock.unlock();
            if (ShouldStop()) {
                CompletePending(batch);
            } else {
                ReschedulePending(batch);
            }
            return;
        }

        std::uint64_t reclaimed_bytes = 0;
        for (const auto &item : batch->items) {
            reclaimed_bytes = SaturatingAdd(reclaimed_bytes, item.item.value_size);
        }

        // The persisted CLS_DELETING records fence readers and new values.
        // Retry only when the backend explicitly guarantees that the same URI
        // cannot name a successor generation. A legacy reusable GA receives
        // one attempt; ambiguity is recorded as a possible physical orphan.
        if (!batch->physical_delete_finished) {
            // Slow backend I/O must not serialize unrelated PutStart calls in
            // the same EMB group. The durable tombstone plus pending-location
            // index already fences this exact key, and its bytes remain in
            // authoritative usage accounting. Reacquire the shard only for
            // the final exact metadata transition.
            quota_lock.unlock();
            physical_delete_attempted_object_count_metrics_ += batch->items.size();
            ErrorCode physical_ec = EC_IO_ERROR;
            const char *failure_kind = "error_code";
            try {
                RequestContext request_context("kv_meta_reclaimer_physical_delete");
                physical_ec = owner_->DeleteAllocatedLocations(&request_context, all_items);
            } catch (const std::exception &) { failure_kind = "standard_exception"; } catch (...) {
                failure_kind = "unknown_exception";
            }
            if (physical_ec != EC_OK) {
                ++error_count_metrics_;
                physical_delete_uncertain_object_count_metrics_ += batch->items.size();
                physical_delete_uncertain_bytes_metrics_ += reclaimed_bytes;
                if (batch->delete_retry_safe) {
                    ++retry_count_metrics_;
                    KVCM_LOG_WARN("KVMeta reclaimer retry-safe physical cleanup will be retried, item_count[%zu], "
                                  "failure[%s], ec[%d]",
                                  all_items.size(),
                                  failure_kind,
                                  physical_ec);
                    if (ShouldStop()) {
                        // Metadata remains durably retired. Leader recovery
                        // will rediscover this retry-safe ownership record.
                        CompletePending(batch);
                    } else {
                        ReschedulePending(batch);
                    }
                    return;
                }
                KVCM_LOG_WARN("KVMeta reclaimer will not replay ambiguous reusable-address delete, "
                              "item_count[%zu], failure[%s], ec[%d]",
                              all_items.size(),
                              failure_kind,
                              physical_ec);
                // Continuing reclamation could convert every live cache
                // entry into an untracked physical orphan during one PACE
                // outage. Stop only KVMeta; recovery will metadata-finalize
                // this durable tombstone without replaying the GA.
                FailClosedMaintenance();
                batch->physical_delete_finished = true;
            } else {
                batch->physical_delete_finished = true;
                ConfirmBackendCapacityFreed(batch->instance_group, batch->items);
            }
            quota_lock.lock();
            if (ShouldStop()) {
                CompletePending(batch);
                return;
            }
        }

        try {
            RequestContext request_context("kv_meta_reclaimer_finalize");
            for (const auto &[internal_instance_id, item_indices] : item_indices_by_instance) {
                std::vector<KvMetaManager::SessionItem> items;
                items.reserve(item_indices.size());
                for (const std::size_t index : item_indices) {
                    items.push_back(batch->items[index].item);
                }
                const auto cleanup = owner_->DeleteRetiredMetadata(&request_context, internal_instance_id, items);
                if (cleanup.metadata_deleted.size() != item_indices.size() ||
                    cleanup.metadata_absent.size() != item_indices.size() ||
                    cleanup.metadata_conflicted.size() != item_indices.size()) {
                    throw std::runtime_error("KVMeta reclaimer metadata cleanup returned malformed evidence");
                }
                bool unexpected_owner = false;
                for (std::size_t i = 0; i < item_indices.size(); ++i) {
                    auto &retired = batch->items[item_indices[i]];
                    if (cleanup.metadata_deleted[i]) {
                        retired.metadata_delete_applied = true;
                    }
                    if ((cleanup.metadata_absent[i] && !retired.metadata_delete_applied) ||
                        cleanup.metadata_conflicted[i]) {
                        unexpected_owner = true;
                    }
                }
                if (unexpected_owner) {
                    // Physical cleanup was already attempted under this
                    // backend's retry policy using the allocation identity
                    // captured in the tombstone. The metadata owner transition
                    // is nevertheless out of protocol and accounting can no
                    // longer be certified. Fail closed rather than CAS-deleting
                    // the unexpected owner.
                    ++error_count_metrics_;
                    KVCM_LOG_ERROR("KVMeta reclaimer observed an unexpected missing or replaced owner for instance "
                                   "[%s] after physical cleanup",
                                   internal_instance_id.c_str());
                    FailClosedMaintenance();
                    quota_lock.unlock();
                    CompletePending(batch);
                    return;
                }
                if (cleanup.ec != EC_OK || !cleanup.metadata_cleanup_complete) {
                    ++retry_count_metrics_;
                    KVCM_LOG_WARN("KVMeta reclaimer metadata cleanup will be retried for instance [%s], "
                                  "item_count[%zu], ec[%d]",
                                  internal_instance_id.c_str(),
                                  items.size(),
                                  cleanup.ec);
                    // The exact delete may already have removed metadata from
                    // the in-memory view before its Sync failed. Close new
                    // admission for this KVMeta group before releasing the
                    // shard; existing sessions may still finish safely.
                    BlockAdmission(batch);
                    quota_lock.unlock();
                    if (ShouldStop()) {
                        CompletePending(batch);
                    } else {
                        ReschedulePending(batch);
                    }
                    return;
                }
            }
        } catch (...) {
            BlockAdmission(batch);
            throw;
        }
        // Logical quota is released after the final metadata persistence
        // barrier. For an at-most-once backend, physical failure remains a
        // separately metered orphan and never authorizes a destructive retry.
        reclaimed_object_count_metrics_ += batch->items.size();
        reclaimed_bytes_metrics_ += reclaimed_bytes;
        CompletePending(batch);
    }

    bool ReclaimGroup(RequestContext *request_context, const std::shared_ptr<const InstanceGroup> &group) {
        if (!group || !HasSupportedKvMetaReclaimConfiguration(*group, owner_->limits_.max_write_timeout_seconds) ||
            !HasSupportedKvMetaReadHeatTracking(*group)) {
            if (group) {
                KVCM_INTERVAL_LOG_WARN(
                    10,
                    "KVMeta reclaimer skipped group [%s] with an unsupported reclaim or read-heat configuration",
                    group->name().c_str());
            }
            return false;
        }
        const auto &strategy = group->cache_config()->reclaim_strategy();
        const double threshold = strategy->trigger_strategy().used_percentage();
        const auto [instances_ec, all_instances] =
            owner_->registry_manager_->ListInstanceInfo(request_context, group->name());
        if (instances_ec != EC_OK || all_instances.empty()) {
            return false;
        }
        std::vector<InstanceInfoConstPtr> instances;
        instances.reserve(all_instances.size());
        for (const auto &instance : all_instances) {
            if (!instance || !IsKvMetaInstance(*instance)) {
                // Never let the side-path worker scan or delete an ordinary KV
                // cache group, even if configuration was changed after KVMeta
                // registration.
                return false;
            }
            instances.push_back(instance);
        }

        Pressure pressure;
        if (!ReadPressure(request_context, *group, instances, threshold, pressure) || !pressure.Any()) {
            return false;
        }
        const auto [configured_sampling_size, configured_batch_size] = SamplingAndBatchSize();
        // Candidate materialization and pending ownership use the same hard
        // object bound. Clamp shared CacheReclaimer knobs locally so an
        // otherwise valid large main-path setting cannot make every KVMeta
        // reclaim round reject its own bounded candidate set.
        const std::size_t sampling_size = std::min<std::size_t>(configured_sampling_size, kPendingObjectLimit);
        const std::size_t batch_size = std::min<std::size_t>(configured_batch_size, kPendingObjectLimit);
        if (sampling_size == 0 || batch_size == 0) {
            KVCM_INTERVAL_LOG_WARN(10,
                                   "KVMeta reclaimer cannot make progress for group [%s]: sample[%zu], batch[%zu]",
                                   group->name().c_str(),
                                   configured_sampling_size,
                                   configured_batch_size);
            return false;
        }
        std::vector<CandidateKey> candidate_keys;
        if (!CollectCandidates(request_context, group->name(), instances, pressure, sampling_size, candidate_keys)) {
            KVCM_INTERVAL_LOG_WARN(
                10, "KVMeta reclaimer failed to collect exact LRU candidates for group [%s]", group->name().c_str());
            return false;
        }
        const auto delay = std::chrono::milliseconds(strategy->delay_before_delete_ms());
        const std::size_t quota_shard =
            std::hash<std::string>{}(group->name()) % owner_->quota_admission_mutexes_.size();
        std::vector<RetiredItem> retired;
        {
            std::unique_lock<std::mutex> quota_lock(owner_->quota_admission_mutexes_[quota_shard]);
            if (ShouldStop()) {
                return false;
            }
            const auto &trimming = owner_->trimming_instances_[quota_shard];
            if (std::any_of(instances.begin(), instances.end(), [&](const auto &instance) {
                    return instance && trimming.count(instance->instance_id()) != 0;
                })) {
                return false;
            }
            // Pressure and admission can change while sampling. Recheck under
            // the same group shard used by Put/Remove/Trim before changing any
            // metadata, so a completed concurrent delete cannot cause an
            // unnecessary retirement.
            Pressure current_pressure;
            if (!ReadPressure(request_context, *group, instances, threshold, current_pressure) ||
                !current_pressure.Any()) {
                return false;
            }
            const PendingCapacity pending_capacity = RemainingPendingCapacity();
            if (pending_capacity.batch_slots == 0 || pending_capacity.object_count == 0 ||
                pending_capacity.bytes == 0) {
                ++pending_limit_reject_count_metrics_;
                KVCM_INTERVAL_LOG_WARN(
                    10, "KVMeta reclaimer pending limit reached for group [%s]", group->name().c_str());
                return false;
            }
            // A selection can contain both retry-safe and at-most-once items,
            // and each policy needs its own bounded batch. With S free slots,
            // (S - 1) * B + 1 items fit for every possible two-way partition;
            // this keeps a nearly full pending queue making progress instead
            // of repeatedly selecting a vector that cannot be published.
            const std::size_t batch_slot_object_capacity =
                (pending_capacity.batch_slots - 1) * kKvMetaDeleteBatchSize + 1;
            auto selected = SelectCandidates(std::move(candidate_keys),
                                             current_pressure,
                                             std::min({batch_size,
                                                       pending_capacity.object_count,
                                                       batch_slot_object_capacity}),
                                             pending_capacity.bytes);
            if (selected.empty()) {
                return false;
            }
            std::size_t retry_safe_count = 0;
            std::size_t at_most_once_count = 0;
            for (const auto &candidate : selected) {
                bool retry_safe = false;
                if (!candidate.location || !owner_->TryGetKvMetaDeleteRetrySafety(*candidate.location, retry_safe)) {
                    AddError(request_context, "KVMeta reclaim candidate has no valid delete policy");
                    FailClosedMaintenance();
                    return false;
                }
                if (retry_safe) {
                    ++retry_safe_count;
                } else {
                    ++at_most_once_count;
                }
            }
            const std::size_t required_batch_count =
                DeleteBatchCount(retry_safe_count) + DeleteBatchCount(at_most_once_count);
            if (required_batch_count > pending_capacity.batch_slots ||
                !HasPendingCapacity(selected, required_batch_count)) {
                ++pending_limit_reject_count_metrics_;
                KVCM_INTERVAL_LOG_WARN(10,
                                       "KVMeta reclaimer pending limit reached for group [%s], selected[%zu]",
                                       group->name().c_str(),
                                       selected.size());
                return false;
            }

            std::chrono::steady_clock::time_point finalization_deadline;
            try {
                retired = RetireCandidates(request_context, selected, delay, finalization_deadline);
            } catch (...) {
                // Retirement may already be durable. Stop only the KVMeta side
                // before releasing the group shard so another round cannot
                // over-evict without pending credit; leader recovery owns any
                // persisted retired record.
                FailClosedMaintenance();
                throw;
            }
            if (!retired.empty()) {
                std::vector<RetiredItem> at_most_once_items;
                std::vector<RetiredItem> retry_safe_items;
                at_most_once_items.reserve(retired.size());
                retry_safe_items.reserve(retired.size());
                for (auto &item : retired) {
                    bool retry_safe = false;
                    if (!item.item.data_location ||
                        !owner_->TryGetKvMetaDeleteRetrySafety(*item.item.data_location, retry_safe)) {
                        FailClosedMaintenance();
                        throw std::logic_error("retired KVMeta object lost its delete policy");
                    }
                    (retry_safe ? retry_safe_items : at_most_once_items).push_back(std::move(item));
                }
                // Publish the pending marker before releasing the same group
                // shard observed by Trim. Otherwise Trim could see the
                // durable CLS_DELETING state in the tiny retire/marker window
                // and bypass the configured read grace period.
                try {
                    const auto publish_bounded_batches = [&](std::vector<RetiredItem> items) {
                        for (std::size_t begin = 0; begin < items.size(); begin += kKvMetaDeleteBatchSize) {
                            const std::size_t count = std::min(kKvMetaDeleteBatchSize, items.size() - begin);
                            std::vector<RetiredItem> batch_items;
                            batch_items.reserve(count);
                            for (std::size_t index = begin; index < begin + count; ++index) {
                                batch_items.push_back(std::move(items[index]));
                            }
                            AddPendingBatch(
                                group->name(), quota_shard, finalization_deadline, std::move(batch_items));
                        }
                    };
                    // Publish reusable addresses first. If their one allowed
                    // attempt becomes ambiguous and closes maintenance, every
                    // retry-safe tombstone remains recoverable without ever
                    // inheriting the reusable-address policy.
                    if (!at_most_once_items.empty()) {
                        publish_bounded_batches(std::move(at_most_once_items));
                    }
                    if (!retry_safe_items.empty()) {
                        publish_bounded_batches(std::move(retry_safe_items));
                    }
                } catch (...) {
                    // AddPendingBatch can allocate before it reaches its
                    // internally guarded map publication. At this point the
                    // CAS+Sync retirement may already be durable, so every
                    // exception must close KVMeta admission before the group
                    // shard is released.
                    FailClosedMaintenance();
                    KVCM_LOG_ERROR("KVMeta reclaimer could not publish pending state for group [%s]; "
                                   "KVMeta maintenance is now fail-closed",
                                   group->name().c_str());
                    throw;
                }
                return true;
            }
        }
        return false;
    }

    bool ReclaimRound() {
        if (ShouldStop()) {
            return false;
        }
        // Share the existing operational pause switch. A pause stops new
        // retirements, while already-retired batches still pass through the
        // metadata/physical finalization path above, matching CacheReclaimer's
        // handling of accepted deletes.
        const auto cache_reclaimer = owner_->cache_manager_->cache_reclaimer();
        if (cache_reclaimer && cache_reclaimer->IsPaused()) {
            return false;
        }
        RequestContext request_context("kv_meta_reclaimer");
        ++round_count_metrics_;
        const std::vector<std::string> group_names = owner_->SnapshotKvMetaGroups();
        const std::set<std::string> active_group_names(group_names.begin(), group_names.end());
        for (auto it = sampling_rotation_by_group_.begin(); it != sampling_rotation_by_group_.end();) {
            if (active_group_names.count(it->first) == 0) {
                it = sampling_rotation_by_group_.erase(it);
            } else {
                ++it;
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = admission_demands_.begin(); it != admission_demands_.end();) {
                if (active_group_names.count(it->first) == 0) {
                    it = admission_demands_.erase(it);
                } else {
                    ++it;
                }
            }
            UpdateAdmissionDemandMetricsLocked();
        }
        bool made_progress = false;
        for (const auto &group_name : group_names) {
            if (ShouldStop()) {
                break;
            }
            const auto [group_ec, group] = owner_->registry_manager_->GetInstanceGroup(&request_context, group_name);
            if (group_ec != EC_OK || !group) {
                KVCM_INTERVAL_LOG_WARN(
                    10, "KVMeta reclaimer failed to load tracked group [%s], ec[%d]", group_name.c_str(), group_ec);
                continue;
            }
            try {
                made_progress = ReclaimGroup(&request_context, group) || made_progress;
            } catch (const std::exception &) {
                ++error_count_metrics_;
                KVCM_LOG_WARN("KVMeta reclaimer contained a standard provider exception for group [%s]",
                              group ? group->name().c_str() : "<null>");
            } catch (...) {
                ++error_count_metrics_;
                KVCM_LOG_WARN("KVMeta reclaimer contained an unknown provider exception for group [%s]",
                              group ? group->name().c_str() : "<null>");
            }
        }
        return made_progress;
    }

    void Loop() noexcept {
        for (;;) {
            try {
                const auto interval = std::chrono::milliseconds(IdleIntervalMs());
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    auto wake_deadline = std::chrono::steady_clock::now() + interval;
                    if (!pending_batches_.empty()) {
                        wake_deadline = std::min(wake_deadline, pending_batches_.begin()->first.first);
                    }
                    condition_.wait_until(lock, wake_deadline, [this]() { return stopping_ || wake_requested_; });
                    if (stopping_) {
                        break;
                    }
                    wake_requested_ = false;
                }
                for (const auto &batch : TakeDueBatches()) {
                    try {
                        FinalizePending(batch);
                    } catch (const std::exception &) {
                        ++error_count_metrics_;
                        KVCM_LOG_WARN("KVMeta reclaimer contained a standard finalization exception");
                        if (ShouldStop()) {
                            CompletePending(batch);
                        } else {
                            ReschedulePending(batch);
                        }
                    } catch (...) {
                        ++error_count_metrics_;
                        KVCM_LOG_WARN("KVMeta reclaimer contained an unknown finalization exception");
                        if (ShouldStop()) {
                            CompletePending(batch);
                        } else {
                            ReschedulePending(batch);
                        }
                    }
                }
                if (ShouldStop()) {
                    break;
                }
                const bool made_progress = ReclaimRound();
                if (made_progress) {
                    // Re-evaluate watermarks immediately after a bounded batch.
                    Wake();
                }
            } catch (const std::exception &) {
                ++error_count_metrics_;
                KVCM_LOG_WARN("KVMeta reclaimer loop contained a standard exception");
                try {
                    std::unique_lock<std::mutex> lock(mutex_);
                    wake_requested_ = false;
                    condition_.wait_for(lock, std::chrono::milliseconds(100), [this]() { return stopping_; });
                    if (stopping_) {
                        break;
                    }
                } catch (...) { break; }
            } catch (...) {
                ++error_count_metrics_;
                KVCM_LOG_WARN("KVMeta reclaimer loop contained an unknown exception");
                try {
                    std::unique_lock<std::mutex> lock(mutex_);
                    wake_requested_ = false;
                    condition_.wait_for(lock, std::chrono::milliseconds(100), [this]() { return stopping_; });
                    if (stopping_) {
                        break;
                    }
                } catch (...) { break; }
            }
        }
    }

    KvMetaManager *owner_ = nullptr;
    mutable std::mutex lifecycle_mutex_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool stopping_ = true;
    bool wake_requested_ = false;
    std::map<std::string, std::size_t> pending_instances_;
    std::map<std::pair<std::string, std::string>, std::size_t> pending_locations_;
    std::map<std::string, PendingCredit> pending_credits_;
    std::map<std::string, AdmissionDemand> admission_demands_;
    std::map<PendingDeadline, std::shared_ptr<PendingBatch>> pending_batches_;
    std::map<std::string, std::size_t> sampling_rotation_by_group_;
    std::uint64_t next_pending_sequence_ = 0;
    std::uint64_t next_admission_demand_sequence_ = 0;
    std::uint64_t pending_batch_count_ = 0;
    std::uint64_t pending_object_count_ = 0;
    std::uint64_t pending_bytes_ = 0;
    Counter round_count_metrics_;
    Counter retired_object_count_metrics_;
    Counter reclaimed_object_count_metrics_;
    Counter reclaimed_bytes_metrics_;
    Counter retry_count_metrics_;
    Counter error_count_metrics_;
    Counter pending_limit_reject_count_metrics_;
    Counter admission_demand_count_metrics_;
    Counter backend_capacity_demand_count_metrics_;
    Counter physical_delete_attempted_object_count_metrics_;
    Counter physical_delete_uncertain_object_count_metrics_;
    Counter physical_delete_uncertain_bytes_metrics_;
    Counter maintenance_touch_key_count_metrics_;
    Gauge pending_batch_count_metrics_;
    Gauge pending_object_count_metrics_;
    Gauge pending_bytes_metrics_;
    Gauge blocked_group_count_metrics_;
    Gauge admission_demand_group_count_metrics_;
    Gauge backend_capacity_demand_bytes_metrics_;
    std::thread thread_;
};

KvMetaManager::KvMetaManager(std::shared_ptr<CacheManager> cache_manager,
                             std::shared_ptr<RegistryManager> registry_manager)
    : KvMetaManager(std::move(cache_manager), std::move(registry_manager), Limits{}) {}

KvMetaManager::KvMetaManager(std::shared_ptr<CacheManager> cache_manager,
                             std::shared_ptr<RegistryManager> registry_manager,
                             Limits limits)
    : cache_manager_(std::move(cache_manager)), registry_manager_(std::move(registry_manager)), limits_(limits) {}

KvMetaManager::~KvMetaManager() { Shutdown(); }

bool KvMetaManager::Init() {
    if (initialized_.load(std::memory_order_acquire)) {
        return true;
    }
    if (!cache_manager_ || !registry_manager_ || !cache_manager_->meta_indexer_manager() ||
        !registry_manager_->data_storage_manager() || limits_.max_batch_items == 0 || limits_.max_key_bytes == 0 ||
        limits_.max_instance_id_bytes == 0 || limits_.max_instance_group_bytes == 0 ||
        limits_.max_write_session_id_bytes == 0 || limits_.max_user_data_bytes == 0 ||
        limits_.max_location_uri_bytes == 0 || limits_.max_location_uri_bytes > kMaxKvMetaLocationUriBytes ||
        limits_.max_active_write_sessions == 0 || limits_.max_value_bytes == 0 || limits_.max_batch_bytes == 0 ||
        limits_.max_write_timeout_seconds <= 0 || limits_.max_failed_write_cleanup_grace_seconds < 0 ||
        limits_.max_write_timeout_seconds > std::numeric_limits<std::int32_t>::max() ||
        limits_.max_failed_write_cleanup_grace_seconds >
            std::numeric_limits<std::int32_t>::max() - limits_.max_write_timeout_seconds) {
        KVCM_LOG_ERROR("KVMeta manager init failed: dependency or limits are invalid");
        return false;
    }
    data_storage_selector_ =
        std::make_unique<DataStorageSelector>(cache_manager_->meta_indexer_manager(), registry_manager_);
    reclaimer_ = std::make_unique<KvMetaReclaimer>(this);
    write_session_manager_ = std::make_unique<KvMetaWriteSessionManager>(this, limits_.max_active_write_sessions);
    if (!write_session_manager_->Start()) {
        write_session_manager_.reset();
        reclaimer_.reset();
        data_storage_selector_.reset();
        return false;
    }
    maintenance_cancelled_.store(false, std::memory_order_release);
    initialized_.store(true, std::memory_order_release);
    return true;
}

void KvMetaManager::Shutdown() {
    CancelMaintenance();
    initialized_.store(false, std::memory_order_release);
    if (reclaimer_) {
        reclaimer_->StopAndJoin();
        reclaimer_.reset();
    }
    if (write_session_manager_) {
        write_session_manager_->StopAndDiscard();
        write_session_manager_.reset();
    }
    data_storage_selector_.reset();
}

void KvMetaManager::DoCleanup() {
    CancelMaintenance();
    if (reclaimer_) {
        reclaimer_->StopAndJoin();
    }
    if (write_session_manager_) {
        write_session_manager_->StopAndDiscard();
    }
}

void KvMetaManager::CancelMaintenance() noexcept {
    maintenance_cancelled_.store(true, std::memory_order_release);
    if (write_session_manager_) {
        write_session_manager_->RequestStop();
    }
    if (reclaimer_) {
        reclaimer_->RequestStop();
    }
}

bool KvMetaManager::ResumeMaintenance() {
    if (!initialized_.load(std::memory_order_acquire) || !write_session_manager_ || !reclaimer_) {
        return false;
    }
    if (!write_session_manager_->Start()) {
        return false;
    }
    maintenance_cancelled_.store(false, std::memory_order_release);
    if (!reclaimer_->Start()) {
        maintenance_cancelled_.store(true, std::memory_order_release);
        write_session_manager_->RequestStop();
        reclaimer_->RequestStop();
        return false;
    }
    return true;
}

std::string KvMetaManager::InternalInstanceId(const std::string &instance_id) {
    return std::string(kKvMetaInternalInstancePrefix) + HexEncode(instance_id);
}

std::int64_t KvMetaManager::InternalKey(const std::string &key) {
    const std::uint64_t hash = Hash64(key.data(), key.size(), kObjectKeyHashSeed);
    std::int64_t result = 0;
    static_assert(sizeof(result) == sizeof(hash));
    std::memcpy(&result, &hash, sizeof(result));
    return result;
}

std::string KvMetaManager::StableLocationId(const std::string &key) {
    return std::string(kKvMetaLocationIdPrefix) + HexEncode(key);
}

void KvMetaManager::RememberKvMetaGroup(const std::string &instance_group) {
    if (instance_group.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(kv_meta_groups_mutex_);
    kv_meta_groups_.insert(instance_group);
}

std::vector<std::string> KvMetaManager::SnapshotKvMetaGroups() const {
    std::lock_guard<std::mutex> lock(kv_meta_groups_mutex_);
    std::vector<std::string> groups(kv_meta_groups_.begin(), kv_meta_groups_.end());
    std::sort(groups.begin(), groups.end());
    return groups;
}

void KvMetaManager::ReplaceKvMetaGroups(std::unordered_set<std::string> instance_groups) {
    std::lock_guard<std::mutex> lock(kv_meta_groups_mutex_);
    kv_meta_groups_ = std::move(instance_groups);
}

bool KvMetaManager::IsOwnedLocation(std::int64_t internal_key, const std::string &location_id) const {
    if (location_id.size() <= kKvMetaLocationIdPrefix.size() ||
        location_id.compare(0, kKvMetaLocationIdPrefix.size(), kKvMetaLocationIdPrefix) != 0) {
        return false;
    }
    const std::string_view encoded_key = std::string_view(location_id).substr(kKvMetaLocationIdPrefix.size());
    if ((encoded_key.size() & 1U) != 0 || encoded_key.size() / 2 > limits_.max_key_bytes) {
        return false;
    }
    std::string original_key;
    if (!HexDecode(encoded_key, original_key)) {
        return false;
    }
    return StableLocationId(original_key) == location_id && InternalKey(original_key) == internal_key;
}

ErrorCode KvMetaManager::ValidateOwnedLocation(RequestContext *request_context,
                                               const std::string &internal_instance_id,
                                               std::int64_t internal_key,
                                               const std::string &location_id,
                                               const CacheLocation &location,
                                               std::uint64_t &value_size) const {
    const auto data_storage_manager = registry_manager_->data_storage_manager();
    const DataStorageUri location_uri(location.location_specs().empty() ? std::string{}
                                                                        : location.location_specs().front().uri());
    const auto location_backend =
        data_storage_manager ? data_storage_manager->GetDataStorageBackend(location_uri.GetHostName()) : nullptr;
    const bool known_state = (location.status() == CLS_NEW && location.create_time() != 0) || IsRetiredObject(location);
    std::uint64_t validated_total_size = 0;
    const bool has_validated_total_size = location.GetValidatedTotalSize(validated_total_size);
    if (!IsOwnedLocation(internal_key, location_id) || location.id() != location_id || !known_state ||
        location.location_specs().size() != 1 ||
        location.location_specs().front().uri().size() > limits_.max_location_uri_bytes ||
        !HasMatchingStorageBackend(location, data_storage_manager) ||
        !UriBelongsToKvMetaNamespace(
            location_uri, location_backend, location.type(), internal_instance_id, internal_key) ||
        !ReadLogicalSize(location, value_size) || (has_validated_total_size && validated_total_size != value_size) ||
        value_size > limits_.max_value_bytes || value_size > std::numeric_limits<std::size_t>::max()) {
        AddError(request_context, "KVMeta location does not match its exact key or registered storage backend");
        return EC_CORRUPTION;
    }
    return EC_OK;
}

ErrorCode KvMetaManager::ValidateInstanceId(RequestContext *request_context, const std::string &instance_id) const {
    if (!request_context || instance_id.empty() || instance_id.size() > limits_.max_instance_id_bytes) {
        AddError(request_context, "KVMeta instance_id is empty or exceeds max_instance_id_bytes");
        return EC_BADARGS;
    }
    return EC_OK;
}

std::pair<ErrorCode, std::shared_ptr<const InstanceInfo>>
KvMetaManager::GetValidatedInstanceInfo(RequestContext *request_context, const std::string &instance_id) const {
    if (const ErrorCode ec = ValidateInstanceId(request_context, instance_id); ec != EC_OK) {
        return {ec, nullptr};
    }
    auto info = registry_manager_->GetInstanceInfo(request_context, InternalInstanceId(instance_id));
    if (!info) {
        return {EC_INSTANCE_NOT_EXIST, nullptr};
    }
    if (!IsKvMetaInstance(*info)) {
        AddError(request_context, "KVMeta internal instance marker does not match the generic-object schema");
        return {EC_CORRUPTION, nullptr};
    }
    return {EC_OK, std::move(info)};
}

ErrorCode KvMetaManager::ValidateKeys(RequestContext *request_context, const std::vector<std::string> &keys) const {
    if (!request_context || keys.empty() || keys.size() > limits_.max_batch_items) {
        AddError(request_context, "KVMeta keys must be non-empty and within max_batch_items");
        return EC_BADARGS;
    }
    std::unordered_set<std::string> unique_keys;
    unique_keys.reserve(keys.size());
    for (const auto &key : keys) {
        if (key.empty() || key.size() > limits_.max_key_bytes) {
            AddError(request_context, "KVMeta key is empty or exceeds max_key_bytes");
            return EC_BADARGS;
        }
        if (!unique_keys.insert(key).second) {
            AddError(request_context, "KVMeta request contains duplicate keys");
            return EC_DUPLICATE_ENTITY;
        }
    }
    return EC_OK;
}

ErrorCode KvMetaManager::ValidateCacheConfiguration(RequestContext *request_context,
                                                    const std::string &instance_group) const {
    if (!request_context || instance_group.empty()) {
        return EC_BADARGS;
    }
    const auto [group_ec, group] = registry_manager_->GetInstanceGroup(request_context, instance_group);
    if (group_ec != EC_OK || !group) {
        return group_ec == EC_OK ? EC_INSTANCE_NOT_EXIST : group_ec;
    }
    if (!HasSupportedKvMetaReclaimConfiguration(*group, limits_.max_write_timeout_seconds)) {
        AddError(request_context,
                 "KVMeta requires an LRU reclaim strategy with a valid watermark and read-grace delay");
        return EC_CONFIG_ERROR;
    }
    if (!HasSupportedKvMetaReadHeatTracking(*group)) {
        AddError(request_context,
                 "KVMeta POLICY_LRU requires a local or cached metadata backend that refreshes read heat");
        return EC_CONFIG_ERROR;
    }
    if (!HasCrashRecoverableKvMetaMetadata(*group)) {
        KVCM_INTERVAL_LOG_WARN(60,
                               "KVMeta group [%s] uses process-local/test metadata; a KVCM restart loses exact-object "
                               "ownership and requires an independent backend TTL/sweeper. Production shared caches "
                               "must use cached metadata with a Redis/async_redis persistent layer",
                               instance_group.c_str());
    }
    const auto data_storage_manager = registry_manager_->data_storage_manager();
    if (!data_storage_manager || group->storage_candidates().empty()) {
        AddError(request_context, "KVMeta requires at least one exact-object storage candidate");
        return EC_CONFIG_ERROR;
    }
    std::unordered_set<std::string_view> unique_storage_names;
    unique_storage_names.reserve(group->storage_candidates().size());
    for (const auto &storage_name : group->storage_candidates()) {
        const auto backend = data_storage_manager->GetDataStorageBackend(storage_name);
        const auto extension = std::dynamic_pointer_cast<KvMetaDataStorageBackendExtension>(backend);
        if (!IsCanonicalKvMetaBackendName(storage_name) || !unique_storage_names.emplace(storage_name).second ||
            !backend || !SupportsKvMetaAdmission(backend->GetType()) || !extension ||
            backend->GetStorageConfig().type() != backend->GetType() ||
            backend->GetStorageConfig().global_unique_name() != storage_name ||
            !HasSafeConfiguredKvMetaNamespace(backend->GetStorageConfig(), limits_.max_location_uri_bytes)) {
            AddError(request_context,
                     "KVMeta storage candidates must be unique registered backends with hard caller-buffer and "
                     "exact-object lifecycle contracts");
            return EC_CONFIG_ERROR;
        }
    }
    if (!reclaimer_ || !reclaimer_->HasExecutableTuning()) {
        AddError(request_context, "KVMeta requires non-zero reclaim sampling and batching sizes");
        return EC_CONFIG_ERROR;
    }
    return EC_OK;
}

ErrorCode KvMetaManager::CheckDynamicByteAdmission(RequestContext *request_context,
                                                   const std::string &instance_group,
                                                   DataStorageType storage_type,
                                                   std::uint64_t requested_bytes) const {
    if (!request_context || instance_group.empty() || requested_bytes == 0) {
        return EC_BADARGS;
    }
    const auto [group_ec, group] = registry_manager_->GetInstanceGroup(request_context, instance_group);
    if (group_ec != EC_OK || !group) {
        return group_ec == EC_OK ? EC_INSTANCE_NOT_EXIST : group_ec;
    }
    const auto [instances_ec, instances] = registry_manager_->ListInstanceInfo(request_context, instance_group);
    if (instances_ec != EC_OK) {
        return instances_ec;
    }

    std::uint64_t total_usage = 0;
    std::uint64_t type_usage = 0;
    const DataStorageType base_type = ToBaseType(storage_type);
    const bool check_type_quota = storage_type != DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
    const auto saturating_add = [](std::uint64_t &sum, std::uint64_t value) {
        sum = value > std::numeric_limits<std::uint64_t>::max() - sum ? std::numeric_limits<std::uint64_t>::max()
                                                                      : sum + value;
    };
    for (const auto &instance : instances) {
        if (!instance || !IsKvMetaInstance(*instance)) {
            AddError(request_context,
                     "KVMeta byte admission requires an instance group containing only generic-object instances");
            return instance ? EC_BADARGS : EC_CORRUPTION;
        }
        auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance->instance_id());
        if (!indexer) {
            AddError(request_context, "KVMeta byte admission could not read an instance indexer");
            return EC_INSTANCE_NOT_EXIST;
        }
        saturating_add(total_usage, indexer->GetStorageUsage());
        if (check_type_quota) {
            saturating_add(type_usage, indexer->GetStorageUsageByType(base_type));
        }
    }

    const auto fits = [requested_bytes](std::int64_t capacity, std::uint64_t used) {
        return capacity >= 0 && used <= static_cast<std::uint64_t>(capacity) &&
               requested_bytes <= static_cast<std::uint64_t>(capacity) - used;
    };
    if (!fits(group->quota().capacity(), total_usage)) {
        AddError(request_context, "KVMeta requested bytes exceed the remaining instance-group capacity");
        return EC_NOSPC;
    }
    // A pre-selection check uses UNKNOWN to validate only the group quota.
    // This preserves the selector's existing behavior while ensuring that an
    // already-full KVMeta group is reported as EC_NOSPC instead of EC_ERROR.
    if (!check_type_quota) {
        return EC_OK;
    }
    for (const auto &type_quota : group->quota().quota_config()) {
        if (ToBaseType(type_quota.storage_spec()) == base_type && !fits(type_quota.capacity(), type_usage)) {
            AddError(request_context, "KVMeta requested bytes exceed the remaining storage-type capacity");
            return EC_NOSPC;
        }
    }
    return EC_OK;
}

std::pair<ErrorCode, std::string> KvMetaManager::RegisterInstance(RequestContext *request_context,
                                                                  const std::string &instance_group,
                                                                  const std::string &instance_id,
                                                                  const std::string &user_data) {
    if (!initialized_.load(std::memory_order_acquire) || !request_context || instance_group.empty() ||
        instance_group.size() > limits_.max_instance_group_bytes || instance_id.empty() ||
        instance_id.size() > limits_.max_instance_id_bytes || user_data.size() > limits_.max_user_data_bytes) {
        AddError(request_context, "KVMeta instance_group, instance_id, or user_data exceeds a configured limit");
        return {EC_BADARGS, {}};
    }
    if (const ErrorCode ec = ValidateCacheConfiguration(request_context, instance_group); ec != EC_OK) {
        return {ec, {}};
    }

    ModelDeployment deployment;
    deployment.set_model_name(std::string(kKvMetaModelName));
    deployment.set_dtype(std::string(kKvMetaDtype));
    deployment.set_tp_size(1);
    deployment.set_dp_size(1);
    deployment.set_pp_size(1);
    deployment.set_extra(std::string(kKvMetaDeploymentExtra));
    deployment.set_user_data(user_data);

    std::lock_guard<std::mutex> lock(registration_mutex_);
    const auto [list_ec, existing_instances] = registry_manager_->ListInstanceInfo(request_context, instance_group);
    if (list_ec != EC_OK) {
        return {list_ec, {}};
    }
    if (std::any_of(existing_instances.begin(), existing_instances.end(), [](const auto &instance) {
            return !instance || !IsKvMetaInstance(*instance);
        })) {
        AddError(request_context,
                 "KVMeta requires a dedicated instance group so generic-object usage cannot affect KV-cache quota");
        return {EC_BADARGS, {}};
    }
    auto result = cache_manager_->RegisterInstance(request_context,
                                                   instance_group,
                                                   InternalInstanceId(instance_id),
                                                   1,
                                                   {LocationSpecInfo(std::string(kKvMetaValueSpecName), 1)},
                                                   deployment,
                                                   {},
                                                   CacheManager::QueryType::QT_BATCH_GET);
    if (result.first == EC_OK) {
        RememberKvMetaGroup(instance_group);
    }
    return result;
}

std::pair<ErrorCode, std::shared_ptr<const InstanceInfo>>
KvMetaManager::GetInstanceInfo(RequestContext *request_context, const std::string &instance_id) const {
    if (!initialized_.load(std::memory_order_acquire)) {
        return {EC_ERROR, nullptr};
    }
    auto [ec, info] = GetValidatedInstanceInfo(request_context, instance_id);
    if (ec != EC_OK) {
        return {ec, nullptr};
    }
    auto public_info = std::make_shared<InstanceInfo>(*info);
    public_info->set_instance_id(instance_id);
    return {EC_OK, std::move(public_info)};
}

ErrorCode KvMetaManager::LoadExactLocations(RequestContext *request_context,
                                            const std::string &internal_instance_id,
                                            const std::vector<std::string> &keys,
                                            std::vector<ExactLocation> &out) const {
    out.assign(keys.size(), ExactLocation{});
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (!indexer) {
        AddError(request_context, "KVMeta instance does not exist or its indexer is unavailable");
        return EC_INSTANCE_NOT_EXIST;
    }

    std::vector<std::int64_t> internal_keys;
    internal_keys.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        out[i].internal_key = InternalKey(keys[i]);
        out[i].location_id = StableLocationId(keys[i]);
        internal_keys.push_back(out[i].internal_key);
    }

    ErrorCode overall = EC_OK;
    for (const auto &layer : MakeUniqueKeyLayers(internal_keys)) {
        KeyVector layer_keys;
        LocationIdsPerKey layer_ids;
        layer_keys.reserve(layer.size());
        layer_ids.reserve(layer.size());
        for (const std::size_t index : layer) {
            layer_keys.push_back(out[index].internal_key);
            layer_ids.push_back({out[index].location_id});
        }
        LocationsPerKey locations;
        const auto result = indexer->GetLocations(request_context, layer_keys, layer_ids, locations);
        if (locations.size() != layer.size() || result.per_location_error_codes.size() != layer.size()) {
            AddError(request_context, "KVMeta exact metadata read returned a malformed batch");
            return EC_MISMATCH;
        }
        for (std::size_t i = 0; i < layer.size(); ++i) {
            const std::size_t output_index = layer[i];
            if (locations[i].size() != 1 || result.per_location_error_codes[i].size() != 1) {
                out[output_index].ec = EC_MISMATCH;
            } else {
                out[output_index].ec = result.per_location_error_codes[i][0];
                out[output_index].location = locations[i][0];
                if (out[output_index].ec == EC_OK &&
                    (!out[output_index].location ||
                     out[output_index].location->id() != out[output_index].location_id)) {
                    out[output_index].ec = EC_CORRUPTION;
                }
            }
            overall = FirstHardError(overall, out[output_index].ec);
        }
    }
    return overall;
}

std::pair<ErrorCode, std::vector<KvMetaManager::GetResult>> KvMetaManager::Get(
    RequestContext *request_context, const std::string &instance_id, const std::vector<std::string> &keys) const {
    if (!initialized_.load(std::memory_order_acquire)) {
        return {EC_ERROR, {}};
    }
    const auto validated_instance = GetValidatedInstanceInfo(request_context, instance_id);
    if (validated_instance.first != EC_OK) {
        return {validated_instance.first, {}};
    }
    if (const ErrorCode ec = ValidateKeys(request_context, keys); ec != EC_OK) {
        return {ec, {}};
    }
    const std::string internal_instance_id = InternalInstanceId(instance_id);
    std::vector<ExactLocation> exact;
    const ErrorCode load_ec = LoadExactLocations(request_context, internal_instance_id, keys, exact);
    if (load_ec != EC_OK) {
        return {load_ec, {}};
    }
    std::vector<GetResult> result(keys.size());
    for (std::size_t i = 0; i < exact.size(); ++i) {
        if (exact[i].ec == EC_NOENT) {
            continue;
        }
        if (exact[i].ec != EC_OK || !exact[i].location) {
            return {exact[i].ec == EC_OK ? EC_CORRUPTION : exact[i].ec, {}};
        }
        std::uint64_t value_size = 0;
        if (const ErrorCode ec = ValidateOwnedLocation(request_context,
                                                       internal_instance_id,
                                                       exact[i].internal_key,
                                                       exact[i].location_id,
                                                       *exact[i].location,
                                                       value_size);
            ec != EC_OK) {
            return {ec, {}};
        }
        if (!IsCommittedObject(*exact[i].location)) {
            continue;
        }
        if (!ToValueLocation(*exact[i].location, result[i].location) || result[i].location.value_size != value_size) {
            AddError(request_context, "KVMeta committed location is malformed");
            return {EC_CORRUPTION, {}};
        }
        result[i].found = true;
    }
    return {EC_OK, std::move(result)};
}

ErrorCode KvMetaManager::DeleteStorageUris(RequestContext *request_context,
                                           const std::string &storage_name,
                                           const std::vector<DataStorageUri> &uris) const {
    if (uris.empty()) {
        return EC_OK;
    }
    auto data_storage_manager = registry_manager_->data_storage_manager();
    if (!data_storage_manager) {
        return EC_ERROR;
    }
    std::vector<ErrorCode> delete_results;
    const char *failure_kind = nullptr;
    try {
        delete_results = data_storage_manager->DeleteForKvMeta(request_context, storage_name, uris, nullptr);
    } catch (const std::exception &) { failure_kind = "standard_exception"; } catch (...) {
        failure_kind = "unknown_exception";
    }
    if (failure_kind) {
        KVCM_LOG_WARN("KVMeta storage delete caught a provider exception, item_count[%zu], failure[%s]",
                      uris.size(),
                      failure_kind);
        AddError(request_context, "KVMeta storage delete caught a provider exception");
        return EC_IO_ERROR;
    }
    if (delete_results.size() != uris.size()) {
        AddError(request_context, "KVMeta storage delete returned a mismatched result count");
        return EC_MISMATCH;
    }
    ErrorCode overall = EC_OK;
    for (const ErrorCode ec : delete_results) {
        if (ec != EC_OK && ec != EC_NOENT) {
            overall = FirstHardError(overall, ec);
        }
    }
    return overall;
}

bool KvMetaManager::TryGetKvMetaDeleteRetrySafety(const CacheLocation &location, bool &retry_safe) const {
    retry_safe = false;
    const auto data_storage_manager = registry_manager_ ? registry_manager_->data_storage_manager() : nullptr;
    if (!data_storage_manager || location.location_specs().size() != 1) {
        return false;
    }
    const DataStorageUri uri(location.location_specs().front().uri());
    if (!HasMatchingStorageBackend(location, data_storage_manager)) {
        return false;
    }
    const auto backend = data_storage_manager->GetDataStorageBackend(uri.GetHostName());
    const auto extension = std::dynamic_pointer_cast<KvMetaDataStorageBackendExtension>(backend);
    if (!extension) {
        return false;
    }
    retry_safe = extension->IsKvMetaDeleteRetrySafe();
    return true;
}

ErrorCode KvMetaManager::DeleteAllocatedLocations(RequestContext *request_context,
                                                  const std::vector<SessionItem> &items) const {
    auto data_storage_manager = registry_manager_->data_storage_manager();
    if (!data_storage_manager) {
        return EC_ERROR;
    }
    std::map<std::string, std::vector<DataStorageUri>> uris_by_storage;
    std::unordered_set<std::string> seen_allocations;
    seen_allocations.reserve(items.size());
    ErrorCode overall = EC_OK;
    for (const auto &item : items) {
        if (!item.data_location) {
            // Every production SessionItem reaching this helper owns one
            // allocation. Silently skipping a missing owner would turn an
            // invariant violation into an unobservable physical orphan.
            overall = FirstHardError(overall, EC_CORRUPTION);
            continue;
        }
        if (!HasMatchingStorageBackend(*item.data_location, data_storage_manager)) {
            overall = FirstHardError(overall, EC_CORRUPTION);
            continue;
        }
        std::string allocation_identity;
        if (!GetPhysicalAllocationIdentity(*item.data_location, allocation_identity)) {
            overall = FirstHardError(overall, EC_CORRUPTION);
            continue;
        }
        if (!seen_allocations.insert(std::move(allocation_identity)).second) {
            continue;
        }
        for (const auto &spec : item.data_location->location_specs()) {
            DataStorageUri uri(spec.uri());
            if (!uri.Valid() || uri.GetHostName().empty()) {
                overall = FirstHardError(overall, EC_CORRUPTION);
                continue;
            }
            uris_by_storage[uri.GetHostName()].push_back(std::move(uri));
        }
    }
    for (auto &[storage_name, uris] : uris_by_storage) {
        overall = FirstHardError(overall, DeleteStorageUris(request_context, storage_name, uris));
    }
    return overall;
}

KvMetaManager::DeleteItemsResult KvMetaManager::DeleteItems(RequestContext *request_context,
                                                            const std::string &internal_instance_id,
                                                            const std::vector<SessionItem> &items,
                                                            const DeleteItemsOptions &options) {
    DeleteItemsResult result;
    result.metadata_deleted.assign(items.size(), false);
    result.metadata_absent.assign(items.size(), false);
    result.metadata_conflicted.assign(items.size(), false);
    if (items.empty()) {
        result.metadata_cleanup_complete = true;
        return result;
    }
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (!indexer) {
        AddError(request_context, "KVMeta indexer is unavailable during exact delete");
        result.ec = EC_INSTANCE_NOT_EXIST;
        return result;
    }

    if (options.delete_physical) {
        for (const auto &item : items) {
            bool item_retry_safe = false;
            if (!item.data_location || !TryGetKvMetaDeleteRetrySafety(*item.data_location, item_retry_safe)) {
                AddError(request_context, "KVMeta physical owner has no valid delete policy");
                result.ec = EC_CORRUPTION;
                return result;
            }
        }
        // First replace each exact owner with a read-invisible tombstone and
        // persist that fence, then issue physical deletion, then erase the
        // tombstone. A generation-aware/path-unique backend may retry from the
        // durable fence. A reusable legacy GA is different: an unknown result
        // is terminal for KVCM, because replay could delete a successor. In
        // that case we finalize logical metadata and report a possible orphan.
        const std::int64_t now_us = TimestampUtil::GetCurrentTimeUs();
        std::int64_t retirement_deadline = 0;
        if (!EncodeTaggedDeadlineUs(now_us, 0, retirement_deadline)) {
            result.ec = EC_ERROR;
            return result;
        }

        std::vector<SessionItem> retired_items;
        std::vector<std::size_t> retired_indices;
        std::vector<std::int64_t> item_keys;
        retired_items.reserve(items.size());
        retired_indices.reserve(items.size());
        item_keys.reserve(items.size());
        for (const auto &item : items) {
            item_keys.push_back(item.internal_key);
        }

        KeyVector keys_to_sync;
        for (const auto &layer : MakeUniqueKeyLayers(item_keys)) {
            KeyVector keys;
            LocationIdsPerKey ids;
            std::vector<CacheLocationConstPtr> expected;
            std::vector<CacheLocationConstPtr> replacements;
            keys.reserve(layer.size());
            ids.reserve(layer.size());
            expected.reserve(layer.size());
            replacements.reserve(layer.size());
            bool valid_layer = true;
            for (const std::size_t index : layer) {
                if (!items[index].metadata_location) {
                    result.ec = FirstHardError(result.ec, EC_BADARGS);
                    valid_layer = false;
                    break;
                }
                keys.push_back(items[index].internal_key);
                ids.push_back({items[index].location_id});
                expected.push_back(items[index].metadata_location);
                auto replacement = std::make_shared<CacheLocation>(*items[index].metadata_location);
                replacement->set_status(CLS_DELETING);
                replacement->set_create_time(retirement_deadline);
                replacements.push_back(std::move(replacement));
            }
            if (!valid_layer) {
                continue;
            }

            std::vector<bool> layer_retired(layer.size(), false);
            auto modifier = [&expected, &replacements, &layer_retired](const std::vector<ErrorCode> &get_ecs,
                                                                       const LocationIdVector &,
                                                                       std::size_t key_index,
                                                                       CacheLocationVector &locations,
                                                                       PropertyMap &) -> LocationModifierResult {
                if (get_ecs.size() != 1 || locations.size() != 1 || key_index >= expected.size()) {
                    return {MA_FAIL, {EC_MISMATCH}};
                }
                if (get_ecs[0] != EC_OK) {
                    return {MA_SKIP, {get_ecs[0]}};
                }
                if (!locations[0] || locations[0]->ToJsonString() != expected[key_index]->ToJsonString()) {
                    return {MA_SKIP, {EC_MISMATCH}};
                }
                locations[0] = replacements[key_index];
                layer_retired[key_index] = true;
                return {MA_OK, {EC_OK}};
            };
            const auto rmw =
                indexer->ReadModifyWriteLocationsForMaintenance(request_context, keys, ids, modifier, false);
            if (rmw.per_location_error_codes.size() != layer.size()) {
                result.ec = FirstHardError(result.ec, EC_MISMATCH);
            }
            const std::size_t reported = std::min(rmw.per_location_error_codes.size(), layer.size());
            for (std::size_t i = 0; i < layer.size(); ++i) {
                const std::size_t item_index = layer[i];
                if (layer_retired[i]) {
                    // The modifier running only proves that the expected owner
                    // was observed. The backend write result below is the
                    // authorization boundary for physical deletion: a failed
                    // or malformed RMW may not have installed the tombstone.
                    // Still Sync the key because the write outcome is
                    // ambiguous and, if it did apply, recovery needs that
                    // tombstone to become durable.
                    keys_to_sync.push_back(items[item_index].internal_key);
                    result.metadata_outcome_changed = true;
                }
                if (i >= reported || rmw.per_location_error_codes[i].size() != 1) {
                    result.ec = FirstHardError(result.ec, EC_MISMATCH);
                    continue;
                }
                const ErrorCode ec = rmw.per_location_error_codes[i][0];
                if (ec == EC_OK && layer_retired[i]) {
                    SessionItem retired = items[item_index];
                    retired.metadata_location = replacements[i];
                    retired_items.push_back(std::move(retired));
                    retired_indices.push_back(item_index);
                    continue;
                }
                if (ec == EC_NOENT && !layer_retired[i]) {
                    result.metadata_absent[item_index] = true;
                    result.metadata_already_absent = true;
                    continue;
                }
                if (ec == EC_MISMATCH) {
                    result.metadata_conflicted[item_index] = true;
                    result.metadata_owner_conflicted = true;
                }
                result.ec = FirstHardError(result.ec, ec == EC_OK ? EC_MISMATCH : ec);
            }
            if (rmw.ec != EC_OK && rmw.ec != EC_PARTIAL_OK) {
                result.ec = FirstHardError(result.ec, rmw.ec);
            }
        }

        std::sort(keys_to_sync.begin(), keys_to_sync.end());
        keys_to_sync.erase(std::unique(keys_to_sync.begin(), keys_to_sync.end()), keys_to_sync.end());
        if (!keys_to_sync.empty() && !indexer->Sync(keys_to_sync)) {
            result.ec = FirstHardError(result.ec, EC_TIMEOUT);
            AddError(request_context,
                     "KVMeta could not persist the physical-delete tombstone; maintenance is fail-closed until "
                     "recovery");
            CancelMaintenance();
            return result;
        }

        bool at_most_once_delete_uncertain = false;
        if (!retired_items.empty()) {
            struct DeletePartition {
                bool retry_safe = false;
                std::vector<SessionItem> items;
                std::vector<std::size_t> original_indices;
            };
            DeletePartition at_most_once;
            DeletePartition retry_safe;
            retry_safe.retry_safe = true;
            at_most_once.items.reserve(retired_items.size());
            at_most_once.original_indices.reserve(retired_items.size());
            retry_safe.items.reserve(retired_items.size());
            retry_safe.original_indices.reserve(retired_items.size());
            for (std::size_t i = 0; i < retired_items.size(); ++i) {
                const auto &item = retired_items[i];
                bool item_retry_safe = false;
                if (!item.data_location || !TryGetKvMetaDeleteRetrySafety(*item.data_location, item_retry_safe)) {
                    AddError(request_context, "KVMeta retired physical owner lost its delete policy");
                    CancelMaintenance();
                    result.ec = EC_OUTCOME_UNKNOWN;
                    return result;
                }
                auto &partition = item_retry_safe ? retry_safe : at_most_once;
                partition.items.push_back(item);
                partition.original_indices.push_back(retired_indices[i]);
            }

            auto delete_and_finalize = [&](const DeletePartition &partition) -> bool {
                if (partition.items.empty()) {
                    return true;
                }
                ErrorCode physical_ec = EC_IO_ERROR;
                try {
                    physical_ec = DeleteAllocatedLocations(request_context, partition.items);
                } catch (const std::exception &) {
                    KVCM_LOG_WARN("KVMeta tombstoned physical cleanup caught a standard internal exception");
                } catch (...) {
                    KVCM_LOG_WARN("KVMeta tombstoned physical cleanup caught an unknown internal exception");
                }
                if (physical_ec == EC_OK && reclaimer_ && !options.capacity_release_group.empty()) {
                    reclaimer_->ConfirmExternalCapacityFreed(options.capacity_release_group, partition.items);
                }
                if (physical_ec != EC_OK) {
                    if (partition.retry_safe) {
                        AddError(request_context,
                                 "KVMeta retry-safe physical cleanup failed; durable tombstones were retained for "
                                 "recovery");
                        CancelMaintenance();
                        result.ec = EC_OUTCOME_UNKNOWN;
                        return false;
                    }
                    at_most_once_delete_uncertain = true;
                    AddError(request_context,
                             "KVMeta legacy-GA delete had an unknown outcome; logical metadata will be finalized "
                             "and the address will not be retried");
                    KVCM_LOG_WARN(
                        "KVMeta will not retry an ambiguous reusable-address delete, item_count[%zu], ec[%d]",
                        partition.items.size(),
                        physical_ec);
                    // Bound damage to one uncertain operation. Ordinary
                    // KV-cache traffic is unaffected; a subsequent KVMeta
                    // recovery skips this reusable address and reconciles
                    // metadata only.
                    CancelMaintenance();
                }

                DeleteItemsOptions finalize_options = options;
                finalize_options.delete_physical = false;
                const auto finalized =
                    DeleteItems(request_context, internal_instance_id, partition.items, finalize_options);
                result.ec = FirstHardError(result.ec, finalized.ec);
                result.metadata_already_absent = result.metadata_already_absent || finalized.metadata_already_absent;
                result.metadata_owner_conflicted =
                    result.metadata_owner_conflicted || finalized.metadata_owner_conflicted;
                result.metadata_outcome_changed =
                    result.metadata_outcome_changed || finalized.metadata_outcome_changed;
                for (std::size_t i = 0; i < partition.original_indices.size(); ++i) {
                    const std::size_t original_index = partition.original_indices[i];
                    if (i < finalized.metadata_deleted.size()) {
                        result.metadata_deleted[original_index] = finalized.metadata_deleted[i];
                    }
                    if (i < finalized.metadata_absent.size()) {
                        result.metadata_absent[original_index] =
                            result.metadata_absent[original_index] || finalized.metadata_absent[i];
                    }
                    if (i < finalized.metadata_conflicted.size()) {
                        result.metadata_conflicted[original_index] =
                            result.metadata_conflicted[original_index] || finalized.metadata_conflicted[i];
                    }
                }
                if (!finalized.metadata_cleanup_complete) {
                    result.metadata_cleanup_complete = false;
                }
                return true;
            };

            // Reusable addresses must receive their sole delete attempt in
            // this process. Recovery cannot distinguish an unattempted GA
            // from an ambiguous attempt and therefore never replays one.
            // Retry-safe objects are processed independently afterwards, so
            // their failures retain a recoverable tombstone instead of
            // inheriting the at-most-once policy.
            delete_and_finalize(at_most_once);
            if (!delete_and_finalize(retry_safe)) {
                return result;
            }
        }

        const bool metadata_state_ok = result.ec == EC_OK;
        result.metadata_cleanup_complete = metadata_state_ok && retired_items.size() == items.size() &&
                                           !result.metadata_already_absent && !result.metadata_owner_conflicted &&
                                           std::all_of(result.metadata_deleted.begin(),
                                                       result.metadata_deleted.end(),
                                                       [](bool value) { return value; });
        if (at_most_once_delete_uncertain && result.ec == EC_OK) {
            result.ec = EC_OUTCOME_UNKNOWN;
        }
        return result;
    }

    MetaSearcher searcher(indexer);
    std::vector<std::int64_t> item_keys;
    item_keys.reserve(items.size());
    for (const auto &item : items) {
        item_keys.push_back(item.internal_key);
    }

    for (const auto &layer : MakeUniqueKeyLayers(item_keys)) {
        KeyVector keys;
        LocationIdsPerKey ids;
        std::vector<std::vector<std::string>> expected_values;
        keys.reserve(layer.size());
        ids.reserve(layer.size());
        expected_values.reserve(layer.size());
        for (const std::size_t index : layer) {
            if (!items[index].metadata_location) {
                result.ec = FirstHardError(result.ec, EC_BADARGS);
                continue;
            }
            keys.push_back(items[index].internal_key);
            ids.push_back({items[index].location_id});
            expected_values.push_back({items[index].metadata_location->ToJsonString()});
        }
        if (keys.size() != layer.size()) {
            continue;
        }
        std::vector<std::vector<ErrorCode>> per_location_ec;
        const ErrorCode delete_ec = searcher.BatchDeleteLocations(request_context,
                                                                  keys,
                                                                  ids,
                                                                  per_location_ec,
                                                                  expected_values,
                                                                  options.adjust_storage_usage,
                                                                  true,
                                                                  options.maintenance_no_touch);
        result.ec = FirstHardError(result.ec, delete_ec);
        if (per_location_ec.size() != layer.size()) {
            result.ec = FirstHardError(result.ec, EC_MISMATCH);
            continue;
        }
        for (std::size_t i = 0; i < layer.size(); ++i) {
            if (per_location_ec[i].size() != 1) {
                result.ec = FirstHardError(result.ec, EC_MISMATCH);
                continue;
            }
            const ErrorCode ec = per_location_ec[i][0];
            if (ec == EC_OK) {
                result.metadata_deleted[layer[i]] = true;
                result.metadata_outcome_changed = true;
            } else if (ec == EC_NOENT) {
                result.metadata_absent[layer[i]] = true;
                result.metadata_already_absent = true;
            } else {
                // EC_MISMATCH is positive evidence that the stable location id
                // now names a different exact value. Reclaimer must fail closed
                // instead of retrying forever or ever deleting the old URI.
                result.metadata_conflicted[layer[i]] = ec == EC_MISMATCH;
                result.metadata_owner_conflicted = result.metadata_owner_conflicted || ec == EC_MISMATCH;
                result.ec = FirstHardError(result.ec, ec);
            }
        }
    }

    KeyVector keys_to_sync;
    std::unordered_set<std::int64_t> unique_keys_to_sync;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if ((result.metadata_deleted[i] || (options.sync_metadata_absent && result.metadata_absent[i])) &&
            unique_keys_to_sync.insert(items[i].internal_key).second) {
            keys_to_sync.push_back(items[i].internal_key);
        }
    }
    const bool metadata_delete_is_durable = keys_to_sync.empty() || indexer->Sync(keys_to_sync);
    if (!metadata_delete_is_durable) {
        result.ec = FirstHardError(result.ec, EC_TIMEOUT);
        // BatchDeleteLocations adjusts the in-memory counter when the delete
        // is accepted. If its persistence barrier fails, restore a
        // conservative upper bound; the next KVMeta recovery rebuilds the
        // exact value from durable metadata.
        if (options.adjust_storage_usage && options.restore_usage_on_sync_failure) {
            for (std::size_t i = 0; i < items.size(); ++i) {
                if (result.metadata_deleted[i] && items[i].metadata_location) {
                    indexer->AddStorageUsageByType(items[i].metadata_location->type(), items[i].value_size);
                }
            }
        }
    }
    // An already-absent item is complete only for the one caller that
    // explicitly persisted that absence while holding a pending ownership
    // fence (the Reclaimer retry path).
    result.metadata_cleanup_complete = result.ec == EC_OK && metadata_delete_is_durable &&
                                       (!result.metadata_already_absent || options.sync_metadata_absent);
    return result;
}

KvMetaManager::DeleteItemsResult KvMetaManager::DeleteRetiredMetadata(RequestContext *request_context,
                                                                      const std::string &internal_instance_id,
                                                                      const std::vector<SessionItem> &items) {
    // Reclaimer finalization deliberately separates metadata and physical
    // deletion. If Sync fails after the in-memory exact delete, a later retry
    // must Sync the now-absent key before releasing the allocation. Keeping
    // the already-decremented usage avoids double accounting on that retry.
    DeleteItemsOptions options;
    options.delete_physical = false;
    options.maintenance_no_touch = true;
    options.sync_metadata_absent = true;
    options.restore_usage_on_sync_failure = false;
    return DeleteItems(request_context, internal_instance_id, items, options);
}

std::pair<ErrorCode, KvMetaManager::StartWriteResult>
KvMetaManager::StartWrite(RequestContext *request_context,
                          const std::string &instance_id,
                          const std::vector<std::string> &keys,
                          const std::vector<std::uint64_t> &value_sizes,
                          std::int64_t write_timeout_seconds) {
    StartWriteResult response;
    if (!initialized_.load(std::memory_order_acquire)) {
        return {EC_ERROR, std::move(response)};
    }
    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        return {EC_SERVICE_NOT_LEADER, std::move(response)};
    }
    if (const ErrorCode ec = ValidateKeys(request_context, keys); ec != EC_OK) {
        return {ec, std::move(response)};
    }
    if (instance_id.empty() || instance_id.size() > limits_.max_instance_id_bytes ||
        value_sizes.size() != keys.size() || write_timeout_seconds <= 0 ||
        write_timeout_seconds > limits_.max_write_timeout_seconds) {
        AddError(request_context, "KVMeta value_sizes or write_timeout_seconds is invalid");
        return {EC_BADARGS, std::move(response)};
    }
    std::uint64_t batch_bytes = 0;
    for (const std::uint64_t size : value_sizes) {
        if (size == 0 || size > limits_.max_value_bytes || size > limits_.max_batch_bytes ||
            size > std::numeric_limits<std::size_t>::max() || batch_bytes > limits_.max_batch_bytes - size) {
            AddError(request_context, "KVMeta value size exceeds a configured limit");
            return {EC_OUT_OF_LIMIT, std::move(response)};
        }
        batch_bytes += size;
    }
    const auto write_deadline = KvMetaWriteSessionManager::Clock::now() + std::chrono::seconds(write_timeout_seconds);
    const std::int64_t write_start_time_us = TimestampUtil::GetCurrentTimeUs();
    if (write_start_time_us <= 0) {
        AddError(request_context, "KVMeta could not establish a persistent write lease clock");
        return {EC_ERROR, StartWriteResult{}};
    }

    const std::string internal_instance_id = InternalInstanceId(instance_id);
    auto [instance_ec, instance_info] = GetValidatedInstanceInfo(request_context, instance_id);
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (instance_ec != EC_OK || !indexer) {
        return {instance_ec != EC_OK ? instance_ec : EC_INSTANCE_NOT_EXIST, std::move(response)};
    }

    // The existing selector admits based on current usage because KV-cache
    // blocks have a fixed registered size. KVMeta additionally holds a
    // side-path-only shard lock and checks used + this request's exact bytes,
    // so differently sized values cannot overshoot group/type quota through
    // concurrent PutStart calls.
    const std::size_t quota_shard =
        std::hash<std::string>{}(instance_info->instance_group_name()) % quota_admission_mutexes_.size();
    std::unique_lock<std::mutex> quota_lock(quota_admission_mutexes_[quota_shard]);
    if (trimming_instances_[quota_shard].count(internal_instance_id) != 0) {
        AddError(request_context, "KVMeta instance is being trimmed");
        return {EC_EXIST, StartWriteResult{}};
    }

    // A retired location normally remains visible as CLS_DELETING until its
    // final persistence barrier succeeds. If that barrier failed after the
    // in-memory delete, the exact metadata can temporarily be absent. Keep the
    // same logical key closed until the pending batch either completes or is
    // handed to leader recovery; otherwise a reusable backend URI could expose
    // an ABA window between old-object deletion and a new allocation.
    if (reclaimer_) {
        if (reclaimer_->IsAdmissionBlocked(instance_info->instance_group_name())) {
            AddError(request_context, "KVMeta admission is paused while reclaim metadata is being persisted");
            return {EC_EXIST, StartWriteResult{}};
        }
        for (const auto &key : keys) {
            if (reclaimer_->HasPendingLocation(internal_instance_id, StableLocationId(key))) {
                AddError(request_context, "KVMeta value is being reclaimed");
                return {EC_EXIST, StartWriteResult{}};
            }
        }
    }

    auto data_storage_manager = registry_manager_->data_storage_manager();
    if (!data_storage_manager) {
        return {EC_ERROR, StartWriteResult{}};
    }

    std::vector<ExactLocation> existing;
    if (const ErrorCode ec = LoadExactLocations(request_context, internal_instance_id, keys, existing); ec != EC_OK) {
        return {ec, std::move(response)};
    }
    response.key_mask.assign(keys.size(), false);
    std::vector<std::size_t> missing_indices;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (existing[i].ec == EC_NOENT) {
            missing_indices.push_back(i);
        } else if (existing[i].ec == EC_OK && existing[i].location) {
            std::uint64_t existing_size = 0;
            if (const ErrorCode ec = ValidateOwnedLocation(request_context,
                                                           internal_instance_id,
                                                           existing[i].internal_key,
                                                           existing[i].location_id,
                                                           *existing[i].location,
                                                           existing_size);
                ec != EC_OK) {
                return {ec, StartWriteResult{}};
            }
            if (IsRetiredObject(*existing[i].location)) {
                AddError(request_context, "KVMeta value is being reclaimed");
                return {EC_EXIST, StartWriteResult{}};
            }
            // Only a committed object is an idempotent cache hit. Treating an
            // active reservation as a hit can make a second object client
            // report SaveObjects success while the first writer later aborts,
            // leaving no readable value behind. Its size is provisional too:
            // after the writer aborts or expires, a different-size request is
            // valid, so report the retryable state before any size comparison.
            if (!IsCommittedObject(*existing[i].location)) {
                AddError(request_context, "KVMeta value is still being written");
                return {EC_EXIST, StartWriteResult{}};
            }
            if (existing_size != value_sizes[i]) {
                AddError(request_context, "KVMeta existing value size does not match PutStart value_sizes");
                return {EC_MISMATCH, StartWriteResult{}};
            }
            response.key_mask[i] = true;
        } else {
            return {existing[i].ec == EC_OK ? EC_CORRUPTION : existing[i].ec, StartWriteResult{}};
        }
    }
    if (missing_indices.empty()) {
        return {EC_OK, std::move(response)};
    }
    // A cache that can accept new objects but has no executable eviction
    // policy eventually degrades into an unbounded/manual object store. Keep
    // existing hits readable, but fail new allocation before touching the
    // backend if the group was registered or hot-updated without the exact
    // LRU, tuning, and storage-ownership contract implemented by KVMeta.
    if (const ErrorCode ec = ValidateCacheConfiguration(request_context, instance_info->instance_group_name());
        ec != EC_OK) {
        return {ec, StartWriteResult{}};
    }
    if (!write_session_manager_) {
        return {EC_SERVICE_NOT_LEADER, StartWriteResult{}};
    }
    switch (write_session_manager_->Availability()) {
    case KvMetaWriteSessionManager::PutResult::kOk:
        break;
    case KvMetaWriteSessionManager::PutResult::kFull:
        AddError(request_context, "KVMeta active write-session limit has been reached");
        return {EC_NOSPC, StartWriteResult{}};
    case KvMetaWriteSessionManager::PutResult::kStopped:
        return {EC_SERVICE_NOT_LEADER, StartWriteResult{}};
    case KvMetaWriteSessionManager::PutResult::kDuplicate:
        return {EC_ERROR, StartWriteResult{}};
    case KvMetaWriteSessionManager::PutResult::kExpired:
        return {EC_TIMEOUT, StartWriteResult{}};
    }

    std::uint64_t missing_bytes = 0;
    for (const std::size_t index : missing_indices) {
        missing_bytes += value_sizes[index];
    }

    if (!data_storage_selector_) {
        return {EC_ERROR, StartWriteResult{}};
    }

    // MetaIndexer enforces key capacity per instance, whereas the periodic
    // watermark is aggregated for a group. A full hot instance can therefore
    // need targeted reclaim even while peer instances keep the group ratio
    // below its watermark. Avoid storage allocation when this request cannot
    // create all of its new primary metadata keys.
    std::uint64_t requested_new_metadata_keys = 0;
    bool metadata_key_admission_blocked = false;
    const std::uint64_t current_key_count = static_cast<std::uint64_t>(indexer->GetKeyCount());
    const std::uint64_t max_key_count = static_cast<std::uint64_t>(indexer->GetMaxKeyCount());
    if (current_key_count > max_key_count ||
        missing_indices.size() > static_cast<std::size_t>(max_key_count - current_key_count)) {
        KeyVector unique_missing_internal_keys;
        unique_missing_internal_keys.reserve(missing_indices.size());
        for (const std::size_t request_index : missing_indices) {
            unique_missing_internal_keys.push_back(existing[request_index].internal_key);
        }
        std::sort(unique_missing_internal_keys.begin(), unique_missing_internal_keys.end());
        unique_missing_internal_keys.erase(
            std::unique(unique_missing_internal_keys.begin(), unique_missing_internal_keys.end()),
            unique_missing_internal_keys.end());
        CacheLocationMapVector existing_location_maps;
        const auto key_result = indexer->GetLocationMapsForMaintenance(
            request_context, unique_missing_internal_keys, existing_location_maps);
        // Result::ec collapses an all-missing maintenance read to EC_ERROR in
        // the legacy indexer aggregation. The aligned per-key codes remain the
        // authoritative shape/identity signal for this read-only admission
        // check, so validate and consume those directly.
        if (key_result.error_codes.size() != unique_missing_internal_keys.size() ||
            existing_location_maps.size() != unique_missing_internal_keys.size()) {
            AddError(request_context,
                     "KVMeta metadata-key admission returned a malformed result: ec=" +
                         std::to_string(static_cast<int>(key_result.ec)) +
                         ", errors=" + std::to_string(key_result.error_codes.size()) +
                         ", maps=" + std::to_string(existing_location_maps.size()) +
                         ", keys=" + std::to_string(unique_missing_internal_keys.size()));
            return {EC_MISMATCH, StartWriteResult{}};
        }
        for (std::size_t i = 0; i < unique_missing_internal_keys.size(); ++i) {
            if (key_result.error_codes[i] == EC_NOENT) {
                ++requested_new_metadata_keys;
            } else if (key_result.error_codes[i] != EC_OK || existing_location_maps[i].empty()) {
                AddError(request_context, "KVMeta metadata-key admission could not verify an existing key");
                return {key_result.error_codes[i] == EC_OK ? EC_CORRUPTION : key_result.error_codes[i],
                        StartWriteResult{}};
            }
        }
        if (current_key_count > max_key_count || requested_new_metadata_keys > max_key_count - current_key_count) {
            AddError(request_context, "KVMeta requested keys exceed the remaining instance metadata capacity");
            if (requested_new_metadata_keys > max_key_count) {
                // The batch can never fit even in an empty instance. Do not
                // evict useful cache entries for a futile admission request.
                return {EC_NOSPC, StartWriteResult{}};
            }
            metadata_key_admission_blocked = true;
        }
    }

    const auto publish_admission_demand = [&](DataStorageType storage_type) {
        if (!reclaimer_) {
            return false;
        }
        const std::uint64_t requested_key_headroom =
            metadata_key_admission_blocked ? std::max<std::uint64_t>(1, requested_new_metadata_keys) : 0;
        reclaimer_->RequestAdmissionCapacity(instance_info->instance_group_name(),
                                             storage_type,
                                             missing_bytes,
                                             metadata_key_admission_blocked ? internal_instance_id : std::string{},
                                             requested_key_headroom);
        return true;
    };
    const auto request_reclaim_target = [&]() -> ErrorCode {
        if (!reclaimer_) {
            return EC_ERROR;
        }
        const auto target = data_storage_selector_->SelectCacheWriteDataStorageBackendForReclaim(
            request_context, instance_info->instance_group_name(), missing_bytes);
        if (target.ec != EC_OK || target.type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN || target.name.empty()) {
            return target.ec == EC_OK ? EC_NOENT : target.ec;
        }
        return publish_admission_demand(target.type) ? EC_OK : EC_ERROR;
    };
    if (const ErrorCode ec = CheckDynamicByteAdmission(request_context,
                                                       instance_info->instance_group_name(),
                                                       DataStorageType::DATA_STORAGE_TYPE_UNKNOWN,
                                                       missing_bytes);
        ec != EC_OK) {
        if (ec == EC_NOSPC && reclaimer_) {
            request_reclaim_target();
        }
        return {ec, StartWriteResult{}};
    }
    const auto selected = data_storage_selector_->SelectCacheWriteDataStorageBackend(
        request_context, instance_info->instance_group_name(), missing_bytes);
    if (selected.ec != EC_OK || selected.name.empty() || selected.type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN) {
        const ErrorCode reclaim_ec = request_reclaim_target();
        if (reclaim_ec == EC_OK) {
            AddError(request_context, "KVMeta exact-size allocation requires cache reclamation");
            return {EC_NOSPC, StartWriteResult{}};
        }
        if (reclaim_ec == EC_NOSPC) {
            AddError(request_context, "KVMeta exact-size allocation exceeds every configured storage-type capacity");
            return {EC_NOSPC, StartWriteResult{}};
        }
        return {selected.ec == EC_OK ? EC_NOENT : selected.ec, StartWriteResult{}};
    }
    if (const ErrorCode ec = CheckDynamicByteAdmission(
            request_context, instance_info->instance_group_name(), selected.type, missing_bytes);
        ec != EC_OK) {
        if (ec == EC_NOSPC && reclaimer_) {
            publish_admission_demand(selected.type);
        }
        return {ec, StartWriteResult{}};
    }
    if (metadata_key_admission_blocked) {
        publish_admission_demand(selected.type);
        return {EC_NOSPC, StartWriteResult{}};
    }
    const auto selected_backend = data_storage_manager->GetDataStorageBackend(selected.name);
    const auto kv_meta_backend = std::dynamic_pointer_cast<KvMetaDataStorageBackendExtension>(selected_backend);
    if (!selected_backend || selected_backend->GetType() != selected.type ||
        !SupportsKvMetaAdmission(selected_backend->GetType()) || !kv_meta_backend) {
        AddError(request_context, "KVMeta selected storage backend changed or lacks exact-object/caller-buffer safety");
        return {EC_CORRUPTION, StartWriteResult{}};
    }
    const StorageConfig &selected_config = selected_backend->GetStorageConfig();
    if (selected_config.type() != selected.type || selected_config.global_unique_name() != selected.name ||
        !HasSafeConfiguredKvMetaNamespace(selected_config, limits_.max_location_uri_bytes)) {
        AddError(request_context, "KVMeta selected storage backend identity does not match its registration");
        return {EC_CORRUPTION, StartWriteResult{}};
    }
    const std::int64_t failed_write_cleanup_grace_seconds = kv_meta_backend->GetFailedWriteCleanupGraceSeconds();
    if (failed_write_cleanup_grace_seconds < 0 ||
        failed_write_cleanup_grace_seconds > limits_.max_failed_write_cleanup_grace_seconds) {
        AddError(request_context, "KVMeta selected storage backend has an invalid failed-write cleanup grace");
        return {EC_CORRUPTION, StartWriteResult{}};
    }
    const std::int64_t persistent_lease_seconds = write_timeout_seconds + failed_write_cleanup_grace_seconds;
    std::int64_t persistent_write_deadline = 0;
    if (!EncodeLeaseDeadline(write_start_time_us, persistent_lease_seconds, persistent_write_deadline)) {
        AddError(request_context, "KVMeta write lease deadline is outside the persistent timestamp range");
        return {EC_OUT_OF_LIMIT, StartWriteResult{}};
    }
    const auto cleanup_deadline = write_deadline + std::chrono::seconds(failed_write_cleanup_grace_seconds);

    // A singleton Create call is intentional. Several existing filesystem
    // backends pack a batch into one file; singleton allocation prevents a
    // later per-key Remove from deleting another generic object.
    std::vector<SessionItem> candidates;
    candidates.reserve(missing_indices.size());
    const auto delete_allocated_noexcept = [&](const std::vector<SessionItem> &items) noexcept {
        try {
            return DeleteAllocatedLocations(request_context, items);
        } catch (const std::exception &) {
            KVCM_LOG_WARN("KVMeta allocation cleanup caught a standard internal exception");
        } catch (...) { KVCM_LOG_WARN("KVMeta allocation cleanup caught an unknown internal exception"); }
        return EC_IO_ERROR;
    };
    const auto release_allocated_once = [&](const std::vector<SessionItem> &items,
                                            const char *failure_message) noexcept {
        const ErrorCode cleanup_ec = delete_allocated_noexcept(items);
        if (cleanup_ec == EC_OK) {
            return true;
        }
        KVCM_LOG_WARN("%s, ec[%d]", failure_message, cleanup_ec);
        AddError(request_context,
                 "KVMeta uncommitted allocation delete had an unknown outcome; the address will not be retried");
        // A legacy reusable GA has no generation token. Replaying an
        // ambiguous delete could remove a successor object, so this helper
        // performs exactly one attempt and leaves any physical orphan to the
        // backend's own lifecycle cleanup. Stop only the KVMeta side path so
        // a provider outage cannot turn request retries into an orphan storm.
        CancelMaintenance();
        return false;
    };
    for (const std::size_t request_index : missing_indices) {
        const bool cancelled = maintenance_cancelled_.load(std::memory_order_acquire);
        const bool expired = KvMetaWriteSessionManager::Clock::now() >= write_deadline;
        if (cancelled || expired) {
            if (!release_allocated_once(
                    candidates, "KVMeta admission stop could not release all uncommitted allocations")) {
                return {EC_OUTCOME_UNKNOWN, StartWriteResult{}};
            }
            return {cancelled ? EC_SERVICE_NOT_LEADER : EC_TIMEOUT, StartWriteResult{}};
        }
        const std::uint64_t instance_path_hash =
            Hash64(internal_instance_id.data(), internal_instance_id.size(), kInstancePathHashSeed);
        const std::string object_key =
            "kvmeta/" + StringUtil::Uint64ToHex(instance_path_hash) + "/" +
            StringUtil::Uint64ToHex(static_cast<std::uint64_t>(existing[request_index].internal_key)) + "/" +
            StringUtil::GenerateRandomString(32);
        std::vector<std::pair<ErrorCode, DataStorageUri>> create_result;
        try {
            create_result = data_storage_manager->CreateForKvMeta(request_context,
                                                                  selected.name,
                                                                  {object_key},
                                                                  static_cast<std::size_t>(value_sizes[request_index]),
                                                                  nullptr);
        } catch (const std::exception &) {
            KVCM_LOG_WARN("KVMeta storage create caught a standard provider exception; "
                          "backend orphan cleanup may be required");
            release_allocated_once(candidates,
                                   "KVMeta could not release every allocation preceding a failed Create");
            AddError(request_context, "KVMeta storage create failed; backend orphan cleanup may be required");
            // The throwing call may already have allocated an object without
            // returning its identity. It cannot be retried or explicitly
            // freed safely; the backend's unmaterialized-object cleanup owns
            // that possible orphan.
            CancelMaintenance();
            return {EC_OUTCOME_UNKNOWN, StartWriteResult{}};
        } catch (...) {
            KVCM_LOG_WARN("KVMeta storage create caught an unknown provider exception; "
                          "backend orphan cleanup may be required");
            release_allocated_once(candidates,
                                   "KVMeta could not release every allocation preceding a failed Create");
            AddError(request_context, "KVMeta storage create failed; backend orphan cleanup may be required");
            CancelMaintenance();
            return {EC_OUTCOME_UNKNOWN, StartWriteResult{}};
        }
        if (create_result.size() == 1 && create_result[0].first != EC_OK) {
            if (!release_allocated_once(
                    candidates, "KVMeta could not release every allocation preceding a failed Create")) {
                return {EC_OUTCOME_UNKNOWN, StartWriteResult{}};
            }
            if (create_result[0].second.Valid()) {
                // A failed Create must not return an allocation identity. It
                // is impossible to tell whether this URI was allocated by the
                // call, is merely diagnostic, or aliases an existing object;
                // none of those cases grants physical Delete authority.
                AddError(request_context,
                         "KVMeta failed storage allocation returned an ambiguous URI; backend cleanup is required");
                CancelMaintenance();
                return {EC_OUTCOME_UNKNOWN, StartWriteResult{}};
            }
            if (create_result[0].first == EC_NOSPC && reclaimer_) {
                // The backend proved that this singleton did not allocate an
                // object, so it is safe to return a retryable result and ask
                // the side-path reclaimer for physical capacity. Use only the
                // failed singleton size: earlier candidates were released
                // above, and summing a whole batch here would over-evict on a
                // burst of equivalent retries.
                reclaimer_->RequestBackendCapacity(
                    instance_info->instance_group_name(), selected.type, value_sizes[request_index]);
            }
            AddError(request_context, "KVMeta singleton storage allocation failed");
            if (!IsKnownPreAllocationKvMetaCreateFailure(create_result[0].first)) {
                // Without an allocation identity, only errors that prove
                // rejection before allocation are safe to retry freely. Be
                // defensive even if a backend violates the side-interface
                // contract: never expose its generic/retryable error after an
                // allocation may already have happened.
                CancelMaintenance();
                return {EC_OUTCOME_UNKNOWN, StartWriteResult{}};
            }
            return {create_result[0].first, StartWriteResult{}};
        }
        if (create_result.size() != 1 ||
            !UriMatchesStorageBackend(create_result[0].second, selected.name, selected.type) ||
            !HasOwnedKvMetaAllocationShape(create_result[0].second, selected.type) ||
            !UriNamesCreatedKvMetaObject(create_result[0].second, selected_backend, selected.type, object_key)) {
            // Once a singleton Create returns the wrong cardinality, no result
            // can be attributed to this operation strongly enough to authorize
            // physical deletion. An EC_OK result with the wrong ownership shape
            // or a different logical object identity is equally unsafe: its
            // URI may alias shared, pre-existing, or otherwise unrelated data.
            // Leave this call's unknown allocations to backend orphan cleanup,
            // while releasing only candidates proven by earlier, independently
            // well-formed singleton calls.
            const bool cleanup_complete = release_allocated_once(
                candidates, "KVMeta could not release every previously validated allocation");
            AddError(request_context,
                     "KVMeta singleton storage allocation failed; malformed results require backend orphan cleanup");
            CancelMaintenance();
            return {cleanup_complete ? EC_CORRUPTION : EC_OUTCOME_UNKNOWN, StartWriteResult{}};
        }
        auto location = std::make_shared<CacheLocation>();
        location->set_id(existing[request_index].location_id);
        location->set_status(CLS_NEW);
        location->set_type(selected.type);
        location->set_spec_size(1);
        // Positive create_time is KVMeta-private for active objects. Persist a
        // tagged wall-clock lease deadline, rather than only the allocation
        // time, so a new leader cannot reclaim this URI while the client is
        // still inside its advertised write window. Older untagged positive
        // markers remain readable and receive the configured maximum lease
        // during rolling-upgrade recovery.
        location->set_create_time(persistent_write_deadline);
        location->push_location_spec(
            LocationSpec(std::string(kKvMetaValueSpecName), create_result[0].second.ToUriString()));
        location->set_validated_total_size(value_sizes[request_index]);
        std::uint64_t uri_size = 0;
        const ErrorCode location_ec = ValidateOwnedLocation(request_context,
                                                            internal_instance_id,
                                                            existing[request_index].internal_key,
                                                            existing[request_index].location_id,
                                                            *location,
                                                            uri_size);
        if (location_ec != EC_OK || uri_size != value_sizes[request_index]) {
            auto cleanup_items = candidates;
            cleanup_items.push_back(SessionItem{request_index,
                                                keys[request_index],
                                                existing[request_index].internal_key,
                                                existing[request_index].location_id,
                                                location,
                                                location,
                                                value_sizes[request_index]});
            const bool cleanup_complete = release_allocated_once(
                cleanup_items, "KVMeta could not release every invalid-size allocation");
            AddError(request_context,
                     location_ec == EC_OK ? "KVMeta storage returned a mismatched allocation size"
                                          : "KVMeta storage returned a malformed allocation URI");
            CancelMaintenance();
            return {cleanup_complete ? EC_CORRUPTION : EC_OUTCOME_UNKNOWN, StartWriteResult{}};
        }
        const bool allocation_reused =
            std::any_of(candidates.begin(), candidates.end(), [&](const SessionItem &candidate) {
                return candidate.data_location && SamePhysicalAllocation(*candidate.data_location, *location);
            });
        if (allocation_reused) {
            // Query fields such as size are metadata, not necessarily part of
            // the backend's Delete identity. Release the shared allocation
            // once through the earlier candidate and publish no metadata for
            // either key.
            const bool cleanup_complete =
                release_allocated_once(candidates, "KVMeta duplicate allocation cleanup failed");
            AddError(request_context, "KVMeta storage reused one singleton allocation for multiple keys");
            CancelMaintenance();
            return {cleanup_complete ? EC_CORRUPTION : EC_OUTCOME_UNKNOWN, StartWriteResult{}};
        }
        candidates.push_back(SessionItem{request_index,
                                         keys[request_index],
                                         existing[request_index].internal_key,
                                         existing[request_index].location_id,
                                         location,
                                         location,
                                         value_sizes[request_index]});
    }

    const bool cancelled_after_allocation = maintenance_cancelled_.load(std::memory_order_acquire);
    if (cancelled_after_allocation || KvMetaWriteSessionManager::Clock::now() >= write_deadline) {
        if (!release_allocated_once(candidates, "KVMeta stopped allocation cleanup failed")) {
            return {EC_OUTCOME_UNKNOWN, StartWriteResult{}};
        }
        return {cancelled_after_allocation ? EC_SERVICE_NOT_LEADER : EC_TIMEOUT, StartWriteResult{}};
    }

    std::vector<std::int64_t> candidate_keys;
    candidate_keys.reserve(candidates.size());
    for (const auto &candidate : candidates) {
        candidate_keys.push_back(candidate.internal_key);
    }
    std::vector<bool> inserted(candidates.size(), false);
    std::vector<bool> lost_race(candidates.size(), false);
    ErrorCode insert_error = EC_OK;
    for (const auto &layer : MakeUniqueKeyLayers(candidate_keys)) {
        KeyVector layer_keys;
        LocationIdsPerKey layer_ids;
        std::vector<CacheLocationConstPtr> layer_locations;
        layer_keys.reserve(layer.size());
        layer_ids.reserve(layer.size());
        layer_locations.reserve(layer.size());
        for (const std::size_t index : layer) {
            layer_keys.push_back(candidates[index].internal_key);
            layer_ids.push_back({candidates[index].location_id});
            layer_locations.push_back(candidates[index].metadata_location);
        }
        std::vector<bool> modifier_inserted(layer.size(), false);
        auto modifier = [&layer_locations, &modifier_inserted](const std::vector<ErrorCode> &get_ecs,
                                                               const LocationIdVector &,
                                                               std::size_t key_index,
                                                               CacheLocationVector &locations,
                                                               PropertyMap &) -> LocationModifierResult {
            if (get_ecs.size() != 1 || locations.size() != 1 || key_index >= layer_locations.size()) {
                return {MA_FAIL, {EC_MISMATCH}};
            }
            if (get_ecs[0] == EC_NOENT) {
                locations[0] = layer_locations[key_index];
                modifier_inserted[key_index] = true;
                return {MA_OK, {EC_OK}};
            }
            if (get_ecs[0] == EC_OK) {
                return {MA_SKIP, {EC_EXIST}};
            }
            return {MA_FAIL, {get_ecs[0]}};
        };
        const auto rmw = indexer->ReadModifyWriteTargetLocations(request_context, layer_keys, layer_ids, modifier);
        if (rmw.per_location_error_codes.size() != layer.size()) {
            insert_error = FirstHardError(insert_error, EC_MISMATCH);
            break;
        }
        for (std::size_t i = 0; i < layer.size(); ++i) {
            if (rmw.per_location_error_codes[i].size() != 1) {
                insert_error = FirstHardError(insert_error, EC_MISMATCH);
                continue;
            }
            const ErrorCode ec = rmw.per_location_error_codes[i][0];
            if (ec == EC_OK && modifier_inserted[i]) {
                inserted[layer[i]] = true;
            } else if (ec == EC_EXIST) {
                lost_race[layer[i]] = true;
                response.key_mask[candidates[layer[i]].request_index] = true;
            } else {
                insert_error = FirstHardError(insert_error, ec == EC_OK ? EC_MISMATCH : ec);
            }
        }
        if (rmw.ec != EC_OK && rmw.ec != EC_PARTIAL_OK) {
            insert_error = FirstHardError(insert_error, rmw.ec);
        }
        if (insert_error != EC_OK) {
            break;
        }
    }

    auto rollback_start = [&](ErrorCode original_error) {
        std::vector<std::string> candidate_original_keys;
        candidate_original_keys.reserve(candidates.size());
        for (const auto &candidate : candidates) {
            candidate_original_keys.push_back(candidate.original_key);
        }
        // The conditional insert result can be partial or malformed. Before
        // deciding that an allocation was never published, make the current
        // metadata view durable. A failed barrier means recovery, not this
        // request, must decide which generation owns the allocation.
        if (!indexer->Sync(candidate_keys)) {
            AddError(request_context,
                     "KVMeta start rollback could not establish a durable metadata view; maintenance is "
                     "fail-closed until recovery");
            CancelMaintenance();
            return EC_OUTCOME_UNKNOWN;
        }
        std::vector<ExactLocation> current;
        const ErrorCode reload_ec =
            LoadExactLocations(request_context, internal_instance_id, candidate_original_keys, current);
        if (reload_ec != EC_OK || current.size() != candidates.size()) {
            AddError(request_context,
                     "KVMeta start rollback could not reload exact ownership; maintenance is fail-closed until "
                     "recovery");
            CancelMaintenance();
            return EC_OUTCOME_UNKNOWN;
        }
        std::vector<SessionItem> exact_deletes;
        std::vector<SessionItem> direct_deletes;
        bool ownership_uncertain = false;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (current[i].ec == EC_OK && current[i].location) {
                std::uint64_t current_size = 0;
                if (ValidateOwnedLocation(request_context,
                                          internal_instance_id,
                                          candidates[i].internal_key,
                                          candidates[i].location_id,
                                          *current[i].location,
                                          current_size) != EC_OK) {
                    ownership_uncertain = true;
                    continue;
                }
                const bool exact_candidate =
                    current[i].location->ToJsonString() == candidates[i].metadata_location->ToJsonString();
                const bool same_allocation = candidates[i].data_location &&
                                             SamePhysicalAllocation(*current[i].location, *candidates[i].data_location);
                if (inserted[i] && exact_candidate) {
                    SessionItem item = candidates[i];
                    item.metadata_location = current[i].location;
                    exact_deletes.push_back(std::move(item));
                } else if (exact_candidate || same_allocation) {
                    // The URI is still referenced, but the exact metadata is
                    // either known to have won the conditional insert, or the
                    // insert result did not prove that this request published
                    // it. Deleting either side would destroy an owner we
                    // cannot identify.
                    ownership_uncertain = true;
                } else if (lost_race[i]) {
                    // EC_EXIST proves this candidate was never published.
                    // Its fresh allocation remains independently owned even
                    // though the winning metadata changed again before the
                    // durable reload.
                    direct_deletes.push_back(candidates[i]);
                } else {
                    // If this request inserted the owner (or the RMW result
                    // was too malformed to prove it did not), a replacement
                    // can also mean that the old URI was freed and reused.
                    // This request no longer has a durable owner ledger, and
                    // cannot prove an unsupported actor did not move the same
                    // generation under another key. Keep it fail-closed;
                    // generation identity alone is not global ownership.
                    ownership_uncertain = true;
                }
            } else if (current[i].ec == EC_NOENT) {
                if (lost_race[i]) {
                    direct_deletes.push_back(candidates[i]);
                } else {
                    ownership_uncertain = true;
                }
            } else {
                ownership_uncertain = true;
            }
        }
        DeleteItemsOptions cleanup_options;
        cleanup_options.adjust_storage_usage = false;
        cleanup_options.restore_usage_on_sync_failure = false;
        const auto metadata_cleanup =
            DeleteItems(request_context, internal_instance_id, exact_deletes, cleanup_options);
        if (metadata_cleanup.ec != EC_OK) {
            KVCM_LOG_WARN("KVMeta start rollback retained durable cleanup tombstones, ec[%d]", metadata_cleanup.ec);
        }
        const ErrorCode direct_cleanup_ec = delete_allocated_noexcept(direct_deletes);
        if (direct_cleanup_ec != EC_OK) {
            KVCM_LOG_WARN("KVMeta start rollback left unpublished allocations for backend orphan cleanup, ec[%d]",
                          direct_cleanup_ec);
        }
        if (!metadata_cleanup.metadata_cleanup_complete || metadata_cleanup.metadata_already_absent ||
            ownership_uncertain) {
            AddError(request_context,
                     "KVMeta start rollback could not prove exclusive ownership; recovery is required");
            CancelMaintenance();
            return EC_OUTCOME_UNKNOWN;
        }
        if (metadata_cleanup.ec != EC_OK || direct_cleanup_ec != EC_OK) {
            AddError(request_context,
                     "KVMeta start rollback completed logically but one physical delete outcome is unknown");
            return EC_OUTCOME_UNKNOWN;
        }
        return original_error;
    };

    if (insert_error != EC_OK) {
        return {rollback_start(insert_error), StartWriteResult{}};
    }

    // A different process can win the exact metadata insertion between our
    // initial read and conditional insert. Treat it as an existing value only
    // after validating the complete schema and the caller-declared size.
    std::vector<std::string> race_winner_keys;
    std::vector<std::size_t> race_winner_indices;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (lost_race[i]) {
            race_winner_keys.push_back(candidates[i].original_key);
            race_winner_indices.push_back(i);
        }
    }
    if (!race_winner_keys.empty()) {
        std::vector<ExactLocation> race_winners;
        const ErrorCode reload_ec =
            LoadExactLocations(request_context, internal_instance_id, race_winner_keys, race_winners);
        if (reload_ec != EC_OK || race_winners.size() != race_winner_indices.size()) {
            return {rollback_start(reload_ec == EC_OK ? EC_MISMATCH : reload_ec), StartWriteResult{}};
        }
        for (std::size_t i = 0; i < race_winners.size(); ++i) {
            const std::size_t candidate_index = race_winner_indices[i];
            std::uint64_t winner_size = 0;
            if (race_winners[i].ec != EC_OK || !race_winners[i].location) {
                return {rollback_start(race_winners[i].ec == EC_OK ? EC_CORRUPTION : race_winners[i].ec),
                        StartWriteResult{}};
            }
            const ErrorCode validate_ec = ValidateOwnedLocation(request_context,
                                                                internal_instance_id,
                                                                race_winners[i].internal_key,
                                                                race_winners[i].location_id,
                                                                *race_winners[i].location,
                                                                winner_size);
            if (validate_ec != EC_OK) {
                return {rollback_start(validate_ec), StartWriteResult{}};
            }
            if (candidates[candidate_index].data_location &&
                SamePhysicalAllocation(*race_winners[i].location, *candidates[candidate_index].data_location)) {
                AddError(request_context,
                         "KVMeta storage reused one allocation across concurrent singleton Create calls");
                return {rollback_start(EC_CORRUPTION), StartWriteResult{}};
            }
            // The conditional insert can lose to a writer whose metadata is
            // valid but not committed yet. Do not turn that transient state
            // or its provisional size into a permanent mismatch: our own
            // candidate allocations must be rolled back and the caller must
            // retry explicitly after the winner settles.
            if (!IsCommittedObject(*race_winners[i].location)) {
                AddError(request_context, "KVMeta concurrent winner is still writing");
                return {rollback_start(EC_EXIST), StartWriteResult{}};
            }
            if (winner_size != candidates[candidate_index].value_size) {
                AddError(request_context, "KVMeta concurrent winner has a different value size");
                return {rollback_start(EC_MISMATCH), StartWriteResult{}};
            }
        }
    }

    std::vector<SessionItem> race_losers;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (lost_race[i]) {
            race_losers.push_back(candidates[i]);
        }
    }
    if (!release_allocated_once(race_losers, "KVMeta failed to release one or more race-loser allocations")) {
        return {EC_OUTCOME_UNKNOWN, StartWriteResult{}};
    }

    std::vector<SessionItem> session_items;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (inserted[i]) {
            session_items.push_back(candidates[i]);
        }
    }
    if (session_items.empty()) {
        return {EC_OK, std::move(response)};
    }

    std::vector<std::int64_t> inserted_keys;
    inserted_keys.reserve(session_items.size());
    for (const auto &item : session_items) {
        inserted_keys.push_back(item.internal_key);
    }

    // Until Put() publishes a write session, no owner exists that can finish
    // or expire these active reservations. Every failure in this interval
    // must therefore prove their metadata absence and physical cleanup before
    // returning an error that callers may retry. Either uncertainty
    // fail-closes KVMeta alone.
    const auto rollback_unpublished_reservations = [&](bool adjust_storage_usage) -> ErrorCode {
        DeleteItemsOptions cleanup_options;
        cleanup_options.adjust_storage_usage = adjust_storage_usage;
        cleanup_options.restore_usage_on_sync_failure = adjust_storage_usage;
        // DeleteItems first persists a read-invisible, immediate tombstone.
        // Retry-safe backends retain it as a cleanup ledger; reusable legacy
        // addresses are attempted once and then finalized logically.
        const auto cleanup = DeleteItems(request_context, internal_instance_id, session_items, cleanup_options);
        if (!cleanup.metadata_cleanup_complete || cleanup.metadata_already_absent) {
            AddError(request_context,
                     "KVMeta unpublished reservation rollback was incomplete; maintenance is fail-closed until "
                     "recovery");
            CancelMaintenance();
            return EC_OUTCOME_UNKNOWN;
        }
        if (cleanup.ec != EC_OK) {
            KVCM_LOG_WARN("KVMeta unpublished reservation rollback completed with a physical cleanup warning, ec[%d]",
                          cleanup.ec);
            return EC_OUTCOME_UNKNOWN;
        }
        return EC_OK;
    };

    if (!indexer->Sync(inserted_keys)) {
        if (rollback_unpublished_reservations(/*adjust_storage_usage=*/false) != EC_OK) {
            return {EC_OUTCOME_UNKNOWN, StartWriteResult{}};
        }
        AddError(request_context, "KVMeta metadata reservation did not reach its persistence barrier");
        return {EC_TIMEOUT, StartWriteResult{}};
    }

    for (const auto &item : session_items) {
        KvMetaManager::ValueLocation location;
        if (!ToValueLocation(*item.metadata_location, location)) {
            if (rollback_unpublished_reservations(/*adjust_storage_usage=*/false) != EC_OK) {
                return {EC_OUTCOME_UNKNOWN, StartWriteResult{}};
            }
            return {EC_CORRUPTION, StartWriteResult{}};
        }
        response.locations.push_back(std::move(location));
    }
    for (const auto &item : session_items) {
        indexer->AddStorageUsageByType(item.metadata_location->type(), item.value_size);
    }

    std::string session_id;
    auto session_result = KvMetaWriteSessionManager::PutResult::kDuplicate;
    bool session_publication_threw = false;
    try {
        for (int attempt = 0; attempt < 8 && session_result == KvMetaWriteSessionManager::PutResult::kDuplicate;
             ++attempt) {
            session_id = StringUtil::GenerateRandomString(32);
            auto items_for_attempt = session_items;
            session_result = write_session_manager_
                                 ? write_session_manager_->Put(session_id,
                                                               internal_instance_id,
                                                               quota_shard,
                                                               std::move(items_for_attempt),
                                                               write_deadline,
                                                               cleanup_deadline,
                                                               failed_write_cleanup_grace_seconds > 0)
                                 : KvMetaWriteSessionManager::PutResult::kStopped;
        }
    } catch (const std::exception &) {
        session_publication_threw = true;
        KVCM_LOG_WARN("KVMeta write-session publication caught a standard internal exception");
    } catch (...) {
        session_publication_threw = true;
        KVCM_LOG_WARN("KVMeta write-session publication caught an unknown internal exception");
    }
    if (session_publication_threw) {
        if (rollback_unpublished_reservations(/*adjust_storage_usage=*/true) != EC_OK) {
            return {EC_OUTCOME_UNKNOWN, StartWriteResult{}};
        }
        AddError(request_context, "KVMeta could not publish the write session");
        return {EC_ERROR, StartWriteResult{}};
    }
    if (session_result != KvMetaWriteSessionManager::PutResult::kOk) {
        if (rollback_unpublished_reservations(/*adjust_storage_usage=*/true) != EC_OK) {
            return {EC_OUTCOME_UNKNOWN, StartWriteResult{}};
        }
        if (session_result == KvMetaWriteSessionManager::PutResult::kFull) {
            AddError(request_context, "KVMeta active write-session limit has been reached");
            return {EC_NOSPC, StartWriteResult{}};
        }
        if (session_result == KvMetaWriteSessionManager::PutResult::kStopped) {
            AddError(request_context, "KVMeta write-session manager is stopped");
            return {EC_SERVICE_NOT_LEADER, StartWriteResult{}};
        }
        if (session_result == KvMetaWriteSessionManager::PutResult::kExpired) {
            AddError(request_context, "KVMeta write lease expired before PutStart completed");
            return {EC_TIMEOUT, StartWriteResult{}};
        }
        AddError(request_context, "KVMeta could not generate a unique write-session id");
        return {EC_ERROR, StartWriteResult{}};
    }
    response.write_session_id = std::move(session_id);
    response.session_item_count = session_items.size();
    for (const auto &location : response.locations) {
        data_storage_manager->RecordWriteBytes(selected.name, location.value_size);
    }
    return {EC_OK, std::move(response)};
}

ErrorCode KvMetaManager::FinishWriteInternal(RequestContext *request_context,
                                             const std::string &internal_instance_id,
                                             const std::vector<bool> &success_keys,
                                             const std::vector<SessionItem> &items) {
    if (items.empty() || success_keys.size() != items.size()) {
        return EC_BADARGS;
    }
    // The protocol uses all-or-nothing failure handling. A single failed value
    // aborts the complete session, which also makes packed/remote backend
    // semantics unsurprising even though current allocations are singleton.
    if (std::any_of(success_keys.begin(), success_keys.end(), [](bool success) { return !success; })) {
        const auto cleanup = DeleteItems(request_context, internal_instance_id, items, DeleteItemsOptions{});
        if (!cleanup.metadata_cleanup_complete || cleanup.metadata_already_absent) {
            // Take/expiry has consumed the only in-memory owner. Do not admit
            // a successor generation while durable metadata may still refer
            // to this allocation. An already-absent record is equally unsafe:
            // the old URI may already have been freed and reused, so it was
            // intentionally not sent to physical Delete.
            AddError(request_context,
                     "KVMeta session rollback did not prove metadata absence; maintenance is fail-closed until "
                     "recovery");
            CancelMaintenance();
            return EC_OUTCOME_UNKNOWN;
        }
        return cleanup.ec;
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (!indexer) {
        return EC_INSTANCE_NOT_EXIST;
    }
    std::vector<CacheLocationConstPtr> committed_locations;
    std::vector<std::int64_t> item_keys;
    committed_locations.reserve(items.size());
    item_keys.reserve(items.size());
    for (const auto &item : items) {
        if (!item.metadata_location || item.metadata_location->status() != CLS_NEW ||
            item.metadata_location->create_time() <= 0) {
            return EC_CORRUPTION;
        }
        committed_locations.push_back(MakeCommittedLocation(*item.metadata_location));
        item_keys.push_back(item.internal_key);
    }

    ErrorCode commit_error = EC_OK;
    for (const auto &layer : MakeUniqueKeyLayers(item_keys)) {
        KeyVector keys;
        LocationIdsPerKey ids;
        std::vector<CacheLocationConstPtr> expected;
        std::vector<CacheLocationConstPtr> replacement;
        for (const std::size_t index : layer) {
            keys.push_back(items[index].internal_key);
            ids.push_back({items[index].location_id});
            expected.push_back(items[index].metadata_location);
            replacement.push_back(committed_locations[index]);
        }
        std::vector<bool> modifier_committed(layer.size(), false);
        auto modifier = [&expected, &replacement, &modifier_committed](const std::vector<ErrorCode> &get_ecs,
                                                                       const LocationIdVector &,
                                                                       std::size_t key_index,
                                                                       CacheLocationVector &locations,
                                                                       PropertyMap &) -> LocationModifierResult {
            if (get_ecs.size() != 1 || locations.size() != 1 || key_index >= expected.size()) {
                return {MA_FAIL, {EC_MISMATCH}};
            }
            if (get_ecs[0] != EC_OK) {
                return {MA_FAIL, {get_ecs[0]}};
            }
            if (!locations[0] || locations[0]->ToJsonString() != expected[key_index]->ToJsonString()) {
                return {MA_SKIP, {EC_MISMATCH}};
            }
            locations[0] = replacement[key_index];
            modifier_committed[key_index] = true;
            return {MA_OK, {EC_OK}};
        };
        const auto rmw = indexer->ReadModifyWriteLocation(request_context, keys, ids, modifier);
        if (rmw.per_location_error_codes.size() != layer.size()) {
            commit_error = FirstHardError(commit_error, EC_MISMATCH);
            break;
        }
        for (std::size_t i = 0; i < layer.size(); ++i) {
            if (rmw.per_location_error_codes[i].size() != 1) {
                commit_error = FirstHardError(commit_error, EC_MISMATCH);
                continue;
            }
            const ErrorCode ec = rmw.per_location_error_codes[i][0];
            if (ec != EC_OK || !modifier_committed[i]) {
                // A reservation disappearing before commit is a failed CAS,
                // not an idempotent absence. FirstHardError intentionally
                // ignores EC_NOENT for read/remove aggregation, so normalize
                // it here or a partially committed batch could be reported
                // as successful without running rollback.
                commit_error = FirstHardError(commit_error, ec == EC_OK || ec == EC_NOENT ? EC_MISMATCH : ec);
            }
        }
        if (rmw.ec != EC_OK && rmw.ec != EC_PARTIAL_OK) {
            commit_error = FirstHardError(commit_error, rmw.ec);
        }
        if (commit_error != EC_OK) {
            break;
        }
    }
    if (commit_error == EC_OK && !indexer->Sync(item_keys)) {
        commit_error = EC_TIMEOUT;
    }
    if (commit_error == EC_OK) {
        return EC_OK;
    }

    // Commit may have changed only a prefix of the session. Persist that
    // exact current view before classifying any allocation as unreferenced;
    // otherwise a failed barrier plus an in-memory miss could make rollback
    // physically delete an allocation still referenced by durable metadata.
    if (!indexer->Sync(item_keys)) {
        AddError(request_context,
                 "KVMeta commit rollback could not establish a durable metadata view; maintenance is fail-closed "
                 "until recovery");
        CancelMaintenance();
        return EC_OUTCOME_UNKNOWN;
    }

    std::vector<std::string> original_keys;
    original_keys.reserve(items.size());
    for (const auto &item : items) {
        original_keys.push_back(item.original_key);
    }
    std::vector<ExactLocation> current;
    const ErrorCode reload_ec = LoadExactLocations(request_context, internal_instance_id, original_keys, current);
    if (reload_ec != EC_OK || current.size() != items.size()) {
        AddError(request_context,
                 "KVMeta commit rollback could not reload exact ownership; maintenance is fail-closed until "
                 "recovery");
        CancelMaintenance();
        return EC_OUTCOME_UNKNOWN;
    }
    std::vector<SessionItem> exact_deletes;
    bool ownership_uncertain = false;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (current[i].ec == EC_OK && current[i].location) {
            std::uint64_t current_size = 0;
            if (ValidateOwnedLocation(request_context,
                                      internal_instance_id,
                                      items[i].internal_key,
                                      items[i].location_id,
                                      *current[i].location,
                                      current_size) != EC_OK) {
                ownership_uncertain = true;
                continue;
            }
            const std::string current_value = current[i].location->ToJsonString();
            if (current_value == items[i].metadata_location->ToJsonString() ||
                current_value == committed_locations[i]->ToJsonString()) {
                SessionItem item = items[i];
                item.metadata_location = current[i].location;
                exact_deletes.push_back(std::move(item));
            } else if (items[i].data_location &&
                       SamePhysicalAllocation(*current[i].location, *items[i].data_location)) {
                // A concurrent actor changed ownership metadata without
                // changing the URI. It is unsafe both to CAS-delete that
                // unexpected owner and to physically delete its allocation.
                ownership_uncertain = true;
            } else {
                // Supported KVMeta operations cannot replace an active
                // session with another allocation. Even though the old URI
                // is no longer referenced by this metadata, it may already
                // have been freed and reused. Preserve it as an orphan and
                // rebuild ownership/counters before admitting mutations.
                ownership_uncertain = true;
            }
        } else if (current[i].ec == EC_NOENT) {
            // StartWrite charged this session's bytes before publication. An
            // absent record here can only come from an unsupported concurrent
            // mutation or an uncertain metadata outcome. It is not authority
            // to delete a reusable address, and the byte counter is no longer
            // trustworthy.
            ownership_uncertain = true;
        } else {
            ownership_uncertain = true;
        }
    }
    const auto exact_cleanup = DeleteItems(request_context, internal_instance_id, exact_deletes, DeleteItemsOptions{});
    if (!exact_cleanup.metadata_cleanup_complete || exact_cleanup.metadata_already_absent) {
        AddError(request_context,
                 "KVMeta commit rollback did not prove metadata absence; maintenance is fail-closed until recovery");
        CancelMaintenance();
        ownership_uncertain = true;
    }
    if (exact_cleanup.ec != EC_OK) {
        KVCM_LOG_WARN("KVMeta commit rollback completed with a physical cleanup warning, ec[%d]", exact_cleanup.ec);
    }
    if (ownership_uncertain) {
        AddError(request_context,
                 "KVMeta commit rollback observed an unexpected replacement or missing owner; recovery is required");
        CancelMaintenance();
    }
    // Unlike a reservation that never became visible, this path follows a
    // partially applied metadata commit. Logical cleanup may be complete even
    // when a legacy backend reports an unknown physical outcome.
    if (exact_cleanup.ec != EC_OK) {
        AddError(request_context, "KVMeta partial commit rollback had an unknown physical cleanup outcome");
        return EC_OUTCOME_UNKNOWN;
    }
    if (ownership_uncertain) {
        return EC_OUTCOME_UNKNOWN;
    }
    return commit_error;
}

ErrorCode KvMetaManager::FinishWrite(RequestContext *request_context,
                                     const std::string &instance_id,
                                     const std::string &write_session_id,
                                     const std::vector<bool> &success_keys) {
    if (!initialized_.load(std::memory_order_acquire) || ValidateInstanceId(request_context, instance_id) != EC_OK ||
        write_session_id.empty() || write_session_id.size() > limits_.max_write_session_id_bytes ||
        success_keys.empty() || !write_session_manager_) {
        AddError(request_context, "KVMeta FinishWrite has an invalid session id or success mask");
        return EC_BADARGS;
    }
    const bool all_success =
        std::all_of(success_keys.begin(), success_keys.end(), [](bool success) { return success; });
    KvMetaWriteSessionManager::Session session;
    auto [take_result, finalization] = write_session_manager_->Take(write_session_id,
                                                                    InternalInstanceId(instance_id),
                                                                    std::optional<std::size_t>{success_keys.size()},
                                                                    all_success,
                                                                    session);
    switch (take_result) {
    case KvMetaWriteSessionManager::TakeResult::kNotFound:
        AddError(request_context, "KVMeta write session does not exist or has expired");
        return EC_NOENT;
    case KvMetaWriteSessionManager::TakeResult::kInstanceMismatch:
        AddError(request_context, "KVMeta write session belongs to another instance");
        return EC_BADARGS;
    case KvMetaWriteSessionManager::TakeResult::kSizeMismatch:
        AddError(request_context, "KVMeta success mask size does not match the write session");
        return EC_MISMATCH;
    case KvMetaWriteSessionManager::TakeResult::kDeferred:
        // The abort is durable through the active metadata lease and owned by
        // the expiry worker. Keeping it charged and invisible is intentional.
        return EC_OK;
    case KvMetaWriteSessionManager::TakeResult::kExpiredDeferred:
        AddError(request_context, "KVMeta write session expired; failed-write cleanup remains quarantined");
        return EC_TIMEOUT;
    case KvMetaWriteSessionManager::TakeResult::kAborted:
        AddError(request_context, "KVMeta write session was already aborted");
        return EC_EXIST;
    case KvMetaWriteSessionManager::TakeResult::kExpired:
    case KvMetaWriteSessionManager::TakeResult::kOk:
        break;
    }
    if (session.items.empty()) {
        AddError(request_context, "KVMeta write session is empty; admission is fail-closed until recovery");
        CancelMaintenance();
        return EC_OUTCOME_UNKNOWN;
    }
    if (session.quota_shard >= quota_admission_mutexes_.size()) {
        AddError(request_context,
                 "KVMeta write session has an invalid quota shard; admission is fail-closed until recovery");
        CancelMaintenance();
        return EC_OUTCOME_UNKNOWN;
    }
    // Keep commit/rollback finalization in the same KVMeta-only group critical
    // section as allocation, Remove and Trim. In particular, rollback must
    // finish deleting the old physical object before a new generation can be
    // allocated for the key.
    std::unique_lock<std::mutex> quota_lock(quota_admission_mutexes_[session.quota_shard]);
    // Take removes the session before waiting for the group shard. Another
    // KVMeta operation in the same group can hold that shard across backend
    // I/O, so the lease may expire while this finalizer is queued. Recheck
    // after acquiring the lock; otherwise a value could become visible after
    // the timeout promised to both the client and leader-recovery logic.
    const auto cleanup_active_session = [&](ErrorCode completed_result) {
        ErrorCode cleanup_ec = EC_IO_ERROR;
        const char *failure_kind = "error_code";
        bool cleanup_threw = false;
        try {
            const std::vector<bool> failed(session.items.size(), false);
            cleanup_ec = FinishWriteInternal(request_context, session.internal_instance_id, failed, session.items);
        } catch (const std::exception &) {
            failure_kind = "standard_exception";
            cleanup_threw = true;
        } catch (...) {
            failure_kind = "unknown_exception";
            cleanup_threw = true;
        }
        if (cleanup_threw) {
            // Take() already consumed the only session owner. Preserve an
            // expired caller's timeout result, but fail-close new mutations
            // until recovery establishes the durable metadata state.
            CancelMaintenance();
            cleanup_ec = EC_OUTCOME_UNKNOWN;
        }
        if (cleanup_ec == EC_OK) {
            return completed_result;
        }
        // The physical-delete outcome may be ambiguous. FinishWriteInternal
        // either retains a retry-safe ownership tombstone or finalizes a
        // non-retry-safe legacy address while closing KVMeta. Do not turn
        // either case into a replay-safe session result.
        KVCM_LOG_WARN("KVMeta active write cleanup did not complete; durable cleanup recovery is required, "
                      "item_count[%zu], failure[%s], ec[%d]",
                      session.items.size(),
                      failure_kind,
                      cleanup_ec);
        AddError(request_context, "KVMeta active write cleanup failed; durable cleanup recovery is required");
        // The session may also have expired, but the cleanup mutation is now
        // ambiguous and has tripped the KVMeta circuit breaker. Preserve the
        // stronger outcome so no caller treats this as a replay-safe timeout.
        return EC_OUTCOME_UNKNOWN;
    };

    const bool expired = take_result == KvMetaWriteSessionManager::TakeResult::kExpired ||
                         KvMetaWriteSessionManager::Clock::now() >= session.commit_deadline;
    if (expired) {
        AddError(request_context, "KVMeta write session expired before FinishWrite");
        return cleanup_active_session(EC_TIMEOUT);
    }
    if (!all_success) {
        return cleanup_active_session(EC_OK);
    }
    // Successful commits are observed by the bounded periodic reclaim round.
    // Waking on every PutFinish would turn sustained write QPS into registry
    // scan QPS even when the group is far below its watermark. EC_NOSPC still
    // wakes the worker immediately from StartWrite's admission path.
    try {
        return FinishWriteInternal(request_context, session.internal_instance_id, success_keys, session.items);
    } catch (const std::exception &) {
        KVCM_LOG_WARN("KVMeta commit finalization caught a standard internal exception; recovery is required");
    } catch (...) {
        KVCM_LOG_WARN("KVMeta commit finalization caught an unknown internal exception; recovery is required");
    }
    AddError(request_context, "KVMeta commit finalization outcome is unknown; recovery is required");
    CancelMaintenance();
    return EC_OUTCOME_UNKNOWN;
}

ErrorCode KvMetaManager::Remove(RequestContext *request_context,
                                const std::string &instance_id,
                                const std::vector<std::string> &keys) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return EC_ERROR;
    }
    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        return EC_SERVICE_NOT_LEADER;
    }
    const auto [instance_ec, instance_info] = GetValidatedInstanceInfo(request_context, instance_id);
    if (instance_ec != EC_OK || !instance_info) {
        return instance_ec == EC_OK ? EC_CORRUPTION : instance_ec;
    }
    if (const ErrorCode ec = ValidateKeys(request_context, keys); ec != EC_OK) {
        return ec;
    }

    // Keep metadata removal, its persistence barrier, and physical deletion
    // in the same KVMeta-only group critical section as StartWrite. Without
    // this ordering, a new generation of the same key could allocate after
    // the old metadata disappears but before its physical delete completes.
    // A backend that quickly reuses allocation URIs would then expose an ABA
    // window in which the old Remove can delete the new generation's object.
    // Regular fixed-block KV-cache operations never take these shard locks.
    const std::size_t quota_shard =
        std::hash<std::string>{}(instance_info->instance_group_name()) % quota_admission_mutexes_.size();
    std::unique_lock<std::mutex> quota_lock(quota_admission_mutexes_[quota_shard]);
    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        return EC_SERVICE_NOT_LEADER;
    }

    const std::string internal_instance_id = InternalInstanceId(instance_id);
    if (trimming_instances_[quota_shard].count(internal_instance_id) != 0) {
        AddError(request_context, "KVMeta instance is being trimmed");
        return EC_EXIST;
    }
    std::vector<ExactLocation> exact;
    if (const ErrorCode ec = LoadExactLocations(request_context, internal_instance_id, keys, exact); ec != EC_OK) {
        return ec;
    }
    std::vector<SessionItem> items;
    for (std::size_t i = 0; i < exact.size(); ++i) {
        if (exact[i].ec == EC_NOENT) {
            continue;
        }
        if (exact[i].ec != EC_OK || !exact[i].location) {
            return exact[i].ec == EC_OK ? EC_CORRUPTION : exact[i].ec;
        }
        std::uint64_t size = 0;
        if (const ErrorCode ec = ValidateOwnedLocation(request_context,
                                                       internal_instance_id,
                                                       exact[i].internal_key,
                                                       exact[i].location_id,
                                                       *exact[i].location,
                                                       size);
            ec != EC_OK) {
            return ec;
        }
        // Do not let an independent Remove invalidate a writer's allocation
        // while its session can still commit or roll back that same URI.
        // Session timeout/PutFinish owns cleanup of active generations.
        if (!IsCommittedObject(*exact[i].location)) {
            AddError(request_context, "KVMeta cannot remove a value while its write session is active");
            return EC_EXIST;
        }
        items.push_back(SessionItem{
            i, keys[i], exact[i].internal_key, exact[i].location_id, exact[i].location, exact[i].location, size});
    }
    DeleteItemsResult deletion;
    deletion.ec = EC_IO_ERROR;
    try {
        DeleteItemsOptions options;
        options.capacity_release_group = instance_info->instance_group_name();
        deletion = DeleteItems(request_context, internal_instance_id, items, options);
    } catch (const std::exception &) {
        KVCM_LOG_WARN("KVMeta Remove caught a standard internal exception; recovery is required");
        AddError(request_context, "KVMeta Remove outcome is unknown after an internal exception");
        CancelMaintenance();
        return EC_OUTCOME_UNKNOWN;
    } catch (...) {
        KVCM_LOG_WARN("KVMeta Remove caught an unknown internal exception; recovery is required");
        AddError(request_context, "KVMeta Remove outcome is unknown after an internal exception");
        CancelMaintenance();
        return EC_OUTCOME_UNKNOWN;
    }
    if (deletion.metadata_already_absent || deletion.metadata_owner_conflicted) {
        // These items were present in the exact read under the group shard.
        // Seeing one disappear or change before the compare-and-delete
        // therefore means an unsupported concurrent mutation (for example,
        // another leader). Its ownership/usage side effect is unknowable, so
        // only recovery may rebuild the authoritative state.
        CancelMaintenance();
        AddError(request_context,
                 "KVMeta Remove observed an unexpected missing or replaced owner; recovery is required");
        return EC_OUTCOME_UNKNOWN;
    }
    if (deletion.ec != EC_OK && deletion.metadata_outcome_changed) {
        // Some keys are durably/in-memory deleted, while a later metadata or
        // physical step failed. Returning the underlying
        // ordinary error would invite an unsafe blind retry that can delete a
        // successor generation created for one of those keys.
        if (!deletion.metadata_cleanup_complete) {
            // In-memory absence without a durable barrier must also block new
            // generations locally. Outcome-unknown alone is only a caller
            // contract; it is not an admission fence against other clients.
            CancelMaintenance();
        }
        AddError(request_context, "KVMeta Remove partially applied; final outcome must be reconciled");
        return EC_OUTCOME_UNKNOWN;
    }
    return deletion.ec;
}

ErrorCode KvMetaManager::TrimAll(RequestContext *request_context, const std::string &instance_id, bool metadata_only) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return EC_ERROR;
    }
    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        return EC_SERVICE_NOT_LEADER;
    }
    const auto [instance_ec, instance_info] = GetValidatedInstanceInfo(request_context, instance_id);
    if (instance_ec != EC_OK || !instance_info) {
        return instance_ec == EC_OK ? EC_CORRUPTION : instance_ec;
    }
    const std::string internal_instance_id = InternalInstanceId(instance_id);
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (!indexer) {
        return EC_INSTANCE_NOT_EXIST;
    }

    // Fail fast when a session is already known to be active/finalizing. A
    // finalizer can hold the group shard across slow physical I/O, and Trim
    // should report WRITE_IN_PROGRESS rather than wait behind that I/O. The
    // second check below remains mandatory to close the check-to-lock race.
    if (write_session_manager_ && write_session_manager_->HasSessionForInstance(internal_instance_id)) {
        AddError(request_context, "KVMeta cannot trim an instance while a write session is active or finalizing");
        return EC_EXIST;
    }
    if (reclaimer_ && reclaimer_->HasPendingForInstance(internal_instance_id)) {
        AddError(request_context, "KVMeta cannot trim an instance while automatic reclaim is pending");
        return EC_EXIST;
    }

    // Publish a per-instance fence under the same shard used by StartWrite,
    // Remove and reclaim retirement. The fence closes the check-to-lock race,
    // while releasing the shard below prevents an unbounded metadata scan or
    // slow physical delete from stalling unrelated instances in this group.
    const std::size_t quota_shard =
        std::hash<std::string>{}(instance_info->instance_group_name()) % quota_admission_mutexes_.size();
    std::unique_lock<std::mutex> quota_lock(quota_admission_mutexes_[quota_shard]);
    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        return EC_SERVICE_NOT_LEADER;
    }
    // StartWrite holds the same group shard until its active metadata and
    // session are both installed (or fully rolled back), so no new writer for
    // this instance can appear after this check. Never let Trim free a remote
    // allocation while its owner can still be writing into it. A session stays
    // protected after Take removes it from the lookup map until FinishWrite or
    // timeout cleanup has completed all metadata and physical-storage I/O.
    if (write_session_manager_ && write_session_manager_->HasSessionForInstance(internal_instance_id)) {
        AddError(request_context, "KVMeta cannot trim an instance while a write session is active or finalizing");
        return EC_EXIST;
    }
    if (reclaimer_ && reclaimer_->HasPendingForInstance(internal_instance_id)) {
        AddError(request_context, "KVMeta cannot trim an instance while automatic reclaim is pending");
        return EC_EXIST;
    }
    try {
        if (!trimming_instances_[quota_shard].insert(internal_instance_id).second) {
            AddError(request_context, "KVMeta instance is already being trimmed");
            return EC_EXIST;
        }
    } catch (...) {
        AddError(request_context, "KVMeta could not publish the trim admission fence");
        return EC_ERROR;
    }

    struct TrimMarkerGuard {
        std::mutex *mutex = nullptr;
        std::unordered_set<std::string> *instances = nullptr;
        const std::string *instance_id = nullptr;

        ~TrimMarkerGuard() noexcept {
            if (!mutex || !instances || !instance_id) {
                return;
            }
            std::lock_guard<std::mutex> lock(*mutex);
            instances->erase(*instance_id);
        }
    } trim_marker{&quota_admission_mutexes_[quota_shard], &trimming_instances_[quota_shard], &internal_instance_id};
    quota_lock.unlock();

    bool trim_metadata_changed = false;
    bool trim_delete_in_progress = false;
    bool trim_metadata_uncertain = false;
    const auto finish_trim = [&](ErrorCode ec) {
        if (ec != EC_OK && (trim_metadata_changed || trim_metadata_uncertain)) {
            AddError(request_context, "KVMeta Trim partially applied; final outcome must be reconciled");
            return EC_OUTCOME_UNKNOWN;
        }
        return ec;
    };

    try {
        for (;;) {
            if (maintenance_cancelled_.load(std::memory_order_acquire)) {
                return finish_trim(EC_SERVICE_NOT_LEADER);
            }
            std::vector<SessionItem> batch;
            batch.reserve(kKvMetaDeleteBatchSize);
            bool found_any = false;
            ErrorCode operation_ec = EC_OK;
            const auto flush = [&]() {
                if (batch.empty() || operation_ec != EC_OK) {
                    return;
                }
                if (maintenance_cancelled_.load(std::memory_order_acquire)) {
                    operation_ec = EC_SERVICE_NOT_LEADER;
                    return;
                }
                // Keep this flag set across stack unwinding. Although normal
                // backend exceptions are contained below DeleteItems, an
                // allocation or unexpected metadata exception can otherwise
                // escape after an in-memory delete without reporting whether
                // persistence advanced.
                trim_delete_in_progress = true;
                DeleteItemsOptions delete_options;
                delete_options.delete_physical = !metadata_only;
                const auto deletion = DeleteItems(request_context, internal_instance_id, batch, delete_options);
                operation_ec = deletion.ec;
                trim_delete_in_progress = false;
                trim_metadata_changed = trim_metadata_changed || deletion.metadata_outcome_changed;
                if (deletion.metadata_already_absent || deletion.metadata_owner_conflicted) {
                    // The scan captured this exact owner while the per-instance
                    // trim fence was active. Its disappearance or replacement
                    // proves an out-of-protocol mutation whose ownership and
                    // storage-usage delta are unknown. It is deliberately not
                    // sent to physical Delete because its URI could already
                    // belong to a successor.
                    trim_metadata_uncertain = true;
                    CancelMaintenance();
                    operation_ec = EC_OUTCOME_UNKNOWN;
                }
                if (deletion.metadata_outcome_changed && !deletion.metadata_cleanup_complete) {
                    // Keep the per-instance trim fence from turning into an
                    // admission hole when it is removed at function exit.
                    // The global KVMeta recovery gate is required until the
                    // in-memory/durable metadata split is reconciled.
                    CancelMaintenance();
                }
                batch.clear();
            };

            std::string cursor = SCAN_BASE_CURSOR;
            do {
                if (maintenance_cancelled_.load(std::memory_order_acquire)) {
                    return finish_trim(EC_SERVICE_NOT_LEADER);
                }
                std::string next_cursor;
                KeyVector keys;
                const ErrorCode scan_ec =
                    indexer->Scan(request_context, cursor, kRecoveryScanBatchSize, next_cursor, keys);
                if (scan_ec != EC_OK || next_cursor.empty()) {
                    return finish_trim(scan_ec == EC_OK ? EC_CORRUPTION : scan_ec);
                }
                if (!keys.empty()) {
                    CacheLocationMapVector locations;
                    const auto get_result = indexer->GetLocations(request_context, keys, locations);
                    if (get_result.ec != EC_OK || locations.size() != keys.size() ||
                        get_result.error_codes.size() != keys.size()) {
                        return finish_trim(get_result.ec == EC_OK ? EC_MISMATCH : get_result.ec);
                    }
                    for (std::size_t i = 0; i < keys.size() && operation_ec == EC_OK; ++i) {
                        if (maintenance_cancelled_.load(std::memory_order_acquire)) {
                            return finish_trim(EC_SERVICE_NOT_LEADER);
                        }
                        if (get_result.error_codes[i] != EC_OK || locations[i].empty()) {
                            return finish_trim(get_result.error_codes[i] == EC_OK ? EC_CORRUPTION
                                                                                  : get_result.error_codes[i]);
                        }
                        for (const auto &[location_id, location] : locations[i]) {
                            if (!location) {
                                operation_ec = EC_CORRUPTION;
                                break;
                            }
                            std::uint64_t size = 0;
                            operation_ec = ValidateOwnedLocation(
                                request_context, internal_instance_id, keys[i], location_id, *location, size);
                            if (operation_ec != EC_OK) {
                                break;
                            }
                            found_any = true;
                            auto copy = std::make_shared<CacheLocation>(*location);
                            batch.push_back(SessionItem{0, {}, keys[i], location_id, copy, copy, size});
                            if (batch.size() == kKvMetaDeleteBatchSize) {
                                flush();
                            }
                            if (operation_ec != EC_OK) {
                                break;
                            }
                        }
                    }
                }
                if (operation_ec != EC_OK) {
                    return finish_trim(operation_ec);
                }
                cursor = std::move(next_cursor);
            } while (cursor != SCAN_BASE_CURSOR);

            flush();
            if (operation_ec != EC_OK) {
                return finish_trim(operation_ec);
            }
            if (maintenance_cancelled_.load(std::memory_order_acquire)) {
                return finish_trim(EC_SERVICE_NOT_LEADER);
            }
            if (!found_any) {
                return EC_OK;
            }
            // Deleting during a cursor scan is safe for a single captured
            // batch, but some backends provide weak cursor guarantees under
            // mutation. Restart until a complete pass observes no remaining
            // locations.
        }
    } catch (const std::exception &) {
        if (trim_delete_in_progress) {
            trim_metadata_uncertain = true;
            CancelMaintenance();
        }
        KVCM_LOG_WARN("KVMeta Trim caught a standard exception after metadata progress[%d]",
                      trim_metadata_changed || trim_metadata_uncertain ? 1 : 0);
        AddError(request_context, "KVMeta Trim failed with a standard internal exception");
        return finish_trim(EC_ERROR);
    } catch (...) {
        if (trim_delete_in_progress) {
            trim_metadata_uncertain = true;
            CancelMaintenance();
        }
        KVCM_LOG_WARN("KVMeta Trim caught an unknown exception after metadata progress[%d]",
                      trim_metadata_changed || trim_metadata_uncertain ? 1 : 0);
        AddError(request_context, "KVMeta Trim failed with an unknown internal exception");
        return finish_trim(EC_ERROR);
    }
}

ErrorCode KvMetaManager::DoRecover(std::function<bool()> should_abort) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return EC_ERROR;
    }
    // Even corrupted/far-future persisted deadlines cannot keep this isolated
    // recovery blocked forever. A legitimate active lease is bounded by the
    // same configured maximum, so waiting at most that long from promotion
    // preserves its data-I/O window without affecting main KV-cache recovery.
    const std::int64_t max_persisted_active_lease_seconds =
        limits_.max_write_timeout_seconds + limits_.max_failed_write_cleanup_grace_seconds;
    const auto recovery_force_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(max_persisted_active_lease_seconds);
    RequestContext request_context("kv_meta_recover");
    const auto [groups_ec, groups] = registry_manager_->ListInstanceGroup(&request_context);
    if (groups_ec != EC_OK) {
        return groups_ec;
    }
    std::unordered_set<std::string> recovered_kv_meta_groups;
    ErrorCode overall = EC_OK;
    for (const auto &group : groups) {
        if (should_abort && should_abort()) {
            return EC_SERVICE_NOT_LEADER;
        }
        if (!group) {
            overall = FirstHardError(overall, EC_CORRUPTION);
            continue;
        }
        const auto [instances_ec, instances] = registry_manager_->ListInstanceInfo(&request_context, group->name());
        if (instances_ec != EC_OK) {
            overall = FirstHardError(overall, instances_ec);
            continue;
        }
        const bool has_kv_meta_instance = std::any_of(instances.begin(), instances.end(), [](const auto &instance) {
            return instance && IsKvMetaInstance(*instance);
        });
        if (has_kv_meta_instance) {
            recovered_kv_meta_groups.insert(group->name());
        }
        for (const auto &instance : instances) {
            if (!instance || !IsKvMetaInstance(*instance)) {
                continue;
            }
            if (should_abort && should_abort()) {
                return EC_SERVICE_NOT_LEADER;
            }
            auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance->instance_id());
            if (!indexer) {
                overall = FirstHardError(overall, EC_INSTANCE_NOT_EXIST);
                continue;
            }
            std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> committed_usage_by_type{};
            ErrorCode instance_recovery_ec = EC_OK;
            bool removed_stale_in_pass = false;
            bool deferred_active_in_pass = false;
            do {
                removed_stale_in_pass = false;
                deferred_active_in_pass = false;
                auto earliest_deferred_deadline = std::chrono::steady_clock::time_point::max();
                const auto scan_steady_time = std::chrono::steady_clock::now();
                const std::int64_t scan_wall_time_us = TimestampUtil::GetCurrentTimeUs();
                if (scan_wall_time_us <= 0) {
                    instance_recovery_ec = FirstHardError(instance_recovery_ec, EC_ERROR);
                    break;
                }
                committed_usage_by_type.fill(0);
                std::vector<SessionItem> stale_batch;
                stale_batch.reserve(kKvMetaDeleteBatchSize);
                const auto flush_stale = [&]() {
                    if (stale_batch.empty()) {
                        return EC_OK;
                    }
                    // A retired record may represent a delete whose response
                    // was lost just before the previous leader stopped. Replay
                    // it only for a backend that explicitly guarantees retry
                    // safety. A stale active record has not yet crossed the
                    // delete fence in this recovery: DeleteItems first makes
                    // that tombstone durable and then performs one attempt.
                    std::vector<SessionItem> retry_safe;
                    std::vector<SessionItem> at_most_once_active;
                    std::vector<SessionItem> at_most_once_retired;
                    for (const auto &item : stale_batch) {
                        bool can_retry = false;
                        if (!item.data_location || !TryGetKvMetaDeleteRetrySafety(*item.data_location, can_retry)) {
                            KVCM_LOG_ERROR("KVMeta recovery cannot bind stale metadata to a valid delete policy");
                            stale_batch.clear();
                            return EC_CORRUPTION;
                        }
                        if (can_retry) {
                            retry_safe.push_back(item);
                        } else if (item.metadata_location && IsRetiredObject(*item.metadata_location)) {
                            at_most_once_retired.push_back(item);
                        } else {
                            at_most_once_active.push_back(item);
                        }
                    }
                    const auto cleanup = [&](const std::vector<SessionItem> &items, bool delete_physical) {
                        if (items.empty()) {
                            return EC_OK;
                        }
                        DeleteItemsResult result;
                        result.ec = EC_IO_ERROR;
                        try {
                            DeleteItemsOptions options;
                            options.delete_physical = delete_physical;
                            options.adjust_storage_usage = false;
                            result = DeleteItems(&request_context, instance->instance_id(), items, options);
                        } catch (const std::exception &) {
                            KVCM_LOG_WARN("KVMeta recovery cleanup caught a standard exception");
                        } catch (...) { KVCM_LOG_WARN("KVMeta recovery cleanup caught an unknown exception"); }
                        if (!result.metadata_cleanup_complete || result.metadata_already_absent ||
                            result.metadata_owner_conflicted) {
                            return result.ec == EC_OK ? EC_OUTCOME_UNKNOWN : result.ec;
                        }
                        if (result.ec == EC_OUTCOME_UNKNOWN) {
                            KVCM_LOG_WARN("KVMeta recovery finalized metadata after an at-most-once delete with an "
                                          "unknown physical outcome, item_count[%zu]",
                                          items.size());
                            return EC_OK;
                        }
                        return result.ec;
                    };

                    // Do retryable work first. If it fails, no at-most-once
                    // address has been touched and the next recovery may retry.
                    ErrorCode cleanup_ec = cleanup(retry_safe, true);
                    if (cleanup_ec == EC_OK) {
                        cleanup_ec = cleanup(at_most_once_active, true);
                    }
                    if (cleanup_ec == EC_OK) {
                        // Never replay a retired reusable GA: it may already
                        // have been freed and handed to a successor.
                        cleanup_ec = cleanup(at_most_once_retired, false);
                    }
                    stale_batch.clear();
                    return cleanup_ec;
                };

                std::string cursor = SCAN_BASE_CURSOR;
                do {
                    if (should_abort && should_abort()) {
                        return EC_SERVICE_NOT_LEADER;
                    }
                    std::string next_cursor;
                    KeyVector keys;
                    const ErrorCode scan_ec =
                        indexer->Scan(&request_context, cursor, kRecoveryScanBatchSize, next_cursor, keys);
                    if (scan_ec != EC_OK || next_cursor.empty()) {
                        instance_recovery_ec =
                            FirstHardError(instance_recovery_ec, scan_ec == EC_OK ? EC_CORRUPTION : scan_ec);
                        break;
                    }
                    if (!keys.empty()) {
                        CacheLocationMapVector locations;
                        const auto get_result = indexer->GetLocations(&request_context, keys, locations);
                        if (get_result.ec != EC_OK || locations.size() != keys.size() ||
                            get_result.error_codes.size() != keys.size()) {
                            instance_recovery_ec = FirstHardError(instance_recovery_ec,
                                                                  get_result.ec == EC_OK ? EC_MISMATCH : get_result.ec);
                            break;
                        }
                        for (std::size_t i = 0; i < keys.size() && instance_recovery_ec == EC_OK; ++i) {
                            if (get_result.error_codes[i] != EC_OK || locations[i].empty()) {
                                instance_recovery_ec = FirstHardError(
                                    instance_recovery_ec,
                                    get_result.error_codes[i] == EC_OK ? EC_CORRUPTION : get_result.error_codes[i]);
                                break;
                            }
                            for (const auto &[location_id, location] : locations[i]) {
                                if (!location) {
                                    instance_recovery_ec = FirstHardError(instance_recovery_ec, EC_CORRUPTION);
                                    break;
                                }
                                std::uint64_t size = 0;
                                const ErrorCode validate_ec = ValidateOwnedLocation(
                                    &request_context, instance->instance_id(), keys[i], location_id, *location, size);
                                const DataStorageType base_type = ToBaseType(location->type());
                                const std::size_t type_index = ToIndex(base_type);
                                if (validate_ec != EC_OK || base_type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN ||
                                    type_index >= committed_usage_by_type.size()) {
                                    instance_recovery_ec = FirstHardError(
                                        instance_recovery_ec, validate_ec == EC_OK ? EC_CORRUPTION : validate_ec);
                                    break;
                                }
                                if (IsRetiredObject(*location)) {
                                    std::int64_t retire_deadline_us = 0;
                                    if (DecodeLeaseDeadline(location->create_time(), retire_deadline_us) &&
                                        retire_deadline_us > scan_wall_time_us &&
                                        scan_steady_time < recovery_force_deadline) {
                                        const auto force_remaining_us =
                                            std::chrono::duration_cast<std::chrono::microseconds>(
                                                recovery_force_deadline - scan_steady_time)
                                                .count();
                                        if (force_remaining_us > 0) {
                                            const std::int64_t wait_us =
                                                std::min(retire_deadline_us - scan_wall_time_us, force_remaining_us);
                                            deferred_active_in_pass = true;
                                            earliest_deferred_deadline =
                                                std::min(earliest_deferred_deadline,
                                                         scan_steady_time + std::chrono::microseconds(wait_us));
                                            continue;
                                        }
                                    }
                                    removed_stale_in_pass = true;
                                    auto copy = std::make_shared<CacheLocation>(*location);
                                    stale_batch.push_back(SessionItem{0, {}, keys[i], location_id, copy, copy, size});
                                    if (stale_batch.size() == kKvMetaDeleteBatchSize) {
                                        if (should_abort && should_abort()) {
                                            return EC_SERVICE_NOT_LEADER;
                                        }
                                        instance_recovery_ec = FirstHardError(instance_recovery_ec, flush_stale());
                                        if (instance_recovery_ec != EC_OK) {
                                            break;
                                        }
                                    }
                                    continue;
                                }
                                if (location->create_time() < 0) {
                                    if (size > std::numeric_limits<std::uint64_t>::max() -
                                                   committed_usage_by_type[type_index]) {
                                        instance_recovery_ec = FirstHardError(instance_recovery_ec, EC_OUT_OF_LIMIT);
                                        break;
                                    }
                                    committed_usage_by_type[type_index] += size;
                                    continue;
                                }
                                std::int64_t lease_deadline_us = 0;
                                if (DecodeRecoveryLeaseDeadline(location->create_time(),
                                                                max_persisted_active_lease_seconds,
                                                                lease_deadline_us) &&
                                    lease_deadline_us > scan_wall_time_us &&
                                    scan_steady_time < recovery_force_deadline) {
                                    const auto force_remaining_us =
                                        std::chrono::duration_cast<std::chrono::microseconds>(recovery_force_deadline -
                                                                                              scan_steady_time)
                                            .count();
                                    if (force_remaining_us > 0) {
                                        const std::int64_t wait_us =
                                            std::min(lease_deadline_us - scan_wall_time_us, force_remaining_us);
                                        deferred_active_in_pass = true;
                                        earliest_deferred_deadline =
                                            std::min(earliest_deferred_deadline,
                                                     scan_steady_time + std::chrono::microseconds(wait_us));
                                        continue;
                                    }
                                }
                                removed_stale_in_pass = true;
                                auto copy = std::make_shared<CacheLocation>(*location);
                                stale_batch.push_back(SessionItem{0, {}, keys[i], location_id, copy, copy, size});
                                if (stale_batch.size() == kKvMetaDeleteBatchSize) {
                                    if (should_abort && should_abort()) {
                                        return EC_SERVICE_NOT_LEADER;
                                    }
                                    instance_recovery_ec = FirstHardError(instance_recovery_ec, flush_stale());
                                    if (instance_recovery_ec != EC_OK) {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if (instance_recovery_ec != EC_OK) {
                        break;
                    }
                    cursor = std::move(next_cursor);
                } while (cursor != SCAN_BASE_CURSOR);
                if (instance_recovery_ec == EC_OK) {
                    if (should_abort && should_abort()) {
                        return EC_SERVICE_NOT_LEADER;
                    }
                    instance_recovery_ec = FirstHardError(instance_recovery_ec, flush_stale());
                }
                if (instance_recovery_ec == EC_OK && deferred_active_in_pass) {
                    // Keep the KVMeta gate closed while an old leader's client
                    // may still be writing. Poll in short intervals so
                    // demotion/Stop cancellation never waits for the lease.
                    const auto poll_interval =
                        std::chrono::duration_cast<std::chrono::steady_clock::duration>(kRecoveryWaitPollInterval);
                    while (std::chrono::steady_clock::now() < earliest_deferred_deadline) {
                        if (should_abort && should_abort()) {
                            return EC_SERVICE_NOT_LEADER;
                        }
                        const auto remaining = earliest_deferred_deadline - std::chrono::steady_clock::now();
                        if (remaining <= std::chrono::steady_clock::duration::zero()) {
                            break;
                        }
                        std::this_thread::sleep_for(std::min(remaining, poll_interval));
                    }
                }
                // If this pass removed anything, cursor semantics under
                // mutation may have skipped entries and the usage sum is no
                // longer authoritative. A deferred live lease also requires a
                // rescan after its deadline. The final no-delete/no-defer pass
                // is stable and is the only one used below.
            } while (instance_recovery_ec == EC_OK && (removed_stale_in_pass || deferred_active_in_pass));
            if (should_abort && should_abort()) {
                return EC_SERVICE_NOT_LEADER;
            }
            if (instance_recovery_ec == EC_OK) {
                // Runtime mutations account exact bytes directly. A crash can
                // leave the periodically persisted usage snapshot behind the
                // durable locations, so rebuild it only after a complete,
                // error-free scan and stale-write cleanup. This is confined
                // to reserved KVMeta instances and runs after main recovery.
                for (std::size_t type_index = 1; type_index < committed_usage_by_type.size(); ++type_index) {
                    const auto type = static_cast<DataStorageType>(type_index);
                    if (ToBaseType(type) != type) {
                        continue;
                    }
                    indexer->SetStorageUsageByType(type, committed_usage_by_type[type_index]);
                }
                indexer->PersistMetaData();
            }
            if (instance_recovery_ec == EC_OUTCOME_UNKNOWN) {
                // An uncertain physical Delete is namespace-wide evidence
                // that this backend cannot currently provide a safe terminal
                // cleanup result.  Do not advance into another instance and
                // manufacture one additional orphan per indexer before the
                // caller observes the failed recovery.
                CancelMaintenance();
                return EC_OUTCOME_UNKNOWN;
            }
            overall = FirstHardError(overall, instance_recovery_ec);
        }
    }
    if (overall == EC_OK) {
        ReplaceKvMetaGroups(std::move(recovered_kv_meta_groups));
    }
    return overall;
}

} // namespace kv_cache_manager
