#include "kv_cache_manager/meta/utils.h"

#include <cstring>
#include <string_view>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/data_storage/snapshot_uri_utils.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"

namespace kv_cache_manager {

namespace {

// Compact v1 stores a 4-byte header, a little-endian int64 timestamp and the raw URI.
constexpr std::string_view kCompactLocationHeader{"\0KV\1", 4};
constexpr size_t kCompactLocationUriOffset = kCompactLocationHeader.size() + sizeof(int64_t);

bool ParseEventReportStorageType(std::string_view id, DataStorageType &out_type) {
    std::string_view storage_type;
    std::string_view medium;
    std::string_view host;
    if (!SnapshotUriUtils::ParseEventReportLocationIdView(id, storage_type, medium, host)) {
        return false;
    }
    out_type = storage_type == "event_report_l1p5" ? DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5
                                                   : DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2;
    return true;
}

} // namespace

RedisLocationCodec::RedisLocationCodec(SharedLocationSpecName simple_spec_name, bool compact_write)
    : simple_spec_name_(std::move(simple_spec_name)), compact_write_(simple_spec_name_ && compact_write) {}

bool RedisLocationCodec::SerializeCompactLocation(const LocationId &id,
                                                  const CacheLocation &location,
                                                  std::string &out_value) const {
    const std::vector<LocationSpec> &specs = location.location_specs();
    if (location.status() != CLS_SERVING || location.spec_size() != 1 || specs.size() != 1 || location.id() != id) {
        return false;
    }
    const LocationSpec &spec = specs.front();
    DataStorageType type;
    if (spec.name() != *simple_spec_name_ || spec.uri().empty() || spec.has_checksum() ||
        !ParseEventReportStorageType(id, type) || type != location.type()) {
        return false;
    }

    const uint64_t timestamp = static_cast<uint64_t>(location.create_time());
    out_value.assign(kCompactLocationHeader);
    out_value.resize(kCompactLocationUriOffset);
    unsigned char *bytes = reinterpret_cast<unsigned char *>(out_value.data() + kCompactLocationHeader.size());
    for (size_t i = 0; i < sizeof(timestamp); ++i) {
        bytes[i] = static_cast<unsigned char>(timestamp >> (8 * i));
    }
    out_value.append(spec.uri());
    return true;
}

FieldMap RedisLocationCodec::SerializeToFieldMap(const CacheLocationMap &locations, PropertyMap properties) const {
    for (const auto &[loc_id, loc_ptr] : locations) {
        auto [it, inserted] = properties.try_emplace(PROPERTY_LOCATION_PREFIX + loc_id);
        if (inserted && loc_ptr && !(compact_write_ && SerializeCompactLocation(loc_id, *loc_ptr, it->second))) {
            it->second = loc_ptr->ToJsonString();
        }
    }
    return properties;
}

bool RedisLocationCodec::DeserializeLocation(const LocationId &id,
                                             const std::string &value,
                                             LocationIdInterner *id_interner,
                                             LocationSpecNameInterner *name_interner,
                                             CacheLocation &out_location) const {
    if (value.empty()) {
        out_location.set_id(id);
        return true;
    }
    if (value.front() != kCompactLocationHeader.front()) {
        return id_interner ? out_location.FromJsonStringInterned(value, *id_interner, name_interner)
                           : out_location.FromJsonString(value);
    }

    DataStorageType type;
    if (!simple_spec_name_ || value.size() <= kCompactLocationUriOffset ||
        value.compare(0, kCompactLocationHeader.size(), kCompactLocationHeader) != 0 ||
        !ParseEventReportStorageType(id, type)) {
        return false;
    }

    const unsigned char *bytes = reinterpret_cast<const unsigned char *>(value.data() + kCompactLocationHeader.size());
    uint64_t timestamp = 0;
    for (size_t i = 0; i < sizeof(timestamp); ++i) {
        timestamp |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    int64_t create_time;
    std::memcpy(&create_time, &timestamp, sizeof(create_time));
    if (id_interner) {
        bool retained = false;
        InternedLocationId interned_id = id_interner->Intern(id, &retained);
        out_location.set_id(std::move(interned_id), retained);
    } else {
        out_location.set_id(id);
    }
    out_location.set_status(CLS_SERVING);
    out_location.set_type(type);
    out_location.set_spec_size(1);
    out_location.set_create_time(create_time);
    out_location.push_location_spec(LocationSpec(simple_spec_name_, value.substr(kCompactLocationUriOffset)));
    return true;
}

ErrorCode RedisLocationCodec::DeserializeFieldMap(const FieldMap &field_map,
                                                  CacheLocationMap &out_locations,
                                                  PropertyMap *out_properties,
                                                  LocationIdInterner *id_interner,
                                                  LocationSpecNameInterner *name_interner) const {
    for (const auto &[field_name, field_value] : field_map) {
        if (field_name.rfind(PROPERTY_LOCATION_PREFIX, 0) != 0) {
            if (out_properties) {
                (*out_properties)[field_name] = field_value;
            }
            continue;
        }
        const std::string loc_id = field_name.substr(PROPERTY_LOCATION_PREFIX.size());
        auto location = std::make_shared<CacheLocation>();
        if (!DeserializeLocation(loc_id, field_value, id_interner, name_interner, *location)) {
            return EC_CORRUPTION;
        }
        out_locations[loc_id] = std::move(location);
    }
    return EC_OK;
}

void ExtractLocationIds(const FieldMap &field_map, std::vector<LocationId> &out_location_ids) {
    for (const auto &[field_name, field_value] : field_map) {
        if (field_name.rfind(PROPERTY_LOCATION_PREFIX, 0) == 0) {
            out_location_ids.push_back(field_name.substr(PROPERTY_LOCATION_PREFIX.size()));
        }
    }
}

std::vector<std::string> AppendPrefixToKeys(const std::string &cache_key_prefix, const KeyTypeVec &keys) {
    std::vector<std::string> keys_with_prefix;
    keys_with_prefix.reserve(keys.size());
    for (const KeyType &key : keys) {
        keys_with_prefix.emplace_back(cache_key_prefix + std::to_string(key));
    }
    return keys_with_prefix;
}

bool StripPrefixInKeys(const std::string &cache_key_prefix,
                       const std::string &instance_id,
                       const std::vector<std::string> &keys_with_prefix,
                       KeyTypeVec &out_keys) {
    out_keys.clear();
    out_keys.reserve(keys_with_prefix.size());
    for (const std::string &key_with_prefix : keys_with_prefix) {
        if (key_with_prefix.size() < cache_key_prefix.size() ||
            key_with_prefix.compare(0, cache_key_prefix.size(), cache_key_prefix) != 0) {
            KVCM_LOG_ERROR("strip prefix invalid key[%s], expected prefix[%s], instance[%s]",
                           key_with_prefix.c_str(),
                           cache_key_prefix.c_str(),
                           instance_id.c_str());
            out_keys.clear();
            return false;
        }
        const std::string key_str = key_with_prefix.substr(cache_key_prefix.size());
        KeyType key = 0;
        if (!StringUtil::StrToInt64(key_str.c_str(), key)) {
            KVCM_LOG_ERROR("strip prefix invalid key[%s], can not convert to int64, instance[%s]",
                           key_with_prefix.c_str(),
                           instance_id.c_str());
            out_keys.clear();
            return false;
        }
        out_keys.emplace_back(key);
    }
    return true;
}

} // namespace kv_cache_manager
