#include "kv_cache_manager/meta/utils.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"

namespace kv_cache_manager {

FieldMap SerializeToFieldMap(const CacheLocationMap &locations, const PropertyMap &properties) {
    FieldMap field_map;
    for (const auto &[loc_id, loc_ptr] : locations) {
        field_map[PROPERTY_LOCATION_PREFIX + loc_id] = loc_ptr ? loc_ptr->ToJsonString() : "";
    }
    for (const auto &[key, value] : properties) {
        field_map[key] = value;
    }
    return field_map;
}

ErrorCode DeserializeFieldMap(const FieldMap &field_map, CacheLocationMap &out_locations, PropertyMap &out_properties) {
    for (const auto &[field_name, field_value] : field_map) {
        if (field_name.rfind(PROPERTY_LOCATION_PREFIX, 0) == 0) {
            std::string loc_id = field_name.substr(PROPERTY_LOCATION_PREFIX.size());
            auto location = std::make_shared<CacheLocation>();
            if (field_value.empty()) {
                location->set_id(loc_id);
            } else if (!location->FromJsonString(field_value)) {
                return EC_CORRUPTION;
            }
            out_locations[loc_id] = std::move(location);
        } else {
            out_properties[field_name] = field_value;
        }
    }
    return EC_OK;
}

ErrorCode DeserializeLocations(const FieldMap &field_map, CacheLocationMap &out_locations) {
    for (const auto &[field_name, field_value] : field_map) {
        if (field_name.rfind(PROPERTY_LOCATION_PREFIX, 0) != 0) {
            continue;
        }
        std::string loc_id = field_name.substr(PROPERTY_LOCATION_PREFIX.size());
        auto location = std::make_shared<CacheLocation>();
        if (field_value.empty()) {
            location->set_id(loc_id);
        } else if (!location->FromJsonString(field_value)) {
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
