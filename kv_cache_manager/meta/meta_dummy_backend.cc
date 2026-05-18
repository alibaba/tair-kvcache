#include "kv_cache_manager/meta/meta_dummy_backend.h"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"

namespace kv_cache_manager {

// special key used in the persistence file to identify the metadata
// entry, non-numeric so it is guaranteed never to collide with any
// numeric cache key
static constexpr const char *kMetadataFileKey = "__metadata__";

std::string MetaDummyBackend::GetStorageType() noexcept { return META_DUMMY_BACKEND_TYPE_STR; }

ErrorCode MetaDummyBackend::Init(const std::string &instance_id,
                                 const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept {
    if (instance_id.empty()) {
        KVCM_LOG_ERROR("init fail, instance_id is empty");
        return ErrorCode::EC_BADARGS;
    }
    if (!config) {
        KVCM_LOG_ERROR("init fail, config is nullptr");
        return ErrorCode::EC_BADARGS;
    }
    if (const std::string storage_uri_str = config->GetStorageUri(); storage_uri_str.empty()) {
        enable_persistence_ = false;
    } else {
        const StandardUri storage_uri = StandardUri::FromUri(storage_uri_str);
        enable_persistence_ = true;
        path_ = storage_uri.GetPath() + "_" + instance_id;
    }
    return ErrorCode::EC_OK;
}

ErrorCode MetaDummyBackend::Open() noexcept {
    std::lock_guard<std::mutex> guard(mutex_);

    table_.Clear();
    metadata_.clear();
    if (!enable_persistence_) {
        return ErrorCode::EC_OK;
    }

    std::error_code ec;
    bool exists = std::filesystem::exists(path_, ec);
    if (ec) {
        KVCM_LOG_ERROR("std::filesystem::exists call failed, err code: [%d], err msg: [%s], path: [%s]",
                       ec.value(),
                       ec.message().c_str(),
                       path_.c_str());
        return ErrorCode::EC_IO_ERROR;
    }
    if (!exists) {
        return ErrorCode::EC_OK;
    }

    std::ifstream ifs(path_);
    if (!ifs.is_open()) {
        KVCM_LOG_ERROR("file open failed, path: [%s]", path_.c_str());
        return ErrorCode::EC_IO_ERROR;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    std::map<std::string, std::string> parsed;
    if (!Jsonizable::FromJsonString(content, parsed)) {
        KVCM_LOG_ERROR("parse top-level json failed, path: [%s], raw content: [%s]", path_.c_str(), content.c_str());
        return ErrorCode::EC_ERROR;
    }
    for (auto &[k, v] : parsed) {
        FieldMap parsed_v;
        if (!Jsonizable::FromJsonString(v, parsed_v)) {
            KVCM_LOG_ERROR("parse field_map json failed, path: [%s], raw content: [%s]", path_.c_str(), v.c_str());
            return ErrorCode::EC_ERROR;
        }

        // parse metadata
        if (k == kMetadataFileKey) {
            metadata_ = std::move(parsed_v);
            continue;
        }

        // parse block data: reconstruct DummyItem from serialized FieldMap
        KeyType key;
        if (!StringUtil::StrToInt64(k.c_str(), key)) {
            KVCM_LOG_ERROR("parse key to int64_t failed, path: [%s], raw key string: [%s]", path_.c_str(), k.c_str());
            return ErrorCode::EC_ERROR;
        }
        DummyItem item;
        for (auto &[field_name, field_value] : parsed_v) {
            if (field_name.rfind(LOCATION_PREFIX, 0) == 0) {
                std::string loc_id = field_name.substr(LOCATION_PREFIX.size());
                auto loc = std::make_shared<CacheLocation>();
                if (loc->FromJsonString(field_value)) {
                    item.locations[loc_id] = std::move(loc);
                }
            } else {
                item.properties[field_name] = std::move(field_value);
            }
        }
        table_.Emplace(key, std::move(item));
    }

    return ErrorCode::EC_OK;
}

ErrorCode MetaDummyBackend::Close() noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    return ErrorCode::EC_OK;
}

// should be used with protection of mutex
ErrorCode MetaDummyBackend::PersistToPath() {
    if (!enable_persistence_) {
        return ErrorCode::EC_OK;
    }

    std::map<std::string, std::string> persist_table;

    // block data: serialize each DummyItem as a FieldMap with locations serialized as JSON values
    table_.ForEachKV([&](const KeyType &key, const DummyItem &item) {
        FieldMap serialized;
        for (const auto &[loc_id, loc_ptr] : item.locations) {
            serialized[LOCATION_PREFIX + loc_id] = loc_ptr ? loc_ptr->ToJsonString() : "";
        }
        for (const auto &[prop_name, prop_value] : item.properties) {
            serialized[prop_name] = prop_value;
        }
        persist_table[std::to_string(key)] = Jsonizable::ToJsonString(serialized);
        return true;
    });

    // metadata
    if (!metadata_.empty()) {
        persist_table[kMetadataFileKey] = Jsonizable::ToJsonString(metadata_);
    }

    const std::string json_content = Jsonizable::ToJsonString(persist_table);
    std::ofstream ofs(path_);
    if (!ofs.is_open()) {
        KVCM_LOG_ERROR("cannot open file for write, path: [%s]", path_.c_str());
        return ErrorCode::EC_IO_ERROR;
    }
    ofs << json_content;
    return ErrorCode::EC_OK;
}

// ---------------------------------------------------------------------------
// Write operations
// ---------------------------------------------------------------------------

std::vector<ErrorCode> MetaDummyBackend::Put(RequestContext * /*request_context*/,
                                             const KeyTypeVec &keys,
                                             const CacheLocationMapVector &locations,
                                             const PropertyMapVector &properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    std::lock_guard<std::mutex> guard(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        DummyItem item;
        item.locations = locations[i];
        item.properties = properties[i];
        item.properties[PROPERTY_LRU_TIME] = std::to_string(TimestampUtil::GetCurrentTimeUs());
        table_.Upsert(keys[i], std::move(item));
    }
    PersistToPath();
    return results;
}

std::vector<ErrorCode> MetaDummyBackend::Upsert(RequestContext * /*request_context*/,
                                                const KeyTypeVec &keys,
                                                const CacheLocationMapVector &locations,
                                                const PropertyMapVector &properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    std::lock_guard<std::mutex> guard(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        const bool found = table_.FindAndModify(keys[i], [&](DummyItem &existing) {
            for (const auto &[loc_id, loc] : locations[i]) {
                existing.locations[loc_id] = loc;
            }
            for (const auto &[prop_name, prop_value] : properties[i]) {
                existing.properties[prop_name] = prop_value;
            }
            existing.properties[PROPERTY_LRU_TIME] = std::to_string(TimestampUtil::GetCurrentTimeUs());
        });
        if (!found) {
            DummyItem item;
            item.locations = locations[i];
            item.properties = properties[i];
            item.properties[PROPERTY_LRU_TIME] = std::to_string(TimestampUtil::GetCurrentTimeUs());
            table_.Upsert(keys[i], std::move(item));
        }
    }
    PersistToPath();
    return results;
}

std::vector<ErrorCode> MetaDummyBackend::Update(RequestContext * /*request_context*/,
                                                const KeyTypeVec &keys,
                                                const CacheLocationMapVector &locations,
                                                const PropertyMapVector &properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    std::lock_guard<std::mutex> guard(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        const bool found = table_.FindAndModify(keys[i], [&](DummyItem &existing) {
            for (const auto &[loc_id, loc] : locations[i]) {
                existing.locations[loc_id] = loc;
            }
            for (const auto &[prop_name, prop_value] : properties[i]) {
                existing.properties[prop_name] = prop_value;
            }
            existing.properties[PROPERTY_LRU_TIME] = std::to_string(TimestampUtil::GetCurrentTimeUs());
        });
        if (!found) {
            results[i] = EC_NOENT;
        }
    }
    PersistToPath();
    return results;
}

std::vector<ErrorCode> MetaDummyBackend::Delete(RequestContext * /*request_context*/, const KeyTypeVec &keys) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    std::lock_guard<std::mutex> guard(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (!table_.Contains(keys[i])) {
            results[i] = EC_NOENT;
        } else {
            table_.Erase(keys[i]);
        }
    }
    PersistToPath();
    return results;
}

std::vector<ErrorCode> MetaDummyBackend::DeleteLocations(RequestContext * /*request_context*/,
                                                         const KeyTypeVec &keys,
                                                         const LocationIdsPerKey &location_ids) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    std::lock_guard<std::mutex> guard(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (location_ids[i].empty()) {
            continue;
        }
        const bool found = table_.FindAndModify(keys[i], [&](DummyItem &existing) {
            for (const auto &loc_id : location_ids[i]) {
                existing.locations.erase(loc_id);
            }
        });
        if (!found) {
            results[i] = EC_NOENT;
        }
    }
    PersistToPath();
    return results;
}

// ---------------------------------------------------------------------------
// Read operations
// ---------------------------------------------------------------------------

std::vector<ErrorCode> MetaDummyBackend::Get(RequestContext * /*request_context*/,
                                             const KeyTypeVec &keys,
                                             CacheLocationMapVector &out_locations,
                                             PropertyMapVector &out_properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_locations.resize(keys.size());
    out_properties.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const bool found = table_.FindAndApply(keys[i], [&](const DummyItem &item) {
            out_locations[i] = item.locations;
            out_properties[i] = item.properties;
        });
        if (!found) {
            results[i] = EC_NOENT;
        } else {
            table_.FindAndModify(keys[i], [](DummyItem &item) {
                item.properties[PROPERTY_LRU_TIME] = std::to_string(TimestampUtil::GetCurrentTimeUs());
            });
        }
    }
    return results;
}

std::vector<ErrorCode> MetaDummyBackend::GetLocations(RequestContext * /*request_context*/,
                                                      const KeyTypeVec &keys,
                                                      CacheLocationMapVector &out_locations) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_locations.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const bool found =
            table_.FindAndApply(keys[i], [&](const DummyItem &item) { out_locations[i] = item.locations; });
        if (!found) {
            results[i] = EC_NOENT;
        } else {
            table_.FindAndModify(keys[i], [](DummyItem &item) {
                item.properties[PROPERTY_LRU_TIME] = std::to_string(TimestampUtil::GetCurrentTimeUs());
            });
        }
    }
    return results;
}

std::vector<std::vector<ErrorCode>> MetaDummyBackend::GetLocations(RequestContext * /*request_context*/,
                                                                   const KeyTypeVec &keys,
                                                                   const LocationIdsPerKey &location_ids,
                                                                   LocationsPerKey &out_locations) noexcept {
    std::vector<std::vector<ErrorCode>> results(keys.size());
    out_locations.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i].resize(location_ids[i].size(), EC_NOENT);
        out_locations[i].resize(location_ids[i].size());
        const bool found = table_.FindAndApply(keys[i], [&](const DummyItem &item) {
            for (size_t j = 0; j < location_ids[i].size(); ++j) {
                auto it = item.locations.find(location_ids[i][j]);
                if (it != item.locations.end()) {
                    out_locations[i][j] = it->second;
                    results[i][j] = EC_OK;
                }
            }
        });
        if (!found) {
            // All remain EC_NOENT (key not found).
        } else {
            table_.FindAndModify(keys[i], [](DummyItem &item) {
                item.properties[PROPERTY_LRU_TIME] = std::to_string(TimestampUtil::GetCurrentTimeUs());
            });
        }
    }
    return results;
}

std::vector<ErrorCode> MetaDummyBackend::GetLocationIds(RequestContext * /*request_context*/,
                                                        const KeyTypeVec &keys,
                                                        LocationIdsPerKey &out_location_ids) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_location_ids.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const bool found = table_.FindAndApply(keys[i], [&](const DummyItem &item) {
            for (const auto &[loc_id, loc] : item.locations) {
                out_location_ids[i].push_back(loc_id);
            }
        });
        if (!found) {
            results[i] = EC_NOENT;
        }
    }
    return results;
}

std::vector<ErrorCode> MetaDummyBackend::ExistsLocation(RequestContext * /*request_context*/,
                                                        const KeyTypeVec &keys,
                                                        std::vector<bool> &out_exists) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_exists.resize(keys.size(), false);
    for (size_t i = 0; i < keys.size(); ++i) {
        const bool found =
            table_.FindAndApply(keys[i], [&](const DummyItem &item) { out_exists[i] = !item.locations.empty(); });
        if (!found) {
            results[i] = EC_NOENT;
        }
    }
    return results;
}

std::vector<ErrorCode> MetaDummyBackend::GetProperties(RequestContext * /*request_context*/,
                                                       const KeyTypeVec &keys,
                                                       const std::vector<std::string> &field_names,
                                                       PropertyMapVector &out_properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_properties.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const bool found = table_.FindAndApply(keys[i], [&](const DummyItem &item) {
            for (const auto &field_name : field_names) {
                auto it = item.properties.find(field_name);
                if (it != item.properties.end()) {
                    out_properties[i][field_name] = it->second;
                }
            }
        });
        if (!found) {
            results[i] = EC_NOENT;
        } else {
            table_.FindAndModify(keys[i], [](DummyItem &item) {
                item.properties[PROPERTY_LRU_TIME] = std::to_string(TimestampUtil::GetCurrentTimeUs());
            });
        }
    }
    return results;
}

// ---------------------------------------------------------------------------
// Key-level operations
// ---------------------------------------------------------------------------

std::vector<ErrorCode> MetaDummyBackend::Exists(RequestContext * /*request_context*/,
                                                const KeyTypeVec &keys,
                                                std::vector<bool> &out_is_exist_vec) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_is_exist_vec.resize(keys.size(), false);
    for (size_t i = 0; i < keys.size(); ++i) {
        out_is_exist_vec[i] = table_.Contains(keys[i]);
        if (out_is_exist_vec[i]) {
            table_.FindAndModify(keys[i], [](DummyItem &item) {
                item.properties[PROPERTY_LRU_TIME] = std::to_string(TimestampUtil::GetCurrentTimeUs());
            });
        }
    }
    return results;
}

ErrorCode MetaDummyBackend::ListKeys(RequestContext * /*request_context*/,
                                     const std::string &cursor,
                                     const std::int64_t limit,
                                     std::string &out_next_cursor,
                                     KeyTypeVec &out_keys) noexcept {
    out_next_cursor.clear();
    out_keys.clear();

    std::int64_t start_index = 0;
    if (cursor != SCAN_BASE_CURSOR) {
        if (!StringUtil::StrToInt64(cursor.c_str(), start_index)) {
            KVCM_LOG_ERROR("list keys fail, cannot convert cursor[%s] to start index", cursor.c_str());
            return ErrorCode::EC_BADARGS;
        }
    }

    std::int64_t current_index = 0;
    const std::int64_t end_index = start_index + limit;
    bool reached_limit = false;
    table_.ForEachKV([&](const KeyType &key, const DummyItem &) {
        if (current_index >= end_index) {
            reached_limit = true;
            return false;
        }
        if (current_index >= start_index) {
            out_keys.emplace_back(key);
        }
        ++current_index;
        return true;
    });

    out_next_cursor = reached_limit ? std::to_string(current_index) : SCAN_BASE_CURSOR;
    return ErrorCode::EC_OK;
}

ErrorCode MetaDummyBackend::RandomSample(RequestContext * /*request_context*/,
                                         const std::int64_t count,
                                         KeyTypeVec &out_keys) noexcept {
    out_keys.clear();
    table_.ForEachKV([&](const KeyType &key, const DummyItem &) {
        if (static_cast<std::int64_t>(out_keys.size()) >= count) {
            return false;
        }
        out_keys.emplace_back(key);
        return true;
    });
    return ErrorCode::EC_OK;
}

ErrorCode MetaDummyBackend::SampleReclaimKeys(RequestContext *request_context,
                                              const std::int64_t count,
                                              KeyTypeVec &out_keys) noexcept {
    return RandomSample(request_context, count, out_keys);
}

// ---------------------------------------------------------------------------
// Metadata operations
// ---------------------------------------------------------------------------

ErrorCode MetaDummyBackend::PutMetaData(const FieldMap &field_map) noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    metadata_ = field_map;
    return PersistToPath();
}

ErrorCode MetaDummyBackend::GetMetaData(FieldMap &out_field_map) noexcept {
    out_field_map.clear();
    std::lock_guard<std::mutex> guard(mutex_);
    if (metadata_.empty()) {
        return ErrorCode::EC_NOENT;
    }
    out_field_map = metadata_;
    return ErrorCode::EC_OK;
}

} // namespace kv_cache_manager
