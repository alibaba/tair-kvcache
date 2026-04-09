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

        // parse block data
        KeyType key;
        if (!StringUtil::StrToInt64(k.c_str(), key)) {
            KVCM_LOG_ERROR("parse key to int64_t failed, path: [%s], raw key string: [%s]", path_.c_str(), k.c_str());
            return ErrorCode::EC_ERROR;
        }
        table_.Emplace(key, std::move(parsed_v));
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

    // block data
    table_.ForEachKV([&](const KeyType &key, const FieldMap &field_map) {
        persist_table[std::to_string(key)] = Jsonizable::ToJsonString(field_map);
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

std::vector<ErrorCode> MetaDummyBackend::Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    if (keys.size() != field_maps.size()) {
        KVCM_LOG_ERROR("put failed, keys.size: [%zu] != field_maps.size: [%zu]", keys.size(), field_maps.size());
        return std::vector<ErrorCode>(keys.size(), ErrorCode::EC_BADARGS);
    }

    std::vector<ErrorCode> ec_vec;
    for (std::size_t i = 0; i != keys.size(); ++i) {
        ec_vec.emplace_back(PutForOneKey(keys[i], field_maps[i]));
        if (ec_vec.back() != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("put failed, key: [%" PRIi64 "], error code: [%" PRIi32 "]",
                          keys[i],
                          static_cast<std::int32_t>(ec_vec.back()));
        }
    }
    return ec_vec;
}

ErrorCode MetaDummyBackend::PutForOneKey(const KeyType &key, const FieldMap &field_map) {
    std::lock_guard<std::mutex> guard(mutex_);
    table_.Upsert(key, field_map);
    return PersistToPath();
}

std::vector<ErrorCode> MetaDummyBackend::UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    if (keys.size() != field_maps.size()) {
        KVCM_LOG_ERROR("update fields failed, keys.size[%zu] != field_maps.size[%zu]", keys.size(), field_maps.size());
        return std::vector<ErrorCode>(keys.size(), ErrorCode::EC_BADARGS);
    }
    std::vector<ErrorCode> ec_vec;
    for (std::size_t i = 0; i != keys.size(); ++i) {
        ec_vec.emplace_back(UpdateFieldsForOneKey(keys[i], field_maps[i]));
        if (ec_vec.back() != ErrorCode::EC_OK && ec_vec.back() != ErrorCode::EC_NOENT) {
            KVCM_LOG_WARN("update fields failed, key: [%" PRIi64 "], ec:[%" PRIi32 "]",
                          keys[i],
                          static_cast<std::int32_t>(ec_vec.back()));
        }
    }
    return ec_vec;
}

ErrorCode MetaDummyBackend::UpdateFieldsForOneKey(const KeyType &key, const FieldMap &field_map) {
    std::lock_guard<std::mutex> guard(mutex_);
    const bool found = table_.FindAndModify(key, [&](FieldMap &existing_map) {
        for (const auto &[field_name, field_value] : field_map) {
            existing_map[field_name] = field_value;
        }
    });
    if (!found) {
        KVCM_LOG_WARN("update fields failed due to key cannot be found, key: [%" PRIi64 "]", key);
        return ErrorCode::EC_NOENT;
    }
    return PersistToPath();
}

std::vector<ErrorCode> MetaDummyBackend::Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    if (keys.size() != field_maps.size()) {
        KVCM_LOG_ERROR("upsert failed, keys.size: [%zu] != field_maps.size: [%zu]", keys.size(), field_maps.size());
        return std::vector<ErrorCode>(keys.size(), ErrorCode::EC_BADARGS);
    }

    std::vector<ErrorCode> ec_vec;
    for (std::size_t i = 0; i != keys.size(); ++i) {
        ec_vec.emplace_back(UpsertForOneKey(keys[i], field_maps[i]));
        if (ec_vec.back() != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("upsert failed, key: [%" PRIi64 "], error code: [%" PRIi32 "]", keys[i], ec_vec.back());
        }
    }
    return ec_vec;
}

ErrorCode MetaDummyBackend::UpsertForOneKey(const KeyType &key, const FieldMap &field_map) {
    std::lock_guard<std::mutex> guard(mutex_);
    const bool found = table_.FindAndModify(key, [&](FieldMap &existing_map) {
        for (const auto &[field_name, field_value] : field_map) {
            existing_map[field_name] = field_value;
        }
    });
    if (!found) {
        table_.Upsert(key, field_map);
    }
    return PersistToPath();
}

std::vector<ErrorCode> MetaDummyBackend::IncrFields(const KeyTypeVec &keys,
                                                    const std::map<std::string, std::int64_t> &field_amounts) noexcept {
    std::vector<ErrorCode> ec_vec;
    for (const KeyType &key : keys) {
        ec_vec.emplace_back(IncrFieldsForOneKey(key, field_amounts));
        if (ec_vec.back() != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("incr fields failed, key: [%" PRIi64 "], error code: [%" PRIi32 "]",
                          key,
                          static_cast<std::int32_t>(ec_vec.back()));
        }
    }
    return ec_vec;
}

ErrorCode MetaDummyBackend::IncrFieldsForOneKey(const KeyType &key,
                                                const std::map<std::string, std::int64_t> &field_amounts) {
    std::lock_guard<std::mutex> guard(mutex_);
    ErrorCode ec = ErrorCode::EC_OK;
    const bool found = table_.FindAndModify(key, [&](FieldMap &field_map) {
        std::map<std::string, std::string> new_field_map;
        for (const auto &[field_name, amount] : field_amounts) {
            const auto field_iter = field_map.find(field_name);
            if (field_iter == field_map.end()) {
                KVCM_LOG_ERROR("incr fields failed due to field cannot be found, field: [%s], key: [%" PRIi64 "]",
                               field_name.c_str(),
                               key);
                ec = ErrorCode::EC_BADARGS;
                return;
            }
            const auto &old_field_value = field_iter->second;
            std::int64_t old_field_value_num = 0;
            if (!StringUtil::StrToInt64(old_field_value.c_str(), old_field_value_num)) {
                KVCM_LOG_ERROR("incr fields failed due to value cannot be interpreted to int64_t, "
                               "field: [%s], value: [%s], key: [%" PRIi64 "]",
                               field_name.c_str(),
                               old_field_value.c_str(),
                               key);
                ec = ErrorCode::EC_BADARGS;
                return;
            }
            new_field_map[field_name] = std::to_string(old_field_value_num + amount);
        }
        for (const auto &[field_name, new_field_value] : new_field_map) {
            field_map[field_name] = new_field_value;
        }
    });
    if (!found) {
        KVCM_LOG_WARN("incr fields failed due to key cannot be found, key: [%" PRIi64 "]", key);
        return ErrorCode::EC_NOENT;
    }
    if (ec != ErrorCode::EC_OK) {
        return ec;
    }
    return PersistToPath();
}

std::vector<ErrorCode> MetaDummyBackend::Delete(const KeyTypeVec &keys) noexcept {
    std::vector<ErrorCode> ec_vec;
    for (const KeyType &key : keys) {
        ec_vec.emplace_back(DeleteForOneKey(key));
        if (ec_vec.back() != ErrorCode::EC_OK && ec_vec.back() != ErrorCode::EC_NOENT) {
            KVCM_LOG_WARN("delete failed, key: [%" PRIi64 "], error code: [%" PRIi32 "]",
                          key,
                          static_cast<std::int32_t>(ec_vec.back()));
        }
    }
    return ec_vec;
}

ErrorCode MetaDummyBackend::DeleteForOneKey(const KeyType &key) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!table_.Contains(key)) {
        return ErrorCode::EC_NOENT;
    }
    table_.Erase(key);
    return PersistToPath();
}

std::vector<ErrorCode> MetaDummyBackend::Get(const KeyTypeVec &keys,
                                             const std::vector<std::string> &field_names,
                                             FieldMapVec &out_field_maps) noexcept {
    std::vector<ErrorCode> ec_vec;
    out_field_maps = FieldMapVec(keys.size());
    for (std::size_t i = 0; i != keys.size(); ++i) {
        ec_vec.emplace_back(GetForOneKey(keys[i], field_names, out_field_maps[i]));
        if (ec_vec.back() != ErrorCode::EC_OK && ec_vec.back() != ErrorCode::EC_NOENT) {
            KVCM_LOG_WARN("get failed, key: [%" PRIi64 "], error code: [%" PRIi32 "]",
                          keys[i],
                          static_cast<std::int32_t>(ec_vec.back()));
        }
    }
    return ec_vec;
}

ErrorCode MetaDummyBackend::GetForOneKey(const KeyType &key,
                                         const std::vector<std::string> &field_names,
                                         FieldMap &out_field_map) const {
    out_field_map.clear();
    const bool found = table_.FindAndApply(key, [&](const FieldMap &field_table) {
        for (const auto &field_name : field_names) {
            const auto field_iter = field_table.find(field_name);
            out_field_map[field_name] = (field_iter == field_table.end() ? "" : field_iter->second);
        }
    });
    if (!found) {
        for (const auto &field_name : field_names) {
            out_field_map[field_name] = "";
        }
    }
    return ErrorCode::EC_OK;
}

std::vector<ErrorCode> MetaDummyBackend::GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept {
    std::vector<ErrorCode> ec_vec;
    out_field_maps = FieldMapVec(keys.size());
    for (std::size_t i = 0; i != keys.size(); ++i) {
        ec_vec.emplace_back(GetAllFieldsForOneKey(keys[i], out_field_maps[i]));
        if (ec_vec.back() != ErrorCode::EC_OK && ec_vec.back() != ErrorCode::EC_NOENT) {
            KVCM_LOG_WARN("get all fields failed, key: [%" PRIi64 "], error code: [%" PRIi32 "]",
                          keys[i],
                          static_cast<std::int32_t>(ec_vec.back()));
        }
    }
    return ec_vec;
}

ErrorCode MetaDummyBackend::GetAllFieldsForOneKey(const KeyType &key, FieldMap &out_field_map) const {
    out_field_map.clear();
    if (!table_.Get(key, out_field_map)) {
        return ErrorCode::EC_NOENT;
    }
    return out_field_map.empty() ? ErrorCode::EC_NOENT : ErrorCode::EC_OK;
}

std::vector<ErrorCode> MetaDummyBackend::Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept {
    out_is_exist_vec.clear();
    out_is_exist_vec.reserve(keys.size());
    std::vector<ErrorCode> ec_vec;
    for (std::size_t i = 0; i != keys.size(); ++i) {
        bool is_exist = false;
        ec_vec.emplace_back(ExistsForOneKey(keys[i], is_exist));
        if (ec_vec.back() != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("get all fields failed, key: [%" PRIi64 "], error code: [%" PRIi32 "]",
                          keys[i],
                          static_cast<std::int32_t>(ec_vec.back()));
        }
        out_is_exist_vec.emplace_back(is_exist);
    }
    return ec_vec;
}

ErrorCode MetaDummyBackend::ExistsForOneKey(const KeyType &key, bool &out_is_exist) const {
    out_is_exist = (table_.Count(key) > 0);
    return ErrorCode::EC_OK;
}

ErrorCode MetaDummyBackend::ListKeys(const std::string &cursor,
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
    table_.ForEachKV([&](const KeyType &key, const FieldMap &) {
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

ErrorCode MetaDummyBackend::RandomSample(const std::int64_t count, KeyTypeVec &out_keys) noexcept {
    out_keys.clear();
    table_.ForEachKV([&](const KeyType &key, const FieldMap &map) {
        if (out_keys.size() >= count) {
            return false;
        }
        out_keys.emplace_back(key);
        return true;
    });
    return ErrorCode::EC_OK;
}

ErrorCode MetaDummyBackend::SampleReclaimKeys(const std::int64_t count, KeyTypeVec &out_keys) noexcept {
    return RandomSample(count, out_keys);
}

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
