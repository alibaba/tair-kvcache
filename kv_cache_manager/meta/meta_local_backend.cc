#include "kv_cache_manager/meta/meta_local_backend.h"

#include <filesystem>
#include <fstream>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"

namespace kv_cache_manager {

std::string MetaLocalBackend::GetStorageType() noexcept { return META_LOCAL_BACKEND_TYPE_STR; }

ErrorCode MetaLocalBackend::Init(const std::string &instance_id,
                                 const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept {
    if (instance_id.empty()) {
        KVCM_LOG_ERROR("init fail, instance id is empty");
        return EC_BADARGS;
    }
    if (!config) {
        KVCM_LOG_ERROR("init fail, config is nullptr");
        return EC_BADARGS;
    }
    std::string storage_uri_str = config->GetStorageUri();
    if (storage_uri_str.empty()) {
        enable_persistence_ = false;
    } else {
        StandardUri storage_uri = StandardUri::FromUri(storage_uri_str);
        enable_persistence_ = true;
        path_ = storage_uri.GetPath();
    }
    return EC_OK;
}

ErrorCode MetaLocalBackend::Open() noexcept {
    table_.Clear();
    if (!enable_persistence_) {
        return EC_OK;
    }

    std::lock_guard<std::mutex> guard(mutex_);
    std::error_code ec;
    bool exists = std::filesystem::exists(path_, ec);
    if (ec) {
        KVCM_LOG_ERROR("file exists error[%s] file[%s]", ec.message().c_str(), path_.c_str());
        return EC_IO_ERROR;
    }
    if (!exists) {
        return EC_OK;
    }

    std::ifstream ifs(path_);
    if (!ifs.is_open()) {
        KVCM_LOG_ERROR("fail to open file[%s]", path_.c_str());
        return EC_IO_ERROR;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    std::map<std::string, std::string> tmp_table;
    if (!Jsonizable::FromJsonString(content, tmp_table)) {
        KVCM_LOG_ERROR("fail to parse full json from file[%s], content[%s]", path_.c_str(), content.c_str());
        return EC_ERROR;
    }
    for (auto &pair : tmp_table) {
        FieldMap tmp_field_table;
        if (!Jsonizable::FromJsonString(pair.second, tmp_field_table)) {
            KVCM_LOG_ERROR("fail to parse field map json, file[%s] content[%s]", path_.c_str(), pair.second.c_str());
            return EC_ERROR;
        }
        KeyType key;
        if (!StringUtil::StrToInt64(pair.first.c_str(), key)) {
            KVCM_LOG_ERROR("fail to parse key, file[%s] content[%s]", path_.c_str(), pair.first.c_str());
            return EC_ERROR;
        }
        table_.Emplace(key, std::move(tmp_field_table));
    }
    return EC_OK;
}

ErrorCode MetaLocalBackend::Close() noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    return EC_OK;
}

ErrorCode MetaLocalBackend::PersistToPath() {
    if (!enable_persistence_) {
        return EC_OK;
    }
    std::map<std::string, std::string> tmp_table;
    table_.ForEachKV([&](const KeyType &key, const FieldMap &field_map) {
        tmp_table[std::to_string(key)] = Jsonizable::ToJsonString(field_map);
        return true;
    });
    std::string json_content = Jsonizable::ToJsonString(tmp_table);
    std::ofstream ofs(path_);
    if (!ofs.is_open()) {
        KVCM_LOG_ERROR("Cannot open file for write: %s", path_.c_str());
        return EC_IO_ERROR;
    }
    ofs << json_content;
    return EC_OK;
}

std::vector<ErrorCode> MetaLocalBackend::Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    if (keys.size() != field_maps.size()) {
        KVCM_LOG_ERROR("put fail, keys size[%lu] != field_maps size[%lu]", keys.size(), field_maps.size());
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    std::vector<ErrorCode> ec_vec;
    for (int32_t i = 0; i < keys.size(); ++i) {
        ec_vec.emplace_back(PutForOneKey(keys[i], field_maps[i]));
        if (ec_vec.back() != EC_OK) {
            KVCM_LOG_WARN("put fail, key[%ld] ec[%d]", keys[i], ec_vec.back());
        }
    }
    return ec_vec;
}

ErrorCode MetaLocalBackend::PutForOneKey(const KeyType &key, const FieldMap &field_map) {
    // PersistToPath will traverse all keys, we should lock the mutex when multi-threads put/update/delete one key
    std::lock_guard<std::mutex> guard(mutex_);
    table_[key] = field_map;
    return PersistToPath();
}

std::vector<ErrorCode> MetaLocalBackend::UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    if (keys.size() != field_maps.size()) {
        KVCM_LOG_ERROR("update fields fail, keys.size[%lu] != field_maps.size[%lu]", keys.size(), field_maps.size());
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    std::vector<ErrorCode> ec_vec;
    for (int32_t i = 0; i < keys.size(); ++i) {
        ec_vec.emplace_back(UpdateFieldsForOneKey(keys[i], field_maps[i]));
        if (ec_vec.back() != EC_OK && ec_vec.back() != EC_NOENT) {
            KVCM_LOG_WARN("update fields fail, key[%ld] ec[%d]", keys[i], ec_vec.back());
        }
    }
    return ec_vec;
}

ErrorCode MetaLocalBackend::UpdateFieldsForOneKey(const KeyType &key, const FieldMap &field_map) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto iter = table_.Find(key);
    if (iter == table_.End()) {
        KVCM_LOG_WARN("update fields fail, cannot find key[%ld]", key);
        return EC_NOENT;
    }
    for (const auto &[field_name, field_value] : field_map) {
        (iter->second)[field_name] = field_value;
    }
    return PersistToPath();
}

std::vector<ErrorCode> MetaLocalBackend::Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    if (keys.size() != field_maps.size()) {
        KVCM_LOG_ERROR("upsert fail, keys size[%lu] != field_maps size[%lu]", keys.size(), field_maps.size());
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    std::vector<ErrorCode> ec_vec;
    for (int32_t i = 0; i < keys.size(); ++i) {
        ec_vec.emplace_back(UpsertForOneKey(keys[i], field_maps[i]));
        if (ec_vec.back() != EC_OK) {
            KVCM_LOG_WARN("upsert fail, key[%ld] ec[%d]", keys[i], ec_vec.back());
        }
    }
    return ec_vec;
}

ErrorCode MetaLocalBackend::UpsertForOneKey(const KeyType &key, const FieldMap &field_map) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto iter = table_.Find(key);
    if (iter == table_.End()) {
        // if not exist, then insert
        table_[key] = field_map;
        return PersistToPath();
    }
    // if exist, then update
    for (const auto &[field_name, field_value] : field_map) {
        (iter->second)[field_name] = field_value;
    }
    return PersistToPath();
}

std::vector<ErrorCode> MetaLocalBackend::IncrFields(const KeyTypeVec &keys,
                                                    const std::map<std::string, int64_t> &field_amounts) noexcept {
    std::vector<ErrorCode> ec_vec;
    for (const KeyType &key : keys) {
        ec_vec.emplace_back(IncrFieldsForOneKey(key, field_amounts));
        if (ec_vec.back() != EC_OK) {
            KVCM_LOG_WARN("incr fields fail, key[%ld] ec[%d]", key, ec_vec.back());
        }
    }
    return ec_vec;
}

ErrorCode MetaLocalBackend::IncrFieldsForOneKey(const KeyType &key,
                                                const std::map<std::string, int64_t> &field_amounts) {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto iter = table_.Find(key);
    if (iter == table_.End()) {
        KVCM_LOG_WARN("incr fields fail, cannot find key[%ld]", key);
        return EC_NOENT;
    }

    auto &field_map = iter->second;
    std::map<std::string, std::string> new_field_map;
    for (const auto &[field_name, amount] : field_amounts) {
        const auto field_iter = field_map.find(field_name);
        if (field_iter == field_map.end()) {
            KVCM_LOG_ERROR("incr fields fail, cannot find field[%s] for key[%ld]", field_name.c_str(), key);
            return EC_BADARGS;
        }
        const auto &old_field_value = field_iter->second;
        int64_t old_field_value_num = 0;
        if (!StringUtil::StrToInt64(old_field_value.c_str(), old_field_value_num)) {
            KVCM_LOG_ERROR("incr fields fail, cannot convert field[%s] value[%s] to int64_t for key[%ld]",
                           field_name.c_str(),
                           old_field_value.c_str(),
                           key);
            return EC_BADARGS;
        }
        new_field_map[field_name] = std::to_string(old_field_value_num + amount);
    }
    for (const auto &[field_name, new_field_value] : new_field_map) {
        field_map[field_name] = new_field_value;
    }
    return PersistToPath();
}

std::vector<ErrorCode> MetaLocalBackend::Delete(const KeyTypeVec &keys) noexcept {
    std::vector<ErrorCode> ec_vec;
    for (const KeyType &key : keys) {
        ec_vec.emplace_back(DeleteForOneKey(key));
        if (ec_vec.back() != EC_OK && ec_vec.back() != EC_NOENT) {
            KVCM_LOG_WARN("delete fail, key[%ld] ec[%d]", key, ec_vec.back());
        }
    }
    return ec_vec;
}

ErrorCode MetaLocalBackend::DeleteForOneKey(const KeyType &key) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!table_.Contains(key)) {
        return EC_NOENT;
    }
    table_.Erase(key);
    return PersistToPath();
}

std::vector<ErrorCode> MetaLocalBackend::Get(const KeyTypeVec &keys,
                                             const std::vector<std::string> &field_names,
                                             FieldMapVec &out_field_maps) noexcept {
    std::vector<ErrorCode> ec_vec;
    out_field_maps = FieldMapVec(keys.size());
    for (int32_t i = 0; i < keys.size(); ++i) {
        ec_vec.emplace_back(GetForOneKey(keys[i], field_names, out_field_maps[i]));
        if (ec_vec.back() != EC_OK && ec_vec.back() != EC_NOENT) {
            KVCM_LOG_WARN("get fail, key[%ld] ec[%d]", keys[i], ec_vec.back());
        }
    }
    return ec_vec;
}

ErrorCode MetaLocalBackend::GetForOneKey(const KeyType &key,
                                         const std::vector<std::string> &field_names,
                                         FieldMap &out_field_map) {
    out_field_map.clear();
    const auto iter = table_.Find(key);
    if (iter == table_.End()) {
        for (const std::string &field_name : field_names) {
            out_field_map[field_name] = "";
        }
        return EC_OK;
    }
    const auto &field_table = iter->second;
    for (const std::string &field_name : field_names) {
        const auto field_iter = field_table.find(field_name);
        out_field_map[field_name] = (field_iter == field_table.end() ? "" : field_iter->second);
    }
    return EC_OK;
}

std::vector<ErrorCode> MetaLocalBackend::GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept {
    std::vector<ErrorCode> ec_vec;
    out_field_maps = FieldMapVec(keys.size());
    for (int32_t i = 0; i < keys.size(); ++i) {
        ec_vec.emplace_back(GetAllFieldsForOneKey(keys[i], out_field_maps[i]));
        if (ec_vec.back() != EC_OK && ec_vec.back() != EC_NOENT) {
            KVCM_LOG_WARN("get all fields fail, key[%ld] ec[%d]", keys[i], ec_vec.back());
        }
    }
    return ec_vec;
}

ErrorCode MetaLocalBackend::GetAllFieldsForOneKey(const KeyType &key, FieldMap &out_field_map) {
    out_field_map.clear();
    const auto iter = table_.Find(key);
    if (iter == table_.End()) {
        return EC_NOENT;
    }
    const auto &field_table = iter->second;
    out_field_map = field_table;
    return out_field_map.empty() ? EC_NOENT : EC_OK;
}

std::vector<ErrorCode> MetaLocalBackend::Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept {
    out_is_exist_vec.clear();
    out_is_exist_vec.reserve(keys.size());
    std::vector<ErrorCode> ec_vec;
    for (int32_t i = 0; i < keys.size(); ++i) {
        bool is_exist = false;
        ec_vec.emplace_back(ExistsForOneKey(keys[i], is_exist));
        if (ec_vec.back() != EC_OK) {
            KVCM_LOG_WARN("get all fields fail, key[%ld] ec[%d]", keys[i], ec_vec.back());
        }
        out_is_exist_vec.emplace_back(is_exist);
    }
    return ec_vec;
}

ErrorCode MetaLocalBackend::ExistsForOneKey(const KeyType &key, bool &out_is_exist) {
    out_is_exist = (table_.Count(key) > 0);
    return EC_OK;
}

ErrorCode MetaLocalBackend::ListKeys(const std::string &cursor,
                                     const int64_t limit,
                                     std::string &out_next_cursor,
                                     std::vector<KeyType> &out_keys) noexcept {
    out_next_cursor.clear();
    out_keys.clear();

    int64_t current_index = 0;
    auto iter = table_.Begin();
    if (cursor != SCAN_BASE_CURSOR) {
        if (!StringUtil::StrToInt64(cursor.c_str(), current_index)) {
            KVCM_LOG_ERROR("list keys fail, cannot convert cursor[%s] to start index", cursor.c_str());
            return EC_BADARGS;
        }
        std::advance(iter, current_index);
    }
    int64_t end_index = current_index + limit;
    while (iter != table_.End()) {
        if (current_index >= end_index) {
            out_next_cursor = std::to_string(current_index);
            return EC_OK;
        }
        const auto &key = iter->first;
        out_keys.emplace_back(key);
        ++iter;
        ++current_index;
    }
    out_next_cursor = SCAN_BASE_CURSOR;
    return EC_OK;
}

ErrorCode MetaLocalBackend::RandomSample(const int64_t count, std::vector<KeyType> &out_keys) noexcept {
    out_keys.clear();
    table_.ForEachKV([&](const KeyType &key, const FieldMap &map) {
        if (out_keys.size() >= count) {
            return false;
        }
        out_keys.emplace_back(key);
        return true;
    });
    return EC_OK;
}

ErrorCode MetaLocalBackend::PutMetaData(const FieldMap &field_map) noexcept { return EC_OK; }

ErrorCode MetaLocalBackend::GetMetaData(FieldMap &field_map) noexcept { return EC_NOENT; }

} // namespace kv_cache_manager
