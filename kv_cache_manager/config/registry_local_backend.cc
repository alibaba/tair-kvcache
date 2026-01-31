#include "kv_cache_manager/config/registry_local_backend.h"

#include <filesystem>
#include <fstream>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

RegistryLocalBackend::~RegistryLocalBackend() {}

ErrorCode RegistryLocalBackend::Init(const StandardUri &standard_uri) noexcept {
    table_.Clear();
    path_ = standard_uri.GetPath();

    if (!path_.empty()) {
        enable_persistence_ = true;
        ErrorCode ec = Recover();
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("fail to recover table, file path: %s", path_.c_str());
            return ec;
        }
    }
    return EC_OK;
}

ErrorCode RegistryLocalBackend::Recover() {
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
        std::map<std::string, std::string> tmp_field_table;
        if (!Jsonizable::FromJsonString(pair.second, tmp_field_table)) {
            KVCM_LOG_ERROR("fail to parse field map json, file[%s] content[%s]", path_.c_str(), pair.second.c_str());
            return EC_ERROR;
        }
        table_.Emplace(pair.first, std::move(tmp_field_table));
    }
    return EC_OK;
}

ErrorCode RegistryLocalBackend::PersistToPath() {
    if (!enable_persistence_) {
        return EC_OK;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    std::map<std::string, std::string> tmp_table;
    table_.ForEachKV([&](const std::string &key, const std::map<std::string, std::string> &field_map) {
        tmp_table[key] = Jsonizable::ToJsonString(field_map);
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

ErrorCode RegistryLocalBackend::Load(const std::string &key, std::map<std::string, std::string> &out_value) noexcept {
    auto iter = table_.Find(key);
    if (iter == table_.End()) {
        return EC_NOENT;
    }
    out_value = iter->second;
    return EC_OK;
}

ErrorCode RegistryLocalBackend::Save(const std::string &key, const std::map<std::string, std::string> &value) noexcept {
    table_[key] = value;
    return PersistToPath();
}

ErrorCode RegistryLocalBackend::Delete(const std::string &key) noexcept {
    auto iter = table_.Find(key);
    if (iter == table_.End()) {
        return EC_NOENT;
    }
    table_.Erase(iter);
    return PersistToPath();
}

} // namespace kv_cache_manager
