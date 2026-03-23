#pragma once

#include <memory>
#include <string>

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {

class MetaStorageBackendConfig : public Jsonizable {
public:
    MetaStorageBackendConfig() = default;
    MetaStorageBackendConfig(const std::string &storage_type) : storage_type_(storage_type) {}

public:
    ~MetaStorageBackendConfig() override;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "storage_type", storage_type_);
        Put(writer, "cache_type", cache_type_);
        Put(writer, "persistent_type", persistent_type_);
        Put(writer, "storage_uri", storage_uri_);
        Put(writer, "mutex_shard_num", mutex_shard_num_);
    }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "storage_type", storage_type_, std::string("local"));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "cache_type", cache_type_, std::string("local"));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "persistent_type", persistent_type_, std::string("redis"));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "storage_uri", storage_uri_, std::string(""));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "mutex_shard_num", mutex_shard_num_, size_t(1024));
        return true;
    }
    bool ValidateRequiredFields(std::string &invalid_fields) const {
        bool valid = true;
        std::string local_invalid_fields;
        if (storage_type_.empty()) {
            valid = false;
            local_invalid_fields += "{storage_type}";
        }
        if (!valid) {
            invalid_fields += "{MetaStorageBackendConfig: " + local_invalid_fields + "}";
        }
        return valid;
    }
    const std::string &GetStorageType() const { return storage_type_; }
    const std::string &GetCacheType() const { return cache_type_; }
    const std::string &GetPersistentType() const { return persistent_type_; }
    const std::string &GetStorageUri() const { return storage_uri_; }
    const size_t GetMutexShardNum() const { return mutex_shard_num_; }

    void SetStorageType(const std::string &storage_type) { storage_type_ = storage_type; }
    void SetCacheType(const std::string &cache_type) { cache_type_ = cache_type; }
    void SetPersistentType(const std::string &persistent_type) { persistent_type_ = persistent_type; }
    void SetStorageUri(const std::string &storage_uri) { storage_uri_ = storage_uri; }
    void SetMutexShardNum(const size_t mutex_shard_num) { mutex_shard_num_ = mutex_shard_num; }

private:
    std::string storage_type_ = "local";
    std::string cache_type_ = "local";
    std::string persistent_type_ = "redis";
    std::string storage_uri_ = "";
    size_t mutex_shard_num_ = 1024;
};
} // namespace kv_cache_manager
