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
        Put(writer, "storage_uri", storage_uri_);
        Put(writer, "memory_primary", memory_primary_);
        Put(writer, "force_deleting_async_enqueue", force_deleting_async_enqueue_);
    }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "storage_type", storage_type_, std::string("local"));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "storage_uri", storage_uri_, std::string(""));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "memory_primary", memory_primary_, false);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "force_deleting_async_enqueue", force_deleting_async_enqueue_, true);
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
    const std::string &GetStorageUri() const { return storage_uri_; }
    bool GetMemoryPrimary() const { return memory_primary_; }
    bool GetForceDeletingAsyncEnqueue() const { return force_deleting_async_enqueue_; }

    void SetStorageType(const std::string &storage_type) { storage_type_ = storage_type; }
    void SetStorageUri(const std::string &storage_uri) { storage_uri_ = storage_uri; }
    void SetMemoryPrimary(bool memory_primary) { memory_primary_ = memory_primary; }
    void SetForceDeletingAsyncEnqueue(bool force_deleting_async_enqueue) {
        force_deleting_async_enqueue_ = force_deleting_async_enqueue;
    }

private:
    std::string storage_type_ = "local";
    std::string storage_uri_ = "";
    bool memory_primary_ = false;
    bool force_deleting_async_enqueue_ = true;
};
} // namespace kv_cache_manager
