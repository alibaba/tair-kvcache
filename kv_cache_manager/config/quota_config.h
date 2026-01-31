#pragma once

#include <string>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/data_storage/common_define.h"

namespace kv_cache_manager {

class QuotaConfig : public Jsonizable {
public:
    QuotaConfig() = default;
    QuotaConfig(int64_t capacity, DataStorageType storage_type) : capacity_(capacity), storage_type_(storage_type) {}

    ~QuotaConfig() override;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const;
    // Getters
    int64_t capacity() const { return capacity_; }
    DataStorageType storage_spec() const { return storage_type_; }

    // Setters
    void set_capacity(int64_t capacity) { capacity_ = capacity; }
    void set_storage_type(DataStorageType storage_type) { storage_type_ = storage_type; }

private:
    int64_t capacity_;
    DataStorageType storage_type_;
};

} // namespace kv_cache_manager