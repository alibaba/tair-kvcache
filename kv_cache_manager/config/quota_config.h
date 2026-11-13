#pragma once

#include <string>
#include <vector>

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
    const std::string &mode() const { return mode_; }
    double ratio() const { return ratio_; }
    const std::vector<std::string> &storage_names() const { return storage_names_; }
    bool automatic() const { return mode_ == "auto"; }
    void set_mode(const std::string &mode) { mode_ = mode; }
    void set_ratio(double ratio) { ratio_ = ratio; }
    void set_storage_names(const std::vector<std::string> &names) { storage_names_ = names; }

private:
    int64_t capacity_ = 0;
    DataStorageType storage_type_ = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
    std::string mode_ = "fixed";
    double ratio_ = 1.0;
    std::vector<std::string> storage_names_;
};

} // namespace kv_cache_manager