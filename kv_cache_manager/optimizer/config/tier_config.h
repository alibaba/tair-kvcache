#pragma once

#include <memory>
#include <string>
#include <variant>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/data_storage/storage_config.h"
namespace kv_cache_manager {
class OptTierConfig : public Jsonizable {
public:
    OptTierConfig() = default;
    ~OptTierConfig() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

public:
    [[nodiscard]] const std::string &unique_name() const { return unique_name_; }
    [[nodiscard]] DataStorageType storage_type() const { return storage_type_; }
    [[nodiscard]] int64_t band_width_mbps() const { return band_width_mbps_; }
    [[nodiscard]] size_t priority() const { return priority_; }
    [[nodiscard]] int64_t capacity() const { return capacity_; }

    void set_unique_name(const std::string &name) { unique_name_ = name; }
    void set_storage_type(DataStorageType type) { storage_type_ = type; }
    void set_band_width_mbps(int64_t band_width_mbps) { band_width_mbps_ = band_width_mbps; }
    void set_priority(size_t priority) { priority_ = priority; }
    void set_capacity(int64_t capacity) { capacity_ = capacity; }

private:
    std::string unique_name_;
    DataStorageType storage_type_ = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
    int64_t band_width_mbps_ = 0;
    size_t priority_ = 0;
    int64_t capacity_ = 0;
};
} // namespace kv_cache_manager