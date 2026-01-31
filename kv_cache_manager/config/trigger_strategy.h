#pragma once

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {

class TriggerStrategy : public Jsonizable {
public:
    TriggerStrategy() = default;
    TriggerStrategy(int64_t used_size, double used_percentage)
        : used_size_(used_size), used_percentage_(used_percentage) {}

    ~TriggerStrategy() override;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    // Getters
    int64_t used_size() const { return used_size_; }
    double used_percentage() const { return used_percentage_; }

    // Setters
    void set_used_size(int64_t used_size) { used_size_ = used_size; }
    void set_used_percentage(double used_percentage) { used_percentage_ = used_percentage; }

private:
    int64_t used_size_;
    double used_percentage_;
};

} // namespace kv_cache_manager