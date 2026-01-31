#pragma once

#include <memory>
#include <string>
#include <variant>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/optimizer/config/types.h"
namespace kv_cache_manager {
struct LruParams : public Jsonizable {
    double sample_rate = 0.0;
    bool FromRapidValue(const rapidjson::Value &v) override {
        KVCM_JSON_GET_MACRO(v, "sample_rate", sample_rate);
        return true;
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "sample_rate", sample_rate);
    }
};
struct RandomLruParams : public Jsonizable {
    double sample_rate = 0.0;
    bool FromRapidValue(const rapidjson::Value &v) override {
        KVCM_JSON_GET_MACRO(v, "sample_rate", sample_rate);
        return true;
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "sample_rate", sample_rate);
    }
};

using EvictionPolicyParam = std::variant<LruParams, RandomLruParams>;

class EvictionConfig : public Jsonizable {
public:
    EvictionConfig() = default;
    ~EvictionConfig() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] EvictionMode eviction_mode() const { return eviction_mode_; }
    [[nodiscard]] int32_t eviction_batch_size_per_instance() const { return eviction_batch_size_per_instance_; }

    void set_eviction_mode(EvictionMode mode) { eviction_mode_ = mode; }
    void set_eviction_batch_size_per_instance(int32_t size) { eviction_batch_size_per_instance_ = size; }

private:
    EvictionMode eviction_mode_ = EvictionMode::EVICTION_MODE_UNSPECIFIED;
    int32_t eviction_batch_size_per_instance_ = 100;
};
} // namespace kv_cache_manager