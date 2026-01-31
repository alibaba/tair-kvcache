#pragma once
#include <string>

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {

class MetaCachePolicyConfig : public Jsonizable {
public:
    static constexpr const char *kDefaultCacheType = "lru";
    static constexpr size_t kDefaultCacheCapacity = 1024; // MB
    static constexpr int32_t kDefaultCacheShardBits = 6;
    static constexpr double kDefaultHighPriPoolRatio = 0.0;

public:
    MetaCachePolicyConfig()
        : capacity_(kDefaultCacheCapacity)
        , type_(kDefaultCacheType)
        , cache_shard_bits_(kDefaultCacheShardBits)
        , high_pri_pool_ratio_(kDefaultHighPriPoolRatio) {}

    MetaCachePolicyConfig(size_t capacity,
                          const std::string &type,
                          int32_t cache_shard_bits,
                          double high_pri_pool_ratio)
        : capacity_(capacity)
        , type_(type)
        , cache_shard_bits_(cache_shard_bits)
        , high_pri_pool_ratio_(high_pri_pool_ratio) {}

    ~MetaCachePolicyConfig() override;
    const std::string &GetCachePolicyType() const { return type_; }
    const size_t &GetCapacity() const { return capacity_; }
    const int32_t &GetCacheShardBits() const { return cache_shard_bits_; }
    const double &GetHighPriPoolRatio() const { return high_pri_pool_ratio_; }

    void SetCapacity(size_t capacity) { capacity_ = capacity; }
    void SetType(const std::string &type) { type_ = type; }
    void SetCacheShardBits(int32_t cache_shard_bits) { cache_shard_bits_ = cache_shard_bits; }
    void SetHighPriPoolRatio(double high_pri_pool_ratio) { high_pri_pool_ratio_ = high_pri_pool_ratio; }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "capacity", capacity_, kDefaultCacheCapacity);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "type", type_, std::string(kDefaultCacheType));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "cache_shard_bits", cache_shard_bits_, kDefaultCacheShardBits);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "high_pri_pool_ratio", high_pri_pool_ratio_, kDefaultHighPriPoolRatio);
        return true;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "type", type_);
        Put(writer, "capacity", capacity_);
        Put(writer, "cache_shard_bits", cache_shard_bits_);
        Put(writer, "high_pri_pool_ratio", high_pri_pool_ratio_);
    }
    bool ValidateRequiredFields(std::string &invalid_fields) const {
        bool valid = true;
        std::string local_invalid_fields;
        if (type_.empty()) {
            valid = false;
            local_invalid_fields += "{type}";
        }
        if (!valid) {
            invalid_fields += "{MetaCachePolicyConfig: " + local_invalid_fields + "}";
        }
        return valid;
    }

private:
    size_t capacity_;
    std::string type_;
    int32_t cache_shard_bits_;
    double high_pri_pool_ratio_;
};

} // namespace kv_cache_manager
