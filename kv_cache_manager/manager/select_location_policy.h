#pragma once

#include <array>
#include <map>
#include <string>

#include "cache_location.h"

namespace kv_cache_manager {

class SelectLocationPolicy {
public:
    // for match : select best location
    virtual CacheLocation *SelectForMatch(CacheLocationMap &location_map) const = 0;
    // for write : return true if exists means that not need write again
    virtual bool ExistsForWrite(const CacheLocationMap &location_map) const = 0;
    virtual ~SelectLocationPolicy() = default;
};

class WeightSLPolicy : public SelectLocationPolicy {
public:
    CacheLocation *SelectForMatch(CacheLocationMap &location_map) const override;
    bool ExistsForWrite(const CacheLocationMap &location_map) const override;

protected:
    virtual uint32_t GetWeight(CacheLocationMap::const_reference kv) const = 0;
};

/*
StaticWeightSLPolicy : 作为默认策略, 不解析uri, 不同类型存储后端权重固定
DynamicWeightSLPoliy : 不解析uri, 如果有某个存储后端挂掉了, 支持动态传入weight
NamedStorageWeightedSLPolicy : 解析uri, 如果有相同类型的多个存储后端(比如多个3fs),
                               支持根据解析得到的unique_name做选择
*/

class StaticWeightSLPolicy : public WeightSLPolicy {
public:
    using WeightArray = std::array<uint32_t, 5>;

protected:
    uint32_t GetWeight(CacheLocationMap::const_reference kv) const override;
    struct StorageTypeWeights {
        static constexpr uint32_t NFS = 5;          // NFS存储权重较高
        static constexpr uint32_t MOONCAKE = 3;     // Mooncake存储权重中等
        static constexpr uint32_t THREEFS = 3;      // 3FS存储权重较低
        static constexpr uint32_t TAIR_MEMPOOL = 3; // Tair存储权重最低
        static constexpr uint32_t DEFAULT = 1;      // 默认权重
    };

protected:
    // 直接用一个简单的映射表来选权重(映射表顺序与DataStorageType一样）
    // 代替switch-case
    inline static WeightArray default_storage_weights_{StorageTypeWeights::DEFAULT,
                                                       StorageTypeWeights::THREEFS,
                                                       StorageTypeWeights::MOONCAKE,
                                                       StorageTypeWeights::TAIR_MEMPOOL,
                                                       StorageTypeWeights::NFS};

    WeightArray &storage_weights_ = default_storage_weights_;
};

class DynamicWeightSLPoliy : public StaticWeightSLPolicy {
public:
    explicit DynamicWeightSLPoliy(WeightArray weight_array) : weight_array_(weight_array) {
        storage_weights_ = weight_array_;
    }

private:
    WeightArray weight_array_;
};

class NamedStorageWeightedSLPolicy : public WeightSLPolicy {
public:
    // support transparent find
    using WeightMap = std::map<std::string, uint32_t, std::less<>>;

    explicit NamedStorageWeightedSLPolicy(WeightMap &&weight_map) : weight_map_(std::move(weight_map)) {}

private:
    uint32_t GetWeight(CacheLocationMap::const_reference kv) const override;
    std::string_view ExtractHostName(std::string_view uri) const;

private:
    WeightMap weight_map_;
};

} // namespace kv_cache_manager