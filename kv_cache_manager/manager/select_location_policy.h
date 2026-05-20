#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <string>

#include "kv_cache_manager/meta/cache_location.h"

namespace kv_cache_manager {

enum class LocCheckResult {
    EXIST,                   // 数据存在且可达，正常参与选择
    TEMPORARILY_UNREACHABLE, // 数据可能仍在但暂时不可达，过滤但不删除
    NOT_EXIST                // 数据不存在或确认不可恢复，过滤且删除
};

using CheckLocDataExistFunc = std::function<LocCheckResult(const CacheLocation &loc)>;

class SelectLocationPolicy {
public:
    // for match : select best location
    virtual CacheLocationConstPtr SelectForMatch(CacheLocationMap &location_map,
                                                 CheckLocDataExistFunc check_loc_data_exist,
                                                 std::vector<std::string> &out_prune_loc_ids) const = 0;

    // for write : return true if exists means that not need write again
    virtual bool ExistsForWrite(const CacheLocationMap &location_map,
                                CheckLocDataExistFunc check_loc_data_exist,
                                std::vector<std::string> &out_prune_loc_ids) const = 0;
    // for write : spec-group aware version, returns true only if some serving
    // location already covers all requested_spec_names
    virtual bool ExistsForWrite(const CacheLocationMap &location_map,
                                const std::vector<std::string> &requested_spec_names,
                                CheckLocDataExistFunc check_loc_data_exist,
                                std::vector<std::string> &out_prune_loc_ids) const = 0;

    // Returns true if candidate belongs to the same data storage
    // as reference (same type AND same hostname).
    bool IsSameDataStorage(const CacheLocation &candidate, const CacheLocation &reference) const;

    virtual ~SelectLocationPolicy() = default;
};

class WeightSLPolicy : public SelectLocationPolicy {
public:
    CacheLocationConstPtr SelectForMatch(CacheLocationMap &location_map,
                                         CheckLocDataExistFunc check_loc_data_exist,
                                         std::vector<std::string> &out_prune_loc_ids) const override;
    bool ExistsForWrite(const CacheLocationMap &location_map,
                        CheckLocDataExistFunc check_loc_data_exist,
                        std::vector<std::string> &out_prune_loc_ids) const override;
    bool ExistsForWrite(const CacheLocationMap &location_map,
                        const std::vector<std::string> &requested_spec_names,
                        CheckLocDataExistFunc check_loc_data_exist,
                        std::vector<std::string> &out_prune_loc_ids) const override;

    // Returns true when the number of effective replicas reaches min_count.
    // A location counts as a replica iff:
    //   - status != CLS_NOT_FOUND
    //   - GetWeight(kv) > 0
    //   - check_loc_data_exist (when supplied) returns EXIST for it
    // Used by V6D eviction (V8 §2.6) to require >= 2 replicas before skipping
    // the write. The optional check_loc_data_exist hook lets the caller
    // exclude stale replicas (e.g. V6D nodes that just timed out heartbeat).
    // Locations returning NOT_EXIST are collected into out_prune_loc_ids so
    // the caller can schedule asynchronous meta cleanup.
    bool ExistsForWriteWithMinCount(const CacheLocationMap &location_map,
                                    int32_t min_count,
                                    CheckLocDataExistFunc check_loc_data_exist,
                                    std::vector<std::string> &out_prune_loc_ids) const;
    bool ExistsForWriteWithMinCount(const CacheLocationMap &location_map,
                                    int32_t min_count,
                                    CheckLocDataExistFunc check_loc_data_exist = nullptr) const;

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
    using WeightArray = std::array<uint32_t, static_cast<std::size_t>(DataStorageType::COUNT)>;

protected:
    uint32_t GetWeight(CacheLocationMap::const_reference kv) const override;
    struct StorageTypeWeights {
        static constexpr uint32_t NFS = 5;
        static constexpr uint32_t MOONCAKE = 3;
        static constexpr uint32_t THREEFS = 3;
        static constexpr uint32_t TAIR_MEMPOOL = 3;
        static constexpr uint32_t DEFAULT = 1;
        static constexpr uint32_t VCNS_HF3FS = THREEFS;
        static constexpr uint32_t VINEYARD = 10;
    };

protected:
    // Array positions correspond to DataStorageType enum values:
    // [0] UNKNOWN, [1] HF3FS, [2] MOONCAKE, [3] TAIR_MEMPOOL,
    // [4] NFS, [5] VCNS_HF3FS, [6] DUMMY, [7] VINEYARD
    inline static WeightArray default_storage_weights_{
        StorageTypeWeights::DEFAULT,      // [0] DATA_STORAGE_TYPE_UNKNOWN
        StorageTypeWeights::THREEFS,      // [1] DATA_STORAGE_TYPE_HF3FS
        StorageTypeWeights::MOONCAKE,     // [2] DATA_STORAGE_TYPE_MOONCAKE
        StorageTypeWeights::TAIR_MEMPOOL, // [3] DATA_STORAGE_TYPE_TAIR_MEMPOOL
        StorageTypeWeights::NFS,          // [4] DATA_STORAGE_TYPE_NFS
        StorageTypeWeights::VCNS_HF3FS,   // [5] DATA_STORAGE_TYPE_VCNS_HF3FS
        StorageTypeWeights::DEFAULT,      // [6] DATA_STORAGE_TYPE_DUMMY
        StorageTypeWeights::VINEYARD,     // [7] DATA_STORAGE_TYPE_VINEYARD
    };

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

private:
    WeightMap weight_map_;
};

} // namespace kv_cache_manager
