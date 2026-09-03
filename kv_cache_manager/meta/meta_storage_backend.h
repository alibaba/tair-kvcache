#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {
class MetaStorageBackendConfig;
class RevisitIntervalHistogram;
class RequestContext;

// MetaStorageBackend — MetaIndexer 后端存储抽象基类
//
// 数据模型：
//   每个 key (int64_t) 对应一组 CacheLocation（按 location_id 索引）和
//   一组 property（string→string KV 对）。
//   后端自行决定内部存储格式：Local 后端直接存结构体，Redis 后端存 JSON string。

class MetaStorageBackend {
public:
    using KeyType = ::kv_cache_manager::KeyType;
    using KeyTypeVec = ::kv_cache_manager::KeyTypeVec;
    using FieldMap = ::kv_cache_manager::FieldMap;
    using FieldMapVec = ::kv_cache_manager::FieldMapVec;

    virtual ~MetaStorageBackend() = default;

    virtual std::string GetStorageType() noexcept = 0;

    // 初始化后端。必须在 Open() 之前调用。
    // @param instance_id  实例标识，用于 key 前缀隔离
    // @param config       后端配置（含 storage_uri 等）
    // @return EC_OK 成功；EC_BADARGS 参数非法；EC_ERROR 内部错误
    virtual ErrorCode Init(const std::string &instance_id,
                           const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept = 0;

    // 打开后端连接/资源。Init 成功后调用。
    // @return EC_OK 成功；EC_ERROR 连接/资源获取失败
    virtual ErrorCode Open() noexcept = 0;

    // 关闭后端，释放连接和资源。
    // @return EC_OK 成功；EC_ERROR 关闭时出错
    virtual ErrorCode Close() noexcept = 0;

    // =====================================================================
    // Write APIs
    // =====================================================================

    // 整 key 覆盖写入 locations + properties。若 key 已存在则全量替换
    // @param request_context  请求上下文，用于 metrics 上报；可为 nullptr
    // @param keys             待写入的 key 列表
    // @param locations        每个 key 对应的 CacheLocationMap，大小必须等于 keys.size()；
    //                         可以为空 map 表示该 key 无 location
    // @param properties       每个 key 对应的 PropertyMap，大小必须等于 keys.size()；
    //                         可以为空 map 表示该 key 无 property
    // @return 每个 key 的错误码：
    //   - EC_OK:    写入成功
    //   - EC_ERROR: 写入失败（网络/IO 错误等）
    virtual std::vector<ErrorCode> Put(RequestContext *request_context,
                                       const KeyTypeVec &keys,
                                       const CacheLocationMapVector &locations,
                                       const PropertyMapVector &properties) noexcept = 0;

    // 若 key 存在则合并 不存在则创建 key 并写入。
    // @param request_context  请求上下文；可为 nullptr
    // @param keys             待操作的 key 列表
    // @param locations        每个 key 要合并的 CacheLocationMap，大小必须等于 keys.size()；
    //                         空 map 表示不更新该 key 的 location
    // @param properties       每个 key 要合并的 PropertyMap，大小必须等于 keys.size()；
    //                         空 map 表示不更新该 key 的 property
    // @return 每个 key 的错误码：
    //   - EC_OK:    操作成功（无论是新建还是合并）
    //   - EC_ERROR: 操作失败
    virtual std::vector<ErrorCode> Upsert(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          const CacheLocationMapVector &locations,
                                          const PropertyMapVector &properties) noexcept = 0;

    // Intent-aware counterpart used by shared RMW code. Backends without
    // process-local access bookkeeping can safely reuse ordinary Upsert.
    virtual std::vector<ErrorCode> Upsert(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          const CacheLocationMapVector &locations,
                                          const PropertyMapVector &properties,
                                          MetaAccessIntent /*intent*/) noexcept {
        return Upsert(request_context, keys, locations, properties);
    }

    // Allocation-light one-location upsert used by pure-local targeted RMW.
    // The default adapter preserves backend semantics; local memory overrides
    // it to avoid constructing one temporary unordered_map per key.
    virtual std::vector<ErrorCode> UpsertSingleLocations(RequestContext *request_context,
                                                         const KeyTypeVec &keys,
                                                         const LocationIdRefVector &location_ids,
                                                         const CacheLocationVector &locations) noexcept {
        if (keys.size() != location_ids.size() || keys.size() != locations.size()) {
            return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
        }
        CacheLocationMapVector location_maps(keys.size());
        PropertyMapVector properties(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            if (location_ids[i] == nullptr) {
                return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
            }
            location_maps[i].emplace(*location_ids[i], locations[i]);
        }
        return Upsert(request_context, keys, location_maps, properties);
    }

    // 删除整个 key 及其所有 locations 和 properties。
    // @param request_context  请求上下文；可为 nullptr
    // @param keys  待删除的 key 列表
    // @return 每个 key 的错误码：
    //   - EC_OK:    删除成功（key 确实存在并被删除）
    //   - EC_NOENT: key 不存在
    //   - EC_ERROR: 删除失败
    virtual std::vector<ErrorCode> Delete(RequestContext *request_context, const KeyTypeVec &keys) noexcept = 0;

    // 删除指定 key 下的指定 location。幂等语义：删除不存在的 location 视为成功。
    // 不影响 key 中的 properties 和其他 location。
    // @param request_context  请求上下文；可为 nullptr
    // @param keys             待操作的 key 列表
    // @param location_ids     每个 key 要删除的 location id 列表，大小必须等于 keys.size()；
    //                         若 location_ids[i] 为空，该 key 为 no-op（返回 EC_OK）
    // @return 每个 key 的错误码：
    //   - EC_OK:    删除成功（含 no-op 和 location 原本不存在的情况）
    //   - EC_NOENT: key 本身不存在
    //   - EC_ERROR: 操作失败
    virtual std::vector<ErrorCode> DeleteLocations(RequestContext *request_context,
                                                   const KeyTypeVec &keys,
                                                   const LocationIdsPerKey &location_ids) noexcept = 0;

    // =====================================================================
    // Read APIs
    // =====================================================================

    // 读取 key 的完整元数据（locations + properties 一起返回）。
    // @param request_context   请求上下文；可为 nullptr
    // @param keys              待查询的 key 列表
    // @param out_locations     [out] 每个 key 的 CacheLocationMap，大小等于 keys.size()
    // @param out_properties    [out] 每个 key 的 PropertyMap，大小等于 keys.size()
    // @return 每个 key 的错误码：
    //   - EC_OK:        key 存在
    //   - EC_NOENT:     key 不存在
    //   - EC_CORRUPTION: location 数据损坏（反序列化失败）
    //   - EC_ERROR:     查询失败
    virtual std::vector<ErrorCode> Get(RequestContext *request_context,
                                       const KeyTypeVec &keys,
                                       CacheLocationMapVector &out_locations,
                                       PropertyMapVector &out_properties) noexcept = 0;

    // 读取 key 的全量 CacheLocationMap（所有 location）。
    // @param request_context   请求上下文；可为 nullptr
    // @param keys              待查询的 key 列表
    // @param out_locations     [out] 每个 key 的 CacheLocationMap，大小等于 keys.size()。
    //                          key 存在但无 location 时为空 map。
    // @return 每个 key 的错误码：
    //   - EC_OK:        key 存在（out_locations[i] 可能为空 map）
    //   - EC_NOENT:     key 不存在
    //   - EC_CORRUPTION: location 数据损坏（反序列化失败，仅 Redis 后端）
    //   - EC_ERROR:     查询失败
    virtual std::vector<ErrorCode> GetLocations(RequestContext *request_context,
                                                const KeyTypeVec &keys,
                                                CacheLocationMapVector &out_locations) noexcept = 0;

    // Lightweight all-location read for consumers that only need immutable
    // CacheLocation values and do not need to look them up again by id. The
    // default implementation preserves every backend's existing behavior by
    // flattening GetLocations. Local memory overrides it to avoid cloning an
    // unordered_map (including every location-id string and hash node) per key.
    virtual std::vector<ErrorCode> GetLocationValues(RequestContext *request_context,
                                                     const KeyTypeVec &keys,
                                                     LocationsPerKey &out_locations) noexcept {
        CacheLocationMapVector location_maps;
        auto results = GetLocations(request_context, keys, location_maps);
        out_locations.clear();
        out_locations.resize(keys.size());
        const std::size_t count = std::min(keys.size(), location_maps.size());
        for (std::size_t i = 0; i < count; ++i) {
            auto &values = out_locations[i];
            values.reserve(location_maps[i].size());
            for (const auto &[location_id, location] : location_maps[i]) {
                (void)location_id;
                values.push_back(location);
            }
        }
        return results;
    }

    // Range-based compact variant used by very large prefix queries. The
    // generic fallback preserves backend behavior; the local backend overrides
    // it so callers avoid both copying the key slice and allocating one vector
    // per key.
    virtual std::vector<ErrorCode> GetLocationValuesCompact(RequestContext *request_context,
                                                            const KeyType *keys,
                                                            size_t key_count,
                                                            CompactLocationsPerKey &out_locations) noexcept {
        if (key_count != 0 && keys == nullptr) {
            out_locations.Clear(key_count);
            for (size_t i = 0; i < key_count; ++i) {
                out_locations.FinishKey();
            }
            return std::vector<ErrorCode>(key_count, EC_BADARGS);
        }
        KeyTypeVec key_vector;
        if (key_count != 0) {
            key_vector.assign(keys, keys + key_count);
        }
        LocationsPerKey locations;
        auto results = GetLocationValues(request_context, key_vector, locations);
        out_locations.Clear(key_count);
        const size_t value_count = std::min(key_count, locations.size());
        for (size_t i = 0; i < key_count; ++i) {
            if (i < value_count) {
                out_locations.values.insert(out_locations.values.end(), locations[i].begin(), locations[i].end());
            }
            out_locations.FinishKey();
        }
        return results;
    }

    // 读取指定 location id 对应的 CacheLocation。
    // @param request_context  请求上下文；可为 nullptr
    // @param keys             待查询的 key 列表
    // @param location_ids     每个 key 要查询的 location id 列表，大小必须等于 keys.size()
    // @param out_locations    [out] 每个 key 的 CacheLocation 列表，out_locations[i] 大小
    //                         等于 location_ids[i].size()。未找到的 location 为默认构造。
    // @return 二维错误码，results[i][j] 对应 keys[i] 的 location_ids[i][j]：
    //   - EC_OK:         location 存在
    //   - EC_NOENT:      key 不存在或该 location 不存在
    //   - EC_CORRUPTION: location 数据损坏（反序列化失败）
    //   - EC_ERROR:      查询失败
    virtual std::vector<std::vector<ErrorCode>> GetLocations(RequestContext *request_context,
                                                             const KeyTypeVec &keys,
                                                             const LocationIdsPerKey &location_ids,
                                                             LocationsPerKey &out_locations) noexcept = 0;

    // Read selected locations and preserve the key-level existence result.
    // A targeted read alone cannot distinguish a missing key from an existing
    // key that does not contain any requested location. RMW callers need that
    // distinction to update key_count correctly when an upsert creates a new
    // block. Backends that can determine both states in one lookup should
    // override this method. The generic fallback only probes key existence for
    // ambiguous all-NOENT rows, so existing-location reads remain one request.
    virtual std::vector<std::vector<ErrorCode>>
    GetLocationsWithKeyStatus(RequestContext *request_context,
                              const KeyTypeVec &keys,
                              const LocationIdsPerKey &location_ids,
                              LocationsPerKey &out_locations,
                              std::vector<ErrorCode> &out_key_error_codes) noexcept {
        auto results = GetLocations(request_context, keys, location_ids, out_locations);
        out_key_error_codes.assign(keys.size(), EC_MISMATCH);
        if (results.size() != keys.size() || out_locations.size() != keys.size()) {
            return results;
        }

        KeyTypeVec ambiguous_keys;
        std::vector<size_t> ambiguous_indices;
        for (size_t i = 0; i < keys.size(); ++i) {
            if (results[i].size() != location_ids[i].size() || out_locations[i].size() != location_ids[i].size()) {
                continue;
            }
            bool found_location = false;
            ErrorCode hard_error = EC_OK;
            for (const ErrorCode ec : results[i]) {
                if (ec == EC_OK) {
                    found_location = true;
                    break;
                }
                if (ec != EC_NOENT && hard_error == EC_OK) {
                    hard_error = ec;
                }
            }
            if (found_location) {
                out_key_error_codes[i] = EC_OK;
            } else if (hard_error != EC_OK) {
                out_key_error_codes[i] = hard_error;
            } else {
                ambiguous_keys.push_back(keys[i]);
                ambiguous_indices.push_back(i);
            }
        }

        if (ambiguous_keys.empty()) {
            return results;
        }
        std::vector<bool> exists;
        const auto exists_results = Exists(request_context, ambiguous_keys, exists);
        if (exists_results.size() != ambiguous_keys.size() || exists.size() != ambiguous_keys.size()) {
            return results;
        }
        for (size_t i = 0; i < ambiguous_keys.size(); ++i) {
            const size_t original_index = ambiguous_indices[i];
            out_key_error_codes[original_index] =
                exists_results[i] == EC_OK ? (exists[i] ? EC_OK : EC_NOENT) : exists_results[i];
        }
        return results;
    }

    // Flat one-location form of GetLocationsWithKeyStatus. The default
    // adapter is deliberately generic; local memory overrides it so the
    // common ReportEvent shape allocates O(1) vectors per batch instead of
    // two tiny vectors per key.
    virtual std::vector<ErrorCode>
    GetSingleLocationsWithKeyStatus(RequestContext *request_context,
                                    const KeyTypeVec &keys,
                                    const LocationIdRefVector &location_ids,
                                    CacheLocationVector &out_locations,
                                    std::vector<ErrorCode> &out_key_error_codes) noexcept {
        out_locations.assign(keys.size(), CacheLocationConstPtr{});
        out_key_error_codes.assign(keys.size(), EC_BADARGS);
        if (keys.size() != location_ids.size()) {
            return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
        }
        LocationIdsPerKey ids_per_key(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            if (location_ids[i] == nullptr) {
                return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
            }
            ids_per_key[i].push_back(*location_ids[i]);
        }
        LocationsPerKey locations_per_key;
        auto nested_results =
            GetLocationsWithKeyStatus(request_context, keys, ids_per_key, locations_per_key, out_key_error_codes);
        std::vector<ErrorCode> results(keys.size(), EC_MISMATCH);
        if (nested_results.size() != keys.size() || locations_per_key.size() != keys.size() ||
            out_key_error_codes.size() != keys.size()) {
            return results;
        }
        for (size_t i = 0; i < keys.size(); ++i) {
            if (nested_results[i].size() == 1 && locations_per_key[i].size() == 1) {
                results[i] = nested_results[i][0];
                out_locations[i] = std::move(locations_per_key[i][0]);
            }
        }
        return results;
    }

    // Targeted maintenance read. It must not update access/LRU/revisit state
    // and must not populate another cache tier. Backends without an online
    // cache can use the ordinary targeted read implementation.
    virtual std::vector<std::vector<ErrorCode>> GetLocationsForMaintenance(RequestContext *request_context,
                                                                           const KeyTypeVec &keys,
                                                                           const LocationIdsPerKey &location_ids,
                                                                           LocationsPerKey &out_locations) noexcept {
        return GetLocations(request_context, keys, location_ids, out_locations);
    }

    // 仅获取 key 的 location id 列表（不读取 location body）。
    // @param request_context    请求上下文；可为 nullptr
    // @param keys               待查询的 key 列表
    // @param out_location_ids   [out] 每个 key 的 location id 列表，大小等于 keys.size()。
    //                           key 存在但无 location 时为空 vector。
    // @return 每个 key 的错误码：
    //   - EC_OK:    key 存在
    //   - EC_NOENT: key 不存在
    //   - EC_ERROR: 查询失败
    virtual std::vector<ErrorCode> GetLocationIds(RequestContext *request_context,
                                                  const KeyTypeVec &keys,
                                                  LocationIdsPerKey &out_location_ids) noexcept = 0;

    // Maintenance counterpart of GetLocationIds(). It must not promote or
    // touch cache entries. Persistent-only backends can use the ordinary
    // implementation because they have no online LRU state to preserve.
    virtual std::vector<ErrorCode> GetLocationIdsForMaintenance(RequestContext *request_context,
                                                                const KeyTypeVec &keys,
                                                                LocationIdsPerKey &out_location_ids) noexcept {
        return GetLocationIds(request_context, keys, out_location_ids);
    }

    // 读取指定字段名的 properties。
    // @param request_context  请求上下文；可为 nullptr
    // @param keys             待查询的 key 列表
    // @param field_names      要读取的 property 字段名列表（所有 key 使用同一组字段名）
    // @param out_properties   [out] 每个 key 的 PropertyMap（仅包含请求的字段），
    //                         大小等于 keys.size()。字段不存在时不出现在 map 中。
    // @return 每个 key 的错误码：
    //   - EC_OK:    key 存在
    //   - EC_NOENT: key 不存在
    //   - EC_ERROR: 查询失败
    virtual std::vector<ErrorCode> GetProperties(RequestContext *request_context,
                                                 const KeyTypeVec &keys,
                                                 const std::vector<std::string> &field_names,
                                                 PropertyMapVector &out_properties) noexcept = 0;

    virtual std::vector<ErrorCode>
    GetPropertiesForMaintenance(RequestContext *request_context,
                                const KeyTypeVec &keys,
                                const std::vector<std::string> &field_names,
                                PropertyMapVector &out_properties) noexcept {
        return GetProperties(request_context, keys, field_names, out_properties);
    }

    virtual MaintenanceReadResult GetForMaintenance(RequestContext *request_context,
                                                     const KeyTypeVec &keys,
                                                     const std::vector<std::string> &field_names,
                                                     CacheLocationMapVector &out_locations,
                                                     PropertyMapVector &out_properties) noexcept {
        MaintenanceReadResult result;
        result.location_error_codes = GetLocations(request_context, keys, out_locations);
        result.property_error_codes =
            GetPropertiesForMaintenance(request_context, keys, field_names, out_properties);
        return result;
    }

    // 检查 key 是否存在。
    // @param request_context   请求上下文；可为 nullptr
    // @param keys              待查询的 key 列表
    // @param out_is_exist_vec  [out] 每个 key 是否存在，大小等于 keys.size()
    // @return 每个 key 的错误码：
    //   - EC_OK:    查询成功（out_is_exist_vec[i] 反映结果）
    //   - EC_ERROR: 查询失败
    virtual std::vector<ErrorCode>
    Exists(RequestContext *request_context, const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept = 0;

    // 判断 key 是否存在至少一个有效 location。
    // @param request_context  请求上下文；可为 nullptr
    // @param keys             待查询的 key 列表
    // @param out_exists       [out] 每个 key 是否拥有至少一个 location，大小等于 keys.size()
    // @return 每个 key 的错误码：
    //   - EC_OK:    查询成功（out_exists[i] 反映结果）
    //   - EC_NOENT: key 不存在（out_exists[i] 为 false）
    //   - EC_ERROR: 查询失败
    virtual std::vector<ErrorCode>
    ExistsLocation(RequestContext *request_context, const KeyTypeVec &keys, std::vector<bool> &out_exists) noexcept = 0;

    // 基于 cursor 分页扫描所有 key。
    // @param request_context 请求上下文；可为 nullptr
    // @param cursor          游标（首次传 SCAN_BASE_CURSOR = "0"）
    // @param limit           本次最多返回的 key 数量（hint，实际可能多于或少于此值）
    // @param out_next_cursor [out] 下一次扫描的 cursor；等于 SCAN_BASE_CURSOR 时表示扫描结束
    // @param out_keys        [out] 本次扫描到的 key 列表
    // @return EC_OK 成功；EC_BADARGS cursor 非法；EC_ERROR 扫描失败
    virtual ErrorCode ListKeys(RequestContext *request_context,
                               const std::string &cursor,
                               const int64_t limit,
                               std::string &out_next_cursor,
                               KeyTypeVec &out_keys) noexcept = 0;

    // Scan keys and their full CacheLocationMap for background maintenance.
    // Unlike the online ListKeys/GetLocations path, this operation must not
    // update access/LRU/revisit state or populate another cache tier.
    //
    // `limit` is a batch-size hint, matching ListKeys/Redis SCAN semantics;
    // a backend may return more or fewer keys while advancing its cursor.
    //
    // `out` is only valid when EC_OK is returned. On success, out.keys,
    // out.locations and out.location_results must have identical sizes.
    virtual ErrorCode ScanLocationsForMaintenance(RequestContext *request_context,
                                                  const std::string &cursor,
                                                  int64_t limit,
                                                  MaintenanceScanBatch &out) noexcept = 0;

    // Maintenance delete counterpart of GetLocationsForMaintenance(). It
    // must not promote/touch cache entries. The caller already performed the
    // expected-value comparison under the MetaIndexer shard lock.
    virtual std::vector<ErrorCode> DeleteLocationsForMaintenance(RequestContext *request_context,
                                                                 const KeyTypeVec &keys,
                                                                 const LocationIdsPerKey &location_ids) noexcept {
        return DeleteLocations(request_context, keys, location_ids);
    }

    // 随机采样 key。
    // @param request_context 请求上下文；可为 nullptr
    // @param count    期望采样数量
    // @param out_keys [out] 采样结果（实际数量可能小于 count）
    // @return EC_OK 成功；EC_ERROR 采样失败
    virtual ErrorCode
    RandomSample(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept = 0;

    // 采样适合回收的 key（通常按 LRU 或 access time 排序）。
    // @param request_context 请求上下文；可为 nullptr
    // @param count    期望采样数量
    // @param out_keys [out] 采样结果（实际数量可能小于 count）
    // @return EC_OK 成功；EC_ERROR 采样失败
    virtual ErrorCode
    SampleReclaimKeys(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept = 0;

    // =====================================================================
    // Metadata APIs — 用于持久化 MetaIndexer 自身的元信息（key_count、storage_usage 等）
    // =====================================================================

    // 写入/更新 MetaIndexer 元数据。
    // @param field_maps  要写入的元数据 field→value
    // @return EC_OK 成功；EC_ERROR 写入失败
    virtual ErrorCode PutMetaData(const FieldMap &field_maps) noexcept = 0;

    // 读取 MetaIndexer 元数据。
    // @param field_maps [out] 读到的元数据
    // @return EC_OK 成功；EC_NOENT 无元数据；EC_ERROR 读取失败
    virtual ErrorCode GetMetaData(FieldMap &field_maps) noexcept = 0;

    // Synchronously flush pending writes for the given keys.
    // Returns true if all writes persisted successfully, false on failure/timeout.
    // Default: no-op (sync backends have no pending writes).
    virtual bool Sync(const KeyTypeVec & /*keys*/) noexcept { return true; }

    struct AsyncWriteStats {
        int64_t max_async_queue_size = 0;
        int64_t avg_async_queue_size = 0;
        int64_t flush_key_count = 0;
        int64_t batch_flush_time_us = 0;
        int64_t pipeline_error_count = 0;
    };
    virtual AsyncWriteStats GetAsyncWriteStats() noexcept { return {}; }

    // =====================================================================
    // Metrics APIs — 用于配置后端的指标收集功能
    // =====================================================================

    // 设置重访间隔直方图（用于统计缓存命中时间间隔分布）。
    // 默认实现为空操作（no-op），只有需要统计重访间隔的后端才需要重写。
    // @param histogram 直方图对象，用于记录重访间隔数据
    virtual void SetRevisitHistogram(std::shared_ptr<RevisitIntervalHistogram> /*histogram*/) {}
};
} // namespace kv_cache_manager
