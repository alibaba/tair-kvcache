#pragma once

#include "kv_cache_manager/meta/meta_storage_backend.h"

namespace kv_cache_manager {

// Intermediate base class for cache-type backends (e.g. MetaLocalBackend).
// Extends MetaStorageBackend with conditional write operations that accept
// previous_error_codes, allowing callers to skip keys that already failed
// in a prior write stage (e.g. persistent write) without copying key/value vectors.
class MetaCacheBaseBackend : public MetaStorageBackend {
public:
    ~MetaCacheBaseBackend() override = default;

    // Bring base-class write overloads into scope (without previous_error_codes).
    using MetaStorageBackend::Delete;
    using MetaStorageBackend::DeleteLocations;
    using MetaStorageBackend::Put;
    using MetaStorageBackend::Upsert;

    virtual size_t GetMemUsage() const noexcept = 0;
    virtual int64_t GetOldestAccessTime() const noexcept = 0;

    // 写入 locations + properties，仅当 key 在 cache 中不存在时才写入。
    // 若 key 已存在，返回 EC_OK 且不修改已有数据（幂等）。
    // @param request_context    请求上下文；可为 nullptr
    // @param keys               待写入的 key 列表
    // @param locations          每个 key 的 CacheLocationMap
    // @param properties         每个 key 的 PropertyMap
    // @return 每个 key 的错误码：
    //   - EC_OK:    写入成功或 key 已存在（均视为成功）
    //   - EC_ERROR: 写入失败
    virtual std::vector<ErrorCode> PutIfAbsent(RequestContext *request_context,
                                               const KeyTypeVec &keys,
                                               const CacheLocationMapVector &locations,
                                               const PropertyMapVector &properties) noexcept = 0;

    // =====================================================================
    // Overloaded writes with previous_error_codes
    // =====================================================================
    // For each key:
    //   - If previous_error_codes[i] != EC_OK → skip, return previous_error_codes[i]
    //   - Otherwise → perform the write normally

    virtual std::vector<ErrorCode> Put(RequestContext *request_context,
                                       const KeyTypeVec &keys,
                                       const CacheLocationMapVector &locations,
                                       const PropertyMapVector &properties,
                                       const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;

    virtual std::vector<ErrorCode> PutIfAbsent(RequestContext *request_context,
                                               const KeyTypeVec &keys,
                                               const CacheLocationMapVector &locations,
                                               const PropertyMapVector &properties,
                                               const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;

    virtual std::vector<ErrorCode> Upsert(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          const CacheLocationMapVector &locations,
                                          const PropertyMapVector &properties,
                                          const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;

    virtual std::vector<ErrorCode> Delete(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;

    virtual std::vector<ErrorCode> DeleteLocations(RequestContext *request_context,
                                                   const KeyTypeVec &keys,
                                                   const LocationIdsPerKey &location_ids,
                                                   const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;
};

} // namespace kv_cache_manager
