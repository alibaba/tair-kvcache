#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {
class MetaStorageBackendConfig;

// MetaIndexer 后端存储抽象基类
class MetaStorageBackend {
public:
    // The storage primitives (KeyType / KeyTypeVec / FieldMap / FieldMapVec)
    // are now defined once in kv_cache_manager/meta/types.h. We re-export them
    // here for the existing call sites that reach for them via this class.
    using KeyType = ::kv_cache_manager::KeyType;
    using KeyTypeVec = ::kv_cache_manager::KeyTypeVec;
    using FieldMap = ::kv_cache_manager::FieldMap;
    using FieldMapVec = ::kv_cache_manager::FieldMapVec;

public:
    virtual ~MetaStorageBackend() = default;

    virtual std::string GetStorageType() noexcept = 0;

    virtual ErrorCode Init(const std::string &instance_id,
                           const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept = 0;
    virtual ErrorCode Open() noexcept = 0;
    virtual ErrorCode Close() noexcept = 0;

    // ----- Write -----
    virtual std::vector<ErrorCode> Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept = 0;
    virtual std::vector<ErrorCode> UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept = 0;
    virtual std::vector<ErrorCode> Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept = 0;
    virtual std::vector<ErrorCode> Delete(const KeyTypeVec &keys) noexcept = 0;

    virtual std::vector<ErrorCode>
    DeleteFields(const KeyTypeVec &keys, const std::vector<std::vector<std::string>> &field_names_vec) noexcept = 0;

    // ----- Read -----
    virtual std::vector<ErrorCode>
    Get(const KeyTypeVec &keys, const std::vector<std::string> &field_names, FieldMapVec &out_field_maps) noexcept = 0;
    virtual std::vector<ErrorCode> Get(const KeyTypeVec &keys,
                                       const std::vector<std::vector<std::string>> &field_names_vec,
                                       FieldMapVec &out_field_maps) noexcept = 0;
    virtual std::vector<ErrorCode> GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept = 0;
    virtual std::vector<ErrorCode>
    GetFieldNamesWithPrefix(const KeyTypeVec &keys,
                            const std::string &field_prefix,
                            std::vector<std::vector<std::string>> &out_field_names_vec) noexcept = 0;
    virtual std::vector<ErrorCode> Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept = 0;
    virtual std::vector<ErrorCode> ExistsFieldWithPrefix(const KeyTypeVec &keys,
                                                         const std::string &field_prefix,
                                                         std::vector<bool> &out_exists_vec) noexcept = 0;
    virtual ErrorCode ListKeys(const std::string &cursor,
                               const int64_t limit,
                               std::string &out_next_cursor,
                               KeyTypeVec &out_keys) noexcept = 0;
    virtual ErrorCode RandomSample(const int64_t count, KeyTypeVec &out_keys) noexcept = 0;
    virtual ErrorCode SampleReclaimKeys(const int64_t count, KeyTypeVec &out_keys) noexcept = 0;

    // meta data
    virtual ErrorCode PutMetaData(const FieldMap &field_maps) noexcept = 0;
    virtual ErrorCode GetMetaData(FieldMap &field_maps) noexcept = 0;
};
} // namespace kv_cache_manager
