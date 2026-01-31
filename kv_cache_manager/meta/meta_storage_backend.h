#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"

namespace kv_cache_manager {
class MetaStorageBackendConfig;

// MetaIndexer 后端存储抽象基类
class MetaStorageBackend {
public:
    using KeyType = int64_t;
    using KeyTypeVec = std::vector<KeyType>;
    using FieldMap = std::map<std::string, std::string>;
    using FieldMapVec = std::vector<std::map<std::string, std::string>>;

public:
    virtual ~MetaStorageBackend() = default;

    virtual std::string GetStorageType() noexcept = 0;

    virtual ErrorCode Init(const std::string &instance_id,
                           const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept = 0;
    virtual ErrorCode Open() noexcept = 0;
    virtual ErrorCode Close() noexcept = 0;

    // write
    virtual std::vector<ErrorCode> Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept = 0;
    virtual std::vector<ErrorCode> UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept = 0;
    virtual std::vector<ErrorCode> Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept = 0;
    virtual std::vector<ErrorCode> IncrFields(const KeyTypeVec &keys,
                                              const std::map<std::string, int64_t> &field_amounts) noexcept = 0;
    virtual std::vector<ErrorCode> Delete(const KeyTypeVec &keys) noexcept = 0;

    // read
    virtual std::vector<ErrorCode>
    Get(const KeyTypeVec &keys, const std::vector<std::string> &field_names, FieldMapVec &out_field_maps) noexcept = 0;
    virtual std::vector<ErrorCode> GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept = 0;
    virtual std::vector<ErrorCode> Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept = 0;
    virtual ErrorCode ListKeys(const std::string &cursor,
                               const int64_t limit,
                               std::string &out_next_cursor,
                               KeyTypeVec &out_keys) noexcept = 0;
    virtual ErrorCode RandomSample(const int64_t count, KeyTypeVec &out_keys) noexcept = 0;

    // meta data
    virtual ErrorCode PutMetaData(const FieldMap &field_maps) noexcept = 0;
    virtual ErrorCode GetMetaData(FieldMap &field_maps) noexcept = 0;
};
} // namespace kv_cache_manager
