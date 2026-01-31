#pragma once

#include <mutex>

#include "kv_cache_manager/common/concurrent_hash_map.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"

namespace kv_cache_manager {
class MetaLocalBackend : public MetaStorageBackend {
public:
    ~MetaLocalBackend() = default;

    std::string GetStorageType() noexcept override;

    ErrorCode Init(const std::string &instance_id,
                   const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept override;
    ErrorCode Open() noexcept override;
    ErrorCode Close() noexcept override;

    // write
    std::vector<ErrorCode> Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> IncrFields(const KeyTypeVec &keys,
                                      const std::map<std::string, int64_t> &field_amounts) noexcept override;
    std::vector<ErrorCode> Delete(const KeyTypeVec &keys) noexcept override;

    // read
    std::vector<ErrorCode> Get(const KeyTypeVec &keys,
                               const std::vector<std::string> &field_names,
                               FieldMapVec &out_field_maps) noexcept override;
    std::vector<ErrorCode> GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept override;
    std::vector<ErrorCode> Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept override;
    ErrorCode ListKeys(const std::string &cursor,
                       const int64_t limit,
                       std::string &out_next_cursor,
                       std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode RandomSample(const int64_t count, std::vector<KeyType> &out_keys) noexcept override;

    // meta data
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;

private:
    ErrorCode PersistToPath();

    ErrorCode PutForOneKey(const KeyType &key, const FieldMap &field_map);
    ErrorCode UpdateFieldsForOneKey(const KeyType &key, const FieldMap &field_map);
    ErrorCode UpsertForOneKey(const KeyType &key, const FieldMap &field_map);
    ErrorCode IncrFieldsForOneKey(const KeyType &key, const std::map<std::string, int64_t> &field_amounts);
    ErrorCode DeleteForOneKey(const KeyType &key);

    ErrorCode GetForOneKey(const KeyType &key, const std::vector<std::string> &field_names, FieldMap &out_field_map);
    ErrorCode GetAllFieldsForOneKey(const KeyType &key, FieldMap &out_field_map);
    ErrorCode ExistsForOneKey(const KeyType &key, bool &out_is_exist);

private:
    std::mutex mutex_;
    std::string path_;
    ConcurrentHashMap<KeyType, FieldMap> table_;
    bool enable_persistence_ = false;
};
} // namespace kv_cache_manager
