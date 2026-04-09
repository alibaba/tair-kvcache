#pragma once

#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "kv_cache_manager/common/cache/advanced_cache.h"
#include "kv_cache_manager/common/cache/cache.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/meta_cache_policy_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_local_base_backend.h"

namespace kv_cache_manager {

struct MetaMemCacheItem {
    using FieldMap = MetaLocalBaseBackend::FieldMap;

    FieldMap fields_;

    // Estimates total memory footprint including the heap memory owned by
    // FieldMap entries, used as the "charge" for LRU cache eviction accounting.
    size_t Size() const {
        size_t total = sizeof(MetaMemCacheItem);
        for (const auto &[field_name, field_value] : fields_) {
            // Each map entry: key string heap + value string heap + tree-node overhead
            total += field_name.size() + field_value.size() + sizeof(void *) * 4;
        }
        return total;
    }

    const FieldMap &GetFields() const { return fields_; }

    FieldMap &GetMutableFields() { return fields_; }

    static MetaMemCacheItem *Create(const FieldMap &fields) {
        auto *item = new MetaMemCacheItem();
        item->fields_ = fields;
        return item;
    }

    static MetaMemCacheItem *Create(FieldMap &&fields) {
        auto *item = new MetaMemCacheItem();
        item->fields_ = std::move(fields);
        return item;
    }

    static void Deleter(void *value, MemoryAllocator * /*allocator*/) {
        delete static_cast<MetaMemCacheItem *>(value);
    }
};

class MetaLocalBackend : public MetaLocalBaseBackend {
public:
    MetaLocalBackend() = default;
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

    // Conditional write: only processes keys where previous_error_codes[i] == EC_OK.
    std::vector<ErrorCode> Put(const KeyTypeVec &keys,
                               const FieldMapVec &field_maps,
                               const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> UpdateFields(const KeyTypeVec &keys,
                                        const FieldMapVec &field_maps,
                                        const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Upsert(const KeyTypeVec &keys,
                                  const FieldMapVec &field_maps,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> IncrFields(const KeyTypeVec &keys,
                                      const std::map<std::string, int64_t> &field_amounts,
                                      const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Delete(const KeyTypeVec &keys,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> PutIfAbsent(const KeyTypeVec &keys,
                                       const FieldMapVec &field_maps,
                                       const std::vector<ErrorCode> &previous_error_codes) noexcept override;

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
    ErrorCode SampleReclaimKeys(const int64_t count, std::vector<KeyType> &out_keys) noexcept override;

    // meta data
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;

    // Returns up to `count` oldest keys from a randomly chosen shard.
    ErrorCode GetOldestKeysFromRandomShard(size_t count, std::vector<KeyType> &out_keys);

private:
    // Per-key write helpers used by both unconditional and conditional write methods.
    ErrorCode UpsertForOneKey(KeyType key, const FieldMap &field_map);
    ErrorCode DeleteForOneKey(KeyType key);

    // Creates a MetaMemCacheItem from the given fields and inserts it into
    // cache_ via Insert().  On failure the item is deleted and the error code
    // is returned.
    ErrorCode CreateAndInsert(const std::string &key_str, const FieldMap &fields);
    ErrorCode CreateAndInsert(const std::string &key_str, FieldMap &&fields);

    // Creates a MetaMemCacheItem and inserts it into cache_ via InsertIfAbsent().
    ErrorCode CreateAndInsertIfAbsent(const std::string &key_str, const FieldMap &fields);

    // Looks up an existing cache item and returns a copy of its fields.
    // Returns true if the key was found, false otherwise.
    bool LookupFields(const std::string &key_str, FieldMap &out_fields);

    // Looks up an existing cache item and merges `updates` into its fields
    // in place (the handle prevents eviction during the update).
    // Returns EC_OK on success, EC_NOENT if key not found.
    ErrorCode UpdateFieldsInPlace(const std::string &key_str, const FieldMap &updates);

    std::shared_ptr<Cache::CacheItemHelper> cache_item_helper_;
    std::shared_ptr<Cache> cache_;
    uint32_t shard_mask_ = 0;
    int64_t sample_times_ = 0;
};

} // namespace kv_cache_manager