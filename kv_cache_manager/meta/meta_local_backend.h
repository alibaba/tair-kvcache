#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <random>
#include <shared_mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/common/cache/advanced_cache.h"
#include "kv_cache_manager/common/cache/cache.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/meta_cache_policy_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_cache_base_backend.h"

namespace kv_cache_manager {

struct MetaMemCacheItem {
    using FieldMap = MetaCacheBaseBackend::FieldMap;

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
    std::shared_mutex &GetMutex() const { return mutex_; }

    int64_t GetLastAccessTime() const { return last_access_time_.load(std::memory_order_relaxed); }
    void TouchAccessTime() { last_access_time_.store(TimestampUtil::GetCurrentTimeUs(), std::memory_order_relaxed); }

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
    static void Deleter(void *value, MemoryAllocator * /*allocator*/) { delete static_cast<MetaMemCacheItem *>(value); }

private:
    mutable std::shared_mutex mutex_;
    FieldMap fields_;
    std::atomic<int64_t> last_access_time_{0};
};

class MetaLocalBackend : public MetaCacheBaseBackend {
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
    std::vector<ErrorCode> PutIfAbsent(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> Delete(const KeyTypeVec &keys) noexcept override;
    std::vector<ErrorCode> DeleteFields(const KeyTypeVec &keys,
                                        const std::vector<std::vector<std::string>> &field_names_vec) noexcept override;

    // Conditional write: only processes keys where previous_error_codes[i] == EC_OK.
    std::vector<ErrorCode> Put(const KeyTypeVec &keys,
                               const FieldMapVec &field_maps,
                               const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> PutIfAbsent(const KeyTypeVec &keys,
                                       const FieldMapVec &field_maps,
                                       const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> UpdateFields(const KeyTypeVec &keys,
                                        const FieldMapVec &field_maps,
                                        const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Upsert(const KeyTypeVec &keys,
                                  const FieldMapVec &field_maps,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Delete(const KeyTypeVec &keys,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    // Adjusts the LRU charge to reflect the size change.
    std::vector<ErrorCode> DeleteFields(const KeyTypeVec &keys,
                                        const std::vector<std::vector<std::string>> &field_names_vec,
                                        const std::vector<ErrorCode> &previous_error_codes) noexcept override;

    // read
    std::vector<ErrorCode> Get(const KeyTypeVec &keys,
                               const std::vector<std::string> &field_names,
                               FieldMapVec &out_field_maps) noexcept override;
    std::vector<ErrorCode> Get(const KeyTypeVec &keys,
                               const std::vector<std::vector<std::string>> &field_names_vec,
                               FieldMapVec &out_field_maps) noexcept override;
    std::vector<ErrorCode> GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept override;
    std::vector<ErrorCode> Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept override;
    std::vector<ErrorCode> ExistsFieldWithPrefix(const KeyTypeVec &keys,
                                                 const std::string &field_prefix,
                                                 std::vector<bool> &out_exists_vec) noexcept override;
    std::vector<ErrorCode>
    GetFieldNamesWithPrefix(const KeyTypeVec &keys,
                            const std::string &field_prefix,
                            std::vector<std::vector<std::string>> &out_field_names_vec) noexcept override;
    ErrorCode ListKeys(const std::string &cursor,
                       const int64_t limit,
                       std::string &out_next_cursor,
                       std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode RandomSample(const int64_t count, std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode SampleReclaimKeys(const int64_t count, std::vector<KeyType> &out_keys) noexcept override;

    // meta data
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;

    size_t GetMemUsage() const noexcept override;

private:
    // Collects up to `count` oldest keys from the given shard, parses them to
    // KeyType and appends to `out_keys`. Returns the number of keys collected.
    size_t CollectOldestKeysFromShard(uint32_t shard_id, size_t count, std::vector<KeyType> &out_keys);

    // Per-key helpers used by both unconditional and conditional write methods.
    ErrorCode UpsertForOneKey(KeyType key, const FieldMap &field_map);
    ErrorCode DeleteForOneKey(KeyType key);
    ErrorCode DeleteFieldsForOneKey(KeyType key, const std::vector<std::string> &field_names);
    ErrorCode GetForOneKey(KeyType key, const std::vector<std::string> &field_names, FieldMap &out_field_map);

    // Creates a MetaMemCacheItem from the given fields and inserts it into
    // cache_ via Insert().  On failure the item is deleted and the error code
    // is returned.
    ErrorCode CreateAndInsert(const std::string &key_str, const FieldMap &fields);
    ErrorCode CreateAndInsert(const std::string &key_str, FieldMap &&fields);

    // Creates a MetaMemCacheItem and inserts it into cache_ via InsertIfAbsent().
    ErrorCode CreateAndInsertIfAbsent(const std::string &key_str, const FieldMap &fields);

    // Looks up an existing cache item and merges `updates` into its fields
    // in place. Adjusts the LRU charge to reflect the size change.
    // Returns EC_OK on success, EC_NOENT if key not found.
    ErrorCode UpdateFieldsInPlace(const std::string &key_str, const FieldMap &updates);

    std::shared_ptr<Cache::CacheItemHelper> cache_item_helper_;
    std::shared_ptr<Cache> cache_;
    uint32_t shard_mask_ = 0;
    size_t sample_times_ = 0;

    // Per-shard approximate oldest access time, updated via tail-change callback
    // from the LRU cache. Used by SampleReclaimKeys to select the oldest shards
    // without acquiring any shard locks.
    std::unique_ptr<std::atomic<int64_t>[]> shard_oldest_access_time_;
};

} // namespace kv_cache_manager
