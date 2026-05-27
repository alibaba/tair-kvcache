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
    // Estimates total memory footprint including the heap memory owned by
    // CacheLocationMap and PropertyMap entries, used as the "charge" for LRU cache eviction accounting.
    size_t Size() const {
        size_t total = sizeof(MetaMemCacheItem);
        for (const auto &[location_id, location] : locations_) {
            // unordered_map node overhead + key string heap + shared_ptr overhead + CacheLocation footprint
            total += sizeof(void *) * 4 + location_id.size() + sizeof(CacheLocationConstPtr) +
                     (location ? location->EstimateMemUsage() : 0);
        }
        for (const auto &[prop_name, prop_value] : properties_) {
            total += sizeof(void *) * 4 + prop_name.size() + prop_value.size();
        }
        return total;
    }

    const CacheLocationMap &GetLocations() const { return locations_; }
    CacheLocationMap &GetMutableLocations() { return locations_; }
    const PropertyMap &GetProperties() const { return properties_; }
    PropertyMap &GetMutableProperties() { return properties_; }
    std::shared_mutex &GetMutex() const { return mutex_; }

    int64_t GetLastAccessTime() const { return last_access_time_.load(std::memory_order_relaxed); }
    void TouchAccessTime() { last_access_time_.store(TimestampUtil::GetCurrentTimeUs(), std::memory_order_relaxed); }

    static MetaMemCacheItem *Create(const CacheLocationMap &locations, const PropertyMap &properties) {
        auto *item = new MetaMemCacheItem();
        item->locations_ = locations;
        item->properties_ = properties;
        return item;
    }
    static MetaMemCacheItem *Create(CacheLocationMap &&locations, PropertyMap &&properties) {
        auto *item = new MetaMemCacheItem();
        item->locations_ = std::move(locations);
        item->properties_ = std::move(properties);
        return item;
    }
    static void Deleter(void *value, MemoryAllocator * /*allocator*/) { delete static_cast<MetaMemCacheItem *>(value); }

private:
    mutable std::shared_mutex mutex_;
    CacheLocationMap locations_;
    PropertyMap properties_;
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
    std::vector<ErrorCode> Put(RequestContext *request_context,
                               const KeyTypeVec &keys,
                               const CacheLocationMapVector &locations,
                               const PropertyMapVector &properties) noexcept override;
    std::vector<ErrorCode> PutIfAbsent(RequestContext *request_context,
                                       const KeyTypeVec &keys,
                                       const CacheLocationMapVector &locations,
                                       const PropertyMapVector &properties) noexcept override;
    std::vector<ErrorCode> Upsert(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &locations,
                                  const PropertyMapVector &properties) noexcept override;
    std::vector<ErrorCode> Update(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &locations,
                                  const PropertyMapVector &properties) noexcept override;
    std::vector<ErrorCode> Delete(RequestContext *request_context, const KeyTypeVec &keys) noexcept override;
    std::vector<ErrorCode> DeleteLocations(RequestContext *request_context,
                                           const KeyTypeVec &keys,
                                           const LocationIdsPerKey &location_ids) noexcept override;

    // Conditional write: only processes keys where previous_error_codes[i] == EC_OK.
    std::vector<ErrorCode> Put(RequestContext *request_context,
                               const KeyTypeVec &keys,
                               const CacheLocationMapVector &locations,
                               const PropertyMapVector &properties,
                               const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> PutIfAbsent(RequestContext *request_context,
                                       const KeyTypeVec &keys,
                                       const CacheLocationMapVector &locations,
                                       const PropertyMapVector &properties,
                                       const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Upsert(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &locations,
                                  const PropertyMapVector &properties,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Update(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &locations,
                                  const PropertyMapVector &properties,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Delete(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> DeleteLocations(RequestContext *request_context,
                                           const KeyTypeVec &keys,
                                           const LocationIdsPerKey &location_ids,
                                           const std::vector<ErrorCode> &previous_error_codes) noexcept override;

    // read
    std::vector<ErrorCode> Get(RequestContext *request_context,
                               const KeyTypeVec &keys,
                               CacheLocationMapVector &out_locations,
                               PropertyMapVector &out_properties) noexcept override;
    std::vector<ErrorCode> GetLocations(RequestContext *request_context,
                                        const KeyTypeVec &keys,
                                        CacheLocationMapVector &out_locations) noexcept override;
    std::vector<std::vector<ErrorCode>> GetLocations(RequestContext *request_context,
                                                     const KeyTypeVec &keys,
                                                     const LocationIdsPerKey &location_ids,
                                                     LocationsPerKey &out_locations) noexcept override;
    std::vector<ErrorCode> GetLocationIds(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          LocationIdsPerKey &out_location_ids) noexcept override;
    std::vector<ErrorCode> GetProperties(RequestContext *request_context,
                                         const KeyTypeVec &keys,
                                         const std::vector<std::string> &field_names,
                                         PropertyMapVector &out_properties) noexcept override;
    std::vector<ErrorCode> Exists(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  std::vector<bool> &out_is_exist_vec) noexcept override;
    std::vector<ErrorCode> ExistsLocation(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          std::vector<bool> &out_exists) noexcept override;
    ErrorCode ListKeys(RequestContext *request_context,
                       const std::string &cursor,
                       const int64_t limit,
                       std::string &out_next_cursor,
                       std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode RandomSample(RequestContext *request_context,
                           const int64_t count,
                           std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode SampleReclaimKeys(RequestContext *request_context,
                                const int64_t count,
                                std::vector<KeyType> &out_keys) noexcept override;

    // meta data
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;

    size_t GetMemUsage() const noexcept override;
    int64_t GetOldestAccessTime() const noexcept override;

private:
    size_t CollectOldestKeysFromShard(uint32_t shard_id, size_t count, std::vector<KeyType> &out_keys);
    ErrorCode
    CreateAndInsert(const std::string &key_str, const CacheLocationMap &locations, const PropertyMap &properties);
    ErrorCode CreateAndInsertIfAbsent(const std::string &key_str,
                                      const CacheLocationMap &locations,
                                      const PropertyMap &properties);
    ErrorCode
    UpdateInPlace(const std::string &key_str, const CacheLocationMap &locations, const PropertyMap &properties);
    ErrorCode UpsertForOneKey(KeyType key, const CacheLocationMap &locations, const PropertyMap &properties);
    ErrorCode DeleteForOneKey(KeyType key);
    ErrorCode DeleteLocationsForOneKey(KeyType key, const std::vector<LocationId> &location_ids);
    // Unified read helper. Fetches data from cache for a single key.
    // Pass nullptr for any output you don't need.
    // - field_names: if non-null, only these properties are returned; otherwise all properties
    // - out_location_map: if non-null, copies the full CacheLocationMap
    // - out_property_map: if non-null, copies properties
    // - out_location_ids: if non-null, collects all location ids from the key
    // Returns EC_OK if key found, EC_NOENT otherwise.
    ErrorCode GetForOneKey(KeyType key,
                           const std::vector<std::string> *field_names,
                           CacheLocationMap *out_location_map,
                           PropertyMap *out_property_map,
                           std::vector<LocationId> *out_location_ids);

    std::shared_ptr<Cache::CacheItemHelper> cache_item_helper_;
    std::shared_ptr<Cache> cache_;
    std::unique_ptr<std::atomic<int64_t>[]> shard_oldest_access_time_;
    uint32_t shard_mask_ = 0;
    size_t sample_times_ = 0;
};

} // namespace kv_cache_manager
