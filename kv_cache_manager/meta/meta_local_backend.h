#pragma once

#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <vector>

#include "kv_cache_manager/common/cache/advanced_cache.h"
#include "kv_cache_manager/common/cache/cache.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/meta_cache_policy_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_cache_base_backend.h"
#include "kv_cache_manager/metrics/revisit_interval_histogram.h"

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
    void TouchAccessTime(int64_t access_time_us) {
        int64_t current = last_access_time_.load(std::memory_order_relaxed);
        while (current < access_time_us &&
               !last_access_time_.compare_exchange_weak(
                   current, access_time_us, std::memory_order_relaxed, std::memory_order_relaxed)) {}
    }

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
    static MetaMemCacheItem *CreateSingleLocation(const LocationId &location_id, CacheLocationConstPtr location) {
        auto *item = new MetaMemCacheItem();
        item->locations_.emplace(location_id, std::move(location));
        return item;
    }
    static void Deleter(void *value, MemoryAllocator * /*allocator*/) { delete static_cast<MetaMemCacheItem *>(value); }

private:
    mutable std::shared_mutex mutex_;
    CacheLocationMap locations_;
    PropertyMap properties_;
    std::atomic<int64_t> last_access_time_{0};
};

// Caller-owned workspace for the pure-local one-location RMW path. It is
// prepared before MetaIndexer acquires metadata shard locks, then reused by
// both the read and write halves of the operation. The fused path may retain
// read handles until its matching write call; the destructor is a final guard
// that releases every handle on early returns.
struct SingleLocationRmwScratch {
    SingleLocationRmwScratch() = default;
    ~SingleLocationRmwScratch();
    SingleLocationRmwScratch(const SingleLocationRmwScratch &) = delete;
    SingleLocationRmwScratch &operator=(const SingleLocationRmwScratch &) = delete;

    void ReleaseRetainedHandles() noexcept;
    [[nodiscard]] bool HasRetainedHandles() const noexcept { return retained_handle_owner != nullptr; }

    std::vector<std::string_view> key_views;
    std::vector<Cache::Handle *> handles;
    CacheLocationVector retired_locations;
    Cache *retained_handle_owner = nullptr;
};

class MetaLocalBackend : public MetaCacheBaseBackend {
public:
    MetaLocalBackend() = default;
    ~MetaLocalBackend() = default;

    // Set histogram for revisit interval tracking (overrides base class no-op).
    void SetRevisitHistogram(std::shared_ptr<RevisitIntervalHistogram> histogram) override {
        revisit_histogram_ = std::move(histogram);
    }

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
    std::vector<ErrorCode> UpsertSingleLocations(RequestContext *request_context,
                                                 const KeyTypeVec &keys,
                                                 const LocationIdRefVector &location_ids,
                                                 const CacheLocationVector &locations) noexcept override;
    void PrepareSingleLocationRmwScratch(size_t max_count, SingleLocationRmwScratch &scratch) noexcept;
    void UpsertSingleLocationsInto(RequestContext *request_context,
                                   const KeyTypeVec &keys,
                                   const LocationIdRefVector &location_ids,
                                   const CacheLocationVector &locations,
                                   std::vector<ErrorCode> &out_results,
                                   SingleLocationRmwScratch &scratch) noexcept;
    void UpsertSingleLocationsUsingRetainedHandlesInto(RequestContext *request_context,
                                                       const KeyTypeVec &keys,
                                                       const LocationIdRefVector &location_ids,
                                                       CacheLocationVector &locations,
                                                       const std::vector<size_t> &read_indices,
                                                       std::vector<ErrorCode> &out_results,
                                                       SingleLocationRmwScratch &scratch) noexcept;
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
    std::vector<ErrorCode> UpsertConsume(RequestContext *request_context,
                                         const KeyTypeVec &keys,
                                         CacheLocationMapVector &locations,
                                         PropertyMapVector &properties,
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
    std::vector<ErrorCode> GetLocationValues(RequestContext *request_context,
                                             const KeyTypeVec &keys,
                                             LocationsPerKey &out_locations) noexcept override;
    std::vector<ErrorCode> GetLocationValuesCompact(RequestContext *request_context,
                                                    const KeyType *keys,
                                                    size_t key_count,
                                                    CompactLocationsPerKey &out_locations) noexcept override;
    std::vector<std::vector<ErrorCode>> GetLocations(RequestContext *request_context,
                                                     const KeyTypeVec &keys,
                                                     const LocationIdsPerKey &location_ids,
                                                     LocationsPerKey &out_locations) noexcept override;
    std::vector<std::vector<ErrorCode>>
    GetLocationsWithKeyStatus(RequestContext *request_context,
                              const KeyTypeVec &keys,
                              const LocationIdsPerKey &location_ids,
                              LocationsPerKey &out_locations,
                              std::vector<ErrorCode> &out_key_error_codes) noexcept override;
    std::vector<ErrorCode>
    GetSingleLocationsWithKeyStatus(RequestContext *request_context,
                                    const KeyTypeVec &keys,
                                    const LocationIdRefVector &location_ids,
                                    CacheLocationVector &out_locations,
                                    std::vector<ErrorCode> &out_key_error_codes) noexcept override;
    void GetSingleLocationsWithKeyStatusInto(RequestContext *request_context,
                                             const KeyTypeVec &keys,
                                             const LocationIdRefVector &location_ids,
                                             CacheLocationVector &out_locations,
                                             std::vector<ErrorCode> &out_key_error_codes,
                                             std::vector<ErrorCode> &out_results,
                                             SingleLocationRmwScratch &scratch,
                                             bool retain_handles = false) noexcept;
    // Pure-local RMW-only variant: borrows immutable location values without
    // incrementing one shared_ptr control block per key. Callers must retain
    // the handles and consume every view before the matching write/release.
    void GetSingleLocationViewsWithKeyStatusInto(RequestContext *request_context,
                                                 const KeyTypeVec &keys,
                                                 const LocationIdRefVector &location_ids,
                                                 CacheLocationViewVector &out_locations,
                                                 std::vector<ErrorCode> &out_key_error_codes,
                                                 std::vector<ErrorCode> &out_results,
                                                 SingleLocationRmwScratch &scratch) noexcept;
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
    ErrorCode ScanLocationsForMaintenance(RequestContext *request_context,
                                          const std::string &cursor,
                                          int64_t limit,
                                          MaintenanceScanBatch &out) noexcept override;
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
    bool GetCacheHashSeed(uint32_t &out_hash_seed) const noexcept;

private:
    static std::string_view KeyToView(const KeyType &key) {
        return {reinterpret_cast<const char *>(&key), sizeof(KeyType)};
    }
    static KeyType ViewToKey(std::string_view sv) {
        KeyType key = 0;
        std::memcpy(&key, sv.data(), sizeof(KeyType));
        return key;
    }

    size_t CollectOldestKeysFromShard(uint32_t shard_id, size_t count, std::vector<KeyType> &out_keys);
    ErrorCode
    CreateAndInsert(std::string_view key_sv,
                    const CacheLocationMap &locations,
                    const PropertyMap &properties,
                    int64_t access_time_us);
    ErrorCode CreateAndInsertConsume(std::string_view key_sv,
                                     CacheLocationMap &locations,
                                     PropertyMap &properties,
                                     int64_t access_time_us);
    ErrorCode
    CreateAndInsertIfAbsent(std::string_view key_sv,
                            const CacheLocationMap &locations,
                            const PropertyMap &properties,
                            int64_t access_time_us);
    ErrorCode UpdateHandleInPlace(Cache::Handle *handle,
                                  const CacheLocationMap &locations,
                                  const PropertyMap &properties,
                                  int64_t access_time_us);
    ErrorCode UpdateHandleInPlaceConsume(Cache::Handle *handle,
                                         CacheLocationMap &locations,
                                         PropertyMap &properties,
                                         int64_t access_time_us);
    ErrorCode UpdateHandleInPlaceSingleLocation(Cache::Handle *handle,
                                                const LocationId &location_id,
                                                CacheLocationConstPtr location,
                                                int64_t access_time_us,
                                                CacheLocationVector *retired_locations = nullptr,
                                                bool release_handle = false);
    ErrorCode UpdateInPlace(std::string_view key_sv,
                            const CacheLocationMap &locations,
                            const PropertyMap &properties,
                            int64_t access_time_us);
    ErrorCode CreateAndInsertSingleLocation(std::string_view key_sv,
                                            const LocationId &location_id,
                                            CacheLocationConstPtr location,
                                            int64_t access_time_us,
                                            CacheLocationVector *retired_locations = nullptr);
    void GetSingleLocationsWithKeyStatusIntoImpl(RequestContext *request_context,
                                                 const KeyTypeVec &keys,
                                                 const LocationIdRefVector &location_ids,
                                                 CacheLocationVector *out_owned_locations,
                                                 CacheLocationViewVector *out_borrowed_locations,
                                                 std::vector<ErrorCode> &out_key_error_codes,
                                                 std::vector<ErrorCode> &out_results,
                                                 SingleLocationRmwScratch &scratch,
                                                 bool retain_handles) noexcept;
    ErrorCode UpsertForOneKey(KeyType key,
                              const CacheLocationMap &locations,
                              const PropertyMap &properties,
                              int64_t access_time_us);
    ErrorCode UpsertConsumeForOneKey(KeyType key,
                                     CacheLocationMap &locations,
                                     PropertyMap &properties,
                                     int64_t access_time_us);
    ErrorCode DeleteForOneKey(KeyType key);
    ErrorCode DeleteLocationsForOneKey(KeyType key,
                                       const std::vector<LocationId> &location_ids,
                                       int64_t access_time_us);
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
    std::shared_ptr<RevisitIntervalHistogram> revisit_histogram_;
};

} // namespace kv_cache_manager
