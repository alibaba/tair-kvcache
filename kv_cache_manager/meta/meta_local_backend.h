#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
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
#include "kv_cache_manager/meta/utils.h"
#include "kv_cache_manager/metrics/revisit_interval_histogram.h"

namespace kv_cache_manager {

struct LocalItemExtra {
    std::unique_ptr<CacheLocationMap> multiple_locations;
    std::unique_ptr<PropertyMap> extra_properties;

    bool Empty() const { return !multiple_locations && !extra_properties; }
};

class LocalLocationStore {
public:
    ssize_t Merge(const CacheLocationMap &locations, std::unique_ptr<LocalItemExtra> &extra);
    ssize_t Upsert(const LocationId &location_id,
                   CacheLocationConstPtr location,
                   std::unique_ptr<LocalItemExtra> &extra,
                   CacheLocationVector *retired_locations = nullptr);
    ssize_t Erase(const LocationIdVector &location_ids, LocalItemExtra *extra);

    const CacheLocationConstPtr *Find(const LocationId &location_id, const LocalItemExtra *extra) const;
    void CopyTo(CacheLocationMap &out, const LocalItemExtra *extra) const;
    size_t Size(const LocalItemExtra *extra) const;
    bool Empty(const LocalItemExtra *extra) const { return Size(extra) == 0; }
    size_t EstimateUsage(const LocalItemExtra *extra) const;

    template <typename Visitor>
    void ForEach(const LocalItemExtra *extra, Visitor &&visitor) const {
        if (extra && extra->multiple_locations) {
            for (const auto &[location_id, location] : *extra->multiple_locations) {
                visitor(location_id, location);
            }
        } else if (inline_location_) {
            visitor(inline_location_->id(), inline_location_);
        }
    }

private:
    // Callers guarantee every stored value is non-null and its id matches the
    // map key. The compact representation therefore reuses CacheLocation::id().
    // Returns the usage delta caused by the mutation instead of relying on a
    // full before/after EstimateUsage() rescan of the container.
    ssize_t UpsertImpl(const LocationId &location_id,
                       CacheLocationConstPtr location,
                       std::unique_ptr<LocalItemExtra> &extra,
                       CacheLocationVector *retired_locations);
    ssize_t NormalizeAfterErase(LocalItemExtra *extra);
    static size_t EstimateMapEntryUsage(const LocationId &location_id, const CacheLocationConstPtr &location);

    CacheLocationConstPtr inline_location_;
};

class LocalPropertyStore {
public:
    ssize_t Merge(const PropertyMap &properties, std::unique_ptr<LocalItemExtra> &extra);
    void CopySelected(const std::vector<std::string> &field_names, PropertyMap &out, const LocalItemExtra *extra) const;
    void CopyAll(PropertyMap &out, const LocalItemExtra *extra) const;
    size_t EstimateUsage(const LocalItemExtra *extra) const;

private:
    enum class PrevKeyState : uint8_t {
        kAbsent,
        kEmpty,
        kCanonical
    };

    // Both helpers return the usage delta they caused so Merge can accumulate
    // it locally without a full before/after EstimateUsage() rescan.
    ssize_t ErasePrevKeyFallback(LocalItemExtra *extra);
    ssize_t UpsertExtraProperty(const std::string &name,
                                const std::string &value,
                                std::unique_ptr<LocalItemExtra> &extra);
    PropertyMap &EnsureExtraProperties(std::unique_ptr<LocalItemExtra> &extra);

    PrevKeyState prev_key_state_{PrevKeyState::kAbsent};
    KeyType prev_key_{0};
};

struct MetaMemCacheItem {
    // Stable LRU charge estimate for this item and its location/property allocations;
    // allocator-wide overhead and shared objects are not exact per-item costs.
    size_t Size() const {
        size_t total = sizeof(MetaMemCacheItem) + locations_.EstimateUsage(extra_.get());
        total += properties_.EstimateUsage(extra_.get());
        if (extra_) {
            total += sizeof(LocalItemExtra);
        }
        return total;
    }

    ssize_t MergeLocations(const CacheLocationMap &locations) {
        return MutateExtra([&] { return locations_.Merge(locations, extra_); });
    }
    ssize_t UpsertLocation(const LocationId &id,
                           CacheLocationConstPtr location,
                           CacheLocationVector *retired_locations = nullptr) {
        return MutateExtra([&] { return locations_.Upsert(id, std::move(location), extra_, retired_locations); });
    }
    ssize_t EraseLocations(const LocationIdVector &ids) {
        return MutateExtra([&] { return locations_.Erase(ids, extra_.get()); });
    }
    const CacheLocationConstPtr *FindLocation(const LocationId &id) const { return locations_.Find(id, extra_.get()); }
    void CopyLocations(CacheLocationMap &out) const { locations_.CopyTo(out, extra_.get()); }
    size_t LocationCount() const { return locations_.Size(extra_.get()); }
    bool LocationsEmpty() const { return locations_.Empty(extra_.get()); }
    template <typename Visitor>
    void ForEachLocation(Visitor &&visitor) const {
        locations_.ForEach(extra_.get(), std::forward<Visitor>(visitor));
    }
    ssize_t MergeProperties(const PropertyMap &properties) {
        return MutateExtra([&] { return properties_.Merge(properties, extra_); });
    }
    void CopySelectedProperties(const std::vector<std::string> &names, PropertyMap &out) const {
        properties_.CopySelected(names, out, extra_.get());
    }
    void CopyAllProperties(PropertyMap &out) const { properties_.CopyAll(out, extra_.get()); }

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
        item->MergeLocations(locations);
        item->MergeProperties(properties);
        return item;
    }
    static MetaMemCacheItem *CreateSingleLocation(const LocationId &location_id, CacheLocationConstPtr location) {
        auto *item = new MetaMemCacheItem();
        item->UpsertLocation(location_id, std::move(location));
        return item;
    }
    static void Deleter(void *value, MemoryAllocator * /*allocator*/) { delete static_cast<MetaMemCacheItem *>(value); }

private:
    template <typename Mutation>
    ssize_t MutateExtra(Mutation &&mutation) {
        const bool had_extra = bool(extra_);
        const ssize_t delta = mutation();
        if (extra_ && extra_->Empty()) {
            extra_.reset();
        }
        return delta + (static_cast<ssize_t>(bool(extra_)) - static_cast<ssize_t>(had_extra)) *
                           static_cast<ssize_t>(sizeof(LocalItemExtra));
    }

    LocalLocationStore locations_;
    LocalPropertyStore properties_;
    std::atomic<int64_t> last_access_time_{0};
    std::unique_ptr<LocalItemExtra> extra_;
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
    Cache::BatchOperationScratch cache_batch;
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
    std::vector<ErrorCode> GetLocationsForMaintenance(RequestContext *request_context,
                                                      const KeyTypeVec &keys,
                                                      CacheLocationMapVector &out_locations) noexcept override;
    std::vector<std::vector<ErrorCode>> GetLocationsForMaintenance(RequestContext *request_context,
                                                                   const KeyTypeVec &keys,
                                                                   const LocationIdsPerKey &location_ids,
                                                                   LocationsPerKey &out_locations) noexcept override;
    std::vector<ErrorCode> GetLocationIds(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          LocationIdsPerKey &out_location_ids) noexcept override;
    std::vector<ErrorCode> GetLocationIdsForMaintenance(RequestContext *request_context,
                                                        const KeyTypeVec &keys,
                                                        LocationIdsPerKey &out_location_ids) noexcept override;
    std::vector<ErrorCode> GetLocationMapsForMaintenance(RequestContext *request_context,
                                                         const KeyTypeVec &keys,
                                                         CacheLocationMapVector &out_locations) noexcept override;
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
    std::vector<ErrorCode> DeleteLocationsForMaintenance(RequestContext *request_context,
                                                         const KeyTypeVec &keys,
                                                         const LocationIdsPerKey &location_ids) noexcept override;
    std::vector<ErrorCode>
    DeleteLocationsForMaintenance(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const LocationIdsPerKey &location_ids,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    ErrorCode RandomSample(RequestContext *request_context,
                           const int64_t count,
                           std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode SampleReclaimKeys(RequestContext *request_context,
                                const int64_t count,
                                std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode SampleReclaimCandidates(RequestContext *request_context,
                                      int64_t count,
                                      ReclaimCandidateVector &out_candidates,
                                      bool require_read_success = false) noexcept override;
    std::vector<ErrorCode>
    GetLastAccessTimesForMaintenance(RequestContext *request_context,
                                     const KeyTypeVec &keys,
                                     std::vector<int64_t> &out_last_access_times) noexcept override;
    size_t TouchKeysForMaintenance(const KeyTypeVec &keys) noexcept override;

    // meta data
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;
    ErrorCode DeleteMetaData() noexcept override;

    // LRU capacity charge, not process RSS or allocator allocated bytes.
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
    CreateAndInsert(std::string_view key_sv, const CacheLocationMap &locations, const PropertyMap &properties);
    ErrorCode
    CreateAndInsertIfAbsent(std::string_view key_sv, const CacheLocationMap &locations, const PropertyMap &properties);
    ErrorCode UpdateHandleInPlace(KeyType key,
                                  Cache::Handle *handle,
                                  const CacheLocationMap &locations,
                                  const PropertyMap &properties);
    ErrorCode UpdateHandleInPlaceSingleLocation(KeyType key,
                                                Cache::Handle *handle,
                                                const LocationId &location_id,
                                                CacheLocationConstPtr location,
                                                CacheLocationVector *retired_locations = nullptr);
    ErrorCode UpdateInPlace(std::string_view key_sv, const CacheLocationMap &locations, const PropertyMap &properties);
    ErrorCode CreateAndInsertSingleLocation(std::string_view key_sv,
                                            const LocationId &location_id,
                                            CacheLocationConstPtr location);
    ErrorCode
    UpsertSingleLocationForOneKey(KeyType key, const LocationId &location_id, const CacheLocationConstPtr &location);
    void GetSingleLocationsWithKeyStatusIntoImpl(RequestContext *request_context,
                                                 const KeyTypeVec &keys,
                                                 const LocationIdRefVector &location_ids,
                                                 CacheLocationVector *out_owned_locations,
                                                 CacheLocationViewVector *out_borrowed_locations,
                                                 std::vector<ErrorCode> &out_key_error_codes,
                                                 std::vector<ErrorCode> &out_results,
                                                 SingleLocationRmwScratch &scratch,
                                                 bool retain_handles) noexcept;
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
    ErrorCode GetForOneKeyForMaintenance(KeyType key,
                                         CacheLocationMap *out_location_map,
                                         std::vector<LocationId> *out_location_ids);

    // The LRU maintenance callback holds its shard lock before taking this
    // lock. Normal paths must release it before AdjustCharge/Release/Erase.
    std::shared_mutex &ItemMutex(KeyType key) const {
        return item_mutexes_[GetShardIndex(key, item_mutex_shard_mask_)];
    }

    std::shared_ptr<Cache::CacheItemHelper> cache_item_helper_;
    std::shared_ptr<Cache> cache_;
    std::unique_ptr<std::shared_mutex[]> item_mutexes_;
    uint32_t item_mutex_shard_mask_ = 0;
    std::unique_ptr<std::atomic<int64_t>[]> shard_oldest_access_time_;
    uint32_t shard_mask_ = 0;
    size_t sample_times_ = 0;
    std::shared_ptr<RevisitIntervalHistogram> revisit_histogram_;
};

} // namespace kv_cache_manager
