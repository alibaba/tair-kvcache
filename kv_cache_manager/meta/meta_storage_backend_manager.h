#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/meta_cache_base_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

class MetaStorageBackendConfig;
class RequestContext;
struct SingleLocationRmwScratch;

// Backend orchestrator with two modes (auto-selected at Init):
//   * Dual-backend: persistent (source-of-truth) + cache (hot cache).
//     Writes go persistent-first then cache; reads are cache-first,
//     falling back to persistent during Recover.
//   * Single-backend: persistent only (cache is null, no Recover).
//
// Callers must partition requests via MetaIndexer::MakeBatches and hold
// shard locks before invoking the manager.
class MetaStorageBackendManager {
public:
    enum class RecoverState {
        kRecover,
        kRunning,
    };

    MetaStorageBackendManager() = default;
    ~MetaStorageBackendManager();

    ErrorCode Init(const std::string &instance_id, const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept;

    ErrorCode Open() noexcept;
    ErrorCode Close() noexcept;

    RecoverState GetRecoverState() const noexcept { return recover_state_.load(std::memory_order_acquire); }

    // ----- Write APIs -----
    // Put / Upsert merge CacheLocations into batch.batch_properties in place.
    std::vector<ErrorCode> Put(RequestContext *request_context, BatchMetaData &batch) noexcept;
    std::vector<ErrorCode> Upsert(RequestContext *request_context, BatchMetaData &batch) noexcept;
    std::vector<ErrorCode> UpsertSingleLocations(RequestContext *request_context,
                                                 const KeyVector &keys,
                                                 const LocationIdRefVector &location_ids,
                                                 const CacheLocationVector &locations) noexcept;
    void PrepareSingleLocationRmwScratch(size_t max_count, SingleLocationRmwScratch &scratch) noexcept;
    void UpsertSingleLocationsInto(RequestContext *request_context,
                                   const KeyVector &keys,
                                   const LocationIdRefVector &location_ids,
                                   const CacheLocationVector &locations,
                                   std::vector<ErrorCode> &out_results,
                                   SingleLocationRmwScratch &scratch) noexcept;
    void UpsertSingleLocationsUsingRetainedHandlesInto(RequestContext *request_context,
                                                       const KeyVector &keys,
                                                       const LocationIdRefVector &location_ids,
                                                       CacheLocationVector &locations,
                                                       const std::vector<size_t> &read_indices,
                                                       std::vector<ErrorCode> &out_results,
                                                       SingleLocationRmwScratch &scratch) noexcept;
    std::vector<ErrorCode> Delete(RequestContext *request_context, const KeyVector &keys) noexcept;
    std::vector<ErrorCode> Delete(RequestContext *request_context,
                                  const KeyVector &keys,
                                  const LocationIdsPerKey &location_ids,
                                  int32_t &out_reclaimed_count) noexcept;
    std::vector<ErrorCode> DeleteLocationsForMaintenance(RequestContext *request_context,
                                                         const KeyVector &keys,
                                                         const LocationIdsPerKey &location_ids,
                                                         int32_t &out_reclaimed_count) noexcept;

    // ----- Read APIs -----
    std::vector<ErrorCode> Get(RequestContext *request_context,
                               const KeyVector &keys,
                               CacheLocationMapVector &out_locations,
                               PropertyMapVector &out_properties) noexcept;
    std::vector<ErrorCode> GetLocations(RequestContext *request_context,
                                        const KeyVector &keys,
                                        CacheLocationMapVector &out_location_maps) noexcept;
    std::vector<ErrorCode>
    GetLocationValues(RequestContext *request_context, const KeyVector &keys, LocationsPerKey &out_locations) noexcept;
    std::vector<ErrorCode> GetLocationValuesCompact(RequestContext *request_context,
                                                    const KeyType *keys,
                                                    size_t key_count,
                                                    CompactLocationsPerKey &out_locations) noexcept;
    // Read the source-of-truth backend directly without touching the hot cache.
    // Maintenance admission uses this to revalidate a persistent scan result.
    std::vector<ErrorCode> GetLocationsFromPersistent(RequestContext *request_context,
                                                      const KeyVector &keys,
                                                      CacheLocationMapVector &out_location_maps) noexcept;
    // Refresh complete keys from persistent storage into the hot cache before
    // a maintenance RMW. The caller must hold the corresponding shard locks.
    // In single-backend mode this is a no-op.
    std::vector<ErrorCode> RefreshCacheFromPersistent(RequestContext *request_context, const KeyVector &keys) noexcept;
    std::vector<std::vector<ErrorCode>> GetLocations(RequestContext *request_context,
                                                     const KeyVector &keys,
                                                     const LocationIdsPerKey &location_ids,
                                                     LocationsPerKey &out_locations) noexcept;
    std::vector<std::vector<ErrorCode>> GetLocationsForMaintenance(RequestContext *request_context,
                                                                   const KeyVector &keys,
                                                                   const LocationIdsPerKey &location_ids,
                                                                   LocationsPerKey &out_locations) noexcept;
    std::vector<std::vector<ErrorCode>> GetLocationsWithKeyStatus(RequestContext *request_context,
                                                                  const KeyVector &keys,
                                                                  const LocationIdsPerKey &location_ids,
                                                                  LocationsPerKey &out_locations,
                                                                  std::vector<ErrorCode> &out_key_error_codes) noexcept;
    std::vector<ErrorCode> GetSingleLocationsWithKeyStatus(RequestContext *request_context,
                                                           const KeyVector &keys,
                                                           const LocationIdRefVector &location_ids,
                                                           CacheLocationVector &out_locations,
                                                           std::vector<ErrorCode> &out_key_error_codes) noexcept;
    void GetSingleLocationsWithKeyStatusInto(RequestContext *request_context,
                                             const KeyVector &keys,
                                             const LocationIdRefVector &location_ids,
                                             CacheLocationVector &out_locations,
                                             std::vector<ErrorCode> &out_key_error_codes,
                                             std::vector<ErrorCode> &out_results,
                                             SingleLocationRmwScratch &scratch,
                                             bool retain_handles = false) noexcept;
    void GetSingleLocationViewsWithKeyStatusInto(RequestContext *request_context,
                                                 const KeyVector &keys,
                                                 const LocationIdRefVector &location_ids,
                                                 CacheLocationViewVector &out_locations,
                                                 std::vector<ErrorCode> &out_key_error_codes,
                                                 std::vector<ErrorCode> &out_results,
                                                 SingleLocationRmwScratch &scratch) noexcept;
    std::vector<ErrorCode> GetLocationIds(RequestContext *request_context,
                                          const KeyVector &keys,
                                          LocationIdsPerKey &out_location_ids) noexcept;
    std::vector<ErrorCode> GetProperties(RequestContext *request_context,
                                         const KeyVector &keys,
                                         const std::vector<std::string> &field_names,
                                         PropertyMapVector &out_properties) noexcept;
    std::vector<ErrorCode>
    Exists(RequestContext *request_context, const KeyVector &keys, std::vector<bool> &out_is_exist_vec) noexcept;

    // ----- Cross-batch APIs (no shard locks) -----
    ErrorCode ListKeys(RequestContext *request_context,
                       const std::string &cursor,
                       const int64_t limit,
                       std::string &out_next_cursor,
                       KeyTypeVec &out_keys) noexcept;
    // Scan the in-memory cache when dual-backend metadata is configured;
    // single-backend deployments scan their only persistent backend.
    ErrorCode ScanLocationsForMaintenance(RequestContext *request_context,
                                          const std::string &cursor,
                                          int64_t limit,
                                          MaintenanceScanBatch &out) noexcept;
    ErrorCode RandomSample(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept;
    ErrorCode SampleReclaimKeys(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept;
    ErrorCode SampleReclaimCandidates(RequestContext *request_context,
                                      int64_t count,
                                      ReclaimCandidateVector &out_candidates) noexcept;

    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept;

    // Synchronously flush pending writes for the given keys to persistent storage.
    // Returns true on success, false on failure/timeout.
    bool Sync(const KeyVector &keys) noexcept;

    // A persistent-only backend has no synchronous hot view that can expose an
    // accepted maintenance delete to the next same-key RMW. Conservatively
    // retain the trailing barrier there (it is a no-op for synchronous
    // backends). Cached mode updates the hot view under the shard fence and
    // preserves persistent write order, so it can unlock without that barrier.
    bool RequiresMaintenancePostDeleteSync() const noexcept { return cache_backend_ == nullptr; }

    // Returns async write stats from persistent backend.
    MetaStorageBackend::AsyncWriteStats GetAsyncWriteStats() noexcept;

    size_t GetMemUsage() const noexcept;
    int64_t GetOldestAccessTime() const noexcept;

    // Set revisit interval histogram for cache backend (optional, for metrics tracking).
    void SetRevisitHistogram(std::shared_ptr<RevisitIntervalHistogram> histogram);

    // Only the single local backend is safe and useful to fan out: its cache
    // and items are independently sharded/locked and it ignores RequestContext.
    // Redis and cached modes retain their existing batched request semantics.
    bool SupportsConcurrentLocationValueReads() const noexcept;
    bool SupportsSingleLocationRmw() const noexcept;
    bool GetPureLocalCacheHashSeed(uint32_t &out_hash_seed) const noexcept;

private:
    void AsyncRecoverTask() noexcept;
    int64_t BackfillKeysToCache(const KeyTypeVec &keys,
                                const CacheLocationMapVector &locations,
                                const PropertyMapVector &properties,
                                const std::vector<ErrorCode> &get_error_codes,
                                // Reports whether every source entry and
                                // conditional cache write completed safely.
                                bool *out_success = nullptr) noexcept;
    // Hydrate missing keys from persistent into cache during Recover. Returns
    // false when a backend violates the positional response contract or the
    // full pre-update value cannot be made available safely.
    bool EnsureKeyInCache(RequestContext *request_context, const KeyTypeVec &keys) noexcept;
    // Delete keys that have no remaining location fields. Returns reclaimed count.
    int32_t MaybeReclaimEmptyKeys(RequestContext *request_context,
                                  const KeyVector &keys,
                                  const std::vector<ErrorCode> &delete_results) noexcept;
    std::string instance_id_;
    std::unique_ptr<MetaStorageBackend> persistent_backend_;
    std::unique_ptr<MetaCacheBaseBackend> cache_backend_;

    std::atomic<RecoverState> recover_state_{RecoverState::kRecover};
    std::atomic<bool> is_closed_{false};
    std::thread recover_thread_;
    // Serializes lifecycle transitions and prevents assigning a second
    // recovery thread over an already-joinable std::thread (which would call
    // std::terminate even though Open() is noexcept).
    mutable std::mutex lifecycle_mutex_;
    bool opened_ = false;

    mutable std::mutex deleted_keys_mutex_;
    std::unordered_set<KeyType> deleted_keys_;
};

} // namespace kv_cache_manager
