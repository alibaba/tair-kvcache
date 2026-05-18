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
    std::vector<ErrorCode> Delete(RequestContext *request_context, const KeyVector &keys) noexcept;
    std::vector<ErrorCode> Delete(RequestContext *request_context,
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
    std::vector<std::vector<ErrorCode>> GetLocations(RequestContext *request_context,
                                                     const KeyVector &keys,
                                                     const LocationIdsPerKey &location_ids,
                                                     LocationsPerKey &out_locations) noexcept;
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
    ErrorCode RandomSample(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept;
    ErrorCode SampleReclaimKeys(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept;

    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept;

    // Synchronously flush pending writes for the given keys to persistent storage.
    // Returns true on success, false on failure/timeout.
    bool Sync(const KeyVector &keys) noexcept;

    // Returns per-queue pending key sizes from persistent backend.
    std::vector<int64_t> GetAsyncQueueSizes() const noexcept;

    size_t GetMemUsage() const noexcept;
    int64_t GetOldestAccessTime() const noexcept;

private:
    void AsyncRecoverTask() noexcept;
    int64_t BackfillKeysToCache(const KeyTypeVec &keys,
                                const CacheLocationMapVector &locations,
                                const PropertyMapVector &properties,
                                const std::vector<ErrorCode> &get_error_codes) noexcept;
    // Hydrate missing keys from persistent into cache during Recover.
    void EnsureKeyInCache(RequestContext *request_context, const KeyTypeVec &keys) noexcept;
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

    mutable std::mutex deleted_keys_mutex_;
    std::unordered_set<KeyType> deleted_keys_;
};

} // namespace kv_cache_manager
