#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/meta_local_base_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

class MetaStorageBackendConfig;
class RequestContext;

// Backend orchestrator with two modes (auto-selected at Init):
//   * Dual-backend: persistent (source-of-truth) + local (hot cache).
//     Writes go persistent-first then local; reads are local-first,
//     falling back to persistent during Recover.
//   * Single-backend: persistent only (local is null, no Recover).
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

    ErrorCode Init(const std::string &instance_id,
                   const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept;

    ErrorCode Open() noexcept;
    ErrorCode Close() noexcept;

    RecoverState GetRecoverState() const noexcept { return recover_state_.load(std::memory_order_acquire); }

    // ----- Write APIs -----
    // Put / UpdateFields / Upsert merge CacheLocations into batch.batch_properties in place.
    std::vector<ErrorCode> Put(RequestContext *request_context, BatchMetaData &batch) noexcept;
    std::vector<ErrorCode> UpdateFields(RequestContext *request_context, BatchMetaData &batch) noexcept;
    std::vector<ErrorCode> Upsert(RequestContext *request_context, BatchMetaData &batch) noexcept;

    // Delete entire block keys.
    std::vector<ErrorCode> Delete(const KeyVector &keys) noexcept;

    // Delete specific location fields within each key.
    std::vector<ErrorCode> Delete(const KeyVector &keys,
                                  const LocationIdsPerKey &location_ids,
                                  int32_t &out_reclaimed_count) noexcept;

    // ----- Read APIs -----
    std::vector<ErrorCode> Get(const KeyVector &keys,
                               const std::vector<std::string> &field_names,
                               FieldMapVec &out_field_maps) noexcept;
    std::vector<ErrorCode> GetLocations(RequestContext *request_context,
                                        const KeyVector &keys,
                                        LocationMapVector &out_location_maps) noexcept;
    std::vector<std::vector<ErrorCode>> GetLocations(RequestContext *request_context,
                                                     const KeyVector &keys,
                                                     const LocationIdsPerKey &location_ids,
                                                     LocationsPerKey &out_locations) noexcept;
    std::vector<ErrorCode> GetAllFields(const KeyVector &keys, FieldMapVec &out_field_maps) noexcept;
    std::vector<ErrorCode> Exists(const KeyVector &keys, std::vector<bool> &out_is_exist_vec) noexcept;

    // ----- Cross-batch APIs (no shard locks) -----
    ErrorCode ListKeys(const std::string &cursor,
                       const int64_t limit,
                       std::string &out_next_cursor,
                       KeyTypeVec &out_keys) noexcept;
    ErrorCode RandomSample(const int64_t count, KeyTypeVec &out_keys) noexcept;
    ErrorCode SampleReclaimKeys(const int64_t count, KeyTypeVec &out_keys) noexcept;

    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept;

    size_t GetMemUsage() const noexcept;

private:
    void AsyncRecoverTask() noexcept;
    int64_t BackfillKeysToLocal(const KeyTypeVec &keys,
                                const FieldMapVec &field_maps,
                                const std::vector<ErrorCode> &get_error_codes) noexcept;
    // Hydrate missing keys from persistent into local during Recover.
    void EnsureKeyInLocal(const KeyTypeVec &keys) noexcept;
    // Merge CacheLocations into batch.batch_properties in place. Accumulates
    // serialization latency onto `request_context` when non-null.
    PropertyMapVector &BuildEffectiveFieldMaps(RequestContext *request_context,
                                               BatchMetaData &batch) const noexcept;
    static std::string MakeLocationFieldName(const std::string &location_id) noexcept;
    // Delete keys that have no remaining location fields. Returns reclaimed count.
    int32_t MaybeReclaimEmptyKeys(const KeyVector &keys, const std::vector<ErrorCode> &delete_results) noexcept;

    std::string instance_id_;
    std::unique_ptr<MetaStorageBackend> persistent_backend_;
    std::unique_ptr<MetaLocalBaseBackend> local_backend_;

    std::atomic<RecoverState> recover_state_{RecoverState::kRecover};
    std::atomic<bool> is_closed_{false};
    std::thread recover_thread_;

    mutable std::mutex deleted_keys_mutex_;
    std::unordered_set<KeyType> deleted_keys_;
};

} // namespace kv_cache_manager
