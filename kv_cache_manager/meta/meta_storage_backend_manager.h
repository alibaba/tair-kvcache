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

// Backend orchestrator. Operates in two modes selected automatically at Init:
//
//   * Dual-backend mode (default): persistent_backend_ is the source-of-truth,
//     local_backend_ is an in-memory hot cache. Every write goes
//     persistent-first, then local as a conditional write keyed by the
//     persistent error codes. Reads are local-first, falling back to
//     persistent during async Recover.
//
//   * Single-backend mode: only persistent_backend_ exists (local_backend_ is
//     null). Every read/write is forwarded directly to persistent_backend_;
//     no recover thread is started and recover_state_ stays kRunning. This
//     mode is selected when the storage URI carries no persistent_type /
//     cache_type params, in which case the backend is built straight from
//     `config->GetStorageType()` via MetaStorageBackendFactory::
//     CreateAndInitStorageBackend.
//
// Lifecycle (dual-backend mode):
//   * Recover: a background thread scans persistent and back-fills local via
//     PutIfAbsent; reads fall back to persistent on miss; deletes record a
//     tombstone so concurrent backfill cannot resurrect removed keys;
//     partial-update writes (UpdateFields / Upsert / per-location Delete)
//     first hydrate the affected keys into local from persistent so the
//     conditional mirror write does not lose pre-restart fields.
//   * Running: reads come from local only.
//
// CacheLocation handling: if batch.batch_locations is populated, every
// CacheLocation is JSON-serialised into the per-key PropertyMap under the
// field name "__loc__{storage_type}" (storage_type = int(CacheLocation::type())).
//
// Callers are responsible for partitioning requests into BatchMetaData
// (typically via MetaIndexer::MakeBatches) and for acquiring the shard mutexes
// named by batch_shard_indexs before invoking the manager; the manager itself
// never grabs shard mutexes.
class MetaStorageBackendManager {
public:
    enum class RecoverState {
        kRecover,
        kRunning,
    };

    MetaStorageBackendManager() = default;
    ~MetaStorageBackendManager();

    // Persistent and cache backends are parsed out of `config->GetStorageUri()`
    // (params: persistent_type / cache_type), defaulting to redis / local.
    ErrorCode Init(const std::string &instance_id,
                   const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept;

    ErrorCode Open() noexcept;
    ErrorCode Close() noexcept;

    RecoverState GetRecoverState() const noexcept { return recover_state_.load(std::memory_order_acquire); }

    // ----- Write APIs -----
    //
    // Put / UpdateFields / Upsert take `batch` by non-const reference because
    // they merge serialised CacheLocations into batch.batch_properties in place
    // to avoid copying the per-key PropertyMaps. On return, batch.batch_properties
    // holds the merged maps actually written to the backends.
    std::vector<ErrorCode> Put(BatchMetaData &batch) noexcept;
    std::vector<ErrorCode> UpdateFields(BatchMetaData &batch) noexcept;
    std::vector<ErrorCode> Upsert(BatchMetaData &batch) noexcept;

    std::vector<ErrorCode> Delete(const BatchMetaData &batch) noexcept;

    // Delete the per-key location fields identified by `location_ids`. The
    // backend exposes no field-level delete primitive, so the targeted fields
    // are overwritten with an empty string and readers treat an empty value
    // as "removed".
    std::vector<ErrorCode> Delete(const BatchMetaData &batch,
                                  const std::vector<std::string> &location_ids) noexcept;

    // ----- Read APIs -----
    // Local-first; during Recover misses fall back to persistent.
    std::vector<ErrorCode> Get(const KeyVector &keys,
                               const std::vector<std::string> &field_names,
                               FieldMapVec &out_field_maps) noexcept;
    std::vector<ErrorCode>
    GetLocations(const KeyVector &keys, LocationMapVector &out_location_maps) noexcept;
    std::vector<std::vector<ErrorCode>> GetLocations(const KeyVector &keys,
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

private:
    // Background scan of persistent -> PutIfAbsent into local. Transitions
    // recover_state_ to kRunning on completion.
    void AsyncRecoverTask() noexcept;

    // Returns the number of keys back-filled into local, skipping tombstoned keys.
    int64_t BackfillKeysToLocal(const KeyTypeVec &keys,
                                const FieldMapVec &field_maps,
                                const std::vector<ErrorCode> &get_error_codes) noexcept;

    // Hydrate `keys` that are missing from local by pulling all fields from
    // persistent and Put-ing them into local. Used by partial-update writes
    // (UpdateFields / Upsert / per-location Delete) during Recover so that
    // the subsequent conditional mirror write sees the complete field set
    // and the async backfill cannot later overwrite this write with stale
    // persistent contents. Best-effort: failures are logged, not surfaced.
    void EnsureKeyInLocal(const KeyTypeVec &keys) noexcept;

    // Merge serialised CacheLocations from batch.batch_locations into
    // batch.batch_properties in place (resizing batch_properties to match
    // batch_keys when empty). Returns a reference to batch.batch_properties
    // for call-site convenience. No copies of existing maps are made.
    PropertyMapVector &BuildEffectiveFieldMaps(BatchMetaData &batch) const noexcept;

    // "__loc__{int(storage_type)}"
    static std::string MakeLocationFieldName(const std::string &location_id) noexcept;

    std::string instance_id_;
    std::unique_ptr<MetaStorageBackend> persistent_backend_;
    std::unique_ptr<MetaLocalBaseBackend> local_backend_;

    std::atomic<RecoverState> recover_state_{RecoverState::kRecover};
    std::atomic<bool> is_closed_{false};
    std::thread recover_thread_;

    // Tombstones prevent Recover backfill from resurrecting keys that foreground
    // traffic has just deleted.
    mutable std::mutex deleted_keys_mutex_;
    std::unordered_set<KeyType> deleted_keys_;
};

} // namespace kv_cache_manager
