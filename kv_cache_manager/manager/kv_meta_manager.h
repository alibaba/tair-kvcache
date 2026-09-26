#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/kv_meta_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

class CacheManager;
class CacheLocation;
class DataStorageSelector;
class InstanceInfo;
class KvMetaReclaimer;
class KvMetaWriteSessionManager;
class RegistryManager;
class RequestContext;

// Generic, exact-key object metadata path used by embedding and other opaque
// values.  It deliberately does not call StartWriteCache/FinishWriteCache and
// never changes the fixed-size KV-cache allocation path.
class KvMetaManager {
public:
    struct Limits {
        std::size_t max_batch_items = 64;
        std::size_t max_key_bytes = 512;
        std::size_t max_instance_id_bytes = 512;
        std::size_t max_instance_group_bytes = 512;
        std::size_t max_write_session_id_bytes = 512;
        std::size_t max_user_data_bytes = 64 * 1024;
        std::size_t max_location_uri_bytes = kMaxKvMetaLocationUriBytes;
        std::size_t max_active_write_sessions = 4096;
        std::uint64_t max_value_bytes = 1ULL * 1024 * 1024 * 1024;
        std::uint64_t max_batch_bytes = 4ULL * 1024 * 1024 * 1024;
        std::int64_t max_write_timeout_seconds = kKvMetaMaxWriteTimeoutSeconds;
        // Upper bound accepted from a backend that must quarantine a failed
        // remote write before releasing its reusable allocation. This is
        // server-side safety time and does not extend the client's commit
        // deadline. The default covers PACE's 180-second quarantine contract.
        std::int64_t max_failed_write_cleanup_grace_seconds = kKvMetaMaxFailedWriteCleanupGraceSeconds;
    };

    struct ValueLocation {
        DataStorageType type = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
        std::uint64_t value_size = 0;
        std::vector<std::pair<std::string, std::string>> specs;
    };

    struct GetResult {
        bool found = false;
        ValueLocation location;
    };

    struct StartWriteResult {
        std::string write_session_id;
        // Exact compact-session cardinality recorded only after the session
        // is published. This is internal rollback metadata and is never sent
        // on the wire.
        std::size_t session_item_count = 0;
        // Request-aligned. true means the caller must not write this key.
        std::vector<bool> key_mask;
        // Only entries whose key_mask is false, in request-relative order.
        std::vector<ValueLocation> locations;
    };

    KvMetaManager(std::shared_ptr<CacheManager> cache_manager, std::shared_ptr<RegistryManager> registry_manager);
    KvMetaManager(std::shared_ptr<CacheManager> cache_manager,
                  std::shared_ptr<RegistryManager> registry_manager,
                  Limits limits);
    ~KvMetaManager();

    KvMetaManager(const KvMetaManager &) = delete;
    KvMetaManager &operator=(const KvMetaManager &) = delete;

    bool Init();
    void Shutdown();
    // Reconciles stale generic-object writes after CacheManager has recreated
    // all indexers. This only scans instances in the reserved KVMeta namespace.
    ErrorCode DoRecover(std::function<bool()> should_abort = nullptr);
    // Stops the expiry worker and forgets in-memory sessions. Their invisible
    // active metadata is reclaimed by the next leader's isolated recovery;
    // demotion never walks an unbounded session set on the main cleanup path.
    void DoCleanup();
    // Non-blockingly cancels Trim, closes new session admission, and wakes the
    // expiry worker. Server demotion can therefore finish the existing
    // KV-cache drain/GC/migration sequence before joining KVMeta workers.
    void CancelMaintenance() noexcept;
    // Called only after a successful leader recovery. It also restarts the
    // write-session expiry worker stopped by DoCleanup.
    bool ResumeMaintenance();

    std::pair<ErrorCode, std::string> RegisterInstance(RequestContext *request_context,
                                                       const std::string &instance_group,
                                                       const std::string &instance_id,
                                                       const std::string &user_data);

    std::pair<ErrorCode, std::shared_ptr<const InstanceInfo>> GetInstanceInfo(RequestContext *request_context,
                                                                              const std::string &instance_id) const;

    std::pair<ErrorCode, std::vector<GetResult>>
    Get(RequestContext *request_context, const std::string &instance_id, const std::vector<std::string> &keys) const;

    std::pair<ErrorCode, StartWriteResult> StartWrite(RequestContext *request_context,
                                                      const std::string &instance_id,
                                                      const std::vector<std::string> &keys,
                                                      const std::vector<std::uint64_t> &value_sizes,
                                                      std::int64_t write_timeout_seconds);

    ErrorCode FinishWrite(RequestContext *request_context,
                          const std::string &instance_id,
                          const std::string &write_session_id,
                          const std::vector<bool> &success_keys);

    ErrorCode
    Remove(RequestContext *request_context, const std::string &instance_id, const std::vector<std::string> &keys);

    ErrorCode TrimAll(RequestContext *request_context, const std::string &instance_id, bool metadata_only);

    const Limits &limits() const noexcept { return limits_; }

private:
    struct SessionItem;
    struct ExactLocation;
    struct DeleteItemsOptions {
        // Physical deletion is authorized only after this call conditionally
        // replaces the exact owner with a durable generation-bearing
        // tombstone. An already-absent record is never sufficient ownership
        // proof for a reusable backend address.
        bool delete_physical = true;
        bool adjust_storage_usage = true;
        bool maintenance_no_touch = false;
        // Reclaimer retry is the sole expected already-absent case: its
        // in-memory pending record fences successor generations while the
        // previous metadata Sync is retried.
        bool sync_metadata_absent = false;
        bool restore_usage_on_sync_failure = true;
    };
    struct DeleteItemsResult {
        ErrorCode ec = EC_OK;
        bool metadata_outcome_changed = false;
        bool metadata_cleanup_complete = false;
        // True only when every unfinished item has a durable CLS_DELETING
        // owner and the online Reclaimer atomically accepted responsibility
        // for retrying physical deletion and metadata finalization. Callers
        // may keep KVMeta admission open in this one incomplete case.
        bool online_cleanup_owned = false;
        bool metadata_already_absent = false;
        bool metadata_owner_conflicted = false;
        // Request-aligned evidence used by the Reclaimer to distinguish its
        // own post-delete Sync retry from an unexpected missing or replaced
        // owner.
        std::vector<bool> metadata_deleted;
        std::vector<bool> metadata_absent;
        std::vector<bool> metadata_conflicted;
    };

    static std::string InternalInstanceId(const std::string &instance_id);
    static std::int64_t InternalKey(const std::string &key);
    static std::string StableLocationId(const std::string &key);

    bool IsOwnedLocation(std::int64_t internal_key, const std::string &location_id) const;
    ErrorCode ValidateOwnedLocation(RequestContext *request_context,
                                    const std::string &internal_instance_id,
                                    std::int64_t internal_key,
                                    const std::string &location_id,
                                    const CacheLocation &location,
                                    std::uint64_t &value_size) const;
    ErrorCode ValidateInstanceId(RequestContext *request_context, const std::string &instance_id) const;
    std::pair<ErrorCode, std::shared_ptr<const InstanceInfo>>
    GetValidatedInstanceInfo(RequestContext *request_context, const std::string &instance_id) const;
    ErrorCode ValidateKeys(RequestContext *request_context, const std::vector<std::string> &keys) const;
    ErrorCode ValidateCacheConfiguration(RequestContext *request_context, const std::string &instance_group) const;
    ErrorCode CheckDynamicByteAdmission(RequestContext *request_context,
                                        const std::string &instance_group,
                                        DataStorageType storage_type,
                                        std::uint64_t requested_bytes) const;
    ErrorCode LoadExactLocations(RequestContext *request_context,
                                 const std::string &internal_instance_id,
                                 const std::vector<std::string> &keys,
                                 std::vector<ExactLocation> &out) const;
    DeleteItemsResult DeleteItems(RequestContext *request_context,
                                  const std::string &internal_instance_id,
                                  const std::vector<SessionItem> &items,
                                  const DeleteItemsOptions &options);
    DeleteItemsResult DeleteRetiredMetadata(RequestContext *request_context,
                                            const std::string &internal_instance_id,
                                            const std::vector<SessionItem> &items);
    ErrorCode FinishWriteInternal(RequestContext *request_context,
                                  const std::string &internal_instance_id,
                                  const std::vector<bool> &success_keys,
                                  const std::vector<SessionItem> &items);
    ErrorCode DeleteStorageUris(RequestContext *request_context,
                                const std::string &storage_name,
                                const std::vector<DataStorageUri> &uris) const;
    ErrorCode DeleteAllocatedLocations(RequestContext *request_context, const std::vector<SessionItem> &items) const;
    void RememberKvMetaGroup(const std::string &instance_group);
    std::vector<std::string> SnapshotKvMetaGroups() const;
    void ReplaceKvMetaGroups(std::unordered_set<std::string> instance_groups);

private:
    friend class KvMetaReclaimer;
    friend class KvMetaWriteSessionManager;
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<RegistryManager> registry_manager_;
    Limits limits_;
    std::unique_ptr<DataStorageSelector> data_storage_selector_;
    std::unique_ptr<KvMetaReclaimer> reclaimer_;
    std::unique_ptr<KvMetaWriteSessionManager> write_session_manager_;
    mutable std::mutex registration_mutex_;
    // Reclaimer rounds must never enumerate every ordinary KV-cache group.
    // Registration and leader recovery are the only supported discovery
    // points for the reserved KVMeta instance schema, so retain that bounded
    // set and keep periodic maintenance entirely on the side path.
    mutable std::mutex kv_meta_groups_mutex_;
    std::unordered_set<std::string> kv_meta_groups_;
    // Serializes exact-byte admission and short metadata transitions within a
    // KVMeta group. Long Trim scans publish a per-instance marker under this
    // shard and then release it, so unrelated instances never wait behind
    // unbounded scan or storage I/O. The existing cache path never takes
    // these side-path-only locks.
    mutable std::array<std::mutex, 64> quota_admission_mutexes_;
    // Each set is accessed only while holding the matching shard above.
    std::array<std::unordered_set<std::string>, 64> trimming_instances_;
    // One promotion/recovery epoch has one bounded force deadline. Transient
    // physical cleanup failures must not restart the full old-writer grace on
    // every retry; successful recovery or demotion clears the epoch.
    mutable std::mutex recovery_window_mutex_;
    std::optional<std::chrono::steady_clock::time_point> recovery_force_deadline_;
    std::atomic<bool> maintenance_cancelled_{false};
    std::atomic<bool> initialized_{false};
};

} // namespace kv_cache_manager
