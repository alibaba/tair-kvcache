#pragma once

#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

class CacheManager;
class CacheLocation;
class DataStorageSelector;
class InstanceInfo;
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
        std::size_t max_user_data_bytes = 64 * 1024;
        std::size_t max_active_write_sessions = 4096;
        std::uint64_t max_value_bytes = 1ULL * 1024 * 1024 * 1024;
        std::uint64_t max_batch_bytes = 4ULL * 1024 * 1024 * 1024;
        std::int64_t max_write_timeout_seconds = 1800;
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
        // Request-aligned. true means the caller must not write this key.
        std::vector<bool> key_mask;
        // Only entries whose key_mask is false, in request-relative order.
        std::vector<ValueLocation> locations;
    };

    KvMetaManager(std::shared_ptr<CacheManager> cache_manager,
                  std::shared_ptr<RegistryManager> registry_manager);
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
    // Trim can span an arbitrary number of objects. Server demotion cancels
    // it before waiting for KVMeta RPCs, so this optional side path cannot
    // indefinitely delay cleanup of the existing KV-cache service.
    void CancelMaintenance() noexcept;
    // Called only after a successful leader recovery. It also restarts the
    // write-session expiry worker stopped by DoCleanup.
    bool ResumeMaintenance();

    std::pair<ErrorCode, std::string> RegisterInstance(RequestContext *request_context,
                                                       const std::string &instance_group,
                                                       const std::string &instance_id,
                                                       const std::string &user_data);

    std::pair<ErrorCode, std::shared_ptr<const InstanceInfo>>
    GetInstanceInfo(RequestContext *request_context, const std::string &instance_id) const;

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

    ErrorCode Remove(RequestContext *request_context,
                     const std::string &instance_id,
                     const std::vector<std::string> &keys);

    ErrorCode TrimAll(RequestContext *request_context, const std::string &instance_id, bool metadata_only);

    const Limits &limits() const noexcept { return limits_; }

private:
    struct SessionItem;
    struct ExactLocation;

    static std::string InternalInstanceId(const std::string &instance_id);
    static std::int64_t InternalKey(const std::string &key);
    static std::string StableLocationId(const std::string &key);

    bool IsOwnedLocation(std::int64_t internal_key, const std::string &location_id) const;
    ErrorCode ValidateOwnedLocation(RequestContext *request_context,
                                    std::int64_t internal_key,
                                    const std::string &location_id,
                                    const CacheLocation &location,
                                    std::uint64_t &value_size) const;
    ErrorCode ValidateInstanceId(RequestContext *request_context, const std::string &instance_id) const;
    std::pair<ErrorCode, std::shared_ptr<const InstanceInfo>>
    GetValidatedInstanceInfo(RequestContext *request_context, const std::string &instance_id) const;
    ErrorCode ValidateKeys(RequestContext *request_context, const std::vector<std::string> &keys) const;
    ErrorCode CheckDynamicByteAdmission(RequestContext *request_context,
                                        const std::string &instance_group,
                                        DataStorageType storage_type,
                                        std::uint64_t requested_bytes) const;
    ErrorCode LoadExactLocations(RequestContext *request_context,
                                 const std::string &internal_instance_id,
                                 const std::vector<std::string> &keys,
                                 std::vector<ExactLocation> &out) const;
    ErrorCode DeleteItems(RequestContext *request_context,
                          const std::string &internal_instance_id,
                          const std::vector<SessionItem> &items,
                          bool metadata_only,
                          bool adjust_storage_usage = true);
    ErrorCode FinishWriteInternal(RequestContext *request_context,
                                  const std::string &internal_instance_id,
                                  const std::vector<bool> &success_keys,
                                  const std::vector<SessionItem> &items);
    ErrorCode DeleteAllocatedLocations(RequestContext *request_context,
                                       const std::vector<SessionItem> &items) const;

private:
    friend class KvMetaWriteSessionManager;
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<RegistryManager> registry_manager_;
    Limits limits_;
    std::unique_ptr<DataStorageSelector> data_storage_selector_;
    std::unique_ptr<KvMetaWriteSessionManager> write_session_manager_;
    mutable std::mutex registration_mutex_;
    // Serializes exact-byte admission and bounded Trim within a KVMeta group.
    // The existing cache path never takes these side-path-only locks.
    mutable std::array<std::mutex, 64> quota_admission_mutexes_;
    std::atomic<bool> maintenance_cancelled_{false};
    std::atomic<bool> initialized_{false};
};

} // namespace kv_cache_manager
