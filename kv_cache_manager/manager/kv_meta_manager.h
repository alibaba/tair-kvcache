#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/kv_meta_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

class CacheLocation;
class CacheManager;
class DataStorageSelector;
class InstanceInfo;
class KvMetaReclaimer;
class KvMetaWriteSessionManager;
class RegistryManager;
class RequestContext;

// Metadata and lifecycle management for variable-size cache objects. This is
// intentionally independent from the fixed-block KV-cache write path.
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
        // Request-aligned. true means an equal-size committed object already exists.
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
    ErrorCode DoRecover();
    void DoCleanup();
    void CancelMaintenance() noexcept;
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

    static std::string InternalInstanceId(const std::string &instance_id);
    static std::int64_t InternalKey(const std::string &key);
    static std::string StableLocationId(const std::string &key);

    ErrorCode ValidateInstanceId(RequestContext *request_context, const std::string &instance_id) const;
    ErrorCode CheckReady(RequestContext *request_context) const;
    ErrorCode ValidateKeys(RequestContext *request_context, const std::vector<std::string> &keys) const;
    ErrorCode ValidateCacheConfiguration(RequestContext *request_context, const std::string &instance_group) const;
    std::pair<ErrorCode, std::shared_ptr<const InstanceInfo>>
    GetValidatedInstanceInfo(RequestContext *request_context, const std::string &instance_id) const;
    ErrorCode ValidateLocation(RequestContext *request_context,
                               std::int64_t internal_key,
                               const std::string &location_id,
                               const CacheLocation &location,
                               std::uint64_t &value_size) const;
    ErrorCode LoadExactLocations(RequestContext *request_context,
                                 const std::string &internal_instance_id,
                                 const std::vector<std::string> &keys,
                                 std::vector<ExactLocation> &out) const;
    ErrorCode DeleteItems(RequestContext *request_context,
                          const std::string &internal_instance_id,
                          const std::vector<SessionItem> &items,
                          bool delete_physical,
                          bool maintenance_read,
                          std::vector<SessionItem> *deleted_items = nullptr);
    void DeletePhysicalBestEffort(RequestContext *request_context, const std::vector<SessionItem> &items) const;
    void ExpireSession(const std::string &session_id,
                       const std::string &internal_instance_id,
                       const std::vector<SessionItem> &items) noexcept;

private:
    friend class KvMetaReclaimer;
    friend class KvMetaWriteSessionManager;

    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<RegistryManager> registry_manager_;
    Limits limits_;
    std::unique_ptr<DataStorageSelector> data_storage_selector_;
    std::unique_ptr<KvMetaReclaimer> reclaimer_;
    std::unique_ptr<KvMetaWriteSessionManager> write_session_manager_;

    // Serializes only SERVING transitions and deletes so their byte-counter
    // updates cannot cross. PutStart never enters this lock.
    mutable std::array<std::mutex, 64> metadata_mutation_mutexes_;
    std::atomic<bool> maintenance_cancelled_{false};
    std::atomic<bool> recovery_complete_{false};
    std::atomic<bool> initialized_{false};
};

} // namespace kv_cache_manager
