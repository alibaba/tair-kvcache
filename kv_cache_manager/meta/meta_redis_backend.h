#pragma once

#include <cassert>
#include <condition_variable>
#include <mutex>
#include <queue>

#include "kv_cache_manager/common/client_pool.h"
#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"
#include "kv_cache_manager/meta/meta_store_health_tracker.h"

namespace kv_cache_manager {

/*
[instance_id]:cache_key_set ---> [key1, key2, key3]

[instance_id]:cache_key:[key1] --- > {{f1, v1-1}, {f2, v1-2}}
[instance_id]:cache_key:[key2] --- > {{f1, v2-1}, {f2, v2-2}}
 */
class MetaRedisBackend : public MetaStorageBackend {
public:
    MetaRedisBackend() = default;
    ~MetaRedisBackend() override;

    std::string GetStorageType() noexcept override;

    // False once the Redis instance behind this backend is detected down, i.e.
    // writes can no longer be persisted. See MetaStorageBackend. While
    // degraded this also drives a rate limited probe, because a synchronous
    // backend that has been gated off no longer issues writes that could
    // report recovery.
    bool IsStoreAvailable() noexcept override;

    ErrorCode Init(const std::string &instance_id,
                   const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept override;
    ErrorCode Open() noexcept override;
    ErrorCode Close() noexcept override;

    // ----- Write APIs -----
    std::vector<ErrorCode> Put(RequestContext *request_context,
                               const KeyTypeVec &keys,
                               const CacheLocationMapVector &locations,
                               const PropertyMapVector &properties) noexcept override;
    std::vector<ErrorCode> Upsert(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &locations,
                                  const PropertyMapVector &properties) noexcept override;
    std::vector<ErrorCode> Delete(RequestContext *request_context, const KeyTypeVec &keys) noexcept override;
    std::vector<ErrorCode> DeleteLocations(RequestContext *request_context,
                                           const KeyTypeVec &keys,
                                           const LocationIdsPerKey &location_ids) noexcept override;

    // ----- Read APIs -----
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
                       KeyTypeVec &out_keys) noexcept override;
    ErrorCode ScanLocationsForMaintenance(RequestContext *request_context,
                                          const std::string &cursor,
                                          int64_t limit,
                                          MaintenanceScanBatch &out) noexcept override;
    ErrorCode
    RandomSample(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept override;
    ErrorCode
    SampleReclaimKeys(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept override;
    ErrorCode SampleReclaimCandidates(RequestContext *request_context,
                                      int64_t count,
                                      ReclaimCandidateVector &out_candidates,
                                      bool require_read_success = false) noexcept override;

    // ----- Metadata APIs -----
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;
    ErrorCode DeleteMetaData() noexcept override;

private:
    // virtual for test
    virtual std::shared_ptr<RedisClient> CreateRedisClient() const;

    // Feeds the outcome of one synchronous store operation into store_health_.
    void RecordStoreOpResult(ErrorCode ec) noexcept;
    // Same for a per key result vector: one unreachable entry is enough.
    std::vector<ErrorCode> RecordStoreOpResults(std::vector<ErrorCode> results) noexcept;

    // Hands out a pooled client and records the outcome in store_health_. An
    // unreachable store usually surfaces here first -- AcquireClient times out
    // before a single command is issued -- so routing every call through this
    // helper is what lets a synchronous backend notice the outage at all.
    ClientPool<RedisClient>::ClientHandle AcquireStoreClient() noexcept;

private:
    std::shared_ptr<DynamicClientPool<RedisClient>> client_pool_;
    StandardUri storage_uri_;
    std::string instance_id_;
    std::string cache_key_prefix_;
    std::string metadata_key_;
    int64_t timeout_ms_ = 1000;

    // Durability signal for the Redis instance behind this backend.
    MetaStoreHealthTracker store_health_;
};
} // namespace kv_cache_manager
