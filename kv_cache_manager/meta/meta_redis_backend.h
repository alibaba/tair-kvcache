#pragma once

#include <cassert>
#include <condition_variable>
#include <mutex>
#include <queue>

#include "kv_cache_manager/common/client_pool.h"
#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"

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
    ErrorCode
    RandomSample(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept override;
    ErrorCode
    SampleReclaimKeys(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept override;

    // ----- Metadata APIs -----
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;

private:
    // virtual for test
    virtual std::shared_ptr<RedisClient> CreateRedisClient() const;

private:
    std::shared_ptr<DynamicClientPool<RedisClient>> client_pool_;
    StandardUri storage_uri_;
    std::string instance_id_;
    std::string cache_key_prefix_;
    std::string metadata_key_;
    int64_t timeout_ms_ = 1000;
};
} // namespace kv_cache_manager
