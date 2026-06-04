#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/common/concurrent_hash_map.h"
#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"

namespace kv_cache_manager {

// Per-key storage item for dummy backend.
struct DummyItem {
    CacheLocationMap locations;
    PropertyMap properties;
};

/*
 * MetaDummyBackend is a meta storage backend implementation with
 * in-memory data store, and optional local filesystem persistence
 * capability, whose *sole* purpose is for testing & validating.
 *
 * WARNING:
 * THIS IS A TEST FACILITY WITHOUT ANY FUNCTIONALITY COMPLETENESS
 * WARRANTY; DO NOT EVER USE IT UNDER ANY PRODUCTION CIRCUMSTANCE.
 */
class MetaDummyBackend : public MetaStorageBackend {
public:
    MetaDummyBackend() = default;
    ~MetaDummyBackend() override = default;

    std::string GetStorageType() noexcept override;

    ErrorCode Init(const std::string &instance_id,
                   const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept override;
    ErrorCode Open() noexcept override;
    ErrorCode Close() noexcept override;

    // =====================================================================
    // Write APIs
    // =====================================================================

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

    // =====================================================================
    // Read APIs
    // =====================================================================

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
    std::vector<ErrorCode> ExistsLocation(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          std::vector<bool> &out_exists) noexcept override;
    std::vector<ErrorCode> GetProperties(RequestContext *request_context,
                                         const KeyTypeVec &keys,
                                         const std::vector<std::string> &field_names,
                                         PropertyMapVector &out_properties) noexcept override;

    // =====================================================================
    // Key-level APIs
    // =====================================================================

    std::vector<ErrorCode> Exists(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  std::vector<bool> &out_is_exist_vec) noexcept override;
    ErrorCode ListKeys(RequestContext *request_context,
                       const std::string &cursor,
                       std::int64_t limit,
                       std::string &out_next_cursor,
                       KeyTypeVec &out_keys) noexcept override;
    ErrorCode RandomSample(RequestContext *request_context, std::int64_t count, KeyTypeVec &out_keys) noexcept override;
    ErrorCode
    SampleReclaimKeys(RequestContext *request_context, std::int64_t count, KeyTypeVec &out_keys) noexcept override;

    // =====================================================================
    // Metadata APIs
    // =====================================================================

    ErrorCode PutMetaData(const FieldMap &field_map) noexcept override;
    ErrorCode GetMetaData(FieldMap &out_field_map) noexcept override;

private:
    ErrorCode PersistToPath();

    std::mutex mutex_;
    ConcurrentHashMap<KeyType, DummyItem> table_;
    FieldMap metadata_;
    std::string path_;
    bool enable_persistence_ = false;
};

} // namespace kv_cache_manager
