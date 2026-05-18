#pragma once

#include <gtest/gtest.h>

#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/meta_cache_base_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"

namespace kv_cache_manager {
class MetaStorageBackendTestBase {
protected:
    using KeyType = MetaStorageBackend::KeyType;
    using KeyTypeVec = MetaStorageBackend::KeyTypeVec;
    using FieldMap = MetaStorageBackend::FieldMap;
    using FieldMapVec = MetaStorageBackend::FieldMapVec;
    // PropertyMap / PropertyMapVector are defined in types.h, not inside MetaStorageBackend.
    using PropertyMap = kv_cache_manager::PropertyMap;
    using PropertyMapVector = kv_cache_manager::PropertyMapVector;

    // ---- Compatibility wrappers ----
    // Split a legacy FieldMapVec into CacheLocationMapVector + PropertyMapVector.
    // Fields starting with PROPERTY_LOCATION_PREFIX are treated as locations (value = raw string
    // stored as-is in a minimal CacheLocation), others go to properties.
    static void SplitFieldMaps(const FieldMapVec &field_maps,
                               CacheLocationMapVector &out_locations,
                               PropertyMapVector &out_properties);
    // Legacy-compatible Put/Upsert/Update that accept FieldMapVec.
    static std::vector<ErrorCode>
    PutWithFieldMaps(MetaStorageBackend *backend, const KeyTypeVec &keys, const FieldMapVec &field_maps);
    static std::vector<ErrorCode>
    UpsertWithFieldMaps(MetaStorageBackend *backend, const KeyTypeVec &keys, const FieldMapVec &field_maps);
    // Legacy-compatible PutIfAbsent that accepts FieldMapVec (for MetaLocalBackend / MetaCacheBaseBackend).
    static std::vector<ErrorCode> PutIfAbsentWithFieldMaps(MetaCacheBaseBackend *backend,
                                                           const KeyTypeVec &keys,
                                                           const FieldMapVec &field_maps,
                                                           const std::vector<ErrorCode> &previous_error_codes);

    // ---- Assert helpers ----
    // Calls GetProperties(nullptr, ...) and asserts the returned per-key PropertyMap.
    static void AssertGetProperties(MetaStorageBackend *meta_storage_backend,
                                    const KeyTypeVec &keys,
                                    const std::vector<std::string> &field_names,
                                    const std::vector<ErrorCode> &expected_ec_vec,
                                    const PropertyMapVector &expected_properties);
    // Calls Get(nullptr, ...) to retrieve locations + properties, then merges
    // them into a FieldMap (location serialised via ToJsonString) for comparison.
    static void AssertGetAllFields(MetaStorageBackend *meta_storage_backend,
                                   const KeyTypeVec &keys,
                                   const std::vector<ErrorCode> &expected_ec_vec,
                                   const FieldMapVec &expected_field_maps);
    static void AssertExists(MetaStorageBackend *meta_storage_backend,
                             const KeyTypeVec &keys,
                             const std::vector<ErrorCode> &expected_ec_vec,
                             const std::vector<bool> &expected_is_exist_vec);
    static void AssertListKeys(MetaStorageBackend *meta_storage_backend,
                               const std::string &cursor,
                               const int64_t limit,
                               const ErrorCode expected_ec,
                               const std::string &expected_next_cursor,
                               const std::set<KeyType> &expected_keys);
    static void AssertListKeysByStep(MetaStorageBackend *meta_storage_backend,
                                     const std::string &cursor,
                                     const int64_t limit,
                                     const ErrorCode expected_ec,
                                     const std::set<KeyType> &expected_keys,
                                     std::string &out_next_cursor);
    static void AssertSampleReclaimKeys(MetaStorageBackend *meta_storage_backend,
                                        const int64_t count,
                                        const ErrorCode expected_ec,
                                        const std::set<KeyType> &expected_keys);
    static void AssertDeleteLocations(MetaStorageBackend *meta_storage_backend,
                                      const KeyTypeVec &keys,
                                      const LocationIdsPerKey &location_ids,
                                      const std::vector<ErrorCode> &expected_ec_vec);
    static void AssertExistsLocation(MetaStorageBackend *meta_storage_backend,
                                     const KeyTypeVec &keys,
                                     const std::vector<ErrorCode> &expected_ec_vec,
                                     const std::vector<bool> &expected_exists_vec);
};
} // namespace kv_cache_manager