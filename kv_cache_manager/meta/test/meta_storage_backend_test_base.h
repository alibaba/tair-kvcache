#pragma once

#include <gtest/gtest.h>

#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_cache_base_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"

namespace kv_cache_manager {
// Inspect the shared typed sampler in tests that only assert key selection.
template <typename Backend>
ErrorCode SampleReclaimKeysForTest(Backend *backend, int64_t count, KeyVector &out_keys) {
    ReclaimCandidateVector candidates;
    const auto ec = backend->SampleReclaimCandidates(nullptr, count, candidates, true);
    out_keys.clear();
    for (const auto &candidate : candidates) {
        out_keys.push_back(candidate.key);
    }
    return ec;
}

template <typename Backend>
class ScriptedReclaimReadBackend : public Backend {
public:
    ErrorCode RandomSample(RequestContext *, int64_t, KeyVector &out) noexcept override {
        out = keys;
        return EC_OK;
    }
    std::vector<ErrorCode> GetProperties(RequestContext *,
                                         const KeyVector &,
                                         const std::vector<std::string> &,
                                         PropertyMapVector &out) noexcept override {
        out = properties;
        return errors;
    }
    KeyVector keys{1, 2};
    PropertyMapVector properties{{{PROPERTY_LRU_TIME, "101"}}, {{PROPERTY_LRU_TIME, "202"}}};
    std::vector<ErrorCode> errors{EC_TIMEOUT, EC_OK};
};

template <typename Backend>
void AssertReclaimCandidateReadModes() {
    ScriptedReclaimReadBackend<Backend> backend;
    ReclaimCandidateVector candidates;
    ASSERT_EQ(EC_OK, backend.SampleReclaimCandidates(nullptr, 2, candidates));
    ASSERT_EQ(2, candidates.size());
    EXPECT_EQ(0, candidates[0].last_access_time_us);
    EXPECT_EQ(0, candidates[1].last_access_time_us);
    EXPECT_EQ(EC_TIMEOUT, backend.SampleReclaimCandidates(nullptr, 2, candidates, true));
    EXPECT_TRUE(candidates.empty());

    backend.errors = {EC_OK};
    EXPECT_EQ(EC_ERROR, backend.SampleReclaimCandidates(nullptr, 2, candidates, true));
    EXPECT_TRUE(candidates.empty());
    backend.errors = {EC_OK, EC_OK};
    backend.properties.resize(1);
    EXPECT_EQ(EC_ERROR, backend.SampleReclaimCandidates(nullptr, 2, candidates, true));
    EXPECT_TRUE(candidates.empty());

    backend.errors = {EC_NOENT, EC_ERROR};
    backend.properties = {{}, {}};
    EXPECT_EQ(EC_ERROR, backend.SampleReclaimCandidates(nullptr, 2, candidates, true));
    EXPECT_TRUE(candidates.empty());

    backend.keys = {1, 2, 3, 4};
    backend.errors = {EC_OK, EC_NOENT, EC_OK, EC_OK};
    backend.properties = {{{PROPERTY_LRU_TIME, "101"}}, {}, {}, {{PROPERTY_LRU_TIME, "invalid"}}};
    ASSERT_EQ(EC_OK, backend.SampleReclaimCandidates(nullptr, 4, candidates, true));
    ASSERT_EQ(4, candidates.size());
    EXPECT_EQ(1, candidates[0].key);
    EXPECT_EQ(101, candidates[0].last_access_time_us);
    // Redis EC_NOENT also means that just the requested LRU field is absent.
    EXPECT_EQ(2, candidates[1].key);
    EXPECT_EQ(0, candidates[1].last_access_time_us);
    EXPECT_EQ(3, candidates[2].key);
    EXPECT_EQ(0, candidates[2].last_access_time_us);
    EXPECT_EQ(4, candidates[3].key);
    EXPECT_EQ(0, candidates[3].last_access_time_us);

    backend.errors.assign(4, EC_NOENT);
    backend.properties.assign(4, {});
    ASSERT_EQ(EC_OK, backend.SampleReclaimCandidates(nullptr, 4, candidates, true));
    ASSERT_EQ(4, candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        EXPECT_EQ(backend.keys[i], candidates[i].key);
        EXPECT_EQ(0, candidates[i].last_access_time_us);
    }
}

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
