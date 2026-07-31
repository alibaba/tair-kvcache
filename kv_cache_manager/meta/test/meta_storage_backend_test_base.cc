#include "kv_cache_manager/meta/test/meta_storage_backend_test_base.h"

#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"

namespace kv_cache_manager {

// ---------------------------------------------------------------------------
// Compatibility wrappers
// ---------------------------------------------------------------------------

void MetaStorageBackendTestBase::SplitFieldMaps(const FieldMapVec &field_maps,
                                                CacheLocationMapVector &out_locations,
                                                PropertyMapVector &out_properties) {
    out_locations.resize(field_maps.size());
    out_properties.resize(field_maps.size());
    for (size_t i = 0; i < field_maps.size(); ++i) {
        for (const auto &[name, value] : field_maps[i]) {
            if (name.rfind(PROPERTY_LOCATION_PREFIX, 0) == 0) {
                std::string loc_id = name.substr(PROPERTY_LOCATION_PREFIX.size());
                auto loc = std::make_shared<CacheLocation>();
                loc->set_id(loc_id);
                // Store raw value in a location_spec uri so round-trip tests
                // that compare serialised JSON can still work. For empty value
                // (tombstone), leave the location default-constructed (no specs).
                if (!value.empty()) {
                    loc->FromJsonString(value);
                    // If FromJsonString failed (value was not valid JSON),
                    // fall back to storing raw value.
                    if (loc->id().empty()) {
                        loc->set_id(loc_id);
                    }
                }
                out_locations[i][loc_id] = std::move(loc);
            } else {
                out_properties[i][name] = value;
            }
        }
    }
}

std::vector<ErrorCode> MetaStorageBackendTestBase::PutWithFieldMaps(MetaStorageBackend *backend,
                                                                    const KeyTypeVec &keys,
                                                                    const FieldMapVec &field_maps) {
    CacheLocationMapVector locations;
    PropertyMapVector properties;
    SplitFieldMaps(field_maps, locations, properties);
    return backend->Put(nullptr, keys, locations, properties);
}

std::vector<ErrorCode> MetaStorageBackendTestBase::UpsertWithFieldMaps(MetaStorageBackend *backend,
                                                                       const KeyTypeVec &keys,
                                                                       const FieldMapVec &field_maps) {
    CacheLocationMapVector locations;
    PropertyMapVector properties;
    SplitFieldMaps(field_maps, locations, properties);
    return backend->Upsert(nullptr, keys, locations, properties);
}

std::vector<ErrorCode>
MetaStorageBackendTestBase::PutIfAbsentWithFieldMaps(MetaCacheBaseBackend *backend,
                                                     const KeyTypeVec &keys,
                                                     const FieldMapVec &field_maps,
                                                     const std::vector<ErrorCode> &previous_error_codes) {
    CacheLocationMapVector locations;
    PropertyMapVector properties;
    SplitFieldMaps(field_maps, locations, properties);
    return backend->PutIfAbsent(nullptr, keys, locations, properties, previous_error_codes);
}

void MetaStorageBackendTestBase::AssertGetProperties(MetaStorageBackend *meta_storage_backend,
                                                     const KeyTypeVec &keys,
                                                     const std::vector<std::string> &field_names,
                                                     const std::vector<ErrorCode> &expected_ec_vec,
                                                     const PropertyMapVector &expected_properties) {
    ASSERT_TRUE(meta_storage_backend);
    PropertyMapVector out_properties;
    std::vector<ErrorCode> ec_vec = meta_storage_backend->GetProperties(nullptr, keys, field_names, out_properties);
    ASSERT_EQ(keys.size(), expected_ec_vec.size());
    ASSERT_EQ(expected_ec_vec, ec_vec);
    ASSERT_EQ(keys.size(), out_properties.size());
    ASSERT_EQ(expected_properties.size(), out_properties.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const KeyType &key = keys[i];
        const PropertyMap &actual = out_properties[i];
        const PropertyMap &expected = expected_properties[i];
        ASSERT_EQ(expected.size(), actual.size()) << key;
        for (const auto &[name, value] : expected) {
            const auto iter = actual.find(name);
            ASSERT_TRUE(iter != actual.end()) << key << " " << name;
            ASSERT_EQ(value, iter->second) << key << " " << name;
        }
    }
}

void MetaStorageBackendTestBase::AssertGetAllFields(MetaStorageBackend *meta_storage_backend,
                                                    const KeyTypeVec &keys,
                                                    const std::vector<ErrorCode> &expected_ec_vec,
                                                    const FieldMapVec &expected_field_maps) {
    ASSERT_TRUE(meta_storage_backend);
    CacheLocationMapVector out_locations;
    PropertyMapVector out_properties;
    std::vector<ErrorCode> ec_vec = meta_storage_backend->Get(nullptr, keys, out_locations, out_properties);
    ASSERT_EQ(keys.size(), expected_ec_vec.size());
    ASSERT_EQ(expected_ec_vec, ec_vec);
    ASSERT_EQ(keys.size(), out_locations.size());
    ASSERT_EQ(keys.size(), out_properties.size());
    ASSERT_EQ(expected_field_maps.size(), out_locations.size());

    // Merge locations + properties back into a FieldMap for comparison.
    for (size_t i = 0; i < keys.size(); ++i) {
        const KeyType &key = keys[i];
        FieldMap merged;
        for (const auto &[loc_id, loc_ptr] : out_locations[i]) {
            merged[PROPERTY_LOCATION_PREFIX + loc_id] = loc_ptr ? loc_ptr->ToJsonString() : "";
        }
        for (const auto &[prop_name, prop_value] : out_properties[i]) {
            merged[prop_name] = prop_value;
        }
        const FieldMap &expected_field_map = expected_field_maps[i];
        // Count actual fields excluding PROPERTY_LRU_TIME (dynamically filled by local backend).
        size_t actual_count_without_lru = merged.size();
        if (merged.count(PROPERTY_LRU_TIME) && !expected_field_map.count(PROPERTY_LRU_TIME)) {
            --actual_count_without_lru;
        }
        ASSERT_EQ(expected_field_map.size(), actual_count_without_lru) << key;
        for (const auto &[expected_field_name, expected_field_value] : expected_field_map) {
            const auto iter = merged.find(expected_field_name);
            ASSERT_TRUE(iter != merged.end()) << key << " " << expected_field_name;
            if (expected_field_name == PROPERTY_LRU_TIME) {
                ASSERT_FALSE(iter->second.empty()) << key << " " << expected_field_name << " should not be empty";
            } else {
                ASSERT_EQ(expected_field_value, iter->second) << key << " " << expected_field_name;
            }
        }
    }
}

void MetaStorageBackendTestBase::AssertExists(MetaStorageBackend *meta_storage_backend,
                                              const KeyTypeVec &keys,
                                              const std::vector<ErrorCode> &expected_ec_vec,
                                              const std::vector<bool> &expected_is_exist_vec) {
    ASSERT_TRUE(meta_storage_backend);
    std::vector<bool> is_exist_vec;
    std::vector<ErrorCode> ec_vec = meta_storage_backend->Exists(nullptr, keys, is_exist_vec);
    ASSERT_EQ(keys.size(), expected_ec_vec.size());
    ASSERT_EQ(expected_ec_vec, ec_vec);
    ASSERT_EQ(keys.size(), is_exist_vec.size());
    ASSERT_EQ(expected_is_exist_vec.size(), is_exist_vec.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const KeyType &key = keys[i];
        ASSERT_EQ(expected_is_exist_vec[i], is_exist_vec[i]) << "key is: " << key;
    }
}

void MetaStorageBackendTestBase::AssertListKeys(MetaStorageBackend *meta_storage_backend,
                                                const std::string &cursor,
                                                const int64_t limit,
                                                const ErrorCode expected_ec,
                                                const std::string &expected_next_cursor,
                                                const std::set<KeyType> &expected_keys) {
    ASSERT_TRUE(meta_storage_backend);
    std::string next_cursor;
    std::vector<KeyType> keys;
    ErrorCode ec = meta_storage_backend->ListKeys(nullptr, cursor, limit, next_cursor, keys);
    ASSERT_EQ(expected_ec, ec);
    if (ec == EC_OK) {
        ASSERT_EQ(expected_next_cursor, next_cursor);
        ASSERT_EQ(std::min(static_cast<size_t>(limit), expected_keys.size()), keys.size());
        for (const auto &key : keys) {
            ASSERT_TRUE(expected_keys.count(key)) << key;
        }
    } else {
        ASSERT_EQ("", next_cursor);
        ASSERT_EQ(0, keys.size());
    }
}

void MetaStorageBackendTestBase::AssertListKeysByStep(MetaStorageBackend *meta_storage_backend,
                                                      const std::string &cursor,
                                                      const int64_t limit,
                                                      const ErrorCode expected_ec,
                                                      const std::set<KeyType> &expected_keys,
                                                      std::string &out_next_cursor) {
    ASSERT_TRUE(meta_storage_backend);
    std::vector<KeyType> keys;
    ErrorCode ec = meta_storage_backend->ListKeys(nullptr, cursor, limit, out_next_cursor, keys);
    ASSERT_EQ(expected_ec, ec);
    if (ec == EC_OK) {
        ASSERT_EQ(std::min(static_cast<size_t>(limit), expected_keys.size()), keys.size());
        for (const auto &key : keys) {
            ASSERT_TRUE(expected_keys.count(key)) << key;
        }
    } else {
        ASSERT_EQ("", out_next_cursor);
        ASSERT_EQ(0, keys.size());
    }
}

void MetaStorageBackendTestBase::AssertSampleReclaimKeys(MetaStorageBackend *meta_storage_backend,
                                                         const int64_t count,
                                                         const ErrorCode expected_ec,
                                                         const std::set<KeyType> &expected_keys) {
    ASSERT_TRUE(meta_storage_backend);
    std::vector<KeyType> keys;
    ErrorCode ec = meta_storage_backend->SampleReclaimKeys(nullptr, count, keys);
    ASSERT_EQ(expected_ec, ec);
    for (const auto &key : keys) {
        ASSERT_TRUE(expected_keys.count(key)) << key;
    }
}

void MetaStorageBackendTestBase::AssertDeleteLocations(MetaStorageBackend *meta_storage_backend,
                                                       const KeyTypeVec &keys,
                                                       const LocationIdsPerKey &location_ids,
                                                       const std::vector<ErrorCode> &expected_ec_vec) {
    ASSERT_TRUE(meta_storage_backend);
    std::vector<ErrorCode> ec_vec = meta_storage_backend->DeleteLocations(nullptr, keys, location_ids);
    ASSERT_EQ(expected_ec_vec, ec_vec);
}

void MetaStorageBackendTestBase::AssertExistsLocation(MetaStorageBackend *meta_storage_backend,
                                                      const KeyTypeVec &keys,
                                                      const std::vector<ErrorCode> &expected_ec_vec,
                                                      const std::vector<bool> &expected_exists_vec) {
    ASSERT_TRUE(meta_storage_backend);
    std::vector<bool> exists_vec;
    std::vector<ErrorCode> ec_vec = meta_storage_backend->ExistsLocation(nullptr, keys, exists_vec);
    ASSERT_EQ(expected_ec_vec, ec_vec);
    ASSERT_EQ(expected_exists_vec, exists_vec);
}

} // namespace kv_cache_manager