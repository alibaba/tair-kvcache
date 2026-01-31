#include "kv_cache_manager/meta/test/meta_storage_backend_test_base.h"

namespace kv_cache_manager {
void MetaStorageBackendTestBase::AssertGet(MetaStorageBackend *meta_storage_backend,
                                           const KeyTypeVec &keys,
                                           const std::vector<std::string> &field_names,
                                           const std::vector<ErrorCode> expected_ec_vec,
                                           const FieldMapVec &expected_field_maps) {
    ASSERT_TRUE(meta_storage_backend);
    FieldMapVec field_maps;
    std::vector<ErrorCode> ec_vec = meta_storage_backend->Get(keys, field_names, field_maps);
    ASSERT_EQ(keys.size(), expected_ec_vec.size());
    ASSERT_EQ(expected_ec_vec, ec_vec);
    ASSERT_EQ(keys.size(), field_maps.size());
    ASSERT_EQ(expected_field_maps.size(), field_maps.size());
    for (int i = 0; i < keys.size(); ++i) {
        const KeyType &key = keys[i];
        const FieldMap &field_map = field_maps[i];
        const FieldMap &expected_field_map = expected_field_maps[i];
        ASSERT_EQ(expected_field_map.size(), field_map.size()) << key;
        for (const auto &[expected_field_name, expected_field_value] : expected_field_map) {
            const auto iter = field_map.find(expected_field_name);
            ASSERT_TRUE(iter != field_map.end()) << key << " " << expected_field_name;
            ASSERT_EQ(expected_field_value, iter->second) << key << " " << expected_field_name;
        }
    }
}

void MetaStorageBackendTestBase::AssertGetAllFields(MetaStorageBackend *meta_storage_backend,
                                                    const KeyTypeVec &keys,
                                                    const std::vector<ErrorCode> expected_ec_vec,
                                                    const FieldMapVec &expected_field_maps) {
    ASSERT_TRUE(meta_storage_backend);
    FieldMapVec field_maps;
    std::vector<ErrorCode> ec_vec = meta_storage_backend->GetAllFields(keys, field_maps);
    ASSERT_EQ(keys.size(), expected_ec_vec.size());
    ASSERT_EQ(expected_ec_vec, ec_vec);
    ASSERT_EQ(keys.size(), field_maps.size());
    ASSERT_EQ(expected_field_maps.size(), field_maps.size());
    for (int i = 0; i < keys.size(); ++i) {
        const KeyType &key = keys[i];
        const FieldMap &field_map = field_maps[i];
        const FieldMap &expected_field_map = expected_field_maps[i];
        ASSERT_EQ(expected_field_map.size(), field_map.size()) << key;
        for (const auto &[expected_field_name, expected_field_value] : expected_field_map) {
            const auto iter = field_map.find(expected_field_name);
            ASSERT_TRUE(iter != field_map.end()) << key << " " << expected_field_name;
            ASSERT_EQ(expected_field_value, iter->second) << key << " " << expected_field_name;
        }
    }
}

void MetaStorageBackendTestBase::AssertExists(MetaStorageBackend *meta_storage_backend,
                                              const KeyTypeVec &keys,
                                              const std::vector<ErrorCode> expected_ec_vec,
                                              const std::vector<bool> &expected_is_exist_vec) {
    ASSERT_TRUE(meta_storage_backend);
    std::vector<bool> is_exist_vec;
    std::vector<ErrorCode> ec_vec = meta_storage_backend->Exists(keys, is_exist_vec);
    ASSERT_EQ(keys.size(), expected_ec_vec.size());
    ASSERT_EQ(expected_ec_vec, ec_vec);
    ASSERT_EQ(keys.size(), is_exist_vec.size());
    ASSERT_EQ(expected_is_exist_vec.size(), is_exist_vec.size());
    for (int i = 0; i < keys.size(); ++i) {
        const KeyType &key = keys[i];
        const bool is_exist = is_exist_vec[i];
        const bool expected_is_exist = expected_is_exist_vec[i];
        ASSERT_EQ(expected_is_exist, is_exist) << "key is: " << key;
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
    ErrorCode ec = meta_storage_backend->ListKeys(cursor, limit, next_cursor, keys);
    ASSERT_EQ(expected_ec, ec);
    if (ec == EC_OK) {
        ASSERT_EQ(expected_next_cursor, next_cursor);
        ASSERT_EQ(std::min((size_t)limit, expected_keys.size()), keys.size());
        for (const auto &key : keys) {
            const auto iter = expected_keys.find(key);
            ASSERT_TRUE(iter != expected_keys.end()) << key;
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
    ErrorCode ec = meta_storage_backend->ListKeys(cursor, limit, out_next_cursor, keys);
    ASSERT_EQ(expected_ec, ec);
    if (ec == EC_OK) {
        ASSERT_EQ(std::min((size_t)limit, expected_keys.size()), keys.size());
        for (const auto &key : keys) {
            const auto iter = expected_keys.find(key);
            ASSERT_TRUE(iter != expected_keys.end()) << key;
        }
    } else {
        ASSERT_EQ("", out_next_cursor);
        ASSERT_EQ(0, keys.size());
    }
}

void MetaStorageBackendTestBase::AssertRandomSample(MetaStorageBackend *meta_storage_backend,
                                                    const int64_t count,
                                                    const ErrorCode expected_ec,
                                                    const std::set<KeyType> &expected_keys) {
    ASSERT_TRUE(meta_storage_backend);
    std::vector<KeyType> keys;
    bool ec = meta_storage_backend->RandomSample(count, keys);
    ASSERT_EQ(expected_ec, ec);
    ASSERT_EQ(std::min((size_t)count, expected_keys.size()), keys.size());
    for (const auto &key : keys) {
        const auto iter = expected_keys.find(key);
        ASSERT_TRUE(iter != expected_keys.end()) << key;
    }
}
} // namespace kv_cache_manager
