#pragma once

#include <gtest/gtest.h>

#include "kv_cache_manager/meta/meta_storage_backend.h"

namespace kv_cache_manager {
class MetaStorageBackendTestBase {
protected:
    using KeyType = MetaStorageBackend::KeyType;
    using KeyTypeVec = MetaStorageBackend::KeyTypeVec;
    using FieldMap = MetaStorageBackend::FieldMap;
    using FieldMapVec = MetaStorageBackend::FieldMapVec;

    static void AssertGet(MetaStorageBackend *meta_storage_backend,
                          const MetaStorageBackend::KeyTypeVec &keys,
                          const std::vector<std::string> &field_names,
                          const std::vector<ErrorCode> expected_ec_vec,
                          const MetaStorageBackend::FieldMapVec &expected_field_maps);
    static void AssertGetAllFields(MetaStorageBackend *meta_storage_backend,
                                   const MetaStorageBackend::KeyTypeVec &keys,
                                   const std::vector<ErrorCode> expected_ec_vec,
                                   const MetaStorageBackend::FieldMapVec &expected_field_maps);
    static void AssertExists(MetaStorageBackend *meta_storage_backend,
                             const MetaStorageBackend::KeyTypeVec &keys,
                             const std::vector<ErrorCode> expected_ec_vec,
                             const std::vector<bool> &expected_is_exist_vec);
    static void AssertListKeys(MetaStorageBackend *meta_storage_backend,
                               const std::string &cursor,
                               const int64_t limit,
                               const ErrorCode expected_ec,
                               const std::string &expected_next_cursor,
                               const std::set<MetaStorageBackend::KeyType> &expected_keys);
    static void AssertListKeysByStep(MetaStorageBackend *meta_storage_backend,
                                     const std::string &cursor,
                                     const int64_t limit,
                                     const ErrorCode expected_ec,
                                     const std::set<MetaStorageBackend::KeyType> &expected_keys,
                                     std::string &out_next_cursor);
    static void AssertRandomSample(MetaStorageBackend *meta_storage_backend,
                                   const int64_t count,
                                   const ErrorCode expected_ec,
                                   const std::set<MetaStorageBackend::KeyType> &expected_keys);
};
} // namespace kv_cache_manager
