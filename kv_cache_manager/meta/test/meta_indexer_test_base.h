#pragma once

#include <gtest/gtest.h>

#include "kv_cache_manager/meta/meta_indexer.h"

namespace kv_cache_manager {
class MetaIndexerTestBase {
protected:
    using KeyVector = MetaIndexer::KeyVector;
    using UriVector = MetaIndexer::UriVector;
    using PropertyMap = MetaIndexer::PropertyMap;
    using PropertyMapVector = MetaIndexer::PropertyMapVector;
    using Result = MetaIndexer::Result;

    struct KVData {
        KeyVector keys;
        UriVector uris;
        PropertyMapVector properties;
    };
    void MakeKVData(const int64_t start, const int64_t end, KVData &data) const;
    void MakeRandomKVData(const int64_t count, const int64_t min, const int64_t max, KVData &data) const;

    void AssertGet(const KeyVector &keys, const UriVector &expect_uris, const Result &expect_result);
    void AssertSearchCacheGet(const KeyVector &keys,
                              const UriVector &expect_uris,
                              const std::vector<ErrorCode> &error_codes);
    void AssertGet(const KeyVector &keys,
                   const UriVector &expect_uris,
                   const PropertyMapVector &expect_properties,
                   const Result &expect_result);
    void AssertGetProperties(const KeyVector &keys,
                             const std::vector<std::string> &property_names,
                             PropertyMapVector &expect_properties,
                             const Result &expect_result);
    void AssertReadModifyWrite(const KeyVector &keys,
                               const MetaIndexer::ModifierFunc &modifier,
                               const Result &expect_result);
    void DoSimpleTest();
    void DoMultiThreadTest();

private:
    void DoPutTest();
    void DoUpdateTest();
    void DoDeleteAndExistTest();
    void DoScanAndRandomSampleTest();
    void DoReadModifyWriteTest();

protected:
    std::shared_ptr<MetaIndexer> meta_indexer_;
    std::shared_ptr<RequestContext> request_context_;
};
} // namespace kv_cache_manager
