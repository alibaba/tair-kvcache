#pragma once

#include <gtest/gtest.h>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

// Test harness shared by MetaIndexer tests across different persistent backends
// (local / redis / dummy). Builds a location-centric KVData fixture and drives
// the high-level MetaIndexer APIs that were rewritten to operate on
// LocationMapVector + PropertyMapVector after the location-granularity refactor.
class MetaIndexerTestBase {
protected:
    using Result = MetaIndexer::Result;
    using LocationResult = MetaIndexer::LocationResult;

    // Per-request payload used by every helper below. Each slot at index `i`
    // describes key_i's locations + block-level properties.
    struct KVData {
        KeyVector keys;
        LocationMapVector location_maps;
        PropertyMapVector properties;
    };

    // Build a deterministic KVData for keys in [start, end). Each key i gets:
    //   * one CacheLocation with id = "loc_<i>", uri = "uri_<i>"
    //   * block-level properties {"p0": "p0_<i>", "p1": "p1_<i>"}
    void MakeKVData(const int64_t start, const int64_t end, KVData &data) const;
    // Same payload shape but with random keys in [min, max] (dedup'd). Used by
    // the multi-thread stress test.
    void MakeRandomKVData(const int64_t count, const int64_t min, const int64_t max, KVData &data) const;

    // Build a single-location CacheLocation with the given id/uri, using the
    // same schema as MakeKVData so producers/consumers stay in lockstep.
    static CacheLocation MakeLocation(const std::string &id, const std::string &uri);

    // Assertions over MetaIndexer::Get(LocationMapVector, PropertyMapVector)
    // (the new whole-block read path).
    void AssertGet(const KeyVector &keys,
                   const LocationMapVector &expect_location_maps,
                   const Result &expect_result);
    void AssertGet(const KeyVector &keys,
                   const LocationMapVector &expect_location_maps,
                   const PropertyMapVector &expect_properties,
                   const Result &expect_result);
    void AssertGetProperties(const KeyVector &keys,
                             const std::vector<std::string> &property_names,
                             const PropertyMapVector &expect_properties,
                             const Result &expect_result);

    // Shared test scenarios. DoSimpleTest sequences put/update/delete/exist/
    // scan/sample/rmw sub-tests; DoMultiThreadTest stresses ReadModifyWriteBlock
    // with concurrent add/delete threads.
    void DoSimpleTest();
    void DoMultiThreadTest();

private:
    void DoPutTest();
    void DoUpdateTest();
    void DoDeleteAndExistTest();
    void DoScanAndSampleReclaimKeysTest();
    void DoReadModifyWriteBlockTest();
    void DoReadModifyWriteLocationTest();

protected:
    std::shared_ptr<MetaIndexer> meta_indexer_;
    std::shared_ptr<RequestContext> request_context_;
};
} // namespace kv_cache_manager
