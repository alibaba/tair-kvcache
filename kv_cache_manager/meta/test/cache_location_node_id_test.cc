// LocationSpec.node_id round-trip tests:
//  (a) C++ struct constructor/getter/setter
//  (b) JSON ToRapidWriter / FromRapidValue round-trip
//  (c) Legacy JSON without node_id field degrades to empty string

#include <string>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/meta/cache_location.h"

namespace kv_cache_manager {

class LocationSpecNodeIdTest : public TESTBASE {};

TEST_F(LocationSpecNodeIdTest, DefaultNodeIdIsEmpty) {
    LocationSpec spec("tp0", "pace://cluster/key");
    EXPECT_EQ("", spec.node_id());
}

TEST_F(LocationSpecNodeIdTest, ThreeArgCtorSetsNodeId) {
    LocationSpec spec("tp0", "pace://cluster/key", "worker.33");
    EXPECT_EQ("tp0", spec.name());
    EXPECT_EQ("pace://cluster/key", spec.uri());
    EXPECT_EQ("worker.33", spec.node_id());
}

TEST_F(LocationSpecNodeIdTest, SetterUpdatesNodeId) {
    LocationSpec spec("tp0", "pace://cluster/key");
    spec.set_node_id("worker.42");
    EXPECT_EQ("worker.42", spec.node_id());
}

TEST_F(LocationSpecNodeIdTest, JsonRoundTrip) {
    LocationSpec origin("tp1", "pace://cluster/key2", "worker.55");
    std::string serialized = origin.ToJsonString();

    LocationSpec parsed;
    ASSERT_TRUE(parsed.FromJsonString(serialized));
    EXPECT_EQ(origin.name(), parsed.name());
    EXPECT_EQ(origin.uri(), parsed.uri());
    EXPECT_EQ(origin.node_id(), parsed.node_id());
}

TEST_F(LocationSpecNodeIdTest, LegacyJsonWithoutNodeIdParsesAsEmpty) {
    // 模拟 v0 持久化数据：只有 name + uri，没有 node_id 字段
    const std::string legacy_json = R"({"name":"tp0","uri":"pace://cluster/key"})";
    LocationSpec parsed;
    ASSERT_TRUE(parsed.FromJsonString(legacy_json));
    EXPECT_EQ("tp0", parsed.name());
    EXPECT_EQ("pace://cluster/key", parsed.uri());
    EXPECT_EQ("", parsed.node_id());
}

TEST_F(LocationSpecNodeIdTest, CacheLocationCarriesNodeIdAcrossRoundTrip) {
    std::vector<LocationSpec> specs{
        LocationSpec("tp0", "pace://c/k0", "node_a"),
        LocationSpec("tp1", "pace://c/k1", "node_b"),
    };
    CacheLocation loc("loc_1", CLS_SERVING, DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, 2, specs);
    std::string s = loc.ToJsonString();
    CacheLocation parsed;
    ASSERT_TRUE(parsed.FromJsonString(s));
    ASSERT_EQ(2u, parsed.location_specs().size());
    EXPECT_EQ("node_a", parsed.location_specs()[0].node_id());
    EXPECT_EQ("node_b", parsed.location_specs()[1].node_id());
}

} // namespace kv_cache_manager
