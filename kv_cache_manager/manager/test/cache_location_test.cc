#include <gtest/gtest.h>
#include <map>
#include <string>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

using namespace kv_cache_manager;

class CacheLocationTest : public TESTBASE {
public:
    static CacheLocation MakeLocation(const std::string &id, DataStorageType type, const std::string &uri) {
        CacheLocation loc;
        loc.set_id(id);
        loc.set_status(CLS_SERVING);
        loc.set_type(type);
        loc.set_spec_size(1);
        loc.set_location_specs({LocationSpec(id, uri)});
        return loc;
    }
};

// (1) ToFieldMap: each location materializes as L#{loc_id} key + JSON value.
TEST_F(CacheLocationTest, ToFieldMapEmitsLPrefixedEntries) {
    BlockCacheLocationsMeta meta;
    meta.location_map()["kvs#v6d#mem#h1:1"] =
        MakeLocation("kvs#v6d#mem#h1:1", DataStorageType::DATA_STORAGE_TYPE_VINEYARD, "vineyard://h1:1/mem");
    meta.location_map()["kvs#v6d#disk#h2:1"] =
        MakeLocation("kvs#v6d#disk#h2:1", DataStorageType::DATA_STORAGE_TYPE_VINEYARD, "vineyard://h2:1/disk");

    std::map<std::string, std::string> fm;
    meta.ToFieldMap(fm);
    ASSERT_EQ(fm.size(), 2u);
    ASSERT_TRUE(fm.count(PROPERTY_LOCATION_PREFIX + "kvs#v6d#mem#h1:1"));
    ASSERT_TRUE(fm.count(PROPERTY_LOCATION_PREFIX + "kvs#v6d#disk#h2:1"));
    // Each value must parse back into a CacheLocation.
    CacheLocation parsed;
    ASSERT_TRUE(parsed.FromJsonString(fm.at(PROPERTY_LOCATION_PREFIX + "kvs#v6d#mem#h1:1")));
    EXPECT_EQ(parsed.id(), "kvs#v6d#mem#h1:1");
    EXPECT_EQ(parsed.type(), DataStorageType::DATA_STORAGE_TYPE_VINEYARD);
    EXPECT_EQ(parsed.location_specs().size(), 1u);
    EXPECT_EQ(parsed.location_specs()[0].uri(), "vineyard://h1:1/mem");
}

// (2) ToFieldMap merges into a caller-supplied map without disturbing other
//     entries (BP#/P# fields the indexer wrote separately must survive).
TEST_F(CacheLocationTest, ToFieldMapMergesWithExistingFields) {
    BlockCacheLocationsMeta meta;
    meta.location_map()["loc1"] = MakeLocation("loc1", DataStorageType::DATA_STORAGE_TYPE_HF3FS, "3fs://x");

    std::map<std::string, std::string> fm{
        {"BP#hit_count", "17"},
        {"BP#lru_time", "1234"},
        {"P#loc1#last_update", "5678"},
    };
    meta.ToFieldMap(fm);

    EXPECT_EQ(fm.at("BP#hit_count"), "17");
    EXPECT_EQ(fm.at("BP#lru_time"), "1234");
    EXPECT_EQ(fm.at("P#loc1#last_update"), "5678");
    EXPECT_TRUE(fm.count(PROPERTY_LOCATION_PREFIX + "loc1"));
}

// (3) FromFieldMap ingests only L# fields, ignoring BP#/P# and other keys.
TEST_F(CacheLocationTest, FromFieldMapIgnoresNonLocationEntries) {
    CacheLocation loc = MakeLocation("loc1", DataStorageType::DATA_STORAGE_TYPE_HF3FS, "3fs://x");
    std::map<std::string, std::string> fm{
        {PROPERTY_LOCATION_PREFIX + "loc1", loc.ToJsonString()},
        {"BP#hit_count", "17"},
        {"P#loc1#last_update", "5678"},
        {"unrelated_key", "noise"},
    };

    BlockCacheLocationsMeta meta;
    ASSERT_TRUE(meta.FromFieldMap(fm));
    EXPECT_EQ(meta.GetLocationCount(), 1u);
    ASSERT_TRUE(meta.location_map().count("loc1"));
    EXPECT_EQ(meta.location_map().at("loc1").type(), DataStorageType::DATA_STORAGE_TYPE_HF3FS);
}

// (4) Round-trip: ToFieldMap then FromFieldMap reproduces the original
//     location_map.
TEST_F(CacheLocationTest, RoundTripPreservesLocations) {
    BlockCacheLocationsMeta original;
    original.location_map()["loc_a"] = MakeLocation("loc_a", DataStorageType::DATA_STORAGE_TYPE_HF3FS, "3fs://a");
    original.location_map()["loc_b"] =
        MakeLocation("loc_b", DataStorageType::DATA_STORAGE_TYPE_VINEYARD, "vineyard://h:1/mem");

    std::map<std::string, std::string> fm;
    original.ToFieldMap(fm);

    BlockCacheLocationsMeta restored;
    ASSERT_TRUE(restored.FromFieldMap(fm));
    EXPECT_EQ(restored.GetLocationCount(), 2u);
    EXPECT_EQ(restored.location_map().at("loc_a").type(), DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    EXPECT_EQ(restored.location_map().at("loc_b").type(), DataStorageType::DATA_STORAGE_TYPE_VINEYARD);
    EXPECT_EQ(restored.location_map().at("loc_b").location_specs()[0].uri(), "vineyard://h:1/mem");
}

// (5) FromFieldMap rejects malformed JSON in a L# value -> false.
TEST_F(CacheLocationTest, FromFieldMapRejectsMalformedLocationJson) {
    std::map<std::string, std::string> fm{
        {PROPERTY_LOCATION_PREFIX + "loc_bad", "{not_json"},
    };
    BlockCacheLocationsMeta meta;
    EXPECT_FALSE(meta.FromFieldMap(fm));
}

// (6) FromFieldMap with an empty L# suffix is rejected (we never want to
//     accept a "L#" key with no location_id).
TEST_F(CacheLocationTest, FromFieldMapRejectsEmptyLocationId) {
    std::map<std::string, std::string> fm{
        {PROPERTY_LOCATION_PREFIX, R"({"id":"x","status":3,"type":1})"},
    };
    BlockCacheLocationsMeta meta;
    EXPECT_FALSE(meta.FromFieldMap(fm));
}

// (7) When the embedded CacheLocation JSON omits "id", FromFieldMap must
//     synthesize the id from the L# field name so callers never observe
//     anonymous entries.
TEST_F(CacheLocationTest, FromFieldMapBackfillsIdFromFieldName) {
    // Build a CacheLocation JSON that intentionally lacks the "id" field.
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> w(buf);
    w.StartObject();
    w.Key("status");
    w.Int(static_cast<int>(CLS_SERVING));
    w.Key("type");
    w.Int(static_cast<int>(DataStorageType::DATA_STORAGE_TYPE_VINEYARD));
    w.Key("spec_size");
    w.Uint64(1);
    w.Key("location_specs");
    w.StartArray();
    w.StartObject();
    w.Key("name");
    w.String("kvs#v6d#mem#h:1");
    w.Key("uri");
    w.String("vineyard://h:1/mem");
    w.EndObject();
    w.EndArray();
    w.EndObject();

    std::map<std::string, std::string> fm{
        {PROPERTY_LOCATION_PREFIX + "kvs#v6d#mem#h:1", buf.GetString()},
    };
    BlockCacheLocationsMeta meta;
    ASSERT_TRUE(meta.FromFieldMap(fm));
    ASSERT_TRUE(meta.location_map().count("kvs#v6d#mem#h:1"));
    EXPECT_EQ(meta.location_map().at("kvs#v6d#mem#h:1").id(), "kvs#v6d#mem#h:1");
}
