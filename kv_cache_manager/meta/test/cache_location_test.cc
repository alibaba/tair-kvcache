#include <gtest/gtest.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/meta/cache_location.h"

using namespace kv_cache_manager;

class CacheLocationTest : public TESTBASE {};

TEST_F(CacheLocationTest, DefaultLocationSpecChecksumIsAbsent) {
    LocationSpec spec("tp0", "file://x/y?offset=0&size=1024");
    EXPECT_EQ(spec.checksum(), "");
    EXPECT_FALSE(spec.has_checksum());
}

TEST_F(CacheLocationTest, PerSpecChecksumsRoundTripThroughJson) {
    CacheLocation loc;
    loc.set_id("loc-1");
    loc.set_status(CacheLocationStatus::CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_spec_size(2);
    std::vector<LocationSpec> specs;
    specs.emplace_back("spec-a", "file://x/y?offset=0&size=1024");
    specs.back().set_checksum("crc64:0123456789abcdef");
    specs.emplace_back("spec-b", "file://x/z?offset=0&size=1024");
    specs.back().set_checksum("sha256:00000123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd");
    loc.set_location_specs(std::move(specs));

    const std::string json = loc.ToJsonString();
    EXPECT_NE(json.find("checksum"), std::string::npos);

    CacheLocation parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    ASSERT_EQ(parsed.location_specs().size(), 2u);
    EXPECT_EQ(parsed.location_specs()[0].checksum(), "crc64:0123456789abcdef");
    EXPECT_TRUE(parsed.location_specs()[0].has_checksum());
    EXPECT_EQ(parsed.location_specs()[1].checksum(),
              "sha256:00000123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd");
    EXPECT_TRUE(parsed.location_specs()[1].has_checksum());
    EXPECT_EQ(parsed.id(), "loc-1");
}

TEST_F(CacheLocationTest, LegacyJsonWithoutChecksumKeepsPresenceFalse) {
    // Old metadata omits checksum; an explicitly stored empty string has presence=true.
    const std::string legacy_json = R"({"id":"loc-1","status":3,"type":4,"spec_size":1,"create_time":0,)"
                                    R"("location_specs":[{"name":"spec-a","uri":"file://x/y?offset=0&size=1024"}]})";
    CacheLocation parsed;
    ASSERT_TRUE(parsed.FromJsonString(legacy_json));
    ASSERT_EQ(parsed.location_specs().size(), 1u);
    EXPECT_EQ(parsed.location_specs()[0].checksum(), "");
    EXPECT_FALSE(parsed.location_specs()[0].has_checksum());
    EXPECT_EQ(parsed.id(), "loc-1");
    EXPECT_EQ(parsed.status(), CacheLocationStatus::CLS_SERVING);
}

TEST_F(CacheLocationTest, OpaqueChecksumIsPreserved) {
    LocationSpec spec("tp0", "file://x");
    const std::string checksum = "校验:00FF/+=\"\\\n";
    spec.set_checksum(checksum);
    const std::string json = spec.ToJsonString();
    LocationSpec parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_EQ(parsed.checksum(), checksum);
    EXPECT_TRUE(parsed.has_checksum());
}

TEST_F(CacheLocationTest, ZeroChecksumHasExplicitJsonPresence) {
    LocationSpec spec("tp0", "file://x");
    spec.set_checksum("0");
    EXPECT_TRUE(spec.has_checksum());

    const std::string json = spec.ToJsonString();
    EXPECT_NE(json.find("\"checksum\":\"0\""), std::string::npos);

    LocationSpec parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_TRUE(parsed.has_checksum());
    EXPECT_EQ(parsed.checksum(), "0");
}

TEST_F(CacheLocationTest, CopyConstructorPreservesChecksumPresence) {
    CacheLocation original;
    std::vector<LocationSpec> specs;
    specs.emplace_back("tp0", "file://x");
    specs.back().set_checksum("");
    original.set_location_specs(std::move(specs));

    CacheLocation copied(original);

    ASSERT_EQ(copied.location_specs().size(), 1u);
    EXPECT_TRUE(copied.location_specs()[0].has_checksum());
    EXPECT_EQ(copied.location_specs()[0].checksum(), "");
}

TEST_F(CacheLocationTest, EmptyChecksumIsDistinctFromAbsentAndReuseClearsPresence) {
    LocationSpec spec("tp0", "file://x");
    spec.set_checksum("");
    const auto json = spec.ToJsonString();
    EXPECT_NE(std::string::npos, json.find("\"checksum\":\"\""));

    LocationSpec parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_TRUE(parsed.has_checksum());
    EXPECT_EQ("", parsed.checksum());

    ASSERT_TRUE(parsed.FromJsonString(R"({"name":"tp0","uri":"file://x"})"));
    EXPECT_FALSE(parsed.has_checksum());
    EXPECT_EQ("", parsed.checksum());
    EXPECT_EQ(std::string::npos, parsed.ToJsonString().find("checksum"));
}

TEST_F(CacheLocationTest, NonStringChecksumIsRejected) {
    LocationSpec spec;
    EXPECT_FALSE(spec.FromJsonString(R"({"name":"tp0","uri":"file://x","checksum":123})"));
}
