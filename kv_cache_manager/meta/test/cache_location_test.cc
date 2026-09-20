#include <gtest/gtest.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/meta/cache_location.h"

using namespace kv_cache_manager;

class CacheLocationTest : public TESTBASE {};

TEST_F(CacheLocationTest, DefaultLocationSpecChecksumIsAbsent) {
    LocationSpec spec("tp0", "file://x/y?offset=0&size=1024");
    EXPECT_EQ(spec.checksum(), 0);
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
    specs.back().set_checksum(0x0123456789ABCDEFLL);
    specs.emplace_back("spec-b", "file://x/z?offset=0&size=1024");
    specs.back().set_checksum(-7);
    loc.set_location_specs(std::move(specs));

    const std::string json = loc.ToJsonString();
    EXPECT_NE(json.find("checksum"), std::string::npos);

    CacheLocation parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    ASSERT_EQ(parsed.location_specs().size(), 2u);
    EXPECT_EQ(parsed.location_specs()[0].checksum(), 0x0123456789ABCDEFLL);
    EXPECT_TRUE(parsed.location_specs()[0].has_checksum());
    EXPECT_EQ(parsed.location_specs()[1].checksum(), -7);
    EXPECT_TRUE(parsed.location_specs()[1].has_checksum());
    EXPECT_EQ(parsed.id(), "loc-1");
}

TEST_F(CacheLocationTest, LegacyJsonWithoutChecksumDefaultsToZero) {
    // 老 meta 存量数据：不带 checksum 字段。数值保持 0，但 presence=false，
    // 读端据此跳过校验；合法的 checksum=0 则由 presence=true 区分。
    const std::string legacy_json = R"({"id":"loc-1","status":3,"type":4,"spec_size":1,"create_time":0,)"
                                    R"("location_specs":[{"name":"spec-a","uri":"file://x/y?offset=0&size=1024"}]})";
    CacheLocation parsed;
    ASSERT_TRUE(parsed.FromJsonString(legacy_json));
    ASSERT_EQ(parsed.location_specs().size(), 1u);
    EXPECT_EQ(parsed.location_specs()[0].checksum(), 0);
    EXPECT_FALSE(parsed.location_specs()[0].has_checksum());
    EXPECT_EQ(parsed.id(), "loc-1");
    EXPECT_EQ(parsed.status(), CacheLocationStatus::CLS_SERVING);
}

TEST_F(CacheLocationTest, NegativeChecksumIsPreserved) {
    // int64 全集都是合法 checksum 值。
    LocationSpec spec("tp0", "file://x");
    spec.set_checksum(-1);
    const std::string json = spec.ToJsonString();
    LocationSpec parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_EQ(parsed.checksum(), -1);
    EXPECT_TRUE(parsed.has_checksum());
}

TEST_F(CacheLocationTest, ZeroChecksumHasExplicitJsonPresence) {
    LocationSpec spec("tp0", "file://x");
    spec.set_checksum(0);
    EXPECT_TRUE(spec.has_checksum());

    const std::string json = spec.ToJsonString();
    EXPECT_NE(json.find("\"checksum\":0"), std::string::npos);

    LocationSpec parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_TRUE(parsed.has_checksum());
    EXPECT_EQ(parsed.checksum(), 0);
}

TEST_F(CacheLocationTest, CopyConstructorPreservesChecksumPresence) {
    CacheLocation original;
    std::vector<LocationSpec> specs;
    specs.emplace_back("tp0", "file://x");
    specs.back().set_checksum(0);
    original.set_location_specs(std::move(specs));

    CacheLocation copied(original);

    ASSERT_EQ(copied.location_specs().size(), 1u);
    EXPECT_TRUE(copied.location_specs()[0].has_checksum());
    EXPECT_EQ(copied.location_specs()[0].checksum(), 0);
}
