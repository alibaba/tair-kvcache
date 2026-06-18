#include <gtest/gtest.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/meta/cache_location.h"

using namespace kv_cache_manager;

class CacheLocationTest : public TESTBASE {};

TEST_F(CacheLocationTest, DefaultChecksumIsZero) {
    CacheLocation loc;
    EXPECT_EQ(loc.checksum(), 0);
}

TEST_F(CacheLocationTest, ChecksumRoundTripsThroughJson) {
    CacheLocation loc;
    loc.set_id("loc-1");
    loc.set_status(CacheLocationStatus::CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_spec_size(1);
    std::vector<LocationSpec> specs;
    specs.emplace_back("spec-a", "file://x/y?offset=0&size=1024");
    loc.set_location_specs(std::move(specs));
    loc.set_checksum(0x0123456789ABCDEFLL);

    const std::string json = loc.ToJsonString();
    EXPECT_NE(json.find("checksum"), std::string::npos);

    CacheLocation parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_EQ(parsed.checksum(), 0x0123456789ABCDEFLL);
    EXPECT_EQ(parsed.id(), "loc-1");
}

TEST_F(CacheLocationTest, LegacyJsonWithoutChecksumDefaultsToZero) {
    // 老 meta 存量数据：不带 checksum 字段。反序列化后 checksum 必须为 0，
    // 这样读端的 "expected_hash == 0 跳过校验" 兼容路径才能触发。
    const std::string legacy_json = R"({"id":"loc-1","status":3,"type":4,"spec_size":1,"create_time":0,)"
                                    R"("location_specs":[{"name":"spec-a","uri":"file://x/y?offset=0&size=1024"}]})";
    CacheLocation parsed;
    ASSERT_TRUE(parsed.FromJsonString(legacy_json));
    EXPECT_EQ(parsed.checksum(), 0);
    EXPECT_EQ(parsed.id(), "loc-1");
    EXPECT_EQ(parsed.status(), CacheLocationStatus::CLS_SERVING);
}

TEST_F(CacheLocationTest, NegativeChecksumIsPreserved) {
    // int64 全集都是合法 hash 值 (不像 0 那样有 sentinel 语义)。
    CacheLocation loc;
    loc.set_id("loc-2");
    loc.set_status(CacheLocationStatus::CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_spec_size(0);
    loc.set_checksum(-1);
    const std::string json = loc.ToJsonString();
    CacheLocation parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_EQ(parsed.checksum(), -1);
}
