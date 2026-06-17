#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.pb.h"
#include "kv_cache_manager/service/util/manager_message_proto_util.h"

namespace kv_cache_manager {

class InstanceGroupTest : public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}
};

// --- StringUtil::ParseBucketBoundaries ---

TEST_F(InstanceGroupTest, ParseEmptyStringReturnsEmpty) {
    auto result = StringUtil::ParseBucketBoundaries("");
    EXPECT_TRUE(result.empty());
}

TEST_F(InstanceGroupTest, ParseValidBoundaries) {
    auto result = StringUtil::ParseBucketBoundaries("1,5,30,60,300,3600");
    ASSERT_EQ(result.size(), 6);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 5.0);
    EXPECT_DOUBLE_EQ(result[2], 30.0);
    EXPECT_DOUBLE_EQ(result[3], 60.0);
    EXPECT_DOUBLE_EQ(result[4], 300.0);
    EXPECT_DOUBLE_EQ(result[5], 3600.0);
}

TEST_F(InstanceGroupTest, ParseWithSpaces) {
    auto result = StringUtil::ParseBucketBoundaries(" 1 , 5 , 30 ");
    ASSERT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 5.0);
    EXPECT_DOUBLE_EQ(result[2], 30.0);
}

TEST_F(InstanceGroupTest, ParseNonAscendingRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("5,1,30").empty());
}

TEST_F(InstanceGroupTest, ParseDuplicateRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("1,5,5,30").empty());
}

TEST_F(InstanceGroupTest, ParseNegativeRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("-1,5,30").empty());
}

TEST_F(InstanceGroupTest, ParseZeroRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("0,5,30").empty());
}

TEST_F(InstanceGroupTest, ParseTrailingCharsRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("1s,5,30").empty());
}

TEST_F(InstanceGroupTest, ParseEmptyTokenRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("1,,5").empty());
}

TEST_F(InstanceGroupTest, ParseLeadingCommaRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries(",1,5").empty());
}

TEST_F(InstanceGroupTest, ParseTrailingCommaRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("1,5,").empty());
}

TEST_F(InstanceGroupTest, ParseNonNumericRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("abc,5,30").empty());
}

TEST_F(InstanceGroupTest, ParseFractionalBoundaries) {
    auto result = StringUtil::ParseBucketBoundaries("0.5,1.5,5.0");
    ASSERT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 0.5);
    EXPECT_DOUBLE_EQ(result[1], 1.5);
    EXPECT_DOUBLE_EQ(result[2], 5.0);
}

// --- InstanceGroup set_revisit_interval_buckets ---

TEST_F(InstanceGroupTest, SetValidBuckets) {
    InstanceGroup group;
    group.set_name("test");
    group.set_revisit_interval_buckets("1,5,30,60");
    ASSERT_EQ(group.revisit_interval_buckets().size(), 4);
    EXPECT_DOUBLE_EQ(group.revisit_interval_buckets()[0], 1.0);
    EXPECT_DOUBLE_EQ(group.revisit_interval_buckets()[3], 60.0);
    EXPECT_EQ(group.revisit_interval_buckets_raw(), "1,5,30,60");
}

TEST_F(InstanceGroupTest, SetInvalidBucketsClearsParsed) {
    InstanceGroup group;
    group.set_name("test");
    group.set_revisit_interval_buckets("5,1,30");  // not ascending
    EXPECT_TRUE(group.revisit_interval_buckets().empty());
    EXPECT_EQ(group.revisit_interval_buckets_raw(), "5,1,30");  // raw preserved
}

TEST_F(InstanceGroupTest, SetEmptyBuckets) {
    InstanceGroup group;
    group.set_name("test");
    group.set_revisit_interval_buckets("");
    EXPECT_TRUE(group.revisit_interval_buckets().empty());
}

// --- JSON round-trip ---

TEST_F(InstanceGroupTest, JsonRoundTripWithBuckets) {
    std::string json = R"({
        "name": "test_group",
        "storage_candidates": ["local"],
        "global_quota_group_name": "default",
        "max_instance_count": 10,
        "quota": {"capacity": 1024},
        "cache_config": {"meta_indexer_config": {"max_key_count": 1000, "mutex_shard_num": 16, "batch_key_size": 32, "persist_meta_data_interval_time_ms": 1000, "meta_storage_backend_config": {"storage_type": "local"}}},
        "version": 1,
        "revisit_interval_buckets": "1,5,30,60"
    })";

    InstanceGroup parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_EQ("test_group", parsed.name());
    EXPECT_EQ("1,5,30,60", parsed.revisit_interval_buckets_raw());
    ASSERT_EQ(parsed.revisit_interval_buckets().size(), 4);
    EXPECT_DOUBLE_EQ(parsed.revisit_interval_buckets()[0], 1.0);
    EXPECT_DOUBLE_EQ(parsed.revisit_interval_buckets()[3], 60.0);
}

TEST_F(InstanceGroupTest, JsonRoundTripWithoutBuckets) {
    std::string json = R"({
        "name": "test_group",
        "storage_candidates": ["local"],
        "global_quota_group_name": "default",
        "max_instance_count": 10,
        "quota": {"capacity": 1024},
        "cache_config": {"meta_indexer_config": {"max_key_count": 1000, "mutex_shard_num": 16, "batch_key_size": 32, "persist_meta_data_interval_time_ms": 1000, "meta_storage_backend_config": {"storage_type": "local"}}},
        "version": 1
    })";

    InstanceGroup parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_EQ("test_group", parsed.name());
    EXPECT_TRUE(parsed.revisit_interval_buckets_raw().empty());
    EXPECT_TRUE(parsed.revisit_interval_buckets().empty());
}

TEST_F(InstanceGroupTest, JsonRoundTripInvalidBuckets) {
    std::string json = R"({
        "name": "test_group",
        "storage_candidates": ["local"],
        "global_quota_group_name": "default",
        "max_instance_count": 10,
        "quota": {"capacity": 1024},
        "cache_config": {"meta_indexer_config": {"max_key_count": 1000, "mutex_shard_num": 16, "batch_key_size": 32, "persist_meta_data_interval_time_ms": 1000, "meta_storage_backend_config": {"storage_type": "local"}}},
        "version": 1,
        "revisit_interval_buckets": "5,1,30"
    })";

    InstanceGroup parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    // Raw string preserved, but parsed vector is empty (invalid)
    EXPECT_EQ("5,1,30", parsed.revisit_interval_buckets_raw());
    EXPECT_TRUE(parsed.revisit_interval_buckets().empty());
}

// --- Proto round-trip ---

TEST_F(InstanceGroupTest, ProtoRoundTripWithBuckets) {
    InstanceGroup original;
    original.set_name("test_group");
    original.set_storage_candidates({"local"});
    original.set_global_quota_group_name("default");
    original.set_max_instance_count(10);
    original.set_version(1);
    original.set_revisit_interval_buckets("1,5,30,60");

    // ToProto
    proto::admin::InstanceGroup proto_msg;
    ProtoConvert::InstanceGroupToProto(original, &proto_msg);
    EXPECT_EQ("1,5,30,60", proto_msg.revisit_interval_buckets());

    // FromProto
    InstanceGroup restored;
    ProtoConvert::InstanceGroupFromProto(&proto_msg, restored);
    EXPECT_EQ("test_group", restored.name());
    EXPECT_EQ("1,5,30,60", restored.revisit_interval_buckets_raw());
    ASSERT_EQ(restored.revisit_interval_buckets().size(), 4);
    EXPECT_DOUBLE_EQ(restored.revisit_interval_buckets()[0], 1.0);
    EXPECT_DOUBLE_EQ(restored.revisit_interval_buckets()[3], 60.0);
}

TEST_F(InstanceGroupTest, ProtoRoundTripWithoutBuckets) {
    InstanceGroup original;
    original.set_name("test_group");
    original.set_storage_candidates({"local"});
    original.set_global_quota_group_name("default");
    original.set_max_instance_count(10);
    original.set_version(1);
    // revisit_interval_buckets not set

    proto::admin::InstanceGroup proto_msg;
    ProtoConvert::InstanceGroupToProto(original, &proto_msg);
    EXPECT_TRUE(proto_msg.revisit_interval_buckets().empty());

    InstanceGroup restored;
    ProtoConvert::InstanceGroupFromProto(&proto_msg, restored);
    EXPECT_TRUE(restored.revisit_interval_buckets_raw().empty());
    EXPECT_TRUE(restored.revisit_interval_buckets().empty());
}

} // namespace kv_cache_manager
