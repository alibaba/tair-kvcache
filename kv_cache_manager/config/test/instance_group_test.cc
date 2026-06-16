#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"

namespace kv_cache_manager {

class InstanceGroupTest : public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}
};

// --- ParseRevisitIntervalBuckets ---

TEST_F(InstanceGroupTest, ParseEmptyStringReturnsEmpty) {
    auto result = InstanceGroup::ParseRevisitIntervalBuckets("");
    EXPECT_TRUE(result.empty());
}

TEST_F(InstanceGroupTest, ParseValidBoundaries) {
    auto result = InstanceGroup::ParseRevisitIntervalBuckets("1,5,30,60,300,3600");
    ASSERT_EQ(result.size(), 6);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 5.0);
    EXPECT_DOUBLE_EQ(result[2], 30.0);
    EXPECT_DOUBLE_EQ(result[3], 60.0);
    EXPECT_DOUBLE_EQ(result[4], 300.0);
    EXPECT_DOUBLE_EQ(result[5], 3600.0);
}

TEST_F(InstanceGroupTest, ParseWithSpaces) {
    auto result = InstanceGroup::ParseRevisitIntervalBuckets(" 1 , 5 , 30 ");
    ASSERT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 5.0);
    EXPECT_DOUBLE_EQ(result[2], 30.0);
}

TEST_F(InstanceGroupTest, ParseNonAscendingRejected) {
    auto result = InstanceGroup::ParseRevisitIntervalBuckets("5,1,30");
    EXPECT_TRUE(result.empty());
}

TEST_F(InstanceGroupTest, ParseDuplicateRejected) {
    auto result = InstanceGroup::ParseRevisitIntervalBuckets("1,5,5,30");
    EXPECT_TRUE(result.empty());
}

TEST_F(InstanceGroupTest, ParseNegativeRejected) {
    auto result = InstanceGroup::ParseRevisitIntervalBuckets("-1,5,30");
    EXPECT_TRUE(result.empty());
}

TEST_F(InstanceGroupTest, ParseZeroRejected) {
    auto result = InstanceGroup::ParseRevisitIntervalBuckets("0,5,30");
    EXPECT_TRUE(result.empty());
}

TEST_F(InstanceGroupTest, ParseTrailingCharsRejected) {
    auto result = InstanceGroup::ParseRevisitIntervalBuckets("1s,5,30");
    EXPECT_TRUE(result.empty());
}

TEST_F(InstanceGroupTest, ParseEmptyTokenRejected) {
    auto result = InstanceGroup::ParseRevisitIntervalBuckets("1,,5");
    EXPECT_TRUE(result.empty());
}

TEST_F(InstanceGroupTest, ParseLeadingCommaRejected) {
    auto result = InstanceGroup::ParseRevisitIntervalBuckets(",1,5");
    EXPECT_TRUE(result.empty());
}

TEST_F(InstanceGroupTest, ParseNonNumericRejected) {
    auto result = InstanceGroup::ParseRevisitIntervalBuckets("abc,5,30");
    EXPECT_TRUE(result.empty());
}

TEST_F(InstanceGroupTest, ParseFractionalBoundaries) {
    auto result = InstanceGroup::ParseRevisitIntervalBuckets("0.5,1.5,5.0");
    ASSERT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 0.5);
    EXPECT_DOUBLE_EQ(result[1], 1.5);
    EXPECT_DOUBLE_EQ(result[2], 5.0);
}

// --- JSON round-trip (new field only) ---

TEST_F(InstanceGroupTest, JsonRoundTripWithBuckets) {
    // Build a minimal valid JSON with revisit_interval_buckets
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
    EXPECT_EQ("1,5,30,60", parsed.revisit_interval_buckets());

    auto boundaries = InstanceGroup::ParseRevisitIntervalBuckets(parsed.revisit_interval_buckets());
    ASSERT_EQ(boundaries.size(), 4);
    EXPECT_DOUBLE_EQ(boundaries[0], 1.0);
    EXPECT_DOUBLE_EQ(boundaries[3], 60.0);
}

TEST_F(InstanceGroupTest, JsonRoundTripWithoutBuckets) {
    // JSON without revisit_interval_buckets — should default to empty
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
    EXPECT_TRUE(parsed.revisit_interval_buckets().empty());
}

} // namespace kv_cache_manager
