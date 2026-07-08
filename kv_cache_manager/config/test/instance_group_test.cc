#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.pb.h"
#include "kv_cache_manager/service/util/manager_message_proto_util.h"

namespace kv_cache_manager {

class InstanceGroupTest : public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}
};

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

// --- ValidateRequiredFields revisit_interval_buckets ---

// Helper: validate only the revisit_interval_buckets check in isolation.
// Builds a fully valid InstanceGroup, then sets the buckets to test.
static std::pair<bool, std::string> ValidateBucketsOnly(const std::string &buckets_str) {
    InstanceGroup group;
    group.set_name("test");
    group.set_storage_candidates({"local"});
    group.set_global_quota_group_name("default");
    group.set_max_instance_count(10);
    auto quota = InstanceGroupQuota();
    quota.set_capacity(1024);
    group.set_quota(quota);

    auto reclaim = std::make_shared<CacheReclaimStrategy>();
    reclaim->set_storage_unique_name("local");
    auto meta_config = std::make_shared<MetaIndexerConfig>();
    auto cache_config = std::make_shared<CacheConfig>(
        CachePreferStrategy::CPS_PREFER_3FS, reclaim, meta_config);
    group.set_cache_config(cache_config);
    group.set_revisit_interval_buckets(buckets_str);

    std::string invalid_fields;
    bool result = group.ValidateRequiredFields(invalid_fields);
    return {result, invalid_fields};
}

TEST_F(InstanceGroupTest, ValidateRejectsInvalidBuckets) {
    auto [valid, invalid_fields] = ValidateBucketsOnly("5,1,30");  // invalid: not ascending
    EXPECT_FALSE(valid);
    EXPECT_NE(invalid_fields.find("revisit_interval_buckets"), std::string::npos);
}

TEST_F(InstanceGroupTest, ValidateAcceptsValidBuckets) {
    auto [valid, invalid_fields] = ValidateBucketsOnly("1,5,30,60");
    EXPECT_TRUE(valid);
}

TEST_F(InstanceGroupTest, ValidateAcceptsEmptyBuckets) {
    auto [valid, invalid_fields] = ValidateBucketsOnly("");
    EXPECT_TRUE(valid);
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
    // FromRapidValue is lenient — accepts and preserves raw, but parsed is empty
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_EQ("5,1,30", parsed.revisit_interval_buckets_raw());
    EXPECT_TRUE(parsed.revisit_interval_buckets().empty());

    // But ValidateRequiredFields should reject it
    std::string invalid_fields;
    EXPECT_FALSE(parsed.ValidateRequiredFields(invalid_fields));
    EXPECT_NE(invalid_fields.find("revisit_interval_buckets"), std::string::npos);
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
