#include <limits>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_group.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_info.h"
#include "kv_cache_manager/optimizer/config/replay_instance_config.h"
#include "kv_cache_manager/optimizer/config/replay_instance_group_config.h"

namespace kv_cache_manager {

class OptimizerInstanceGroupTest : public TESTBASE {};

TEST_F(OptimizerInstanceGroupTest, DefaultValues) {
    OptimizerInstanceGroup group;
    EXPECT_TRUE(group.name().empty());
    EXPECT_TRUE(group.capacity_gb().empty());
    EXPECT_EQ("lru", group.eviction_policy());
    EXPECT_FALSE(group.shared_group_quota());
    EXPECT_FALSE(group.enable_theoretical_max_cache());
    EXPECT_EQ(0, group.ttl_seconds());
}

TEST_F(OptimizerInstanceGroupTest, SerializeDeserialize) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_capacity_gb({40.0, 80.0, 120.0});
    group.set_ttl_seconds(3600);
    group.set_shared_group_quota(true);
    group.set_enable_theoretical_max_cache(true);

    std::string json = group.ToJsonString();

    OptimizerInstanceGroup group2;
    ASSERT_TRUE(group2.FromJsonString(json));
    EXPECT_EQ("g1", group2.name());
    EXPECT_EQ(3, group2.capacity_gb().size());
    EXPECT_DOUBLE_EQ(40.0, group2.capacity_gb()[0]);
    EXPECT_DOUBLE_EQ(80.0, group2.capacity_gb()[1]);
    EXPECT_DOUBLE_EQ(120.0, group2.capacity_gb()[2]);
    EXPECT_EQ("lru", group2.eviction_policy());
    EXPECT_EQ(3600, group2.ttl_seconds());
    EXPECT_TRUE(group2.shared_group_quota());
    EXPECT_TRUE(group2.enable_theoretical_max_cache());
}

TEST_F(OptimizerInstanceGroupTest, ValidateEmptyName) {
    OptimizerInstanceGroup group;
    group.set_capacity_gb({1.0});
    std::string fields;
    EXPECT_FALSE(group.ValidateRequiredFields(fields));
    EXPECT_NE(std::string::npos, fields.find("name"));
}

TEST_F(OptimizerInstanceGroupTest, ValidateNoCapacity) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    std::string fields;
    EXPECT_FALSE(group.ValidateRequiredFields(fields));
    EXPECT_NE(std::string::npos, fields.find("capacity_gb"));
}

TEST_F(OptimizerInstanceGroupTest, ValidateWithCapacity) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_capacity_gb({1.0});
    std::string fields;
    EXPECT_TRUE(group.ValidateRequiredFields(fields));
}

TEST_F(OptimizerInstanceGroupTest, ValidateCapacityTooLargeFails) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_capacity_gb({static_cast<double>(std::numeric_limits<int64_t>::max())});
    std::string fields;
    EXPECT_FALSE(group.ValidateRequiredFields(fields));
    EXPECT_NE(std::string::npos, fields.find("capacity_gb"));
}

TEST_F(OptimizerInstanceGroupTest, ValidateInvalidEvictionPolicyType) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_capacity_gb({1.0});
    group.set_eviction_policy("unknown");
    std::string fields;
    EXPECT_FALSE(group.ValidateRequiredFields(fields));
    EXPECT_NE(std::string::npos, fields.find("eviction_policy"));
}

TEST_F(OptimizerInstanceGroupTest, ValidateNegativeTtlSeconds) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_capacity_gb({1.0});
    group.set_ttl_seconds(-1);
    std::string fields;
    EXPECT_FALSE(group.ValidateRequiredFields(fields));
    EXPECT_NE(std::string::npos, fields.find("ttl_seconds"));
}

TEST_F(OptimizerInstanceGroupTest, ValidateZeroTtlDisablesTtl) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_capacity_gb({1.0});
    std::string fields;
    EXPECT_TRUE(group.ValidateRequiredFields(fields));
}

TEST_F(OptimizerInstanceGroupTest, ValidateTtlWithPositiveTtlSeconds) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_capacity_gb({1.0});
    group.set_ttl_seconds(10);
    std::string fields;
    EXPECT_TRUE(group.ValidateRequiredFields(fields));
}

class OptimizerInstanceInfoTest : public TESTBASE {};

TEST_F(OptimizerInstanceInfoTest, DefaultValues) {
    OptimizerInstanceInfo info;
    EXPECT_TRUE(info.instance_group_name().empty());
    EXPECT_TRUE(info.instance_id().empty());
    EXPECT_EQ(0, info.block_size());
    EXPECT_TRUE(info.location_spec_infos().empty());
    EXPECT_TRUE(info.location_spec_groups().empty());
    EXPECT_EQ(0, info.linear_step());
    EXPECT_TRUE(info.optimizer_state_info().full_location_spec_group_name().empty());
    EXPECT_TRUE(info.optimizer_state_info().linear_location_spec_group_name().empty());
}

TEST_F(OptimizerInstanceInfoTest, ConstructorAndAccessors) {
    std::vector<LocationSpecInfo> specs = {LocationSpecInfo("tp0", 8192), LocationSpecInfo("tp1", 4096)};
    std::vector<LocationSpecGroup> groups = {LocationSpecGroup("F0", {"tp0"}), LocationSpecGroup("L1", {"tp1"})};

    OptimizerInstanceInfo info("grp1", "inst1", 16, specs, groups, 3, OptimizerStateInfo("F0", "L1"));
    EXPECT_EQ("grp1", info.instance_group_name());
    EXPECT_EQ("inst1", info.instance_id());
    EXPECT_EQ(16, info.block_size());
    EXPECT_EQ(2, info.location_spec_infos().size());
    EXPECT_EQ(2, info.location_spec_groups().size());
    EXPECT_EQ(3, info.linear_step());
    EXPECT_EQ("F0", info.optimizer_state_info().full_location_spec_group_name());
    EXPECT_EQ("L1", info.optimizer_state_info().linear_location_spec_group_name());
}

TEST_F(OptimizerInstanceInfoTest, SerializeDeserialize) {
    std::vector<LocationSpecInfo> specs = {LocationSpecInfo("tp0", 8192), LocationSpecInfo("tp1", 4096)};
    std::vector<LocationSpecGroup> groups = {LocationSpecGroup("F0", {"tp0"}), LocationSpecGroup("L1", {"tp1"})};

    OptimizerInstanceInfo info("grp1", "inst1", 16, specs, groups, 4, OptimizerStateInfo("F0", "L1"));
    std::string json = info.ToJsonString();

    OptimizerInstanceInfo info2;
    ASSERT_TRUE(info2.FromJsonString(json));
    EXPECT_EQ("grp1", info2.instance_group_name());
    EXPECT_EQ("inst1", info2.instance_id());
    EXPECT_EQ(16, info2.block_size());
    EXPECT_EQ(2, info2.location_spec_infos().size());
    EXPECT_EQ("tp0", info2.location_spec_infos()[0].name());
    EXPECT_EQ(8192, info2.location_spec_infos()[0].size());
    EXPECT_EQ(2, info2.location_spec_groups().size());
    EXPECT_EQ("F0", info2.location_spec_groups()[0].name());
    EXPECT_EQ(4, info2.linear_step());
    EXPECT_EQ("F0", info2.optimizer_state_info().full_location_spec_group_name());
    EXPECT_EQ("L1", info2.optimizer_state_info().linear_location_spec_group_name());
}

class OptimizerReplayConfigTest : public TESTBASE {};

TEST_F(OptimizerReplayConfigTest, LegacyInstanceConfigCalculatesBytesPerBlock) {
    OptimizerReplayInstanceConfig legacy;
    legacy.set_instance_group_name("legacy_group");
    legacy.set_instance_id("legacy_inst");
    legacy.set_block_size(16);
    legacy.set_bytes_per_token(1024);
    legacy.set_eviction_policy_type(EvictionPolicyType::POLICY_LRU);

    EXPECT_EQ(16 * 1024, legacy.bytes_per_block());
    EXPECT_EQ("legacy_group", legacy.instance_group_name());
    EXPECT_EQ("legacy_inst", legacy.instance_id());
}

TEST_F(OptimizerReplayConfigTest, InstanceConfigJsonRequiresBytesPerToken) {
    const std::string json = R"JSON(
        {
            "instance_group_name": "group",
            "instance_id": "inst",
            "block_size": 16,
            "eviction_policy_type": "lru",
            "eviction_policy_params": {"sample_rate": 1.0}
        }
    )JSON";

    OptimizerReplayInstanceConfig config;
    EXPECT_FALSE(config.FromJsonString(json));
}

} // namespace kv_cache_manager
