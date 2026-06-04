#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/online_optimizer/config/optimizer_instance_group.h"
#include "kv_cache_manager/online_optimizer/config/optimizer_instance_info.h"

namespace kv_cache_manager {

class OptimizerInstanceGroupTest : public TESTBASE {};

TEST_F(OptimizerInstanceGroupTest, DefaultValues) {
    OptimizerInstanceGroup group;
    EXPECT_TRUE(group.name().empty());
    EXPECT_FALSE(group.enabled());
    EXPECT_TRUE(group.capacity_gb().empty());
    EXPECT_EQ("lru", group.indexer_type());
    EXPECT_EQ(0, group.max_key_count());
}

TEST_F(OptimizerInstanceGroupTest, SerializeDeserialize) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_enabled(true);
    group.set_capacity_gb({40.0, 80.0, 120.0});
    group.set_indexer_type("bst_lru");
    group.set_max_key_count(10000);

    std::string json = group.ToJsonString();

    OptimizerInstanceGroup group2;
    ASSERT_TRUE(group2.FromJsonString(json));
    EXPECT_EQ("g1", group2.name());
    EXPECT_TRUE(group2.enabled());
    EXPECT_EQ(3, group2.capacity_gb().size());
    EXPECT_DOUBLE_EQ(40.0, group2.capacity_gb()[0]);
    EXPECT_DOUBLE_EQ(80.0, group2.capacity_gb()[1]);
    EXPECT_DOUBLE_EQ(120.0, group2.capacity_gb()[2]);
    EXPECT_EQ("bst_lru", group2.indexer_type());
    EXPECT_EQ(10000, group2.max_key_count());
}

TEST_F(OptimizerInstanceGroupTest, ValidateEmptyName) {
    OptimizerInstanceGroup group;
    group.set_enabled(true);
    group.set_capacity_gb({1.0});
    std::string fields;
    EXPECT_FALSE(group.ValidateRequiredFields(fields));
    EXPECT_NE(std::string::npos, fields.find("name"));
}

TEST_F(OptimizerInstanceGroupTest, ValidateEnabledNoCapacity) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_enabled(true);
    std::string fields;
    EXPECT_FALSE(group.ValidateRequiredFields(fields));
    EXPECT_NE(std::string::npos, fields.find("capacity_gb"));
}

TEST_F(OptimizerInstanceGroupTest, ValidateEnabledWithCapacity) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_enabled(true);
    group.set_capacity_gb({1.0});
    std::string fields;
    EXPECT_TRUE(group.ValidateRequiredFields(fields));
}

TEST_F(OptimizerInstanceGroupTest, ValidateInvalidIndexerType) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_enabled(true);
    group.set_capacity_gb({1.0});
    group.set_indexer_type("unknown");
    std::string fields;
    EXPECT_FALSE(group.ValidateRequiredFields(fields));
    EXPECT_NE(std::string::npos, fields.find("indexer_type"));
}

TEST_F(OptimizerInstanceGroupTest, ValidateNegativeMaxKeyCount) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_enabled(true);
    group.set_capacity_gb({1.0});
    group.set_max_key_count(-1);
    std::string fields;
    EXPECT_FALSE(group.ValidateRequiredFields(fields));
    EXPECT_NE(std::string::npos, fields.find("max_key_count"));
}

TEST_F(OptimizerInstanceGroupTest, ValidateDisabledAlwaysOk) {
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_enabled(false);
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
    EXPECT_TRUE(info.full_group_name().empty());
}

TEST_F(OptimizerInstanceInfoTest, ConstructorAndAccessors) {
    std::vector<LocationSpecInfo> specs = {LocationSpecInfo("tp0", 8192), LocationSpecInfo("tp1", 4096)};
    std::vector<LocationSpecGroup> groups = {LocationSpecGroup("F0", {"tp0", "tp1"})};

    OptimizerInstanceInfo info("grp1", "inst1", 16, specs, groups, 3, "F0");
    EXPECT_EQ("grp1", info.instance_group_name());
    EXPECT_EQ("inst1", info.instance_id());
    EXPECT_EQ(16, info.block_size());
    EXPECT_EQ(2, info.location_spec_infos().size());
    EXPECT_EQ(1, info.location_spec_groups().size());
    EXPECT_EQ(3, info.linear_step());
    EXPECT_EQ("F0", info.full_group_name());
}

TEST_F(OptimizerInstanceInfoTest, SerializeDeserialize) {
    std::vector<LocationSpecInfo> specs = {LocationSpecInfo("tp0", 8192), LocationSpecInfo("tp1", 4096)};
    std::vector<LocationSpecGroup> groups = {LocationSpecGroup("F0", {"tp0", "tp1"})};

    OptimizerInstanceInfo info("grp1", "inst1", 16, specs, groups, 4, "F0");
    std::string json = info.ToJsonString();

    OptimizerInstanceInfo info2;
    ASSERT_TRUE(info2.FromJsonString(json));
    EXPECT_EQ("grp1", info2.instance_group_name());
    EXPECT_EQ("inst1", info2.instance_id());
    EXPECT_EQ(16, info2.block_size());
    EXPECT_EQ(2, info2.location_spec_infos().size());
    EXPECT_EQ("tp0", info2.location_spec_infos()[0].name());
    EXPECT_EQ(8192, info2.location_spec_infos()[0].size());
    EXPECT_EQ(1, info2.location_spec_groups().size());
    EXPECT_EQ("F0", info2.location_spec_groups()[0].name());
    EXPECT_EQ(4, info2.linear_step());
    EXPECT_EQ("F0", info2.full_group_name());
}

} // namespace kv_cache_manager
