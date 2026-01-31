#include <memory>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/instance_config.h"
#include "kv_cache_manager/optimizer/config/instance_group_config.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/config/tier_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/manager/optimizer_manager.h"

using namespace kv_cache_manager;

class OptimizerManagerTest : public TESTBASE {
public:
    void SetUp() override { config_ = CreateTestOptimizerConfig(); }

protected:
    OptimizerConfig CreateTestOptimizerConfig();
    OptimizerConfig config_;
};

OptimizerConfig OptimizerManagerTest::CreateTestOptimizerConfig() {
    OptimizerConfig config;
    config.set_trace_file_path("/tmp/test_trace.json");
    config.set_output_result_path("/tmp/test_result.json");
    config.set_trace_type(TraceType::TRACE_PUBLISHER_LOG);

    EvictionConfig eviction_config;
    eviction_config.set_eviction_mode(EvictionMode::EVICTION_MODE_INSTANCE_PRECISE);
    eviction_config.set_eviction_batch_size_per_instance(10);
    config.set_eviction_params(eviction_config);
    // 创建实例组配置
    OptInstanceGroupConfig instance_group;
    instance_group.set_group_name("test_group");
    instance_group.set_quota_capacity(1024 * 1024 * 100); // 100MB
    instance_group.set_used_percentage(0.0);
    instance_group.set_hierarchical_eviction_enabled(false);

    OptTierConfig tier1;
    tier1.set_unique_name("tier1");
    tier1.set_capacity(1024 * 1024 * 10);
    tier1.set_storage_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    tier1.set_band_width_mbps(1000);
    tier1.set_priority(1);
    instance_group.set_storages({tier1});

    // 添加实例配置到实例组
    OptInstanceConfig instance1;
    instance1.set_instance_id("instance1");
    instance1.set_instance_group_name("test_group");
    instance1.set_block_size(1024);
    LruParams params;
    params.sample_rate = 1.0;
    EvictionPolicyParam policy_param;
    policy_param = params;
    instance1.set_eviction_policy_param(policy_param);
    instance1.set_eviction_policy_type(EvictionPolicyType::POLICY_LRU);

    instance_group.set_instances({instance1});

    config.set_instance_groups({instance_group});

    return config;
}

TEST_F(OptimizerManagerTest, BasicInitialization) {
    OptimizerManager manager(config_);
    EXPECT_TRUE(manager.Init());
}