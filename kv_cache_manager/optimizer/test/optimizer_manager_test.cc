#include <fstream>
#include <memory>
#include <stdexcept>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/config/replay_instance_config.h"
#include "kv_cache_manager/optimizer/config/replay_instance_group_config.h"
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

    EvictionConfig eviction_config;
    eviction_config.set_eviction_mode(EvictionMode::EVICTION_MODE_INSTANCE_PRECISE);
    eviction_config.set_eviction_batch_size_per_instance(10);
    config.set_eviction_params(eviction_config);
    // 创建实例组配置
    OptimizerReplayInstanceGroupConfig instance_group;
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
    OptimizerReplayInstanceConfig instance1;
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

TEST_F(OptimizerManagerTest, WriteCacheTtlSecondsUsesNanosecondTimestamps) {
    auto config = CreateTestOptimizerConfig();
    auto instance_groups = config.instance_groups();
    ASSERT_EQ(instance_groups.size(), 1);

    auto group = instance_groups[0];
    group.set_default_block_ttl_seconds(3600);
    auto instances = group.instances();
    ASSERT_EQ(instances.size(), 1);

    TtlParams ttl_params;
    ttl_params.fallback_on_pressure = false;
    instances[0].set_eviction_policy_type(EvictionPolicyType::POLICY_TTL);
    instances[0].set_eviction_policy_param(ttl_params);
    group.set_instances(instances);
    config.set_instance_groups({group});

    OptimizerManager manager(config);
    ASSERT_TRUE(manager.Init());

    const int64_t write_ts_ns = 1'000'000'000;
    manager.WriteCache("instance1", "write", write_ts_ns, {1}, 1);

    BlockMask remote_read_mask = std::vector<bool>{false};
    auto hit_before_expire = manager.GetCacheLocation(
        "instance1", "read_before_expire", write_ts_ns + 1'000'000, {1}, remote_read_mask, 1024);
    EXPECT_EQ(hit_before_expire.kvcm_hit_length, 1);

    auto hit_after_expire = manager.GetCacheLocation(
        "instance1", "read_after_expire", write_ts_ns + 2'000'000'000, {1}, remote_read_mask, 1024);
    EXPECT_EQ(hit_after_expire.kvcm_hit_length, 0);
}

TEST_F(OptimizerManagerTest, TemplateAnalysisReadRecordKeepsTraceIdAndKeys) {
    OptimizerManager manager(config_, false, true);
    ASSERT_TRUE(manager.Init());
    ASSERT_NE(manager.template_prefix_tracker_, nullptr);

    const std::vector<int64_t> keys = {1, 2, 3};
    manager.WriteCache("instance1", "write_trace", 1000, keys);

    BlockMask remote_read_mask = std::vector<bool>{false, false, false};
    manager.GetCacheLocation("instance1", "read_trace", 2000, keys, remote_read_mask, 3 * 1024);

    auto data_it = manager.template_prefix_tracker_->instance_data_.find("instance1");
    ASSERT_NE(data_it, manager.template_prefix_tracker_->instance_data_.end());
    ASSERT_EQ(data_it->second.trace_reads.size(), 1);

    EXPECT_EQ(data_it->second.trace_reads[0].trace_id, "read_trace");
    EXPECT_EQ(data_it->second.trace_reads[0].keys, keys);
}

TEST_F(OptimizerManagerTest, ReadUsesExplicitInputLen) {
    OptimizerManager manager(config_);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {1};
    manager.WriteCache("instance1", "write_trace", 1000, keys);

    BlockMask remote_read_mask = std::vector<bool>{false};
    auto res = manager.GetCacheLocation("instance1", "read_trace", 2000, keys, remote_read_mask, 1537);
    EXPECT_EQ(res.kvcm_hit_length, 1);

    const auto *last_read = manager.hit_rate_tracker_->LastReadRecord("instance1");
    ASSERT_NE(last_read, nullptr);
    EXPECT_EQ(last_read->input_tokens, 1537);
    EXPECT_EQ(last_read->remote_hit_blocks, 1);
}

TEST_F(OptimizerManagerTest, ReadRejectsPartialTailBlockKeys) {
    OptimizerManager manager(config_);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {1, 2};
    manager.WriteCache("instance1", "write_trace", 1000, keys);

    BlockMask remote_read_mask = std::vector<bool>{false, false};
    EXPECT_THROW(manager.GetCacheLocation("instance1", "read_trace", 2000, keys, remote_read_mask, 1537),
                 std::runtime_error);
}

TEST_F(OptimizerManagerTest, ReadWithoutFullBlocksCountsInputTokens) {
    OptimizerManager manager(config_);
    ASSERT_TRUE(manager.Init());

    BlockMask remote_read_mask = std::vector<bool>{};
    auto res = manager.GetCacheLocation("instance1", "short_read", 1000, {}, remote_read_mask, 128);
    EXPECT_EQ(res.kvcm_hit_length, 0);

    const auto *last_read = manager.hit_rate_tracker_->LastReadRecord("instance1");
    ASSERT_NE(last_read, nullptr);
    EXPECT_EQ(last_read->input_tokens, 128);
    EXPECT_EQ(last_read->local_read_blocks, 0);
    EXPECT_EQ(last_read->remote_read_blocks, 0);
    EXPECT_EQ(last_read->local_hit_blocks, 0);
    EXPECT_EQ(last_read->remote_hit_blocks, 0);
}

TEST_F(OptimizerManagerTest, RequestTraceSchedulesDelayedWrite) {
    auto config = CreateTestOptimizerConfig();
    config.set_trace_file_path(GetTestTempRootPath() + "/request_trace.jsonl");
    config.set_output_result_path(GetTestTempRootPath() + "/request_trace_result");

    OptTraceReplayConfig trace_replay_config;
    trace_replay_config.set_write_delay_ns(1000);
    config.set_trace_replay_config(trace_replay_config);

    auto groups = config.instance_groups();
    ASSERT_EQ(groups.size(), 1);
    auto group = groups[0];
    group.set_quota_capacity(-1);
    group.set_used_percentage(1.0);
    auto instances = group.instances();
    ASSERT_EQ(instances.size(), 1);
    instances[0].set_block_size(16);
    instances[0].set_bytes_per_token(1);
    group.set_instances(instances);
    config.set_instance_groups({group});

    std::ofstream out(config.trace_file_path());
    out << R"({"type":"request","instance_id":"instance1","trace_id":"r1","timestamp_ns":1000,"keys":[1],"input_len":16,"block_mask":[]})"
        << "\n";
    out << R"({"type":"request","instance_id":"instance1","trace_id":"r2","timestamp_ns":1500,"keys":[1],"input_len":16,"block_mask":[]})"
        << "\n";
    out << R"({"type":"request","instance_id":"instance1","trace_id":"r3","timestamp_ns":2000,"keys":[1],"input_len":16,"block_mask":[]})"
        << "\n";
    out.close();

    OptimizerManager manager(config);
    ASSERT_TRUE(manager.Init());
    manager.DirectRun();

    auto data_it = manager.hit_rate_tracker_->instance_data_.find("instance1");
    ASSERT_NE(data_it, manager.hit_rate_tracker_->instance_data_.end());
    ASSERT_EQ(data_it->second.read_records.size(), 3);
    ASSERT_EQ(data_it->second.write_records.size(), 3);

    EXPECT_EQ(data_it->second.read_records[0].remote_hit_blocks, 0);
    EXPECT_EQ(data_it->second.read_records[1].remote_hit_blocks, 0);
    EXPECT_EQ(data_it->second.read_records[2].remote_hit_blocks, 1);
    EXPECT_EQ(data_it->second.write_records[0].timestamp_ns, 2000);
    EXPECT_EQ(data_it->second.write_records[1].timestamp_ns, 2500);
    EXPECT_EQ(data_it->second.write_records[2].timestamp_ns, 3000);
}
