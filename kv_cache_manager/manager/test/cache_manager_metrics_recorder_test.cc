#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager_metrics_recorder.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend_manager.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"

using namespace kv_cache_manager;

class CacheManagerMetricsRecorderTest : public TESTBASE {
public:
    void SetUp() override {
        auto lifecycle = std::make_shared<MetricsLifecycle>();
        recorder_ = std::make_shared<CacheManagerMetricsRecorder>(nullptr, nullptr, nullptr, lifecycle);
    }

    // directly populate the recorder's snapshot maps (via
    // -fno-access-control) to unit-test the mutator methods without
    // needing a full RecorderLoop
    void SeedSnapshot() {
        recorder_->group_usage_ratio_map_["grp1"] = 0.5;
        recorder_->group_usage_ratio_map_["grp2"] = 0.3;

        recorder_->group_instance_id_metric_map_["grp1"]["inst1"] = {10, 100};
        recorder_->group_instance_id_metric_map_["grp1"]["inst2"] = {20, 200};
        recorder_->group_instance_id_metric_map_["grp2"]["inst3"] = {30, 300};
        recorder_->group_instance_id_metric_map_["grp2"]["inst1"] = {5, 50};
    }

    std::shared_ptr<CacheManagerMetricsRecorder> recorder_;
};

TEST_F(CacheManagerMetricsRecorderTest, RemoveGroupErasesGroupAndItsInstances) {
    SeedSnapshot();

    recorder_->RemoveGroup("grp1");

    auto ratio_map = recorder_->group_usage_ratio_map();
    ASSERT_EQ(ratio_map.end(), ratio_map.find("grp1"));
    ASSERT_NE(ratio_map.end(), ratio_map.find("grp2"));

    auto inst_map = recorder_->group_instance_id_metric_map();
    ASSERT_EQ(inst_map.end(), inst_map.find("grp1"));
    ASSERT_NE(inst_map.end(), inst_map.find("grp2"));
    // grp2 instances untouched
    ASSERT_EQ(2, inst_map["grp2"].size());
}

TEST_F(CacheManagerMetricsRecorderTest, RemoveInstanceErasesAcrossAllGroups) {
    SeedSnapshot();

    // inst1 exists in both grp1 and grp2
    recorder_->RemoveInstance("inst1");

    auto inst_map = recorder_->group_instance_id_metric_map();
    // grp1 should only have inst2
    ASSERT_EQ(1, inst_map["grp1"].size());
    ASSERT_NE(inst_map["grp1"].end(), inst_map["grp1"].find("inst2"));
    ASSERT_EQ(inst_map["grp1"].end(), inst_map["grp1"].find("inst1"));
    // grp2 should only have inst3
    ASSERT_EQ(1, inst_map["grp2"].size());
    ASSERT_NE(inst_map["grp2"].end(), inst_map["grp2"].find("inst3"));
    ASSERT_EQ(inst_map["grp2"].end(), inst_map["grp2"].find("inst1"));

    // group entries themselves remain (not erased)
    auto ratio_map = recorder_->group_usage_ratio_map();
    ASSERT_NE(ratio_map.end(), ratio_map.find("grp1"));
    ASSERT_NE(ratio_map.end(), ratio_map.find("grp2"));
}

TEST_F(CacheManagerMetricsRecorderTest, RemoveGroupOnAbsentGroupIsNoOp) {
    SeedSnapshot();

    // removing a non-existent group should not crash or affect others
    recorder_->RemoveGroup("nonexistent");

    auto ratio_map = recorder_->group_usage_ratio_map();
    ASSERT_EQ(2, ratio_map.size());
    auto inst_map = recorder_->group_instance_id_metric_map();
    ASSERT_EQ(2, inst_map.size());
}

TEST_F(CacheManagerMetricsRecorderTest, EmptyCachedInstanceHasZeroLruAge) {
    auto registry_manager = std::make_shared<RegistryManager>("", nullptr);
    auto group = std::make_shared<InstanceGroup>();
    group->set_name("grp1");
    registry_manager->instance_group_configs_["grp1"] = group;
    auto instance = std::make_shared<InstanceInfo>();
    instance->set_instance_group_name("grp1");
    instance->set_instance_id("inst1");
    registry_manager->instance_infos_["inst1"] = instance;

    auto indexer_manager = std::make_shared<MetaIndexerManager>();
    auto config = std::make_shared<MetaIndexerConfig>();
    auto backend_config = std::make_shared<MetaStorageBackendConfig>("cached");
    backend_config->SetStorageUri("file://" + GetPrivateTestRuntimeDataPath() +
                                  "lru_age?persistent_type=dummy&cache_type=local&num_shard_bits=0");
    config->SetMetaStorageBackendConfig(backend_config);
    ASSERT_EQ(EC_OK, indexer_manager->CreateMetaIndexer("inst1", config));
    auto indexer = indexer_manager->GetMetaIndexer("inst1");
    auto *backend = dynamic_cast<MetaLocalBackend *>(indexer->backend_manager_->cache_backend_.get());
    ASSERT_NE(nullptr, backend);
    indexer->backend_manager_->recover_thread_.join();

    // Start a fresh recorder for each sample so no five-second polling delay
    // or previously published snapshot can hide a state transition.
    auto record_metric = [&](CacheManagerMetricsRecorder::InstanceMetric &metric) {
        CacheManagerMetricsRecorder recorder(
            indexer_manager, nullptr, registry_manager, std::make_shared<MetricsLifecycle>());
        recorder.Start();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            auto snapshot = recorder.group_instance_id_metric_map();
            if (snapshot.count("grp1") && snapshot.at("grp1").count("inst1")) {
                metric = snapshot.at("grp1").at("inst1");
                recorder.Stop();
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        recorder.Stop();
        return false;
    };
    CacheManagerMetricsRecorder::InstanceMetric metric;
    ASSERT_TRUE(record_metric(metric));
    EXPECT_EQ(0u, metric.key_count);
    EXPECT_EQ(0, metric.max_lru_age_us);

    RequestContext context("lru_age_test");
    auto location = std::make_shared<CacheLocation>();
    location->set_id("loc");
    location->set_status(CLS_SERVING);
    CacheLocationMapVector locations(2, {{"loc", location}});
    PropertyMapVector properties(2);
    ASSERT_EQ(EC_OK, indexer->Put(&context, {1, 2}, locations, properties).ec);
    ASSERT_TRUE(record_metric(metric));
    EXPECT_EQ(2u, metric.key_count);
    EXPECT_GT(metric.max_lru_age_us, 0);

    ASSERT_EQ(EC_OK, indexer->Delete(&context, {1}).ec);
    ASSERT_TRUE(record_metric(metric));
    EXPECT_EQ(1u, metric.key_count);
    EXPECT_GT(metric.max_lru_age_us, 0);

    ASSERT_EQ(EC_OK, indexer->Delete(&context, {2}).ec);
    ASSERT_EQ(0u, indexer->GetKeyCount());
    ASSERT_TRUE(record_metric(metric));
    EXPECT_EQ(0, metric.max_lru_age_us);

    // Reproduce a stale timestamp after the instance has lost all its keys.
    // Empty-instance metrics must remain zero even when this signal lingers.
    const int64_t stale_time = TimestampUtil::GetCurrentTimeUs() - 60 * 1000000;
    backend->shard_oldest_access_time_[0].store(stale_time);
    ASSERT_EQ(stale_time, indexer->GetOldestAccessTime());
    for (int sample = 0; sample < 2; ++sample) {
        ASSERT_TRUE(record_metric(metric));
        EXPECT_EQ(0u, metric.key_count);
        EXPECT_EQ(0, metric.max_lru_age_us);
    }

    locations.resize(1);
    properties.resize(1);
    ASSERT_EQ(EC_OK, indexer->Put(&context, {3}, locations, properties).ec);
    ASSERT_GT(indexer->GetOldestAccessTime(), stale_time);
    ASSERT_TRUE(record_metric(metric));
    EXPECT_EQ(1u, metric.key_count);
    EXPECT_GT(metric.max_lru_age_us, 0);
    EXPECT_LT(metric.max_lru_age_us, TimestampUtil::GetCurrentTimeUs() - stale_time);
}
