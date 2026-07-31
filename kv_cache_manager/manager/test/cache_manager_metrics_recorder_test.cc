#include <memory>
#include <string>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/manager/cache_manager_metrics_recorder.h"
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
