#include <memory>
#include <string>
#include <thread>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/service/meta_service_metrics_base.h"

using namespace kv_cache_manager;

class MetaServiceMetricsBaseTest : public TESTBASE {
public:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("local://", metrics_registry_);
        registry_manager_->Init();

        base_ = std::make_unique<MetaServiceMetricsBase>(metrics_registry_, registry_manager_);
        base_->InitMetrics();
    }

    // helper: seed a fake instance into registry_manager_ so that
    // GetInstanceGroupName returns the group name for the getter
    void SeedInstance(const std::string &instance_id, const std::string &group_name) {
        auto info = std::make_shared<InstanceInfo>();
        info->set_instance_id(instance_id);
        info->set_instance_group_name(group_name);
        registry_manager_->instance_infos_[instance_id] = info;
    }

    void RemoveInstance(const std::string &instance_id) { registry_manager_->instance_infos_.erase(instance_id); }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::unique_ptr<MetaServiceMetricsBase> base_;
};

TEST_F(MetaServiceMetricsBaseTest, InvalidateCollectorCacheEmptyIdIsNoOp) {
    // the global GetClusterInfo collector is seeded at empty key
    auto collector = base_->get_metrics_collector_from_map_for_GetClusterInfo("");
    ASSERT_NE(nullptr, collector);

    base_->InvalidateCollectorCache("");

    // still accessible — empty id guard prevented erasure
    auto after = base_->get_metrics_collector_from_map_for_GetClusterInfo("");
    ASSERT_NE(nullptr, after);
    ASSERT_EQ(collector, after);
}

TEST_F(MetaServiceMetricsBaseTest, InvalidateCollectorCacheRemovesEntry) {
    SeedInstance("inst1", "grp1");

    // create collector via getter (slow path)
    auto collector = base_->get_metrics_collector_from_map_for_GetCacheMeta("inst1");
    ASSERT_NE(nullptr, collector);

    // verify cached (fast path returns same pointer)
    auto cached = base_->get_metrics_collector_from_map_for_GetCacheMeta("inst1");
    ASSERT_EQ(collector, cached);

    // invalidate
    base_->InvalidateCollectorCache("inst1");

    // since instance still exists, getter recreates a new collector
    auto recreated = base_->get_metrics_collector_from_map_for_GetCacheMeta("inst1");
    ASSERT_NE(nullptr, recreated);
    ASSERT_NE(collector, recreated);
}

TEST_F(MetaServiceMetricsBaseTest, InvalidateCollectorCacheAllMaps) {
    SeedInstance("inst1", "grp1");

    // populate collectors in multiple API maps
    auto c1 = base_->get_metrics_collector_from_map_for_GetCacheMeta("inst1");
    auto c2 = base_->get_metrics_collector_from_map_for_GetCacheLocation("inst1");
    auto c3 = base_->get_metrics_collector_from_map_for_StartWriteCache("inst1");
    auto c4 = base_->get_metrics_collector_from_map_for_FinishWriteCache("inst1");
    auto c5 = base_->get_metrics_collector_from_map_for_RemoveCache("inst1");
    auto c6 = base_->get_metrics_collector_from_map_for_TrimCache("inst1");
    ASSERT_NE(nullptr, c1);
    ASSERT_NE(nullptr, c2);
    ASSERT_NE(nullptr, c3);
    ASSERT_NE(nullptr, c4);
    ASSERT_NE(nullptr, c5);
    ASSERT_NE(nullptr, c6);

    // invalidate
    base_->InvalidateCollectorCache("inst1");

    // all maps produce new (different) collectors
    ASSERT_NE(c1, base_->get_metrics_collector_from_map_for_GetCacheMeta("inst1"));
    ASSERT_NE(c2, base_->get_metrics_collector_from_map_for_GetCacheLocation("inst1"));
    ASSERT_NE(c3, base_->get_metrics_collector_from_map_for_StartWriteCache("inst1"));
    ASSERT_NE(c4, base_->get_metrics_collector_from_map_for_FinishWriteCache("inst1"));
    ASSERT_NE(c5, base_->get_metrics_collector_from_map_for_RemoveCache("inst1"));
    ASSERT_NE(c6, base_->get_metrics_collector_from_map_for_TrimCache("inst1"));
}

TEST_F(MetaServiceMetricsBaseTest, InvalidateAfterRemovalPreventsRecreation) {
    SeedInstance("inst1", "grp1");

    // create collector
    auto collector = base_->get_metrics_collector_from_map_for_GetCacheMeta("inst1");
    ASSERT_NE(nullptr, collector);

    // remove instance from registry
    RemoveInstance("inst1");

    // invalidate
    base_->InvalidateCollectorCache("inst1");

    // getter returns nullptr since instance no longer exists
    auto after = base_->get_metrics_collector_from_map_for_GetCacheMeta("inst1");
    ASSERT_EQ(nullptr, after);
}

TEST_F(MetaServiceMetricsBaseTest, InvalidateDoesNotAffectOtherInstances) {
    SeedInstance("inst1", "grp1");
    SeedInstance("inst2", "grp1");

    auto c1 = base_->get_metrics_collector_from_map_for_GetCacheMeta("inst1");
    auto c2 = base_->get_metrics_collector_from_map_for_GetCacheMeta("inst2");
    ASSERT_NE(nullptr, c1);
    ASSERT_NE(nullptr, c2);

    // invalidate only inst1
    base_->InvalidateCollectorCache("inst1");

    // inst2 still cached
    auto c2_after = base_->get_metrics_collector_from_map_for_GetCacheMeta("inst2");
    ASSERT_EQ(c2, c2_after);
}

TEST_F(MetaServiceMetricsBaseTest, SlowPathCollectorCreationRacesWithRemoval) {
    auto lifecycle = std::make_shared<MetricsLifecycle>();
    auto base = std::make_unique<MetaServiceMetricsBase>(metrics_registry_, registry_manager_, lifecycle);
    base->InitMetrics();

    SeedInstance("inst1", "grp1");

    constexpr int kIterations = 200;

    // writer: simulates RemoveInstance path — takes unique lock,
    // invalidates cache, removes metrics by tag filter
    std::thread writer([&]() {
        for (int i = 0; i < kIterations; ++i) {
            std::unique_lock<std::shared_mutex> guard(lifecycle->mut_);
            base->InvalidateCollectorCache("inst1");
            metrics_registry_->RemoveByTagFilter({{"instance_id", "inst1"}});
        }
    });

    // reader: simulates the slow-path getter which internally takes
    // shared lock on lifecycle->mu
    std::thread reader([&]() {
        for (int i = 0; i < kIterations; ++i) {
            auto collector = base->get_metrics_collector_from_map_for_GetCacheMeta("inst1");
            // collector may be non-null (created) or null (if the writer
            // raced and removed the instance from the registry); both are
            // valid outcomes — we only care about no crashes / no data
            // races
            (void)collector;
        }
    });

    writer.join();
    reader.join();

    // instance still seeded, so a final get should succeed
    auto final_collector = base->get_metrics_collector_from_map_for_GetCacheMeta("inst1");
    ASSERT_NE(nullptr, final_collector);
}
