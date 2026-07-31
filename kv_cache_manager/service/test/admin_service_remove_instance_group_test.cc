#include <memory>
#include <set>
#include <string>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/service/admin_service_impl.h"

using namespace kv_cache_manager;

class AdminServiceRemoveInstanceGroupTest : public TESTBASE {
public:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        auto rm_registry = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("local://", rm_registry);
        registry_manager_->Init();

        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);

        admin_service_ =
            std::make_unique<AdminServiceImpl>(cache_manager_, nullptr, metrics_registry_, registry_manager_, nullptr);
        // explicitly enable leader-only requests so this test does not
        // silently depend on the default in ServiceImplBase
        admin_service_->EnableLeaderOnlyRequests();
    }

    void SeedInstanceGroup(const std::string &group_name) {
        RequestContext rc("seed-" + group_name);
        InstanceGroup ig;
        ig.set_name(group_name);
        auto ec = registry_manager_->CreateInstanceGroup(&rc, ig);
        ASSERT_EQ(EC_OK, ec);
    }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::unique_ptr<AdminServiceImpl> admin_service_;
};

TEST_F(AdminServiceRemoveInstanceGroupTest, PurgesGroupTaggedMetrics) {
    SeedInstanceGroup("grp1");

    // seed metrics: group-tagged and unrelated
    metrics_registry_->GetGauge("g1", {{"instance_group", "grp1"}});
    metrics_registry_->GetGauge("g2", {{"instance_group", "grp1"}});
    metrics_registry_->GetCounter("m1", {{"instance_id", "other"}});
    ASSERT_EQ(3, metrics_registry_->GetSize());

    RequestContext rc("test-trace");
    proto::admin::RemoveInstanceGroupRequest request;
    request.set_name("grp1");
    proto::admin::CommonResponse response;
    admin_service_->RemoveInstanceGroup(&rc, &request, &response);

    ASSERT_EQ(proto::admin::OK, response.header().status().code());

    // group-tagged metrics should be purged; only "other" remains
    ASSERT_EQ(1, metrics_registry_->GetSize());
    std::vector<MetricsRegistry::metrics_tuple_t> all;
    metrics_registry_->GetAllMetrics(all);
    ASSERT_EQ(1, all.size());
    auto &[name, tags, _] = all[0];
    ASSERT_EQ("other", tags.at("instance_id"));
}

TEST_F(AdminServiceRemoveInstanceGroupTest, EmptyGroupStillSucceeds) {
    SeedInstanceGroup("empty_grp");

    // seed metrics: one tagged with the group being removed, one unrelated
    metrics_registry_->GetGauge("g1", {{"instance_group", "empty_grp"}});
    metrics_registry_->GetCounter("m1", {{"instance_id", "other"}});
    ASSERT_EQ(2, metrics_registry_->GetSize());

    RequestContext rc("test-trace");
    proto::admin::RemoveInstanceGroupRequest request;
    request.set_name("empty_grp");
    proto::admin::CommonResponse response;
    admin_service_->RemoveInstanceGroup(&rc, &request, &response);

    ASSERT_EQ(proto::admin::OK, response.header().status().code());
    // instance_group-tagged metric should be purged
    ASSERT_EQ(1, metrics_registry_->GetSize());
    std::vector<MetricsRegistry::metrics_tuple_t> all;
    metrics_registry_->GetAllMetrics(all);
    ASSERT_EQ(1, all.size());
    auto &[name, tags, _] = all[0];
    ASSERT_EQ("other", tags.at("instance_id"));
}

// documents the contract-violation behavior: caller is responsible
// for draining instances before removing the group
TEST_F(AdminServiceRemoveInstanceGroupTest, RemoveInstanceGroupWithRegisteredInstancesLeavesInstancesIntact) {
    SeedInstanceGroup("grp1");

    // seed an instance directly into the registry (bypasses full
    // RegisterInstance flow; sufficient to test removal semantics)
    auto info = std::make_shared<InstanceInfo>();
    info->set_instance_id("inst1");
    info->set_instance_group_name("grp1");
    registry_manager_->instance_infos_["inst1"] = info;

    // seed group-tagged and instance-tagged metrics
    metrics_registry_->GetGauge("group_metric", {{"instance_group", "grp1"}});
    metrics_registry_->GetGauge("inst_metric", {{"instance_id", "inst1"}});
    ASSERT_EQ(2, metrics_registry_->GetSize());

    RequestContext rc("test-trace");
    proto::admin::RemoveInstanceGroupRequest request;
    request.set_name("grp1");
    proto::admin::CommonResponse response;
    admin_service_->RemoveInstanceGroup(&rc, &request, &response);

    ASSERT_EQ(proto::admin::OK, response.header().status().code());

    // group config is gone
    RequestContext rc2("verify");
    auto [ec, _ig] = registry_manager_->GetInstanceGroup(&rc2, "grp1");
    ASSERT_NE(EC_OK, ec);

    // group-tagged metrics are purged, but instance-tagged metrics remain
    ASSERT_EQ(1, metrics_registry_->GetSize());
    std::vector<MetricsRegistry::metrics_tuple_t> all;
    metrics_registry_->GetAllMetrics(all);
    ASSERT_EQ(1, all.size());
    auto &[name, tags, metric] = all[0];
    ASSERT_EQ("inst1", tags.at("instance_id"));

    // instance record still present in registry
    ASSERT_NE(registry_manager_->instance_infos_.end(), registry_manager_->instance_infos_.find("inst1"));
}
