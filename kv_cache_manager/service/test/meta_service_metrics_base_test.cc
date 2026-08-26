#include <memory>
#include <set>
#include <string>
#include <thread>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
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

TEST_F(MetaServiceMetricsBaseTest, TypedReportEventCollectorUsesTypeTagAndStableKey) {
    SeedInstance("inst1", "grp1");

    auto l1p5 = base_->GetTypedMetricsCollectorForReportEvent("inst1", "event_report_l1p5");
    auto l2 = base_->GetTypedMetricsCollectorForReportEvent("inst1", "event_report_l2");
    ASSERT_NE(nullptr, l1p5);
    ASSERT_NE(nullptr, l2);
    ASSERT_NE(l1p5, l2);

    MetricsTags l1p5_tags = {{"api_name", "ReportEvent"},
                             {"instance_group", "grp1"},
                             {"instance_id", "inst1"},
                             {"type", "event_report_l1p5"}};
    MetricsTags l2_tags = {
        {"api_name", "ReportEvent"}, {"instance_group", "grp1"}, {"instance_id", "inst1"}, {"type", "event_report_l2"}};
    ASSERT_EQ(l1p5_tags, l1p5->GetMetricsTags());
    ASSERT_EQ(l2_tags, l2->GetMetricsTags());

    auto l1p5_cached = base_->GetTypedMetricsCollectorForReportEvent("inst1", "event_report_l1p5");
    ASSERT_EQ(l1p5, l1p5_cached);
}

TEST_F(MetaServiceMetricsBaseTest, ReportEventMetricsFallbackDoesNotGateRequestValidation) {
    proto::meta::ReportEventRequest request;
    request.set_instance_id("unknown-instance");
    request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);

    std::string metrics_type;
    auto collector = base_->ResolveReportEventMetricsCollector(request, metrics_type);
    ASSERT_NE(nullptr, collector);
    EXPECT_EQ("event_report_l2", metrics_type);
    EXPECT_EQ((MetricsTags{{"api_name", "ReportEvent"}}), collector->GetMetricsTags());

    request.clear_instance_id();
    request.set_storage_type(proto::meta::ST_UNSPECIFIED);
    auto invalid_request_collector = base_->ResolveReportEventMetricsCollector(request, metrics_type);
    EXPECT_EQ(collector, invalid_request_collector);
    EXPECT_TRUE(metrics_type.empty());
}

TEST_F(MetaServiceMetricsBaseTest, InvalidateCollectorCacheRemovesTypedReportEventEntries) {
    SeedInstance("inst1", "grp1");

    auto report_event = base_->GetTypedMetricsCollectorForReportEvent("inst1", "event_report_l1p5");
    auto block_add = base_->GetTypedMetricsCollectorForReportEventType("inst1", "event_report_l1p5", "block_add");
    auto block_delete = base_->GetTypedMetricsCollectorForReportEventType("inst1", "event_report_l2", "block_delete");
    ASSERT_NE(nullptr, report_event);
    ASSERT_NE(nullptr, block_add);
    ASSERT_NE(nullptr, block_delete);
    ASSERT_NE(
        nullptr,
        dynamic_cast<EventReportMetricsCollector *>(
            base_->GetTypedMetricsCollectorForReportEventType("inst1", "event_report_l1p5", "block_snapshot").get()));

    base_->InvalidateCollectorCache("inst1");

    ASSERT_NE(report_event, base_->GetTypedMetricsCollectorForReportEvent("inst1", "event_report_l1p5"));
    ASSERT_NE(block_add, base_->GetTypedMetricsCollectorForReportEventType("inst1", "event_report_l1p5", "block_add"));
    ASSERT_NE(block_delete,
              base_->GetTypedMetricsCollectorForReportEventType("inst1", "event_report_l2", "block_delete"));
}

TEST_F(MetaServiceMetricsBaseTest, ReportEventTypeCollectorUsesBoundedTagsAndStableKey) {
    SeedInstance("inst1", "grp1");

    auto snapshot = base_->GetTypedMetricsCollectorForReportEventType("inst1", "event_report_l2", "block_snapshot");
    ASSERT_NE(nullptr, snapshot);
    ASSERT_NE(nullptr, dynamic_cast<EventReportMetricsCollector *>(snapshot.get()));
    MetricsTags expected_tags = {{"api_name", "ReportEvent"},
                                 {"instance_group", "grp1"},
                                 {"instance_id", "inst1"},
                                 {"type", "event_report_l2"},
                                 {"event_type", "block_snapshot"}};
    EXPECT_EQ(expected_tags, snapshot->GetMetricsTags());
    EXPECT_EQ(snapshot,
              base_->GetTypedMetricsCollectorForReportEventType("inst1", "event_report_l2", "block_snapshot"));
}

TEST_F(MetaServiceMetricsBaseTest, AttachesCollectorsOnlyForBlockMutationEventTypes) {
    SeedInstance("inst1", "grp1");
    proto::meta::ReportEventRequest request;
    request.set_instance_id("inst1");
    auto *snapshot_event = request.add_events();
    snapshot_event->set_event_type(proto::meta::EVENT_BLOCK_SNAPSHOT);
    snapshot_event->mutable_block_snapshot()->add_blocks()->set_block_key("1");
    snapshot_event->mutable_block_snapshot()->add_blocks()->set_block_key("2");
    request.add_events()->set_event_type(proto::meta::EVENT_BLOCK_SNAPSHOT);
    request.add_events()->set_event_type(proto::meta::EVENT_HEARTBEAT);
    request.add_events()->set_event_type(static_cast<proto::meta::ReportEventType>(99));
    RequestContext request_context("trace");

    base_->AttachReportEventTypeMetricsCollectors(request, "event_report_l2", &request_context);

    const auto collectors = request_context.GetMetricsCollectorsVehicle().GetMetricsCollectors();
    ASSERT_EQ(1, collectors.size());
    std::set<std::string> event_types;
    for (const auto &collector : collectors) {
        auto *event_collector = dynamic_cast<EventReportMetricsCollector *>(collector.get());
        ASSERT_NE(nullptr, event_collector);
        const auto &event_type = collector->GetMetricsTags().at("event_type");
        event_types.insert(event_type);
        if (event_type == "block_snapshot") {
            EXPECT_TRUE(event_collector->HasRequestKeyCountSample());
            EXPECT_DOUBLE_EQ(2., event_collector->GetRequestKeyCountSample());
        }
    }
    EXPECT_EQ((std::set<std::string>{"block_snapshot"}), event_types);
}

TEST_F(MetaServiceMetricsBaseTest, AttachedEventCollectorsHaveRequestLocalSamplesAndSharedCounters) {
    SeedInstance("inst1", "grp1");
    proto::meta::ReportEventRequest request;
    request.set_instance_id("inst1");
    request.add_events()->set_event_type(proto::meta::EVENT_BLOCK_SNAPSHOT);
    RequestContext first_context("first");
    RequestContext second_context("second");

    base_->AttachReportEventTypeMetricsCollectors(request, "event_report_l2", &first_context);
    base_->AttachReportEventTypeMetricsCollectors(request, "event_report_l2", &second_context);
    const auto first_collectors = first_context.GetMetricsCollectorsVehicle().GetMetricsCollectors();
    const auto second_collectors = second_context.GetMetricsCollectorsVehicle().GetMetricsCollectors();
    ASSERT_EQ(1, first_collectors.size());
    ASSERT_EQ(1, second_collectors.size());
    auto first = std::dynamic_pointer_cast<EventReportMetricsCollector>(first_collectors.front());
    auto second = std::dynamic_pointer_cast<EventReportMetricsCollector>(second_collectors.front());
    ASSERT_NE(nullptr, first);
    ASSERT_NE(nullptr, second);
    EXPECT_NE(first.get(), second.get());

    first->SetRequestSample(10.0, 0.0);
    second->SetRequestSample(20.0, 1.0);
    EXPECT_DOUBLE_EQ(10.0, first->GetRequestRtUsSample());
    EXPECT_DOUBLE_EQ(0.0, first->GetErrorCodeSample());
    EXPECT_DOUBLE_EQ(20.0, second->GetRequestRtUsSample());
    EXPECT_DOUBLE_EQ(1.0, second->GetErrorCodeSample());

    Counter first_counter;
    Counter second_counter;
    first->copy_service_query_counter_metrics(first_counter);
    second->copy_service_query_counter_metrics(second_counter);
    ++first_counter;
    EXPECT_EQ(1u, second_counter.Get());
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
