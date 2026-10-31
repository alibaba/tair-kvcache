#include <array>
#include <memory>
#include <vector>

#include "kmonitor/client/core/MetricsTags.h"
#include "kmonitor/client/core/MutableMetric.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/metrics/kmonitor_metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "stub.h"

using namespace kv_cache_manager;

namespace {
thread_local double reported_error_qps = 0;
thread_local double reported_grpc_qps = 0;
thread_local MetricsTags reported_grpc_tags;

struct ReportedLifetime {
    std::string name;
    MetricsTags tags;
    double value;
};
thread_local std::vector<ReportedLifetime> reported_lifetimes;

void CaptureEventAges(kmonitor::MutableMetric *metric, const kmonitor::MetricsTags *tags, double value) {
    const auto pos = metric->metric_name_.find("cache_manager_instance.event_report_location_lifetime_");
    if (pos != std::string::npos) {
        reported_lifetimes.push_back({metric->metric_name_.substr(pos), tags->GetTagsMap(), value});
    }
}

void ExpectLifetime(const std::vector<ReportedLifetime> &reported,
                    size_t offset,
                    const MetricsTags &tags,
                    const EventReportMetricsCollector::LocationLifetimeSample &sample) {
    ASSERT_GE(reported.size(), offset + 3);
    const std::array<const char *, 3> suffixes{"min", "max", "avg"};
    const std::array<int64_t, 3> values{sample.min_us, sample.max_us, sample.avg_us};
    for (size_t i = 0; i < suffixes.size(); ++i) {
        EXPECT_EQ(std::string("cache_manager_instance.event_report_location_lifetime_") + suffixes[i] + "_us",
                  reported[offset + i].name);
        EXPECT_EQ(tags, reported[offset + i].tags);
        EXPECT_DOUBLE_EQ(values[i], reported[offset + i].value);
    }
}

void CaptureErrorQps(kmonitor::MutableMetric *metric, const kmonitor::MetricsTags *tags, double value) {
    if (metric->metric_name_.find("service.error_qps") != std::string::npos) {
        reported_error_qps += value;
    }
    if (metric->metric_name_.find("grpc.response_qps") != std::string::npos) {
        reported_grpc_qps += value;
        reported_grpc_tags = tags->GetTagsMap();
    }
}
} // namespace

class KmonitorMetricsReporterTest : public TESTBASE {
protected:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry_);
        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        reporter_ = std::make_unique<KmonitorMetricsReporter>();
        reporter_->Init(cache_manager_, metrics_registry_, "");
    }

    void TearDown() override {}

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::unique_ptr<KmonitorMetricsReporter> reporter_;
};

TEST_F(KmonitorMetricsReporterTest, TestConstructorAndDestructor) { EXPECT_NE(reporter_, nullptr); }

TEST_F(KmonitorMetricsReporterTest, GrpcStatusQpsPreservesTagsAndUpdatesLocalCounter) {
    const MetricsTags rejected_tags{
        {"server", "meta"}, {"method", "MetaService/GetClusterInfo"}, {"grpc_code", "RESOURCE_EXHAUSTED"}};
    const MetricsTags accepted_tags{{"server", "admin"}, {"method", "AdminService/CheckHealth"}, {"grpc_code", "OK"}};
    GrpcMetricsCollector rejected(metrics_registry_, rejected_tags);
    GrpcMetricsCollector accepted(metrics_registry_, accepted_tags);
    ASSERT_TRUE(rejected.Init());
    ASSERT_TRUE(accepted.Init());
    Stub stub;
    auto report = static_cast<void (kmonitor::MutableMetric::*)(const kmonitor::MetricsTags *, double)>(
        &kmonitor::MutableMetric::Report);
    reported_grpc_qps = 0;
    reported_grpc_tags.clear();
    stub.set(report, CaptureErrorQps);

    reporter_->ReportPerQuery(&rejected);
    EXPECT_DOUBLE_EQ(1., reported_grpc_qps);
    EXPECT_EQ(rejected_tags, reported_grpc_tags);
    reporter_->ReportPerQuery(&accepted);
    EXPECT_DOUBLE_EQ(2., reported_grpc_qps);
    EXPECT_EQ(accepted_tags, reported_grpc_tags);
    EXPECT_EQ(1u, rejected.get_grpc_response_counter_metrics());
    EXPECT_EQ(1u, accepted.get_grpc_response_counter_metrics());
}

TEST_F(KmonitorMetricsReporterTest, TestInitWithNull) {
    EXPECT_FALSE(reporter_->Init(nullptr, nullptr, ""));

    EXPECT_FALSE(reporter_->Init(nullptr, metrics_registry_, ""));

    EXPECT_FALSE(reporter_->Init(cache_manager_, nullptr, ""));

    EXPECT_TRUE(reporter_->Init(cache_manager_, metrics_registry_, ""));
}

TEST_F(KmonitorMetricsReporterTest, TestReportPerQuery) {
    EXPECT_NO_THROW(reporter_->ReportPerQuery(nullptr));

    {
        DummyMetricsCollector collector;
        collector.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));
        EXPECT_EQ(3, metrics_registry_->GetSize());
    }

    {
        EventReportMetricsCollector collector(metrics_registry_, {{"event_type", "block_snapshot"}});
        ASSERT_TRUE(collector.Init());
        SET_METRICS_(&collector, service, query_rt_us, 123.);
        SET_METRICS_(&collector, service, error_code, 1.);
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));
        EXPECT_EQ(1, collector.get_service_query_counter_metrics());
        EXPECT_EQ(1, collector.get_service_error_counter_metrics());
    }

    {
        // simulate the uninitialised case 1
        reporter_->cache_manager_ = nullptr;
        reporter_->metrics_registry_ = std::make_shared<MetricsRegistry>();

        ServiceMetricsCollector collector{metrics_registry_};
        collector.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));

        DataStorageMetricsCollector collector2{metrics_registry_};
        collector2.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));
    }

    {
        // simulate the uninitialised case 2
        reporter_->cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        reporter_->metrics_registry_ = nullptr;

        ServiceMetricsCollector collector{metrics_registry_};
        collector.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));

        DataStorageMetricsCollector collector2{metrics_registry_};
        collector2.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));
    }

    {
        // simulate the uninitialised case 3
        reporter_->cache_manager_ = nullptr;
        reporter_->metrics_registry_ = nullptr;

        ServiceMetricsCollector collector{metrics_registry_};
        collector.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));

        DataStorageMetricsCollector collector2{metrics_registry_};
        collector2.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));
    }
}

TEST_F(KmonitorMetricsReporterTest, TestReportInterval) {
    {
        metrics_registry_->GetCounter("cache_gc.scan_round_count") += 1;
        metrics_registry_->GetCounter("cache_gc.candidate_count", {{"reason", "storage_missing"}}) += 2;
        metrics_registry_->GetCounter("cache_gc.delete_result_count", {{"status", "0"}}) += 1;
        metrics_registry_->GetCounter("cache_gc.operation_error_count", {{"stage", "scan"}}) += 1;
        metrics_registry_->GetGauge("cache_gc.inflight_delete_count") = 2;
        metrics_registry_->GetCounter("cache_gc.candidate_dropped_count",
                                      {{"reason", "event_report_down_host"}, {"cause", "total_budget"}}) += 1;
        metrics_registry_->GetCounter("cache_gc.event_report_probe_count", {{"result", "delete"}}) += 2;
        metrics_registry_->GetCounter("cache_gc.event_report_probe_unknown_count", {{"cause", "malformed"}}) += 1;
        metrics_registry_->GetCounter("cache_gc.event_report_delete_location_count",
                                      {{"reason", "down_host"}, {"status", "deleted"}}) += 2;
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());
    }

    {
        // simulate the uninitialised case 1
        reporter_->cache_manager_ = nullptr;
        reporter_->metrics_registry_ = std::make_shared<MetricsRegistry>();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());
    }

    {
        // simulate the uninitialised case 2
        reporter_->cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        reporter_->metrics_registry_ = nullptr;
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());
    }

    {
        // simulate the uninitialised case 3
        reporter_->cache_manager_ = nullptr;
        reporter_->metrics_registry_ = nullptr;
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());
    }
}

TEST_F(KmonitorMetricsReporterTest, TestReportIntervalWriteBytes) {
    // the per-storage counter lives on the backend's DataStorageMetricsCollector,
    // which is only created after Open()
    auto be = std::make_shared<NfsBackend>(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_root_path(GetPrivateTestRuntimeDataPath() + "nfs_root/");
    StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_NFS, "abc", spec);
    ASSERT_EQ(ErrorCode::EC_OK, be->Open(config, "trace_1"));

    registry_manager_->data_storage_manager_ = std::make_shared<DataStorageManager>(metrics_registry_);
    registry_manager_->data_storage_manager_->storage_map_.emplace("abc", be);

    registry_manager_->data_storage_manager_->RecordWriteBytes("abc", 2048);
    EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());

    // the kmonitor side has no mock injection point; assert the registry-side value
    // that ReportInterval reads from instead
    const auto &tags = be->GetMetricsCollector()->GetMetricsTags();
    EXPECT_EQ(2048u, metrics_registry_->GetCounter("data_storage.write_bytes_dispatched_total", tags).Get());
}

TEST_F(KmonitorMetricsReporterTest, RequestOutcomeControlsKmonitorAndLocalErrorCounts) {
    ServiceMetricsCollector collector(metrics_registry_);
    ASSERT_TRUE(collector.Init());
    Stub stub;
    auto report = static_cast<void (kmonitor::MutableMetric::*)(const kmonitor::MetricsTags *, double)>(
        &kmonitor::MutableMetric::Report);
    reported_error_qps = 0;
    stub.set(report, CaptureErrorQps);

    SET_METRICS_(&collector, service, error_code, 1.);
    reporter_->ReportPerQuery(&collector, false);
    EXPECT_DOUBLE_EQ(0., reported_error_qps);
    EXPECT_EQ(0u, collector.get_service_error_counter_metrics());

    SET_METRICS_(&collector, service, error_code, 0.);
    reporter_->ReportPerQuery(&collector, true);
    EXPECT_DOUBLE_EQ(1., reported_error_qps);
    EXPECT_EQ(1u, collector.get_service_error_counter_metrics());

    SET_METRICS_(&collector, service, error_code, 1.);
    reporter_->ReportPerQuery(&collector, false);
    EXPECT_DOUBLE_EQ(1., reported_error_qps);
    EXPECT_EQ(1u, collector.get_service_error_counter_metrics());

    // Existing callers without a request outcome still use the collector gauge.
    reporter_->ReportPerQuery(&collector);
    EXPECT_DOUBLE_EQ(2., reported_error_qps);
    EXPECT_EQ(2u, collector.get_service_error_counter_metrics());
    EXPECT_EQ(4u, collector.get_service_query_counter_metrics());
}

TEST_F(KmonitorMetricsReporterTest, LifetimeSamplesAreReportedPerRequestWithInstanceTags) {
    Stub stub;
    auto report = static_cast<void (kmonitor::MutableMetric::*)(const kmonitor::MetricsTags *, double)>(
        &kmonitor::MutableMetric::Report);
    stub.set(report, CaptureEventAges);
    reported_lifetimes.clear();
    const MetricsTags tags{{"api_name", "ReportEvent"},
                           {"instance_group", "group_a"},
                           {"instance_id", "instance_a"},
                           {"type", "event_report_l2"},
                           {"event_type", "block_delete"}};
    const MetricsTags lifetime_tags{{"instance_group", "group_a"}, {"instance_id", "instance_a"}};
    EventReportMetricsCollector cached(metrics_registry_, tags);
    ASSERT_TRUE(cached.Init());
    EventReportMetricsCollector first(cached);
    EventReportMetricsCollector second(cached);
    first.SetLocationLifetimeSample({10, 30, 20});
    second.SetLocationLifetimeSample({100, 300, 200});
    // The shared gauge can hold another request's value; KMonitor must read
    // the private sample, without stealing or clearing the other request.
    second.UpdateLocationLifetimeMetrics();
    reporter_->ReportPerQuery(&first);
    reporter_->ReportPerQuery(&second);
    ASSERT_EQ(6u, reported_lifetimes.size());
    ExpectLifetime(reported_lifetimes, 0, lifetime_tags, {10, 30, 20});
    ExpectLifetime(reported_lifetimes, 3, lifetime_tags, {100, 300, 200});

    EventReportMetricsCollector empty(first);
    reporter_->ReportPerQuery(&empty);
    EXPECT_EQ(6u, reported_lifetimes.size());
    reporter_->ReportInterval();
    reporter_->ReportInterval();
    EXPECT_EQ(6u, reported_lifetimes.size());

    EventReportMetricsCollector zero(cached);
    zero.SetLocationLifetimeSample({0, 0, 0});
    // Completed requests still report their private sample after instance removal.
    metrics_registry_->RemoveByTagFilter({{"instance_id", "instance_a"}});
    reporter_->ReportPerQuery(&zero);
    ASSERT_EQ(9u, reported_lifetimes.size());
    ExpectLifetime(reported_lifetimes, 6, lifetime_tags, {0, 0, 0});
}
