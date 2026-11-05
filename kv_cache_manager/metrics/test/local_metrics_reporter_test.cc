#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/registry_local_backend.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_manager_metrics_recorder.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/local_metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/prometheus_exporter.h"
#include "kv_cache_manager/service/util/service_call_guard.h"
#include "stub.h"

using namespace kv_cache_manager;

std::size_t MetaIndexer_GetMemUsage_stub() noexcept { return 16; }

class LocalMetricsReporterTest : public TESTBASE {
public:
    void SetUp() override {
        stub_.set(ADDR(MetaIndexer, GetMemUsage), MetaIndexer_GetMemUsage_stub);

        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry_);
        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        reporter_ = std::make_unique<LocalMetricsReporter>();
        reporter_->Init(cache_manager_, metrics_registry_, "");
    }

    void TearDown() override { stub_.reset(ADDR(MetaIndexer, GetMemUsage)); }

    Stub stub_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::unique_ptr<LocalMetricsReporter> reporter_;
};

TEST_F(LocalMetricsReporterTest, TestConstructor) { EXPECT_NE(reporter_, nullptr); }

TEST_F(LocalMetricsReporterTest, GrpcResponseCountsAreSeparateFromBusinessMetrics) {
    ServiceMetricsCollector business(metrics_registry_);
    ASSERT_TRUE(business.Init());
    const std::string method = "MetaService/GetClusterInfo";
    GrpcMetricsCollector rejected(metrics_registry_,
                                  {{"server", "meta"}, {"method", method}, {"grpc_code", "RESOURCE_EXHAUSTED"}});
    GrpcMetricsCollector accepted(metrics_registry_,
                                  {{"server", "admin"}, {"method", "AdminService/CheckHealth"}, {"grpc_code", "OK"}});
    ASSERT_TRUE(rejected.Init());
    ASSERT_TRUE(accepted.Init());

    reporter_->ReportPerQuery(&rejected);
    reporter_->ReportPerQuery(&rejected);
    reporter_->ReportPerQuery(&accepted);
    EXPECT_EQ(2u, rejected.get_grpc_response_counter_metrics());
    EXPECT_EQ(1u, accepted.get_grpc_response_counter_metrics());
    EXPECT_EQ(0u, business.get_service_query_counter_metrics());
    EXPECT_EQ(0u, business.get_service_error_counter_metrics());

    const auto output = PrometheusExporter::Expose(*metrics_registry_);
    EXPECT_NE(output.find("kvcm_grpc_response_counter{grpc_code=\"RESOURCE_EXHAUSTED\",method=\"" + method +
                          "\",server=\"meta\"} 2\n"),
              std::string::npos)
        << output;
}

TEST_F(LocalMetricsReporterTest, TestInitWithNull) {
    EXPECT_FALSE(reporter_->Init(nullptr, nullptr, ""));

    EXPECT_FALSE(reporter_->Init(nullptr, metrics_registry_, ""));

    EXPECT_FALSE(reporter_->Init(cache_manager_, nullptr, ""));

    EXPECT_TRUE(reporter_->Init(cache_manager_, metrics_registry_, ""));
}

TEST_F(LocalMetricsReporterTest, TestReportPerQuery00) {
    EXPECT_EQ(3, metrics_registry_->GetSize());
    EXPECT_NO_THROW(reporter_->ReportPerQuery(nullptr));
    EXPECT_EQ(3, metrics_registry_->GetSize());

    {
        DummyMetricsCollector collector;
        collector.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));
        EXPECT_EQ(3, metrics_registry_->GetSize());
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

TEST_F(LocalMetricsReporterTest, TestReportPerQuery01) {
    constexpr int kDataStorageMetricsCount = 10;
    EXPECT_EQ(3, metrics_registry_->GetSize());

    DataStorageMetricsCollector collector(metrics_registry_);
    collector.Init();

    EXPECT_EQ(kDataStorageMetricsCount, metrics_registry_->GetSize());

    {
        reporter_->ReportPerQuery(&collector);
        EXPECT_EQ(kDataStorageMetricsCount, metrics_registry_->GetSize());

        std::uint64_t v;
        GET_METRICS_(&collector, data_storage, create_counter, v);
        EXPECT_EQ(1, v);

        GET_METRICS_(&collector, data_storage, create_keys_counter, v);
        EXPECT_EQ(0, v);
    }

    {
        Gauge g;
        COPY_METRICS_(&collector, data_storage, create_keys_qps, g);
        g = 6.0;

        reporter_->ReportPerQuery(&collector);
        EXPECT_EQ(kDataStorageMetricsCount, metrics_registry_->GetSize());

        std::uint64_t v;
        GET_METRICS_(&collector, data_storage, create_counter, v);
        EXPECT_EQ(2, v);

        GET_METRICS_(&collector, data_storage, create_keys_counter, v);
        EXPECT_EQ(6, v);
    }

    {
        Gauge g;
        COPY_METRICS_(&collector, data_storage, create_keys_qps, g);
        g = 2.0;

        reporter_->ReportPerQuery(&collector);
        EXPECT_EQ(kDataStorageMetricsCount, metrics_registry_->GetSize());

        std::uint64_t v;
        GET_METRICS_(&collector, data_storage, create_counter, v);
        EXPECT_EQ(3, v);

        GET_METRICS_(&collector, data_storage, create_keys_counter, v);
        EXPECT_EQ(8, v);
    }
}

TEST_F(LocalMetricsReporterTest, TestReportPerQuery02) {
    constexpr int kServiceMetricsCount = 3 + 5 + 14 + 6 + 26;
    EXPECT_EQ(3, metrics_registry_->GetSize());

    ServiceMetricsCollector collector(metrics_registry_);
    collector.Init();

    EXPECT_EQ(kServiceMetricsCount, metrics_registry_->GetSize());

    {
        reporter_->ReportPerQuery(&collector);

        EXPECT_EQ(kServiceMetricsCount, metrics_registry_->GetSize());

        std::uint64_t v;
        GET_METRICS_(&collector, service, query_counter, v);
        EXPECT_EQ(1, v);

        GET_METRICS_(&collector, service, error_counter, v);
        EXPECT_EQ(0, v);
    }

    {
        Gauge ec;
        COPY_METRICS_(&collector, service, error_code, ec);
        ec = 1.0;

        reporter_->ReportPerQuery(&collector);

        EXPECT_EQ(kServiceMetricsCount, metrics_registry_->GetSize());

        std::uint64_t v;
        GET_METRICS_(&collector, service, query_counter, v);
        EXPECT_EQ(2, v);

        GET_METRICS_(&collector, service, error_counter, v);
        EXPECT_EQ(1, v);
    }
}

TEST_F(LocalMetricsReporterTest, ExplicitRequestOutcomeOverridesSharedGauge) {
    ServiceMetricsCollector collector(metrics_registry_);
    ASSERT_TRUE(collector.Init());

    SET_METRICS_(&collector, service, error_code, 1.);
    reporter_->ReportPerQuery(&collector, false);
    EXPECT_EQ(0u, collector.get_service_error_counter_metrics());

    SET_METRICS_(&collector, service, error_code, 0.);
    reporter_->ReportPerQuery(&collector, true);
    EXPECT_EQ(2u, collector.get_service_query_counter_metrics());
    EXPECT_EQ(1u, collector.get_service_error_counter_metrics());
}

TEST_F(LocalMetricsReporterTest, ServiceCallGuardReportsSuccessFailureAndRecovery) {
    auto collector = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    ASSERT_TRUE(collector->Init());
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*metrics_registry_).find("kvcm_service_error_counter"));

    for (int status : {RequestContext::kOkStatusCode, 10, RequestContext::kOkStatusCode}) {
        RequestContext context("request-outcome", collector);
        context.set_status_code(status);
        { ServiceCallGuard guard(cache_manager_.get(), &context, reporter_.get()); }
        EXPECT_DOUBLE_EQ(status == RequestContext::kOkStatusCode ? 0. : 1.,
                         collector->get_service_error_code_metrics());
    }

    EXPECT_EQ(3u, collector->get_service_query_counter_metrics());
    EXPECT_EQ(1u, collector->get_service_error_counter_metrics());
    EXPECT_NE(std::string::npos, PrometheusExporter::Expose(*metrics_registry_).find("kvcm_service_error_counter 1\n"));
}

TEST_F(LocalMetricsReporterTest, ConcurrentServiceCallsCountTheirOwnFailures) {
    auto collector = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    ASSERT_TRUE(collector->Init());
    std::atomic<int> ready{0};
    std::vector<std::thread> workers;
    constexpr int kWorkers = 4;
    constexpr int kRequests = 100;
    for (int i = 0; i < kWorkers; ++i) {
        workers.emplace_back([&, i]() {
            ++ready;
            while (ready.load() != kWorkers) {
                std::this_thread::yield();
            }
            for (int j = 0; j < kRequests; ++j) {
                RequestContext context("concurrent-outcome", collector);
                context.set_status_code(i % 2 == 0 ? RequestContext::kOkStatusCode : 10);
                ServiceCallGuard guard(cache_manager_.get(), &context, reporter_.get());
            }
        });
    }
    for (auto &worker : workers) {
        worker.join();
    }
    EXPECT_EQ(kWorkers * kRequests, collector->get_service_query_counter_metrics());
    EXPECT_EQ(kWorkers * kRequests / 2, collector->get_service_error_counter_metrics());
}

TEST_F(LocalMetricsReporterTest, TestReportPerQueryEventReport) {
    EventReportMetricsCollector collector(metrics_registry_, {{"event_type", "block_snapshot"}});
    ASSERT_TRUE(collector.Init());

    reporter_->ReportPerQuery(&collector);
    std::uint64_t value = 0;
    GET_METRICS_(&collector, service, query_counter, value);
    EXPECT_EQ(1, value);
    GET_METRICS_(&collector, service, error_counter, value);
    EXPECT_EQ(0, value);

    SET_METRICS_(&collector, service, error_code, 10.);
    reporter_->ReportPerQuery(&collector);
    GET_METRICS_(&collector, service, query_counter, value);
    EXPECT_EQ(2, value);
    GET_METRICS_(&collector, service, error_counter, value);
    EXPECT_EQ(1, value);
}

TEST_F(LocalMetricsReporterTest, LifetimeSamplesReachPrometheusAtRequestCompletion) {
    const MetricsTags tags{{"api_name", "ReportEvent"},
                           {"instance_group", "group_a"},
                           {"instance_id", "instance_a"},
                           {"type", "event_report_l2"},
                           {"event_type", "block_delete"}};
    auto service_collector = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    ASSERT_TRUE(service_collector->Init());
    EventReportMetricsCollector cached(metrics_registry_, tags);
    ASSERT_TRUE(cached.Init());
    auto request_collector = std::make_shared<EventReportMetricsCollector>(cached);
    request_collector->SetLocationLifetimeSample({0, 0, 0});
    EXPECT_EQ(std::string::npos,
              PrometheusExporter::Expose(*metrics_registry_).find("event_report_location_lifetime_"));
    RequestContext context("lifetime_sample", service_collector);
    context.set_status_code(RequestContext::kOkStatusCode);
    context.GetMetricsCollectorsVehicle().AddMetricsCollector(request_collector);
    { ServiceCallGuard guard(cache_manager_.get(), &context, reporter_.get()); }

    const auto output = PrometheusExporter::Expose(*metrics_registry_);
    for (const auto *suffix : {"min", "max", "avg"}) {
        const auto series = std::string("kvcm_cache_manager_instance_event_report_location_lifetime_") + suffix +
                            "_us{instance_group=\"group_a\",instance_id=\"instance_a\"} 0\n";
        EXPECT_NE(std::string::npos, output.find(series));
    }
}

TEST_F(LocalMetricsReporterTest, ConcurrentEventCollectorsDoNotShareErrorSamples) {
    const MetricsTags tags{{"event_type", "block_snapshot"}};
    EventReportMetricsCollector success(metrics_registry_, tags);
    EventReportMetricsCollector failure(metrics_registry_, tags);
    ASSERT_TRUE(success.Init());
    ASSERT_TRUE(failure.Init());

    success.SetRequestSample(10.0, 0.0);
    failure.SetRequestSample(20.0, 1.0);
    // Both objects point at the same tagged registry gauge. Deliberately leave
    // that shared gauge in the failed state before reporting the success.
    SET_METRICS_(&success, service, error_code, 0.0);
    SET_METRICS_(&failure, service, error_code, 1.0);

    reporter_->ReportPerQuery(&success);
    reporter_->ReportPerQuery(&failure);

    EXPECT_EQ(2u, success.get_service_query_counter_metrics());
    EXPECT_EQ(1u, success.get_service_error_counter_metrics());
}

TEST_F(LocalMetricsReporterTest, ServiceCallGuardCopiesRequestOutcomeToEventReportMetrics) {
    auto service_collector = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    auto snapshot_collector =
        std::make_shared<EventReportMetricsCollector>(metrics_registry_, MetricsTags{{"event_type", "block_snapshot"}});
    ASSERT_TRUE(service_collector->Init());
    ASSERT_TRUE(snapshot_collector->Init());
    SET_METRICS_(service_collector, service, error_code, 10.);

    RequestContext request_context("event-report-metrics", service_collector);
    request_context.set_status_code(10);
    request_context.GetMetricsCollectorsVehicle().AddMetricsCollector(snapshot_collector);
    {
        ServiceCallGuard guard(cache_manager_.get(), &request_context, reporter_.get());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_GT(snapshot_collector->get_service_query_rt_us_metrics(), 0.);
    EXPECT_DOUBLE_EQ(1., snapshot_collector->get_service_error_code_metrics());
    EXPECT_EQ(1, snapshot_collector->get_service_query_counter_metrics());
    EXPECT_EQ(1, snapshot_collector->get_service_error_counter_metrics());
}

TEST_F(LocalMetricsReporterTest, EventReportMetricsUseRequestLocalStatusInsteadOfSharedServiceGauge) {
    auto service_collector = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    auto heartbeat_collector =
        std::make_shared<EventReportMetricsCollector>(metrics_registry_, MetricsTags{{"event_type", "heartbeat"}});
    ASSERT_TRUE(service_collector->Init());
    ASSERT_TRUE(heartbeat_collector->Init());
    // Model another concurrent request leaving a failure sample in the
    // instance-scoped master collector.
    SET_METRICS_(service_collector, service, error_code, 1.);

    RequestContext request_context("event-report-success", service_collector);
    request_context.set_status_code(RequestContext::kOkStatusCode);
    request_context.GetMetricsCollectorsVehicle().AddMetricsCollector(heartbeat_collector);
    { ServiceCallGuard guard(cache_manager_.get(), &request_context, reporter_.get()); }

    EXPECT_DOUBLE_EQ(0., heartbeat_collector->get_service_error_code_metrics());
    EXPECT_EQ(1, heartbeat_collector->get_service_query_counter_metrics());
    EXPECT_EQ(0, heartbeat_collector->get_service_error_counter_metrics());
}

TEST_F(LocalMetricsReporterTest, TestReportInterval00) {
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

TEST_F(LocalMetricsReporterTest, TestReportInterval01) {
    {
        registry_manager_->data_storage_manager_ = nullptr;
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());
    }

    {
        auto be = std::make_shared<NfsBackend>(metrics_registry_);
        be->SetOpen(true);
        be->SetAvailable(true);
        registry_manager_->data_storage_manager_ = std::make_shared<DataStorageManager>(metrics_registry_);
        registry_manager_->data_storage_manager_->storage_map_.emplace("abc", be);
        reporter_->ReportInterval();

        const auto mc_vec = reporter_->data_storage_interval_metrics_collectors_.GetMetricsCollectors();
        EXPECT_EQ(1, mc_vec.size());
        auto p = std::dynamic_pointer_cast<DataStorageIntervalMetricsCollector>(mc_vec.front());
        EXPECT_TRUE(p);
        double v;
        GET_METRICS_(p, data_storage, healthy_status, v);
        EXPECT_DOUBLE_EQ(1., v);
        GET_METRICS_(p, data_storage, writable_status, v);
        EXPECT_DOUBLE_EQ(1., v);
        GET_METRICS_(p, data_storage, storage_usage_ratio, v);
        EXPECT_DOUBLE_EQ(0., v);
    }

    {
        auto be = std::make_shared<NfsBackend>(metrics_registry_);
        be->SetOpen(true);
        be->SetAvailable(false);
        registry_manager_->data_storage_manager_ = std::make_shared<DataStorageManager>(metrics_registry_);
        registry_manager_->data_storage_manager_->storage_map_.emplace("abc", be);
        reporter_->ReportInterval();

        const auto mc_vec = reporter_->data_storage_interval_metrics_collectors_.GetMetricsCollectors();
        EXPECT_EQ(1, mc_vec.size());
        auto p = std::dynamic_pointer_cast<DataStorageIntervalMetricsCollector>(mc_vec.front());
        EXPECT_TRUE(p);
        double v;
        GET_METRICS_(p, data_storage, healthy_status, v);
        EXPECT_DOUBLE_EQ(0., v);
        GET_METRICS_(p, data_storage, writable_status, v);
        EXPECT_DOUBLE_EQ(0., v);
        GET_METRICS_(p, data_storage, storage_usage_ratio, v);
        EXPECT_DOUBLE_EQ(0., v);
    }
}


namespace {
// NfsBackend with a controllable usage ratio, used to verify how
// ReportInterval handles valid readings, default zero and negative values.
class FakeRatioNfsBackend : public NfsBackend {
public:
    explicit FakeRatioNfsBackend(std::shared_ptr<MetricsRegistry> registry) : NfsBackend(std::move(registry)) {}
    double GetStorageUsageRatio(const std::string & /*trace_id*/) const override {
        ++usage_queries_;
        return ratio_;
    }
    mutable int usage_queries_ = 0;
    void SetUsageRatio(double ratio) { ratio_ = ratio; }
    bool Writable() override { return Available() && writable_; }
    void SetWritable(bool writable) { writable_ = writable; }

private:
    double ratio_ = -1.0;
    bool writable_ = true;
};
} // namespace


TEST_F(LocalMetricsReporterTest, StorageUsageRatioPreservesZeroAndSkipsNegativeValues) {
    auto be = std::make_shared<FakeRatioNfsBackend>(metrics_registry_);
    be->SetOpen(true);
    be->SetAvailable(true);
    registry_manager_->data_storage_manager_ = std::make_shared<DataStorageManager>(metrics_registry_);
    registry_manager_->data_storage_manager_->storage_map_.emplace("ratio_store", be);

    auto published_ratio = [this]() {
        const auto mc_vec = reporter_->data_storage_interval_metrics_collectors_.GetMetricsCollectors();
        EXPECT_GE(mc_vec.size(), 1u);
        auto p = std::dynamic_pointer_cast<DataStorageIntervalMetricsCollector>(mc_vec.front());
        EXPECT_TRUE(p);
        double v = -12345.0;
        GET_METRICS_(p, data_storage, storage_usage_ratio, v);
        return v;
    };

    // A valid ratio gets published.
    be->SetUsageRatio(0.75);
    reporter_->ReportInterval();
    EXPECT_DOUBLE_EQ(0.75, published_ratio());

    // Negative values are not valid readings and must not be exported.
    // Backends are not required to return a negative value on query failure.
    be->SetUsageRatio(-1.0);
    reporter_->ReportInterval();
    EXPECT_DOUBLE_EQ(0.75, published_ratio());

    // Preserve the existing default-zero behavior, including after an error:
    // zero replaces a previous reading while health still reflects failure.
    be->SetUsageRatio(0.0);
    be->SetAvailable(false);
    reporter_->ReportInterval();
    EXPECT_DOUBLE_EQ(0.0, published_ratio());
    EXPECT_DOUBLE_EQ(0.0,
                     metrics_registry_->GetGauge("data_storage.healthy_status",
                                                 MetricsTags{{"type", "file"}, {"unique_name", "ratio_store"}}).Get());
    be->SetUsageRatio(0.75);
    be->SetAvailable(true);
    reporter_->ReportInterval();
    EXPECT_DOUBLE_EQ(0.75, published_ratio());

    // A backend that never had a valid ratio leaves its gauge at the
    // registry default, and the exported text never shows a bogus full
    // reading (or the raw sentinel) for it.
    auto never_ok = std::make_shared<FakeRatioNfsBackend>(metrics_registry_);
    never_ok->SetOpen(true);
    never_ok->SetAvailable(false);
    registry_manager_->data_storage_manager_->storage_map_.emplace("never_ok_store", never_ok);
    reporter_->ReportInterval();
    const auto mc_vec = reporter_->data_storage_interval_metrics_collectors_.GetMetricsCollectors();
    ASSERT_EQ(2u, mc_vec.size());
    for (const auto &mc : mc_vec) {
        auto p = std::dynamic_pointer_cast<DataStorageIntervalMetricsCollector>(mc);
        ASSERT_TRUE(p);
        double v = -12345.0;
        GET_METRICS_(p, data_storage, storage_usage_ratio, v);
        EXPECT_TRUE(v == 0.75 || v == 0.0) << "unexpected published usage ratio: " << v;
    }
    const std::string exported_text = PrometheusExporter::Expose(*metrics_registry_);
    EXPECT_EQ(std::string::npos, exported_text.find("unique_name=\"never_ok_store\"} 1"));
    EXPECT_EQ(std::string::npos, exported_text.find("unique_name=\"never_ok_store\"} -"));

    // Filtering a negative ratio must not suppress the backend health signal.
    // An unavailable backend still exports healthy_status 0; a healthy one
    // continues to export healthy_status 1.
    auto exported_value = [&exported_text](const std::string &metric, const std::string &unique_name) {
        const std::string tag = "unique_name=\"" + unique_name + "\"} ";
        size_t pos = 0;
        while ((pos = exported_text.find(metric + "{", pos)) != std::string::npos) {
            auto line_end = exported_text.find('\n', pos);
            if (line_end == std::string::npos) {
                line_end = exported_text.size();
            }
            const auto tag_pos = exported_text.find(tag, pos);
            if (tag_pos != std::string::npos && tag_pos < line_end) {
                const auto value_begin = tag_pos + tag.size();
                return exported_text.substr(value_begin, line_end - value_begin);
            }
            pos = line_end + 1;
        }
        return std::string{};
    };
    EXPECT_EQ("0", exported_value("kvcm_data_storage_healthy_status", "never_ok_store"));
    EXPECT_EQ("1", exported_value("kvcm_data_storage_healthy_status", "ratio_store"));
}

TEST_F(LocalMetricsReporterTest, FollowerUsageIsAbsentAcrossRolesAndStorageStates) {
    auto backend = std::make_shared<FakeRatioNfsBackend>(metrics_registry_);
    backend->SetOpen(true);
    registry_manager_->data_storage_manager_ = std::make_shared<DataStorageManager>(metrics_registry_);
    registry_manager_->data_storage_manager_->storage_map_.emplace("role_store", backend);

    bool leader = false;
    int role_samples = 0;
    reporter_->SetStorageUsageProvider([&]() {
        ++role_samples;
        return leader;
    });
    const MetricsTags tags{{"type", "file"}, {"unique_name", "role_store"}};
    const auto verify = [&](double expected_ratio) {
        const int previous_queries = backend->usage_queries_;
        role_samples = 0;
        reporter_->ReportInterval();
        EXPECT_EQ(1, role_samples);
        EXPECT_EQ(previous_queries + (leader ? 1 : 0), backend->usage_queries_);
        if (leader) {
            EXPECT_DOUBLE_EQ(expected_ratio,
                             metrics_registry_->GetGauge("data_storage.storage_usage_ratio", tags).Get());
        } else {
            const auto ratios = metrics_registry_->GetMetricsData("data_storage.storage_usage_ratio");
            EXPECT_TRUE(!ratios || ratios->GetMetricsValues().empty());
        }
        EXPECT_DOUBLE_EQ(backend->Available() ? 1.0 : 0.0,
                         metrics_registry_->GetGauge("data_storage.healthy_status", tags).Get());
        EXPECT_DOUBLE_EQ(backend->Writable() ? 1.0 : 0.0,
                         metrics_registry_->GetGauge("data_storage.writable_status", tags).Get());

        // Check the text exposed by /metrics, including connectivity signals.
        const auto output = PrometheusExporter::Expose(*metrics_registry_);
        const std::string labels = "{type=\"file\",unique_name=\"role_store\"} ";
        const auto expect_export = [&](const std::string &metric, const std::string &value) {
            EXPECT_NE(std::string::npos, output.find("kvcm_data_storage_" + metric + labels + value + "\n"));
        };
        if (!leader) {
            EXPECT_EQ(std::string::npos, output.find("kvcm_data_storage_storage_usage_ratio"));
        }
        expect_export("healthy_status", backend->Available() ? "1" : "0");
        expect_export("writable_status", backend->Writable() ? "1" : "0");
    };

    // Cold follower, including a failed capacity query and failed health check.
    verify(0.0);
    for (bool available : {false, true}) {
        for (bool writable : {false, true}) {
            backend->SetAvailable(available);
            backend->SetWritable(writable);
            for (double ratio : {-1.0, 0.0, 0.75, 1.0}) {
                SCOPED_TRACE(::testing::Message() << "available=" << available << " writable=" << writable
                                                 << " ratio=" << ratio);
                // A real full reading must remain visible on the leader.
                leader = true;
                backend->SetUsageRatio(1.0);
                verify(1.0);
                backend->SetUsageRatio(ratio);
                verify(ratio < 0.0 ? 1.0 : ratio);

                // Demotion clears even a retained 100% reading. Repeated
                // follower reports omit capacity independently of backend state.
                leader = false;
                verify(0.0);
                verify(0.0);

                // Promotion resumes normal sampling; unknown usage retains
                // the last published value until a valid reading arrives.
                leader = true;
                verify(ratio < 0.0 ? 0.0 : ratio);
                backend->SetUsageRatio(0.75);
                verify(0.75);
            }
        }
    }
}

TEST_F(LocalMetricsReporterTest, DemotionCleanupRemovesPreviouslyPublishedCapacity) {
    auto backend = std::make_shared<FakeRatioNfsBackend>(metrics_registry_);
    backend->SetOpen(true);
    backend->SetAvailable(true);
    backend->SetUsageRatio(1.0);
    auto manager = std::make_shared<DataStorageManager>(metrics_registry_);
    registry_manager_->data_storage_manager_ = manager;
    manager->storage_map_.emplace("demoted_store", backend);
    bool leader = true;
    reporter_->SetStorageUsageProvider([&]() { return leader; });
    reporter_->ReportInterval();

    const MetricsTags tags{{"type", "file"}, {"unique_name", "demoted_store"}};
    EXPECT_DOUBLE_EQ(1.0, metrics_registry_->GetGauge("data_storage.storage_usage_ratio", tags).Get());
    EXPECT_DOUBLE_EQ(1.0, metrics_registry_->GetGauge("data_storage.healthy_status", tags).Get());
    EXPECT_DOUBLE_EQ(1.0, metrics_registry_->GetGauge("data_storage.writable_status", tags).Get());

    // Server::OnNoLongerLeader closes and removes storage before the next
    // metrics tick. Exercise that real cleanup path, not just a role toggle.
    ASSERT_EQ(EC_OK, manager->DoCleanup());
    ASSERT_TRUE(manager->GetAllStorageNames().empty());
    // A report that still sees leader=true after cleanup must not recreate
    // metrics for the unloaded backend.
    reporter_->ReportInterval();
    leader = false;
    for (int i = 0; i < 2; ++i) {
        reporter_->ReportInterval();
        const auto output = PrometheusExporter::Expose(*metrics_registry_);
        EXPECT_EQ(std::string::npos, output.find("kvcm_data_storage_storage_usage_ratio"));
    }

    // Recovering the same storage on promotion resumes real measurements.
    backend->SetOpen(true);
    backend->SetAvailable(true);
    backend->SetUsageRatio(0.25);
    manager->storage_map_.emplace("demoted_store", backend);
    leader = true;
    reporter_->ReportInterval();
    EXPECT_DOUBLE_EQ(0.25, metrics_registry_->GetGauge("data_storage.storage_usage_ratio", tags).Get());
    EXPECT_DOUBLE_EQ(1.0, metrics_registry_->GetGauge("data_storage.healthy_status", tags).Get());
    EXPECT_DOUBLE_EQ(1.0, metrics_registry_->GetGauge("data_storage.writable_status", tags).Get());
}

TEST_F(LocalMetricsReporterTest, TestReportInterval02) {
    {
        cache_manager_->meta_indexer_manager_ = nullptr;
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());
    }

    {
        cache_manager_->meta_indexer_manager_ = std::make_shared<MetaIndexerManager>();

        auto midx = std::make_shared<MetaIndexer>();
        midx->key_count_.store(8);
        cache_manager_->meta_indexer_manager_->meta_indexers_.emplace("abc", midx);
        cache_manager_->meta_indexer_manager_->meta_indexers_.emplace("abc2", nullptr);
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());

        auto p = std::dynamic_pointer_cast<MetaIndexerAccumulativeMetricsCollector>(
            reporter_->metrics_collector_for_MetaIndexerAccumulative_);
        EXPECT_TRUE(p);

        double v;
        GET_METRICS_(p, meta_indexer, total_key_count, v);
        EXPECT_DOUBLE_EQ(8., v);

        GET_METRICS_(p, meta_indexer, total_cache_usage, v);
        EXPECT_DOUBLE_EQ(16., v);
    }
}

TEST_F(LocalMetricsReporterTest, TestReportIntervalCacheManagerMetrics) {
    cache_manager_->meta_indexer_manager_ = std::make_shared<MetaIndexerManager>();
    cache_manager_->write_location_manager_ = std::make_shared<WriteLocationManager>();
    cache_manager_->metrics_recorder_ =
        std::make_shared<CacheManagerMetricsRecorder>(cache_manager_->meta_indexer_manager_,
                                                      cache_manager_->write_location_manager_,
                                                      registry_manager_,
                                                      cache_manager_->metrics_lifecycle());

    RequestContext request_context("test_trace");
    InstanceGroup instance_group;
    instance_group.name_ = "test_group";
    instance_group.quota_.capacity_ = 10240;

    registry_manager_->storage_ = std::make_unique<RegistryLocalBackend>();
    ASSERT_EQ(ErrorCode::EC_OK, registry_manager_->storage_->Init({}));
    ASSERT_EQ(ErrorCode::EC_OK, registry_manager_->CreateInstanceGroup(&request_context, instance_group));

    ASSERT_EQ(ErrorCode::EC_OK,
              registry_manager_->RegisterInstance(
                  &request_context, "test_group", "test_instance_id", 8, {LocationSpecInfo("tp0", 1024)}, {}));

    auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    ASSERT_EQ(ErrorCode::EC_OK,
              cache_manager_->meta_indexer_manager_->CreateMetaIndexer("test_instance_id", meta_indexer_config));
    auto meta_indexer = cache_manager_->meta_indexer_manager_->GetMetaIndexer("test_instance_id");
    meta_indexer->key_count_.store(5);
    meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 1024 * 5);
    cache_manager_->metrics_recorder_->Start();
    std::this_thread::sleep_for(std::chrono::seconds(6));
    EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());
    {
        const auto &vec = reporter_->cache_manager_group_interval_metrics_collectors_.GetMetricsCollectors();
        ASSERT_EQ(1, vec.size());
        auto p = std::dynamic_pointer_cast<CacheManagerGroupMetricsCollector>(vec.at(0));
        EXPECT_TRUE(p);
        double usage_ratio_v;
        GET_METRICS_(p, cache_manager_group, usage_ratio, usage_ratio_v);
        EXPECT_DOUBLE_EQ(0.5, usage_ratio_v);
    }
    {
        const auto &vec = reporter_->cache_manager_instance_interval_metrics_collectors_.GetMetricsCollectors();
        ASSERT_EQ(1, vec.size());
        auto p = std::dynamic_pointer_cast<CacheManagerInstanceMetricsCollector>(vec.at(0));
        EXPECT_TRUE(p);
        double key_count_v, byte_size_v;
        GET_METRICS_(p, cache_manager_instance, key_count, key_count_v);
        GET_METRICS_(p, cache_manager_instance, byte_size, byte_size_v);
        EXPECT_DOUBLE_EQ(5, key_count_v);
        EXPECT_DOUBLE_EQ(5 * 1024, byte_size_v);

        // async_queue metrics should be zero since the default meta backend has no async queues
        double async_max_v, async_avg_v;
        GET_METRICS_(p, cache_manager_instance, async_queue_max_size, async_max_v);
        GET_METRICS_(p, cache_manager_instance, async_queue_avg_size, async_avg_v);
        EXPECT_DOUBLE_EQ(0., async_max_v);
        EXPECT_DOUBLE_EQ(0., async_avg_v);

        // async write stats metrics should be zero since the default meta backend has no async write path
        double flush_key_v, flush_time_v, pipeline_err_v;
        GET_METRICS_(p, cache_manager_instance, async_flush_key_count, flush_key_v);
        GET_METRICS_(p, cache_manager_instance, async_batch_flush_time_us, flush_time_v);
        GET_METRICS_(p, cache_manager_instance, async_pipeline_error_count, pipeline_err_v);
        EXPECT_DOUBLE_EQ(0., flush_key_v);
        EXPECT_DOUBLE_EQ(0., flush_time_v);
        EXPECT_DOUBLE_EQ(0., pipeline_err_v);
    }
}

TEST_F(LocalMetricsReporterTest, InstanceLocalMetadataMetricsTrackEntriesAndLifecycle) {
    // Exercise real local accounting, including the recorder and Prometheus export.
    stub_.reset(ADDR(MetaIndexer, GetMemUsage));
    cache_manager_->meta_indexer_manager_ = std::make_shared<MetaIndexerManager>();
    cache_manager_->write_location_manager_ = std::make_shared<WriteLocationManager>();
    registry_manager_->storage_ = std::make_unique<RegistryLocalBackend>();
    ASSERT_EQ(EC_OK, registry_manager_->storage_->Init({}));
    RequestContext context("instance_meta_cache_usage");
    for (const auto *group_name : {"group_a", "group_b"}) {
        InstanceGroup group;
        group.name_ = group_name;
        ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&context, group));
    }
    const std::map<std::string, std::string> instance_groups{
        {"small", "group_a"}, {"large", "group_b"}, {"empty", "group_a"}};
    for (const auto &[instance_id, group] : instance_groups) {
        ASSERT_EQ(
            EC_OK,
            registry_manager_->RegisterInstance(&context, group, instance_id, 8, {LocationSpecInfo("tp0", 1024)}, {}));
        ASSERT_EQ(EC_OK,
                  cache_manager_->meta_indexer_manager_->CreateMetaIndexer(instance_id,
                                                                           std::make_shared<MetaIndexerConfig>()));
    }
    auto small = cache_manager_->meta_indexer_manager_->GetMetaIndexer("small");
    auto large = cache_manager_->meta_indexer_manager_->GetMetaIndexer("large");
    CacheLocationMapVector locations(1);
    PropertyMapVector small_properties{{{"payload", std::string(128, 'a')}}};
    PropertyMapVector large_properties(2, {{"payload", std::string(4096, 'b')}});
    ASSERT_EQ(EC_OK, small->Put(&context, {7}, locations, small_properties).ec);
    locations.resize(2);
    ASSERT_EQ(EC_OK, large->Put(&context, {7, 8}, locations, large_properties).ec);
    // Deliberately diverge the persisted logical count from local occupancy.
    small->key_count_.store(99);
    const auto small_usage = small->GetMemUsage();
    const auto large_usage = large->GetMemUsage();
    ASSERT_GT(small_usage, 0u);
    ASSERT_GT(large_usage, small_usage);

    const auto sample = [&] {
        auto recorder = std::make_shared<CacheManagerMetricsRecorder>(cache_manager_->meta_indexer_manager_,
                                                                      cache_manager_->write_location_manager_,
                                                                      registry_manager_,
                                                                      cache_manager_->metrics_lifecycle());
        cache_manager_->metrics_recorder_ = recorder;
        recorder->Start();
        bool ready = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            const auto snapshot = recorder->group_instance_id_metric_map();
            if (snapshot.count("group_a") && snapshot.at("group_a").size() == 2 && snapshot.count("group_b") &&
                snapshot.at("group_b").size() == 1) {
                ready = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        recorder->Stop();
        reporter_->ReportInterval();
        return ready;
    };
    const auto expect_usage = [&](const std::string &id, size_t expected) {
        const auto output = PrometheusExporter::Expose(*metrics_registry_);
        const auto series = "kvcm_cache_manager_instance_meta_cache_usage{instance_group=\"" + instance_groups.at(id) +
                            "\",instance_id=\"" + id + "\"} " + std::to_string(expected) + "\n";
        EXPECT_NE(output.find(series), std::string::npos) << output;
    };
    const auto expect_counts = [&](const std::string &id, size_t local_count, size_t logical_count) {
        const auto output = PrometheusExporter::Expose(*metrics_registry_);
        const auto series = "kvcm_cache_manager_instance_local_key_count{instance_group=\"" + instance_groups.at(id) +
                            "\",instance_id=\"" + id + "\"} " + std::to_string(local_count) + "\n";
        EXPECT_NE(output.find(series), std::string::npos) << output;
        EXPECT_DOUBLE_EQ(logical_count,
                         metrics_registry_
                             ->GetGauge("cache_manager_instance.key_count",
                                        {{"instance_group", instance_groups.at(id)}, {"instance_id", id}})
                             .Get());
    };
    ASSERT_TRUE(sample());
    expect_usage("small", small_usage);
    expect_usage("large", large_usage);
    expect_usage("empty", 0);
    expect_counts("small", 1, 99);
    expect_counts("large", 2, 2);
    expect_counts("empty", 0, 0);
    EXPECT_DOUBLE_EQ(101, metrics_registry_->GetGauge("meta_indexer.total_key_count").Get());
    EXPECT_DOUBLE_EQ(small_usage + large_usage, metrics_registry_->GetGauge("meta_indexer.total_cache_usage").Get());
    // Metadata-only entries consume local memory without consuming data-storage quota.
    EXPECT_DOUBLE_EQ(
        0,
        metrics_registry_
            ->GetGauge("cache_manager_instance.byte_size", {{"instance_group", "group_a"}, {"instance_id", "small"}})
            .Get());

    ASSERT_EQ(EC_OK, small->Delete(&context, {7}).ec);
    ASSERT_TRUE(sample());
    expect_usage("small", 0);
    expect_usage("large", large_usage);
    expect_usage("empty", 0);
    expect_counts("small", 0, 98);
    expect_counts("large", 2, 2);
    expect_counts("empty", 0, 0);
    EXPECT_DOUBLE_EQ(100, metrics_registry_->GetGauge("meta_indexer.total_key_count").Get());

    cache_manager_->InvalidateInstanceMetrics("small");
    reporter_->ReportInterval();
    const auto data = metrics_registry_->GetMetricsData("cache_manager_instance.meta_cache_usage");
    ASSERT_NE(nullptr, data);
    EXPECT_EQ(2u, data->GetSize());
    for (const auto &[tags, value] : data->GetMetricsValues()) {
        EXPECT_NE("small", tags.at("instance_id"));
    }
    expect_usage("large", large_usage);
    expect_usage("empty", 0);
    const auto local_counts = metrics_registry_->GetMetricsData("cache_manager_instance.local_key_count");
    ASSERT_NE(nullptr, local_counts);
    EXPECT_EQ(2u, local_counts->GetSize());
    for (const auto &[tags, value] : local_counts->GetMetricsValues()) {
        EXPECT_NE("small", tags.at("instance_id"));
    }
    expect_counts("large", 2, 2);
    expect_counts("empty", 0, 0);
}

TEST_F(LocalMetricsReporterTest, TestStorageHealthFailureRecoveryAndClose) {
    auto backend = std::make_shared<NfsBackend>(metrics_registry_);
    backend->SetOpen(true);
    registry_manager_->data_storage_manager_ = std::make_shared<DataStorageManager>(metrics_registry_);
    registry_manager_->data_storage_manager_->storage_map_.emplace("health_transition", backend);
    const auto expect_health = [&](double expected) {
        reporter_->ReportInterval();
        const auto collectors = reporter_->data_storage_interval_metrics_collectors_.GetMetricsCollectors();
        ASSERT_EQ(1u, collectors.size());
        auto collector = std::dynamic_pointer_cast<DataStorageIntervalMetricsCollector>(collectors.front());
        ASSERT_NE(nullptr, collector);
        double value;
        GET_METRICS_(collector, data_storage, healthy_status, value);
        EXPECT_DOUBLE_EQ(expected, value);
    };
    backend->SetAvailable(true);
    expect_health(1.0);
    backend->SetAvailable(false);
    expect_health(0.0);
    backend->SetAvailable(true);
    expect_health(1.0);
    backend->SetOpen(false);
    expect_health(0.0);
}


TEST_F(LocalMetricsReporterTest, DemotionCleanupClearsRemovedBackendHealth) {
    auto backend = std::make_shared<NfsBackend>(metrics_registry_);
    backend->SetOpen(true);
    backend->SetAvailable(true);
    auto manager = std::make_shared<DataStorageManager>(metrics_registry_);
    registry_manager_->data_storage_manager_ = manager;
    manager->storage_map_.emplace("demoted_store", backend);
    bool leader = true;
    reporter_->SetStorageUsageProvider([&]() { return leader; });
    reporter_->ReportInterval();

    const MetricsTags tags{{"type", "file"}, {"unique_name", "demoted_store"}};
    EXPECT_DOUBLE_EQ(1.0, metrics_registry_->GetGauge("data_storage.healthy_status", tags).Get());
    EXPECT_DOUBLE_EQ(1.0, metrics_registry_->GetGauge("data_storage.writable_status", tags).Get());

    // Server::OnNoLongerLeader closes and removes storage before the next
    // metrics tick. Exercise that real cleanup path, not just a role toggle.
    ASSERT_EQ(EC_OK, manager->DoCleanup());
    ASSERT_TRUE(manager->GetAllStorageNames().empty());
    // A report that still sees leader=true after cleanup must not recreate
    // metrics for the unloaded backend.
    reporter_->ReportInterval();
    leader = false;
    for (int i = 0; i < 2; ++i) {
        reporter_->ReportInterval();
        const auto output = PrometheusExporter::Expose(*metrics_registry_);
        for (const auto &name : {"healthy_status", "writable_status"}) {
            EXPECT_EQ(std::string::npos, output.find(std::string("kvcm_data_storage_") + name));
        }
    }

    // Recovering the same storage on promotion resumes real measurements.
    backend->SetOpen(true);
    backend->SetAvailable(true);
    manager->storage_map_.emplace("demoted_store", backend);
    leader = true;
    reporter_->ReportInterval();
    EXPECT_DOUBLE_EQ(1.0, metrics_registry_->GetGauge("data_storage.healthy_status", tags).Get());
    EXPECT_DOUBLE_EQ(1.0, metrics_registry_->GetGauge("data_storage.writable_status", tags).Get());
}

namespace {
class BlockingUsageBackend : public NfsBackend {
public:
    explicit BlockingUsageBackend(std::shared_ptr<MetricsRegistry> registry)
        : NfsBackend(std::move(registry)), release_(released.get_future().share()) {}
    double GetStorageUsageRatio(const std::string &) const override {
        entered.set_value();
        release_.wait();
        return 1.0;
    }
    mutable std::promise<void> entered;
    std::promise<void> released;
    std::shared_future<void> release_;
};
} // namespace

TEST_F(LocalMetricsReporterTest, CleanupDuringCapacitySamplingCannotRecreateUnloadedSeries) {
    auto manager = std::make_shared<DataStorageManager>(metrics_registry_);
    registry_manager_->data_storage_manager_ = manager;
    auto old_backend = std::make_shared<BlockingUsageBackend>(metrics_registry_);
    old_backend->SetOpen(true);
    old_backend->SetAvailable(true);
    manager->storage_map_.emplace("reused_store", old_backend);
    reporter_->SetStorageUsageProvider([] { return true; });
    const MetricsTags tags{{"type", "file"}, {"unique_name", "reused_store"}};
    metrics_registry_->GetGauge("data_storage.storage_usage_ratio", tags) = 1.0;
    auto sampling = std::async(std::launch::async, [&] { reporter_->ReportInterval(); });
    const auto started = old_backend->entered.get_future().wait_for(std::chrono::seconds(2));
    if (started != std::future_status::ready) {
        old_backend->released.set_value();
        FAIL() << "capacity sampling did not start";
    }
    auto cleanup = std::async(std::launch::async, [&] { return registry_manager_->DoCleanup(); });
    const bool cleanup_ready = cleanup.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    EXPECT_TRUE(cleanup_ready) << "cleanup must not wait for backend capacity I/O";
    if (!cleanup_ready) {
        old_backend->released.set_value();
        cleanup.wait();
        sampling.wait();
        return;
    }
    EXPECT_EQ(EC_OK, cleanup.get());
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*metrics_registry_).find("unique_name=\"reused_store\""));
    auto replacement = std::make_shared<FakeRatioNfsBackend>(metrics_registry_);
    replacement->SetOpen(true);
    replacement->SetAvailable(true);
    replacement->SetUsageRatio(0.25);
    manager->storage_map_.emplace("reused_store", replacement);
    old_backend->released.set_value();
    sampling.get();
    // The old 100% result must not be published for the new backend of the same name.
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*metrics_registry_).find("unique_name=\"reused_store\""));
    reporter_->ReportInterval();
    EXPECT_DOUBLE_EQ(0.25, metrics_registry_->GetGauge("data_storage.storage_usage_ratio", tags).Get());
}
