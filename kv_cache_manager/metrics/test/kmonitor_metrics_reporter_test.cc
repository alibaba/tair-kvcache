#include <memory>
#include <vector>

#include "kmonitor/client/core/MutableMetric.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/dummy_backend.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/metrics/kmonitor_metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "stub.h"

using namespace kv_cache_manager;

namespace {
thread_local double reported_error_qps = 0;
thread_local double reported_create_qps = 0;
thread_local std::vector<double> reported_create_keys;
thread_local std::vector<double> reported_create_times;

void CaptureErrorQps(kmonitor::MutableMetric *metric, const kmonitor::MetricsTags *, double value) {
    if (metric->metric_name_.find("service.error_qps") != std::string::npos) {
        reported_error_qps += value;
    }
}

void CaptureCreateMetrics(kmonitor::MutableMetric *metric, const kmonitor::MetricsTags *, double value) {
    if (metric->metric_name_.find("data_storage.create_qps") != std::string::npos) {
        reported_create_qps += value;
    } else if (metric->metric_name_.find("data_storage.create_keys_qps") != std::string::npos) {
        reported_create_keys.push_back(value);
    } else if (metric->metric_name_.find("data_storage.create_time_us") != std::string::npos) {
        reported_create_times.push_back(value);
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

TEST_F(KmonitorMetricsReporterTest, DataStorageCreateSamplesSurviveReorderedReports) {
    Stub stub;
    auto report = static_cast<void (kmonitor::MutableMetric::*)(const kmonitor::MetricsTags *, double)>(
        &kmonitor::MutableMetric::Report);
    stub.set(report, CaptureCreateMetrics);

    for (const bool batch_first : {false, true}) {
        SCOPED_TRACE(batch_first);
        DataStorageManager manager(metrics_registry_);
        auto backend = std::make_shared<DummyBackend>(metrics_registry_);
        auto spec = std::make_shared<DummyStorageSpec>();
        spec->set_root_path("memory://");
        const auto storage = "samples_" + std::to_string(batch_first);
        ASSERT_EQ(EC_OK, backend->Open(StorageConfig(DataStorageType::DATA_STORAGE_TYPE_DUMMY, storage, spec), "open"));
        manager.storage_map_[storage] = backend;
        const auto shared = backend->GetMetricsCollector();
        const auto registry_size = metrics_registry_->GetSize();
        RequestContext first("first"), second("second");
        if (batch_first) {
            manager.CreateBatch(&first, storage, {{{"a", "b"}, 128}, {{"c"}, 256}}, nullptr);
            manager.Create(&second, storage, {"d", "e", "f", "g"}, 128, nullptr);
        } else {
            manager.Create(&first, storage, {"a", "b", "c"}, 128, nullptr);
            manager.CreateBatch(&second, storage, {{{"d"}, 128}, {{"e", "f", "g"}, 256}}, nullptr);
        }
        manager.Create(&first, storage, {"h", "i"}, 128, nullptr);
        const auto first_samples = first.GetMetricsCollectorsVehicle().GetMetricsCollectors();
        const auto second_samples = second.GetMetricsCollectorsVehicle().GetMetricsCollectors();
        ASSERT_EQ(2u, first_samples.size());
        ASSERT_EQ(1u, second_samples.size());
        const std::vector<std::shared_ptr<MetricsCollector>> report_order = {
            second_samples[0], first_samples[0], first_samples[1]};
        std::vector<double> expected_times;
        for (const auto &sample : report_order) {
            const auto create = std::dynamic_pointer_cast<DataStorageMetricsCollector>(sample);
            ASSERT_NE(nullptr, create);
            EXPECT_NE(shared, create);
            EXPECT_GE(create->GetCreateTimeUsSample(), 0.);
            expected_times.push_back(create->GetCreateTimeUsSample());
        }
        // Later allocations may replace both gauges before either request reports.
        shared->set_data_storage_create_keys_qps_metrics(999.);
        shared->set_data_storage_create_time_us_metrics(-123.);
        reported_create_qps = 0;
        reported_create_keys.clear();
        reported_create_times.clear();
        for (const auto &sample : report_order) {
            reporter_->ReportPerQuery(sample.get());
        }

        EXPECT_DOUBLE_EQ(3., reported_create_qps);
        EXPECT_EQ((std::vector<double>{4., 3., 2.}), reported_create_keys);
        EXPECT_EQ(expected_times, reported_create_times);
        EXPECT_EQ(3u, shared->get_data_storage_create_counter_metrics());
        EXPECT_EQ(9u, shared->get_data_storage_create_keys_counter_metrics());
        EXPECT_EQ(registry_size, metrics_registry_->GetSize());
    }
}
