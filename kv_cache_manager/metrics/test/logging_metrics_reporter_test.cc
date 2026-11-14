#include <memory>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/metrics/logging_metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "stub.h"

using namespace kv_cache_manager;

namespace {
thread_local std::vector<std::string> captured_metrics_logs;
void CaptureMetricsLog(int, const std::string &message) { captured_metrics_logs.push_back(message); }
} // namespace

class LoggingMetricsReporterTest : public TESTBASE {
protected:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry_);
        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        reporter_ = std::make_unique<LoggingMetricsReporter>();
        reporter_->Init(cache_manager_, metrics_registry_, "");
    }

    void TearDown() override {}

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::unique_ptr<LoggingMetricsReporter> reporter_;
};

TEST_F(LoggingMetricsReporterTest, ExecutorSummaryIsLoggedSeparatelyFromFullSnapshot) {
    metrics_registry_->GetGauge("service.request_queue_length") = 3;
    metrics_registry_->GetGauge("service.request_queue_capacity") = 2048;
    metrics_registry_->GetGauge("service.request_wait_time_us_max") = 24000;
    metrics_registry_->GetGauge("service.request_wait_time_us_avg") = 12000;
    metrics_registry_->GetGauge("service.request_execution_time_us_max") = 8000;
    metrics_registry_->GetCounter("service.request_completed") += 17;
    // A large unrelated family must not hide the executor in a truncated line.
    metrics_registry_->GetGauge("other." + std::string(210000, 'x')) = 1;
    Stub stub;
    stub.set(&LoggerBroker::MetricsLog, CaptureMetricsLog);
    captured_metrics_logs.clear();
    reporter_->ReportInterval();
    ASSERT_EQ(2u, captured_metrics_logs.size());
    const auto &summary = captured_metrics_logs.front();
    EXPECT_EQ(0u, summary.find("[Report_Request_Executor_Metrics_By_Print]|"));
    EXPECT_NE(std::string::npos, summary.find("[service.request_queue_capacity]:2048|"));
    EXPECT_NE(std::string::npos, summary.find("[service.request_wait_time_us_max]:24000|"));
    EXPECT_NE(std::string::npos, summary.find("[service.request_wait_time_us_avg]:12000|"));
    EXPECT_NE(std::string::npos, summary.find("[service.request_execution_time_us_max]:8000|"));
    EXPECT_NE(std::string::npos, summary.find("[service.request_completed]:17|"));
    EXPECT_EQ(std::string::npos, summary.find("other."));
    EXPECT_LT(summary.size(), 2048u);
    EXPECT_EQ(0u, captured_metrics_logs.back().find("[Report_Interval_By_Print]|"));
}

TEST_F(LoggingMetricsReporterTest, TestConstructor) { EXPECT_NE(reporter_, nullptr); }

TEST_F(LoggingMetricsReporterTest, TestInitWithNull) {
    EXPECT_FALSE(reporter_->Init(nullptr, nullptr, ""));

    EXPECT_FALSE(reporter_->Init(nullptr, metrics_registry_, ""));

    EXPECT_FALSE(reporter_->Init(cache_manager_, nullptr, ""));

    EXPECT_TRUE(reporter_->Init(cache_manager_, metrics_registry_, ""));
}

TEST_F(LoggingMetricsReporterTest, TestReportPerQuery) {
    EXPECT_NO_THROW(reporter_->ReportPerQuery(nullptr));

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

TEST_F(LoggingMetricsReporterTest, TestReportInterval) {
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

TEST_F(LoggingMetricsReporterTest, ForwardsRequestOutcomeToLocalCounters) {
    ServiceMetricsCollector collector(metrics_registry_);
    ASSERT_TRUE(collector.Init());
    SET_METRICS_(&collector, service, error_code, 1.);
    reporter_->ReportPerQuery(&collector, false);
    EXPECT_EQ(0u, collector.get_service_error_counter_metrics());

    SET_METRICS_(&collector, service, error_code, 0.);
    reporter_->ReportPerQuery(&collector, true);
    EXPECT_EQ(1u, collector.get_service_error_counter_metrics());
    EXPECT_EQ(2u, collector.get_service_query_counter_metrics());
}
