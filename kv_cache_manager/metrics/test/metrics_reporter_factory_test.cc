#include <memory>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/dummy_metrics_reporter.h"
#include "kv_cache_manager/metrics/kmonitor_metrics_reporter.h"
#include "kv_cache_manager/metrics/local_metrics_reporter.h"
#include "kv_cache_manager/metrics/logging_metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_reporter_factory.h"

using namespace kv_cache_manager;

class MetricsReporterFactoryTest : public TESTBASE {
protected:
    void SetUp() override { factory_ = std::make_unique<MetricsReporterFactory>(); }

    void TearDown() override { factory_.reset(); }

    std::unique_ptr<MetricsReporterFactory> factory_;
};

TEST_F(MetricsReporterFactoryTest, TestConstructor) { EXPECT_NE(factory_, nullptr); }

TEST_F(MetricsReporterFactoryTest, TestInitWithNull) {
    // nullptr is allowed
    EXPECT_TRUE(factory_->Init(nullptr, nullptr));
}

TEST_F(MetricsReporterFactoryTest, TestCreateDummyReporter) {
    auto reporter = factory_->Create("dummy", "");
    EXPECT_NE(reporter, nullptr);
    auto dummy_reporter = std::dynamic_pointer_cast<DummyMetricsReporter>(reporter);
    EXPECT_NE(dummy_reporter, nullptr);
}

TEST_F(MetricsReporterFactoryTest, TestCreateKmonitorReporter) {
    auto reporter = factory_->Create("kmonitor", "");
    EXPECT_NE(reporter, nullptr);
    auto kmonitor_reporter = std::dynamic_pointer_cast<KmonitorMetricsReporter>(reporter);
    EXPECT_NE(kmonitor_reporter, nullptr);
}

TEST_F(MetricsReporterFactoryTest, TestCreateLocalReporter) {
    auto reporter = factory_->Create("local", "");
    EXPECT_NE(reporter, nullptr);
    auto cached_reporter = std::dynamic_pointer_cast<LocalMetricsReporter>(reporter);
    EXPECT_NE(cached_reporter, nullptr);
}

TEST_F(MetricsReporterFactoryTest, TestCreateLoggingReporter) {
    auto reporter = factory_->Create("logging", "");
    EXPECT_NE(reporter, nullptr);
    auto logging_reporter = std::dynamic_pointer_cast<LoggingMetricsReporter>(reporter);
    EXPECT_NE(logging_reporter, nullptr);
}

TEST_F(MetricsReporterFactoryTest, TestCreateDefaultReporter) {
    auto reporter = factory_->Create("unknown", "");
    EXPECT_NE(reporter, nullptr);

    auto default_reporter = std::dynamic_pointer_cast<LoggingMetricsReporter>(reporter);
    EXPECT_NE(default_reporter, nullptr);
}
